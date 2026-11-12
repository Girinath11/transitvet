"""E2 neural models on the Kepler star split. Modes:
  cnn       AstroNet CNN (Adam 2e-4, batch 256), up to --steps, checkpoint chosen by val PR-AUC
  pretrain  masked-patch pretraining of the Transformer on Kepler TRAINING stars + unlabelled TESS (PC/APC)
  tf        Transformer fine-tune (from --init or from scratch), checkpoint chosen by val PR-AUC"""
import sys, os, json, time, argparse, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, '/home/user/tv/src'); import tv_common as c, tv_models as M
from sklearn.metrics import average_precision_score
ap = argparse.ArgumentParser()
ap.add_argument('mode'); ap.add_argument('--seeds', type=int, default=5); ap.add_argument('--steps', type=int, default=1000)
ap.add_argument('--init', default=None); ap.add_argument('--out', required=True); ap.add_argument('--label_frac', type=float, default=1.0)
ap.add_argument('--views', default='both'); ap.add_argument('--seed_list', default=''); ap.add_argument('--detrend', default='astronet')
a = ap.parse_args(); torch.set_num_threads(int(os.environ.get('TV_THREADS', '4')))
G, L, t = c.load_kepler()
if a.detrend == 'biweight':
    _d = np.load('/home/user/tv/out/transitvet_kepler_dr24.npz'); G, L = _d['global_biweight'], _d['local_biweight']
if a.views == 'global': L = np.zeros_like(L)   # ablation: local view removed
if a.views == 'local': G = np.zeros_like(G)    # ablation: global view removed
s = json.load(open('/home/user/tv/e2/kepler_star_split.json'))
tr, va, te = map(np.array, (s['train'], s['val'], s['test'])); y = t.y.values.astype(np.float32)
Gt, Lt, yt = torch.tensor(G), torch.tensor(L), torch.tensor(y)

def predict(m, ix, Gs=Gt, Ls=Lt):
    m.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(ix), 512): j = ix[i:i + 512]; out.append(torch.sigmoid(m(Gs[j], Ls[j])))
    m.train(); return torch.cat(out).numpy()

if a.mode == 'pretrain':
    Gtess, Ltess, tt = c.load_tess(); unl = np.where(tt.tfopwg_disp.isin(['PC', 'APC']))[0]
    PG = torch.cat([Gt[tr], torch.tensor(Gtess[unl])]); PL = torch.cat([Lt[tr], torch.tensor(Ltess[unl])])
    print('pretraining on', len(PG), 'views (', len(tr), 'Kepler training-star signals +', len(unl), 'unlabelled TESS)', flush=True)
    torch.manual_seed(0); np.random.seed(0); m = M.PatchTransformer()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=a.steps, pct_start=0.05)
    t0 = time.time()
    for step in range(1, a.steps + 1):
        j = torch.randint(0, len(PG), (128,)); loss = m.pretrain_loss(PG[j], PL[j])
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step()
        if step % 250 == 0: print(f'step {step} recon_mse {loss.item():.4f} {time.time()-t0:.0f}s', flush=True)
    torch.save(m.state_dict(), a.out); sys.exit()

trn = tr
if a.label_frac < 1.0:      # E8: subsample training STARS
    st = np.array(sorted(set(t.star.values[tr]))); np.random.RandomState(7).shuffle(st)
    keep = set(st[:int(a.label_frac * len(st))]); trn = tr[[x in keep for x in t.star.values[tr]]]
print(a.mode, 'train', len(trn), 'val', len(va), 'test', len(te), 'init', a.init, flush=True)
runs, probs, vprobs = [], [], []
for seed in ([int(x) for x in a.seed_list.split(',')] if a.seed_list else range(a.seeds)):
    torch.manual_seed(seed); np.random.seed(seed)
    if a.mode == 'cnn':
        m = M.AstroNet(); opt = torch.optim.Adam(m.parameters(), lr=2e-4); bs = 256; sched = None
    elif a.mode == 'mamba':
        E = lambda k, d: float(os.environ.get(k, d))
        m = M.PatchMamba(d=int(E('TV_D', 128)), drop=E('TV_DROP', 0.1)); lr = E('TV_LR', 5e-4)
        opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=E('TV_WD', 0.05)); bs = 128
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=a.steps, pct_start=0.1)
    else:
        m = M.PatchTransformer()
        if a.init: m.load_state_dict(torch.load(a.init))
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=0.05); bs = 128
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-4, total_steps=a.steps, pct_start=0.1)
    best, state, t0 = -1, None, time.time()
    for step in range(1, a.steps + 1):
        j = torch.tensor(np.random.choice(trn, bs)); gb, lb = Gt[j], Lt[j]
        if os.environ.get('TV_AUG') == '1':     # a transit looks the same played backwards; small extra noise
            flip = torch.rand(bs) < 0.5; gb = torch.where(flip[:, None], gb.flip(1), gb); lb = torch.where(flip[:, None], lb.flip(1), lb)
            gb = gb + 0.02 * torch.randn_like(gb); lb = lb + 0.02 * torch.randn_like(lb)
        loss = F.binary_cross_entropy_with_logits(m(gb, lb), yt[j])
        opt.zero_grad(); loss.backward(); opt.step()
        if sched: sched.step()
        if step % 125 == 0:
            ap_v = average_precision_score(y[va], predict(m, va))
            if ap_v > best: best, state, bstep = ap_v, {k: v.clone() for k, v in m.state_dict().items()}, step
            print(f'seed {seed} step {step} loss {loss.item():.4f} val_pr_auc {ap_v:.4f} {time.time()-t0:.0f}s', flush=True)
    m.load_state_dict(state); pv = predict(m, va); thr = c.best_threshold(y[va], pv); pt = predict(m, te)
    r = c.metrics(y[te], pt, thr); r.update(seed=seed, best_step=bstep, val_pr_auc=best, train_seconds=time.time() - t0)
    runs.append(r); probs.append(pt); vprobs.append(pv); print(r, flush=True)
    torch.save(state, a.out.replace('.json', f'_seed{seed}.pt'))
keys = [k for k in runs[0] if k not in ('seed',)]
res = dict(args=vars(a), n_train=len(trn), runs=runs, mean={k: float(np.mean([r[k] for r in runs])) for k in keys},
           std={k: float(np.std([r[k] for r in runs])) for k in keys})
pe = np.mean(probs, 0); res['ensemble'] = c.metrics(y[te], pe, c.best_threshold(y[va], np.mean(vprobs, 0)))
np.save(a.out.replace('.json', '_test_probs.npy'), np.array(probs))
json.dump(res, open(a.out, 'w'), indent=1); print('MEAN', {k: round(v, 4) for k, v in res['mean'].items()}, flush=True)
