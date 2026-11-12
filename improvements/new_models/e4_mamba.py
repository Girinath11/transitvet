"""E4 for Mamba (same protocol as e4.py). E4: Kepler -> TESS. TESS labelled TOIs (CP/KP=1, FP/FA=0) split BY STAR 50/10/40 (seed 123).
Variants per model: (a) Kepler-trained, unchanged; (b) Kepler-trained then fine-tuned on TESS train;
(c) trained on TESS train only. Thresholds from TESS validation. Saves val/test probs for E5."""
import sys, json, time, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, '/home/user/tv/src'); import tv_common as c, tv_models as M
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
import os; torch.set_num_threads(int(os.environ.get('TV_THREADS', '4')))
Gk, Lk, tk = c.load_kepler(); Gs, Ls, ts = c.load_tess()
lab = np.where(ts.y.notna())[0]; ts_l = ts.iloc[lab].reset_index(drop=True)
Gs, Ls = Gs[lab], Ls[lab]; ys = ts_l.y.values.astype(np.float32)
tr, va, te = c.star_split(ts_l.star.values, seed=123, frac=(0.5, 0.1, 0.4))
assert json.load(open('/home/user/tv/e4/tess_star_split.json'))['test'] == te.tolist()   # same split as every other model
print('TESS labelled', len(ys), 'train/val/test', len(tr), len(va), len(te), 'pos frac test', ys[te].mean(), flush=True)
GT, LT, yT = torch.tensor(Gs), torch.tensor(Ls), torch.tensor(ys)
ks = json.load(open('/home/user/tv/e2/kepler_star_split.json')); kva = np.array(ks['val']); kte = np.array(ks['test'])
res, probs = {}, {}
def pred(m, G, L, ix):
    m.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(ix), 512): j = ix[i:i + 512]; out.append(torch.sigmoid(m(torch.as_tensor(G[j]), torch.as_tensor(L[j]))))
    m.train(); return torch.cat(out).numpy()
def finetune(m, steps, lr, bs=64):
    opt = torch.optim.Adam(m.parameters(), lr=lr); best, state = -1, None
    for step in range(1, steps + 1):
        j = torch.tensor(np.random.choice(tr, bs)); loss = F.binary_cross_entropy_with_logits(m(GT[j], LT[j]), yT[j])
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 25 == 0:
            a = average_precision_score(ys[va], pred(m, Gs, Ls, va))
            if a > best: best, state = a, {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(state); return m
def record(name, pv, pt, extra=None):
    r = c.metrics(ys[te], pt, c.best_threshold(ys[va], pv)); res.setdefault(name, []).append(r)
    probs.setdefault(name, []).append(dict(val=pv.tolist(), test=pt.tolist()))
import os
os.environ.setdefault('TV_D', '64')
def mk(): return M.PatchMamba(d=64, drop=0.2)
for seed in range(5):
    torch.manual_seed(seed); np.random.seed(seed); t0 = time.time()
    m = mk(); m.load_state_dict(torch.load(f'/home/user/tv/e2/e2_mamba_s{seed}_seed{seed}.pt'))
    record('mamba_kepler_unchanged', pred(m, Gs, Ls, va), pred(m, Gs, Ls, te))
    probs.setdefault('mamba_kepler_on_kepler', []).append(dict(val=pred(m, Gk, Lk, kva).tolist(), test=pred(m, Gk, Lk, kte).tolist()))
    m = finetune(m, 400, 1e-4); record('mamba_kepler_finetuned', pred(m, Gs, Ls, va), pred(m, Gs, Ls, te))
    torch.save(m.state_dict(), f'/home/user/tv/e4/mamba_tess_ft_seed{seed}.pt')
    m = finetune(mk(), 800, 1e-3); record('mamba_tess_only', pred(m, Gs, Ls, va), pred(m, Gs, Ls, te))
    print('mamba', seed, {k: round(res[k][-1]['pr_auc'], 3) for k in res}, f'{time.time()-t0:.0f}s', flush=True)
summ = {k: dict(mean={m: float(np.mean([r[m] for r in v])) for m in v[0]}, std={m: float(np.std([r[m] for r in v])) for m in v[0]}, runs=v) for k, v in res.items()}
json.dump(summ, open('/home/user/tv/e4/e4_mamba_results.json', 'w'), indent=1)
json.dump(dict(probs=probs), open('/home/user/tv/e4/probs_mamba.json', 'w'))
for k, v in summ.items(): print(k, {m: round(v['mean'][m], 3) for m in ['pr_auc', 'roc_auc', 'precision', 'recall']}, flush=True)
