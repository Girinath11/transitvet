"""5-fold cross-validation BY STAR on every labelled star (extra result for the paper; main held-out results unchanged).

For each mission, stars are shuffled (seed 2026) and cut into 5 folds. For fold k:
  test  = stars of fold k (scored once)
  val   = 15/85 of the remaining stars (all choices: checkpoints, combiner, conformal cut-offs)
  train = the rest
Kepler: GB (5 seeds), Transformer (masked pretraining on this fold's TRAIN views + unlabelled TESS, then fine-tune), Mamba-2 (1 seed).
TESS:  GB with Gaia (5 seeds); Mamba-2 and Transformer start from the shipped KEPLER-trained weights (Kepler stars only,
       no TESS star) and are fine-tuned on this fold's TESS train stars (5 seeds, as in the tool).
Combiner = logistic regression on log-odds (gb, mamba, tf) fitted on val; conformal = half A of val stars fits a combiner,
half B calibrates, alpha = 0.05 (keep / drop / needs a person, counted as for the held-out tables).
Usage: python cv.py kepler K   |   python cv.py tess K   |   python cv.py summary"""
import sys, os, json, time, numpy as np, pandas as pd, joblib, torch, torch.nn.functional as F, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/user/tv/src'); sys.path.insert(0, '/home/user/tv/repo')
import tv_common as c
from transitvet import tv_models as M
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from transitvet.pipeline import verdict
from sklearn.metrics import average_precision_score as ap, roc_auc_score
OUT = '/home/user/tv/cv'; MD = '/home/user/tv/repo/transitvet/models'
NEW = ['cen_sigma', 'cen_offset_arcsec', 'rp_est_re', 'contratio']
GA = ['g_aen', 'g_aen_sig', 'g_gof', 'g_rv_err', 'g_bprp_exc', 'g_plx_snr', 'g_n21', 'g_fr21', 'g_n42', 'g_fr42', 'g_nb_dmag', 'g_nb_sep', 'g_log_nb_depth']
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
SMOKE = os.environ.get('TV_SMOKE') == '1'
S = lambda n: max(40, n // 50) if SMOKE else n
torch.set_num_threads(int(os.environ.get('TV_THREADS', '2')))


def gaia_cols(t):
    g = pd.read_csv('/home/user/tv/gaia/gaia_star_feats.csv').set_index('tid').reindex(t.tid.astype('int64').values)
    g['g_log_nb_depth'] = np.log10(np.clip(t.depth_ppm.values * 1e-6 * (1 + g.g_fr42.values) / np.maximum(g.g_nb_fr.values, 1e-6), 1e-6, 1e3))
    return g[GA].astype(float).values


def data(mis):
    if mis == 'kepler':
        G, L, t = c.load_kepler(); sid = t.kepid.values
        X = np.hstack([c.feature_matrix(t), pd.read_csv('/home/user/tv/imp/new_feats_kepler.csv')[NEW].astype(float).values])
    else:
        G, L, t = c.load_tess(); lab = np.where(t.label.notna())[0]; G, L, t = G[lab], L[lab], t.iloc[lab].reset_index(drop=True); sid = t.tid.values
        X = np.hstack([c.feature_matrix(t), pd.read_csv('/home/user/tv/imp/new_feats_tess.csv')[NEW].astype(float).values, gaia_cols(t)])
    y = t.label.values.astype(int) if mis == 'tess' else t.y.values.astype(int)
    return G.astype(np.float32), L.astype(np.float32), X, y, sid


def folds(sid, k):
    u = np.array(sorted(set(sid))); np.random.RandomState(2026).shuffle(u)
    parts = np.array_split(u, 5); test = set(parts[k])
    rest = np.array([s for s in u if s not in test]); np.random.RandomState(100 + k).shuffle(rest)
    nv = int(round(len(rest) * 15 / 85)); val = set(rest[:nv])
    te = np.where([s in test for s in sid])[0]; va = np.where([s in val for s in sid])[0]
    tr = np.where([(s not in test) and (s not in val) for s in sid])[0]
    return tr, va, te


def pred(m, G, L, ix):
    m.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(ix), 512):
            j = ix[i:i + 512]; out.append(torch.sigmoid(m(torch.as_tensor(G[j]), torch.as_tensor(L[j]))).numpy())
    m.train(); return np.concatenate(out)


def train_net(m, G, L, y, tr, va, steps, opt, sched=None, bs=128, every=None, aug=False):
    Gt, Lt, yt = torch.as_tensor(G), torch.as_tensor(L), torch.as_tensor(y.astype(np.float32))
    best, state = -1, None; every = every or (4 if SMOKE else 125)
    for step in range(1, steps + 1):
        j = torch.as_tensor(np.random.choice(tr, bs)); gb, lb = Gt[j], Lt[j]
        if aug:
            flip = torch.rand(bs) < 0.5; gb = torch.where(flip[:, None], gb.flip(1), gb); lb = torch.where(flip[:, None], lb.flip(1), lb)
            gb = gb + 0.02 * torch.randn_like(gb); lb = lb + 0.02 * torch.randn_like(lb)
        loss = F.binary_cross_entropy_with_logits(m(gb, lb), yt[j])
        opt.zero_grad(); loss.backward(); opt.step()
        if sched: sched.step()
        if step % every == 0:
            a = ap(y[va], pred(m, G, L, va))
            if a > best: best, state = a, {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(state); return m


def gb_scores(X, y, tr, va, te):
    gbs = [HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, early_stopping=True, validation_fraction=0.1, random_state=s).fit(X[tr], y[tr]) for s in range(5)]
    return [np.mean([g.predict_proba(X[ix])[:, 1] for g in gbs], 0) for ix in (va, te)]


def run_fold(mis, k):
    path = f'{OUT}/{mis}_fold{k}' + ('_smoke' if SMOKE else '') + '.json'
    if os.path.exists(path):
        return
    t0 = time.time(); G, L, X, y, sid = data(mis); tr, va, te = folds(sid, k)
    print(mis, 'fold', k, 'train/val/test', len(tr), len(va), len(te), flush=True)
    raw = {}
    raw['gb'] = gb_scores(X, y, tr, va, te); print(' gb done', round(time.time() - t0), flush=True)
    if mis == 'kepler':
        # Transformer: masked pretraining on this fold's training views + unlabelled TESS views, then fine-tune
        Gs, Ls, ts = c.load_tess(); unl = np.where(ts.tfopwg_disp.isin(['PC', 'APC']))[0]
        PG = torch.as_tensor(np.concatenate([G[tr], Gs[unl]]).astype(np.float32)); PL = torch.as_tensor(np.concatenate([L[tr], Ls[unl]]).astype(np.float32))
        torch.manual_seed(0); np.random.seed(0); m = M.PatchTransformer()
        opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.05); sch = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=S(3000), pct_start=0.05)
        for step in range(S(3000)):
            j = torch.randint(0, len(PG), (128,)); loss = m.pretrain_loss(PG[j], PL[j])
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sch.step()
        print(' pretrain done', round(time.time() - t0), flush=True)
        torch.manual_seed(0); np.random.seed(0)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=0.05); sch = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-4, total_steps=S(1500), pct_start=0.1)
        m = train_net(m, G, L, y, tr, va, S(1500), opt, sch); raw['tf'] = [pred(m, G, L, va), pred(m, G, L, te)]
        print(' tf done', round(time.time() - t0), flush=True)
        torch.manual_seed(0); np.random.seed(0); m = M.PatchMamba(d=64, drop=0.2)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.1); sch = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=S(1000), pct_start=0.1)
        m = train_net(m, G, L, y, tr, va, S(1000), opt, sch, aug=True); raw['mamba'] = [pred(m, G, L, va), pred(m, G, L, te)]
        print(' mamba done', round(time.time() - t0), flush=True)
    else:
        for kind, mk, lr in (('mamba', lambda: M.PatchMamba(d=64, drop=0.2), 1e-4), ('tf', M.PatchTransformer, 3e-5)):
            pv, pt = [], []
            for seed, sd in enumerate(torch.load(f'{MD}/kepler_{kind}.pt')):
                torch.manual_seed(seed); np.random.seed(seed); m = mk(); m.load_state_dict(sd)
                m = train_net(m, G, L, y, tr, va, S(400), torch.optim.Adam(m.parameters(), lr=lr), bs=64, every=4 if SMOKE else 25)
                pv.append(pred(m, G, L, va)); pt.append(pred(m, G, L, te))
            raw[kind] = [np.mean(pv, 0), np.mean(pt, 0)]
            print(f' {kind} done', round(time.time() - t0), flush=True)
    ks = ['gb', 'mamba', 'tf']
    Zv = np.column_stack([lg(raw[x][0]) for x in ks]); Zt = np.column_stack([lg(raw[x][1]) for x in ks])
    comb = LogisticRegression(C=1.0).fit(Zv, y[va]); pt = comb.predict_proba(Zt)[:, 1]
    stars = np.unique(sid[va]); rng = np.random.default_rng(0); A = set(rng.choice(stars, len(stars) // 2, replace=False))
    a = np.array([s in A for s in sid[va]]); b = ~a
    lr_ = LogisticRegression().fit(Zv[a], y[va][a]); pb = lr_.predict_proba(Zv[b])[:, 1]; pc = lr_.predict_proba(Zt)[:, 1]; yb = y[va][b]

    def qhat(sc, alpha=0.05):
        n = len(sc); kk = int(np.ceil((n + 1) * (1 - alpha))); return np.inf if kk > n else float(np.sort(sc)[kk - 1])
    q1, q0 = qhat(1 - pb[yb == 1]), qhat(pb[yb == 0])
    out = dict(mission=mis, fold=k, n_train=len(tr), n_val=len(va), n_test=len(te), n_calib=int(b.sum()), coef=comb.coef_.round(3).tolist()[0],
               test_idx=te.tolist(), y=y[te].tolist(), star=[int(s) for s in sid[te]], combined=pt.tolist(), conf_p=pc.tolist(), q1=q1, q0=q0,
               **{x: raw[x][1].tolist() for x in ks}, seconds=round(time.time() - t0))
    json.dump(out, open(path, 'w')); print(mis, 'fold', k, 'PR-AUC', round(ap(y[te], pt), 4), 'done', out['seconds'], 's', flush=True)


def boot_ci(y, p, sid, n=2000):
    rng = np.random.default_rng(0); st = np.unique(sid); idx = {s: np.where(sid == s)[0] for s in st}; v = []
    for _ in range(n):
        j = np.concatenate([idx[s] for s in rng.choice(st, st.size)])
        if y[j].min() != y[j].max():
            v.append(ap(y[j], p[j]))
    return [round(float(np.percentile(v, q)), 4) for q in (2.5, 97.5)]


def summary():
    rep = {}
    for mis in ('kepler', 'tess'):
        fs = [json.load(open(f'{OUT}/{mis}_fold{k}' + ('_smoke' if SMOKE else '') + '.json')) for k in range(5)]
        y = np.concatenate([f['y'] for f in fs]); sid = np.concatenate([f['star'] for f in fs])
        pc = np.concatenate([f['combined'] for f in fs])
        keep = drop = person = tk = td = tp_ = pdrop = kp = df_ = 0
        c1, c0 = [], []
        for f in fs:
            yy = np.array(f['y']); p = np.array(f['combined']); q = np.array(f['conf_p'])
            in1, in0 = (1 - q) <= f['q1'], q <= f['q0']
            c1 += list(in1[yy == 1]); c0 += list(in0[yy == 0])
            k2 = in1 & ~in0; d2 = in0 & ~in1   # conformal sets alone
            keep += int(k2.sum()); drop += int(d2.sum()); person += int(len(yy) - k2.sum() - d2.sum())
            # tool rule: a keep/drop that the score verdict contradicts goes to a person (single-model scores uncalibrated here)
            vd = np.array([verdict(p[i], [f['gb'][i], f['mamba'][i], f['tf'][i]]) for i in range(len(yy))])
            k3 = k2 & (vd != 'false alarm'); d3 = d2 & (vd != 'candidate')
            tk += int(k3.sum()); td += int(d3.sum()); tp_ += int(len(yy) - k3.sum() - d3.sum()); pdrop += int((d3 & (yy == 1)).sum())
            kp += int((k3 & (yy == 1)).sum()); df_ += int((d3 & (yy == 0)).sum())
        r = dict(n_signals=int(len(y)), n_stars=int(len(np.unique(sid))), n_planets=int(y.sum()), n_false=int((1 - y).sum()),
                 pr_auc_combined=round(float(ap(y, pc)), 4), pr_auc_ci=boot_ci(y, pc, sid), roc_auc_combined=round(float(roc_auc_score(y, pc)), 4),
                 per_fold_pr_auc=[round(float(ap(np.array(f['y']), np.array(f['combined']))), 4) for f in fs],
                 pr_auc_models={x: round(float(ap(y, np.concatenate([f[x] for f in fs]))), 4) for x in ('gb', 'mamba', 'tf')},
                 false_rejected_at_05=int(((pc < 0.5) & (y == 0)).sum()), planets_kept_at_05=int(((pc >= 0.5) & (y == 1)).sum()),
                 conformal_coverage_planets=round(float(np.mean(c1)), 4), conformal_coverage_false=round(float(np.mean(c0)), 4),
                 sets_keep=keep, sets_drop=drop, sets_person=person, tool_keep=tk, tool_drop=td, tool_person=tp_, tool_planets_dropped=pdrop,
                 tool_keep_share_planets=round(kp / max(tk, 1), 4), tool_drop_share_false=round(df_ / max(td, 1), 4), coef_per_fold=[f['coef'] for f in fs], n_calib_per_fold=[f['n_calib'] for f in fs])
        r['per_fold_mean'] = round(float(np.mean(r['per_fold_pr_auc'])), 4); r['per_fold_sd'] = round(float(np.std(r['per_fold_pr_auc'])), 4)
        rep[mis] = r; print(mis, json.dumps(r), flush=True)
    json.dump(rep, open(f'{OUT}/cv_summary' + ('_smoke' if SMOKE else '') + '.json', 'w'), indent=1)


if __name__ == '__main__':
    if sys.argv[1] == 'summary':
        summary()
    else:
        run_fold(sys.argv[1], int(sys.argv[2]))
