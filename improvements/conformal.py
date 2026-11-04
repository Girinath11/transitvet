"""Conformal triage for TransitVet (label-conditional / Mondrian split conformal).
Validation stars are split in two halves BY STAR: half A fits the combiner, half B calibrates conformal.
Test stars: measure coverage, share of 'sure' calls and the error rate among sure calls."""
import sys, json, numpy as np, pandas as pd, joblib, torch, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/user/tv/src'); sys.path.insert(0, '/home/user/tv/app')
import tv_common as c
from transitvet import tv_models as M
from sklearn.linear_model import LogisticRegression
MD = '/home/user/tv/app/transitvet/models'; NEW = ['cen_sigma', 'cen_offset_arcsec', 'rp_est_re', 'contratio']
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
def raw(mis, G, L, X, ix):
    out = {'gb': np.mean([g.predict_proba(X[ix])[:, 1] for g in joblib.load(f'{MD}/{mis}_gb.joblib')], 0)}
    g, l = torch.tensor(G[ix]), torch.tensor(L[ix])
    for kind, cls in (('cnn', M.AstroNet), ('tf', M.PatchTransformer)):
        ps = []
        for sd in torch.load(f'{MD}/{mis}_{kind}.pt'):
            m = cls(); m.load_state_dict(sd); m.eval()
            with torch.no_grad(): ps.append(np.concatenate([torch.sigmoid(m(g[i:i+512], l[i:i+512])).numpy() for i in range(0, len(ix), 512)]))
        out[kind] = np.mean(ps, 0)
    return np.column_stack([lg(out[k]) for k in ('gb', 'cnn', 'tf')])
def qhat(scores, alpha):
    n = len(scores); k = int(np.ceil((n + 1) * (1 - alpha)))
    return np.inf if k > n else np.sort(scores)[k - 1]
res = {}; save = {}
for mis in ('kepler', 'tess'):
    if mis == 'kepler':
        G, L, t = c.load_kepler(); sp = json.load(open('/home/user/tv/e2/kepler_star_split.json')); sid = t.kepid.values
    else:
        G, L, t = c.load_tess(); lab = np.where(t.label.notna())[0]; G, L, t = G[lab], L[lab], t.iloc[lab].reset_index(drop=True)
        sp = json.load(open('/home/user/tv/e4/tess_star_split.json')); sid = t.tid.values
    y = t.label.values.astype(int); va, te = np.array(sp['val']), np.array(sp['test'])
    nf = pd.read_csv(f'/home/user/tv/imp/new_feats_{mis}.csv'); X = np.hstack([c.feature_matrix(t), nf[NEW].values])
    Zv, Zt = raw(mis, G, L, X, va), raw(mis, G, L, X, te)
    stars = np.unique(sid[va]); rng = np.random.default_rng(0); A = set(rng.choice(stars, len(stars) // 2, replace=False))
    a = np.array([s in A for s in sid[va]]); b = ~a
    lr = LogisticRegression().fit(Zv[a], y[va][a])
    pv, pt = lr.predict_proba(Zv[b])[:, 1], lr.predict_proba(Zt)[:, 1]; yv, yt = y[va][b], y[te]
    out = {'n_calib': int(b.sum()), 'n_calib_planets': int(yv.sum()), 'n_test': len(te)}
    for alpha in (0.05, 0.10, 0.20):
        # label-conditional: nonconformity for class 1 = 1-p, for class 0 = p; quantile per class
        q1, q0 = qhat(1 - pv[yv == 1], alpha), qhat(pv[yv == 0], alpha)
        in1, in0 = (1 - pt) <= q1, pt <= q0
        sure_pc, sure_fa, both, empty = in1 & ~in0, in0 & ~in1, in1 & in0, ~in1 & ~in0
        sure = sure_pc | sure_fa
        wrong = (sure_pc & (yt == 0)) | (sure_fa & (yt == 1))
        out[f'alpha_{alpha}'] = dict(
            coverage_planets=float(in1[yt == 1].mean()), coverage_false=float(in0[yt == 0].mean()),
            share_sure=float(sure.mean()), share_needs_human=float((both | empty).mean()),
            sure_candidate=int(sure_pc.sum()), sure_false_alarm=int(sure_fa.sum()), needs_human=int((both | empty).sum()),
            errors_among_sure=int(wrong.sum()), error_rate_among_sure=float(wrong.sum() / max(sure.sum(), 1)),
            planets_lost_as_sure_false=int((sure_fa & (yt == 1)).sum()), thresholds=dict(candidate_if_p_ge=float(1 - q1), false_if_p_le=float(q0)))
    save[mis] = dict(combiner=lr, alphas={al: dict(q1=float(qhat(1 - pv[yv == 1], al)), q0=float(qhat(pv[yv == 0], al))) for al in (0.05, 0.10, 0.20)}, n_calib=int(b.sum()))
    res[mis] = out; print(mis, json.dumps(out, indent=1), flush=True)
json.dump(res, open('/home/user/tv/conf/conformal.json', 'w'), indent=1)
joblib.dump(save, '/home/user/tv/app/transitvet/models/conformal.joblib')
