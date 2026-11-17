"""Recompute every test number the journal paper reports, from the SHIPPED models (repo transitvet/models), no refitting."""
import sys, json, joblib, numpy as np, pandas as pd, torch
sys.path.insert(0, '/home/user/tv/src'); sys.path.insert(0, '/home/user/tv/repo')
import tv_common as c
from transitvet import tv_models as M
from sklearn.metrics import average_precision_score as ap, roc_auc_score as auc
exec(open('/home/user/tv/gaia/eval_gaia.py').read().split('def gb_fit')[0].split('torch.set_num_threads')[1].split('\n', 1)[1])
MD = '/home/user/tv/repo/transitvet/models'; NEWF = ['cen_sigma', 'cen_offset_arcsec', 'rp_est_re', 'contratio']
comb = joblib.load(f'{MD}/combiner.joblib'); conf = joblib.load(f'{MD}/conformal.joblib'); cal = joblib.load(f'{MD}/calibrators.joblib')
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
def boot(y, a, b, sid, n=2000):
    rng = np.random.default_rng(1); st = np.unique(sid); idx = {s: np.where(sid == s)[0] for s in st}; d = []
    for _ in range(n):
        ii = np.concatenate([idx[s] for s in rng.choice(st, len(st))])
        if y[ii].min() == y[ii].max(): continue
        d.append(ap(y[ii], a[ii]) - (ap(y[ii], b[ii]) if b is not None else 0))
    return np.percentile(d, [2.5, 97.5]).round(4).tolist()
out = {}; curves = {}
for mis in ('kepler', 'tess'):
    if mis == 'kepler':
        G, L, t = c.load_kepler(); sp = json.load(open('/home/user/tv/e2/kepler_star_split.json')); sid = t.kepid.values
        X = np.hstack([c.feature_matrix(t), pd.read_csv('/home/user/tv/imp/new_feats_kepler.csv')[NEWF].astype(float).values])
    else:
        G, L, t = c.load_tess(); lab = np.where(t.label.notna())[0]; G, L, t = G[lab], L[lab], t.iloc[lab].reset_index(drop=True)
        sp = json.load(open('/home/user/tv/e4/tess_star_split.json')); sid = t.tid.values
        X = np.hstack([c.feature_matrix(t), pd.read_csv('/home/user/tv/imp/new_feats_tess.csv')[NEWF].astype(float).values, gaia_cols(t)])
    y = t.label.values.astype(int); te = np.array(sp['test']); yt = y[te]
    raw = {'gb': np.mean([g.predict_proba(X[te])[:, 1] for g in joblib.load(f'{MD}/{mis}_gb.joblib')], 0)}
    g, l = torch.tensor(G[te]), torch.tensor(L[te])
    for kind, mk in {'tf': lambda: M.PatchTransformer(), 'mamba': lambda: M.PatchMamba(d=64, drop=0.2)}.items():
        ps = []
        for sd in torch.load(f'{MD}/{mis}_{kind}.pt'):
            m = mk(); m.load_state_dict(sd); m.eval()
            with torch.no_grad(): ps.append(np.concatenate([torch.sigmoid(m(g[i:i + 512], l[i:i + 512])).numpy() for i in range(0, len(te), 512)]))
        raw[kind] = np.mean(ps, 0)
    Z = np.column_stack([lg(raw[k]) for k in ('gb', 'mamba', 'tf')])
    p = comb[mis].predict_proba(Z)[:, 1]; pc = conf[mis]['combiner'].predict_proba(Z)[:, 1]
    q = conf[mis]['alphas'][0.05]; in1, in0 = (1 - pc) <= q['q1'], pc <= q['q0']
    keep, drop = in1 & ~in0, in0 & ~in1
    r = dict(n_test=int(len(te)), n_planets=int(yt.sum()), n_false=int((1 - yt).sum()), n_test_stars=int(len(np.unique(sid[te]))),
             n_train=len(sp['train']), n_val=len(sp['val']),
             pr_auc={k: round(ap(yt, v), 4) for k, v in dict(raw, combined=p).items()},
             roc_auc={k: round(auc(yt, v), 4) for k, v in dict(raw, combined=p).items()},
             combined_ci=boot(yt, p, None, sid[te]),
             at05=dict(false_caught=int(((p < 0.5) & (yt == 0)).sum()), planets_kept=int(((p >= 0.5) & (yt == 1)).sum())),
             conformal=dict(cov_planets=round(float(in1[yt == 1].mean()), 4), cov_false=round(float(in0[yt == 0].mean()), 4),
                            keep=int(keep.sum()), drop=int(drop.sum()), person=int((~keep & ~drop).sum()),
                            keep_planet_frac=round(float(yt[keep].mean()), 4), drop_false_frac=round(float(1 - yt[drop].mean()), 4),
                            planets_dropped=int((drop & (yt == 1)).sum()), n_calib=conf[mis]['n_calib'],
                            q1=q['q1'], q0=q['q0']),
             coef=comb[mis].coef_.round(3).tolist()[0])
    out[mis] = r; curves[mis] = dict(y=yt.tolist(), **{k: np.round(v, 5).tolist() for k, v in dict(raw, combined=p, conf_p=pc).items()})
    print(mis, json.dumps(r), flush=True)
json.dump(out, open('/home/user/tv/journal/eval.json', 'w'), indent=1); json.dump(curves, open('/home/user/tv/journal/curves.json', 'w'))
