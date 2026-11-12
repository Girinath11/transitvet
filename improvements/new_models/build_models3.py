"""Models v3 (Nov 2026): the tool's neural pair becomes Mamba-2 + Transformer (the CNN stays in the paper as the
reproduced baseline). Mamba weights: 5 seeds trained on Kepler TRAINING stars (e2_neural.py mamba), and for TESS
the same 5 fine-tuned on TESS TRAINING stars (e4_mamba.py). Isotonic calibrator, combiner (logistic regression on
gb/mamba/tf log-odds) fitted on VALIDATION stars; conformal combiner/thresholds re-fitted exactly as in
improvements/conformal.py (half A of validation stars fits, half B calibrates). Test stars only for the report."""
import sys, os, json, shutil, joblib, numpy as np, pandas as pd, torch
sys.path.insert(0, '/home/user/tv/src'); sys.path.insert(0, '/home/user/tv/repo')
import tv_common as c
from transitvet import tv_models as M
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score as ap
SRC = '/home/user/tv/repo/transitvet/models'; OUT = '/home/user/tv/v3/models'; NEW = ['cen_sigma', 'cen_offset_arcsec', 'rp_est_re', 'contratio']
os.makedirs(OUT, exist_ok=True)
for f in os.listdir(SRC):
    if not f.endswith('_cnn.pt'): shutil.copy(f'{SRC}/{f}', f'{OUT}/{f}')
torch.save([torch.load(f'/home/user/tv/e2/e2_mamba_s{s}_seed{s}.pt') for s in range(5)], f'{OUT}/kepler_mamba.pt')
torch.save([torch.load(f'/home/user/tv/e4/mamba_tess_ft_seed{s}.pt') for s in range(5)], f'{OUT}/tess_mamba.pt')
cal = joblib.load(f'{OUT}/calibrators.joblib'); man = json.load(open(f'{OUT}/manifest.json')); comb, conf = {}, {}
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
def qhat(sc, alpha):
    n = len(sc); k = int(np.ceil((n + 1) * (1 - alpha))); return np.inf if k > n else np.sort(sc)[k - 1]
KINDS = {'tf': lambda: M.PatchTransformer(), 'mamba': lambda: M.PatchMamba(d=64, drop=0.2)}
report = {}
for mis in ('kepler', 'tess'):
    if mis == 'kepler':
        G, L, t = c.load_kepler(); sp = json.load(open('/home/user/tv/e2/kepler_star_split.json')); sid = t.kepid.values
    else:
        G, L, t = c.load_tess(); lab = np.where(t.label.notna())[0]; G, L, t = G[lab], L[lab], t.iloc[lab].reset_index(drop=True)
        sp = json.load(open('/home/user/tv/e4/tess_star_split.json')); sid = t.tid.values
    y = t.label.values.astype(int); va, te = np.array(sp['val']), np.array(sp['test'])
    X = np.hstack([c.feature_matrix(t), pd.read_csv(f'/home/user/tv/imp/new_feats_{mis}.csv')[NEW].astype(float).values])
    raw = {}
    for nm, ix in (('va', va), ('te', te)):
        raw[('gb', nm)] = np.mean([g.predict_proba(X[ix])[:, 1] for g in joblib.load(f'{OUT}/{mis}_gb.joblib')], 0)
        g, l = torch.tensor(G[ix]), torch.tensor(L[ix])
        for kind, mk in KINDS.items():
            ps = []
            for sd in torch.load(f'{OUT}/{mis}_{kind}.pt'):
                m = mk(); m.load_state_dict(sd); m.eval()
                with torch.no_grad(): ps.append(np.concatenate([torch.sigmoid(m(g[i:i + 512], l[i:i + 512])).numpy() for i in range(0, len(ix), 512)]))
            raw[(kind, nm)] = np.mean(ps, 0)
    cal[f'{mis}_mamba'] = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1).fit(raw[('mamba', 'va')], y[va])
    ks = ['gb', 'mamba', 'tf']; Zv = np.column_stack([lg(raw[(k, 'va')]) for k in ks]); Zt = np.column_stack([lg(raw[(k, 'te')]) for k in ks])
    comb[mis] = LogisticRegression(C=1.0).fit(Zv, y[va]); pt = comb[mis].predict_proba(Zt)[:, 1]
    # conformal, same recipe as improvements/conformal.py
    stars = np.unique(sid[va]); rng = np.random.default_rng(0); A = set(rng.choice(stars, len(stars) // 2, replace=False))
    a = np.array([s in A for s in sid[va]]); b = ~a
    lr = LogisticRegression().fit(Zv[a], y[va][a]); pv_b, pt_c = lr.predict_proba(Zv[b])[:, 1], lr.predict_proba(Zt)[:, 1]; yv = y[va][b]
    conf[mis] = dict(combiner=lr, alphas={al: dict(q1=float(qhat(1 - pv_b[yv == 1], al)), q0=float(qhat(pv_b[yv == 0], al))) for al in (0.05, 0.10, 0.20)}, n_calib=int(b.sum()))
    q1, q0 = conf[mis]['alphas'][0.05]['q1'], conf[mis]['alphas'][0.05]['q0']; yt = y[te]
    in1, in0 = (1 - pt_c) <= q1, pt_c <= q0
    report[mis] = dict(test_pr_auc_combined=round(ap(yt, pt), 4), test_pr_auc_mamba=round(ap(yt, raw[('mamba', 'te')]), 4),
                       val_pr_auc_combined=round(ap(y[va], comb[mis].predict_proba(Zv)[:, 1]), 4), coef_gb_mamba_tf=comb[mis].coef_.round(3).tolist()[0],
                       conformal95_coverage_planets=round(float(in1[yt == 1].mean()), 4), conformal95_coverage_false=round(float(in0[yt == 0].mean()), 4),
                       conformal95_needs_human=int(((in1 & in0) | (~in1 & ~in0)).sum()), n_test=len(te))
    man['models'][f'{mis}_mamba'] = dict(val_pr_auc=round(ap(y[va], raw[('mamba', 'va')]), 4), n_val=len(va), cfg=dict(d=64, layers=4, d_state=32, drop=0.2))
    man['models'][f'{mis}_combined'] = dict(val_pr_auc=report[mis]['val_pr_auc_combined'], n_val=len(va), coef_gb_mamba_tf=report[mis]['coef_gb_mamba_tf'])
    print(mis, report[mis], flush=True)
man['nets'] = ['mamba', 'tf']; man['combine'] = ['gb', 'mamba', 'tf']; man['built_v3'] = '2026-11-12'
man['note'] += ' v3: the tool uses Mamba-2 + Transformer as its neural models (CNN kept in the paper as the reproduced baseline); combiner on gb/mamba/tf.'
joblib.dump(cal, f'{OUT}/calibrators.joblib'); joblib.dump(comb, f'{OUT}/combiner.joblib'); joblib.dump(conf, f'{OUT}/conformal.joblib')
json.dump(man, open(f'{OUT}/manifest.json', 'w'), indent=1); json.dump(report, open('/home/user/tv/v3/report.json', 'w'), indent=1); print('ALLDONE')
