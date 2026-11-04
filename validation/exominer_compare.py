"""Our TESS test signals vs NASA's published ExoMiner++ out-of-fold scores (Zenodo 15466293, labeled TCEs S1-S67).
Match: same TIC and period within 1%; several TCE runs per TOI -> score of the longest multi-sector run (as the tool shows).
Our scores: TransitVet v2 models (GB with new checks, CNN, Transformer, combiner) on our TESS TEST split."""
import sys, json, numpy as np, pandas as pd, joblib, torch, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/user/tv/src'); sys.path.insert(0, '/home/user/tv/app')
import tv_common as c
from transitvet import tv_models as M
from sklearn.metrics import average_precision_score as ap, roc_auc_score as auc
MD = '/home/user/tv/app/transitvet/models'; NEW = ['cen_sigma', 'cen_offset_arcsec', 'rp_est_re', 'contratio']
G, L, t = c.load_tess(); lab = np.where(t.label.notna())[0]; G, L, t = G[lab], L[lab], t.iloc[lab].reset_index(drop=True)
y = t.label.values.astype(int); te = np.array(json.load(open('/home/user/tv/e4/tess_star_split.json'))['test'])
nf = pd.read_csv('/home/user/tv/imp/new_feats_tess.csv'); X = np.hstack([c.feature_matrix(t), nf[NEW].values])
raw = {'gb': np.mean([g.predict_proba(X[te])[:, 1] for g in joblib.load(f'{MD}/tess_gb.joblib')], 0)}
g, l = torch.tensor(G[te]), torch.tensor(L[te])
for kind, cls in (('cnn', M.AstroNet), ('tf', M.PatchTransformer)):
    ps = []
    for sd in torch.load(f'{MD}/tess_{kind}.pt'):
        m = cls(); m.load_state_dict(sd); m.eval()
        with torch.no_grad(): ps.append(torch.sigmoid(m(g, l)).numpy())
    raw[kind] = np.mean(ps, 0)
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
ours = joblib.load(f'{MD}/combiner.joblib')['tess'].predict_proba(np.column_stack([lg(raw[k]) for k in ('gb', 'cnn', 'tf')]))[:, 1]
ex = pd.read_csv('labeled.csv', comment='#'); exg = {k: v for k, v in ex.groupby('target_id')}
tt = t.iloc[te].reset_index(drop=True); exo, exlab, nmatch = [], [], []
for tid, per in zip(tt.tid.astype(int), tt.tce_period):
    e = exg.get(tid)
    if e is None: exo.append(np.nan); exlab.append(''); nmatch.append(0); continue
    m = e[np.abs(e.tce_period - per) / per < 0.01]
    def span(r):
        a = str(r).split('-')[2:] if False else str(r).lstrip('S').split('-')
        try: return int(a[-1]) - int(a[0]) + 1
        except ValueError: return 0
    runs = m.uid.str.split('-').str[2:].str.join('-') if len(m) else []
    exo.append(m.score.values[int(np.argmax([span(r) for r in runs]))] if len(m) else np.nan); exlab.append(','.join(sorted(set(m.label))) if len(m) else ''); nmatch.append(len(m))
tt['y'] = y[te]; tt['ours'] = ours; tt['ours_gb'] = raw['gb']; tt['exominer'] = exo; tt['exominer_labels'] = exlab; tt['n_tce'] = nmatch
both = tt[tt.exominer.notna()]
res = dict(test_signals=len(tt), matched=len(both), matched_planets=int(both.y.sum()), matched_fp=int((1 - both.y).sum()),
           ours_pr_auc=ap(both.y, both.ours), ours_roc_auc=auc(both.y, both.ours),
           exominer_pr_auc=ap(both.y, both.exominer), exominer_roc_auc=auc(both.y, both.exominer),
           ours_all_test_pr_auc=ap(tt.y, tt.ours))
for name, col in (('ours', 'ours'), ('exominer', 'exominer')):
    p = both[col] >= 0.5
    res[f'{name}_at_0.5'] = dict(planets_kept=f"{int((p & (both.y == 1)).sum())}/{int(both.y.sum())}", false_alarms_caught=f"{int((~p & (both.y == 0)).sum())}/{int((both.y == 0).sum())}")
# label agreement: how often their training label differs from ours
both_lab = both.exominer_labels.str.split(',').str[0]
res['label_disagreement'] = int(((both_lab.isin(['KP', 'CP'])) != (both.y == 1)).sum())
# average of the two
res['average_ours_exominer_pr_auc'] = ap(both.y, (both.ours + both.exominer) / 2)
print(json.dumps(res, indent=1, default=float)); tt.to_csv('compare_test.csv', index=False); json.dump(res, open('compare.json', 'w'), indent=1, default=float)
