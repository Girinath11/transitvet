"""Pixel check on labelled TESS signals: choose the warning thresholds on VALIDATION, measure on TEST."""
import sys, json, numpy as np, pandas as pd, joblib, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/user/tv/src'); sys.path.insert(0, '/home/user/tv/app'); sys.path.insert(0, '/home/user/tv/conf')
import tv_common as c
mis = sys.argv[1] if len(sys.argv) > 1 else 'tess'
rows = []
for l in open(f'/home/user/tv/pix/pix_{mis}.jsonl'):
    j = json.loads(l)
    for s, r in zip(j['signals'], j.get('results', [{}] * len(j['signals']))):
        rows.append(dict(star=j['star'], split=j['split'], id=s['id'], y=s['y'], lab=s['lab'], off=r.get('px_offset_arcsec', np.nan),
                         sig=r.get('px_sigma', np.nan), n=r.get('px_n_segments', 0), err=j.get('error')))
d = pd.DataFrame(rows).astype({'off': float, 'sig': float})
# combined scores (deployed combiner) for the same signals
import importlib.util
G, L, t = c.load_tess() if mis == 'tess' else c.load_kepler()
if mis == 'tess':
    lab = np.where(t.label.notna())[0]; G, L, t = G[lab], L[lab], t.iloc[lab].reset_index(drop=True); key = t.toi.astype(str).values
else:
    key = (t.kepid.astype(str) + '_' + t.tce_plnt_num.astype(str)).values
src = open('/home/user/tv/conf/conformal.py').read().split('res = {}')[0]
exec(src)
nf = pd.read_csv(f'/home/user/tv/imp/new_feats_{mis}.csv'); X = np.hstack([c.feature_matrix(t), nf[NEW].values])
pos = {k: i for i, k in enumerate(key)}
ix = np.array([pos[k] for k in d.id])
Z = raw(mis, G, L, X, ix)
d['p'] = joblib.load(f'{MD}/combiner.joblib')[mis].predict_proba(Z)[:, 1]
d.to_csv(f'/home/user/tv/pix/pix_{mis}_signals.csv', index=False)
print('signals', len(d), d.groupby('split').size().to_dict(), 'measured (n>0):', (d.n > 0).mean().round(3))
v, te = d[d.split == 'val'], d[d.split == 'test']
def rate(df, s, o):
    f = (df.sig > s) & (df.off > o)
    return f[df.y == 1].mean(), f[df.y == 0].mean()
print('VALIDATION grid: sigma, arcsec -> planets flagged, false alarms flagged')
best = None
for s in (2, 3, 4, 5, 7, 10):
    for o in ((2, 4, 6, 8, 12) if mis == 'kepler' else (10, 15, 21, 30, 42)):
        rp, rf = rate(v, s, o)
        print(f'  {s:>3} {o:>3}  {rp:.3f}  {rf:.3f}')
        if rp <= 0.02 and (best is None or rf > best[2]):
            best = (s, o, rf)
print('chosen (<=2% planets flagged on validation):', best)
s, o = best[:2]
rp, rf = rate(te, s, o)
print(f'TEST: planets flagged {rp:.3f} ({int(((te.sig > s) & (te.off > o) & (te.y == 1)).sum())}/{int((te.y == 1).sum())}), '
      f'false alarms flagged {rf:.3f} ({int(((te.sig > s) & (te.off > o) & (te.y == 0)).sum())}/{int((te.y == 0).sum())})')
fl = (te.sig > s) & (te.off > o)
cand = te.p >= 0.5
print('TEST false alarms called candidate (p>=0.5): before', int((cand & (te.y == 0)).sum()), 'after rule (flag -> not candidate)', int((cand & ~fl & (te.y == 0)).sum()))
print('TEST planets called candidate: before', int((cand & (te.y == 1)).sum()), 'after', int((cand & ~fl & (te.y == 1)).sum()), 'of', int((te.y == 1).sum()))
from sklearn.metrics import average_precision_score as ap
print('TEST PR-AUC combined', round(ap(te.y, te.p), 4), ' with flag (p x 0.2 if flagged)', round(ap(te.y, np.where(fl, te.p * 0.2, te.p)), 4))
json.dump(dict(chosen=dict(sigma=s, arcsec=o)), open(f'/home/user/tv/pix/chosen_{mis}.json', 'w'))
