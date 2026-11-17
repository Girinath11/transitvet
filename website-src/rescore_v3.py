"""Re-score the same 70 showcase signals with the current model set (v4: gradient boosting + Mamba-2 + Transformer; TESS GB also reads Gaia)."""
import sys, json, numpy as np, pandas as pd, joblib, torch, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/user/tv/src'); sys.path.insert(0, '/home/user/tv/repo')
import tv_common as c
from transitvet import pipeline as P
MD = '/home/user/tv/repo/transitvet/models'; NEW = ['cen_sigma', 'cen_offset_arcsec', 'rp_est_re', 'contratio']
sc = P.Scorer(MD); lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
S = json.load(open('signals.json'))
for mis in ('kepler', 'tess'):
    if mis == 'kepler':
        G, L, t = c.load_kepler(); te = np.array(json.load(open('/home/user/tv/e2/kepler_star_split.json'))['test']); sid = t.kepid.values
    else:
        G, L, t = c.load_tess(); lab = np.where(t.label.notna())[0]; G, L, t = G[lab], L[lab], t.iloc[lab].reset_index(drop=True)
        te = np.array(json.load(open('/home/user/tv/e4/tess_star_split.json'))['test']); sid = t.tid.values.astype(int)
    X = np.hstack([c.feature_matrix(t), pd.read_csv(f'/home/user/tv/imp/new_feats_{mis}.csv')[NEW].values])
    extra = sc.manifest.get('feats_extra', {}).get(mis, [])
    if extra:      # v4: TESS gradient boosting also reads Gaia numbers (same table as training, transitvet/gaia.py)
        g = pd.read_csv('/home/user/tv/gaia/gaia_star_feats.csv').set_index('tid').reindex(t.tid.astype('int64').values)
        g['g_log_nb_depth'] = np.log10(np.clip(t.depth_ppm.values * 1e-6 * (1 + g.g_fr42.values) / np.maximum(g.g_nb_fr.values, 1e-6), 1e-6, 1e3))
        X = np.hstack([X, g[extra].astype(float).values])
    items = [s for s in S if s['mission'] == mis]; ix = []
    for s in items:
        num = int(s['star'].split()[1]); m = [i for i in te if sid[i] == num and abs(t.tce_period.values[i] - s['period']) < 1e-3]
        assert len(m) >= 1, s["name"]; ix.append(m[0])
    ix = np.array(ix); sc._load_nets(mis)
    raw = {'gb': np.mean([g.predict_proba(X[ix])[:, 1] for g in sc.gb[mis]], 0)}
    g, l = torch.tensor(G[ix]), torch.tensor(L[ix])
    with torch.no_grad():
        for k in sc.net_kinds: raw[k] = np.mean([torch.sigmoid(n(g, l)).numpy() for n in sc.nets[(mis, k)]], 0)
    out = {k: sc.cal[f'{mis}_{k}'].predict(v) for k, v in raw.items()}
    out['combined'] = sc.comb[mis].predict_proba(np.column_stack([lg(raw[k]) for k in sc.combine_kinds]))[:, 1]
    for j, s in enumerate(items):
        s['s'] = {k: round(float(v[j]), 3) for k, v in out.items()}
json.dump(S, open('signals.json', 'w'))
print(pd.Series([(s['mission'], s['label'], (s['s']['combined'] >= .5) == s['label']) for s in S]).value_counts().to_dict())
