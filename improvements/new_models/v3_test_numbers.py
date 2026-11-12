import sys, json, numpy as np, pandas as pd, torch, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/user/tv/src'); sys.path.insert(0, '/home/user/tv/repo')
import tv_common as c
from transitvet import pipeline as P
from sklearn.metrics import average_precision_score as ap
MD = '/home/user/tv/repo/transitvet/models'; NEW = ['cen_sigma', 'cen_offset_arcsec', 'rp_est_re', 'contratio']
sc = P.Scorer(MD); lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
rng = np.random.default_rng(0); out = {}
for mis in ('kepler', 'tess'):
    if mis == 'kepler':
        G, L, t = c.load_kepler(); te = np.array(json.load(open('/home/user/tv/e2/kepler_star_split.json'))['test'])
    else:
        G, L, t = c.load_tess(); lab = np.where(t.label.notna())[0]; G, L, t = G[lab], L[lab], t.iloc[lab].reset_index(drop=True)
        te = np.array(json.load(open('/home/user/tv/e4/tess_star_split.json'))['test'])
    y = t.label.values[te].astype(int)
    X = np.hstack([c.feature_matrix(t), pd.read_csv(f'/home/user/tv/imp/new_feats_{mis}.csv')[NEW].values])
    sc._load_nets(mis)
    raw = {'gb': np.mean([g.predict_proba(X[te])[:, 1] for g in sc.gb[mis]], 0)}
    g, l = torch.tensor(G[te]), torch.tensor(L[te])
    with torch.no_grad():
        for k in sc.net_kinds: raw[k] = np.mean([np.concatenate([torch.sigmoid(n(g[i:i+256], l[i:i+256])).numpy() for i in range(0, len(te), 256)]) for n in sc.nets[(mis, k)]], 0)
    z = np.column_stack([lg(raw[k]) for k in sc.combine_kinds])
    comb = sc.comb[mis].predict_proba(z)[:, 1]
    cp = sc.conf[mis]['combiner'].predict_proba(z)[:, 1]
    calls = [P.triage(float(p), sc.conf[mis], 0.05) for p in cp]
    calls = [x['call'] if isinstance(x, dict) else x for x in calls]
    r = dict(pr_auc=ap(y, comb), single={k: ap(y, v) for k, v in raw.items()}, calls=pd.Series(calls).value_counts().to_dict())
    if mis == 'tess':
        ct = pd.read_csv('/home/user/tv/exo/compare_test.csv'); assert (ct.toi.values == t.toi.values[te]).all()
        m = ct.exominer.notna().values; e = ct.exominer.values[m]; o = comb[m]; yy = y[m]
        r.update(n=int(m.sum()), exo=ap(yy, e), ours=ap(yy, o), avg=ap(yy, (e + o) / 2))
        d = []
        for _ in range(2000):
            b = rng.integers(0, len(yy), len(yy)); d.append(ap(yy[b], o[b]) - ap(yy[b], e[b]))
        r['diff_ci'] = np.percentile(d, [2.5, 97.5]).tolist()
    out[mis] = r; print(mis, r, flush=True)
