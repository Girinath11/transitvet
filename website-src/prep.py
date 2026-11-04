"""Sample of real held-out TEST signals for the showcase page (Kepler DR24 test split, TESS TOI test split)."""
import sys, json, numpy as np, pandas as pd, joblib, torch, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/user/tv/src'); sys.path.insert(0, '/home/user/tv/app')
import tv_common as c
from transitvet import tv_models as M
MD = '/home/user/tv/app/transitvet/models'; NEW = ['cen_sigma', 'cen_offset_arcsec', 'rp_est_re', 'contratio']
cal = joblib.load(f'{MD}/calibrators.joblib'); comb = joblib.load(f'{MD}/combiner.joblib')
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
koi = pd.read_csv('/home/user/tv/out/labels/cumulative.csv', usecols=['kepid', 'kepoi_name', 'kepler_name', 'koi_period', 'koi_disposition'])
exo = pd.read_csv('/home/user/tv/exo/compare_test.csv')
def score(mis, G, L, X, ix):
    raw = {'gb': np.mean([g.predict_proba(X[ix])[:, 1] for g in joblib.load(f'{MD}/{mis}_gb.joblib')], 0)}
    g, l = torch.tensor(G[ix]), torch.tensor(L[ix])
    for kind, cls in (('cnn', M.AstroNet), ('tf', M.PatchTransformer)):
        ps = []
        for sd in torch.load(f'{MD}/{mis}_{kind}.pt'):
            m = cls(); m.load_state_dict(sd); m.eval()
            with torch.no_grad(): ps.append(np.concatenate([torch.sigmoid(m(g[i:i+512], l[i:i+512])).numpy() for i in range(0, len(ix), 512)]))
        raw[kind] = np.mean(ps, 0)
    out = {k: cal[f'{mis}_{k}'].predict(v) for k, v in raw.items()}
    out['combined'] = comb[mis].predict_proba(np.column_stack([lg(raw[k]) for k in ('gb', 'cnn', 'tf')]))[:, 1]
    return out
rng = np.random.default_rng(7); items = []
for mis in ('kepler', 'tess'):
    if mis == 'kepler':
        G, L, t = c.load_kepler(); te = np.array(json.load(open('/home/user/tv/e2/kepler_star_split.json'))['test'])
    else:
        G, L, t = c.load_tess(); lab = np.where(t.label.notna())[0]; G, L, t = G[lab], L[lab], t.iloc[lab].reset_index(drop=True)
        te = np.array(json.load(open('/home/user/tv/e4/tess_star_split.json'))['test'])
    nf = pd.read_csv(f'/home/user/tv/imp/new_feats_{mis}.csv'); X = np.hstack([c.feature_matrix(t), nf[NEW].values])
    sc = score(mis, G, L, X, te); y = t.label.values[te].astype(int); pred = sc['combined'] >= 0.5
    # balanced sample: planets right, false alarms right, and a few honest mistakes
    groups = [((y == 1) & pred, 14 if mis == 'kepler' else 16), ((y == 0) & ~pred, 14 if mis == 'kepler' else 14),
              ((y == 1) & ~pred, 3), ((y == 0) & pred, 3)]
    if mis == 'tess':   # prefer TESS signals with a NASA ExoMiner++ score
        has = np.isin(t.toi.values[te], exo[exo.exominer.notna()].toi.values)
    for mask, n in groups:
        cand = np.where(mask & (has if mis == 'tess' else True))[0]
        for j in rng.choice(cand, min(n, len(cand)), replace=False):
            i = te[j]; r = t.iloc[i]
            if mis == 'kepler':
                k = koi[(koi.kepid == r.kepid) & (np.abs(koi.koi_period - r.tce_period) / r.tce_period < 0.01)]
                name = (k.kepler_name.dropna().iloc[0] if k.kepler_name.notna().any() else k.kepoi_name.iloc[0]) if len(k) else f'KIC {r.kepid} signal {r.tce_plnt_num}'
                star = f'KIC {r.kepid}'; truth = {'PC': 'planet candidate', 'AFP': 'false alarm (astrophysical)', 'NTP': 'false alarm (not transit-like)'}[r.av_training_set]
                ex = None
            else:
                name = f'TOI-{r.toi}'; star = f'TIC {int(r.tid)}'
                truth = {'CP': 'confirmed planet', 'KP': 'known planet', 'FP': 'false positive', 'FA': 'false alarm'}[r.tfopwg_disp]
                e = exo[exo.toi == r.toi]; ex = float(e.exominer.iloc[0]) if len(e) and e.exominer.notna().iloc[0] else None
            g = G[i].reshape(-1)[:2000].reshape(400, 5).mean(1)
            f = lambda v: None if v is None or not np.isfinite(v) else round(float(v), 3)
            items.append(dict(mission=mis, name=str(name), star=star, truth=truth, label=int(y[j]), period=round(float(r.tce_period), 5),
                              duration_h=round(float(r.tce_duration) * 24, 2), depth=f(r.depth_ppm), odd_even=f(r.odd_even_sigma),
                              secondary=f(r.secondary_sigma), n_transits=int(r.n_transits) if np.isfinite(r.n_transits) else None,
                              size_re=f(nf.rp_est_re.iloc[i]), offset=f(nf.cen_offset_arcsec.iloc[i]), crowd=f(nf.contratio.iloc[i]),
                              s={k: round(float(v[j]), 3) for k, v in sc.items()}, exo=ex,
                              g=np.round(g, 3).tolist(), l=np.round(L[i], 3).tolist()))
rng.shuffle(items)
json.dump(items, open('/home/user/tv/show/signals.json', 'w'))
print(len(items), pd.Series([(i['mission'], i['label'], (i['s']['combined'] >= .5) == i['label']) for i in items]).value_counts().to_dict())
