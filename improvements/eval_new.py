"""Do the new checks (light position, planet size, crowding) improve the verdict?
Same splits as before (Kepler by star 70/15/15; TESS by star 50/10/40). Everything fitted on TRAIN
(models) or VALIDATION (combiner, thresholds); scored once on TEST."""
import sys, json, gzip, pickle, os, numpy as np, pandas as pd, joblib, torch, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/user/tv/src'); sys.path.insert(0, '/home/user/tv/app')
import tv_common as c, tv_centroid as C
from transitvet import tv_models as M
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score as ap, roc_auc_score
torch.set_num_threads(4)
MD = '/home/user/tv/app/transitvet/models'
NEW = ['cen_sigma', 'cen_offset_arcsec', 'rp_est_re', 'contratio']
tic = pd.read_csv('/home/user/tv/out/labels/tic_props.csv')
tic = tic.sort_values('rad', na_position='last')
by_kic = tic.dropna(subset=['KIC']).drop_duplicates('KIC').set_index(tic.dropna(subset=['KIC']).drop_duplicates('KIC').KIC.astype('int64'))
by_tic = tic.set_index('ID')

def add_new(t, mis):
    sid = t.kepid if mis == 'kepler' else t.tid; key = t.tce_plnt_num if mis == 'kepler' else t.toi
    cs, co, n_ok = [], [], 0
    cache = {}
    for s, k, d in zip(sid.astype('int64'), key, t.depth_ppm):
        if s not in cache:
            p = f'/home/user/tv/out/cen/{mis}/{s}.pkl.gz'
            cache[s] = pickle.load(gzip.open(p)) if os.path.exists(p) else None
        r = cache[s]
        if isinstance(r, dict) and k in r['shifts']:
            a, b = C.combine(r['shifts'][k], d, mis); n_ok += 1
        else:
            a, b = np.nan, np.nan
        cs.append(a); co.append(b)
    t = t.copy(); t['cen_sigma'] = cs; t['cen_offset_arcsec'] = co
    props = (by_kic if mis == 'kepler' else by_tic).reindex(sid.astype('int64').values)
    rad = props.rad.values; t['contratio'] = props.contratio.values; t['star_rad'] = rad
    t['rp_est_re'] = np.sqrt(np.clip(t.depth_ppm.values, 0, None) * 1e-6) * rad * 109.1
    print(mis, 'centroid available for', n_ok, 'of', len(t), '; star radius for', np.isfinite(rad).sum(), flush=True)
    return t

def nets(mis, G, L, ix):
    out = {}
    g, l = torch.tensor(G[ix]), torch.tensor(L[ix])
    for kind, cls in (('cnn', M.AstroNet), ('tf', M.PatchTransformer)):
        ps = []
        for sd in torch.load(f'{MD}/{mis}_{kind}.pt'):
            m = cls(); m.load_state_dict(sd); m.eval()
            with torch.no_grad():
                ps.append(np.concatenate([torch.sigmoid(m(g[i:i + 512], l[i:i + 512])).numpy() for i in range(0, len(ix), 512)]))
        out[kind] = np.mean(ps, 0)
    return out

def gb_fit(X, y, tr):
    return [HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, early_stopping=True, validation_fraction=0.1, random_state=s).fit(X[tr], y[tr]) for s in range(5)]
def gb_pred(gbs, X, ix): return np.mean([g.predict_proba(X[ix])[:, 1] for g in gbs], 0)
def lg(p): p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))

def report(name, y_v, p_v, y_t, p_t):
    thr = c.best_threshold(y_v, p_v); pred = p_t >= thr
    tp = (pred & (y_t == 1)).sum(); fp = (pred & (y_t == 0)).sum(); fn = (~pred & (y_t == 1)).sum(); tn = (~pred & (y_t == 0)).sum()
    return dict(model=name, pr_auc=round(ap(y_t, p_t), 4), roc_auc=round(roc_auc_score(y_t, p_t), 4),
                precision=round(tp / max(tp + fp, 1), 3), recall=round(tp / max(tp + fn, 1), 3),
                false_alarms_caught=f'{tn}/{tn + fp}', planets_kept=f'{tp}/{tp + fn}')

res = {}
for mis in sys.argv[1:]:
    if mis == 'kepler':
        G, L, t = c.load_kepler(); sp = json.load(open('/home/user/tv/e2/kepler_star_split.json'))
        tr, va, te = map(np.array, (sp['train'], sp['val'], sp['test'])); y = t.label.values.astype(int)
    else:
        G, L, t = c.load_tess(); lab = np.where(t.label.notna())[0]
        G, L, t = G[lab], L[lab], t.iloc[lab].reset_index(drop=True); y = t.label.values.astype(int)
        sp = json.load(open('/home/user/tv/e4/tess_star_split.json')); tr, va, te = map(np.array, (sp['train'], sp['val'], sp['test']))
    t = add_new(t, mis)
    X0 = c.feature_matrix(t); X1 = np.hstack([X0, t[NEW].astype(float).values])
    gb0 = joblib.load(f'{MD}/{mis}_gb.joblib'); gb1 = gb_fit(X1, y, tr)
    nv, nt = nets(mis, G, L, va), nets(mis, G, L, te)
    P = {'gb_old': (gb_pred(gb0, X0, va), gb_pred(gb0, X0, te)), 'gb_new': (gb_pred(gb1, X1, va), gb_pred(gb1, X1, te)),
         'cnn': (nv['cnn'], nt['cnn']), 'tf': (nv['tf'], nt['tf'])}
    rows = [report(k, y[va], P[k][0], y[te], P[k][1]) for k in P]
    for name, keys in (('combined_old (gb_old+cnn+tf)', ['gb_old', 'cnn', 'tf']), ('combined_new (gb_new+cnn+tf)', ['gb_new', 'cnn', 'tf'])):
        Zv = np.column_stack([lg(P[k][0]) for k in keys]); Zt = np.column_stack([lg(P[k][1]) for k in keys])
        lr = LogisticRegression(C=1.0).fit(Zv, y[va]); rows.append(report(name, y[va], lr.predict_proba(Zv)[:, 1], y[te], lr.predict_proba(Zt)[:, 1]))
        P[name] = (lr.predict_proba(Zv)[:, 1], lr.predict_proba(Zt)[:, 1]); P[name + '_coef'] = lr.coef_.round(3).tolist()
    # which new checks matter: drop each one from gb_new (retrain) and see test PR-AUC
    drops = {}
    for j, f in enumerate(NEW):
        keep = [i for i in range(X1.shape[1]) if i != X0.shape[1] + j]
        drops[f] = round(ap(y[te], gb_pred(gb_fit(X1[:, keep], y, tr), X1[:, keep], te)), 4)
    res[mis] = dict(n=dict(train=len(tr), val=len(va), test=len(te)), rows=rows, drop_one_gb_new_pr_auc=drops,
                    combiner_coef={k: P[k] for k in P if k.endswith('_coef')})
    print(mis, pd.DataFrame(rows).to_string(index=False), '\ndrop one:', drops, flush=True)
    t[['cen_sigma', 'cen_offset_arcsec', 'rp_est_re', 'contratio', 'star_rad', 'label']].to_csv(f'/home/user/tv/imp/new_feats_{mis}.csv', index=False)
json.dump(res, open(f'/home/user/tv/imp/eval_new_{"_".join(sys.argv[1:])}.json', 'w'), indent=1, default=str)
print('ALLDONE')
