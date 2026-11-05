"""Experiment (5 Nov 2026): does training the TESS gradient-boosting model on Kepler + TESS training stars help TESS?
Selection on TESS VALIDATION stars only; TESS TEST reported once at the end."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, '/home/user/tv/src')
import tv_common as c
from sklearn.ensemble import HistGradientBoostingClassifier as H
from sklearn.metrics import average_precision_score as ap, roc_auc_score as auc
NEW = ['cen_sigma', 'cen_offset_arcsec', 'rp_est_re', 'contratio']
def feats(mis):
    if mis == 'kepler':
        G, L, t = c.load_kepler(); sp = json.load(open('/home/user/tv/e2/kepler_star_split.json'))
    else:
        G, L, t = c.load_tess(); lab = np.where(t.label.notna())[0]; t = t.iloc[lab].reset_index(drop=True)
        sp = json.load(open('/home/user/tv/e4/tess_star_split.json'))
    nf = pd.read_csv(f'/home/user/tv/imp/new_feats_{mis}.csv'); assert len(nf) == len(t)
    X = np.hstack([c.feature_matrix(t), nf[NEW].astype(float).values]); y = t.label.values.astype(int)
    return X, y, {k: np.array(v) for k, v in sp.items() if k in ('train', 'val', 'test')}
Xk, yk, sk = feats('kepler'); Xt, yt, st = feats('tess')
def fit(X, y, w=None):
    return [H(max_iter=500, learning_rate=0.05, early_stopping=True, validation_fraction=0.1, random_state=s).fit(X, y, sample_weight=w) for s in range(5)]
def pred(gbs, X): return np.mean([g.predict_proba(X)[:, 1] for g in gbs], 0)
flag = lambda X, v: np.hstack([X, np.full((len(X), 1), v)])
out = {}
tr, va, te = st['train'], st['val'], st['test']
variants = {}
variants['tess_only'] = (fit(flag(Xt[tr], 1), yt[tr]), 1)
Xj = np.vstack([flag(Xk[sk['train']], 0), flag(Xt[tr], 1)]); yj = np.concatenate([yk[sk['train']], yt[tr]])
variants['joint'] = (fit(Xj, yj), 1)
for wt in (3.0, 10.0):
    w = np.concatenate([np.ones(len(sk['train'])), np.full(len(tr), wt)])
    variants[f'joint_w{int(wt)}'] = (fit(Xj, yj, w), 1)
for name, (gbs, _) in variants.items():
    pv, pt = pred(gbs, flag(Xt[va], 1)), pred(gbs, flag(Xt[te], 1))
    out[name] = dict(val_pr_auc=round(ap(yt[va], pv), 4), val_roc=round(auc(yt[va], pv), 4), test_pr_auc=round(ap(yt[te], pt), 4), test_roc=round(auc(yt[te], pt), 4))
    np.save(f'/home/user/tv/imp2/{name}_val.npy', pv); np.save(f'/home/user/tv/imp2/{name}_test.npy', pt)
    print(name, out[name], flush=True)
json.dump(out, open('/home/user/tv/imp2/joint.json', 'w'), indent=1)
