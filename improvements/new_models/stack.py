"""Does an extra model add anything to TransitVet's combined score? Same recipe as the shipped combiner:
logistic regression on the models' log-odds, fitted on VALIDATION stars, TEST reported once.
Usage: python stack.py kepler moment   (extra probs from fm/moment_<mis>_{val,test}_probs.npy)"""
import sys, json, joblib, numpy as np, pandas as pd, torch
sys.path.insert(0, '/home/user/tv/src'); sys.path.insert(0, '/home/user/tv/app')
import tv_common as c
from transitvet import tv_models as M
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score as ap
mis, extra = sys.argv[1], sys.argv[2:]
OUT = '/home/user/tv/app/transitvet/models'; NEW = ['cen_sigma', 'cen_offset_arcsec', 'rp_est_re', 'contratio']
if mis == 'kepler':
    G, L, t = c.load_kepler(); sp = json.load(open('/home/user/tv/e2/kepler_star_split.json'))
else:
    G, L, t = c.load_tess(); lab = np.where(t.label.notna())[0]; G, L, t = G[lab], L[lab], t.iloc[lab].reset_index(drop=True)
    sp = json.load(open('/home/user/tv/e4/tess_star_split.json'))
y = t.label.values.astype(int); va, te = np.array(sp['val']), np.array(sp['test'])
X = np.hstack([c.feature_matrix(t), pd.read_csv(f'/home/user/tv/imp/new_feats_{mis}.csv')[NEW].astype(float).values])
gbs = joblib.load(f'{OUT}/{mis}_gb.joblib'); raw = {}
for nm, ix in (('va', va), ('te', te)):
    raw[('gb', nm)] = np.mean([g.predict_proba(X[ix])[:, 1] for g in gbs], 0)
    for kind, cls in (('cnn', M.AstroNet), ('tf', M.PatchTransformer)):
        ps = []
        for sd in torch.load(f'{OUT}/{mis}_{kind}.pt'):
            m = cls(); m.load_state_dict(sd); m.eval()
            with torch.no_grad(): ps.append(np.concatenate([torch.sigmoid(m(torch.tensor(G[ix[i:i+512]]), torch.tensor(L[ix[i:i+512]]))).numpy() for i in range(0, len(ix), 512)]))
        raw[(kind, nm)] = np.mean(ps, 0)
for e in extra:
    raw[(e, 'va')] = np.load(f'/home/user/tv/fm/{e}_{mis}_val_probs.npy'); raw[(e, 'te')] = np.load(f'/home/user/tv/fm/{e}_{mis}_test_probs.npy')
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
rng = np.random.default_rng(0); stars = t.star.values[te] if 'star' in t else None
res = {}
combos = [['gb', 'cnn', 'tf'], ['gb', 'cnn', 'tf'] + extra] + ([[ 'gb', 'mamba', 'tf'], ['gb', 'mamba'], ['gb', 'cnn']] if 'mamba' in extra else [])
for name, ks in [('+'.join(k), k) for k in combos]:
    Zv = np.column_stack([lg(raw[(k, 'va')]) for k in ks]); Zt = np.column_stack([lg(raw[(k, 'te')]) for k in ks])
    m = LogisticRegression(C=1.0).fit(Zv, y[va]); p = m.predict_proba(Zt)[:, 1]
    res[name] = dict(test_pr_auc=round(ap(y[te], p), 4), coef=m.coef_.round(3).tolist()[0]); res[name + '_p'] = p
for k in ['gb', 'cnn', 'tf'] + extra: res['single ' + k] = round(ap(y[te], raw[(k, 'te')]), 4)
ps = [res['gb+cnn+tf_p'], res['+'.join(['gb', 'cnn', 'tf'] + extra) + '_p']]
base = ps[0]
if stars is not None:
    us = np.unique(stars); idx = {s: np.where(stars == s)[0] for s in us}; d = []
    for _ in range(1000):
        ii = np.concatenate([idx[s] for s in rng.choice(us, len(us))]); d.append(ap(y[te][ii], ps[1][ii]) - ap(y[te][ii], ps[0][ii]))
    res['gain_CI95'] = np.percentile(d, [2.5, 97.5]).round(4).tolist()
for k in [k for k in res if k.endswith('_p')]: del res[k]
print(json.dumps(res, indent=1)); json.dump(res, open(f'/home/user/tv/fm/stack_{mis}_{"_".join(extra)}.json', 'w'), indent=1)
