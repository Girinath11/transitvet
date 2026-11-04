"""Step 2 of the models (3 Nov 2026): gradient boosting retrained with the new checks
(light position, planet size, crowding) and a combiner that joins GB + CNN + Transformer.
GB: fitted on TRAIN stars. Combiner (logistic regression on the three models' log-odds) and the
isotonic calibrators: fitted on VALIDATION stars. CNN / Transformer weights unchanged.
Usage: python build_models2.py kepler tess   (needs /home/user/tv/imp/new_feats_<mission>.csv from imp/eval_new.py)"""
import sys, json, joblib, numpy as np, pandas as pd, torch, shutil, os
sys.path.insert(0, '/home/user/tv/src'); sys.path.insert(0, '/home/user/tv/app')
import tv_common as c
from transitvet import tv_models as M
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score as ap
OUT = '/home/user/tv/app/transitvet/models'; NEW = ['cen_sigma', 'cen_offset_arcsec', 'rp_est_re', 'contratio']
torch.set_num_threads(4)
cal = joblib.load(f'{OUT}/calibrators.joblib'); man = json.load(open(f'{OUT}/manifest.json'))
comb = joblib.load(f'{OUT}/combiner.joblib') if os.path.exists(f'{OUT}/combiner.joblib') else {}
def lg(p): p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))
for mis in sys.argv[1:]:
    if mis == 'kepler':
        G, L, t = c.load_kepler(); sp = json.load(open('/home/user/tv/e2/kepler_star_split.json'))
    else:
        G, L, t = c.load_tess(); lab = np.where(t.label.notna())[0]; G, L, t = G[lab], L[lab], t.iloc[lab].reset_index(drop=True)
        sp = json.load(open('/home/user/tv/e4/tess_star_split.json'))
    tr, va, te = map(np.array, (sp['train'], sp['val'], sp['test'])); y = t.label.values.astype(int)
    nf = pd.read_csv(f'/home/user/tv/imp/new_feats_{mis}.csv'); assert len(nf) == len(t)
    X = np.hstack([c.feature_matrix(t), nf[NEW].astype(float).values])
    if not os.path.exists(f'{OUT}/{mis}_gb_v1.joblib'):
        shutil.copy(f'{OUT}/{mis}_gb.joblib', f'{OUT}/{mis}_gb_v1.joblib')
    gbs = [HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, early_stopping=True, validation_fraction=0.1, random_state=s).fit(X[tr], y[tr]) for s in range(5)]
    joblib.dump(gbs, f'{OUT}/{mis}_gb.joblib')
    raw = {'gb': np.mean([g.predict_proba(X[va])[:, 1] for g in gbs], 0)}
    g, l = torch.tensor(G[va]), torch.tensor(L[va])
    for kind, cls in (('cnn', M.AstroNet), ('tf', M.PatchTransformer)):
        ps = []
        for sd in torch.load(f'{OUT}/{mis}_{kind}.pt'):
            m = cls(); m.load_state_dict(sd); m.eval()
            with torch.no_grad(): ps.append(np.concatenate([torch.sigmoid(m(g[i:i + 512], l[i:i + 512])).numpy() for i in range(0, len(va), 512)]))
        raw[kind] = np.mean(ps, 0)
    cal[f'{mis}_gb'] = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1).fit(raw['gb'], y[va])
    Z = np.column_stack([lg(raw[k]) for k in ('gb', 'cnn', 'tf')])
    comb[mis] = LogisticRegression(C=1.0).fit(Z, y[va])
    man['models'][f'{mis}_gb'] = dict(val_pr_auc=float(ap(y[va], raw['gb'])), n_val=int(len(va)), feats='FEATS + new checks')
    man['models'][f'{mis}_combined'] = dict(val_pr_auc=float(ap(y[va], comb[mis].predict_proba(Z)[:, 1])), n_val=int(len(va)),
                                           coef_gb_cnn_tf=comb[mis].coef_.round(3).tolist()[0])
    print(mis, man['models'][f'{mis}_gb'], man['models'][f'{mis}_combined'], flush=True)
man['feats'] = c.FEATS + NEW; man['built_v2'] = '2026-11-03'
man['note'] += ' v2: GB retrained with light-position (centroid), planet-size and crowding features; combined score = logistic regression on GB, CNN, Transformer log-odds, fitted on validation stars.'
joblib.dump(cal, f'{OUT}/calibrators.joblib'); joblib.dump(comb, f'{OUT}/combiner.joblib')
json.dump(man, open(f'{OUT}/manifest.json', 'w'), indent=1); print('ALLDONE')
