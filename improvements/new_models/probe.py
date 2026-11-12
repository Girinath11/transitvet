"""MOMENT (frozen) + a small classifier, same star splits as every other model.
Classifier and its setting are chosen on VALIDATION stars; TEST reported once."""
import sys, json, numpy as np
sys.path.insert(0, '/home/user/tv/src'); import tv_common as c
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import average_precision_score as ap
mis = sys.argv[1]; E = np.load(f'/home/user/tv/fm/moment_{mis}.npz'); X = np.hstack([E['g'], E['l']])
if mis == 'kepler':
    _, _, t = c.load_kepler(); y = t.y.values.astype(int); sp = json.load(open('/home/user/tv/e2/kepler_star_split.json'))
else:
    _, _, t = c.load_tess(); lab = np.where(t.label.notna())[0]; X, t = X[lab], t.iloc[lab].reset_index(drop=True)
    y = t.label.values.astype(int); sp = json.load(open('/home/user/tv/e4/tess_star_split.json'))
tr, va, te = map(np.array, (sp['train'], sp['val'], sp['test']))
cands = {f'logreg_C{C}': make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)) for C in (0.003, 0.01, 0.03, 0.1)}
for h in (128, 256):
    cands[f'mlp_{h}'] = make_pipeline(StandardScaler(), MLPClassifier((h,), alpha=1e-2, early_stopping=True, max_iter=300, random_state=0))
val = {}
for k, m in cands.items():
    m.fit(X[tr], y[tr]); val[k] = ap(y[va], m.predict_proba(X[va])[:, 1]); print(mis, k, 'val PR-AUC', round(val[k], 4), flush=True)
best = max(val, key=val.get); m = cands[best]
pv, pt = m.predict_proba(X[va])[:, 1], m.predict_proba(X[te])[:, 1]
r = c.metrics(y[te], pt, c.best_threshold(y[va], pv)); r.update(chosen=best, val_pr_auc=val[best])
np.save(f'/home/user/tv/fm/moment_{mis}_test_probs.npy', pt); np.save(f'/home/user/tv/fm/moment_{mis}_val_probs.npy', pv)
print('TEST', {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
json.dump(dict(val=val, test=r), open(f'/home/user/tv/fm/moment_{mis}.json', 'w'), indent=1, default=float)
