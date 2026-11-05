"""Experiment (5 Nov 2026): add the TESS pixel (difference-image) check to the final combiner.
The combiner is already fitted on VALIDATION stars; here it also gets the pixel numbers. TEST is reported once."""
import json, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score as ap, roc_auc_score as auc
d = pd.read_csv('/home/user/tv/repo/validation/pixel_tess_signals.csv')
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
def X(df, pix=True):
    cols = [lg(df.p.values)]
    if pix:
        has = df.off.notna().values.astype(float)
        cols += [has, has * np.log1p(np.clip(df.sig.fillna(0).values, 0, 100)), has * np.log1p(df.off.fillna(0).values),
                 has * ((df.sig.fillna(0) > 5) & (df.off.fillna(0) > 10)).values]
    return np.column_stack(cols)
va, te = d[d.split == 'val'], d[d.split == 'test']
out = {'n_val': len(va), 'n_test': len(te), 'pixel_available_test': round(te.off.notna().mean(), 3)}
base_te = te.p.values
out['base'] = dict(test_pr_auc=round(ap(te.y, base_te), 4), test_roc=round(auc(te.y, base_te), 4))
for C in (0.1, 1.0):
    m = LogisticRegression(C=C, max_iter=1000).fit(X(va), va.y)
    pv, pt = m.predict_proba(X(va))[:, 1], m.predict_proba(X(te))[:, 1]
    out[f'pixel_C{C}'] = dict(val_pr_auc=round(ap(va.y, pv), 4), test_pr_auc=round(ap(te.y, pt), 4), test_roc=round(auc(te.y, pt), 4), coef=m.coef_.round(3).tolist()[0])
    # false alarms caught while keeping the same share of planets as the base score at 0.5
    keep = (base_te >= 0.5)[te.y.values == 1].mean(); thr = np.quantile(pt[te.y.values == 1], 1 - keep)
    out[f'pixel_C{C}']['same_planets_kept'] = f"{int(((pt >= thr) & (te.y.values == 1)).sum())}/{int((te.y == 1).sum())}"
    out[f'pixel_C{C}']['false_alarms_caught'] = f"{int(((pt < thr) & (te.y.values == 0)).sum())}/{int((te.y == 0).sum())}"
out['base']['planets_kept'] = f"{int(((base_te >= 0.5) & (te.y.values == 1)).sum())}/{int((te.y == 1).sum())}"
out['base']['false_alarms_caught'] = f"{int(((base_te < 0.5) & (te.y.values == 0)).sum())}/{int((te.y == 0).sum())}"
print(json.dumps(out, indent=1)); json.dump(out, open('pix_comb.json', 'w'), indent=1)
