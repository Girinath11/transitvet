"""Build the final models used by the TransitVet tool, from the experiment code and splits.
Kepler: GB, AstroNet CNN (5 seeds, from E2), Transformer (5 seeds, from E2).
TESS:   GB trained on TESS, CNN and Transformer fine-tuned from the Kepler models (5 seeds each, as E4).
Each (mission, model) gets an isotonic calibrator fitted on that mission's VALIDATION split only."""
import sys, json, joblib, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, '/home/user/tv/src'); import tv_common as c, tv_models as M
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score
torch.set_num_threads(4); OUT = '/home/user/tv/app/transitvet/models'
Gk, Lk, tk = c.load_kepler(); ks = json.load(open('/home/user/tv/e2/kepler_star_split.json'))
ktr, kva = np.array(ks['train']), np.array(ks['val']); yk = tk.y.values
Gs, Ls, ts = c.load_tess(); lab = np.where(ts.y.notna())[0]; ts = ts.iloc[lab].reset_index(drop=True); Gs, Ls = Gs[lab], Ls[lab]; ys = ts.y.values.astype(np.float32)
tsp = json.load(open('/home/user/tv/e4/tess_star_split.json')); str_, sva = np.array(tsp['train']), np.array(tsp['val'])
def pred(m, G, L, ix):
    m.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(ix), 512): j = ix[i:i + 512]; out.append(torch.sigmoid(m(torch.as_tensor(G[j]), torch.as_tensor(L[j]))))
    return torch.cat(out).numpy()
calib, manifest = {}, {}
def save_cal(name, yv, pv):
    calib[name] = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1).fit(pv, yv)
    manifest[name] = dict(val_pr_auc=float(average_precision_score(yv, pv)), n_val=int(len(yv)))
# Kepler GB
Xk = c.feature_matrix(tk); gbs = []
for s in range(5):
    gbs.append(HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, early_stopping=True, validation_fraction=0.1, random_state=s).fit(Xk[ktr], yk[ktr]))
joblib.dump(gbs, f'{OUT}/kepler_gb.joblib'); save_cal('kepler_gb', yk[kva], np.mean([g.predict_proba(Xk[kva])[:, 1] for g in gbs], 0))
# Kepler CNN / TF (copy E2 weights)
for kind, cls, ck in [('cnn', M.AstroNet, 'e2_cnn'), ('tf', M.PatchTransformer, 'e2_tf_pre')]:
    sds = [torch.load(f'/home/user/tv/e2/{ck}_seed{s}.pt') for s in range(5)]; torch.save(sds, f'{OUT}/kepler_{kind}.pt')
    pv = []
    for sd in sds: m = cls(); m.load_state_dict(sd); pv.append(pred(m, Gk, Lk, kva))
    save_cal(f'kepler_{kind}', yk[kva], np.mean(pv, 0))
print('kepler done', manifest, flush=True)
# TESS GB
Xs = c.feature_matrix(ts); gbs = []
for s in range(5):
    gbs.append(HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, early_stopping=True, validation_fraction=0.1, random_state=s).fit(Xs[str_], ys[str_]))
joblib.dump(gbs, f'{OUT}/tess_gb.joblib'); save_cal('tess_gb', ys[sva], np.mean([g.predict_proba(Xs[sva])[:, 1] for g in gbs], 0))
# TESS fine-tuned CNN / TF (same recipe as E4)
GT, LT, yT = torch.tensor(Gs), torch.tensor(Ls), torch.tensor(ys)
for kind, cls, lr in [('cnn', M.AstroNet, 5e-5), ('tf', M.PatchTransformer, 3e-5)]:
    sds, pv = [], []
    for s, sd in enumerate(torch.load(f'{OUT}/kepler_{kind}.pt')):
        torch.manual_seed(s); np.random.seed(s); m = cls(); m.load_state_dict(sd)
        opt = torch.optim.Adam(m.parameters(), lr=lr); best, state = -1, None
        for step in range(1, 401):
            m.train(); j = torch.tensor(np.random.choice(str_, 64)); loss = F.binary_cross_entropy_with_logits(m(GT[j], LT[j]), yT[j])
            opt.zero_grad(); loss.backward(); opt.step()
            if step % 25 == 0:
                a = average_precision_score(ys[sva], pred(m, Gs, Ls, sva))
                if a > best: best, state = a, {k: v.clone() for k, v in m.state_dict().items()}
        m.load_state_dict(state); sds.append(state); pv.append(pred(m, Gs, Ls, sva)); print('tess', kind, s, round(best, 3), flush=True)
    torch.save(sds, f'{OUT}/tess_{kind}.pt'); save_cal(f'tess_{kind}', ys[sva], np.mean(pv, 0))
joblib.dump(calib, f'{OUT}/calibrators.joblib')
json.dump(dict(built='2026-11-03', models=manifest, feats=c.FEATS,
               note='Kepler models trained on Kepler DR24 training stars (70%); TESS models on TESS training stars (50%). Calibrators: isotonic on validation stars.'),
          open(f'{OUT}/manifest.json', 'w'), indent=1)
print('ALLDONE', json.dumps(manifest), flush=True)
