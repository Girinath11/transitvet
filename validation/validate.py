"""End-to-end check of the TransitVet tool on held-out TEST stars (never used for training or calibration).
For each star: download -> BLS search -> score. A catalogue signal counts as 'recovered' if a found
period matches it within 1% (or at 1/2x or 2x). The verdict of the matching signal is compared with the label."""
import sys, os, json, time, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/user/tv/app'); sys.path.insert(0, '/home/user/tv/src')
from concurrent.futures import ProcessPoolExecutor
import tv_common as c
OUT = '/home/user/tv/app/validation'
def pick():
    _, _, tk = c.load_kepler(); ks = json.load(open('/home/user/tv/e2/kepler_star_split.json')); te = tk.iloc[ks['test']]
    rng = np.random.RandomState(1)
    rows = []
    for lab, n in [('PC', 15), ('AFP', 15)]:
        st = te[te.av_training_set == lab].kepid.unique(); st = rng.choice(st, n, replace=False)
        rows += [dict(target=f'KIC {s}', mission='kepler', label=lab, cat_periods=te[te.kepid == s].tce_period.tolist(),
                      cat_labels=te[te.kepid == s].av_training_set.tolist()) for s in st]
    _, _, ts = c.load_tess(); lab_ = np.where(ts.y.notna())[0]; tl = ts.iloc[lab_].reset_index(drop=True)
    tsp = json.load(open('/home/user/tv/e4/tess_star_split.json')); tt = tl.iloc[tsp['test']]
    for lab, n in [(1.0, 10), (0.0, 10)]:
        st = rng.choice(tt[tt.y == lab].tid.unique(), n, replace=False)
        rows += [dict(target=f'TIC {s}', mission='tess', label='planet' if lab else 'FP', cat_periods=tt[tt.tid == s].tce_period.tolist(),
                      cat_labels=tt[tt.tid == s].tfopwg_disp.tolist()) for s in st]
    return rows
_scorer = None
def one(row):
    global _scorer
    from transitvet import pipeline as P
    import torch; torch.set_num_threads(1)
    if _scorer is None: _scorer = P.Scorer()
    t0 = time.time()
    try:
        segs, meta = P.load_star(row['target'], row['mission'])
        res = P.run(segs, meta, _scorer, progress=lambda m: None)
        found = [dict(period=s['period'], verdict=s['verdict'], p_main=s['scores'][s['main_model']], scores=s['scores']) for s in res['signals']]
        match = []
        for cp, cl in zip(row['cat_periods'], row['cat_labels']):
            m = None
            for k, s in enumerate(found):
                for h in (1, 2, 0.5):
                    if abs(s['period'] * h - cp) / cp < 0.01: m = k
                if m is not None: break
            match.append(dict(cat_period=cp, cat_label=cl, found_index=m, verdict=None if m is None else found[m]['verdict'],
                              p_main=None if m is None else found[m]['p_main']))
        return dict(row, found=found, match=match, seconds=time.time() - t0)
    except Exception as e:
        return dict(row, error=str(e)[:300], seconds=time.time() - t0)
if __name__ == '__main__':
    rows = pick(); print(len(rows), 'stars', flush=True); res = []
    with ProcessPoolExecutor(4) as ex:
        for r in ex.map(one, rows):
            res.append(r); print(r['target'], r['label'], r.get('error', ''), [(m['cat_label'], round(m['cat_period'], 3), m['verdict']) for m in r.get('match', [])], round(r['seconds']), flush=True)
            json.dump(res, open(f'{OUT}/validation_raw.json', 'w'), indent=1, default=float)
    print('ALLDONE', flush=True)
