"""Collect centroid shifts per signal for all labelled Kepler DR24 and TESS TOI signals.
Usage: python run_cen.py kepler|tess NPROC [LIMIT]"""
import os, sys, gzip, pickle, traceback, time, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/user/tv/src')
import numpy as np, pandas as pd, requests
from concurrent.futures import ProcessPoolExecutor, as_completed
MIS = sys.argv[1]; NP = int(sys.argv[2]); LIM = int(sys.argv[3]) if len(sys.argv) > 3 else None
OUT = f'/home/user/tv/out/cen/{MIS}'; os.makedirs(OUT, exist_ok=True)
if MIS == 'kepler':
    df = pd.read_csv('/home/user/tv/out/labels/q1_q17_dr24_tce.csv')
    df = df[df.av_training_set.isin(['PC', 'AFP', 'NTP'])].copy(); df['tce_duration'] /= 24.0
    df['sid'] = df.kepid; df['key'] = df.tce_plnt_num
else:
    urls = json.load(open('/home/user/tv/out/tess/tic_urls.json'))
    df = pd.read_csv('/home/user/tv/out/labels/toi.csv')
    df = df[df.tid.astype(str).isin(urls) & df.pl_orbper.notna() & df.pl_tranmid.notna() & df.pl_trandurh.notna()
            & (df.pl_orbper > 0) & df.tfopwg_disp.isin(['CP', 'KP', 'FP', 'FA'])].copy()
    df['tce_period'] = df.pl_orbper; df['tce_time0bk'] = df.pl_tranmid - 2457000.0; df['tce_duration'] = df.pl_trandurh / 24.0
    df['sid'] = df.tid; df['key'] = df.toi
by_star = {int(k): g[['key', 'tce_period', 'tce_time0bk', 'tce_duration']].to_dict('records') for k, g in df.groupby('sid')}
stars = sorted(by_star)
if LIM: stars = list(np.random.default_rng(0).choice(stars, LIM, replace=False))
_s = None
def run(sid):
    global _s
    import tv_centroid as C
    if _s is None: _s = requests.Session()
    path = f'{OUT}/{sid}.pkl.gz'
    if os.path.exists(path): return 'skip'
    try:
        segs = C.download_kepler_cen(sid, _s) if MIS == 'kepler' else C.download_tess_cen(urls[str(sid)], _s)
        rows = by_star[sid]; res = {}
        for r in rows:
            others = [(o['tce_period'], o['tce_time0bk'], o['tce_duration']) for o in rows if o is not r]
            res[r['key']] = C.segment_shifts(segs, r['tce_period'], r['tce_time0bk'], r['tce_duration'], others)
        res = dict(n_segs=len(segs), shifts=res)
    except Exception:
        res = 'error: ' + traceback.format_exc(limit=2)
    with gzip.open(path + '.tmp', 'wb') as fh: pickle.dump(res, fh)
    os.replace(path + '.tmp', path)
    return 'ok' if isinstance(res, dict) else res[:300]
if __name__ == '__main__':
    print(MIS, len(df), 'signals on', len(stars), 'stars', flush=True); t0 = time.time(); done = 0
    with ProcessPoolExecutor(NP) as pool:
        futs = {pool.submit(run, int(k)): k for k in stars}
        for fut in as_completed(futs):
            done += 1; res = fut.result()
            if res not in ('ok', 'skip'): print(futs[fut], res, flush=True)
            if done % 50 == 0 or done == len(stars): print(f'{done}/{len(stars)} {time.time()-t0:.0f}s', flush=True)
    print('finished', flush=True)
