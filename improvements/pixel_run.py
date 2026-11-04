"""Pixel-level (difference image) check on labelled validation + test signals, catalogue ephemerides.
Resumable: one JSON line per star in out/pix_{mission}.jsonl. Downloaded TPFs are deleted after use."""
import sys, os, json, time, warnings, numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/user/tv/src'); sys.path.insert(0, '/home/user/tv/app')
from concurrent.futures import ProcessPoolExecutor
mission = sys.argv[1]
OUT = f'/home/user/tv/pix/pix_{mission}.jsonl'

def jobs():
    import tv_common as c
    if mission == 'tess':
        _, _, ts = c.load_tess(); tl = ts.iloc[np.where(ts.y.notna())[0]].reset_index(drop=True)
        sp = json.load(open('/home/user/tv/e4/tess_star_split.json'))
        parts = [('val', tl.iloc[sp['val']]), ('test', tl.iloc[sp['test']])]
        key, idc = 'tid', 'toi'
    else:
        _, _, tk = c.load_kepler(); sp = json.load(open('/home/user/tv/e2/kepler_star_split.json'))
        parts = [('val', tk.iloc[sp['val']]), ('test', tk.iloc[sp['test']])]
        key, idc = 'kepid', None
    all_ = {}
    for part, df in parts:
        for sid, g in df.groupby(key):
            all_[int(sid)] = dict(star=int(sid), split=part, signals=[dict(id=str(r[idc]) if idc else f"{r['kepid']}_{r['tce_plnt_num']}",
                period=float(r.tce_period), t0=float(r.tce_time0bk), duration=float(r.tce_duration),
                y=float(r.y), lab=str(r.tfopwg_disp if mission == 'tess' else r.av_training_set)) for _, r in g.iterrows()])
    return list(all_.values())

def one(job):
    from transitvet import tv_pixel as X
    t = time.time()
    try:
        tpfs = X.download_tpfs(('TIC ' if mission == 'tess' else 'KIC ') + str(job['star']), mission)
        res = X.check_signals(tpfs, job['signals'], mission, keep_images=False)
        for tp in tpfs:
            try: os.remove(tp.path)
            except Exception: pass
        out = dict(job, results=res, n_files=len(tpfs))
    except Exception as e:
        out = dict(job, error=str(e)[:300])
    out['seconds'] = time.time() - t
    return out

if __name__ == '__main__':
    J = jobs()
    done = set()
    if os.path.exists(OUT):
        done = {json.loads(l)['star'] for l in open(OUT)}
    J = [j for j in J if j['star'] not in done]
    print(len(J), 'stars to do', flush=True)
    with ProcessPoolExecutor(int(os.environ.get('W', 4))) as ex, open(OUT, 'a') as fo:
        for k, r in enumerate(ex.map(one, J, chunksize=1)):
            fo.write(json.dumps(r, default=float) + '\n'); fo.flush()
            if k % 20 == 0:
                print(k, r['star'], r.get('error', ''), round(r['seconds']), flush=True)
    print('ALLDONE', flush=True)
