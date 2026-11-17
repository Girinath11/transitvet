"""Injection-recovery test. Fake planets (batman transit model, known size and orbit) are multiplied into the RAW
light curves of real held-out test stars, then the whole dashboard pipeline runs unchanged (search, checks, models,
verdict; pixel check off). A planet counts as found if a signal has its period (or half/double) within 1% and
t0 within one duration; the verdict of that signal is recorded. Resumable: one JSON line per injection."""
import sys, os, json, gzip, pickle, time, numpy as np, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/user/tv/src'); sys.path.insert(0, '/home/user/tv/repo')
os.environ.setdefault('TV_LAZY_MODELS', '1')
from transitvet import pipeline as P, extras as E, gaia as GA
import batman
C = '/home/user/tv/inj/cache'; R = '/home/user/tv/inj/res'

def star_data(target, mission):
    p = f"{C}/{target.replace(' ', '_')}.pkl.gz"
    if os.path.exists(p):
        return pickle.load(gzip.open(p))
    segs, meta = P.load_star(target, mission)
    star = E.star_props(target, mission)
    g = {}
    if mission == 'tess' and np.isfinite(star.get('ra', np.nan)):
        ra, dec = GA.to_gaia_epoch(star['ra'], star['dec'], star.get('pmra', np.nan), star.get('pmdec', np.nan)); g = GA.star_features(round(ra, 6), round(dec, 6))
    d = dict(segs=segs, meta=meta, star=star, gaia=g)
    with gzip.open(p + '.tmp', 'wb') as fh: pickle.dump(d, fh)
    os.replace(p + '.tmp', p); return d

def inject(segs, per, t0, rp_rs, a_rs, inc):
    pm = batman.TransitParams(); pm.t0, pm.per, pm.rp, pm.a, pm.inc, pm.ecc, pm.w = t0, per, rp_rs, a_rs, inc, 0., 90.
    pm.u, pm.limb_dark = [0.4, 0.25], 'quadratic'
    out = []
    for t, f in segs:
        m = batman.TransitModel(pm, t, supersample_factor=5, exp_time=np.median(np.diff(t)) if t.size > 1 else 0.02).light_curve(pm)
        out.append((t, f * m))
    return out

def draw(rng, span, rstar):
    per = float(np.exp(rng.uniform(np.log(0.7), np.log(max(1.0, min(span / 3, 200))))))
    rp = float(np.exp(rng.uniform(np.log(0.7), np.log(8.0))))         # Earth radii
    rs = rstar if np.isfinite(rstar) and 0.1 < rstar < 5 else 1.0
    ms = rs ** 1.0 if rs < 1.5 else 1.3                                   # rough main-sequence mass (Sun = 1)
    a_au = (ms * (per / 365.25) ** 2) ** (1 / 3); a_rs = a_au * 215.03 / rs
    b = float(rng.uniform(0, 0.8)); inc = float(np.degrees(np.arccos(b / a_rs)))
    return per, rp, rp / 109.1 / rs, a_rs, inc, b

SNR_MODE = os.environ.get('TV_INJ_SNR') == '1'

def draw_snr(rng, span, rstar, sd, npts):
    """Pick the period at random, then the size that gives an expected SNR drawn log-uniform in 3..80, so the
    recovery curve is sampled where it changes (sizes capped at 0.5 to 20 Earth radii)."""
    per = float(np.exp(rng.uniform(np.log(0.7), np.log(max(1.0, min(span / 3, 200))))))
    snr = float(np.exp(rng.uniform(np.log(3), np.log(80))))
    rs = rstar if np.isfinite(rstar) and 0.1 < rstar < 5 else 1.0
    ms = rs ** 1.0 if rs < 1.5 else 1.3
    a_rs = (ms * (per / 365.25) ** 2) ** (1 / 3) * 215.03 / rs
    b = float(rng.uniform(0, 0.8)); k = 0.02
    for _ in range(3):
        dur = per / np.pi * np.arcsin(min(1, np.sqrt(max((1 + k) ** 2 - b ** 2, 0)) / a_rs))
        n_in = max(npts * dur / per, 1)
        k = float(np.sqrt(snr * sd / np.sqrt(n_in)))
    rp = float(np.clip(k * 109.1 * rs, 0.5, 20)); k = rp / 109.1 / rs
    inc = float(np.degrees(np.arccos(b / a_rs)))
    return per, rp, k, a_rs, inc, b

def main(mission, stars, n_per_star, seed):
    sc = P.Scorer(); rng = np.random.default_rng(seed)
    out = f'{R}/{mission}' + ('_snr' if SNR_MODE else '') + os.environ.get('TV_INJ_TAG', '') + '.jsonl'
    done = set()
    if os.path.exists(out):
        done = {(json.loads(l)['star'], json.loads(l)['k']) for l in open(out)}
    for s in stars:
        tg = ('KIC ' if mission == 'kepler' else 'TIC ') + str(s)
        try:
            d = star_data(tg, mission)
        except Exception as e:
            print('skip', tg, e, flush=True); continue
        segs = d['segs']; tt = np.concatenate([x[0] for x in segs]); span = float(tt.max() - tt.min())
        if SNR_MODE:
            ft0, ff0 = P.flatten_for_search(segs); sd0 = 1.4826 * np.median(np.abs(ff0 - np.median(ff0)))
            draws = [draw_snr(rng, span, d['star'].get('star_rad', np.nan), sd0, ft0.size) for _ in range(n_per_star)]
        else:
            draws = [draw(rng, span, d['star'].get('star_rad', np.nan)) for _ in range(n_per_star)]
        for k, (per, rp, rp_rs, a_rs, inc, b) in enumerate(draws):
            if (tg, k) in done: continue
            t0 = float(tt.min() + rng.uniform(0, per))
            meta = dict(d['meta'], _pixel=False, _star=d['star'], _gaia=d['gaia'])
            t1 = time.time()
            try:
                res = P.run(inject(segs, per, t0, rp_rs, a_rs, inc), meta, sc, lambda m: None)
            except Exception as e:
                print('fail', tg, k, e, flush=True); continue
            dur = per / np.pi * np.arcsin(min(1, np.sqrt((1 + rp_rs) ** 2 - b ** 2) / a_rs / np.sin(np.radians(inc))))
            hit = None
            for r in res['signals']:
                for mult in (1, 0.5, 2):
                    if abs(r['period'] - per * mult) / (per * mult) < 0.01:
                        ph = ((r['t0'] - t0) / per + 0.5) % 1 - 0.5 if mult == 1 else 0
                        if abs(ph * per) < max(dur, r['duration']) or mult != 1:
                            hit = dict(verdict=r['verdict'], score=r['scores'][r['main_model']], mult=mult, triage=(r.get('triage') or {}).get('call'),
                                       found_depth=r['depth_ppm'], snr=r.get('bls_snr')); break
                if hit: break
            # expected signal-to-noise: depth / (point scatter) * sqrt(points in transit)
            ft, ff = P.flatten_for_search(segs); sd = 1.4826 * np.median(np.abs(ff - np.median(ff)))
            ph = ((ft - t0) / per + 0.5) % 1 - 0.5; n_in = int((np.abs(ph * per) < dur / 2).sum())
            depth = rp_rs ** 2
            row = dict(star=tg, k=k, period=per, rp_re=rp, rp_rs=rp_rs, b=b, dur_h=dur * 24, depth_ppm=depth * 1e6, n_in=n_in, sd_ppm=sd * 1e6,
                       exp_snr=depth / sd * np.sqrt(max(n_in, 0)), n_tr=int(len(np.unique(np.round((ft[np.abs(ph * per) < dur / 2] - t0) / per)))),
                       found=bool(hit), **({} if not hit else hit), seconds=time.time() - t1, n_signals=len(res['signals']))
            with open(out, 'a') as fh: fh.write(json.dumps(row) + '\n')
            print(tg, k, f"P={per:.2f} Rp={rp:.2f} SNRexp={row['exp_snr']:.1f}", 'FOUND ' + hit['verdict'] if hit else 'missed', f"{row['seconds']:.0f}s", flush=True)

if __name__ == '__main__':
    mission, n = sys.argv[1], int(sys.argv[2])
    stars = [int(x) for x in sys.argv[3].split(',')]
    main(mission, stars, n, int(sys.argv[4]) if len(sys.argv) > 4 else 0)
