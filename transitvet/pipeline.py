"""TransitVet pipeline: light curve -> dip search -> model inputs -> calibrated scores -> verdict.

Steps (each is a standard signal-processing step):
 1. load      Kepler / TESS light curve from MAST (by star ID) or from an uploaded file
 2. clean     keep good-quality, finite points; normalise each segment; TESS binned to 30 min
 3. detrend   biweight filter (for the search) and the AstroNet spline (for the model views)
 4. search    Box Least Squares (BLS): find the strongest repeating dip, mask it, repeat
 5. fold      global view (whole orbit, 2001 bins) and local view (dip, 201 bins) + check numbers
 6. classify  gradient boosting, AstroNet CNN and Transformer, each calibrated on validation stars
 7. verdict   candidate / false alarm / uncertain
Outputs are planet CANDIDATES, never confirmed planets.
"""
import io, json, os, time, warnings
import numpy as np

from . import tv_kepler as tv

MISSIONS = ('kepler', 'tess')
SEARCH_DURATIONS = np.array([0.04, 0.06, 0.08, 0.12, 0.17, 0.25, 0.35])   # days
MIN_PERIOD, MAX_SIGNALS, MIN_SNR = 0.6, 3, 7.1
# light-position warning: (min sigma, min source offset in arcsec). Chosen on TRAINING signals so that the warning
# hits <1% of Kepler planet candidates (0.8%) and 9% of Kepler false alarms. On TESS no setting separated planets
# from false alarms (big 21-arcsec pixels), so no warning there; the models still use the numbers.
LIGHT_OFFSET_WARN = {'kepler': (10.0, 30.0), 'tess': None}
# pixel-level (difference image) warning: (min sigma, min offset of the dip's source from the target, arcsec).
# Chosen on VALIDATION signals; see validation/PIXEL_CHECK.md.
PIXEL_WARN = {'kepler': None, 'tess': (5.0, 10.0)}   # TESS: 2.0% of validation planets, 24% of validation false alarms;
# Kepler: not validated yet, so the numbers are shown but no warning is raised


# ------------------------------------------------------------------ 1-2. load and clean

def _bin(t, f, minutes=30.0):
    w = minutes / 1440.0
    k = np.floor((t - t[0]) / w).astype(int)
    n = np.bincount(k); good = n > 0
    return np.bincount(k, t)[good] / n[good], np.bincount(k, f)[good] / n[good]


def _clean_segment(t, f, q=None, mission='kepler'):
    ok = np.isfinite(t) & np.isfinite(f)
    if q is not None:
        ok &= (q == 0) if mission == 'tess' else np.ones_like(ok)
    t, f = np.asarray(t, float)[ok], np.asarray(f, float)[ok]
    if t.size < 50:
        return None
    o = np.argsort(t); t, f = t[o], f[o]
    if mission == 'tess' and np.median(np.diff(t)) < 0.015:     # 2-min data -> 30-min, as in training
        t, f = _bin(t, f)
    return t, f


def _cen_segment(t, f, cx, cy, q, mission):
    """(t, f, cx, cy) for the light-position check: good-quality points, TESS binned to 30 min (as in training)."""
    t, f, cx, cy = (np.asarray(a, float) for a in (t, f, cx, cy))
    ok = np.isfinite(t) & np.isfinite(f) & np.isfinite(cx) & np.isfinite(cy)
    if q is not None:
        ok &= (np.asarray(q) == 0)
    t, f, cx, cy = t[ok], f[ok], cx[ok], cy[ok]
    if t.size < 50:
        return None
    o = np.argsort(t); t, f, cx, cy = t[o], f[o], cx[o], cy[o]
    if mission == 'tess' and np.median(np.diff(t)) < 0.015:
        k = np.floor((t - t[0]) / (30.0 / 1440.0)).astype(int); n = np.bincount(k); g = n > 0
        t, f, cx, cy = [np.bincount(k, a)[g] / n[g] for a in (t, f, cx, cy)]
    return t, f, cx, cy


def load_star(target, mission=None):
    """target like 'KIC 11442793', 'TIC 261136679', 'Kepler-90', or a plain number (mission then required)."""
    import lightkurve as lk
    target = str(target).strip()
    if target.isdigit():
        if mission is None:
            raise ValueError('Give the mission for a plain number, or write KIC/TIC before it.')
        target = ('KIC ' if mission == 'kepler' else 'TIC ') + target
    if mission is None:
        mission = 'tess' if target.upper().startswith(('TIC', 'TOI')) else 'kepler'
    if mission == 'kepler':
        sr = lk.search_lightcurve(target, author='Kepler', exptime=1800)
    else:
        sr = lk.search_lightcurve(target, author='SPOC', exptime=120)
    if len(sr) == 0:
        raise ValueError(f'No {mission} light curves found on MAST for {target}.')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        lcs = sr.download_all(quality_bitmask='default')
    segs, cen = [], []
    for lc in lcs:
        flux = lc.pdcsap_flux if 'pdcsap_flux' in lc.colnames else lc.flux
        q = np.asarray(lc.quality.value if hasattr(lc.quality, 'value') else lc.quality) if 'quality' in lc.colnames else None
        s = _clean_segment(lc.time.value, np.asarray(flux.value, float), q, mission)
        if s is not None:
            segs.append(s)
        if 'centroid_col' in lc.colnames and 'centroid_row' in lc.colnames:
            c = _cen_segment(lc.time.value, np.asarray(flux.value, float), np.asarray(lc.centroid_col.value, float),
                             np.asarray(lc.centroid_row.value, float), q, mission)
            if c is not None:
                cen.append(c)
    return segs, dict(target=target, mission=mission, n_files=len(lcs), source='MAST via Lightkurve', _cen=cen)


def load_file(name, data, mission=None):
    """Uploaded file: a Kepler/TESS light-curve FITS file, or a CSV/TXT with columns time, flux."""
    name = name.lower()
    if name.endswith(('.fits', '.fit', '.fits.gz')):
        from astropy.io import fits
        with fits.open(io.BytesIO(data)) as h:
            hdr = h[0].header; d = h[1].data; cols = d.columns.names
            tel = str(hdr.get('TELESCOP', '')).lower()
            m = mission or ('tess' if 'tess' in tel else 'kepler')
            fcol = 'PDCSAP_FLUX' if 'PDCSAP_FLUX' in cols else ('FLUX' if 'FLUX' in cols else 'SAP_FLUX')
            q = np.asarray(d['QUALITY']) if 'QUALITY' in cols else None
            seg = _clean_segment(np.asarray(d['TIME'], float), np.asarray(d[fcol], float), q, m)
            cen = None
            if 'MOM_CENTR1' in cols and 'MOM_CENTR2' in cols:
                cen = _cen_segment(d['TIME'], d[fcol], d['MOM_CENTR1'], d['MOM_CENTR2'], q, m)
            obj = hdr.get('OBJECT', name)
        return ([seg] if seg else []), dict(target=str(obj), mission=m, n_files=1, source=f'uploaded {name}', _cen=[cen] if cen else [])
    import pandas as pd
    df = pd.read_csv(io.BytesIO(data), sep=None, engine='python', comment='#')
    cols = {c.lower().strip(): c for c in df.columns}
    tcol = next((cols[c] for c in ('time', 'bjd', 'btjd', 'bkjd', 't') if c in cols), df.columns[0])
    fcol = next((cols[c] for c in ('pdcsap_flux', 'flux', 'f', 'sap_flux') if c in cols), df.columns[1])
    m = mission or 'kepler'
    t, f = df[tcol].values.astype(float), df[fcol].values.astype(float)
    # split into segments at gaps > 2 days, so each is normalised on its own
    o = np.argsort(t); t, f = t[o], f[o]
    cuts = np.where(np.diff(t) > 2.0)[0] + 1
    segs = [s for s in (_clean_segment(a, b, None, m) for a, b in zip(np.split(t, cuts), np.split(f, cuts))) if s]
    return segs, dict(target=os.path.splitext(name)[0], mission=m, n_files=1, source=f'uploaded {name}')


# ------------------------------------------------------------------ 3-4. detrend and search

def flatten_for_search(segs):
    from wotan import flatten
    T, F = [], []
    for t, f in segs:
        f = f / np.nanmedian(f)
        fl = flatten(t, f, method='biweight', window_length=1.0, break_tolerance=0.5, edge_cutoff=0)
        ok = np.isfinite(fl)
        T.append(t[ok]); F.append(fl[ok])
    t, f = np.concatenate(T), np.concatenate(F)
    o = np.argsort(t); t, f = t[o], f[o]
    sd = 1.4826 * np.median(np.abs(f - np.median(f)))
    up = f > np.median(f) + 5 * sd            # remove upward spikes only (dips are kept)
    return t[~up], f[~up]


def _bls_chunk(args):
    from astropy.timeseries import BoxLeastSquares
    t, f, sd, periods = args
    r = BoxLeastSquares(t, f, dy=np.full(t.size, sd)).power(periods, SEARCH_DURATIONS, objective='snr')
    return (np.asarray(r.period), np.asarray(r.power), np.asarray(r.depth), np.asarray(r.depth_snr),
            np.asarray(r.duration), np.asarray(r.transit_time))


class _R:   # minimal stand-in for astropy's BLS result, merged from chunks
    def __init__(self, parts):
        (self.period, self.power, self.depth, self.depth_snr, self.duration, self.transit_time) = [np.concatenate(x) for x in zip(*parts)]


def _power(t, f, sd, periods, workers=None, on_chunk=None):
    """BLS power, split over CPU cores by period chunks (same result as one call).
    on_chunk(parts_so_far) is called after each chunk, for the live view (results are identical)."""
    workers = workers or int(os.environ.get('TV_BLS_WORKERS', max(1, min(8, os.cpu_count() or 1))))
    if on_chunk is None and (workers == 1 or periods.size < 20000):
        return _R([_bls_chunk((t, f, sd, periods))])
    chunks = np.array_split(periods, max(8, workers * 2))
    parts = []
    if workers == 1:
        for c in chunks:
            parts.append(_bls_chunk((t, f, sd, c)))
            if on_chunk: on_chunk(parts)
        return _R(parts)
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(workers) as ex:
        for part in ex.map(_bls_chunk, [(t, f, sd, c) for c in chunks]):
            parts.append(part)
            if on_chunk: on_chunk(parts)
    return _R(parts)


def _second_dip(tt, ff, sd, g, sig):
    """The new signal repeats an earlier one (same period, half or double). Its dips fall at some phases of
    the earlier signal's orbit; if one of them is clearly apart from the first dip and is a real, significant
    dip in the remaining data, return (phase, depth_ppm) of that second dip, else None.
    (At half the period, the dips of the new signal alternate between phase 0 and the second dip.)"""
    P0, D0 = g['period'], g['duration']
    n = max(1, int(round(P0 / sig['period'])))
    best = None
    for k in range(n):
        ph = ((sig['t0'] + k * sig['period'] - g['t0']) / P0) % 1.0
        phd = min(ph, 1 - ph)            # 0 = same place as the first dip, 0.5 = half an orbit away
        if phd * P0 <= 1.5 * D0:
            continue
        dt = (tt - (g['t0'] + ph * P0) + 0.5 * P0) % P0 - 0.5 * P0
        inn = np.abs(dt) < 0.5 * D0
        near = (np.abs(dt) > D0) & (np.abs(dt) < 3 * D0)
        if inn.sum() < 3 or near.sum() < 3:
            continue
        depth = np.median(ff[near]) - np.mean(ff[inn])
        snr = depth / (sd / np.sqrt(inn.sum()))
        if snr >= MIN_SNR and (best is None or depth > best[1]):
            best = (float(phd), float(depth * 1e6))
    return best


def bls_search(t, f, max_signals=MAX_SIGNALS, min_snr=MIN_SNR, progress=None, on_chunk=None):
    """Box Least Squares, repeated: find the best dip, record it, remove its points, search again."""
    found, keep, periodograms = [], np.ones(t.size, bool), []
    span = t.max() - t.min()
    max_p = max(MIN_PERIOD * 2, min(span / 3.0, 300.0))   # need >= 3 dips
    for k in range(max_signals):
        tt, ff = t[keep], f[keep]
        if tt.size < 100:
            break
        sd = 1.4826 * np.median(np.abs(ff - np.median(ff)))
        # log-spaced periods; spacing chosen so a dip drifts by <= ~0.03 d over the whole span (capped for speed)
        n = int(np.clip(span * np.log(max_p / MIN_PERIOD) / 0.03, 20000, 150000))
        periods = np.exp(np.linspace(np.log(MIN_PERIOD), np.log(max_p), n))
        r = _power(tt, ff, sd, periods, on_chunk=(lambda parts, k=k: on_chunk(k, parts, periods.size)) if on_chunk else None)
        snr = np.asarray(r.depth_snr)
        coarse = (np.asarray(r.period), np.asarray(r.power))
        # take the strongest peak that has at least 3 separate dips with data (rejects one-off glitches)
        order = np.argsort(np.nan_to_num(np.asarray(r.power), nan=-1))[::-1]
        i = int(order[0])
        for cand in order[:200]:
            P_, T0_, D_ = r.period[cand], r.transit_time[cand], r.duration[cand]
            ph = np.abs((tt - T0_ + 0.5 * P_) % P_ - 0.5 * P_) < 0.5 * D_
            if np.unique(np.round((tt[ph] - T0_) / P_)).size >= 3 and r.depth[cand] > 0:
                i = int(cand); break
        # refine the period around the chosen peak with a much finer grid
        step = np.log(periods[1] / periods[0])
        fine = r.period[i] * np.exp(np.linspace(-3 * step, 3 * step, 600))
        rf = _power(tt, ff, sd, fine, workers=1); j = int(np.nanargmax(rf.power))
        if rf.power[j] >= r.power[i] and rf.depth[j] > 0:
            r, i, snr = rf, j, np.asarray(rf.depth_snr)
        sig = dict(period=float(r.period[i]), t0=float(r.transit_time[i]), duration=float(r.duration[i]),
                   depth_ppm=float(r.depth[i] * 1e6), bls_snr=float(snr[i]))
        st_ = max(1, len(coarse[0]) // 4000)
        periodograms.append(dict(period=coarse[0][::st_], power=coarse[1][::st_]))
        if progress:
            progress(k, sig)
        if not np.isfinite(sig['bls_snr']) or sig['bls_snr'] < min_snr or sig['depth_ppm'] <= 0:
            break
        # the same period again means the first signal was not fully removed (common for eclipsing
        # binaries and pulsating stars): report it once and stop searching this star
        dup = [g for g in found if any(abs(sig['period'] * h - g['period']) / g['period'] < 0.02 for h in (1, 2, 0.5))]
        if dup:
            g = dup[0]
            sec = _second_dip(tt, ff, sd, g, sig)
            if sec:
                # a second, separate dip at the same period: typical of an eclipsing binary (secondary eclipse)
                g['second_dip_phase'], g['second_dip_ppm'] = sec
            break   # either way, stop searching this star
        found.append(sig)
        mask = tv.in_transit_mask(t, [(sig['period'], sig['t0'], sig['duration'])], width_factor=3.0)
        keep &= ~mask
    return found, periodograms


# ------------------------------------------------------------------ 5-7. views, scores, verdict

class Scorer:
    def __init__(self, model_dir=os.path.join(os.path.dirname(__file__), 'models')):
        import joblib, torch
        from . import tv_models as M
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
        self.M, self.torch = M, torch
        from .fetch_models import ensure
        ensure(model_dir)                 # the two big CNN files come from the GitHub release on first use
        self.cal = joblib.load(os.path.join(model_dir, 'calibrators.joblib'))
        self.manifest = json.load(open(os.path.join(model_dir, 'manifest.json')))
        self.feats = self.manifest['feats']
        self.gb, self.nets = {}, {}
        cf = os.path.join(model_dir, 'conformal.joblib')
        self.conf = joblib.load(cf) if os.path.exists(cf) else {}
        cp = os.path.join(model_dir, 'combiner.joblib')
        self.comb = joblib.load(cp) if os.path.exists(cp) else {}
        for m in MISSIONS:
            self.gb[m] = joblib.load(os.path.join(model_dir, f'{m}_gb.joblib'))
            for kind, cls in (('cnn', M.AstroNet), ('tf', M.PatchTransformer)):
                nets = []
                for sd in torch.load(os.path.join(model_dir, f'{m}_{kind}.pt')):
                    net = cls(); net.load_state_dict(sd); net.eval(); nets.append(net)
                self.nets[(m, kind)] = nets

    def feature_row(self, rec):
        x = np.array([float(rec.get(k, np.nan)) if rec.get(k) is not None else np.nan for k in self.feats], float)
        for j in [0, 1, 3, 4, 6, 7, 8]:     # same signed log as training (tv_common.feature_matrix)
            x[j] = np.sign(x[j]) * np.log1p(np.abs(x[j]))
        return x

    def score(self, recs, mission):
        torch = self.torch
        X = np.stack([self.feature_row(r) for r in recs])
        g = torch.tensor(np.stack([r['global_astronet'] for r in recs]))
        l = torch.tensor(np.stack([r['local_astronet'] for r in recs]))
        raw = {'gb': np.mean([m.predict_proba(X)[:, 1] for m in self.gb[mission]], 0)}
        with torch.no_grad():
            for kind in ('cnn', 'tf'):
                raw[kind] = np.mean([torch.sigmoid(n(g, l)).numpy() for n in self.nets[(mission, kind)]], 0)
        out = {k: self.cal[f'{mission}_{k}'].predict(v) for k, v in raw.items()}
        if mission in self.comb:      # combined score: logistic regression on the three models' log-odds
            z = np.column_stack([np.log(np.clip(raw[k], 1e-4, 1 - 1e-4) / (1 - np.clip(raw[k], 1e-4, 1 - 1e-4))) for k in ('gb', 'cnn', 'tf')])
            out['combined'] = self.comb[mission].predict_proba(z)[:, 1]
            if mission in self.conf:   # separate combiner used only for the conformal guarantee
                out['_conformal_p'] = self.conf[mission]['combiner'].predict_proba(z)[:, 1]
        return out


# best model per mission in our own tests (E2 for Kepler, E4 for TESS)
MAIN_MODEL_V1 = {'kepler': 'cnn', 'tess': 'gb'}
MAIN_MODEL = {'kepler': 'combined', 'tess': 'combined'}    # v2 (3 Nov 2026), if the combiner file is present
MODEL_NAMES = {'gb': 'Gradient boosting', 'cnn': 'AstroNet CNN', 'tf': 'Transformer', 'combined': 'Combined score'}


CONFORMAL_ALPHA = float(os.environ.get('TV_CONFORMAL_ALPHA', 0.05))


def triage(p, conf, alpha=0.05):
    """Conformal triage (label-conditional split conformal, calibrated on half of the validation stars).
    Guarantee (for stars like the calibration stars): at least (1 - alpha) of real planets keep 'planet' in their
    set, so at most alpha of real planets are called 'sure false alarm'; same for false alarms.
    Returns 'keep' (false alarm ruled out), 'drop' (planet ruled out) or 'needs a human'."""
    a = conf['alphas'].get(alpha) or conf['alphas'][min(conf['alphas'], key=lambda k: abs(k - alpha))]
    has_planet, has_false = (1 - p) <= a['q1'], p <= a['q0']
    if has_planet and not has_false:
        return dict(call='keep', text='keep: a false alarm is ruled out', alpha=alpha, p=p)
    if has_false and not has_planet:
        return dict(call='drop', text='drop: a planet is ruled out', alpha=alpha, p=p)
    return dict(call='needs a human', alpha=alpha, p=p, text='needs a human: ' + ('both answers are still possible' if has_planet else 'neither answer fits well'))


def verdict(p_main, others):
    """candidate if the main model's calibrated probability >= 0.5, false alarm if < 0.5;
    'uncertain' if it is within 0.35-0.65 or the other two models both disagree with it."""
    side = p_main >= 0.5
    disagree = all((p >= 0.5) != side for p in others)
    if 0.35 <= p_main <= 0.65 or disagree:
        return 'uncertain'
    return 'candidate' if side else 'false alarm'


def pixel_warning(rec):
    lim = PIXEL_WARN.get(rec.get('mission_', 'kepler'))
    ps, po = rec.get('px_sigma', np.nan), rec.get('px_offset_arcsec', np.nan)
    return bool(lim and ps is not None and np.isfinite(ps) and ps > lim[0] and np.isfinite(po) and po > lim[1])


def flags(rec):
    out = []
    if np.isfinite(rec.get('odd_even_sigma', np.nan)) and rec['odd_even_sigma'] > 3:
        out.append(f"odd and even dips differ ({rec['odd_even_sigma']:.1f} sigma): possible eclipsing binary, or the period is half the true one, or the dip times wander")
    if (np.isfinite(rec.get('secondary_sigma', np.nan)) and rec['secondary_sigma'] > 5
            and rec.get('secondary_ppm', 0) > 0.1 * max(rec.get('depth_ppm', 0), 1)):
        out.append(f"dip at half phase ({rec['secondary_sigma']:.1f} sigma): possible eclipsing binary")
    if np.isfinite(rec.get('depth_ppm', np.nan)) and rec['depth_ppm'] > 30000:
        out.append('very deep dip (> 3%): too deep for most planets')
    if rec.get('n_transits', 99) < 3:
        out.append('fewer than 3 dips seen: period is less certain')
    rp = rec.get('rp_est_re', np.nan)
    if np.isfinite(rp) and rp > 2 * 11.2:
        out.append(f"estimated size {rp:.0f} Earth radii (> 2 Jupiters): too big for a planet, likely a star")
    cs, co = rec.get('cen_sigma', np.nan), rec.get('cen_offset_arcsec', np.nan)
    lw = LIGHT_OFFSET_WARN[rec.get('mission_', 'kepler')]
    if lw and np.isfinite(cs) and cs > lw[0] and np.isfinite(co) and co > lw[1]:
        out.append(f"the star's light moves during the dip ({cs:.1f} sigma; source about {co:.0f} arcsec away): "
                   "the dip may come from a nearby star")
    if pixel_warning(rec):
        src = rec.get('px_source') or {}
        out.append(f"camera images: the dip comes from about {rec['px_offset_arcsec']:.0f} arcsec away from the target "
                   f"({rec['px_sigma']:.1f} sigma): likely a nearby star"
                   + (f" (closest catalogue star: TIC {src['tic']}, T mag {src['tmag']:.1f})" if src.get('tic') else ''))
    ft = rec.get('fit') or {}
    if ft.get('fit_ok') and ft.get('grazing'):
        out.append(f"V-shaped / grazing dip (impact {ft['b']:.2f}, size ratio {ft['rp_rs']:.2f}): common for eclipsing binaries")
    if ft.get('fit_ok') and 'rho_star_cat' in ft and ft['rho_star_cat'] > 0:
        ratio = ft['rho_star_fit'] / ft['rho_star_cat']
        if ratio > 5 or ratio < 0.2:
            out.append(f"the dip's shape implies a star {ratio:.1f}x as dense as the catalogue says: "
                       "maybe not a planet on this star, an eccentric orbit, or a wrong period")
    ex = rec.get('exominer') or {}
    pm = rec.get('scores', {}).get(rec.get('main_model', ''), np.nan)
    if ex.get('score', np.nan) == ex.get('score', np.nan) and abs(ex['score'] - pm) > 0.5:
        out.append(f"NASA's ExoMiner++ disagrees (its score {ex['score']:.2f}, ours {pm:.2f}): look closely at this one")
    if 'second_dip_phase' in rec:
        out.append(f"a second, separate dip at the same period ({rec['second_dip_ppm']:.0f} ppm, {rec['second_dip_phase']:.2f} of an orbit away): "
                   "looks like an eclipsing binary")
    return out


def run(segs, meta, scorer, progress=print, stage=None):
    """stage(name, **data), if given, receives each step's real intermediate data for the live view."""
    t_start = time.time()
    stage = stage or (lambda *a, **k: None)
    if not segs:
        raise ValueError('No usable data points in this light curve.')
    progress('Detrending for the search (biweight)...')
    t, f = flatten_for_search(segs)
    stage('flatten', t=t, f=f)
    progress(f'Searching for repeating dips with BLS on {t.size:,} points...')
    def _found(k, s):
        progress(f"  search {k + 1}: period {s['period']:.4f} d, depth {s['depth_ppm']:.0f} ppm, BLS SNR {s['bls_snr']:.1f}")
        stage('peak', k=k, sig=s)
    sigs, pgrams = bls_search(t, f, progress=_found, on_chunk=lambda k, parts, n: stage('search', k=k, parts=parts, n=n))
    out = dict(meta=meta, n_points=int(t.size), time_span_days=float(t.max() - t.min()), signals=[],
               search_curve=(t, f), periodograms=pgrams)
    stage('searched', sigs=sigs)
    if not sigs:
        out['seconds'] = time.time() - t_start
        return out
    progress(f'Found {len(sigs)} signal(s). Building views (AstroNet spline) and check numbers...')
    rows = [dict(tce_period=s['period'], tce_time0bk=s['t0'], tce_duration=s['duration']) for s in sigs]
    recs = tv.process_star([s[0] for s in segs], [s[1] for s in segs], rows)
    stage('fold', sigs=sigs, recs=recs)
    progress('Light-position check, star size and catalogue match...')
    from . import extras as E, tv_centroid as C
    if '_star' in meta:            # star properties given by the caller (tests, offline use)
        star = meta['_star']
    else:
        star = E.star_props(meta['target'], meta['mission']) if meta['target'].upper().startswith(('KIC', 'TIC')) else {}
    out['star'] = star
    span = float(t.max() - t.min())
    for s, r in zip(sigs, recs):
        others = [(o['period'], o['t0'], o['duration']) for o in sigs if o is not s]
        sh = C.segment_shifts(meta.get('_cen', []), s['period'], s['t0'], s['duration'], others)
        r['cen_sigma'], r['cen_offset_arcsec'] = C.combine(sh, r.get('depth_ppm', np.nan), meta['mission'])
        r['contratio'] = star.get('contratio', np.nan); r['star_rad'] = star.get('star_rad', np.nan)
        r['rp_est_re'] = E.planet_radius(r.get('depth_ppm', np.nan), r['star_rad'])
        r['known'] = E.known_match(meta['target'], meta['mission'], s['period'])
        r['exominer'] = E.exominer_match(meta['target'], meta['mission'], s['period'])
        from .fit import fit_transit
        r['fit'] = fit_transit(t, f, s['period'], s['t0'], s['duration'], s['depth_ppm'], r['star_rad'],
                               star.get('star_logg', np.nan), meta['mission'])
        fp = r['fit'] if r['fit'] and r['fit']['fit_ok'] else None
        r['next_dips'] = E.next_dips(fp['period'] if fp else s['period'], fp['t0'] if fp else s['t0'],
                                     meta['mission'], span, s['duration'])
        stage('check', i=sigs.index(s), sig=s, rec=r, star=star)
    if meta.get('_pixel') and meta['target'].upper().startswith(('KIC', 'TIC')):
        progress('Pixel check: downloading camera images (Target Pixel Files) and making difference images...')
        try:
            stage('pixel_start')
            pixel_check(meta, sigs, recs, star, keep_files=meta.get('_keep_files', True))
            stage('pixel', recs=recs)
        except Exception as e:      # the pixel check is optional: never fail the whole run because of it
            progress(f'  pixel check skipped: {e}')
    progress('Scoring with the three models and the combined score...')
    probs = scorer.score(recs, meta['mission'])
    main = MAIN_MODEL[meta['mission']] if meta['mission'] in scorer.comb else MAIN_MODEL_V1[meta['mission']]
    for i, (s, r) in enumerate(zip(sigs, recs)):
        p = {k: float(v[i]) for k, v in probs.items() if not k.startswith('_')}
        if '_conformal_p' in probs:
            r['triage'] = triage(float(probs['_conformal_p'][i]), scorer.conf[meta['mission']], CONFORMAL_ALPHA)
        r.update(s); r['scores'] = p; r['main_model'] = main; r['mission_'] = meta['mission']
        r['verdict'] = verdict(p[main], [p[k] for k in p if k != main])
        # (with the combined score, 'others' are the three single models: uncertain only if all three disagree)
        r['flags'] = flags(r)
        if r['verdict'] == 'candidate' and (r.get('n_transits', 99) < 3 or 'second_dip_phase' in r or pixel_warning(r)):
            r['verdict'] = 'uncertain'   # too few dips to trust the period, eclipsing-binary sign, or the dip comes from a neighbour
        out['signals'].append(r)
    out['seconds'] = time.time() - t_start
    stage('verdict', signals=out['signals'])
    return out


def pixel_check(meta, sigs, recs, star, keep_files=True):
    """Difference-image check for every signal (see tv_pixel). Adds px_* fields to each record."""
    from . import tv_pixel as X
    tpfs = X.download_tpfs(meta['target'], meta['mission'])
    meta['n_pixel_files'] = len(tpfs)
    res = X.check_signals(tpfs, [dict(period=s['period'], t0=s['t0'], duration=s['duration']) for s in sigs], meta['mission'])
    for r, x in zip(recs, res):
        r.update(x)
        lim = PIXEL_WARN.get(meta['mission'])
        if lim and np.isfinite(x['px_sigma']) and x['px_sigma'] > lim[0] and x['px_offset_arcsec'] > lim[1] and np.isfinite(star.get('ra', np.nan)):
            used = [g for g in x['px_segments'] if g.get('used')]
            if used:
                w = np.array([1 / g['err'] ** 2 for g in used])
                e = float(np.sum(w * [g['east'] for g in used]) / w.sum()); n = float(np.sum(w * [g['north'] for g in used]) / w.sum())
                r['px_source'] = X.nearby_source(star['ra'], star['dec'], e, n, meta['mission'])
    if not keep_files:
        for tp in tpfs:
            try:
                os.remove(tp.path)
            except Exception:
                pass


def summary_table(result):
    import pandas as pd
    rows = []
    for i, r in enumerate(result['signals']):
        rows.append({'target': result['meta']['target'], 'mission': result['meta']['mission'], 'signal': i + 1,
                     'verdict': r['verdict'], 'main_model': MODEL_NAMES[r['main_model']],
                     'p_main': round(r['scores'][r['main_model']], 3),
                     'p_gradient_boosting': round(r['scores']['gb'], 3), 'p_cnn': round(r['scores']['cnn'], 3),
                     'p_transformer': round(r['scores']['tf'], 3),
                     'p_combined': round(r['scores'].get('combined', np.nan), 3),
                     'trust_check_95': (r.get('triage') or {}).get('call', ''),
                     'period_days': round(r['period'], 5), 't0': round(r['t0'], 5), 'duration_hours': round(24 * r['duration'], 2),
                     'depth_ppm': round(r['depth_ppm'], 1), 'bls_snr': round(r['bls_snr'], 1),
                     'n_transits': r.get('n_transits'), 'odd_even_sigma': round(float(r.get('odd_even_sigma', np.nan)), 2),
                     'secondary_sigma': round(float(r.get('secondary_sigma', np.nan)), 2),
                     'planet_size_earth_radii': round(float(r.get('rp_est_re', np.nan)), 2),
                     'light_shift_sigma': round(float(r.get('cen_sigma', np.nan)), 2),
                     'source_offset_arcsec': round(float(r.get('cen_offset_arcsec', np.nan)), 1),
                     'pixel_offset_arcsec': round(float(r.get('px_offset_arcsec', np.nan)), 1),
                     'pixel_sigma': round(float(r.get('px_sigma', np.nan)), 2),
                     'nasa_exominer_pp': round(float((r.get('exominer') or {}).get('score', np.nan)), 3),
                     'catalogue': r.get('known', {}).get('status', ''), 'catalogue_name': r.get('known', {}).get('name', ''),
                     'next_dip_utc': (r.get('next_dips') or [{}])[0].get('utc', ''),
                     'flags': '; '.join(r['flags'])})
    return pd.DataFrame(rows)


def manifest(result, scorer):
    import platform, sklearn, torch, astropy
    return dict(tool='TransitVet', run_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), target={k: v for k, v in result['meta'].items() if not k.startswith('_')}, star=result.get('star', {}),
                n_points=result['n_points'], time_span_days=result['time_span_days'], seconds=result.get('seconds'),
                search=dict(method='BLS (astropy)', durations_days=SEARCH_DURATIONS.tolist(), min_period_days=MIN_PERIOD,
                            max_signals=MAX_SIGNALS, min_snr=MIN_SNR),
                models=scorer.manifest, versions=dict(python=platform.python_version(), numpy=np.__version__,
                                                      sklearn=sklearn.__version__, torch=torch.__version__, astropy=astropy.__version__))


def report_card(rec):
    """Plain-words list of every check for one signal: (check, what we found, status).
    status: 'pass' (looks like a planet), 'warn' (points to a false alarm), 'n/a' (could not check)."""
    f = lambda k: float(rec.get(k, np.nan)) if rec.get(k) is not None else np.nan
    rows = []
    def add(check, ok, text):
        rows.append((check, text, 'n/a' if ok is None else ('pass' if ok else 'warn')))
    tr = rec.get('triage')
    if tr:
        add(f"Trust check ({100 * (1 - tr['alpha']):.0f}% guarantee)", None if tr['call'] == 'needs a human' else tr['call'] == 'keep', tr['text'])
    n = rec.get('n_transits')
    add('Repeats', None if n is None else n >= 3, f'{n} dips seen' if n is not None else 'unknown')
    oe = f('odd_even_sigma')
    add('Odd and even dips alike', None if not np.isfinite(oe) else oe <= 3, f'difference {oe:.1f} sigma' if np.isfinite(oe) else 'not measured')
    ss = f('secondary_sigma')
    add('No dip half an orbit later', None if not np.isfinite(ss) else ss <= 5, f'{ss:.1f} sigma' if np.isfinite(ss) else 'not measured')
    if 'second_dip_phase' in rec:
        add('No second dip at the same period', False, f"second dip {rec['second_dip_ppm']:.0f} ppm, {rec['second_dip_phase']:.2f} of an orbit away")
    else:
        add('No second dip at the same period', True, 'none found')
    cs, co = f('cen_sigma'), f('cen_offset_arcsec')
    lim = LIGHT_OFFSET_WARN[rec.get('mission_', 'kepler')]
    if lim is None:
        add('Light stays on the target star', None, 'not reliable on TESS (pixels 21 arcsec wide); the models still use it')
    elif np.isfinite(cs):
        bad = cs > lim[0] and np.isfinite(co) and co > lim[1]
        add('Light stays on the target star', not bad, f'shift {cs:.1f} sigma, source about {co:.0f} arcsec away' if np.isfinite(co) else f'shift {cs:.1f} sigma')
    else:
        add('Light stays on the target star', None, 'no position data in the files')
    if 'px_sigma' in rec:
        n = rec.get('px_n_segments', 0)
        if not n:
            add('Dip comes from the target (camera images)', None, 'dip too faint in the camera images, or no images')
        else:
            add('Dip comes from the target (camera images)', not pixel_warning(rec),
                f"source {rec['px_offset_arcsec']:.0f} arcsec from the target, {rec['px_sigma']:.1f} sigma ({n} segment{'s' if n > 1 else ''})")
    ft = rec.get('fit') or {}
    rp = ft.get('planet_radius_re', f('rp_est_re'))
    if np.isfinite(rp):
        err = ft.get('planet_radius_err')
        add('Size fits a planet', rp <= 2 * 11.2, f'about {rp:.1f}' + (f' ± {err:.1f}' if err and np.isfinite(err) else '') + ' Earth radii'
            + (f' ({rp / 11.2:.1f} Jupiters)' if rp > 6 else ''))
    else:
        add('Size fits a planet', None, 'star radius unknown')
    if ft.get('fit_ok'):
        add('Dip shape (U, not V)', not ft['grazing'], f"impact {ft['b']:.2f}, size ratio {ft['rp_rs']:.3f}")
        if 'rho_star_cat' in ft and ft['rho_star_cat'] > 0:
            r = ft['rho_star_fit'] / ft['rho_star_cat']
            add('Dip matches the star', 0.2 <= r <= 5, f'implied star density {r:.1f}x the catalogue value')
    else:
        add('Dip shape (U, not V)', None, 'transit fit failed')
    return rows
