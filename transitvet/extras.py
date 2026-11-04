"""Extra information shown with each signal (not all of it is used by the models):
 - star properties from the TESS Input Catalog (TIC v8.2, queried at MAST): radius, temperature, crowding
 - planet size estimate from the dip depth and the star's radius
 - match against the official lists: Kepler KOI cumulative table and the TESS TOI table
   (downloaded 1 Nov 2026 from the NASA Exoplanet Archive). Used for display only, never in the score.
 - predicted next dip times
"""
import os, functools, numpy as np, pandas as pd

R_SUN_IN_EARTH = 109.1
R_JUP_IN_EARTH = 11.2
TIME_OFFSET = {'kepler': 2454833.0, 'tess': 2457000.0}     # BKJD / BTJD -> BJD
CATALOG_DATE = '1 Nov 2026'


@functools.lru_cache(maxsize=512)
def star_props(target, mission):
    """TIC row for 'KIC n' or 'TIC n'. Returns a dict (values may be NaN); {} if the lookup fails."""
    try:
        from astroquery.mast import Catalogs
        kind, num = target.split()[0].upper(), target.split()[-1]
        if not num.isdigit():
            return {}
        r = Catalogs.query_criteria(catalog='Tic', **({'KIC': [num]} if kind == 'KIC' else {'ID': [num]}))
        if len(r) == 0:
            return {}
        r = r.to_pandas().sort_values('rad', na_position='last').iloc[0]
        f = lambda k: float(r[k]) if k in r and pd.notna(r[k]) else np.nan
        return dict(tic=int(r['ID']), star_rad=f('rad'), star_teff=f('Teff'), star_logg=f('logg'),
                    tmag=f('Tmag'), vmag=f('Vmag'), ra=f('ra'), dec=f('dec'), contratio=f('contratio'))
    except Exception:
        return {}


def planet_radius(depth_ppm, star_rad):
    """Rough size of the object causing the dip, in Earth radii: sqrt(depth) x star radius.
    Ignores limb darkening and grazing dips, so treat it as +-30% at best."""
    if not (np.isfinite(depth_ppm) and depth_ppm > 0 and np.isfinite(star_rad)):
        return np.nan
    return float(np.sqrt(depth_ppm * 1e-6) * star_rad * R_SUN_IN_EARTH)


@functools.lru_cache(maxsize=1)
def _known():
    return pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'known_signals.csv.gz'))


PLANET = {'CONFIRMED', 'CP', 'KP'}
CANDIDATE = {'CANDIDATE', 'PC', 'APC'}
FALSE_POS = {'FALSE POSITIVE', 'FP', 'FA'}


def known_match(target, mission, period):
    """Is this signal already in the official lists? Match: same star and period within 1%
    (or half / double). Returns dict(status, name, disposition, catalogue_period)."""
    try:
        num = int(str(target).split()[-1])
    except ValueError:
        return dict(status='not checked (no KIC/TIC number)')
    k = _known(); k = k[(k.mission == mission) & (k.star == num)]
    if k.empty:
        return dict(status='star not in the official lists')
    for h in (1.0, 2.0, 0.5):
        m = k[np.abs(period * h - k.period) / k.period < 0.01]
        if len(m):
            r = m.iloc[0]; d = str(r.disposition)
            what = ('known planet' if d in PLANET else 'known candidate' if d in CANDIDATE
                    else 'known false positive' if d in FALSE_POS else 'listed')
            if h != 1.0:
                what += f' (our period is {"half" if h == 2.0 else "double"} the listed one)'
            return dict(status=what, name=str(r['name']), disposition=d, catalogue_period=float(r.period))
    return dict(status='not in the official lists for this star (could be a new signal, or an artefact: check the plots)',
                others=', '.join(f"{n} ({d}, {p:.3f} d)" for n, d, p in zip(k['name'], k.disposition, k.period)))


def next_dips(period, t0, mission, span_days, duration, n=5, now=None):
    """Next n predicted dip mid-times (UTC) after today, with a rough timing uncertainty.
    Rough period error ~ 0.1 x duration / (number of orbits in the data); it grows with every orbit
    after the data end, so old Kepler signals can be hours or days off by now."""
    from astropy.time import Time
    now = now or Time.now()
    t0_bjd = t0 + TIME_OFFSET[mission]
    k0 = int(np.ceil((now.jd - t0_bjd) / period))
    sig_p = 0.1 * duration / max(span_days / period, 1.0)
    out = []
    for i in range(n):
        k = k0 + i
        out.append(dict(utc=Time(t0_bjd + k * period, format='jd', scale='tdb').utc.iso[:16],
                        plus_minus_hours=round(float(k * sig_p * 24), 1)))
    return out


@functools.lru_cache(maxsize=1)
def _exominer():
    return pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'exominer_pp_scores.csv.gz'))


def exominer_match(target, mission, period):
    """NASA's ExoMiner++ score for this signal, from its published catalogue (Valizadegan, Martinho et al.,
    AJ 170, 2025; Zenodo record 15466293): TESS SPOC 2-min signals, sectors 1-67. Only TESS stars, and only
    signals that NASA's own pipeline also detected. Labelled signals carry out-of-fold (cross-validation) scores.
    Shown as a second opinion; never used in TransitVet's own score."""
    if mission != 'tess':
        return None
    try:
        num = int(str(target).split()[-1])
    except ValueError:
        return None
    e = _exominer(); e = e[e.target_id == num]
    for h in (1.0, 2.0, 0.5):
        m = e[np.abs(period * h - e.tce_period) / e.tce_period < 0.01]
        if len(m):
            def span(r):
                a = str(r).lstrip('S').split('-')
                try: return int(a[-1]) - int(a[0]) + 1
                except ValueError: return 0
            best = m.iloc[int(np.argmax([span(r) for r in m.run]))]     # longest multi-sector run: most data
            return dict(score=float(best.score), best_run='S' + str(best.run).lstrip('S'), n_runs=int(len(m)),
                        score_range=f'{m.score.min():.2f}-{m.score.max():.2f}', runs=', '.join(m.run.astype(str)[:6]),
                        nasa_label=', '.join(sorted(set(m.label.astype(str)))),
                        note='' if h == 1.0 else f'matched at {"half" if h == 2.0 else "double"} our period')
    return dict(score=np.nan, note='not in the ExoMiner++ catalogue (NASA pipeline did not report this signal in sectors 1-67)')
