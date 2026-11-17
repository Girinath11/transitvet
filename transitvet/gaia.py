"""Gaia DR2 information about a TESS star and its neighbours (queried at MAST, 60 arcsec around the star).
Used by the TESS gradient-boosting model since models v4 (17 Nov 2026):
 - the star's own astrometry: excess noise, fit quality, radial-velocity error, colour excess. A star with a hidden
   companion (a binary) often shows extra wobble or a large radial-velocity scatter.
 - neighbours: how many Gaia stars lie within 21" (one TESS pixel) and 42", how much light they add, and the brightest one.
 - g_log_nb_depth: how deep the brightest neighbour's own dip would have to be to fake this signal (log10 of a fraction;
   above 0 means it would need more than 100%, so that neighbour cannot be the source).
The same code built the training table, so live and training numbers match."""
import functools
import numpy as np

FEATS = ['g_aen', 'g_aen_sig', 'g_gof', 'g_rv_err', 'g_bprp_exc', 'g_plx_snr', 'g_n21', 'g_fr21', 'g_n42', 'g_fr42', 'g_nb_dmag', 'g_nb_sep', 'g_log_nb_depth']
COLS = ['source_id', 'ra', 'dec', 'parallax', 'parallax_error', 'phot_g_mean_mag', 'bp_rp', 'astrometric_excess_noise', 'astrometric_excess_noise_sig',
        'astrometric_gof_al', 'radial_velocity_error', 'phot_bp_rp_excess_factor']


def _sep(ra0, de0, ra, de):
    r = np.radians
    c = np.sin(r(de0)) * np.sin(r(de)) + np.cos(r(de0)) * np.cos(r(de)) * np.cos(r(np.asarray(ra) - ra0))
    return 3600 * np.degrees(np.arccos(np.clip(c, -1, 1)))


def from_rows(ra0, dec0, rows):
    """Star-level numbers from a list of Gaia rows (dicts). {} when no Gaia star lies within 10" of the position
    (the brightest star within 3" is the target; if there is none, the brightest within 10")."""
    f = lambda x: np.nan if x is None else float(x)
    rows = [r for r in rows if r.get('phot_g_mean_mag') is not None]
    if not rows:
        return {}
    sep = _sep(ra0, dec0, [r['ra'] for r in rows], [r['dec'] for r in rows])
    order = np.argsort([r['phot_g_mean_mag'] for r in rows])
    near = [i for i in order if sep[i] < 3]
    if not near:       # fast-moving stars: the catalogue position (epoch 2000) can be several arcsec from Gaia's (epoch 2015.5)
        near = [i for i in order if sep[i] < 10]
    if not near:
        return {}
    k = near[0]; tg = rows[k]
    oth = [i for i in range(len(rows)) if rows[i]['source_id'] != tg['source_id']]
    mag = np.array([rows[i]['phot_g_mean_mag'] for i in oth], float); s = sep[oth] if oth else np.array([])
    fr = 10 ** (-0.4 * (mag - tg['phot_g_mean_mag']))
    pe = f(tg.get('parallax_error'))
    out = dict(g_mag=f(tg['phot_g_mean_mag']), g_aen=f(tg.get('astrometric_excess_noise')), g_aen_sig=f(tg.get('astrometric_excess_noise_sig')),
               g_gof=f(tg.get('astrometric_gof_al')), g_rv_err=f(tg.get('radial_velocity_error')), g_bprp_exc=f(tg.get('phot_bp_rp_excess_factor')),
               g_plx_snr=f(tg.get('parallax')) / pe if pe else np.nan)
    for r in (21, 42):
        m = s < r
        out[f'g_n{r}'] = int(m.sum()); out[f'g_fr{r}'] = float(fr[m].sum())
    m = s < 42
    if m.any():
        j = int(np.argmax(np.where(m, fr, -1)))
        out.update(g_nb_dmag=float(mag[j] - tg['phot_g_mean_mag']), g_nb_sep=float(s[j]), g_nb_fr=float(fr[j]))
    else:
        out.update(g_nb_dmag=10.0, g_nb_sep=60.0, g_nb_fr=0.0)
    return out


def to_gaia_epoch(ra, dec, pmra, pmdec, years=15.5):
    """Move a catalogue position (epoch 2000, as in the TIC) to Gaia DR2's epoch 2015.5 using its proper motion (mas/yr)."""
    if not (np.isfinite(pmra) and np.isfinite(pmdec)):
        return ra, dec
    return ra + pmra * years / 3.6e6 / max(np.cos(np.radians(dec)), 1e-6), dec + pmdec * years / 3.6e6


def nb_depth(depth_ppm, g):
    """log10 of the dip depth (as a fraction) the brightest neighbour within 42" would need to fake this signal."""
    if not g or not np.isfinite(depth_ppm if depth_ppm is not None else np.nan):
        return np.nan
    return float(np.log10(np.clip(depth_ppm * 1e-6 * (1 + g['g_fr42']) / max(g['g_nb_fr'], 1e-6), 1e-6, 1e3)))


@functools.lru_cache(maxsize=256)
def star_features(ra, dec):
    """Query Gaia DR2 at MAST around (ra, dec) in degrees. {} if the lookup fails (the model then treats these inputs as missing)."""
    try:
        from astroquery.mast import Catalogs
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        tab = Catalogs.query_region(SkyCoord(ra, dec, unit='deg'), radius=60 * u.arcsec, catalog='Gaia', version=2)
        rows = []
        for x in tab:
            d = {}
            for c in COLS:
                v = x[c] if c in tab.colnames else None
                d[c] = None if v is None or np.ma.is_masked(v) or (isinstance(v, float) and not np.isfinite(v)) else (int(v) if c == 'source_id' else float(v))
            rows.append(d)
        return from_rows(float(ra), float(dec), rows)
    except Exception:
        return {}
