"""TransitVet: light-position (centroid) check.

Idea: each light-curve file also records, for every measurement, where the star's light
falls on the camera (MOM_CENTR1 = column, MOM_CENTR2 = row, in pixels). If the dip comes
from the target star, that spot hardly moves during the dip. If the dip comes from a nearby
star blended in the same pixels, the spot shifts during the dip, towards or away from it.

For each segment (Kepler quarter / TESS sector; the camera orientation changes between them,
so segments are never averaged in pixel space):
  1. remove the slow drift of the centroid (running median in short bins, transits masked out),
  2. shift = mean(in-dip) - 0, error from the scatter of dip-length blocks outside the dips
     (handles correlated noise).
Combined over segments:
  cen_sigma     how significant the shift is (chi-square over all segments, turned into sigma)
  cen_offset    how far away the source of the dip seems to be: shift / depth, in arcsec
                (a 1% dip that moves the centroid by 0.01 px comes from ~1 px away).
"""
import io, time as _time, numpy as np

PIXEL_ARCSEC = {'kepler': 3.98, 'tess': 21.0}


def _fetch(session, url, retries=3, timeout=90):
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=timeout)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.content
        except Exception:
            _time.sleep(2 ** attempt)
    return None


def _read(data, bin_minutes=None):
    """time, PDCSAP flux, centroid column, centroid row; good-quality finite points only."""
    from astropy.io import fits
    with fits.open(io.BytesIO(data)) as h:
        lc = h['LIGHTCURVE'].data
        t = np.asarray(lc['TIME'], float); f = np.asarray(lc['PDCSAP_FLUX'], float)
        cx = np.asarray(lc['MOM_CENTR1'], float); cy = np.asarray(lc['MOM_CENTR2'], float)
        q = np.asarray(lc['QUALITY'] if 'QUALITY' in lc.names else lc['SAP_QUALITY'])
    ok = np.isfinite(t) & np.isfinite(f) & np.isfinite(cx) & np.isfinite(cy) & (q == 0)
    t, f, cx, cy = t[ok], f[ok], cx[ok], cy[ok]
    if t.size < 50:
        return None
    if bin_minutes:
        k = np.floor((t - t[0]) / (bin_minutes / 1440.0)).astype(int)
        n = np.bincount(k); g = n > 0
        t, f, cx, cy = [np.bincount(k, a)[g] / n[g] for a in (t, f, cx, cy)]
    return t, f, cx, cy


def download_kepler_cen(kepid, session):
    from .tv_kepler import kepler_urls
    out = []
    for _, url in kepler_urls(kepid):
        d = _fetch(session, url)
        if d is not None:
            s = _read(d)
            if s is not None:
                out.append(s)
    return out


def download_tess_cen(urls, session):
    out = []
    for url in sorted(urls):
        d = _fetch(session, url, timeout=120)
        if d is not None:
            s = _read(d, bin_minutes=30.0)
            if s is not None:
                out.append(s)
    return out


def _detrend(t, y, mask, window):
    """y minus a slow trend: medians of the out-of-dip points in bins of `window` days,
    joined by straight lines (fast running-median; the dips are masked out)."""
    keep = ~mask
    if keep.sum() < 20:
        return None
    k = np.floor((t[keep] - t[0]) / window).astype(int)
    order = np.argsort(k, kind='stable'); kk = k[order]; yy = y[keep][order]; tt = t[keep][order]
    cuts = np.flatnonzero(np.diff(kk)) + 1
    groups_y = np.split(yy, cuts); groups_t = np.split(tt, cuts)
    ok = [g.size >= 3 for g in groups_y]
    if sum(ok) < 2:
        return None
    ct = np.array([g.mean() for g, o in zip(groups_t, ok) if o]); cy = np.array([np.median(g) for g, o in zip(groups_y, ok) if o])
    return y - np.interp(t, ct, cy)


def segment_shifts(segs, period, t0, duration, other_tces=()):
    """Per segment: (dx, ex, dy, ey, n_in) for one signal. segs: list of (t, f, cx, cy)."""
    out = []
    window = max(0.5, 2.0 * duration)
    for t, f, cx, cy in segs:
        dt = (t - t0 + 0.5 * period) % period - 0.5 * period
        in_dip = np.abs(dt) < 0.5 * duration
        mask = np.abs(dt) < 1.0 * duration
        for p2, t2, d2 in other_tces:          # keep other signals out of the baseline
            mask |= np.abs((t - t2 + 0.5 * p2) % p2 - 0.5 * p2) < 1.0 * d2
        if in_dip.sum() < 3:
            continue
        row = []
        for c in (cx, cy):
            r = _detrend(t, c, mask, window)
            if r is None:
                break
            base = r[~mask]
            # error of a dip-length average, from blocks of the same length outside the dips
            blk = np.floor((t[~mask] - t[0]) / duration).astype(int)
            n = np.bincount(blk); g = n >= max(2, 0.5 * np.median(n[n > 0]))
            means = (np.bincount(blk, base)[g] / n[g])
            if means.size < 10:
                break
            sig_blk = 1.4826 * np.median(np.abs(means - np.median(means)))
            pts_per_dip = np.median(n[g])
            n_eff = max(in_dip.sum() / pts_per_dip, 1.0)
            row += [float(np.mean(r[in_dip]) - np.median(base)), float(sig_blk / np.sqrt(n_eff))]
        if len(row) == 4 and row[1] > 0 and row[3] > 0:
            out.append((row[0], row[1], row[2], row[3], int(in_dip.sum())))
    return out


def combine(shifts, depth_ppm, mission):
    """cen_sigma, cen_offset_arcsec from the per-segment shifts (see module text)."""
    from scipy import stats
    if not shifts:
        return np.nan, np.nan
    a = np.asarray(shifts, float)
    dx, ex, dy, ey = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    chi2 = float(np.sum((dx / ex) ** 2 + (dy / ey) ** 2)); dof = 2 * len(a)
    sf = stats.chi2.sf(chi2, dof)
    sigma = float(np.clip(stats.norm.isf(sf) if sf > 0 else 40.0, -5, 40))
    w = 1.0 / (ex ** 2 + ey ** 2)
    s2 = np.sum(w * (dx ** 2 + dy ** 2 - ex ** 2 - ey ** 2)) / np.sum(w)   # noise-debiased squared shift
    shift_px = np.sqrt(max(s2, 0.0))
    if not np.isfinite(depth_ppm) or depth_ppm <= 0:
        return sigma, np.nan
    offset = shift_px / (depth_ppm * 1e-6) * PIXEL_ARCSEC[mission]
    return sigma, float(np.clip(offset, 0, 600))
