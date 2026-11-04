"""TransitVet: pixel-level check (difference imaging) - where on the sky does the dip come from?

The light curve adds up the light of a small patch of pixels. A nearby eclipsing star that falls in
the same patch can make a dip that looks like a planet on the target. The camera images
(Target Pixel Files, TPFs) let us see which pixels actually get darker.

For each segment (TESS sector / Kepler quarter):
  1. for every dip, take the mean image during the dip (central 80%) and the mean image just
     before and just after it (from 0.75 to 1.75 dip-lengths from the centre);
  2. difference image = out-of-dip minus in-dip, averaged over the dips. Its bright spot marks the
     source of the dip; the out-of-dip image's bright spot marks where most light comes from;
  3. fit a 2-D Gaussian (star image) to both; errors from the dip-to-dip scatter of the pixels;
  4. turn the pixel positions into sky positions (the file's WCS) and measure how far the dip's
     source is from the catalogue position of the target.
Segments are combined on the sky (camera orientation changes between sectors).
This is the same idea as the difference-image centroid test in NASA's Kepler and TESS pipelines
(Bryson et al. 2013; Twicken et al. 2018), done more simply: a Gaussian instead of the real PRF.

Outputs per signal:
  px_offset_arcsec   distance of the dip's source from the target's catalogue position
  px_sigma           how significant that offset is
  px_n_segments      number of segments used
  px_best            images of the clearest segment, for plotting
  px_source          nearest catalogue star to the dip's source, if the offset is significant
"""
import os, warnings
import numpy as np

PIXEL_ARCSEC = {'kepler': 3.98, 'tess': 21.0}
SYS_FLOOR_PX = {'kepler': 0.05, 'tess': 0.05}    # systematic error floor per segment (pixels)
MIN_DIFF_SNR = 5.0                               # skip segments where the dip is not visible in the pixels
MAX_SEGMENTS = {'kepler': 17, 'tess': int(os.environ.get('TV_PIXEL_MAX_SECTORS', 3))}


def _gauss(p, yy, xx):
    a, x0, y0, s, b = p
    return a * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * s * s)) + b


def fit_star(img, err=None, mask=None):
    """2-D Gaussian fit to an image. Returns (x, y, ex, ey) in pixels (column, row), or None."""
    from scipy.optimize import least_squares
    ny, nx = img.shape
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    good = np.isfinite(img) & (mask if mask is not None else True)
    if err is not None:
        good &= np.isfinite(err) & (err > 0)
    if good.sum() < 9:
        return None
    w = 1.0 / err[good] if err is not None else np.ones(good.sum())
    j = np.nanargmax(np.where(good, img, -np.inf))
    y0, x0 = np.unravel_index(j, img.shape)
    p0 = [float(img[y0, x0] - np.nanmedian(img[good])), float(x0), float(y0), 1.0, float(np.nanmedian(img[good]))]
    if p0[0] <= 0:
        return None
    lo, hi = [0, -0.5, -0.5, 0.3, -np.inf], [np.inf, nx - 0.5, ny - 0.5, 4.0, np.inf]
    p0 = list(np.clip(p0, np.array(lo) + 1e-6, np.array(hi) - 1e-6))
    fn = lambda p: (_gauss(p, yy[good], xx[good]) - img[good]) * w
    try:
        r = least_squares(fn, p0, bounds=(lo, hi), x_scale='jac', max_nfev=300)
        dof = max(good.sum() - 5, 1)
        s2 = max(float(np.sum(r.fun ** 2) / dof), 1.0) if err is not None else float(np.sum(r.fun ** 2) / dof)
        cov = np.linalg.pinv(r.jac.T @ r.jac) * s2
        ex, ey = np.sqrt(max(cov[1, 1], 0)), np.sqrt(max(cov[2, 2], 0))
    except Exception:
        return None
    if not r.success:
        return None
    return float(r.x[1]), float(r.x[2]), float(ex), float(ey)


def _diffs(t, cube, dt, epoch, ins, oot):
    out, oots = [], []
    for e in np.unique(epoch[ins]):
        i_in = ins & (epoch == e)
        i_before = oot & (epoch == e) & (dt < 0)
        i_after = oot & (epoch == e) & (dt > 0)
        if i_in.sum() < 2 or i_before.sum() < 2 or i_after.sum() < 2:
            continue
        o = 0.5 * (np.nanmean(cube[i_before], 0) + np.nanmean(cube[i_after], 0))
        out.append(o - np.nanmean(cube[i_in], 0)); oots.append(o)
    return out, oots


def difference_image(t, cube, period, t0, duration, other=()):
    """Mean out-of-dip minus in-dip image over all dips with data on both sides.
    Pixel errors come from the same measurement made at fake dip times (between the real dips):
    their scatter is the noise of a difference image (handles slow drifts and correlated noise).
    Returns (diff, diff_err, oot_mean, n_dips) or None."""
    busy = np.zeros(t.size, bool)
    for p2, t2, d2 in other:
        busy |= np.abs((t - t2 + 0.5 * p2) % p2 - 0.5 * p2) < 1.0 * d2
    def pattern(shift):
        dt = (t - (t0 + shift) + 0.5 * period) % period - 0.5 * period
        epoch = np.round((t - (t0 + shift)) / period).astype(int)
        ins = (np.abs(dt) < 0.4 * duration) & ~busy
        oot = (np.abs(dt) > 0.75 * duration) & (np.abs(dt) < 1.75 * duration) & ~busy
        return dt, epoch, ins, oot
    diffs, oots = _diffs(t, cube, *pattern(0.0))
    n = len(diffs)
    if n < 1:
        return None
    # fake dips: shifted by a fraction of the period, far from the real dips (and from half phase)
    real = np.abs((t - t0 + 0.5 * period) % period - 0.5 * period) < 1.0 * duration
    fake = []
    gap = 2.0 * duration / period
    for f in np.linspace(0.1, 0.9, 17):
        if min(f, 1 - f) < gap or abs(f - 0.5) < gap:
            continue
        dt, ep, ins, oot = pattern(f * period)
        fake += _diffs(t, cube, dt, ep, ins & ~real, oot & ~real)[0]
    if len(fake) >= 8:
        F = np.array(fake)
        mad = 1.4826 * np.nanmedian(np.abs(F - np.nanmedian(F, 0)), 0)
        err = mad / np.sqrt(n)
    elif n < 2:
        return None
    else:
        err = np.nanstd(np.array(diffs), 0, ddof=1) / np.sqrt(n)
    return np.nanmean(np.array(diffs), 0), err, np.nanmean(np.array(oots), 0), n


def _segment(tpf, period, t0, duration, other, mission):
    """One TPF -> dict with sky offset (arcsec, east and north) of the dip's source from the target, or None."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    t = np.asarray(tpf.time.value, float)
    cube = np.asarray(tpf.flux.value, float)
    q = np.asarray(tpf.quality.value if hasattr(tpf.quality, 'value') else tpf.quality)
    ok = np.isfinite(t) & (q == 0) & np.isfinite(cube).any(axis=(1, 2))
    t, cube = t[ok], cube[ok]
    if t.size < 50:
        return None
    r = difference_image(t, cube, period, t0, duration, other)
    if r is None:
        return None
    D, E, O, n = r
    E = np.where(np.isfinite(E) & (E > 0), E, np.nan)
    ap = np.asarray(tpf.pipeline_mask, bool) if tpf.pipeline_mask is not None else np.isfinite(D)
    snr = np.nansum(D[ap]) / np.sqrt(np.nansum(E[ap] ** 2)) if ap.any() else np.nan
    if not np.isfinite(snr) or snr < MIN_DIFF_SNR:
        return dict(n_dips=n, diff_snr=float(snr) if np.isfinite(snr) else None, used=False)
    cd = fit_star(D, E)
    co = fit_star(O)
    if cd is None or co is None:
        return dict(n_dips=n, diff_snr=float(snr), used=False)
    wcs = tpf.wcs
    target = SkyCoord(tpf.ra * u.deg, tpf.dec * u.deg)
    # pixel position of the target, from the WCS (TPF pixel 0,0 = image corner)
    tx, ty = wcs.world_to_pixel(target)
    src = wcs.pixel_to_world(cd[0], cd[1])
    oot = wcs.pixel_to_world(co[0], co[1])
    scale = PIXEL_ARCSEC[mission]
    dra, ddec = target.spherical_offsets_to(src)
    ora, odec = target.spherical_offsets_to(oot)
    err_px = np.hypot(cd[2], cd[3]) / np.sqrt(2)
    err = np.hypot(err_px, SYS_FLOOR_PX[mission]) * scale
    return dict(used=True, n_dips=n, diff_snr=float(snr), east=float(dra.to(u.arcsec).value), north=float(ddec.to(u.arcsec).value),
                err=float(err), oot_east=float(ora.to(u.arcsec).value), oot_north=float(odec.to(u.arcsec).value),
                diff=D, oot=O, target_px=(float(tx), float(ty)), source_px=(cd[0], cd[1]), oot_px=(co[0], co[1]),
                label=str(getattr(tpf, 'sector', None) or getattr(tpf, 'quarter', '')))


def combine(segs):
    """Weighted mean sky offset over segments -> (offset_arcsec, sigma, n_used, oot_offset_arcsec)."""
    from scipy import stats
    used = [s for s in segs if s and s.get('used')]
    if not used:
        return np.nan, np.nan, 0, np.nan
    e = np.array([s['north'] for s in used]); n_ = e  # noqa
    E = np.array([s['east'] for s in used]); N = np.array([s['north'] for s in used]); er = np.array([s['err'] for s in used])
    w = 1 / er ** 2
    me, mn, se = np.sum(w * E) / w.sum(), np.sum(w * N) / w.sum(), 1 / np.sqrt(w.sum())
    # if segments disagree more than their errors say, widen the error (scatter-based)
    if len(used) > 1:
        chi = np.sum(w * ((E - me) ** 2 + (N - mn) ** 2)) / (2 * (len(used) - 1))
        se *= np.sqrt(max(chi, 1.0))
    off = float(np.hypot(me, mn))
    chi2 = (me ** 2 + mn ** 2) / se ** 2
    sf = stats.chi2.sf(chi2, 2)
    sigma = float(np.clip(stats.norm.isf(sf) if sf > 0 else 40.0, -5, 40))
    oo = float(np.hypot(np.average([s['oot_east'] for s in used], weights=w), np.average([s['oot_north'] for s in used], weights=w)))
    return off, sigma, len(used), oo


def download_tpfs(target, mission, max_segments=None):
    import lightkurve as lk
    max_segments = max_segments or MAX_SEGMENTS[mission]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if mission == 'kepler':
            sr = lk.search_targetpixelfile(target, author='Kepler', exptime=1800)
        else:
            sr = lk.search_targetpixelfile(target, author='SPOC', exptime=120)
        if len(sr) == 0:
            return []
        sr = sr[-max_segments:] if mission == 'tess' else sr[:max_segments]
        out = []
        for i in range(len(sr)):
            try:
                out.append(sr[i].download(quality_bitmask='none'))
            except Exception:
                pass
    return out


def check_signals(tpfs, signals, mission, keep_images=True):
    """signals: list of dicts with period, t0 (in the TPF time system), duration (days).
    Returns one dict per signal."""
    out = []
    for s in signals:
        other = [(o['period'], o['t0'], o['duration']) for o in signals if o is not s]
        segs = []
        for tpf in tpfs:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    segs.append(_segment(tpf, s['period'], s['t0'], s['duration'], other, mission))
            except Exception:
                segs.append(None)
        off, sig, n, oo = combine(segs)
        r = dict(px_offset_arcsec=off, px_sigma=sig, px_n_segments=n, px_oot_offset_arcsec=oo,
                 px_segments=[{k: v for k, v in (g or {}).items() if k not in ('diff', 'oot')} for g in segs])
        used = [g for g in segs if g and g.get('used')]
        if keep_images and used:
            best = max(used, key=lambda g: g['diff_snr'])
            r['px_best'] = {k: best[k] for k in ('diff', 'oot', 'target_px', 'source_px', 'oot_px', 'label', 'diff_snr')}
        out.append(r)
    return out


def nearby_source(ra, dec, east, north, mission, radius_arcsec=None):
    """Nearest catalogue (TIC) star to the dip's source position. Returns dict or None."""
    try:
        from astroquery.mast import Catalogs
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        c = SkyCoord(ra * u.deg, dec * u.deg)
        src = c.spherical_offsets_by(east * u.arcsec, north * u.arcsec)
        rad = radius_arcsec or 3 * PIXEL_ARCSEC[mission]
        tab = Catalogs.query_region(src, radius=rad * u.arcsec, catalog='TIC')
        if len(tab) == 0:
            return None
        sep = SkyCoord(np.asarray(tab['ra'], float) * u.deg, np.asarray(tab['dec'], float) * u.deg).separation(src).arcsec
        i = int(np.argmin(sep))
        return dict(tic=str(tab['ID'][i]), tmag=float(tab['Tmag'][i]), sep_from_source_arcsec=float(sep[i]),
                    sep_from_target_arcsec=float(SkyCoord(float(tab['ra'][i]) * u.deg, float(tab['dec'][i]) * u.deg).separation(c).arcsec))
    except Exception:
        return None
