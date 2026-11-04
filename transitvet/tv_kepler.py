"""TransitVet: turn Kepler light curves into model inputs.

For every labelled signal (TCE) this produces:
  * global view (2001 bins) and local view (201 bins), exactly as AstroNet builds them
  * the same two views from our biweight-detrended curve (for the detrending ablation)
  * a table of numbers: depth, SNR, odd-even difference, secondary dip, dip shape, ...

AstroNet settings were checked in google-research/exoplanet-ml (astronet/data/preprocess.py,
light_curve/binning.py, light_curve/kepler_io.py) on 30 Oct 2026.
"""
import io
import time as _time

import numpy as np

from . import astronet_spline

# Kepler long-cadence file prefixes per quarter (from AstroNet's kepler_io.py).
LONG_CADENCE_QUARTER_PREFIXES = {
    0: ["2009131105131"], 1: ["2009166043257"], 2: ["2009259160929"],
    3: ["2009350155506"], 4: ["2010078095331", "2010009091648"],
    5: ["2010174085026"], 6: ["2010265121752"], 7: ["2010355172524"],
    8: ["2011073133259"], 9: ["2011177032512"], 10: ["2011271113734"],
    11: ["2012004120508"], 12: ["2012088054726"], 13: ["2012179063303"],
    14: ["2012277125453"], 15: ["2013011073258"], 16: ["2013098041711"],
    17: ["2013131215648"],
}
MAST_BASE = "https://archive.stsci.edu/missions/kepler/lightcurves"

GLOBAL_BINS, LOCAL_BINS = 2001, 201
LOCAL_BIN_WIDTH_FACTOR, LOCAL_NUM_DURATIONS = 0.16, 4


# ---------------------------------------------------------------- download

def kepler_urls(kepid):
    kid = "%09d" % int(kepid)
    for quarter, prefixes in LONG_CADENCE_QUARTER_PREFIXES.items():
        for prefix in prefixes:
            yield quarter, "%s/%s/%s/kplr%s-%s_llc.fits" % (MAST_BASE, kid[:4], kid, kid, prefix)


def download_kepler(kepid, session, retries=3):
    """Returns lists (time, flux) per quarter: PDCSAP flux, finite points only (as AstroNet)."""
    from astropy.io import fits
    all_time, all_flux = [], []
    for quarter, url in kepler_urls(kepid):
        data = None
        for attempt in range(retries):
            try:
                r = session.get(url, timeout=60)
                if r.status_code == 404:
                    break
                r.raise_for_status()
                data = r.content
                break
            except Exception:
                _time.sleep(2 ** attempt)
        if data is None:
            continue
        with fits.open(io.BytesIO(data)) as hdul:
            lc = hdul["LIGHTCURVE"].data
            t = np.asarray(lc["TIME"], dtype=np.float64)
            f = np.asarray(lc["PDCSAP_FLUX"], dtype=np.float64)
        ok = np.isfinite(t) & np.isfinite(f)
        if ok.sum() > 0:
            all_time.append(t[ok])
            all_flux.append(f[ok])
    return all_time, all_flux


# ---------------------------------------------------------------- detrending

def split_at_gaps(all_time, all_flux, gap_width=0.75):
    out_t, out_f = [], []
    for t, f in zip(all_time, all_flux):
        cuts = np.where(np.diff(t) > gap_width)[0] + 1
        out_t.extend(np.split(t, cuts))
        out_f.extend(np.split(f, cuts))
    return out_t, out_f


def detrend_astronet(all_time, all_flux):
    """AstroNet: split at 0.75 d gaps, fit the Kepler spline (BIC-chosen spacing), divide."""
    seg_t, seg_f = split_at_gaps(all_time, all_flux)
    spline = astronet_spline.fit_kepler_spline(seg_t, seg_f, verbose=False)[0]
    t, f, s = np.concatenate(seg_t), np.concatenate(seg_f), np.concatenate(spline)
    ok = np.isfinite(s)
    return t[ok], f[ok] / s[ok]


def in_transit_mask(t, tces, width_factor=1.5):
    """True for points within width_factor * duration of any transit of the star's TCEs."""
    mask = np.zeros(t.size, dtype=bool)
    for period, t0, duration in tces:
        phase = np.abs((t - t0 + 0.5 * period) % period - 0.5 * period)
        mask |= phase < 0.5 * width_factor * duration
    return mask


def detrend_biweight(all_time, all_flux, tces, window_factor=3.0, min_window=0.5):
    """TransitVet: robust biweight filter with all known transits masked.

    The trend is fitted on out-of-transit points only and interpolated across the
    transits, so the dips are not flattened. Window = 3 x longest transit duration
    (at least 0.5 d).
    """
    from wotan import flatten
    window = max(min_window, window_factor * max(d for _, _, d in tces))
    seg_t, seg_f = split_at_gaps(all_time, all_flux)
    out_t, out_f = [], []
    for t, f in zip(seg_t, seg_f):
        f = f / np.median(f)
        oot = ~in_transit_mask(t, tces)
        if oot.sum() < 10:
            continue
        _, trend = flatten(t[oot], f[oot], method="biweight", window_length=window,
                           edge_cutoff=0, break_tolerance=0.75, return_trend=True)
        good = np.isfinite(trend)
        if good.sum() < 2:
            continue
        full_trend = np.interp(t, t[oot][good], trend[good])
        out_t.append(t)
        out_f.append(f / full_trend)
    if not out_t:
        return np.array([]), np.array([])
    return np.concatenate(out_t), np.concatenate(out_f)


# ---------------------------------------------------------------- views

def phase_fold(t, period, t0):
    """Folded time in [-period/2, period/2), 0 at t0, sorted (AstroNet util.phase_fold_time)."""
    half = period / 2
    folded = np.mod(t + (half - t0), period) - half
    order = np.argsort(folded)
    return folded, order


def bin_and_aggregate(x, y, num_bins, bin_width, x_min, x_max):
    """AstroNet binning.bin_and_aggregate: overlapping uniform bins, median per bin."""
    spacing = (x_max - x_min - bin_width) / (num_bins - 1)
    lo = x_min + spacing * np.arange(num_bins)
    start = np.searchsorted(x, lo, side="left")
    end = np.searchsorted(x, lo + bin_width, side="left")
    out = np.zeros(num_bins)
    counts = end - start
    for i in np.nonzero(counts)[0]:
        out[i] = np.median(y[start[i]:end[i]])
    return out, counts


def make_view(x, y, num_bins, bin_width, x_min, x_max):
    view, counts = bin_and_aggregate(x, y, num_bins, bin_width, x_min, x_max)
    view = np.where(counts > 0, view, np.median(y))
    view = view - np.median(view)
    low = np.abs(np.min(view))
    if low > 0:
        view = view / low
    return view.astype(np.float32), counts


def views_for_tce(t, f, period, t0, duration):
    x, order = phase_fold(t, period, t0)
    x, y = x[order], f[order]
    g, _ = make_view(x, y, GLOBAL_BINS, period / GLOBAL_BINS, -period / 2, period / 2)
    lw = LOCAL_NUM_DURATIONS * duration
    loc, lc = make_view(x, y, LOCAL_BINS, duration * LOCAL_BIN_WIDTH_FACTOR,
                        max(-period / 2, -lw), min(period / 2, lw))
    return g, loc, float((lc > 0).mean())


# ---------------------------------------------------------------- numbers table

def _depth(y_in, y_out):
    """Depth (ppm) and its 1-sigma error, from in-transit vs out-of-transit points."""
    if y_in.size < 2 or y_out.size < 10:
        return np.nan, np.nan
    mad = 1.4826 * np.median(np.abs(y_out - np.median(y_out)))
    depth = np.median(y_out) - np.mean(y_in)
    return 1e6 * depth, 1e6 * mad / np.sqrt(y_in.size)


def features_for_tce(t, f, period, t0, duration):
    """Eclipsing-binary diagnostics on the biweight-detrended curve."""
    epoch = np.round((t - t0) / period).astype(int)
    dt = t - (t0 + epoch * period)                      # time from nearest transit centre
    in_full = np.abs(dt) < 0.5 * duration
    in_core = np.abs(dt) < 0.25 * duration
    near = np.abs(dt) < 1.5 * duration
    # out-of-transit reference: 1.5 to 3 durations from centre
    ref = (np.abs(dt) > 1.5 * duration) & (np.abs(dt) < 3 * duration)

    depth, depth_err = _depth(f[in_full], f[ref])
    core_depth, _ = _depth(f[in_core], f[ref])
    odd, odd_err = _depth(f[in_full & (epoch % 2 == 1)], f[ref])
    even, even_err = _depth(f[in_full & (epoch % 2 == 0)], f[ref])

    dt2 = t - (t0 + period / 2 + np.round((t - t0 - period / 2) / period) * period)
    sec_in = np.abs(dt2) < 0.5 * duration
    sec_ref = (np.abs(dt2) > 1.5 * duration) & (np.abs(dt2) < 3 * duration) & ~near
    sec, sec_err = _depth(f[sec_in & ~near], f[sec_ref])

    n_transits = np.unique(epoch[in_full]).size
    return {
        "depth_ppm": depth,
        "snr": depth / depth_err if depth_err and depth_err > 0 else np.nan,
        "odd_even_sigma": abs(odd - even) / np.hypot(odd_err, even_err)
        if np.isfinite(odd_err * even_err) and np.hypot(odd_err, even_err) > 0 else np.nan,
        "secondary_ppm": sec,
        "secondary_sigma": sec / sec_err if sec_err and sec_err > 0 else np.nan,
        "shape_ratio": depth / core_depth if core_depth and core_depth > 0 else np.nan,
        "n_transits": n_transits,
    }


# ---------------------------------------------------------------- one star

def process_star(all_time, all_flux, tce_rows):
    """tce_rows: list of dicts with tce_period, tce_time0bk, tce_duration (days).

    Returns a list of dicts (one per TCE) with views and features.
    """
    tces = [(r["tce_period"], r["tce_time0bk"], r["tce_duration"]) for r in tce_rows]
    t_a, f_a = detrend_astronet(all_time, all_flux)
    t_b, f_b = detrend_biweight(all_time, all_flux, tces)
    out = []
    for r, (period, t0, duration) in zip(tce_rows, tces):
        rec = dict(r)
        rec["global_astronet"], rec["local_astronet"], rec["local_cover_astronet"] = \
            views_for_tce(t_a, f_a, period, t0, duration)
        if t_b.size:
            rec["global_biweight"], rec["local_biweight"], rec["local_cover_biweight"] = \
                views_for_tce(t_b, f_b, period, t0, duration)
            rec.update(features_for_tce(t_b, f_b, period, t0, duration))
        out.append(rec)
    return out
