"""Chance that TransitVet would catch a planet of a given size and orbit around THIS star.

Built from the injection-recovery test (validation/INJECTION.md): fake planets were added to the raw light curves of
held-out test stars and the whole pipeline was run unchanged. The fraction found, and the fraction found AND called
candidate, were fitted as a logistic curve in log10(expected SNR), where

    expected SNR = depth / (robust scatter of the flattened curve) * sqrt(points in transit).

For a new star we compute that expected SNR from the star's own scatter and number of points, then read the curve.
This is an estimate for planning, not a guarantee: it assumes a circular orbit, impact parameter 0.4 and even
coverage, and it is limited to periods the search covers (at most a third of the data span, and 500 d)."""
import json, os, numpy as np

_HERE = os.path.dirname(__file__)
CURVE_FILE = os.path.join(_HERE, 'sensitivity.json')
SIZES = (1.0, 2.0, 4.0)          # Earth radii
PERIODS = (3.0, 10.0, 30.0)      # days


def load_curve(path=CURVE_FILE):
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return None


def _logistic(x, a, b):
    return 1.0 / (1.0 + np.exp(-(a + b * x)))


def prob(exp_snr, curve, kind='candidate'):
    """Probability from the fitted curve; kind is 'found' or 'candidate'."""
    if curve is None or not np.isfinite(exp_snr) or exp_snr <= 0:
        return float('nan')
    a, b = curve['fit'][kind]
    return float(_logistic(np.log10(exp_snr), a, b))


def expected_snr(rp_re, period, sd, n_points, rstar=1.0, b=0.4):
    """Expected SNR of a planet of rp_re Earth radii and the given period (days) on a curve with robust scatter sd
    (relative flux) and n_points points in all."""
    rs = rstar if (rstar is not None and np.isfinite(rstar) and 0.1 < rstar < 5) else 1.0
    ms = rs if rs < 1.5 else 1.3                       # rough main-sequence mass, as in the injection test
    a_rs = (ms * (period / 365.25) ** 2) ** (1 / 3) * 215.03 / rs
    k = rp_re / 109.1 / rs
    arg = np.sqrt(max((1 + k) ** 2 - b ** 2, 0)) / a_rs
    dur = period / np.pi * np.arcsin(min(1.0, arg))
    n_in = n_points * dur / period                     # points in transit, summed over all transits (even coverage)
    return float(k ** 2 / sd * np.sqrt(max(n_in, 0)))


def star_table(res, curve=None, sizes=SIZES, periods=PERIODS):
    """Rows of {size, period, exp_snr, p_found, p_candidate} for a pipeline result `res`, or [] if not possible."""
    curve = curve if curve is not None else load_curve()
    sc = res.get('search_curve')
    if curve is None or sc is None:
        return []
    t, f = sc
    t, f = np.asarray(t), np.asarray(f)
    if t.size < 50:
        return []
    sd = 1.4826 * np.median(np.abs(f - np.median(f)))
    span = float(t.max() - t.min())
    rstar = (res.get('star') or {}).get('star_rad', np.nan)
    pmax = min(span / 3, 500)
    rows = []
    for rp in sizes:
        for P in periods:
            if P > pmax:
                rows.append(dict(size=rp, period=P, exp_snr=float('nan'), p_found=0.0, p_candidate=0.0, note='orbit too long for this data'))
                continue
            s = expected_snr(rp, P, sd, t.size, rstar)
            rows.append(dict(size=rp, period=P, exp_snr=s, p_found=prob(s, curve, 'found'), p_candidate=prob(s, curve, 'candidate'), note=''))
    return rows
