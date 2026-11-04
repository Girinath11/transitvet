import numpy as np
from transitvet import fit, planner as PL, tv_centroid as C
from conftest import transit_lc


def test_transit_fit_recovers_size_ratio():
    import batman
    t = np.arange(0, 60, 30 / 1440.0)
    pm = batman.TransitParams(); pm.t0, pm.per, pm.rp, pm.a, pm.inc = 1.0, 4.0, 0.1, 12.0, 89.0
    pm.ecc, pm.w, pm.u, pm.limb_dark = 0, 90, [0.4, 0.25], 'quadratic'
    f = batman.TransitModel(pm, t, supersample_factor=7, exp_time=0.0208).light_curve(pm)
    f = f + np.random.default_rng(0).normal(0, 2e-4, t.size)
    r = fit.fit_transit(t, f, 4.0, 1.0, 0.13, 10000, 1.0, 4.44, 'kepler')
    assert r['fit_ok'] and abs(r['rp_rs'] - 0.1) < 0.01 and abs(r['period'] - 4.0) < 1e-3
    assert not r['grazing']


def test_planner_noise_model_behaves():
    a = PL.expected_snr(5000, 3, 12, 20); b = PL.expected_snr(5000, 3, 12, 40); c = PL.expected_snr(5000, 3, 15, 20)
    assert b > a > c > 0
    assert PL.rating(8) == 'clear' and PL.rating(5) == 'marginal' and PL.rating(1) == 'too faint'


def test_never_visible_far_south_star_from_india():
    assert PL.never_visible(-79.8, 10.94)          # pi Mensae from Coimbatore
    assert not PL.never_visible(-32.8, 10.94)


def test_planner_lists_dips_one_period_apart():
    from astropy.time import Time
    ev = PL.plan(73.36, -32.80, 3.27622, 2460000.5, 0.12, 0.0, 11600, 14.7, 10.94, 76.74, 450, 20, nights=14,
                 now=Time('2026-11-04T00:00:00'))
    mids = [Time(e['mid_utc']).jd for e in ev if 'mid_utc' in e]
    assert len(mids) >= 3
    assert np.allclose(np.diff(mids) / 3.27622, np.round(np.diff(mids) / 3.27622), atol=0.01)


def _cen(shift):
    t, f = transit_lc(period=3.3, depth=0.01, noise=2e-4)
    rng = np.random.default_rng(1)
    cx = 100 + rng.normal(0, 1e-3, t.size); cy = 50 + rng.normal(0, 1e-3, t.size)
    ph = (t - 1.0 + 1.65) % 3.3 - 1.65
    cx[np.abs(ph) < 0.06] += shift
    return C.combine(C.segment_shifts([(t, f, cx, cy)], 3.3, 1.0, 0.12), 10000, 'kepler')


def test_centroid_shift_detected_only_when_present():
    assert _cen(0.0)[0] < 3
    assert _cen(0.01)[0] > 10
