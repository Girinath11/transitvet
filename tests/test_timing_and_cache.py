import numpy as np
from transitvet import pipeline as P, examples as EX


def _lc(shift_min=None, P_=4.0, t0=1.0, dur=0.15, depth=3000, n_days=60, noise=200e-6, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(0, n_days, 2 / 1440)
    f = 1 + rng.normal(0, noise, t.size)
    for e in range(int(n_days / P_) + 1):
        mid = t0 + e * P_ + (0 if shift_min is None else shift_min * np.sin(e) / 1440)
        f[np.abs(t - mid) < dur / 2] -= depth * 1e-6
    return t, f, dict(period=P_, t0=t0, duration=dur, depth_ppm=depth)


def test_regular_dips_do_not_wander():
    t, f, s = _lc()
    r = P.transit_timing(t, f, s)
    assert r['n'] >= 10 and not r['wanders'], r


def test_shifted_dips_wander():
    t, f, s = _lc(shift_min=40)
    r = P.transit_timing(t, f, s)
    assert r['wanders'], r


def test_result_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(EX, 'CACHE_DIR', str(tmp_path))
    class Rec:
        events = [(0.0, 'x', {})]
    EX.store('TIC 1', None, True, Rec(), {'seconds': 3.0, 'signals': []})
    assert EX.cached('tic 1 ', None, True)['res']['seconds'] == 3.0
    assert EX.cached('TIC 1', None, False) is None and EX.cached('TIC 1', 'tess', True) is None


def test_gaia_numbers_from_rows():
    from transitvet import gaia as G
    rows = [dict(source_id=1, ra=10.0, dec=-20.0, phot_g_mean_mag=10.0, parallax=5.0, parallax_error=0.05, astrometric_excess_noise=0.1,
                 astrometric_excess_noise_sig=1.0, astrometric_gof_al=2.0, radial_velocity_error=0.5, phot_bp_rp_excess_factor=1.2),
            dict(source_id=2, ra=10.0, dec=-20.0 + 10 / 3600, phot_g_mean_mag=12.5),      # 10" away, 10x fainter
            dict(source_id=3, ra=10.0, dec=-20.0 + 30 / 3600, phot_g_mean_mag=15.0)]      # 30" away, 100x fainter
    g = G.from_rows(10.0, -20.0, rows)
    assert g['g_n21'] == 1 and g['g_n42'] == 2 and abs(g['g_nb_dmag'] - 2.5) < 1e-9 and abs(g['g_nb_sep'] - 10) < 0.01
    assert abs(g['g_plx_snr'] - 100) < 1e-9
    # a 1% dip on the target would need about a 11% dip on the neighbour (it gives 1/10 of the target's light, plus all light in the aperture)
    assert abs(10 ** G.nb_depth(10000, g) - 0.01 * (1 + 0.1 + 0.01) / 0.1) < 1e-9
    assert G.from_rows(11.0, -20.0, rows) == {}
