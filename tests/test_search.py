import numpy as np
from transitvet import pipeline as P
from conftest import transit_lc


def _search(t, f):
    tt, ff = P.flatten_for_search([(t, f)])
    return P.bls_search(tt, ff)[0]


def test_finds_injected_planet():
    t, f = transit_lc(period=3.3, depth=0.002, noise=3e-4)
    sigs = _search(t, f)
    assert sigs, 'no signal found'
    s = sigs[0]
    assert abs(s['period'] - 3.3) / 3.3 < 0.003
    assert 1500 < s['depth_ppm'] < 2500
    assert s['bls_snr'] >= P.MIN_SNR


def test_finds_two_planets():
    t, f = transit_lc(period=3.3, depth=0.002, noise=3e-4)
    _, f2 = transit_lc(period=7.9, t0=2.2, depth=0.0015, noise=0, seed=5)
    sigs = _search(t, f + f2 - 1)
    periods = sorted(s['period'] for s in sigs)
    assert any(abs(p - 3.3) / 3.3 < 0.003 for p in periods)
    assert any(abs(p - 7.9) / 7.9 < 0.003 for p in periods)


def test_pure_noise_gives_no_strong_signal():
    t, f = transit_lc(depth=0.0, noise=3e-4, seed=3)
    sigs = _search(t, f)
    assert len(sigs) == 0


def test_secondary_eclipse_is_reported():
    t, f = transit_lc(period=2.7, depth=0.01, secondary=0.004, noise=3e-4)
    sigs = _search(t, f)
    assert sigs and abs(sigs[0]['period'] - 2.7) / 2.7 < 0.003
    assert 'second_dip_phase' in sigs[0] and abs(sigs[0]['second_dip_phase'] - 0.5) < 0.05


def test_half_period_is_doubled_when_every_second_dip_is_empty():
    rng = np.random.default_rng(3)
    t = np.arange(0, 1400, 0.0204)
    f = 1 + rng.normal(0, 3e-4, t.size)
    sd = 3e-4
    P_true, dur = 330.0, 0.6
    f[np.abs((t - 50 + 0.5 * P_true) % P_true - 0.5 * P_true) < 0.5 * dur] -= 0.003
    sig = dict(period=P_true / 2, t0=50.0, duration=dur, depth_ppm=1500.0, bls_snr=50.0)
    hp = P._half_period(t, f, sd, sig)
    assert hp is not None and abs(hp[0] - P_true) < 1e-6 and abs(hp[1] - 50.0) < 1e-6
    assert 2500 < hp[3] < 3500


def test_eclipsing_binary_with_two_dips_is_not_doubled():
    rng = np.random.default_rng(4)
    t = np.arange(0, 400, 0.0204)
    f = 1 + rng.normal(0, 3e-4, t.size)
    P_bin, dur = 10.0, 0.3
    ph = (t - 2) % P_bin
    f[np.abs(ph) < 0.5 * dur] -= 0.004                     # primary eclipse
    f[np.abs(ph - 0.5 * P_bin) < 0.5 * dur] -= 0.002       # shallower secondary eclipse
    sig = dict(period=P_bin / 2, t0=2.0, duration=dur, depth_ppm=3000.0, bls_snr=80.0)
    assert P._half_period(t, f, 3e-4, sig) is None
