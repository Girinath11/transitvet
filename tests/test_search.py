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
