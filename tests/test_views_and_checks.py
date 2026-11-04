import numpy as np
from transitvet import tv_kepler as tv, pipeline as P
from conftest import transit_lc


def _rec(**kw):
    t, f = transit_lc(**kw)
    row = dict(tce_period=kw.get('period', 3.3), tce_time0bk=kw.get('t0', 1.0), tce_duration=kw.get('duration', 0.12))
    return tv.process_star([t], [f], [row])[0]


def test_views_have_astronet_shapes():
    r = _rec()
    assert r['global_astronet'].shape == (2001,) and r['local_astronet'].shape == (201,)
    assert np.isfinite(r['global_astronet']).all() and np.isfinite(r['local_astronet']).all()
    assert r['local_astronet'].min() < -0.5          # the dip is visible in the zoomed view


def test_odd_even_difference_is_detected():
    assert _rec(odd_even=0.0)['odd_even_sigma'] < 3
    assert _rec(odd_even=0.4, depth=0.005)['odd_even_sigma'] > 3


def test_secondary_check():
    assert _rec(secondary=0.003, depth=0.01, period=2.7)['secondary_sigma'] > 5


def test_verdict_rules():
    assert P.verdict(0.9, [0.8, 0.9, 0.7]) == 'candidate'
    assert P.verdict(0.1, [0.2, 0.1, 0.3]) == 'false alarm'
    assert P.verdict(0.5, [0.9, 0.9, 0.9]) == 'uncertain'          # middle band
    assert P.verdict(0.9, [0.1, 0.2, 0.3]) == 'uncertain'          # all single models disagree


def test_triage_calls():
    conf = dict(alphas={0.05: dict(q1=0.6, q0=0.3)})
    assert P.triage(0.9, conf)['call'] == 'keep'       # 1-p=0.1<=0.6 planet possible; p=0.9>0.3 false alarm ruled out
    assert P.triage(0.1, conf)['call'] == 'drop'       # 1-p=0.9>0.6 planet ruled out
    assert P.triage(0.35, conf)['call'] == 'needs a human'


def test_pixel_warning_rule():
    lim = P.PIXEL_WARN['tess']
    assert P.pixel_warning(dict(mission_='tess', px_sigma=lim[0] + 1, px_offset_arcsec=lim[1] + 5))
    assert not P.pixel_warning(dict(mission_='tess', px_sigma=lim[0] + 1, px_offset_arcsec=lim[1] - 5))
    assert not P.pixel_warning(dict(mission_='tess', px_sigma=float('nan'), px_offset_arcsec=99.0))
    assert not P.pixel_warning(dict(mission_='kepler', px_sigma=40.0, px_offset_arcsec=99.0))   # not validated on Kepler
