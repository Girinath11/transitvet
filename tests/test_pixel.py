import numpy as np
from transitvet import tv_pixel as X


def _cube(dip_on_neighbour, n=4000, period=2.0, t0=0.5, duration=0.15):
    """11x11 pixel movie: bright target at (5,5), fainter neighbour at (8,3). One of them dims during the dips."""
    rng = np.random.default_rng(0)
    t = np.arange(n) * (10 / 1440.0)
    yy, xx = np.mgrid[0:11, 0:11]
    star = lambda x0, y0, a: a * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * 0.9 ** 2))
    ph = (t - t0 + 0.5 * period) % period - 0.5 * period
    dip = np.abs(ph) < duration / 2
    cube = np.empty((n, 11, 11))
    for i in range(n):
        a_t, a_n = 1000.0, 300.0
        if dip[i]:
            if dip_on_neighbour:
                a_n *= 0.8
            else:
                a_t *= 0.99
        cube[i] = star(5, 5, a_t) + star(8, 3, a_n) + 20 + rng.normal(0, 1.0, (11, 11))
    return t, cube, period, t0, duration


def test_gaussian_fit_finds_star():
    yy, xx = np.mgrid[0:11, 0:11]
    img = 500 * np.exp(-((xx - 6.3) ** 2 + (yy - 4.2) ** 2) / 2) + 10
    x, y, ex, ey = X.fit_star(img)
    assert abs(x - 6.3) < 0.05 and abs(y - 4.2) < 0.05


def test_difference_image_points_at_the_star_that_dims():
    for on_neighbour, expect in ((False, (5, 5)), (True, (8, 3))):
        t, cube, p, t0, d = _cube(on_neighbour)
        D, E, O, n = X.difference_image(t, cube, p, t0, d)
        assert n >= 5
        x, y, ex, ey = X.fit_star(D, E)
        assert abs(x - expect[0]) < 0.3 and abs(y - expect[1]) < 0.3, (on_neighbour, x, y)


def test_combine_flags_consistent_offsets_only():
    segs = [dict(used=True, east=30.0, north=-5.0, err=5.0, oot_east=0, oot_north=0) for _ in range(3)]
    off, sig, n, _ = X.combine(segs)
    assert n == 3 and abs(off - np.hypot(30, 5)) < 1 and sig > 5
    segs = [dict(used=True, east=e, north=0.0, err=5.0, oot_east=0, oot_north=0) for e in (3, -2, 1)]
    off, sig, n, _ = X.combine(segs)
    assert sig < 3
