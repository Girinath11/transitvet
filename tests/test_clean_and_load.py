import io
import numpy as np
from transitvet import pipeline as P
from conftest import transit_lc


def test_clean_drops_bad_points_and_sorts():
    t = np.array([5, 1, 2, np.nan, 4, 3] * 20, float) + np.repeat(np.arange(20) * 10, 6)
    f = np.ones_like(t); f[2] = np.nan
    out = P._clean_segment(t, f, None, 'kepler')
    assert out is not None
    tt, ff = out
    assert np.all(np.isfinite(tt)) and np.all(np.isfinite(ff))
    assert np.all(np.diff(tt) >= 0)


def test_tess_two_minute_data_is_binned_to_30_minutes():
    t = np.arange(0, 5, 2 / 1440.0)
    out = P._clean_segment(t, np.ones_like(t), np.zeros(t.size, int), 'tess')
    assert abs(np.median(np.diff(out[0])) * 1440 - 30) < 0.5


def test_tess_quality_flags_are_applied():
    t = np.arange(0, 5, 30 / 1440.0)
    q = np.zeros(t.size, int); q[:100] = 1
    out = P._clean_segment(t, np.ones_like(t), q, 'tess')
    assert out[0].size == t.size - 100


def test_too_few_points_returns_none():
    assert P._clean_segment(np.arange(10.0), np.ones(10), None, 'kepler') is None


def test_csv_upload_is_split_at_gaps():
    t1, f1 = transit_lc(days=20)
    t2, f2 = transit_lc(days=20, seed=1)
    t = np.concatenate([t1, t2 + 30]); f = np.concatenate([f1, f2])
    buf = io.StringIO(); buf.write('time,flux\n')
    for a, b in zip(t, f):
        buf.write(f'{a},{b}\n')
    segs, meta = P.load_file('star.csv', buf.getvalue().encode(), 'kepler')
    assert len(segs) == 2 and meta['mission'] == 'kepler'
