"""Shared helpers for the TransitVet tests. Run from the app folder:  python -m pytest -q
Tests marked 'network' need internet access to NASA's MAST archive; skip them with  -m "not network"."""
import os, sys, gzip, pickle
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(os.path.dirname(__file__), 'data')


def transit_lc(period=3.3, t0=1.0, depth=0.002, duration=0.12, days=90.0, cadence_min=30.0, noise=3e-4, seed=0,
               secondary=0.0, odd_even=0.0):
    """Synthetic light curve with box-shaped dips (plus optional secondary eclipse / odd-even difference)."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, days, cadence_min / 1440.0)
    f = 1 + rng.normal(0, noise, t.size)
    ph = (t - t0 + 0.5 * period) % period - 0.5 * period
    ep = np.round((t - t0) / period).astype(int)
    d = np.where(ep % 2 == 0, depth * (1 + odd_even), depth * (1 - odd_even))
    f[np.abs(ph) < duration / 2] -= d[np.abs(ph) < duration / 2]
    if secondary:
        ph2 = (t - t0 - 0.5 * period + 0.5 * period) % period - 0.5 * period
        f[np.abs(ph2) < duration / 2] -= secondary
    return t, f


@pytest.fixture(scope='session')
def scorer():
    from transitvet import pipeline as P
    return P.Scorer()


@pytest.fixture(scope='session')
def tic1333933():
    with gzip.open(os.path.join(DATA, 'tic1333933.pkl.gz'), 'rb') as fh:
        return pickle.load(fh)
