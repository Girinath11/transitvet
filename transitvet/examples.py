"""Saved runs of the dashboard's example stars.

A saved run is a real, complete analysis of that star (same code, same models), recorded once with
`python -m transitvet.examples` and stored in transitvet/examples/. The dashboard replays its live-view
steps in about 15-20 seconds and shows the stored result, so visitors don't wait minutes or load the
hosted server. Any other star, or "Run it live" on an example, does the full analysis as usual.

A saved run is used only if it was made with the same models (checked through the models' manifest)."""
import datetime as _dt
import gzip
import hashlib
import os
import pickle
import time

import numpy as np

_HERE = os.path.join(os.path.dirname(__file__), 'examples')
_MANIFEST = os.path.join(os.path.dirname(__file__), 'models', 'manifest.json')
REPLAY_SECONDS = 16.0
_MAX_SEARCH_POINTS = 6000      # the periodogram is thinned to this many points per stored snapshot


def _models_id():
    try:
        with open(_MANIFEST, 'rb') as fh:
            return hashlib.sha256(fh.read()).hexdigest()[:16]
    except OSError:
        return ''


def _path(target):
    return os.path.join(_HERE, ''.join(c if c.isalnum() else '_' for c in target.strip().upper()) + '.pkl.gz')


class Recorder:
    """Stands in for the live view during a recording: keeps every stage event with its time."""

    def __init__(self):
        self.t0, self.events = time.time(), []

    def _keep(self, name, d):
        self.events.append((time.time() - self.t0, name, d))

    def loaded(self, segs, meta):
        self._keep('loaded', dict(segs=segs, meta={k: v for k, v in meta.items() if not k.startswith('_')}))

    def __call__(self, name, **d):
        if name == 'search':
            parts, n = d['parts'], d['n']
            done = sum(len(p[0]) for p in parts)
            per = np.concatenate([p[0] for p in parts]); pw = np.concatenate([p[1] for p in parts])
            st = max(1, per.size // _MAX_SEARCH_POINTS)
            d = dict(k=d['k'], parts=[(per[::st], pw[::st])], n=n, done=done)
        self._keep(name, d)


def record(target, mission=None):
    from . import pipeline as P
    os.environ.setdefault('TV_LAZY_MODELS', '1')
    rec = Recorder()
    segs, meta = P.load_star(target, mission)
    meta['_pixel'] = True
    rec.loaded(segs, meta)
    res = P.run(segs, meta, P.Scorer(), print, stage=rec)
    os.makedirs(_HERE, exist_ok=True)
    blob = dict(target=target, made=_dt.date.today().isoformat(), models=_models_id(), seconds=res['seconds'],
                events=rec.events, res=res)
    with gzip.open(_path(target), 'wb', compresslevel=9) as fh:
        pickle.dump(blob, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return _path(target)


def saved(target):
    """The saved run for this star, or None (missing, unreadable, or made with other models)."""
    p = _path(target)
    if not os.path.exists(p):
        return None
    try:
        with gzip.open(p, 'rb') as fh:
            blob = pickle.load(fh)      # files we made ourselves, shipped inside the package
    except Exception:
        return None
    return blob if blob.get('models') == _models_id() else None


def replay(view, blob, seconds=REPLAY_SECONDS):
    """Plays the stored stage events into a LiveView, squeezed into about `seconds`."""
    ev = blob['events']
    total = max(ev[-1][0], 1e-6)
    prev = 0.0
    for t, name, d in ev:
        time.sleep(min(2.0, max(0.12, (t - prev) / total * seconds)))
        prev = t
        view.replay_clock = t
        if name == 'loaded':
            view.loaded(d['segs'], d['meta'])
        else:
            view(name, **d)
    return blob['res']


if __name__ == '__main__':
    import sys
    targets = sys.argv[1:] or ['TIC 1333933', 'TIC 184952758', 'TIC 261136679', 'KIC 11442793']
    for tg in targets:
        t = time.time()
        print(tg, '->', record(tg), f'{os.path.getsize(_path(tg)) / 1e6:.1f} MB', f'{time.time() - t:.0f} s', flush=True)
