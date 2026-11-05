"""Golden test: a real TESS star (TIC 1333933, known planet TOI-7719.01), data stored in tests/data,
so the full pipeline runs offline. Guards against silent changes in the results."""
import json
import numpy as np
from transitvet import pipeline as P


def test_known_planet_is_found_and_kept(scorer, tic1333933):
    segs, meta = tic1333933
    res = P.run(segs, dict(meta), scorer, progress=lambda m: None)
    s = res['signals'][0]
    assert abs(s['period'] - 3.2762) < 0.002
    assert s['verdict'] == 'candidate'
    assert s['scores']['combined'] > 0.9
    assert s['triage']['call'] == 'keep'
    assert s['known']['name'].startswith('TOI-7719')
    assert s['fit'] and s['fit']['fit_ok'] and not s['fit']['grazing']
    tab = P.summary_table(res)
    assert len(tab) == len(res['signals']) and {'verdict', 'trust_check_95', 'p_combined'} <= set(tab.columns)
    json.dumps(P.manifest(res, scorer), default=float)
    card = P.report_card(s)
    assert card[0][0].startswith('Trust check')


def test_synthetic_star_runs_offline(scorer):
    from conftest import transit_lc
    t, f = transit_lc(days=60)
    res = P.run([(t, f)], dict(target='synthetic', mission='kepler', _star={}), scorer, progress=lambda m: None)
    assert res['signals'] and abs(res['signals'][0]['period'] - 3.3) < 0.01


def test_live_view_events_do_not_change_results(scorer, tic1333933):
    """The live view gets every stage in order, and watching it does not change the answer."""
    segs, meta = tic1333933
    seen = []
    res = P.run(segs, dict(meta), scorer, progress=lambda m: None, stage=lambda name, **d: seen.append(name))
    order = [n for i, n in enumerate(seen) if i == 0 or seen[i - 1] != n]
    for a, b in [('flatten', 'search'), ('search', 'peak'), ('peak', 'fold'), ('fold', 'check'), ('check', 'verdict')]:
        assert a in order and b in order and order.index(a) < order.index(b)
    assert order[-1] == 'verdict'
    base = P.run(segs, dict(meta), scorer, progress=lambda m: None)
    assert [round(s['period'], 6) for s in res['signals']] == [round(s['period'], 6) for s in base['signals']]
    assert [s['verdict'] for s in res['signals']] == [s['verdict'] for s in base['signals']]


def test_live_view_draws_offline(scorer, tic1333933):
    """Every live-view handler runs on real data with a stand-in for Streamlit."""
    from transitvet import live as LV

    class _Slot:
        def markdown(self, *a, **k): pass
        def pyplot(self, *a, **k): pass

    class _St:
        def empty(self): return _Slot()

    segs, meta = tic1333933
    v = LV.LiveView(_St(), meta['target'])
    v.loaded(segs, meta)
    res = P.run(segs, dict(meta), scorer, progress=lambda m: None, stage=v)
    assert v.active == len(LV.STAGES) - 1 and res['signals']


def test_saved_run_replays_and_share_card(scorer, tic1333933, monkeypatch):
    """A recorded run replays into the live view and gives back the same result; the share image is a PNG."""
    from transitvet import live as LV, examples as EX, plots

    class _Slot:
        def markdown(self, *a, **k): pass
        def pyplot(self, *a, **k): pass

    class _St:
        def empty(self): return _Slot()

    segs, meta = tic1333933
    rec = EX.Recorder(); rec.loaded(segs, meta)
    res = P.run(segs, dict(meta), scorer, progress=lambda m: None, stage=rec)
    monkeypatch.setattr(EX.time, 'sleep', lambda s: None)
    back = EX.replay(LV.LiveView(_St(), meta['target']), dict(events=rec.events, res=res), seconds=0.1)
    assert back is res and [e[1] for e in rec.events][0] == 'loaded'
    r = res['signals'][0]
    png = plots.share_card(meta['target'], meta['mission'], r, 1, P.report_card(r))
    assert png[:8] == b'\x89PNG\r\n\x1a\n' and len(png) > 20000
