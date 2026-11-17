import json
import numpy as np
from transitvet import pipeline as P, tv_kepler as tv
from conftest import transit_lc


def _recs():
    t, f = transit_lc()
    r = tv.process_star([t], [f], [dict(tce_period=3.3, tce_time0bk=1.0, tce_duration=0.12)])[0]
    r.update(cen_sigma=0.5, cen_offset_arcsec=2.0, contratio=0.01, rp_est_re=2.0, star_rad=1.0)
    return [r]


def test_models_load_and_scores_are_probabilities(scorer):
    for mission in P.MISSIONS:
        out = scorer.score(_recs(), mission)
        for k in ['gb', 'combined'] + scorer.net_kinds:
            assert 0.0 <= float(out[k][0]) <= 1.0, (mission, k)


def test_scores_are_deterministic(scorer):
    a = scorer.score(_recs(), 'kepler'); b = scorer.score(_recs(), 'kepler')
    for k in a:
        assert np.allclose(a[k], b[k])


def test_conformal_thresholds_present(scorer):
    for m in P.MISSIONS:
        assert set(scorer.conf[m]['alphas']) >= {0.05, 0.1, 0.2}


def test_model_manifest_lists_features(scorer):
    assert len(scorer.feats) == len(scorer.feature_row(_recs()[0]))
    json.dumps(scorer.manifest)


def test_tool_uses_mamba_and_transformer(scorer):
    """Model set v3: Mamba-2 + Transformer are the tool's networks; the CNN is only the paper's baseline."""
    assert scorer.net_kinds == ['mamba', 'tf'] and scorer.combine_kinds == ['gb', 'mamba', 'tf']


def test_explanations_add_up_to_the_scores(scorer):
    """The live view's model panel: patch effects for both networks, and the combined score as a sum of weighted log-odds."""
    for mission in P.MISSIONS:
        recs = _recs()
        out = scorer.score(recs, mission)
        e = scorer.explain(recs, mission)[0]
        assert set(e['nets']) == set(scorer.net_kinds)
        for v in e['nets'].values():
            assert len(v['importance']) == e['n_global'] + e['n_local'] == 89
        c = e['combine']
        z = c['intercept'] + sum(w * l for w, l in zip(c['weights'], c['logodds']))
        assert abs(1 / (1 + np.exp(-z)) - float(out['combined'][0])) < 1e-6
        for k in ['gb'] + scorer.net_kinds:
            assert abs(e['calibrated'][k] - float(out[k][0])) < 1e-6
        assert len(e['gb']['inputs']) == len(scorer.feats_for(mission))


def test_animation_pages_build(scorer):
    from transitvet import anim
    t, f = transit_lc()
    segs = [(t[:2000], f[:2000] * 1.01), (t[2000:], f[2000:])]
    html = anim.page('load', 'STEP 1', anim.load_data(segs, dict(mission='tess')), 300)
    assert '<canvas id="cv">' in html and 'window.D=' in html and '</script>' in html
    sig = dict(period=3.3, t0=1.0, duration=0.12, depth_ppm=2000.0, bls_snr=40.0)
    assert anim.transits(t, f, sig) and anim.half_orbit(t, f, sig)['x']
