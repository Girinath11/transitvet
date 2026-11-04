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
        for k in ('gb', 'cnn', 'tf', 'combined'):
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
