# TransitVet v2: what the new checks changed

All numbers are on held-out TEST stars, never used to train, calibrate or tune. Same splits as the experiments: Kepler DR24 split by star 70/15/15; TESS labelled TOIs split by star 50/10/40. Gradient boosting is fitted on training stars; the combiner (logistic regression on the three models' log-odds) and thresholds on validation stars.

"False alarms caught" and "planets kept" use the threshold that maximised F1 on validation stars.

## KEPLER (train 11053, validation 2311, test 2373 signals)

| model                        |   pr_auc |   roc_auc |   precision |   recall | false_alarms_caught   | planets_kept   |
|:-----------------------------|---------:|----------:|------------:|---------:|:----------------------|:---------------|
| gb_old                       |   0.8879 |    0.9722 |       0.82  |    0.902 | 1751/1854             | 468/519        |
| gb_new                       |   0.9474 |    0.9864 |       0.883 |    0.915 | 1791/1854             | 475/519        |
| cnn                          |   0.9467 |    0.9855 |       0.893 |    0.904 | 1798/1854             | 469/519        |
| tf                           |   0.9321 |    0.9793 |       0.848 |    0.873 | 1773/1854             | 453/519        |
| combined_old (gb_old+cnn+tf) |   0.9476 |    0.987  |       0.879 |    0.923 | 1788/1854             | 479/519        |
| combined_new (gb_new+cnn+tf) |   0.9675 |    0.9919 |       0.915 |    0.954 | 1808/1854             | 495/519        |

Test PR-AUC of the new gradient boosting when one new check is left out: cen_sigma 0.9439, cen_offset_arcsec 0.95, rp_est_re 0.9365, contratio 0.9459

## TESS (train 1243, validation 259, test 1000 signals)

| model                        |   pr_auc |   roc_auc |   precision |   recall | false_alarms_caught   | planets_kept   |
|:-----------------------------|---------:|----------:|------------:|---------:|:----------------------|:---------------|
| gb_old                       |   0.8825 |    0.872  |       0.798 |    0.838 | 343/458               | 454/542        |
| gb_new                       |   0.9403 |    0.9327 |       0.756 |    0.97  | 288/458               | 526/542        |
| cnn                          |   0.8652 |    0.8532 |       0.741 |    0.865 | 294/458               | 469/542        |
| tf                           |   0.8442 |    0.8208 |       0.736 |    0.821 | 298/458               | 445/542        |
| combined_old (gb_old+cnn+tf) |   0.9018 |    0.8904 |       0.732 |    0.923 | 275/458               | 500/542        |
| combined_new (gb_new+cnn+tf) |   0.9457 |    0.9391 |       0.884 |    0.86  | 397/458               | 466/542        |

Test PR-AUC of the new gradient boosting when one new check is left out: cen_sigma 0.9397, cen_offset_arcsec 0.942, rp_est_re 0.9135, contratio 0.9308

## Notes

- gb_old / combined_old = the previous models; gb_new / combined_new = with the new checks (cen_sigma, cen_offset_arcsec = light-position check; rp_est_re = planet size from depth and star radius; contratio = light from neighbouring stars).
- On TESS, planet size and crowding carry most of the gain (TESS test PR-AUC with the old features plus only those two: 0.940; plus only the light-position check: 0.890). TESS pixels are 21 arcsec wide, so the light-position check sees less there.
- On Kepler (4 arcsec pixels) each new check adds a little; with the combiner the test PR-AUC goes from 0.947 (AstroNet CNN alone) to 0.968, and planets missed at the chosen threshold drop from 50 to 24 of 519.
- Caution: the official labels were partly decided by people looking at the same kinds of checks (centroid offsets, planet size). So part of the gain means "agrees better with the official vetting", which is what the tool is for, but it is not independent proof.
- Star radii are from the TESS Input Catalog v8.2 (MAST), available for 93% of Kepler and 95% of TESS signals; gradient boosting handles missing values.
- Code: imp/run_cen.py, imp/fetch_tic.py, imp/eval_new.py, build_models2.py; feature code in transitvet/tv_centroid.py, extras.py, fit.py.
- Light-position numbers: the significance (cen_sigma) is inflated (median 7 sigma even for Kepler planet candidates, because of correlated noise our error estimate misses). The models learn its real scale, but for the user warning we use thresholds set on training signals: Kepler shift > 10 sigma and source > 30 arcsec away (flags 0.8% of planet candidates, 9% of false alarms). On TESS no threshold separated planets from false alarms, so the dashboard shows no light-position warning for TESS.
- Comparison with NASA's ExoMiner++: see EXOMINER_COMPARISON.md.
