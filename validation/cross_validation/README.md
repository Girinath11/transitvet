# 5-fold cross-validation by star (extra result)

The main paper tables use one fixed split by star (Kepler 70/15/15, TESS 50/10/40).
As an extra check, `cv.py` uses every labelled star once as a test star: 15,737 Kepler signals on 9,865 stars and 2,502 TESS signals on 2,357 stars.

- Stars are shuffled (seed 2026) and cut into 5 folds. Each fold in turn is the test set; 15/85 of the other stars are validation stars, the rest training stars.
- Kepler, per fold: gradient boosting (5 seeds), Transformer masked pretraining (training-fold views + unlabelled TESS) and fine-tuning, Mamba-2 (1 seed per neural model).
- TESS, per fold: gradient boosting with Gaia numbers (5 seeds); Mamba-2 and Transformer fine-tuned (5 seeds each) from the Kepler-trained weights, which saw no TESS star.
- Combiner and 95% check are refitted on each fold's validation stars, as in the tool.

| Mission | Signals (stars) | Combined PR-AUC (95% interval) | Per fold | GB | Mamba-2 | Transformer | Planet cov. | False cov. |
|---|---|---|---|---|---|---|---|---|
| Kepler | 15,737 (9,865) | 0.966 (0.960–0.971) | 0.966 ± 0.004 | 0.950 | 0.926 | 0.921 | 95.0% | 94.5% |
| TESS | 2,502 (2,357) | 0.957 (0.949–0.964) | 0.957 ± 0.004 | 0.951 | 0.860 | 0.846 | 96.0% | 95.2% |

All numbers are in `cv_summary.json`. Run: `python cv.py kepler K` and `python cv.py tess K` for K = 0..4, then `python cv.py summary` (`run_cv.sh` does all of it).
About 7 hours on 4 CPU cores. The script uses the working-copy paths of the project (`/home/user/tv/...`) for the data files; change `OUT`, `MD` and the data paths to run it elsewhere.
