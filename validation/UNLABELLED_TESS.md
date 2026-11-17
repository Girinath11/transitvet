# Open TESS signals, ranked (17 Nov 2026)

We scored all 4,408 TOIs that are still open on ExoFOP (3,964 PC, 439 APC, 5 blank). We used the dashboard's v4 models, the same features and views as in training, and the same verdict and 95%-check rules. The full table is `unlabelled_tess_scores.csv`. These are **candidates for follow-up**, not confirmations.

| verdict | 95% check: keep | drop | needs a person |
|---|---|---|---|
| candidate | 1,304 | 0 | 709 |
| false alarm | 0 | 1,028 | 575 |
| uncertain | 88 | 0 | 704 |

- "Strong candidates" (candidate + keep): 1,304, of which 398 have an estimated size under 4 Earth radii.
- APC signals (ambiguous on ExoFOP) come out as false alarms 66% of the time, and PC signals 33% of the time. That is the expected direction.
- NASA's ExoMiner++ catalogue also scores 2,502 of these signals. The two land on the same side of 0.5 for 72% of them, with a Spearman rank correlation of 0.59.
- For 214 signals our light curve did not show the dip at the catalogue ephemeris (depth ≤ 0), and these are marked. "Uncertain" also covers signals with fewer than 3 dips.
- Notes such as "light moves during the dip", "odd and even dips differ" or "dip at half orbit" are warnings only. As in the dashboard, they do not change the verdict.

Caveats [CHECK]: the TOI list and dispositions were downloaded on 1 Nov 2026, and some of these signals may have been resolved since. Our scores are only as good as the test numbers in GAIA.md, which come from labelled TOIs. Open TOIs may differ from labelled ones (fainter stars, smaller dips).

Script: `unlab/score_unlab.py` (working folder).
