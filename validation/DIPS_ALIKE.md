# "Single dips look alike" rule (17 Nov 2026)

**Problem.** Searching up to 5 signals per star and orbits up to 500 days sometimes finds long-period "signals" built from one or two odd events. On Pi Mensae (7 years of TESS data), for example, the three single dips of a 288.8-day signal measured 261, 632 and 135 ppm.

**Rule.** Each single dip is measured on its own. Its depth is the mean of the points in the middle 80% of the dip, compared with the median of points 0.75–3 durations away in the same orbit (`pipeline.dip_depths`). A dip "looks like the typical one" when it is seen at 3 sigma or more and is within 50% of the median single-dip depth. Signals with 6 dips or fewer need at least 3 such dips, or a "candidate" becomes "uncertain". The rule never makes a signal a false alarm. We fixed these thresholds before testing.

**Test.** We used held-out TEST signals with 6 dips or fewer: 106 Kepler (10 planets, 96 false alarms) and 165 TESS (73 planets, 92 false alarms). Each light curve was downloaded with the dashboard's own loader and flattening, and measured at the catalogue period. Only signals the combined score calls candidate (≥ 0.5) are affected.

| | candidates (≥ 0.5) | planets moved to uncertain | false alarms moved to uncertain |
|---|---|---|---|
| Kepler | 9 planets, 0 false alarms | 0 of 9 | – |
| TESS | 69 planets, 22 false alarms | 15 of 69 | 11 of 22 |

On TESS few-dip signals, the share of "candidate" calls that are real planets rises from 76% (69/91) to 83% (54/65). The cost is that 15 real planets are called uncertain instead of candidate. The sample is small, so treat these numbers as rough. The open-TOI ranked list was made before this rule. The rule needs the light curve, so the list does not use it.

Scripts: `junk/val_dips.py`, `junk/rules2.py`, `junk/summ2.py` (working folder).
