# Trust guarantee and follow-up planner

## 1. Trust guarantee (conformal prediction)

**What it does.** Besides the score, every signal gets one of three calls: **keep** (a false alarm is ruled out), **drop** (a planet is ruled out) or **needs a human** (the data cannot rule either out, at the chosen level).

**The guarantee.** With label-conditional ("Mondrian") split conformal prediction, for new signals drawn like the calibration signals: at least 95% of real planets keep "planet" as a possible answer (so they are never dropped), and at least 95% of real false alarms keep "false alarm" as a possible answer (so they are never kept) (at the 95% level; also shown at 90% and 80%). This is a finite-sample guarantee that needs no assumption about the model, only that calibration and new signals come from the same population.

**How it was set up, without leaking.** Validation stars were split in two halves by star: half A fits a combiner (the same three models, GB + AstroNet CNN + Transformer), half B (calibration) sets the two thresholds. Test stars were never touched. Kepler: 1,157 calibration signals; TESS: 129.

| mission | level | coverage: real planets (planet still possible) | coverage: real false alarms (false alarm still possible) | keep | drop | needs a human | planets wrongly dropped |
|---|---|---|---|---|---|---|---|
| KEPLER (2373 test signals) | 95% | 95.8% | 95.7% | 559 | 1792 | 22 | 17 |
| KEPLER (2373 test signals) | 90% | 90.8% | 91.1% | 506 | 1695 | 172 | 6 |
| KEPLER (2373 test signals) | 80% | 82.5% | 81.1% | 443 | 1504 | 426 | 1 |
| TESS (1000 test signals) | 95% | 97.8% | 98.0% | 284 | 268 | 448 | 12 |
| TESS (1000 test signals) | 90% | 94.3% | 95.2% | 418 | 355 | 227 | 31 |
| TESS (1000 test signals) | 80% | 82.3% | 84.3% | 489 | 448 | 63 | 62 |

**What this shows.**
- The guarantee holds on held-out test stars in every row: the measured coverage (columns 3 and 4) is at or above the promised level (95%, 90%, 80%), up to sampling noise.
- On Kepler the classes are well separated, so at 95% almost every signal gets a firm call (only 22 of 2,373 need a human).
- On TESS the same 95% promise sends 448 of 1,000 signals to a human. The tool is telling the truth about TESS being harder (fewer labels, Kepler-to-TESS shift), instead of giving a confident score it cannot back up. At 90%, 227 need a human.
- The guarantee is per class (how many real planets are lost), not "x% of keep calls are planets". The table shows both counts so readers can see the trade-off.
- Limits: the guarantee assumes new stars resemble the calibration stars; for TESS the calibration set is small (129 signals), so the thresholds are noisier. Prior work: a June 2026 preprint (EXOVEIL, arXiv 2606.02778) applies conformal prediction to transit detection; we do not claim to be first [CHECK before the paper].

**In the tool.** Column "trust_check_95" in the results table and a "Trust check (95% guarantee)" line in each report card. The level can be changed with the environment variable TV_CONFORMAL_ALPHA (0.05, 0.10, 0.20).

## 2. Follow-up planner

For every candidate or uncertain signal, the dashboard lists the next dips (over a chosen number of nights) that a telescope at the user's site can observe: the whole dip plus timing uncertainty and 30 minutes of baseline on each side must fall in astronomical night (Sun below -12°) with the star above 30° altitude. It also estimates the signal-to-noise of one dip for the user's telescope from a simple noise budget: photon noise (V-band zero point, 30% system efficiency), atmospheric scintillation (Young's formula) and a 500 ppm ground-systematics floor. SNR ≥ 7 is "clear", 4-7 "marginal", below 4 "too faint".

Default site: Karunya campus, Coimbatore (10.94° N, 76.74° E, 450 m), 20 cm telescope, IST. All settings are in the sidebar. Batch mode builds one observing plan across all analysed stars.

Sanity checks:
- pi Mensae c (V = 5.7, 271 ppm): reported as never above 30° from Coimbatore (declination -80°), and its dip would give SNR 0.4 on a 20 cm telescope, so it is a space-telescope-only planet. Both match what is known about it.
- TOI-7719.01 on TIC 1333933 (V = 14.7, 1.2% deep): one observable dip in the next 14 nights (6 Nov 2026, 00:38-04:00 IST, star at 40° or higher), SNR 6.5, "marginal" for 20 cm.
- These are planning estimates, not guarantees: weather, the moon, the comparison stars and the camera also matter. The noise model has not been checked against real observations from a college telescope [CHECK with a real observation if possible].

Code: conf/conformal.py (guarantee), app/transitvet/planner.py (planner).
