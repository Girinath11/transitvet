# Gaia DR2 numbers for TESS false alarms (17 Nov 2026)

**Question.** Can Gaia data about the star and its neighbours help the TESS model catch more false alarms?

**What we added** (`transitvet/gaia.py`; queried at MAST, 60 arcsec around the star):
- The star's own astrometry: excess noise and its significance, fit quality, radial-velocity error, BP/RP colour excess, parallax signal-to-noise. A hidden companion often shows up as extra wobble or a large radial-velocity scatter.
- Neighbours: the number of Gaia stars within 21" (one TESS pixel) and 42", how much light they add, and the brightest one's magnitude difference and distance.
- How deep the brightest neighbour's own dip would have to be to fake the signal (log10).

Gaia DR2 is used because MAST serves DR2. It has no RUWE column, so excess noise stands in for it. The table was downloaded for all 7,826 TOI host stars, and 7,815 matched a Gaia star: the brightest within 3", or, for fast-moving stars whose catalogue position is out of date, the brightest within 10". The dashboard first moves the TIC position to Gaia's epoch using the proper motion.

**Method.** We used the same TESS star split as before: 1,243 training, 259 validation and 1,000 test signals. The gradient-boosting model (5 seeds) was trained on the training stars, with the old inputs plus the 13 Gaia numbers. The combiner, calibration and conformal cut-offs were refitted on the validation stars. We chose on validation and looked at the test stars at the end.

| PR-AUC | validation | test |
|---|---|---|
| Gradient boosting, before | 0.942 | 0.940 |
| Gradient boosting + Gaia | 0.965 | 0.947 |
| Combined score, before (v3) | 0.946 | 0.946 |
| **Combined score + Gaia (v4)** | 0.967 | **0.950** |

The test gain of the combined score is +0.004, with a 95% bootstrap interval over stars of −0.008 to +0.016. That gain is likely but not proven.

At the tool's own rule (score ≥ 0.5), on the 1,000 test signals:

| | false alarms caught | planets kept |
|---|---|---|
| v3 | 363 / 458 | 492 / 542 |
| v4 | 382 / 458 | 493 / 542 |

With 98% of planets kept, v4 catches 291 false alarms against 261 for v3. The 95% check now sends 318 of the 1,000 test signals to a person, down from 508. Coverage is 98.0% for planets and 96.3% for false alarms, against a promise of 95%.

ExoMiner++ comparison (542 shared test signals): TransitVet 0.968 (was 0.966), ExoMiner++ 0.973. The difference is −0.005, with an interval of −0.019 to +0.010. The average of the two reaches 0.982.

**Which numbers matter.** We dropped one Gaia number at a time and measured the PR-AUC loss on validation. The radial-velocity error mattered most (0.014), followed by the neighbour-depth test and parallax S/N (0.005 each). Radial-velocity errors only exist for stars brighter than about G = 13, so about half the stars have this input missing. Gradient boosting handles missing inputs on its own.

**Side effect.** In the v4 combiner, the TESS weights are 0.99 for gradient boosting, 0.01 for Mamba-2 and 0.17 for the Transformer. On TESS the combined score now leans almost fully on gradient boosting, and Mamba-2 adds next to nothing there. Kepler is unchanged.

**Bug found on the way.** The first feature table compared Gaia source IDs as floating-point numbers. Close neighbours could then be mistaken for the star itself. The fixed table is built by the package code, so live and training numbers match, and all numbers above use it. The first run gave a similar picture (test 0.952).

Scripts: `gaia/fetch.py`, `gaia/feats.py`, `gaia/eval_gaia.py`, `gaia/eq_recall.py`, `v4/build_models4.py` (outside the package, in the working folder).
