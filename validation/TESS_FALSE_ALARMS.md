# Trying to catch more TESS false alarms (5 Nov 2026)

TESS false alarms are the weakest part of TransitVet. On the 1,000 TESS test signals (542 planets, 458 false alarms),
the combined score at 0.5 keeps 493/542 planets and calls 376/458 false alarms correctly; PR-AUC 0.946.
The models already use odd/even depth difference and secondary-eclipse depth, so two other ideas were tested.

Rules: choices made on TESS **validation** stars (259 signals) only; TESS **test** stars (1,000 signals) looked at once.
Scripts: `improvements/joint.py`, `improvements/pix_comb.py`. Raw numbers: `improvements/joint.json`, `improvements/pix_comb.json`.

## Test 1: train the TESS gradient-boosting model on Kepler + TESS training stars

The check numbers (depth, size, odd/even, crowding...) mean roughly the same thing on both telescopes, so more
training examples could help. Kepler training signals were added, with a column saying which telescope, and
TESS signals weighted 1x, 3x or 10x.

| GB trained on | TESS val PR-AUC | TESS test PR-AUC |
|---|---|---|
| TESS only (current) | **0.942** | 0.940 |
| Kepler + TESS | 0.928 | 0.940 |
| Kepler + TESS, TESS weighted 3x | 0.926 | 0.944 |
| Kepler + TESS, TESS weighted 10x | 0.937 | 0.943 |

Result: no gain on validation, so **not adopted**. (Test is slightly higher for two variants, but choosing by test would be unfair.)

## Test 2: add the pixel check to the final score

The pixel (difference-image) check finds which star the dip comes from. It was run on the validation and test stars
(see `PIXEL_CHECK.md`) and gave a result for 70% of test signals. The final combiner is already fitted on validation
stars, so it was refitted with the pixel numbers added (whether a result exists, its sigma, its offset, and the warning flag).

| Score | TESS test PR-AUC | Planets kept | False alarms caught |
|---|---|---|---|
| Current combined score | 0.946 | 493/542 | 376/458 |
| + pixel check | 0.950 | 493/542 (same) | 383/458 |

Difference in PR-AUC +0.004, 95% bootstrap interval over stars **−0.001 to +0.009**: the gain is small and not
certain. Kept as a finding; the dashboard keeps showing the pixel check as a separate warning instead of folding it into
the score, because the pixel check cannot run on every signal or on uploaded files.

## What this says

The TESS false-alarm problem is not fixed by more of the same features or more data from Kepler. A likely reason (not checked
signal by signal) is that the remaining false alarms look like planets in the light curve itself. Information from outside the light curve
(pixel images, Gaia star data, follow-up observations) is what separates them, which is also how NASA's own vetting works.
