# Batch run on 170 held-out test stars (4 Nov 2026)

Command: `python -m transitvet run --file stars.txt --out results --workers 4 --timeout 3600` with the pixel check on (2 TESS sectors at most).
Stars: 20 Kepler and 150 TESS test stars, picked at random (seed 7) from the held-out test split. None of them was used in the 50-star test. Each star was downloaded, searched and scored from scratch.

## Reliability
- 170 of 170 stars finished, with 0 errors. In total the run produced 359 signals.
- The machine restarted when 84 stars were done. Running the same command again skipped the finished stars and completed the other 86 (4,305 s for the second part). This tested the resume feature for real.
- Time per star on one core, including downloads:
  - Kepler: median 363 s, slowest 10% over 555 s.
  - TESS: median 184 s, slowest 10% over 329 s.
- Speed with 4 workers: about 80 TESS stars per hour.

## Agreement with expert labels
A catalogue signal counts as found if a TransitVet signal has the same period (within 1%, or at half or double). Verdicts use the final rules, including the TESS pixel warning.

| | Signals | Found | Candidate | Uncertain | False alarm |
|---|---|---|---|---|---|
| Kepler planet candidates | 10 | 9 | 7 | 1 | 1 |
| Kepler false alarms | 24 | 10 | 1 | 0 | 9 |
| TESS confirmed/known planets | 84 | 81 | 64 | 14 | 3 |
| TESS false positives | 70 | 58 | 11 | 13 | 34 |

- **Planets:** of the planets that were found, 79% of TESS planets and 78% of Kepler planets were called candidate. 4 of 90 were called false alarm.
- **TESS false positives:** 19% of the ones found were still called candidate. That is about the same as the 50-star test (2 of 10), and it is still the main weak spot.
- **Pixel check:** 144 of 323 TESS signals were strong enough to measure in the pixels. The warning fired on 21 matched false positives and on 2 confirmed planets. It turned 2 false-positive candidates into "uncertain", and the 2 planets also became "uncertain".
- **Unfound signals:** Kepler false alarms that were not found are mostly not-transit-like signals, which our search (SNR ≥ 7.1, at least 3 dips) does not pick up. That is the intended behaviour.

Files: `scale_eval.json`, `scale_matches.csv`.
