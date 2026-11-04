# TransitVet tool v2: test on held-out stars

Stars picked at random from the TEST split (never used for training, fine-tuning or calibration): 15 Kepler stars with a planet candidate (PC), 15 Kepler stars with a false alarm (AFP), 10 TESS planet stars (CP/KP/PC), 10 TESS false-positive stars (FP). For each star the tool downloads the light curve itself, runs its own dip search, and scores what it finds. Catalogue entries at the same period (within 1%) on the same star are counted once. A catalogue signal counts as found if one of the tool's periods matches it within 1% (or at half or double the period).

## KEPLER (30 stars, 0 failed to load)

| catalogue label (PC/CP/KP = planet; AFP, NTP, FP, FA = not a planet) | signals | not found by search | candidate | uncertain | false alarm |
|---|---|---|---|---|---|
| PC | 18 | 2 | 15 | 0 | 1 |
| AFP | 16 | 5 | 1 | 1 | 9 |

Time per star (4 stars at once, 1 core each): median 6.6 min, max 11.3 min.


## TESS (20 stars, 0 failed to load)

| catalogue label (PC/CP/KP = planet; AFP, NTP, FP, FA = not a planet) | signals | not found by search | candidate | uncertain | false alarm |
|---|---|---|---|---|---|
| CP | 6 | 0 | 6 | 0 | 0 |
| KP | 5 | 0 | 4 | 1 | 0 |
| FP | 10 | 1 | 2 | 3 | 4 |

Time per star (4 stars at once, 1 core each): median 2.5 min, max 8.9 min.


## v1 (before the new checks) vs v2 (now), same 50 stars

| group | v1 | v2 |
|---|---|---|
| Kepler planet candidates found by the search (16 of 18) | 13 candidate, 2 uncertain, 1 false alarm | 15 candidate, 0 uncertain, 1 false alarm |
| Kepler false alarms found by the search (11 of 16) | 9 false alarm, 2 candidate | 9 false alarm, 1 uncertain, 1 candidate |
| TESS planets (11) | 11 candidate | 10 candidate, 1 uncertain |
| TESS false positives (10) | 3 false alarm, 2 uncertain, 5 candidate | 4 false alarm, 3 uncertain, 2 candidate, 1 not found |

- The weak spot (TESS false positives wrongly called candidate) went from 5 of 10 to 2 of 10. The price: one TESS known planet (TIC 169461816, 17.86 d) moved from candidate to uncertain.
- KIC 2852528 (a Kepler false alarm) is now uncertain instead of candidate, because the search finds a second dip at the same period (eclipsing-binary sign).
- TIC 4294779 (FP, 0.344 d): v1 matched it through a double-period signal; v2 stops the search when the same period comes back, so it is now listed as not found. Periods under 0.6 days are not searched.
- Same 2 Kepler PC signals missed by the search as before (KIC 4736569 at 7.07 d, KIC 7772914 at 23.13 d); the search itself did not change.
- 50 stars is a small sample; the main evidence is the held-out test on thousands of signals in IMPROVEMENTS.md.

## Per star

| star | label | catalogue signals (label, period d, tool verdict) | other signals the tool found |
|---|---|---|---|
| KIC 10090854 | PC | PC 12.631: candidate |  |
| KIC 9847239 | PC | PC 28.508: candidate | 226.045 d false alarm (0.02); 2.331 d false alarm (0.00) |
| KIC 4736569 | PC | PC 10.132: candidate; PC 7.074: not found | 269.192 d false alarm (0.05); 184.634 d false alarm (0.05) |
| KIC 3632089 | PC | PC 31.779: candidate; PC 5.328: candidate | 258.768 d false alarm (0.08) |
| KIC 6206214 | PC | PC 13.540: candidate | 298.006 d false alarm (0.10) |
| KIC 11192235 | PC | PC 1.615: candidate | 246.036 d false alarm (0.01); 139.787 d false alarm (0.13) |
| KIC 10190777 | PC | PC 1.411: candidate | 267.476 d false alarm (0.02); 278.609 d false alarm (0.02) |
| KIC 5966322 | PC | PC 60.928: candidate | 252.781 d candidate (0.67); 285.557 d false alarm (0.12) |
| KIC 7772914 | PC | PC 23.134: not found | 2.964 d false alarm (0.00) |
| KIC 6584273 | PC | PC 51.111: candidate |  |
| KIC 3218908 | PC | PC 18.925: candidate; PC 1.475: candidate; AFP 4.152: candidate |  |
| KIC 9992325 | PC | PC 24.086: candidate | 143.342 d uncertain (0.43); 234.843 d false alarm (0.05) |
| KIC 6869184 | PC | PC 43.317: candidate | 284.824 d false alarm (0.14); 200.894 d false alarm (0.05) |
| KIC 5978170 | PC | PC 5.689: candidate | 203.883 d uncertain (0.40) |
| KIC 8639908 | PC | PC 0.910: false alarm | 280.730 d false alarm (0.03); 240.142 d false alarm (0.02) |
| KIC 8177798 | AFP | AFP 1.450: false alarm |  |
| KIC 8559589 | AFP | AFP 223.112: false alarm |  |
| KIC 10621643 | AFP | AFP 1.480: not found | 0.614 d false alarm (0.00) |
| KIC 6877871 | AFP | AFP 0.534: false alarm |  |
| KIC 7732461 | AFP | AFP 4.212: false alarm | 0.702 d false alarm (0.00) |
| KIC 5894825 | AFP | AFP 1.363: not found | 1.431 d false alarm (0.00) |
| KIC 6717417 | AFP | AFP 0.698: false alarm |  |
| KIC 5622759 | AFP | AFP 1.868: false alarm |  |
| KIC 10259029 | AFP | AFP 7.061: false alarm |  |
| KIC 3238073 | AFP | AFP 0.734: not found | 0.757 d false alarm (0.00) |
| KIC 6614492 | AFP | AFP 0.584: false alarm |  |
| KIC 5036037 | AFP | AFP 0.545: not found | 0.680 d false alarm (0.00) |
| KIC 2852528 | AFP | AFP 11.962: uncertain |  |
| KIC 10798605 | AFP | AFP 56.048: false alarm |  |
| KIC 6842139 | AFP | AFP 1.448: not found | 0.785 d false alarm (0.00) |
| TIC 288735205 | planet | CP 3.478: candidate | 220.939 d uncertain (0.62); 244.955 d false alarm (0.25) |
| TIC 260004324 | planet | CP 3.814: candidate | 263.637 d uncertain (0.88); 250.628 d uncertain (0.88) |
| TIC 138294130 | planet | KP 3.057: candidate | 23.955 d uncertain (0.49) |
| TIC 4610830 | planet | KP 8.262: candidate | 3.559 d candidate (0.94) |
| TIC 169461816 | planet | KP 17.855: uncertain | 0.636 d false alarm (0.13) |
| TIC 234523599 | planet | CP 3.796: candidate | 201.893 d uncertain (0.42); 180.793 d false alarm (0.32) |
| TIC 1333933 | planet | KP 3.276: candidate | 32.863 d uncertain (0.63) |
| TIC 395393265 | planet | CP 3.377: candidate | 75.050 d false alarm (0.16); 103.757 d false alarm (0.17) |
| TIC 270380593 | planet | KP 3.836: candidate | 52.516 d candidate (0.72) |
| TIC 243185500 | planet | CP 15.532: candidate; CP 1.881: candidate | 16.970 d candidate (0.68) |
| TIC 143526444 | FP | FP 15.305: false alarm |  |
| TIC 319735319 | FP | FP 2.904: uncertain |  |
| TIC 233569098 | FP | FP 23.485: candidate | 64.524 d uncertain (0.56); 113.605 d false alarm (0.11) |
| TIC 291013124 | FP | FP 6.037: false alarm |  |
| TIC 265077027 | FP | FP 4.796: uncertain | 2.253 d false alarm (0.11) |
| TIC 4294779 | FP | FP 0.344: not found | 0.681 d false alarm (0.07) |
| TIC 268371979 | FP | FP 2.781: false alarm | 0.897 d false alarm (0.05) |
| TIC 441334829 | FP | FP 4.443: uncertain |  |
| TIC 394509844 | FP | FP 6.891: candidate | 1.497 d false alarm (0.06) |
| TIC 179159972 | FP | FP 1.371: false alarm | 0.753 d false alarm (0.01) |
