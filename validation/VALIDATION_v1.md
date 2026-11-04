# TransitVet tool: test on held-out stars

Stars picked at random from the TEST split (never used for training, fine-tuning or calibration): 15 Kepler stars with a planet candidate (PC), 15 Kepler stars with a false alarm (AFP), 10 TESS planet stars (CP/KP/PC), 10 TESS false-positive stars (FP). For each star the tool downloads the light curve itself, runs its own dip search, and scores what it finds. Catalogue entries at the same period (within 1%) on the same star are counted once. A catalogue signal counts as found if one of the tool's periods matches it within 1% (or at half or double the period).

## KEPLER (30 stars, 0 failed to load)

| catalogue label (PC/CP/KP = planet; AFP, NTP, FP, FA = not a planet) | signals | not found by search | candidate | uncertain | false alarm |
|---|---|---|---|---|---|
| PC | 18 | 2 | 13 | 2 | 1 |
| AFP | 16 | 5 | 2 | 0 | 9 |

Time per star (4 stars at once, 1 core each): median 9.1 min, max 11.7 min.


## TESS (20 stars, 0 failed to load)

| catalogue label (PC/CP/KP = planet; AFP, NTP, FP, FA = not a planet) | signals | not found by search | candidate | uncertain | false alarm |
|---|---|---|---|---|---|
| CP | 6 | 0 | 6 | 0 | 0 |
| KP | 5 | 0 | 5 | 0 | 0 |
| FP | 10 | 0 | 5 | 2 | 3 |

Time per star (4 stars at once, 1 core each): median 2.7 min, max 9.1 min.


## What this shows (written by hand from the tables above)

- **Kepler planet candidates:** the search found 16 of 18 catalogue PC signals. Of those 16, the tool called 13 candidate, 2 uncertain and 1 false alarm (KIC 8639908, period 0.91 d). The 2 missed: a weaker second planet on KIC 4736569 (7.07 d) and KIC 7772914 (23.13 d), where the search locked onto a 2.96 d signal instead.
- **Kepler false alarms (AFP):** 11 of 16 found; 9 called false alarm, 2 candidate (KIC 2852528 at 11.96 d, and an AFP sharing a star with real PCs on KIC 3218908). The 5 not found are short-period eclipsing binaries or variable stars (0.55 to 1.48 d). For 4 of them the tool found a nearby but not matching period and called it false alarm anyway; one (0.545 d) is below the 0.6 d shortest period searched.
- **TESS planets (CP/KP):** all 11 found and all 11 called candidate.
- **TESS false positives:** all 10 found, but only 3 called false alarm, 2 uncertain and 5 candidate. This is the weak spot and matches E4: TESS false positives are often light from a nearby star, which the light curve alone cannot show.
- These are 50 stars only, so each percentage here has a wide margin; the experiment results (thousands of signals) are the main evidence. This test checks that the full tool (download, search, score) works end to end.
- Run with the code before the "same period again" fix, so some eclipsing binaries show the same period 2 or 3 times in the raw output; the fix only removes those repeats.

## Per star

| star | label | catalogue signals (label, period d, tool verdict) | other signals the tool found |
|---|---|---|---|
| KIC 10090854 | PC | PC 12.631: candidate |  |
| KIC 9847239 | PC | PC 28.508: uncertain | 226.045 d false alarm (0.00); 2.331 d false alarm (0.00) |
| KIC 4736569 | PC | PC 10.132: candidate; PC 7.074: not found | 269.192 d false alarm (0.00); 184.634 d false alarm (0.00) |
| KIC 3632089 | PC | PC 31.779: candidate; PC 5.328: candidate | 258.768 d false alarm (0.08) |
| KIC 6206214 | PC | PC 13.540: candidate | 298.006 d false alarm (0.03) |
| KIC 11192235 | PC | PC 1.615: uncertain | 246.036 d false alarm (0.00); 139.787 d false alarm (0.11) |
| KIC 10190777 | PC | PC 1.411: candidate | 267.476 d false alarm (0.00); 278.609 d false alarm (0.00) |
| KIC 5966322 | PC | PC 60.928: candidate | 252.781 d uncertain (0.28); 285.557 d false alarm (0.03) |
| KIC 7772914 | PC | PC 23.134: not found | 2.964 d false alarm (0.00); 1.482 d false alarm (0.00); 88.065 d false alarm (0.00) |
| KIC 6584273 | PC | PC 51.111: candidate |  |
| KIC 3218908 | PC | PC 18.925: candidate; PC 1.475: candidate; AFP 4.152: candidate |  |
| KIC 9992325 | PC | PC 24.086: candidate | 143.342 d false alarm (0.11); 234.843 d false alarm (0.03) |
| KIC 6869184 | PC | PC 43.317: candidate | 284.824 d uncertain (0.03); 200.894 d false alarm (0.00) |
| KIC 5978170 | PC | PC 5.689: candidate | 203.883 d false alarm (0.11) |
| KIC 8639908 | PC | PC 0.910: false alarm | 280.730 d false alarm (0.00); 240.142 d false alarm (0.00) |
| KIC 8177798 | AFP | AFP 1.450: false alarm | 0.725 d false alarm (0.00) |
| KIC 8559589 | AFP | AFP 223.112: false alarm | 223.117 d uncertain (0.00); 216.462 d false alarm (0.03) |
| KIC 10621643 | AFP | AFP 1.480: not found | 0.614 d false alarm (0.00); 0.614 d false alarm (0.00) |
| KIC 6877871 | AFP | AFP 0.534: false alarm | 1.075 d false alarm (0.00) |
| KIC 7732461 | AFP | AFP 4.212: false alarm | 0.702 d false alarm (0.00) |
| KIC 5894825 | AFP | AFP 1.363: not found | 1.431 d false alarm (0.00); 0.717 d false alarm (0.00) |
| KIC 6717417 | AFP | AFP 0.698: false alarm | 0.697 d false alarm (0.00); 0.697 d false alarm (0.00) |
| KIC 5622759 | AFP | AFP 1.868: false alarm | 0.934 d false alarm (0.00) |
| KIC 10259029 | AFP | AFP 7.061: false alarm | 7.061 d false alarm (0.00); 3.152 d false alarm (0.00) |
| KIC 3238073 | AFP | AFP 0.734: not found | 0.757 d false alarm (0.00) |
| KIC 6614492 | AFP | AFP 0.584: false alarm | 1.169 d false alarm (0.00); 1.168 d false alarm (0.00) |
| KIC 5036037 | AFP | AFP 0.545: not found | 0.680 d false alarm (0.00); 0.680 d false alarm (0.00); 0.680 d false alarm (0.00) |
| KIC 2852528 | AFP | AFP 11.962: candidate | 11.963 d false alarm (0.00); 293.127 d false alarm (0.28) |
| KIC 10798605 | AFP | AFP 56.048: false alarm | 56.048 d false alarm (0.00); 203.384 d false alarm (0.08) |
| KIC 6842139 | AFP | AFP 1.448: not found | 0.785 d false alarm (0.00); 0.785 d false alarm (0.00); 0.785 d false alarm (0.00) |
| TIC 288735205 | planet | CP 3.478: candidate | 220.939 d uncertain (0.87); 244.955 d false alarm (0.17) |
| TIC 260004324 | planet | CP 3.814: candidate | 263.637 d uncertain (0.92); 250.628 d uncertain (0.87) |
| TIC 138294130 | planet | KP 3.057: candidate | 23.955 d uncertain (0.87) |
| TIC 4610830 | planet | KP 8.262: candidate | 3.559 d candidate (0.87) |
| TIC 169461816 | planet | KP 17.855: candidate | 0.636 d uncertain (0.69); 0.636 d uncertain (0.87) |
| TIC 234523599 | planet | CP 3.796: candidate | 201.893 d false alarm (0.25); 180.793 d uncertain (0.39) |
| TIC 1333933 | planet | KP 3.276: candidate | 32.863 d uncertain (0.87) |
| TIC 395393265 | planet | CP 3.377: candidate | 75.050 d uncertain (0.92); 103.757 d uncertain (0.39) |
| TIC 270380593 | planet | KP 3.836: candidate | 52.516 d uncertain (0.92) |
| TIC 243185500 | planet | CP 15.532: candidate; CP 1.881: candidate | 16.970 d uncertain (0.39) |
| TIC 143526444 | FP | FP 15.305: candidate | 15.306 d uncertain (0.92) |
| TIC 319735319 | FP | FP 2.904: uncertain | 5.917 d uncertain (0.39); 0.743 d false alarm (0.00) |
| TIC 233569098 | FP | FP 23.485: candidate | 64.524 d uncertain (0.69); 113.605 d uncertain (0.39) |
| TIC 291013124 | FP | FP 6.037: false alarm | 3.019 d uncertain (0.50); 13.894 d uncertain (0.39) |
| TIC 265077027 | FP | FP 4.796: candidate | 2.253 d false alarm (0.17) |
| TIC 4294779 | FP | FP 0.344: false alarm | 0.681 d false alarm (0.17) |
| TIC 268371979 | FP | FP 2.781: uncertain | 0.897 d false alarm (0.00) |
| TIC 441334829 | FP | FP 4.443: candidate |  |
| TIC 394509844 | FP | FP 6.891: candidate | 1.497 d false alarm (0.25) |
| TIC 179159972 | FP | FP 1.371: false alarm | 0.753 d false alarm (0.00); 0.752 d false alarm (0.00) |
