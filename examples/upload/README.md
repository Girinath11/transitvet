# Example files for the upload feature

Ten light-curve files to test **Upload a file** on the dashboard (or the Batch tab, which takes several files or a ZIP).
Nine are real light curves of real stars. One has a fake planet added, so the right answer is known exactly.

Each file is a plain CSV: a few header lines that start with `#`, then the columns `time, flux, flux_err`.
The flux is NASA's PDCSAP flux, with each Kepler quarter or TESS sector divided by its median.
The TESS files hold the first 3 sectors only, to keep them small.

## How the upload works

1. The file is read: a FITS light curve from MAST, or a CSV/TXT with a time column and a flux column.
2. If the file name holds a star number (`KIC 3847138.csv`, `tic_19342878.csv`, ...), the tool uses it. It then picks the mission and looks up the star's size, crowding, Gaia numbers and catalogue match, just as for a star typed by ID. Without a number, the tool runs on the light curve alone and these checks are left empty.
3. From there the upload follows the same path as a star ID: cleaning, transit search (up to 5 signals), folded views, vetting checks, the three models, the combined score, the 95% check, and the report card for each signal.
4. The pixel (difference-image) check needs the pixel files, so it is skipped for uploads.

## Answer key

"What the tool says" is the tool's call for the main signal, from a run of the upload path on 20 Nov 2026 (scores are the combined score).
The split column says where the star sits in the paper's split by star. Only stars marked *test* or *val* were never used to train the models.
Calls for *train* stars are therefore not a fair test; they are there to show how false alarms look in the report.

| File | What it is | Known answer | Split | What the tool says |
|---|---|---|---|---|
| `KIC 3847138.csv` | Kepler planet candidate, about 2 Earth radii | planet candidate, 11.72 d | test | **candidate**, 0.965, period 11.723 d, 95% check: keep |
| `KIC 6046540.csv` | Kepler planet candidate, giant (about 15 Earth radii) | planet candidate, 7.34 d | test | **candidate**, 0.903, period 7.341 d, keep |
| `KIC 9291378.csv` | Kepler eclipsing binary (has a secondary eclipse) | false positive, 3.80 d | test | **false alarm**, 0.012, period 3.796 d, drop |
| `KIC 6341932.csv` | Kepler signal that is not a transit | false alarm, 1.46 d | test | **false alarm**, 0.001, period 1.460 d, drop |
| `KIC 9139151 with fake planet.csv` | Real light curve of a quiet Sun-like star (quarters 1-8) with a **fake planet** added | planet, 9.30 d, 302 ppm, about 2 Earth radii, 70 dips | (star not in the labelled set) | search finds it exactly (9.300 d, 267 ppm, 2.1 Earth radii, SNR 145); **uncertain**, 0.392, keep. See the note below |
| `TIC 19342878.csv` | TOI-6101.01 | confirmed planet, 1.33 d | test | **candidate**, 0.957, period 1.333 d, keep |
| `TIC 1129033.csv` | TOI-398.01 | known planet, 1.36 d | test | **uncertain**, 0.521, period 1.360 d, needs a person |
| `TIC 261136679.csv` | pi Mensae (TOI-144.01, the planet pi Men c) | confirmed planet, 6.27 d | val | **candidate**, 0.950, period 6.268 d, keep (signal 2; signal 1 at 62.5 d is uncertain) |
| `TIC 357416482.csv` | TOI-2017.01 | false positive, 5.50 d | train | **false alarm**, 0.073, period 5.498 d, drop |
| `TIC 87597187.csv` | TOI-3740.01 | false alarm, 2.04 d | train | **false alarm**, 0.286, period 2.045 d, needs a person |

On the 7 real stars that were never used for training (test and val), the tool finds the known signal at the right period every time.
It calls 6 of the 7 correctly and marks TOI-398.01 as uncertain, which sends it to a person rather than making a wrong call.
Several files show extra, weaker signals (signal 2, 3, ...); these are mostly called false alarm, as expected for leftover noise.

**Note on the fake planet.** The search finds the injected planet at the right period, depth and size, but the models give it only 0.39.
This matches the injection-recovery test in the paper: injected planets score lower than real planets of the same size, and the cause is not yet known [CHECK].
The 95% check still says *keep*, so the signal is not thrown away.

A full run of one file took 1.4 to 21 minutes here (one CPU thread each, four files at a time); Kepler files with 4 years of data take longest.

## Rebuild the files

`scripts/download_examples.py` downloads the nine real light curves from MAST with lightkurve.
`scripts/make_synthetic.py` builds the fake-planet file (needs `batman-package`). The fake planet's settings are written in that file's header.
