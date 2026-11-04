# TransitVet

**Website:** https://girinath11.github.io/transitvet/ · **Live dashboard:** coming soon

TransitVet looks at a star's brightness over time (its **light curve**), finds small dips that repeat, and scores how likely each dip is a **planet candidate**. A candidate still needs follow-up by astronomers: TransitVet never confirms a planet.

## Install and run

**Windows:** double-click `install_windows.bat` once. It creates a private Python environment, installs everything and runs the tests. After that, double-click `run_dashboard.bat`. You need Python 3.11 to 3.13 from python.org, with "Add python.exe to PATH" ticked.

**Linux / macOS:** `bash install.sh`, then `. .venv/bin/activate && streamlit run app.py`.

**Docker:** `docker build -t transitvet .` then `docker run -p 8501:8501 transitvet`. The build runs the tests.

The dashboard opens at http://localhost:8501. The models are in `transitvet/models/`. The two CNN files (199 MB each) are too big for normal GitHub storage, so they are published as a release download; the first run fetches them automatically and checks their SHA-256 checksums (`python -m transitvet.fetch_models` does it by hand).

## Command line (many stars)

```bash
python -m transitvet run "TIC 1333933" "KIC 11442793"
python -m transitvet run --file stars.txt --out results --workers 4
python -m transitvet test            # automated tests (add --network for the MAST download tests)
```

What a batch run does:

- **Parallel:** one star per worker. Each worker loads the models once.
- **Resumable:** stars that already have a result in `results/stars/` are skipped, so you can re-run the same command after a crash or a stop.
- **Time limit:** each star gets `--timeout` seconds (default 1800). A star that runs over is recorded as an error, and the run carries on.
- **Retries:** downloads are retried with back-off.
- **Logged:** `results/run.log`, with failures in `results/errors.csv`.
- **Outputs:** `results/summary.csv` has one row per signal, ranked by score. `results/run_manifest.json` records the settings, model versions and timing.
- **Pixel check:** on by default. Use `--no-pixel` to skip it. The camera images are deleted after use unless you give `--keep-downloads`.

From Python:

```python
from transitvet import pipeline as P
scorer = P.Scorer()
segs, meta = P.load_star('KIC 11442793')          # or 'TIC 261136679', or a plain number + mission
meta['_pixel'] = True                             # optional pixel-level check
result = P.run(segs, meta, scorer)
print(P.summary_table(result))
```

## Tests

There are 31 automated tests in `tests/`, and they cover every step:

- cleaning and loading
- the dip search, including two planets, pure noise and an eclipsing binary
- the folded views and check numbers
- model loading and determinism
- the verdict and trust-check rules
- the transit fit, the planner and the light-position check
- the pixel check
- a full offline run on a real TESS star (TIC 1333933, stored in `tests/data/`)
- the live pipeline view: it receives every stage in order and does not change the results

The tests already caught one real bug, which is now fixed: when an eclipsing binary's second dip was found at half the period, the eclipsing-binary warning was missed.

## What it does (one star)

| Step | What happens | Signal-processing idea |
|---|---|---|
| 1. Load | Light curve from NASA's MAST archive (Kepler long cadence; TESS 2-minute, averaged to 30 min), or an uploaded FITS / CSV file | sampling, resampling |
| 2. Clean | Keep good-quality points; each segment normalised; upward spikes removed | outlier rejection |
| 3. Detrend | Biweight filter (for the search) and AstroNet's spline (for the model inputs) | high-pass filtering |
| 4. Search | Box Least Squares (BLS) finds the strongest repeating dip, removes it, searches again (up to 3 signals, SNR ≥ 7.1, at least 3 dips) | matched filtering |
| 5. Fold | Whole-orbit view (2001 bins) and zoomed dip view (201 bins), plus check numbers (odd/even dips, dip at half phase, shape) | coherent averaging |
| 6. Score | Gradient boosting (with light-position, size and crowding checks), AstroNet CNN, Transformer, and a combined score | detection / classification, data fusion |
| 7. Verdict | candidate / false alarm / uncertain | thresholding |

**New checks (3 Nov 2026):**
- *Light-position check*: each file records where the star's light falls on the camera. If that spot moves during the dip, the dip may come from a nearby star.
- *Planet size*: dip depth plus the star's radius from the TESS Input Catalog gives the object's size; more than 2 Jupiters is almost surely a star.
- *Crowding*: how much light in the aperture comes from neighbouring stars (TESS Input Catalog).
- *Transit fit* (batman): sharper period, planet size with an error bar, V-shape check, and the star density implied by the dip.
- *Official lists*: each signal is matched against the Kepler KOI and TESS TOI tables (display only, never used in the score).
- *Next dips*: predicted times with a rough ± error.
- *NASA second opinion*: for TESS signals in the published ExoMiner++ catalogue (NASA Ames, sectors 1-67), its score is shown next to ours. On 542 shared test signals: ExoMiner++ PR-AUC 0.973, TransitVet 0.967 (no clear difference), average of both 0.981. See `validation/EXOMINER_COMPARISON.md`.

**Added 4 Nov 2026:**
- *Trust check (95% guarantee)*: each signal gets one of three calls: **keep** (a false alarm is ruled out), **drop** (a planet is ruled out) or **needs a human**. The thresholds come from split conformal prediction, fitted on half of the validation stars, so that at least 95% of real planets are never dropped and at least 95% of real false alarms are never kept. On test stars: Kepler 95.8% / 95.7%, TESS 97.8% / 98.0%. On TESS 448 of 1,000 test signals go to "needs a human". Change the level with the environment variable `TV_CONFORMAL_ALPHA` (0.05, 0.10 or 0.20). See `validation/GUARANTEE_AND_PLANNER.md`.
- *Follow-up planner*: set your site and telescope in the sidebar. For each signal the app lists the next dips that happen at night with the star above 30°, and the expected signal-to-noise for your telescope (photon noise, scintillation, a 500 ppm ground floor): clear (SNR ≥ 7), marginal (4-7) or too faint. The Batch tab builds one observing plan for all candidates. These are planning estimates; weather, the moon and comparison stars also matter.

- *Pixel check (difference images)*: downloads the camera images (Target Pixel Files) and checks which pixels get darker during the dip. If the dip comes from more than 10 arcsec away at more than 5 sigma, the tool warns, names the closest catalogue star to the dip's source and turns a "candidate" into "uncertain". On 1,000 TESS test signals it flagged 25% of false alarms and 3.5% of planets, and cut the false alarms wrongly called candidates from 82 to 65. This is TESS only; it is not validated on Kepler yet. See `validation/PIXEL_CHECK.md`.

**Main score:** the *combined score*, a logistic regression on the log-odds of gradient boosting (now with the new checks), the AstroNet CNN and the Transformer, fitted on validation stars.
Held-out test (PR-AUC): Kepler 0.968 (AstroNet CNN alone 0.947), TESS 0.946 (best single model before 0.883). Details in `validation/IMPROVEMENTS.md`.
**Verdict rule:** candidate if the combined score is at least 0.5, false alarm if below; *uncertain* if it is between 0.35 and 0.65, if all three single models disagree with it, if fewer than 3 dips were seen, or if a second separate dip shows up at the same period. Warnings (odd/even difference, dip at half phase, light moving, too big, V-shape, density mismatch) are shown in a report card but do not change the verdict.

## Outputs
- Table per signal: verdict, the three scores, period, duration, depth, BLS SNR, check numbers, warnings. Download as CSV.
- Plots: detrended light curve with the dips marked, BLS periodogram, folded views.
- Run manifest (JSON): time, settings, model versions and library versions, so a run can be repeated.

## Batch
Paste many star IDs, or upload several FITS/CSV files or a ZIP. You get one combined table ranked by score.

## Known limits
- A nearby eclipsing star closer than about half a pixel (10 arcsec on TESS) cannot be separated by the pixel check, and a third of dips are too faint to measure in the pixels.
- TESS has far fewer labelled examples; TESS scores are less reliable (E4/E5).
- The BLS search can lock onto half or double the true period, or onto instrument artefacts; check the plots.
- Periods shorter than 0.6 days are not searched. If the same period comes back in a later search, the search stops for that star: a dip in the same place is a leftover and is dropped; a separate dip elsewhere in the orbit is reported as an eclipsing-binary warning.
- Speed: a Kepler star (4 years of data) takes a few minutes, mostly the BLS search; a TESS star with a few sectors takes under a minute, one with many sectors a few minutes.
- Names like "Kepler-90" need the MAST name resolver; KIC / TIC numbers always work.

## Where the models come from
`build_models.py` builds them from the experiment code: Kepler models trained on 70% of Kepler DR24 stars (split by star), TESS models fine-tuned on 50% of labelled TESS stars; calibrators fitted on the validation stars only. See `transitvet-data/results/` for the experiments.

## Test on 50 held-out stars
See `validation/VALIDATION.md`. In short: Kepler planet candidates were found 16 of 18 times and 13 of those 16 were called candidate; TESS confirmed or known planets 11 of 11 called candidate. Weak spot: TESS false positives, where only 3 of 10 were called false alarm (5 candidate, 2 uncertain).


## Website source

`docs/index.html` is the published website (GitHub Pages). It is built from `website-src/` with `python build_space.py` (background images: `python make_space.py`; they are computer-generated, not telescope photos).
