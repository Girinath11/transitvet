# Pixel-level check (difference imaging)

Added 4 Nov 2026. Code: `transitvet/tv_pixel.py`. Validation scripts: `improvements/pixel_run.py`, `improvements/pixel_analyse.py`.

## What it does

A light curve adds up the light from a small patch of camera pixels. If a nearby eclipsing star falls inside that patch, it can make a dip that looks like a planet on the target star. The camera images, called Target Pixel Files, show which pixels actually get darker.

For each TESS sector or Kepler quarter, the check works like this:

1. For every dip, take the mean image during the dip (the central 80%) and the mean image just before and just after it.
2. Make the difference image: out-of-dip minus in-dip, averaged over all dips. Its bright spot marks where the dip comes from.
3. Estimate the pixel noise by repeating the same measurement at fake dip times between the real dips.
4. Fit a 2-D Gaussian (a model of a star's image) to the difference image. The file's sky coordinates then give how far the dip's source is from the target's catalogue position.
5. Combine the sectors on the sky. This gives an offset in arcseconds and how significant that offset is (sigma).

If the offset is significant, the tool looks up the closest star in the TESS Input Catalog to the dip's source and names it.

This is the same idea as the difference-image centroid test in NASA's Kepler and TESS pipelines (Bryson et al. 2013; Twicken et al. 2018) [CHECK citations]. Our version is simpler: it uses a Gaussian instead of the instrument's real point-spread model.

## How it was tested

- **Signals:** all 259 TESS validation signals and all 1,000 TESS test signals (1,179 stars), using the TOI catalogue's period, epoch and duration. At most 2 sectors per star.
- **Measured share:** the dip was strong enough to measure in the pixels for 69% of signals (73% of planets, 66% of false alarms).
- **Threshold:** chosen on the validation signals only. The warning fires when sigma > 5 and the offset is more than 10 arcsec (about half a TESS pixel). This flags 2.0% of validation planets and 24% of validation false alarms.

## Results on TESS test signals

The test set has 542 planets and 458 false alarms.

| | Count |
|---|---|
| Planets flagged | 19 of 542 (3.5%) |
| False alarms flagged | 113 of 458 (24.7%) |
| False alarms called candidate by the combined score (p ≥ 0.5), before | 82 |
| ... after the pixel warning turns them to "uncertain" | 65 |
| Planets called candidate that move to "uncertain" | 18 of 493 |
| PR-AUC of the combined score | 0.946 (unchanged when the flag is used to lower the score) |

Other thresholds, for comparison only. These were not used to choose the threshold.

| sigma, arcsec | Planets flagged | False alarms flagged | False-alarm candidates left (of 82) |
|---|---|---|---|
| 5, 10 (chosen) | 19 | 113 | 65 |
| 3, 15 | 15 | 81 | 69 |
| 10, 21 | 2 | 53 | 76 |

## What this means

- The pixel check catches about 1 in 4 TESS false alarms. Most of those were already rejected by the scores. It adds 17 false alarms that the scores had wrongly called candidates.
- It does not change the ranking quality (PR-AUC), so we use it as a warning and a verdict rule, not as a score input. A flagged candidate becomes "uncertain", which means it goes to a human and is not dropped.
- The cost is that 18 test planets (3.6% of found planets) are sent to a human. Some of these may sit in crowded fields where the real planet host is not the catalogue target.
- **Caveat:** many official TESS false-positive labels were themselves decided with difference imaging. Part of this agreement is therefore built in, as with the light-position check.
- **Kepler:** this has not been validated yet. The tool shows the numbers, but it does not raise a warning on Kepler. The light-position check (validated) covers Kepler.
- **Speed and size:** a TESS sector's pixel file is about 100 MB, and the check adds about 10 to 20 s per star. The batch runner deletes the files after use unless `--keep-downloads` is given.
