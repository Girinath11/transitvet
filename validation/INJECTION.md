# Injection-recovery test (17 Nov 2026)

**Question.** If a planet of a given size and orbit is really there, how often does the whole TransitVet pipeline find it, and what does it then call it?

**Method.** We added fake planets (batman transit model, quadratic limb darkening, circular orbit, impact parameter 0 to 0.8) to the
RAW light curves of held-out TESS test stars, then ran the dashboard pipeline unchanged: detrend, BLS search, checks, Gaia numbers,
models, verdict and 95% check (the pixel check was off to save time). The period was drawn log-uniform from 0.7 d to a third of the
data span (at most 200 d). The size was set so the expected signal-to-noise was spread log-uniform between 3 and 80, so the test samples
the range where recovery changes. A fake planet counts as found when a signal matches its period within 1% (or half or double) and its
time of dip. Scripts: `scripts/injection_test.py` (runs the test, one result row per fake planet) and
`scripts/injection_analyse.py` (curves and figure). Their folder paths are set for our machine; edit them before re-running.

Expected signal-to-noise = depth / (robust scatter of the flattened curve) x sqrt(points in transit).

**What was left out.**
- 31 fake planets had fewer than 3 dips land in the data (gaps or long orbits). The search needs 3 dips by design, so these are not counted.
- 4 fake planets went on a giant star (radius 60 Suns). There the injected depth corresponds to an object far larger than a planet, so a false-alarm call is correct; these are not counted.
- 2 Kepler fake planets were run (about 17 minutes each); too few to report, so they are only included in the curve.

**Result: 90 fake planets on 18 stars (88 TESS, 2 Kepler).**

| Expected SNR | Fake planets | Found | Found, called candidate | Found, called uncertain |
| --- | --- | --- | --- | --- |
| 0 to 3 | 6 | 0 | 0 | 0 |
| 3 to 7.1 | 31 | 2 | 0 | 0 |
| 7.1 to 10 | 9 | 3 | 1 | 0 |
| 10 to 15 | 7 | 6 | 1 | 2 |
| 15 to 25 | 9 | 9 | 4 | 0 |
| 25 to 50 | 14 | 13 | 6 | 3 |
| 50 and up | 14 | 14 | 4 | 4 |

- The search finds half of the planets at an expected SNR of about 10 and 90% at about 16 (logistic fit in log SNR). The BLS cut-off is 7.1.
- Above SNR 15, 36 of 37 were found.

**What the found planets were called depends strongly on the star.**

| Host star (its own TOI turned out to be) | Stars | Found | Candidate | Uncertain | False alarm | 95% check says keep |
| --- | --- | --- | --- | --- | --- | --- |
| a planet | 7 | 20 | 13 | 3 | 4 | 14 |
| a false alarm | 10 | 27 | 3 | 6 | 18 | 1 |

On stars whose own signal was a false alarm, most real-looking fake planets were also called false alarms. A re-run of four such cases
showed that all three models score them low, not only the gradient-boosting model that reads Gaia; removing the Gaia numbers made the
score lower, not higher. Several of these hosts are hot stars (7,000 to 10,000 K), where pulsation and other eclipses in the light curve
can make any folded dip look less like a planet [CHECK the cause]. The practical meaning: a "false alarm" call on a star that already
hosts a false alarm should be read with care. The 95% check said keep for only 1 of these 27 (against 14 of 20 on planet hosts).

By size of the found planet: under 4 Earth radii 9 of 20 were called candidate;
4 to 10 Earth radii 7 of 19; 10 to 16 Earth radii 0 of 8.

**Dashboard.** For every star, the dashboard now shows the chance that the search would find planets of 1, 2 and 4 Earth radii at
3, 10 and 30 days, from the star's own scatter and number of points and the fitted curve above (`transitvet/sensitivity.py`,
`transitvet/sensitivity.json`). Its signal-to-noise estimate matches the one measured in the injection test to within about 5%
(median ratio 0.99, 10th to 90th percentile 0.95 to 1.10, 17 checks). It assumes a circular orbit and even coverage.

**Limits.** 90 planets on 18 stars is a small test; the curve has wide uncertainty, especially between SNR 7 and 15. The host
split is based on 7 and 9 stars. Figure: `paper/figures/fig_sensitivity.png`.
