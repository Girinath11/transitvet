# TransitVet: calibrated vetting of Kepler and TESS transit signals with state-space and attention models, a 95% coverage check, and Gaia context

*Draft of 17 November 2026. Authors: Akilan S, Misham Samuel Daystar, Sachin Sarlin K R S, Girinath A. Guide: Mrs. R. A. Rakhi.*

## Abstract

We built TransitVet, an open pipeline that searches Kepler and TESS light curves for repeating dips and labels each signal as a planet candidate or a false alarm. We first reproduced the AstroNet CNN of Shallue & Vanderburg (2018) on Kepler DR24 (accuracy 0.955, AUC 0.989 under its split by signal). We then re-ran it under a stricter split by star. The final tool joins gradient boosting on vetting numbers, a Mamba-2 state-space model and a patch Transformer through a logistic combiner fitted on validation stars. On held-out stars it reaches PR-AUC 0.963 on Kepler (2,373 signals) and 0.950 on TESS (1,000 signals). On 542 TESS signals also scored by NASA's ExoMiner++, the two systems are within the uncertainty (0.968 vs 0.973). Split conformal prediction turns the score into keep / drop / needs a person, with at least 95% coverage measured on test stars. An injection-recovery test with 90 fake planets shows the search finds half of them at an expected signal-to-noise of about 10 and 90% at about 16, and that the call on a found planet depends strongly on the host star. We release the code, a live dashboard and a ranked list of 4,408 TESS signals that are still open on ExoFOP.

Keywords: exoplanets, transit photometry, vetting, Kepler, TESS, state-space models, conformal prediction, Gaia.

## 1. Introduction

Transit surveys flag far more signals than people can check by hand, and most flagged signals are not planets. They are eclipsing binaries, light from a nearby star, or instrument noise. Machine-learning vetting started with AstroNet, a 1D CNN on folded light-curve views (Shallue & Vanderburg 2018). Later work added more inputs and moved to TESS, for example ExoMiner (Valizadegan et al. 2022) and ExoMiner++ (Valizadegan et al. 2025) from NASA Ames.

This work makes four contributions:

1. A reproduction of AstroNet. It shows that AstroNet's split by signal and a split by star give different test numbers (accuracy 0.955 vs 0.951).
2. One protocol (split by star, choices made on validation only, test used once) under which we compare a rule, gradient boosting, the CNN, a Transformer and a Mamba-2 model, on Kepler and on TESS.
3. A combined score with a split conformal check. Signals it cannot place go to a person instead of getting a guessed label.
4. Context for TESS from Gaia DR2 and difference images, an injection-recovery test of the whole pipeline, a working dashboard, and a ranked list of open TESS signals.

Conformal prediction has been applied to transit detection before, in ExoVeil (Priyanshu 2026), so we make no priority claim for it.

## 2. Data and labels

| Set | Source | Signals | Stars | Labels | Split |
| --- | --- | --- | --- | --- | --- |
| Kepler | DR24 TCE table (Coughlin et al. 2016), long-cadence light curves | 15,737 | 9,865 | PC = 1; AFP, NTP = 0 | by star 70/15/15 (test 2,373) |
| TESS labelled | TOI table (Guerrero et al. 2021), SPOC 2-min light curves (TESS: Ricker et al. 2015) | 2,502 | 2,357 | CP, KP = 1; FP, FA = 0 | by star 50/10/40 (test 1,000) |
| TESS open | TOI table, PC / APC | 4,408 | 4,294 | none | scored only |

All 6,910 TOIs with SPOC 2-min data, on 6,611 stars, were processed with 0 failures. TESS sectors were binned to 30 minutes before detrending. Gaia DR2 (Gaia Collaboration 2018) sources within 60 arcsec were fetched from MAST for all 7,826 TOI host stars. 7,815 of them matched a Gaia star (the brightest within 3 arcsec, or within 10 arcsec for fast-moving stars).

Label tables were downloaded from the NASA Exoplanet Archive on 1 Nov 2026.

## 3. Pipeline and checks

Figure 1 (paper/figures/fig_pipeline.png) shows the whole pipeline.

1. **Clean and detrend.** Keep quality-flag 0 points. A biweight filter (wotan; Hippke et al. 2019) prepares the curve for the search. A piecewise spline, as in AstroNet, prepares it for the folded views.
2. **Search.** Box least squares (Kovács et al. 2002) from 0.6 d up to a third of the data span (at most 500 d). Up to 5 signals per star are kept, each needing SNR ≥ 7.1. Each found signal is masked before the next search. If every second dip is empty, the period is doubled.
3. **Folded views.** A global view of 2,001 bins and a local view of 201 bins, as in AstroNet.
4. **Check numbers.** Depth, SNR, odd/even difference, dip at half orbit, dip shape, number of dips, single dips alike, period, duration and close-up coverage.
5. **Light-position check.** Shift of the flux-weighted centroid during the dip, plus a difference-image check on TESS camera pixels.
6. **Star context.** TIC radius and crowding (contratio), planet size from depth × radius, and for TESS 13 Gaia DR2 numbers. These cover the star's astrometric excess noise, radial-velocity error and colour excess, plus neighbours within 21 and 42 arcsec and the dip depth the brightest neighbour would need to fake the signal. The TIC position is moved to the Gaia epoch (2015.5) with the star's proper motion before matching.
7. **Dip timing.** Each dip is timed with a trapezoid model. Errors follow Carter et al. (2008). A signal is marked as having timing variations when at least 4 dips are timed, the robust scatter is at least 3 times the errors, and the excess is at least 5 sigma. This is display only.
8. **Transit fit** (batman; Kreidberg 2015) for the period, size ratio and shape. This is display only.

## 4. Models, combined score and the 95% check

| Model | Input | Notes |
| --- | --- | --- |
| Gradient boosting | check numbers (+ Gaia on TESS) | 5 seeds, histogram GB, missing values handled natively |
| Mamba-2 (Dao & Gu 2024) | both folded views, as patches | bidirectional SSD form, d = 64, dropout 0.2; trained on Kepler, fine-tuned on TESS |
| Patch Transformer | both folded views, as patches | masked-patch pretraining on training stars |
| AstroNet CNN | both folded views | reproduced baseline; reported, not used in the tool |

Each model is calibrated with isotonic regression on validation stars. The combined score is a logistic regression on the three models' log-odds, also fitted on validation stars.

The 95% check uses class-conditional (Mondrian) split conformal prediction (Vovk et al. 2005; Lei et al. 2018). One half of the validation stars fits a combiner, and the other half sets per-class cut-offs at α = 0.05. If only "planet" stays in the set, the call is keep. If only "false alarm" stays, the call is drop. Otherwise the signal goes to a person. It also goes to a person when the call and the score point different ways. The verdict is candidate at a score of at least 0.5. It becomes uncertain when the score is between 0.35 and 0.65, all single models disagree, fewer than 3 dips are seen, a second dip appears at the same period, the single dips do not look alike (Section 6), or the difference image points to a neighbour.

## 5. Results

All numbers are on held-out test stars, with model choices made on validation stars only.

**AstroNet reproduction (Kepler DR24).** Under AstroNet's split by signal (seed 123, 10 models), we get accuracy 0.955 and AUC 0.989; the paper reports 0.960 and 0.988. Under a split by star (80/10/10), we get accuracy 0.951 and AUC 0.982. The two test sets differ, so this is not a paired comparison.

**Kepler, split by star (test 2,373 signals), PR-AUC**

| Model | PR-AUC |
| --- | --- |
| Threshold rule | 0.463 |
| Gradient boosting (light curve checks only) | 0.885 |
| Transformer, pretrained | 0.931 |
| Mamba-2 (5-model average) | 0.934 |
| AstroNet CNN | 0.941 |
| **Combined (GB + Mamba-2 + Transformer)** | **0.963** |

**TESS, split by star (test 1,000 signals, 54% planets)**

| Model | PR-AUC | False alarms caught at 0.5 | Planets kept at 0.5 |
| --- | --- | --- | --- |
| Best single model, light curve only | 0.883 | | |
| Combined, before Gaia (v3) | 0.946 | 363 / 458 | 492 / 542 |
| **Combined with Gaia (v4)** | **0.950** | **382 / 458** | **493 / 542** |

The Gaia gain in PR-AUC is +0.004, with a 95% bootstrap interval over stars of −0.008 to +0.016. The clearer effect is in the 95% check: "needs a person" falls from 508 to 318 of 1,000.

**95% check on test stars**

| Mission | Planets kept in set | False alarms kept in set | Keep | Drop | Needs a person |
| --- | --- | --- | --- | --- | --- |
| Kepler (2,373) | 96.3% | 97.3% | 532 | 1,786 | 55 |
| TESS (1,000) | 98.0% | 96.3% | 387 | 295 | 318 |

**Against ExoMiner++.** On 542 TESS test signals with a published ExoMiner++ score, TransitVet reaches 0.968 and ExoMiner++ 0.973. The difference is −0.005, with an interval of −0.019 to +0.010. The average of the two scores reaches 0.982.

**Other findings.** Masked-patch pretraining helps the Transformer most with few labels: +0.026 PR-AUC at 10% of the training stars. Models trained on Kepler and used unchanged on TESS are poorly calibrated (ECE 0.15–0.18). Isotonic calibration on TESS validation stars brings this to about 0.06. On a 4-CPU machine, inference costs 1.37 ms per signal for the CNN, 0.68 ms for the Transformer and 7 µs for GB. A whole-star run is dominated by the BLS search.

**Dip timing example.** On Kepler-90 the timing check flags planet d (scatter 4.2 times the expected errors). Published timing work on Kepler-90 covers the outer planets g and h (Cabrera et al. 2014; Liang et al. 2021); we found no published timing variation for d, so we treat this flag as unconfirmed.

## 6. Open TESS signals

We scored all 4,408 TOIs still open on ExoFOP (3,964 PC, 439 APC) with the v4 tool. 1,304 are strong candidates (score ≥ 0.5, at least 3 dips, 95% check says keep). 398 of those have an estimated size under 4 Earth radii. A further 1,603 look like false alarms, and 1,988 go to a person.

As a sanity check, APC signals come out as false alarms 66% of the time and PC signals 33% of the time. ExoMiner++ also scores 2,502 of these signals. The two land on the same side of 0.5 for 72% of them, with a Spearman correlation of 0.59.

The list is meant for planning follow-up, not as confirmation. The full table is released with the code.

**Forward test.** Our calls on these 4,408 signals are frozen as of 17 Nov 2026. Our training labels date from 1 Nov. Every week a script compares them with NASA's latest TOI table, and the website shows the tally. On 17 Nov, 6 had been resolved. Of the 5 we scored at 0.5 or above, 4 are now planets and 1 a false alarm. The 1 we scored below 0.5 is a false alarm. This sample is far too small to judge; it is the start of a test on signals no model could have seen, which will grow over the coming months.

**Single dips look alike (a check for our own search).** When the search keeps up to 5 signals, some long-period signals come from one or two odd events. We measure each single dip on its own. A signal with 6 dips or fewer needs at least 3 dips seen at 3 sigma and within 50% of the median depth, or a candidate becomes uncertain. These thresholds were fixed before testing. We tested the rule on 271 held-out few-dip signals, with light curves downloaded by the dashboard's own loader. On Kepler it moved none of the 9 candidate planets. On TESS it moved 11 of 22 candidate false alarms and 15 of 69 candidate planets to uncertain. That raises the share of TESS few-dip candidates that are planets from 76% to 83%.

## 7. Injection-recovery test

We added 90 fake planets to the raw light curves of 18 held-out test stars and ran the whole pipeline unchanged (Figure 6;
details in validation/INJECTION.md). Sizes were chosen so the expected signal-to-noise (depth / scatter x sqrt(points in transit))
spread from 3 to 80. Fake planets with fewer than 3 dips in the data (31) and those on a giant star (4),
where the dip would come from a star-sized body, were left out.

| Expected SNR | Fake planets | Found | Found, called candidate |
| --- | --- | --- | --- |
| 0 to 3 | 6 | 0 | 0 |
| 3 to 7.1 | 31 | 2 | 0 |
| 7.1 to 10 | 9 | 3 | 1 |
| 10 to 15 | 7 | 6 | 1 |
| 15 to 25 | 9 | 9 | 4 |
| 25 to 50 | 14 | 13 | 6 |
| 50 and up | 14 | 14 | 4 |

The search finds half of the fake planets at an expected SNR of about 10 and 90% at about 16. What a found planet is then
called depends on the host star. On 7 stars whose own TOI is a planet, 13 of 20 found fake planets
were called candidates. On 10 stars whose own TOI is a false alarm, only 3 of 27 were, and
18 were called false alarms. All three models score these low, not only the model that reads Gaia, so this is not
caused by the Gaia numbers alone. The 95% check said keep for 1 of them. One possible reading is that the models have learned that some
kinds of stars rarely host planets [CHECK]; it is a real limit for planets around such stars, and the dashboard says so next to each star's detection
chance.


## 8. Limits, the tool, and conclusion

**Limits.**

- TESS labels are few: 2,502 signals, against 15,737 for Kepler.
- The official dispositions were made partly with the same kinds of checks we feed the model, such as centroid shifts and crowding. That can inflate the scores.
- On TESS, the combiner gives Mamba-2 almost no weight (0.01), so there the result rests mainly on gradient boosting.
- Gaia DR2 is used because MAST serves it. DR3 and RUWE may do better.
- The light-position check cannot resolve neighbours closer than about one pixel. The search can miss very small planets: it does not find Kepler-910 b (0.82 Earth radii), and we have not yet checked why.
- In the injection test, fake planets on stars whose own signal was a false alarm were mostly called false alarms (18 of 27 found). A planet around such a star may be missed by the score even when the search finds it.

**The tool.** It is released under the MIT licence, with a Streamlit dashboard and a command-line batch mode. The dashboard shows each step live and gives a plain-words summary and a report card per signal. It also shows a top-down view of the system, plays each dip as sound for accessibility, estimates the chance of catching planets of a few sizes around the star, plans follow-up for the user's telescope, and has a tab with the 4,408 open TESS signals and the weekly forward check. There are 43 automated tests and 9 example stars with saved runs. A star someone already analysed on the server opens at once.

**Conclusion.** One honest protocol, split by star with test used once, puts a light open pipeline within the uncertainty of ExoMiner++ on shared TESS signals. A conformal check then tells users which signals it cannot place, and an injection-recovery test tells them which planets the whole pipeline can catch.

## Figures

1. fig_pipeline.png: the pipeline.
2. fig_pr_curves.png: precision-recall curves on test stars.
3. fig_reliability.png: calibration before and after isotonic fitting.
4. fig_gaia_importance.png: which Gaia numbers the TESS model uses.
5. fig_kepler90_timing.png: dip timing on Kepler-90.
6. fig_sensitivity.png: injection-recovery curves.

## To do before submission

- [ ] Choose the venue and its template
- [ ] Write the related-work section
- [ ] Ask the guide to review the draft

## References

- Cabrera, J., et al. 2014, ApJ, 781, 18. The planetary system to KIC 11442793: a compact analogue to the solar system.
- Carter, J. A., Yee, J. C., Eastman, J., Gaudi, B. S., & Winn, J. N. 2008, ApJ, 689, 499. Analytic approximations for transit light-curve observables, uncertainties, and covariances.
- Coughlin, J. L., et al. 2016, ApJS, 224, 12. Planetary candidates observed by Kepler. VII. The first fully uniform catalog based on the entire 48-month data set (Q1–Q17 DR24).
- Dao, T., & Gu, A. 2024, Proc. ICML (PMLR 235). Transformers are SSMs: generalized models and efficient algorithms through structured state space duality. arXiv:2405.21060.
- Gaia Collaboration, Brown, A. G. A., et al. 2018, A&A, 616, A1. Gaia Data Release 2: summary of the contents and survey properties.
- Guerrero, N. M., et al. 2021, ApJS, 254, 39. The TESS Objects of Interest catalog from the TESS prime mission.
- Hippke, M., David, T. J., Mulders, G. D., & Heller, R. 2019, AJ, 158, 143. Wōtan: comprehensive time-series detrending in Python.
- Kovács, G., Zucker, S., & Mazeh, T. 2002, A&A, 391, 369. A box-fitting algorithm in the search for periodic transits.
- Kreidberg, L. 2015, PASP, 127, 1161. batman: BAsic Transit Model cAlculatioN in Python.
- Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J., & Wasserman, L. 2018, JASA, 113, 1094. Distribution-free predictive inference for regression.
- Liang, Y., Robnik, J., & Seljak, U. 2021, AJ, 161, 202. Kepler-90: giant transit-timing variations reveal a super-puff.
- Priyanshu, P. 2026, arXiv:2606.02778. One transit is all you need: detecting exoplanets through learned stellar behaviour with ExoVeil.
- Ricker, G. R., et al. 2015, JATIS, 1, 014003. Transiting Exoplanet Survey Satellite.
- Shallue, C. J., & Vanderburg, A. 2018, AJ, 155, 94. Identifying exoplanets with deep learning: a five-planet resonant chain around Kepler-80 and an eighth planet around Kepler-90.
- Valizadegan, H., et al. 2022, ApJ, 926, 120. ExoMiner: a highly accurate and explainable deep learning classifier that validates 301 new exoplanets.
- Valizadegan, H., et al. 2025, arXiv:2502.09790 (submitted to AJ). ExoMiner++: enhanced transit classification and a new vetting catalog for 2-minute TESS data.
- Vovk, V., Gammerman, A., & Shafer, G. 2005, Algorithmic Learning in a Random World (Springer).
