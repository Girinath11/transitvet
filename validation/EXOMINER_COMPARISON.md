# TransitVet vs NASA's ExoMiner++ (TESS)

**ExoMiner++** (NASA Ames; Valizadegan, Martinho et al., Astronomical Journal 170, 2025; code github.com/nasa/ExoMiner) is NASA's own deep-learning vetting model for TESS. Its team published scores for every TESS SPOC 2-minute signal from sectors 1-67 (Zenodo record 15466293). For labelled signals the scores are out-of-fold cross-validation scores, so each one comes from a model that did not train on that signal [CHECK in the paper whether their folds are split by star].

**How we compared.** Our TESS test split (1,000 labelled TOI signals, split by star, never used to train, calibrate or tune). A test signal is matched to ExoMiner++ when the TIC number is the same and the period agrees within 1%. When NASA's pipeline found it in several runs, we use the run covering the most sectors. 542 of our 1,000 test signals matched (347 planets CP/KP, 195 false positives FP/FA). The others were not detected by NASA's 2-min pipeline in sectors 1-67 (for example, signals only seen in full-frame images), so ExoMiner++ has no score for them. Labels: ours (TOI table, 1 Nov 2026); NASA's training label differs on only 2 of the 542.

| on the same 542 test signals | PR-AUC | ROC-AUC | planets kept at 0.5 | false alarms caught at 0.5 |
|---|---|---|---|---|
| TransitVet (combined score) | 0.967 | 0.946 | 321/347 | 160/195 |
| NASA ExoMiner++ | 0.973 | 0.963 | 322/347 | 172/195 |
| average of the two | 0.981 | | | |

Bootstrap by star (2,000 resamples):
- ExoMiner++ minus TransitVet PR-AUC: +0.006, 95% interval -0.009 to +0.020. **No clear difference.**
- Average minus ExoMiner++ alone: +0.008, 95% interval +0.001 to +0.018. **Averaging helps**: the two models make partly different mistakes (score correlation 0.80).

**Fairness notes**
- ExoMiner++ uses more information than TransitVet: NASA pipeline products (difference images, which locate the source of the dip on the sky), 2-minute data, and its own training set. TransitVet uses only the light curve files and the TIC catalogue.
- On ROC-AUC and false alarms caught at 0.5, ExoMiner++ is ahead, mostly from its difference-image check, which TransitVet cannot do from light-curve files.
- ExoMiner++ only covers signals NASA's pipeline detected (sectors 1-67). TransitVet also scores new dips found by its own search, and Kepler stars.
- The average-of-both result was looked at on the test set, so treat it as a finding to confirm, not a tuned setting. The tool shows ExoMiner++ as a second opinion and does **not** change its verdict with it.

**In the tool:** for TESS signals found in the ExoMiner++ catalogue, the dashboard shows NASA's score, the run it comes from, NASA's label, and a warning when the two scores differ by more than 0.5.

Code: validation/exominer_compare.py; per-signal results: validation/exominer_compare_test.csv.
