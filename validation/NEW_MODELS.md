# Newer model families: Mamba-2 and a time-series foundation model (12 Nov 2026)

Question: do newer architectures beat the models TransitVet already uses? Two were tested under exactly the same
rules as the CNN and Transformer: same views (global 2001 bins, local 201 bins), same star splits, settings chosen
on validation stars only, and test stars reported once.

## Models

- **Mamba-2** (selective state-space model; Gu & Dao 2023, Dao & Gu 2024). It is written in plain PyTorch
  (`transitvet/tv_models.py`, `Mamba2Block`, `PatchMamba`). The scan uses its exact matrix ("state-space duality") form,
  and it was checked against the step-by-step recurrence (largest difference 1.5e-8). The model reads the same
  89 patches as the Transformer, runs in both directions, and has 4 blocks.
  The setting was chosen on validation PR-AUC from three one-seed runs:
  d=128, dropout 0.1 (val 0.906); d=128 + flip/noise augmentation (0.916 at step 500, stopped);
  **d=64, lr 1e-3, dropout 0.2, weight decay 0.1, time-flip + noise augmentation (0.935, chosen)**.
- **MOMENT-1-small** (AutonLab, 2024): a 35M-parameter Transformer pretrained on a large collection of time series.
  It is used frozen. Each view is resampled to 512 samples, and its embeddings (512 + 512) feed a logistic regression
  or a small MLP, chosen on validation (`improvements/new_models/probe.py`).

## Kepler (DR24, split by star, 2,373 test signals)

| Model | Test PR-AUC (mean ± sd of 5 seeds) | 5-model average | ROC AUC |
|---|---|---|---|
| AstroNet CNN (reproduced baseline) | 0.941 ± 0.003 | 0.947 | 0.984 |
| **Mamba-2** | 0.926 ± 0.004 | 0.934 | 0.979 |
| Transformer, pretrained | 0.931 ± 0.002 | 0.932 | 0.979 |
| Transformer, no pretraining | 0.923 ± 0.002 | 0.930 | 0.976 |
| MOMENT (frozen) + logistic regression | 0.918 (one fit) | | 0.975 |
| Gradient boosting (check numbers) | 0.885 ± 0.005 | | 0.972 |

## TESS (labelled TOIs, split by star, 1,000 test signals)

| Model | Kepler-trained, unchanged | Fine-tuned on TESS | TESS only |
|---|---|---|---|
| AstroNet CNN | 0.835 | 0.858 | 0.822 |
| **Mamba-2** | 0.840 | 0.852 | 0.808 |
| Transformer | 0.819 | 0.842 | 0.795 |
| MOMENT (frozen) + logistic regression | | | 0.823 |

## Inside the combined score (logistic regression on log-odds, fitted on validation stars)

| Combination | Kepler test PR-AUC | TESS test PR-AUC |
|---|---|---|
| GB + CNN + Transformer (shipped up to v2) | 0.9675 | 0.9457 |
| GB + CNN + Transformer + MOMENT | 0.9674 | 0.9462 |
| GB + CNN + Transformer + Mamba | 0.9675 | 0.9459 |
| **GB + Mamba + Transformer** | 0.9632 | 0.9460 |
| GB + Mamba | 0.9625 | 0.9460 |
| GB + CNN | 0.9674 | 0.9461 |

With GB + Mamba + Transformer, the 95% guarantee still holds on test stars: Kepler planets kept 95.8%, false alarms 95.3%;
TESS 98.3% / 98.5%. The "needs a human" count on TESS rises from 448 to 508 of 1,000.

## What this says

- Mamba-2 reaches the Transformer's level on Kepler and sits between the CNN and the Transformer on TESS.
  At this data size (about 11,000 training signals), it does not beat the CNN.
- A general time-series foundation model, used frozen, falls below every model trained on the task, and it adds nothing
  to the combined score.
- The final score is driven by the gradient-boosting check numbers plus one strong view-based network. Adding more
  networks gives no measurable gain (every bootstrap interval of the gain includes zero).

Scripts: `improvements/new_models/` (Mamba training is `e2_neural.py mamba` with the settings above,
TESS is `e4_mamba.py`, MOMENT is `embed.py` + `probe.py`, combinations are `stack.py`, and the model set is `build_models3.py`).
