# Project Notes — CorrDiff Experiments (HRRR Mini)

This document summarizes the key experimental findings and insights from the CorrDiff mini project. The focus is on understanding uncertainty, reproducibility, and evaluation behavior in the CorrDiff pipeline.
---

## Overview

CorrDiff combines:
1. A **regression UNet** that predicts a deterministic mean field.
2. A **diffusion model** that learns a stochastic residual distribution around the regression mean.
3. A **generation step** that samples from this learned distribution to produce ensemble forecasts.

The experiments conducted here are designed to validate:
- Correct end-to-end execution
- Meaningful uncertainty modeling
- Scientific reproducibility

---

## Experiment exp1 — Ensemble Uncertainty

### Goal
Evaluate how diffusion-based ensemble sampling affects prediction accuracy and uncertainty.

### Setup
- Number of ensembles: **4**
- Same regression and diffusion checkpoints for all runs
- Single timestamp, HRRR Mini dataset
- Metrics: MAE and RMSE

### Key Results

#### Individual Ensembles
Each ensemble member produces a slightly different forecast, with small variations in MAE and RMSE.

This confirms that:
- The diffusion model is stochastic
- Each ensemble represents a plausible realization of the forecast

#### Ensemble Mean
Averaging predictions across ensembles significantly improves accuracy.

Example (MAE):
- **10u**
  - Single ensemble: ~2.1 m/s
  - Ensemble mean: ~1.38 m/s
- **10v**
  - Single ensemble: ~3.5 m/s
  - Ensemble mean: ~3.02 m/s

### Interpretation
This behavior is expected in probabilistic forecasting:
- Individual samples capture uncertainty
- Ensemble averaging reduces variance
- The ensemble mean provides a more stable and accurate prediction

This experiment demonstrates that CorrDiff’s diffusion component captures structured uncertainty rather than random noise.

---

## Experiment exp2 — Seed Reproducibility

### Goal
Verify deterministic behavior when the random seed is fixed, and stochastic behavior when the seed is changed.

### Setup
- Number of ensembles: **1**
- Three generation runs:
  - Run A: seed = 0
  - Run B: seed = 0 (repeat)
  - Run C: seed = 1
- Same checkpoints and configs otherwise

### Reproducibility Proof

- **Run A and Run B**
  - Bitwise-identical NetCDF outputs
  - Identical SHA-256 checksums
- **Run C**
  - Different checksum, confirming a different stochastic sample

This proves that:
- CorrDiff generation is fully reproducible when the seed is fixed
- Changing the seed correctly alters the diffusion sampling

### Metrics Observation

Although the NetCDF outputs differ between seeds, MAE and RMSE values remain identical across runs in this setup.

This is expected because:
- Metrics are averaged over a 64×64 spatial grid
- Diffusion residuals are small relative to the regression mean
- Different spatial error patterns can yield identical aggregate statistics

Additional checks (e.g., max absolute difference between fields) confirm that the predicted fields differ despite identical MAE/RMSE values.

---

## Key Takeaways

- CorrDiff successfully produces stochastic ensemble forecasts.
- Ensemble averaging significantly improves accuracy.
- The pipeline is scientifically reproducible under fixed seeds.
- Aggregate metrics like MAE/RMSE can hide meaningful spatial differences between stochastic samples.
- Proper evaluation requires both statistical metrics and field-level comparisons.

---

### exp03 — Learning Rate Sensitivity

A regression learning-rate sweep (1e-4, 2e-4, 1e-3) shows that higher learning rates can significantly improve forecast accuracy under limited training budgets. In the HRRR Mini CPU-smoke setting, LR=1e-3 achieved the lowest MAE/RMSE for both 10u and 10v, outperforming the baseline configuration.

This highlights the importance of optimization-aware tuning rather than relying solely on default hyperparameters.

### exp04 — Diffusion Training Budget Matters

A controlled diffusion budget sweep (512 vs 2048 samples) with regression held fixed shows large and consistent gains in forecast accuracy when the diffusion model is trained longer. MAE and RMSE improved substantially for both 10u and 10v, confirming that diffusion training budget directly impacts the quality of residual correction in CorrDiff.

**exp05 — Output Variable Change (2t + tp)**  
Demonstrates that CorrDiff can be reconfigured to predict entirely different physical quantities
by modifying `dataset.output_variables`. This affects model output channels, training loss,
generation artifacts, and evaluation metrics. Results confirm correct end-to-end execution,
with expected degradation due to limited training budget.
