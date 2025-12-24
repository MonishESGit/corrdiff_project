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

### Experiment exp0_baseline_smoke — End-to-End Sanity Check

**Goal:**  
Validate the complete CorrDiff pipeline (regression → diffusion → generation)
using a minimal training budget and CPU-friendly configuration.

**What was done:**
- Trained regression UNet on HRRR mini dataset
- Trained diffusion residual model conditioned on regression output
- Generated forecasts using both checkpoints
- Verified NetCDF structure and computed MAE / RMSE metrics

**Why this matters:**  
This experiment establishes that:
- Dataset loading and normalization are correct
- Hydra configs resolve properly
- Checkpointing and reload work as expected
- Regression and diffusion models integrate correctly at inference time

All subsequent experiments (ensembles, learning rate sweeps, training budget,
and output-variable changes) build on this validated baseline.

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

### Experiment exp3 — Learning Rate Sensitivity

A regression learning-rate sweep (1e-4, 2e-4, 1e-3) shows that higher learning rates can significantly improve forecast accuracy under limited training budgets. In the HRRR Mini CPU-smoke setting, LR=1e-3 achieved the lowest MAE/RMSE for both 10u and 10v, outperforming the baseline configuration.

This highlights the importance of optimization-aware tuning rather than relying solely on default hyperparameters.

### Experiment exp4 — Diffusion Training Budget Matters

A controlled diffusion budget sweep (512 vs 2048 samples) with regression held fixed shows large and consistent gains in forecast accuracy when the diffusion model is trained longer. MAE and RMSE improved substantially for both 10u and 10v, confirming that diffusion training budget directly impacts the quality of residual correction in CorrDiff.

### Experiment exp5 — Output Variable Change (2t + tp)  
Demonstrates that CorrDiff can be reconfigured to predict entirely different physical quantities
by modifying `dataset.output_variables`. This affects model output channels, training loss,
generation artifacts, and evaluation metrics. Results confirm correct end-to-end execution,
with expected degradation due to limited training budget.

### Experiment exp6 — Best Feasible CPU Run

`exp06_best_cpu` represents the strongest CorrDiff configuration achievable
under CPU-only constraints.

It includes:
- Full regression and diffusion training logs
- Learning-curve visualizations
- Spatial forecast diagnostics
- Quantitative metrics (MAE/RMSE)

This experiment is used as the primary qualitative reference in the report,
while earlier experiments analyze architectural and hyperparameter sensitivity.



## Understanding & Analysis

In addition to running experiments, I spent time reading, tracing, and understanding
the full CorrDiff pipeline and its implementation in PhysicsNeMo.

This includes:

- **Regression model**
  - UNet-based deterministic predictor used to estimate the conditional mean
  - Role of regression as a low-frequency / large-scale baseline
  - Output channel configuration driven by `dataset.output_variables`
  - Interaction with normalization via `stats.json`

- **Diffusion model**
  - EDM-style residual diffusion model trained to predict stochastic corrections
  - Conditioning on regression output during training and generation
  - Noise scheduling, residual targets, and ensemble sampling behavior
  - Effect of training budget, learning rate, and diffusion steps on forecast quality

- **Generation pipeline**
  - How regression and diffusion are combined at inference time
  - Role of ensembles and stochastic sampling
  - Deterministic vs stochastic components of generation
  - Ensemble mean vs individual ensemble members

- **Configuration system (Hydra)**
  - How base configs, model configs, dataset configs, and overrides compose
  - Mapping from YAML configs to code paths in `train.py` and `generate.py`
  - Practical implications of changing hyperparameters such as
    learning rate, training duration, number of ensembles, and output variables

The experiments in this repository were designed not only to produce outputs,
but to validate understanding of how architectural choices, hyperparameters,
and configuration decisions affect model behavior and results.
