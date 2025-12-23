# CorrDiff Mini Project — HRRR Mini Dataset

This repository documents a series of controlled experiments using NVIDIA PhysicsNeMo’s **CorrDiff** model on the HRRR Mini weather dataset. The goal is to understand, run, and evaluate the full CorrDiff pipeline (regression + diffusion + generation), with an emphasis on reproducibility, uncertainty estimation, and scientific reporting.

This work was completed as part of a small technical evaluation project.

---

## Project Overview

CorrDiff is a two-stage probabilistic forecasting model:

1. **Regression UNet**
   - Learns a deterministic mean prediction for target variables.
2. **Diffusion Model**
   - Learns a stochastic residual distribution around the regression mean.
3. **Generation**
   - Combines regression + diffusion to produce one or more ensemble forecasts.

This repo demonstrates:
- End-to-end execution of the pipeline
- Ensemble-based uncertainty estimation
- Metric-based evaluation using NetCDF outputs
- Clean experiment tracking and documentation

---

## Dataset

- **Dataset:** HRRR Mini (Continental US)
- **Spatial Resolution:** 64 × 64 grid
- **Targets:** 10u, 10v (10m zonal & meridional wind)
- **Inputs:** Meteorological fields such as u/v winds at multiple pressure levels, temperature, humidity, surface pressure, etc.

---

## Experiments

| Experiment ID | Description |
|--------------|------------|
| exp0 | Baseline smoke test (single ensemble) |
| exp1 | Ensemble uncertainty experiment (4 ensembles) |

Each experiment contains:
- Configs used
- Runbook with commands
- NetCDF outputs
- Metric files (`metrics.txt`, `metrics.json`)
- Interpretation notes

---
All experiments were run using CPU smoke configurations on macOS. Exact commands and checkpoints are recorded per experiment in each `runbook.md`.

---

## Key Takeaways

- CorrDiff successfully produces stochastic ensemble forecasts.
- Ensemble averaging significantly improves accuracy over individual samples.
- Proper experiment documentation makes results reproducible and interpretable.
