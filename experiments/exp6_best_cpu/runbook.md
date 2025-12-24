# Experiment 06 — Best Feasible CPU Run (CorrDiff)

## Objective
This experiment represents the **best feasible CorrDiff configuration under CPU-only constraints**.
It serves as the qualitative and quantitative reference run for this project.

Unlike earlier experiments that isolate individual factors (ensembles, learning rate, training budget, output variables),
exp06 combines the strongest settings that were practical to run locally.

## Configuration Summary
- Dataset: HRRR-mini
- Outputs: `10u`, `10v`
- Regression model: CorrDiffRegressionUNet (UNet-based mean predictor)
- Diffusion model: EDM-based residual diffusion model
- Hardware: CPU-only (Apple Silicon)
- Training mode: Non-patched
- Ensembles: 1 (deterministic generation)

### Key hyperparameters
- Regression learning rate: tuned based on prior LR sweep
- Diffusion training budget: selected from budget sweep results
- Generation: fixed configuration for evaluation consistency

See `configs/` and training logs for full details.

## Artifacts
- Training logs:
  - `logs/regression_train.log`
  - `logs/diffusion_train.log`
- Generation output:
  - `results/corrdiff_output_best_cpu.nc`
- Metrics:
  - `results/metrics.json`
  - `results/metrics.txt`

## Figures
Generated figures are stored under `figures/exp06_best_cpu/`:
- Training loss curves (regression and diffusion)
- Spatial truth / prediction / error maps
- Error histograms
- Truth vs prediction scatter plots

## Purpose in Project
exp06 is used to:
- Visualize **learning dynamics**
- Demonstrate **forecast quality**
- Act as a baseline for comparing earlier experiments
