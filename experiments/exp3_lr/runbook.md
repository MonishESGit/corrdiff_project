# exp03_lr_sweep — Runbook

## Objective
Test sensitivity of CorrDiff pipeline to regression learning rate by training regression with different LR values and evaluating generated forecasts.

## Runs
- lr = 1e-4
- lr = 2e-4 (baseline)
- lr = 1e-3

## Outputs
- `results/corrdiff_output_lr_1e-4.nc`
- `results/corrdiff_output_lr_2e-4.nc`
- `results/corrdiff_output_lr_1e-3.nc`

## Evaluation
Metrics computed vs truth for each output:
- `results/metrics.txt`
- `results/metrics.json`

## Notes
Interpretation is provided in `notes.md`.
