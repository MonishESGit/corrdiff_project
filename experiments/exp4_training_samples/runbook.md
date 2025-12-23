# exp04_diffusion_budget — Runbook

## Objective
Measure how diffusion training budget impacts downstream forecast quality, while keeping the regression checkpoint and generation config fixed.

## Runs
- Diffusion trained with ~512 samples (short budget)
- Diffusion trained with ~2048 samples (longer budget)

## Outputs
- `results/corrdiff_output_samples512.nc`
- `results/corrdiff_output2048.nc`

## Evaluation
- `results/metrics.txt`
- `results/metrics.json`

## Notes
Interpretation is in `notes.md`.
