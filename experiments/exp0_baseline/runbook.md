# exp0_baseline — Runbook

This runbook documents the exact steps used to reproduce the baseline CorrDiff
pipeline (regression → diffusion → generation) under a reduced, CPU-only
configuration. The goal of this experiment is pipeline validation rather than
model performance.

---

## Configs Used

The following configuration files were used for this experiment:
- `configs/config_training_hrrr_mini_regression_cpu_smoke.yaml`
- `configs/config_training_hrrr_mini_diffusion_cpu_smoke.yaml`
- `configs/config_generate_hrrr_mini_cpu_smoke.yaml`

These configs are stored here for record-keeping.  
During execution, they were either referenced directly or copied into the
CorrDiff config directory, depending on Hydra resolution behavior.

## Regression Training

Command:

cd physicsnemo/examples/weather/corrdiff
python train.py --config-name=/Users/monish/Desktop/Research Job/corrdiff_project/experiments/exp0_baseline/configs/config_training_hrrr_mini_regression_cpu_smoke.yaml

Outputs (local):

- `checkpoints_regression/CorrDiffRegressionUNet.0.512.mdlus`
- `checkpoints_regression/checkpoint.0.512.pt`

---

## Diffusion Training

Command:

cd physicsnemo/examples/weather/corrdiff  
python train.py --config-name=/Users/monish/Desktop/Research Job/corrdiff_project/experiments/exp0_baseline/configs/config_training_hrrr_mini_diffusion_cpu_smoke.yaml

Outputs (local):

- `checkpoints_diffusion/EDMPrecondSuperResolution.0.512.mdlus`
- `checkpoints_diffusion/checkpoint.0.512.pt`

---

## Generation

Command:

cd physicsnemo/examples/weather/corrdiff  
python generate.py --config-name=/Users/monish/Desktop/Research Job/corrdiff_project/experiments/exp0_baseline/configs/config_generate_hrrr_mini_cpu_smoke.yaml

Output (local):

- `corrdiff_output.nc`

---

## Verification

NetCDF structure inspection command:

ncdump -h corrdiff_output.nc

Saved output:

- results/ncdump_header.txt

This confirms the presence of spatial dimensions, ensemble and time axes, and
the input, truth, and prediction groups.

---

## Metrics

Sanity-check metrics were computed for the 10m zonal wind field (10u) by
comparing predictions against ground truth.

Saved results:

- results/metrics.txt
- results/metrics.json

Metric values:

- MAE ≈ 2.1556 m/s
- RMSE ≈ 2.7167 m/s

These metrics are used only to validate numerical correctness and output
scaling for the smoke-test configuration.

---

## Notes

- This experiment was run entirely on CPU with reduced training duration and
  batch size.
- Default CorrDiff configurations assume GPU hardware and were adapted for
  local execution.
- Due to Hydra configuration search-path behavior, relative --config-name
  paths may not resolve in all environments. When necessary, configs were copied
  into the CorrDiff configs directory for execution and preserved here as a
  record of the exact settings used.

---