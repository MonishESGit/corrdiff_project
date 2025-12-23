# exp1_ensembles4 — Runbook

## Objective

Evaluate CorrDiff’s probabilistic forecasting behavior by generating multiple diffusion-based ensembles and analyzing their variability and ensemble-mean performance.

---

## What Changed vs Baseline (exp0)

Only the generation configuration was modified.

### Config Change

- ```generation: num_ensembles = 4```

All other components (dataset, regression checkpoint, diffusion checkpoint) remained unchanged.

Checkpoints Used:

### Regression UNet:
- ```CorrDiffRegressionUNet.0.514.mdlus```

### Diffusion Model:
- ```EDMPrecondSuperResolution.0.513.mdlus```

### Command Executed
cd physicsnemo/examples/weather/corrdiff

```python generate.py --config-name=config_generate_hrrr_mini_regression_exp1.yaml```

## Outputs

#### NetCDF output: ```results/corrdiff_output.nc```


### Metrics:

- ```results/metrics.txt```
- ```results/metrics.json```

The generation progress bar showed 4/4, confirming that four ensemble members were produced.

### Metrics Summary

Metrics were computed per ensemble and for the ensemble mean.

See:
- ```results/metrics.txt for readable summary```
- ```results/metrics.json for structured data```

### Notes

This experiment demonstrates CorrDiff’s ability to model uncertainty via diffusion and the practical benefit of ensemble averaging.