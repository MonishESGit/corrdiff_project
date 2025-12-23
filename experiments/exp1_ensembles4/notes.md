# Experiment 1 Interpretation Notes

## Observations

- Individual ensemble members show small but consistent variability in MAE and RMSE.
- This variability reflects stochastic sampling in the diffusion model rather than instability.

## Ensemble Mean Effect

Averaging across ensembles significantly improves accuracy:

### 10u (Zonal Wind)
- Single ensemble MAE ≈ 2.1 m/s
- Ensemble-mean MAE ≈ 1.38 m/s

### 10v (Meridional Wind)
- Single ensemble MAE ≈ 3.5 m/s
- Ensemble-mean MAE ≈ 3.02 m/s

This confirms that:
- Diffusion captures meaningful uncertainty
- Ensemble averaging reduces variance and improves predictive accuracy

## Conclusion

CorrDiff’s diffusion component does not simply add noise; it learns a structured residual distribution that, when sampled multiple times, yields improved forecasts through ensemble statistics.
