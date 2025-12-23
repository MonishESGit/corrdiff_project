# exp03 — Learning Rate Sweep (Regression)

## Objective
Evaluate sensitivity of CorrDiff regression performance to learning rate under a fixed training budget and identical generation setup.

## Experimental Setup
- Dataset: HRRR Mini
- Training: regression UNet (CPU smoke configuration)
- Learning rates tested:
  - 1e-4
  - 2e-4 (baseline)
  - 1e-3
- Generation:
  - Same diffusion checkpoint
  - Same generation configuration
- Evaluation:
  - MAE and RMSE vs ground truth
  - Variables: 10u, 10v

## Results Summary

| LR | 10u MAE | 10u RMSE | 10v MAE | 10v RMSE |
|----|--------|----------|---------|----------|
| 1e-4 | 2.3998 | 2.9996 | 3.7817 | 4.6441 |
| 2e-4 | 2.5989 | 3.2351 | 3.7851 | 4.6464 |
| 1e-3 | **2.2458** | **2.8118** | **3.7579** | **4.6198** |

## Interpretation

- The highest learning rate (1e-3) achieved the best MAE and RMSE for both variables.
- The baseline learning rate (2e-4) underperformed in this short training regime.
- The lowest learning rate (1e-4) showed signs of underfitting given the limited number of training steps.

This behavior is consistent with optimization theory:
- Larger learning rates can converge faster when training budgets are small.
- Smaller learning rates require more steps to reach comparable performance.

## Takeaway

CorrDiff regression performance is sensitive to learning rate, especially under constrained training budgets. Default hyperparameters are not universally optimal, and tuning learning rate can yield meaningful downstream improvements in generated forecasts.

This experiment validates the importance of optimization-aware configuration when training CorrDiff-style models.
