# exp04 — Notes (Diffusion training budget)

## Objective
Evaluate how diffusion training budget affects downstream forecast quality, while keeping the regression checkpoint and generation configuration fixed.

## Setup
- Dataset: HRRR Mini
- Regression: fixed (same checkpoint for all runs)
- Diffusion training budgets:
  - ~512 samples (short budget)
  - ~2048 samples (longer budget)
- Generation:
  - Same seed and configuration
  - Single timestamp
- Evaluation:
  - MAE and RMSE vs ground truth
  - Variables: 10u, 10v

## Results Summary

| Diffusion budget | 10u MAE | 10u RMSE | 10v MAE | 10v RMSE |
|------------------|--------|----------|---------|----------|
| 512 samples | 2.87 | 3.55 | 3.73 | 4.59 |
| 2048 samples | **1.87** | **2.29** | **2.64** | **3.25** |

## Interpretation

- Increasing diffusion training budget leads to substantial improvements in forecast accuracy.
- The effect is consistent across both wind components (10u and 10v).
- Since the regression model and generation configuration are fixed, the observed improvements can be attributed directly to better learning of the residual distribution by the diffusion model.

This confirms that diffusion training budget is a critical factor in CorrDiff performance, especially for capturing fine-scale corrections beyond the regression mean.

## Takeaway

Under identical regression and generation settings, allocating more training budget to the diffusion model significantly improves final forecast quality. This experiment highlights the compute–quality tradeoff inherent in CorrDiff-style probabilistic models.
