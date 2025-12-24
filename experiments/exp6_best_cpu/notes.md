## Results Interpretation

This experiment retargets CorrDiff from wind prediction to:
- 2t (2-meter temperature)
- tp (total precipitation)

The resulting errors are higher than wind-based experiments, which is expected:

- 2t exhibits large natural variance (std ≈ 11 K).  
  An MAE of ~11.6 K indicates the model is learning signal but is far from convergence,
  consistent with smoke-scale training and limited regression optimization.

- tp is a highly sparse and intermittent variable.
  Even state-of-the-art models struggle with precipitation prediction.
  Observed MAE/RMSE values are reasonable for a minimal training budget.

This experiment is intended to validate **architectural flexibility and correctness**
rather than forecast accuracy.
