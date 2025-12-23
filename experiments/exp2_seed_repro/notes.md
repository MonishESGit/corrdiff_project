Although the outputs generated with different seeds are bitwise different (verified via SHA-256 checksums), the MAE and RMSE values remain identical in this experiment.

This is expected because:
- Errors are averaged over a 64×64 spatial grid
- Diffusion residuals are small compared to the regression mean
- Different stochastic samples can yield the same global error statistics

Additional checks (e.g., max absolute difference) confirm that the predicted fields differ despite identical aggregate metrics.
