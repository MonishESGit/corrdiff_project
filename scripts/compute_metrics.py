import xarray as xr
import numpy as np
import json

truth = xr.open_dataset("experiments/exp1_ensembles4/results/corrdiff_output_exp01_ensembles4.nc", group="truth")
pred  = xr.open_dataset("experiments/exp1_ensembles4/results/corrdiff_output_exp01_ensembles4.nc", group="prediction")

var = "10u"

t = truth[var]
p = pred[var].isel(ensemble=0)

diff = p - t

mae = float(np.abs(diff).mean())
rmse = float(np.sqrt((diff ** 2).mean()))

metrics = {
    "experiment": "exp1_ensembles4",
    "variable": var,
    "units": "m/s",
    "metrics": {
        "mae": mae,
        "rmse": rmse
    },
}

with open("experiments/exp1_ensembles4/results/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

with open("experiments/exp1_ensembles4/results/metrics.txt", "w") as f:
    f.write(
        f"Experiment: exp1_ensembles4\n"
        f"Variable: {var}\n"
        f"Units: m/s\n\n"
        f"MAE:  {mae:.4f}\n"
        f"RMSE: {rmse:.4f}\n\n"
    )

print("Metrics written to results/")
