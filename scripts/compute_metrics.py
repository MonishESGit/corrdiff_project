import xarray as xr
import numpy as np
import json

truth = xr.open_dataset("experiments/exp0_baseline/results/corrdiff_output.nc", group="truth")
pred  = xr.open_dataset("experiments/exp0_baseline/results/corrdiff_output.nc", group="prediction")

var = "10u"

t = truth[var]
p = pred[var].isel(ensemble=0)

diff = p - t

mae = float(np.abs(diff).mean())
rmse = float(np.sqrt((diff ** 2).mean()))

metrics = {
    "experiment": "exp0_baseline",
    "variable": var,
    "units": "m/s",
    "metrics": {
        "mae": mae,
        "rmse": rmse
    },
    "notes": "Sanity check only. CPU-only smoke-test training configuration."
}

with open("experiments/exp0_baseline/results/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

with open("experiments/exp0_baseline/results/metrics.txt", "w") as f:
    f.write(
        f"Experiment: exp0_baseline\n"
        f"Variable: {var}\n"
        f"Units: m/s\n\n"
        f"MAE:  {mae:.4f}\n"
        f"RMSE: {rmse:.4f}\n\n"
        "Notes:\n"
        "Metrics computed as a sanity check only.\n"
        "Training used reduced CPU-only smoke-test configuration.\n"
    )

print("Metrics written to results/")
