import xarray as xr
import numpy as np
import json
import os

# Paths
NC_PATH = "experiments/exp1_ensembles4/results/corrdiff_output_exp01_ensembles4.nc"
OUT_DIR = "experiments/exp1_ensembles4/results"

os.makedirs(OUT_DIR, exist_ok=True)

# Load datasets
truth = xr.open_dataset(NC_PATH, group="truth")
pred  = xr.open_dataset(NC_PATH, group="prediction")

def mae_rmse(pred, truth):
    diff = pred - truth
    mae = float(np.abs(diff).mean())
    rmse = float(np.sqrt((diff ** 2).mean()))
    return mae, rmse

metrics = {
    "experiment": "exp1_ensembles4",
    "description": "CorrDiff generation with num_ensembles=4. Metrics reported per-ensemble and for ensemble mean.",
    "units": "m/s",
    "variables": {}
}

txt_lines = [
    "Experiment: exp1_ensembles4",
    "Description: Generation with 4 diffusion ensembles",
    "Units: m/s",
    ""
]

for var in ["10u", "10v"]:
    if var not in truth or var not in pred:
        continue

    t = truth[var]
    p = pred[var]

    txt_lines.append(f"Variable: {var}")

    per_ensemble = []
    for e in range(p.sizes["ensemble"]):
        pe = p.isel(ensemble=e)
        mae, rmse = mae_rmse(pe, t)
        per_ensemble.append({
            "ensemble": int(e),
            "mae": mae,
            "rmse": rmse
        })
        txt_lines.append(f"  Ensemble {e}: MAE {mae:.4f}, RMSE {rmse:.4f}")

    p_mean = p.mean(dim="ensemble")
    mae_m, rmse_m = mae_rmse(p_mean, t)

    txt_lines.append(f"  Ensemble-mean: MAE {mae_m:.4f}, RMSE {rmse_m:.4f}")
    txt_lines.append("")

    metrics["variables"][var] = {
        "per_ensemble": per_ensemble,
        "ensemble_mean": {
            "mae": mae_m,
            "rmse": rmse_m
        }
    }

# Write outputs
with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

with open(os.path.join(OUT_DIR, "metrics.txt"), "w") as f:
    f.write("\n".join(txt_lines))

print("✔ Metrics written:")
print(f"  - {OUT_DIR}/metrics.txt")
print(f"  - {OUT_DIR}/metrics.json")
