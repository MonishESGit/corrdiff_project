import xarray as xr
import numpy as np
import json
import os

FILES = {
    "samples_512": "experiments/exp4_training_samples/results/corrdiff_output_samples512.nc",
    "samples_2048": "experiments/exp4_training_samples/results/corrdiff_output_samples2048.nc",
}

OUT_DIR = "experiments/exp4_training_samples/results"
os.makedirs(OUT_DIR, exist_ok=True)

def mae_rmse(pred, truth):
    d = pred - truth
    return float(np.abs(d).mean()), float(np.sqrt((d**2).mean()))

results = {
    "experiment": "exp04_diffusion_budget",
    "description": "Diffusion training budget sweep (512 vs 2048 samples) with fixed regression checkpoint and fixed generation config",
    "units": "m/s",
    "runs": {}
}

lines = []
lines.append("Experiment: exp04_diffusion_budget")
lines.append("Description: Diffusion training budget sweep (512 vs 2048 samples), regression fixed")
lines.append("Units: m/s\n")

for tag, path in FILES.items():
    truth = xr.open_dataset(path, group="truth")
    pred  = xr.open_dataset(path, group="prediction")

    lines.append(f"Run: {tag}  ({path})")
    results["runs"][tag] = {}

    for var in ["10u", "10v"]:
        if var not in truth or var not in pred:
            continue
        mae, rmse = mae_rmse(pred[var], truth[var])
        results["runs"][tag][var] = {"mae": mae, "rmse": rmse}
        lines.append(f"  {var}: MAE {mae:.4f}, RMSE {rmse:.4f}")
    lines.append("")

with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
    json.dump(results, f, indent=2)

with open(os.path.join(OUT_DIR, "metrics.txt"), "w") as f:
    f.write("\n".join(lines))

print("✔ exp04 metrics written:")
print(f"  {OUT_DIR}/metrics.txt")
print(f"  {OUT_DIR}/metrics.json")
