import xarray as xr
import numpy as np
import json
import os

FILES = {
    "lr_1e-4": "experiments/exp3_lr/results/corrdiff_output_lr_1e-4.nc",
    "lr_2e-4": "experiments/exp3_lr/results/corrdiff_output_lr_2e-4.nc",
    "lr_1e-3": "experiments/exp3_lr/results/corrdiff_output_lr_1e-3.nc",
}

OUT_DIR = "experiments/exp3_lr/results"
os.makedirs(OUT_DIR, exist_ok=True)

def mae_rmse(pred, truth):
    d = pred - truth
    return float(np.abs(d).mean()), float(np.sqrt((d**2).mean()))

results = {
    "experiment": "exp3_lr",
    "description": "Learning-rate sweep on regression training; generated outputs evaluated against truth.",
    "units": "m/s",
    "runs": {}
}

txt = []
txt.append("Experiment: exp3_lr")
txt.append("Description: LR sweep (regression) with fixed generation evaluation")
txt.append("Units: m/s\n")

for tag, path in FILES.items():
    truth = xr.open_dataset(path, group="truth")
    pred  = xr.open_dataset(path, group="prediction")

    txt.append(f"Run: {tag}  ({path})")
    results["runs"][tag] = {}

    for var in ["10u", "10v"]:
        if var not in truth or var not in pred:
            continue
        mae, rmse = mae_rmse(pred[var], truth[var])
        results["runs"][tag][var] = {"mae": mae, "rmse": rmse}
        txt.append(f"  {var}: MAE {mae:.4f}, RMSE {rmse:.4f}")
    txt.append("")

# write outputs
with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
    json.dump(results, f, indent=2)

with open(os.path.join(OUT_DIR, "metrics.txt"), "w") as f:
    f.write("\n".join(txt))

print("✔ Wrote:")
print(f"  {OUT_DIR}/metrics.txt")
print(f"  {OUT_DIR}/metrics.json")
