import xarray as xr
import numpy as np
import json
import os

FILES = {
    "seed0_runA": "experiments/exp2_seed_repro/results/corrdiff_output_seed0_runA.nc",
    "seed0_runB": "experiments/exp2_seed_repro/results/corrdiff_output_seed0_runB.nc",
    "seed1_runC": "experiments/exp2_seed_repro/results/corrdiff_output_seed1_runC.nc",
}

OUT_DIR = "experiments/exp2_seed_repro/results"
os.makedirs(OUT_DIR, exist_ok=True)

def mae_rmse(pred, truth):
    d = pred - truth
    return float(np.abs(d).mean()), float(np.sqrt((d**2).mean()))

results = {
    "experiment": "exp2_seed_repro",
    "description": "Seed reproducibility test for CorrDiff generation",
    "units": "m/s",
    "runs": {}
}

txt_lines = [
    "Experiment: exp2_seed_repro",
    "Description: Same seed produces identical outputs; different seed produces different output",
    "Units: m/s",
    ""
]

for tag, path in FILES.items():
    truth = xr.open_dataset(path, group="truth")
    pred  = xr.open_dataset(path, group="prediction")
    results["runs"][tag] = {}
    txt_lines.append(f"Run: {tag}")

    for var in ["10u", "10v"]:
        if var not in truth or var not in pred:
            continue
        mae, rmse = mae_rmse(pred[var], truth[var])
        results["runs"][tag][var] = {"mae": mae, "rmse": rmse}
        txt_lines.append(f"  {var}: MAE {mae:.4f}, RMSE {rmse:.4f}")
    txt_lines.append("")

with open(f"{OUT_DIR}/metrics.json", "w") as f:
    json.dump(results, f, indent=2)

with open(f"{OUT_DIR}/metrics.txt", "w") as f:
    f.write("\n".join(txt_lines))

print("✔ exp02 metrics written")
