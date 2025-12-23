import xarray as xr
import numpy as np
import json
import os

PATH = "experiments/exp5_output_2t-tp/results/corrdiff_output_exp5.nc"
OUT_DIR = "experiments/exp5_output_2t-tp/results"
os.makedirs(OUT_DIR, exist_ok=True)

# Optional: you can edit these if you know them precisely
UNITS = {
    "2t": "K",          # 2m temperature is usually Kelvin in HRRR-style datasets
    "tp": "unknown"     # total precipitation varies by dataset; leave unknown if unsure
}

def mae_rmse(pred, truth):
    d = pred - truth
    return float(np.abs(d).mean()), float(np.sqrt((d**2).mean()))

truth = xr.open_dataset(PATH, group="truth")
pred  = xr.open_dataset(PATH, group="prediction")

results = {
    "experiment": "exp05_output_2t_tp",
    "description": "Output variable change: train+generate CorrDiff with output_variables=['2t','tp']",
    "runs": {"2t_tp": {}},
}

lines = []
lines.append("Experiment: exp05_output_2t_tp")
lines.append("Description: output_variables = ['2t','tp']")
lines.append("")

for var in ["2t", "tp"]:
    if var not in truth or var not in pred:
        raise RuntimeError(f"{var} not found in truth/prediction groups. Check output_variables config.")

    mae, rmse = mae_rmse(pred[var], truth[var])
    results["runs"]["2t_tp"][var] = {
        "mae": mae,
        "rmse": rmse,
        "units": UNITS.get(var, "unknown"),
    }

    lines.append(f"Variable: {var}  Units: {UNITS.get(var, 'unknown')}")
    lines.append(f"  MAE : {mae:.4f}")
    lines.append(f"  RMSE: {rmse:.4f}")
    lines.append("")

with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
    json.dump(results, f, indent=2)

with open(os.path.join(OUT_DIR, "metrics.txt"), "w") as f:
    f.write("\n".join(lines))

print("✔ exp05 metrics written:")
print(f"  {OUT_DIR}/metrics.txt")
print(f"  {OUT_DIR}/metrics.json")
