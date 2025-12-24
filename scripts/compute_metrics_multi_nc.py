#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import numpy as np
import xarray as xr


def _open_groups(nc_path: str):
    # CorrDiff writes netCDF4 groups: truth, prediction, input
    # xarray needs group=... to open them.
    base = xr.open_dataset(nc_path)  # lat/lon/time only
    truth = xr.open_dataset(nc_path, group="truth")
    pred = xr.open_dataset(nc_path, group="prediction")
    return base, truth, pred


def _mae_rmse(truth_arr: np.ndarray, pred_arr: np.ndarray):
    diff = pred_arr - truth_arr
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    return mae, rmse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ncs", nargs="+", required=True, help="List of .nc files")
    ap.add_argument("--labels", required=True, help="Comma-separated labels, one per nc")
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--description", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--vars", default="10u,10v", help="Comma-separated output vars")
    args = ap.parse_args()

    labels = [s.strip() for s in args.labels.split(",") if s.strip()]
    if len(labels) != len(args.ncs):
        raise SystemExit(f"--labels count {len(labels)} must match --ncs count {len(args.ncs)}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    vars_ = [v.strip() for v in args.vars.split(",") if v.strip()]

    result = {
        "experiment": args.experiment,
        "description": args.description,
        "runs": {}
    }

    # Units: for wind vars use m/s; otherwise unknown
    def units_for(v):
        if v in ("10u", "10v"):
            return "m/s"
        if v == "2t":
            return "K"
        return "unknown"

    for nc_path, label in zip(args.ncs, labels):
        _, truth, pred = _open_groups(nc_path)
        run_out = {}

        for v in vars_:
            if v not in truth.variables:
                raise SystemExit(f"{nc_path}: truth group missing variable {v}")
            if v not in pred.variables:
                raise SystemExit(f"{nc_path}: prediction group missing variable {v}")

            # truth: [time,y,x]
            t = truth[v].values
            # prediction: [ensemble,time,y,x] OR [time,y,x] depending on config
            p = pred[v].values

            if p.ndim == 4:
                # Compute metrics for each ensemble and ensemble-mean
                per_ens = []
                for e in range(p.shape[0]):
                    mae, rmse = _mae_rmse(t, p[e])
                    per_ens.append({"ensemble": int(e), "mae": mae, "rmse": rmse})
                p_mean = np.mean(p, axis=0)
                mae_m, rmse_m = _mae_rmse(t, p_mean)
                run_out[v] = {
                    "units": units_for(v),
                    "per_ensemble": per_ens,
                    "ensemble_mean": {"mae": mae_m, "rmse": rmse_m},
                }
            else:
                mae, rmse = _mae_rmse(t, p)
                run_out[v] = {"units": units_for(v), "mae": mae, "rmse": rmse}

        result["runs"][label] = run_out

    # Write JSON
    json_path = outdir / "metrics.json"
    json_path.write_text(json.dumps(result, indent=2))

    # Write TXT (human-readable)
    lines = []
    lines.append(f"Experiment: {args.experiment}")
    lines.append(f"Description: {args.description}")
    lines.append("")
    for label, run_out in result["runs"].items():
        lines.append(f"Run: {label}")
        for v in vars_:
            info = run_out[v]
            if "per_ensemble" in info:
                lines.append(f"  {v} ({info['units']}):")
                for pe in info["per_ensemble"]:
                    lines.append(f"    Ensemble {pe['ensemble']}: MAE {pe['mae']:.4f}, RMSE {pe['rmse']:.4f}")
                em = info["ensemble_mean"]
                lines.append(f"    Ensemble-mean: MAE {em['mae']:.4f}, RMSE {em['rmse']:.4f}")
            else:
                lines.append(f"  {v} ({info['units']}): MAE {info['mae']:.4f}, RMSE {info['rmse']:.4f}")
        lines.append("")

    (outdir / "metrics.txt").write_text("\n".join(lines))

    print(f"Wrote: {json_path}")
    print(f"Wrote: {outdir / 'metrics.txt'}")


if __name__ == "__main__":
    main()
