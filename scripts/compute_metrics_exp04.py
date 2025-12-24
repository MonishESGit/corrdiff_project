#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import xarray as xr


def _open_group(nc_path: str, group: Optional[str]) -> xr.Dataset:
    # xarray: root is group=None
    return xr.open_dataset(nc_path, engine="netcdf4", group=group)


def _select_time(da: xr.DataArray, time_index: Optional[int]) -> xr.DataArray:
    if time_index is None:
        return da
    if "time" in da.dims:
        return da.isel(time=time_index)
    return da


def _detect_ensemble_dim(da: xr.DataArray) -> Optional[str]:
    for d in da.dims:
        if d.lower() in ("ensemble", "ens", "member", "sample", "samples"):
            return d
    return None


def _mae_rmse(truth: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    diff = pred - truth
    mae = float(np.mean(np.abs(diff)))
    rmse = float(math.sqrt(float(np.mean(diff * diff))))
    return mae, rmse


def _compute_var(nc_path: str, var: str, time_index: Optional[int]) -> Dict:
    # required layout:
    # group 'truth' -> var [time,y,x]
    # group 'prediction' -> var [ensemble,time,y,x] OR [time,y,x]
    truth_ds = _open_group(nc_path, "truth")
    pred_ds = _open_group(nc_path, "prediction")

    if var not in truth_ds.data_vars:
        raise KeyError(f"Var '{var}' not found in group 'truth'. Vars={list(truth_ds.data_vars)}")
    if var not in pred_ds.data_vars:
        raise KeyError(f"Var '{var}' not found in group 'prediction'. Vars={list(pred_ds.data_vars)}")

    truth_da = _select_time(truth_ds[var], time_index)
    pred_da = _select_time(pred_ds[var], time_index)

    # align if coordinates exist
    try:
        truth_da, pred_da = xr.align(truth_da, pred_da, join="inner")
    except Exception:
        pass

    t = truth_da.values.astype(np.float32)
    ens_dim = _detect_ensemble_dim(pred_da)

    out = {"truth_group": "truth", "pred_group": "prediction", "per_ensemble": [], "ensemble_mean": None}

    if ens_dim is not None:
        p = pred_da.transpose(ens_dim, ...).values.astype(np.float32)  # [E, ...]
        E = p.shape[0]
        for e in range(E):
            mae, rmse = _mae_rmse(t, p[e])
            out["per_ensemble"].append({"ensemble": int(e), "mae": mae, "rmse": rmse})
        p_mean = np.mean(p, axis=0)
        mae_m, rmse_m = _mae_rmse(t, p_mean)
        out["ensemble_mean"] = {"mae": mae_m, "rmse": rmse_m}
    else:
        p = pred_da.values.astype(np.float32)
        mae, rmse = _mae_rmse(t, p)
        out["per_ensemble"].append({"ensemble": 0, "mae": mae, "rmse": rmse})
        out["ensemble_mean"] = {"mae": mae, "rmse": rmse}

    return out


def _write_txt(metrics: Dict, out_path: Path) -> None:
    lines = []
    lines.append(f"Experiment: {metrics['experiment']}")
    lines.append(f"Description: {metrics['description']}")
    if metrics.get("units"):
        lines.append(f"Units: {metrics['units']}")
    if metrics.get("time_index") is not None:
        lines.append(f"Time index: {metrics['time_index']}")
    lines.append("")

    for run_name, run_blob in metrics["runs"].items():
        lines.append(f"Run: {run_name}")
        for var, blob in run_blob.items():
            lines.append(f"  Variable: {var}")
            for pe in blob["per_ensemble"]:
                lines.append(f"    Ensemble {pe['ensemble']}: MAE {pe['mae']:.4f}, RMSE {pe['rmse']:.4f}")
            em = blob["ensemble_mean"]
            lines.append(f"    Ensemble-mean: MAE {em['mae']:.4f}, RMSE {em['rmse']:.4f}")
        lines.append("")
    out_path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nc512", required=True)
    ap.add_argument("--nc2048", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--description", required=True)
    ap.add_argument("--vars", default="10u,10v")
    ap.add_argument("--units", default="")
    ap.add_argument("--time-index", type=int, default=None)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    vars_list = [v.strip() for v in args.vars.split(",") if v.strip()]

    metrics = {
        "experiment": args.experiment,
        "description": args.description,
        "units": args.units,
        "time_index": args.time_index,
        "runs": {},
    }

    run_map = {"samples_512": args.nc512, "samples_2048": args.nc2048}

    for run_name, nc_path in run_map.items():
        run_blob = {}
        for var in vars_list:
            blob = _compute_var(nc_path, var, args.time_index)
            blob["units"] = args.units
            run_blob[var] = blob
        metrics["runs"][run_name] = run_blob

    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    _write_txt(metrics, outdir / "metrics.txt")

    print(f"Wrote: {outdir/'metrics.json'}")
    print(f"Wrote: {outdir/'metrics.txt'}")


if __name__ == "__main__":
    main()
