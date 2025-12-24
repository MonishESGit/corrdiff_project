import json
import math
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import xarray as xr


def _to_np(da: xr.DataArray) -> np.ndarray:
    # Ensure float32 to keep memory lower on CPU
    return np.asarray(da.values, dtype=np.float32)


def mae_rmse(pred: np.ndarray, truth: np.ndarray) -> Tuple[float, float]:
    diff = pred - truth
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    return mae, rmse


def get_variables(nc_path: Path) -> List[str]:
    truth = xr.open_dataset(nc_path, group="truth")
    vars_ = list(truth.data_vars.keys())
    truth.close()
    return vars_


def compute_metrics(
    nc_path: Path,
    variables: Optional[List[str]] = None,
    time_idx: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Returns a metrics dict:
      - if ensembles exist: metrics per ensemble + ensemble_mean
      - else: single metrics
    """
    truth_ds = xr.open_dataset(nc_path, group="truth")
    pred_ds = xr.open_dataset(nc_path, group="prediction")

    if variables is None:
        variables = list(truth_ds.data_vars.keys())

    # Determine if ensemble dimension exists for the first variable
    sample_var = variables[0]
    has_ensemble = "ensemble" in pred_ds[sample_var].dims

    out: Dict[str, Any] = {
        "file": str(nc_path),
        "has_ensemble": bool(has_ensemble),
        "variables": {},
    }

    for var in variables:
        truth_da = truth_ds[var]
        pred_da = pred_ds[var]

        if time_idx is not None:
            truth_da = truth_da.isel(time=time_idx)
            pred_da = pred_da.isel(time=time_idx)

        truth_np = _to_np(truth_da)  # shape: (time?, y, x) or (y, x)

        if has_ensemble:
            # pred: (ensemble, time?, y, x)
            ens_dim = pred_da.sizes["ensemble"]
            var_entry: Dict[str, Any] = {"per_ensemble": {}, "ensemble_mean": {}}

            # Per-ensemble
            for e in range(ens_dim):
                pred_e = _to_np(pred_da.isel(ensemble=e))
                m, r = mae_rmse(pred_e, truth_np)
                var_entry["per_ensemble"][str(e)] = {"mae": m, "rmse": r}

            # Ensemble mean
            pred_mean = _to_np(pred_da.mean("ensemble"))
            m_mean, r_mean = mae_rmse(pred_mean, truth_np)
            var_entry["ensemble_mean"] = {"mae": m_mean, "rmse": r_mean}

            out["variables"][var] = var_entry
        else:
            pred_np = _to_np(pred_da)
            m, r = mae_rmse(pred_np, truth_np)
            out["variables"][var] = {"mae": m, "rmse": r}

    truth_ds.close()
    pred_ds.close()
    return out


def write_metrics(metrics: Dict[str, Any], out_json: Path, out_txt: Path, meta: Dict[str, Any]) -> None:
    payload = {
        **meta,
        "metrics": metrics,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))

    # Pretty text
    lines = []
    lines.append(f"Experiment: {meta.get('experiment','')}")
    lines.append(f"Description: {meta.get('description','')}")
    if "units" in meta:
        lines.append(f"Units: {meta['units']}")
    lines.append(f"File: {metrics.get('file','')}")
    lines.append(f"Has ensemble: {metrics.get('has_ensemble')}")
    lines.append("")

    for var, v in metrics["variables"].items():
        lines.append(f"Variable: {var}")
        if isinstance(v, dict) and "per_ensemble" in v:
            per = v["per_ensemble"]
            for e, mr in per.items():
                lines.append(f"  Ensemble {e}: MAE {mr['mae']:.4f}, RMSE {mr['rmse']:.4f}")
            em = v["ensemble_mean"]
            lines.append(f"  Ensemble-mean: MAE {em['mae']:.4f}, RMSE {em['rmse']:.4f}")
        else:
            lines.append(f"  MAE  {v['mae']:.4f}")
            lines.append(f"  RMSE {v['rmse']:.4f}")
        lines.append("")

    out_txt.write_text("\n".join(lines))


def main():
    """
    Usage:
      python scripts/compute_metrics_nc.py \
        --nc experiments/exp01/results/corrdiff_output.nc \
        --experiment exp01_ensembles4 \
        --description "Generation with 4 diffusion ensembles" \
        --units "m/s" \
        --outdir experiments/exp01/results
      Optional:
        --vars 10u,10v
        --time-idx 0
    """
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--nc", required=True)
    p.add_argument("--experiment", required=True)
    p.add_argument("--description", required=True)
    p.add_argument("--units", default=None)
    p.add_argument("--outdir", required=True)
    p.add_argument("--vars", default=None)
    p.add_argument("--time-idx", type=int, default=None)
    args = p.parse_args()

    nc_path = Path(args.nc)
    outdir = Path(args.outdir)
    variables = args.vars.split(",") if args.vars else None

    metrics = compute_metrics(nc_path, variables=variables, time_idx=args.time_idx)
    meta = {
        "experiment": args.experiment,
        "description": args.description,
    }
    if args.units is not None:
        meta["units"] = args.units

    out_json = outdir / "metrics.json"
    out_txt = outdir / "metrics.txt"
    write_metrics(metrics, out_json, out_txt, meta)

    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_txt}")


if __name__ == "__main__":
    main()
