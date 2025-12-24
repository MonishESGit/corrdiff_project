#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


# -----------------------------
# Log parsing (training curves)
# -----------------------------

SAMPLE_RE = re.compile(
    r"samples\s+(?P<samples>[0-9]+(?:\.[0-9]+)?)\s+training_loss\s+(?P<loss>[-+0-9.eE]+)"
)

def parse_training_curve(log_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Parse lines like:
       [..] samples 500.0 training_loss 1363.40 ...
    Returns: samples[], loss[]
    """
    xs: List[float] = []
    ys: List[float] = []
    with open(log_path, "r", errors="ignore") as f:
        for line in f:
            m = SAMPLE_RE.search(line)
            if not m:
                continue
            xs.append(float(m.group("samples")))
            ys.append(float(m.group("loss")))
    return np.array(xs, dtype=float), np.array(ys, dtype=float)


def plot_loss_curves(curves: Dict[str, Tuple[np.ndarray, np.ndarray]], title: str, outpath: Path) -> None:
    plt.figure()
    for label, (x, y) in curves.items():
        if len(x) == 0:
            continue
        plt.plot(x, y, label=label)
    plt.title(title)
    plt.xlabel("samples")
    plt.ylabel("training_loss")
    if len(curves) > 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# -----------------------------
# NetCDF loading + plotting
# -----------------------------

def open_group(nc_path: str, group: Optional[str]) -> xr.Dataset:
    return xr.open_dataset(nc_path, engine="netcdf4", group=group)

def detect_ensemble_dim(da: xr.DataArray) -> Optional[str]:
    for d in da.dims:
        if d.lower() in ("ensemble", "ens", "member", "sample", "samples"):
            return d
    return None

def select_time(da: xr.DataArray, time_index: int) -> xr.DataArray:
    if "time" in da.dims:
        return da.isel(time=time_index)
    return da

def get_truth_pred(nc_path: str, var: str, time_index: int) -> Tuple[np.ndarray, np.ndarray]:
    truth = open_group(nc_path, "truth")[var]
    pred = open_group(nc_path, "prediction")[var]

    truth = select_time(truth, time_index)
    pred = select_time(pred, time_index)

    # ensemble mean if needed
    ens_dim = detect_ensemble_dim(pred)
    if ens_dim is not None:
        pred = pred.mean(dim=ens_dim)

    # align if possible
    try:
        truth, pred = xr.align(truth, pred, join="inner")
    except Exception:
        pass

    t = truth.values.astype(np.float32)
    p = pred.values.astype(np.float32)
    return t, p

def plot_truth_pred_error(
    truth: np.ndarray,
    pred: np.ndarray,
    title_prefix: str,
    outpath: Path,
) -> None:
    err = pred - truth

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(truth)
    plt.title(f"{title_prefix}\nTruth")
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.subplot(1, 3, 2)
    plt.imshow(pred)
    plt.title(f"{title_prefix}\nPrediction")
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.subplot(1, 3, 3)
    plt.imshow(err)
    plt.title(f"{title_prefix}\nError (pred-truth)")
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# -----------------------------
# Metrics plots
# -----------------------------

def load_metrics(metrics_json_path: str) -> Dict:
    with open(metrics_json_path, "r") as f:
        return json.load(f)

def extract_run_metric(metrics: Dict, run_key: str, var: str, metric: str) -> float:
    """
    Works with the schema we used:
    metrics["runs"][run_key][var]["ensemble_mean"][metric]
    """
    blob = metrics["runs"][run_key][var]
    return float(blob["ensemble_mean"][metric])

def plot_metrics_bar(metrics: Dict, vars_list: List[str], outdir: Path) -> None:
    run_keys = ["samples_512", "samples_2048"]

    for var in vars_list:
        for metric in ["mae", "rmse"]:
            vals = [extract_run_metric(metrics, rk, var, metric) for rk in run_keys]

            plt.figure()
            x = np.arange(len(run_keys))
            plt.bar(x, vals)
            plt.xticks(x, run_keys)
            plt.title(f"{var}: {metric.upper()} vs diffusion training budget")
            plt.ylabel(metric.upper())
            for i, v in enumerate(vals):
                plt.text(i, v, f"{v:.4f}", ha="center", va="bottom")
            plt.tight_layout()
            plt.savefig(outdir / f"exp4_{var}_{metric}_bar.png", dpi=200)
            plt.close()


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expdir", required=True, help="experiments/exp4_training_samples")
    ap.add_argument("--outdir", default=None, help="default: <expdir>/figures")
    ap.add_argument("--metrics-json", default=None, help="default: <expdir>/results/metrics.json")

    ap.add_argument("--nc512", default=None, help="default: <expdir>/results/corrdiff_output_samples512.nc")
    ap.add_argument("--nc2048", default=None, help="default: <expdir>/results/corrdiff_output_samples2048.nc")

    ap.add_argument("--reg-log", default=None, help="optional regression_train.log (fixed regression)")
    ap.add_argument("--diff512-log", default=None, help="optional diffusion_train_512.log")
    ap.add_argument("--diff2048-log", default=None, help="optional diffusion_train_2048.log")

    ap.add_argument("--time-index", type=int, default=0)
    ap.add_argument("--vars", default="10u,10v")
    args = ap.parse_args()

    expdir = Path(args.expdir)
    outdir = Path(args.outdir) if args.outdir else (expdir / "figures")
    outdir.mkdir(parents=True, exist_ok=True)

    metrics_json = args.metrics_json or str(expdir / "results" / "metrics.json")
    nc512 = args.nc512 or str(expdir / "results" / "corrdiff_output_samples512.nc")
    nc2048 = args.nc2048 or str(expdir / "results" / "corrdiff_output_samples2048.nc")

    vars_list = [v.strip() for v in args.vars.split(",") if v.strip()]

    # A) Training loss plots
    # Regression (single curve)
    if args.reg_log:
        x, y = parse_training_curve(args.reg_log)
        plot_loss_curves({"regression": (x, y)}, "Regression training loss (fixed in exp4)", outdir / "exp4_regression_loss.png")

    # Diffusion overlay (512 vs 2048)
    curves = {}
    if args.diff512_log:
        x, y = parse_training_curve(args.diff512_log)
        curves["samples_512"] = (x, y)
    if args.diff2048_log:
        x, y = parse_training_curve(args.diff2048_log)
        curves["samples_2048"] = (x, y)
    if curves:
        plot_loss_curves(curves, "Diffusion training loss: budget sweep", outdir / "exp4_diffusion_loss_overlay.png")

    # B) Metrics plots
    metrics = load_metrics(metrics_json)
    plot_metrics_bar(metrics, vars_list, outdir)

    # C) Truth/Pred/Error maps
    # for each run (512, 2048) and each variable
    run_map = {
        "samples_512": nc512,
        "samples_2048": nc2048,
    }

    for run_name, nc_path in run_map.items():
        for var in vars_list:
            truth, pred = get_truth_pred(nc_path, var, time_index=args.time_index)
            plot_truth_pred_error(
                truth,
                pred,
                title_prefix=f"{run_name} | {var} | t={args.time_index}",
                outpath=outdir / f"exp4_{run_name}_{var}_truth_pred_error_t{args.time_index}.png",
            )

    print(f"Done. Figures written to: {outdir}")


if __name__ == "__main__":
    main()
