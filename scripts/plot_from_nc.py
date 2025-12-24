import sys
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


def _np(da: xr.DataArray) -> np.ndarray:
    return np.asarray(da.values, dtype=np.float32)


def _pick_pred(pred_da: xr.DataArray) -> xr.DataArray:
    # If ensembles exist, use ensemble mean for "main" prediction
    if "ensemble" in pred_da.dims:
        return pred_da.mean("ensemble")
    return pred_da


def plot_truth_pred_error(nc_path: Path, var: str, time_idx: int, out_png: Path) -> None:
    truth = xr.open_dataset(nc_path, group="truth")
    pred = xr.open_dataset(nc_path, group="prediction")

    t = _np(truth[var].isel(time=time_idx))
    p = _np(_pick_pred(pred[var]).isel(time=time_idx))
    err = np.abs(p - t)

    plt.figure(figsize=(12, 4))

    ax1 = plt.subplot(1, 3, 1)
    im1 = ax1.imshow(t)
    ax1.set_title(f"Truth: {var}")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = plt.subplot(1, 3, 2)
    im2 = ax2.imshow(p)
    ax2.set_title(f"Pred (ens-mean if applicable): {var}")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    ax3 = plt.subplot(1, 3, 3)
    im3 = ax3.imshow(err)
    ax3.set_title("|Pred - Truth|")
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    for ax in (ax1, ax2, ax3):
        ax.set_xticks([])
        ax.set_yticks([])

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    truth.close()
    pred.close()
    print(f"Saved: {out_png}")


def plot_ensemble_spread(nc_path: Path, var: str, time_idx: int, out_png: Path) -> None:
    pred = xr.open_dataset(nc_path, group="prediction")
    if "ensemble" not in pred[var].dims:
        print(f"Skip spread: {var} has no ensemble dimension in {nc_path}")
        return

    spread = _np(pred[var].isel(time=time_idx).std("ensemble"))

    plt.figure(figsize=(5, 4))
    im = plt.imshow(spread)
    plt.title(f"Ensemble spread (std): {var}")
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(im, fraction=0.046, pad=0.04)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    pred.close()
    print(f"Saved: {out_png}")


def plot_error_histogram(nc_path: Path, var: str, time_idx: int, out_png: Path, max_points: int = 200_000) -> None:
    truth = xr.open_dataset(nc_path, group="truth")
    pred = xr.open_dataset(nc_path, group="prediction")

    t = _np(truth[var].isel(time=time_idx)).ravel()
    p = _np(_pick_pred(pred[var]).isel(time=time_idx)).ravel()
    err = (p - t)

    if err.size > max_points:
        idx = np.random.default_rng(0).choice(err.size, size=max_points, replace=False)
        err = err[idx]

    plt.figure(figsize=(6, 4))
    plt.hist(err, bins=60)
    plt.title(f"Error histogram (pred - truth): {var}")
    plt.xlabel("error")
    plt.ylabel("count")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    truth.close()
    pred.close()
    print(f"Saved: {out_png}")


def plot_truth_vs_pred_scatter(nc_path: Path, var: str, time_idx: int, out_png: Path, max_points: int = 50_000) -> None:
    truth = xr.open_dataset(nc_path, group="truth")
    pred = xr.open_dataset(nc_path, group="prediction")

    t = _np(truth[var].isel(time=time_idx)).ravel()
    p = _np(_pick_pred(pred[var]).isel(time=time_idx)).ravel()

    n = t.size
    if n > max_points:
        idx = np.random.default_rng(0).choice(n, size=max_points, replace=False)
        t = t[idx]
        p = p[idx]

    plt.figure(figsize=(5, 5))
    plt.scatter(t, p, s=2, alpha=0.3)
    plt.title(f"Truth vs Pred (sampled): {var}")
    plt.xlabel("truth")
    plt.ylabel("pred")

    # y=x line for reference (computed from data range)
    lo = float(min(t.min(), p.min()))
    hi = float(max(t.max(), p.max()))
    plt.plot([lo, hi], [lo, hi])

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    truth.close()
    pred.close()
    print(f"Saved: {out_png}")


def main():
    """
    Examples:
      python scripts/plot_from_nc.py --nc <file.nc> --var 10u --time-idx 0 --outdir figures/exp01 --all
    """
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--nc", required=True)
    p.add_argument("--var", required=True)
    p.add_argument("--time-idx", type=int, default=0)
    p.add_argument("--outdir", required=True)
    p.add_argument("--all", action="store_true")
    p.add_argument("--only", choices=["tpe", "spread", "hist", "scatter"], default=None)
    args = p.parse_args()

    nc_path = Path(args.nc)
    outdir = Path(args.outdir)
    var = args.var
    t = args.time_idx

    if args.only in (None, "tpe") and (args.all or args.only == "tpe"):
        plot_truth_pred_error(nc_path, var, t, outdir / f"{var}_truth_pred_error.png")
    if args.only in (None, "spread") and (args.all or args.only == "spread"):
        plot_ensemble_spread(nc_path, var, t, outdir / f"{var}_ensemble_spread.png")
    if args.only in (None, "hist") and (args.all or args.only == "hist"):
        plot_error_histogram(nc_path, var, t, outdir / f"{var}_error_hist.png")
    if args.only in (None, "scatter") and (args.all or args.only == "scatter"):
        plot_truth_vs_pred_scatter(nc_path, var, t, outdir / f"{var}_truth_vs_pred_scatter.png")


if __name__ == "__main__":
    main()
