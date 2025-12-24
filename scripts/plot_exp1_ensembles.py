#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


def _find_var(ds: xr.Dataset, var: str) -> str:
    """Return actual var name in ds (handles '\\10u' style names)."""
    if var in ds.variables:
        return var
    alt = f"\\{var}"
    if alt in ds.variables:
        return alt
    # sometimes vars can be stored as DataArray names without being in .variables list as expected
    candidates = [k for k in ds.data_vars.keys()]
    if var in candidates:
        return var
    if alt in candidates:
        return alt
    raise KeyError(f"Could not find '{var}' or '{alt}'. Available vars: {list(ds.variables.keys())}")


def _plot_ens_grid(pred_ens: np.ndarray, title: str, outpath: Path):
    """
    pred_ens: [E, H, W]
    """
    E = pred_ens.shape[0]
    ncols = min(4, E)
    nrows = int(np.ceil(E / ncols))

    vmin = float(np.nanpercentile(pred_ens, 2))
    vmax = float(np.nanpercentile(pred_ens, 98))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    im = None
    for i in range(nrows * ncols):
        ax = axes[i]
        ax.axis("off")
        if i < E:
            im = ax.imshow(pred_ens[i], vmin=vmin, vmax=vmax)
            ax.set_title(f"ens {i}")

    fig.suptitle(title, fontsize=14)
    if im is not None:
        cbar = fig.colorbar(im, ax=axes.tolist(), fraction=0.025, pad=0.02)
        cbar.set_label("value")

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def _plot_truth_pred_err(truth: np.ndarray, pred: np.ndarray, title: str, outpath: Path):
    """
    truth, pred: [H, W]
    """
    err = np.abs(pred - truth)

    vmin = float(np.nanpercentile(np.stack([truth, pred]), 2))
    vmax = float(np.nanpercentile(np.stack([truth, pred]), 98))
    emin = 0.0
    emax = float(np.nanpercentile(err, 98))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    im0 = axes[0].imshow(truth, vmin=vmin, vmax=vmax)
    axes[0].set_title("Truth")

    im1 = axes[1].imshow(pred, vmin=vmin, vmax=vmax)
    axes[1].set_title("Pred (ensemble-mean)")

    im2 = axes[2].imshow(err, vmin=emin, vmax=emax)
    axes[2].set_title("|Pred - Truth|")

    fig.suptitle(title, fontsize=14)

    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nc", required=True, help="Path to corrdiff_output_exp1_ensembles4.nc")
    ap.add_argument("--outdir", required=True, help="Where to save plots")
    ap.add_argument("--time-index", type=int, default=0, help="Time index to plot (default 0)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Open groups explicitly (this is the key fix)
    ds_root = xr.open_dataset(args.nc, engine="netcdf4")
    ds_truth = xr.open_dataset(args.nc, engine="netcdf4", group="truth")
    ds_pred = xr.open_dataset(args.nc, engine="netcdf4", group="prediction")

    # time index sanity
    t = int(args.time_index)
    if "time" in ds_truth.dims:
        if t < 0 or t >= ds_truth.sizes["time"]:
            raise ValueError(f"time-index {t} out of range. time size={ds_truth.sizes['time']}")

    for var in ["10u", "10v"]:
        truth_name = _find_var(ds_truth, var)
        pred_name = _find_var(ds_pred, var)

        truth_da = ds_truth[truth_name]          # (time, y, x)
        pred_da = ds_pred[pred_name]             # (ensemble, time, y, x)

        truth = truth_da.isel(time=t).values.astype(np.float32)            # [H,W]
        pred_ens = pred_da.isel(time=t).values.astype(np.float32)          # [E,H,W]
        pred_mean = np.nanmean(pred_ens, axis=0)                           # [H,W]

        # Ensemble grid
        _plot_ens_grid(
            pred_ens,
            title=f"exp1 ensembles: {var} (time_idx={t})",
            outpath=outdir / f"exp1_{var}_ensembles_grid.png",
        )

        # Truth vs pred(mean) vs error
        _plot_truth_pred_err(
            truth,
            pred_mean,
            title=f"exp1 truth vs pred(mean) vs error: {var} (time_idx={t})",
            outpath=outdir / f"exp1_{var}_truth_pred_error.png",
        )

    # close datasets
    ds_root.close()
    ds_truth.close()
    ds_pred.close()

    print(f"Saved exp1 plots to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
