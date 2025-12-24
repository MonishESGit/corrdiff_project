#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


def open_truth_pred(nc_path: str):
    truth = xr.open_dataset(nc_path, group="truth")
    pred = xr.open_dataset(nc_path, group="prediction")
    return truth, pred


def get_pred_field(pred_ds, var: str, time_idx: int):
    arr = pred_ds[var].values
    # pred could be [ensemble,time,y,x] or [time,y,x]
    if arr.ndim == 4:
        # For exp2, keep ensembles=1 ideally. If >1, use ensemble-mean.
        arr = np.mean(arr, axis=0)
    return arr[time_idx]


def get_truth_field(truth_ds, var: str, time_idx: int):
    return truth_ds[var].values[time_idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ncA", required=True)
    ap.add_argument("--ncB", required=True)
    ap.add_argument("--ncC", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--time-index", type=int, default=0)
    ap.add_argument("--vars", default="10u,10v")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    vars_ = [v.strip() for v in args.vars.split(",") if v.strip()]

    truthA, predA = open_truth_pred(args.ncA)
    truthB, predB = open_truth_pred(args.ncB)
    truthC, predC = open_truth_pred(args.ncC)

    # truth should be identical across runs; we use A.
    for v in vars_:
        T = get_truth_field(truthA, v, args.time_index)
        PA = get_pred_field(predA, v, args.time_index)
        PB = get_pred_field(predB, v, args.time_index)
        PC = get_pred_field(predC, v, args.time_index)

        abs_A_B = np.abs(PA - PB)
        abs_A_C = np.abs(PA - PC)

        errA = np.abs(PA - T)
        errC = np.abs(PC - T)
        err_diff = errA - errC  # positive means C is better in that region

        fig, axes = plt.subplots(2, 3, figsize=(14, 8))

        im0 = axes[0, 0].imshow(T)
        axes[0, 0].set_title(f"Truth: {v}")
        plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

        im1 = axes[0, 1].imshow(PA)
        axes[0, 1].set_title(f"Pred runA (seed0): {v}")
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

        im2 = axes[0, 2].imshow(PC)
        axes[0, 2].set_title(f"Pred runC (seed1): {v}")
        plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

        im3 = axes[1, 0].imshow(abs_A_B)
        axes[1, 0].set_title("|PredA - PredB| (same seed)")
        plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)

        im4 = axes[1, 1].imshow(abs_A_C)
        axes[1, 1].set_title("|PredA - PredC| (diff seed)")
        plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)

        im5 = axes[1, 2].imshow(err_diff)
        axes[1, 2].set_title("|A-T| - |C-T| (positive => seed1 better)")
        plt.colorbar(im5, ax=axes[1, 2], fraction=0.046, pad=0.04)

        for ax in axes.ravel():
            ax.set_xticks([])
            ax.set_yticks([])

        fig.suptitle(f"exp2 seed reproducibility: {v} (time_idx={args.time_index})", fontsize=14)
        fig.tight_layout()

        out_path = outdir / f"exp2_seed_repro_{v}_time{args.time_index}.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
