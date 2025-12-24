#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


LOSS_RE = re.compile(
    r"\bsamples\s+([0-9]+(?:\.[0-9]+)?)\s+training_loss\s+([0-9]+(?:\.[0-9]+)?(?:[eE][-+]?\d+)?)"
)

def parse_lr_key(run_key: str) -> float:
    """
    Expects keys like: lr_1e-4, lr_2e-4, lr_1e-3
    Returns float LR for sorting/plotting.
    """
    m = re.search(r"lr_([0-9]+)e-([0-9]+)", run_key)
    if not m:
        raise ValueError(f"Could not parse LR from run key: {run_key}")
    a = int(m.group(1))
    b = int(m.group(2))
    return a * (10 ** (-b))

def lr_label(lr: float) -> str:
    # prefer scientific-style labels that match your filenames
    # 0.0001 -> 1e-4, 0.001 -> 1e-3, 0.0002 -> 2e-4
    if lr == 0:
        return "0"
    exp = int(math.floor(math.log10(abs(lr))))
    mant = lr / (10 ** exp)
    # convert to integer mantissa when clean (e.g., 1.0, 2.0)
    if abs(mant - round(mant)) < 1e-10:
        mant = int(round(mant))
    return f"{mant:g}e{exp:d}"

def extract_loss_curve(log_path: Path):
    """
    Returns two lists: samples, training_loss
    """
    xs, ys = [], []
    if not log_path.exists():
        return xs, ys

    with log_path.open("r", errors="ignore") as f:
        for line in f:
            m = LOSS_RE.search(line)
            if not m:
                continue
            xs.append(float(m.group(1)))
            ys.append(float(m.group(2)))
    return xs, ys

def get_metric_value(var_blob: dict, metric_name: str):
    """
    Supports your schema:
      {"ensemble_mean":{"mae":...,"rmse":...}, "per_ensemble":[...], ...}
    plus fallbacks.
    """
    if var_blob is None:
        return None

    if "ensemble_mean" in var_blob and isinstance(var_blob["ensemble_mean"], dict):
        if metric_name in var_blob["ensemble_mean"]:
            return float(var_blob["ensemble_mean"][metric_name])

    if metric_name in var_blob:
        return float(var_blob[metric_name])

    if "per_ensemble" in var_blob and isinstance(var_blob["per_ensemble"], list) and len(var_blob["per_ensemble"]) > 0:
        pe0 = var_blob["per_ensemble"][0]
        if metric_name in pe0:
            return float(pe0[metric_name])

    return None

def plot_loss_sweep(expdir: Path, outdir: Path, kind: str, run_keys_sorted, lr_map):
    """
    kind: "regression" or "diffusion"
    log filename pattern:
      experiments/exp3_lr/logs/{kind}_train_lr_1e-4.log
    """
    plt.figure()
    missing = 0

    for rk in run_keys_sorted:
        lr = lr_map[rk]
        label = f"lr={lr_label(lr)}"
        log_path = expdir / "logs" / f"{kind}_train_lr_{lr_label(lr)}.log"
        xs, ys = extract_loss_curve(log_path)
        if not xs:
            missing += 1
            continue
        plt.plot(xs, ys, label=label)

    plt.title(f"exp3 {kind}: training loss vs samples (CPU)")
    plt.xlabel("samples")
    plt.ylabel("training_loss")
    plt.legend()
    plt.tight_layout()

    outpath = outdir / f"exp3_{kind}_loss_lr_sweep.png"
    plt.savefig(outpath, dpi=200)
    plt.close()

    if missing == len(run_keys_sorted):
        print(f"[WARN] No {kind} loss curves were found. Check log filenames in {expdir/'logs'}.")

def plot_metric_vs_lr(metrics, outdir: Path, metric_name: str, vars_list, run_keys_sorted, lr_map):
    plt.figure()

    # x-axis: sorted lr values
    xs = [lr_map[rk] for rk in run_keys_sorted]
    xlabels = [lr_label(x) for x in xs]

    # for each var, plot a line
    for var in vars_list:
        ys = []
        for rk in run_keys_sorted:
            var_blob = metrics["runs"][rk].get(var)
            ys.append(get_metric_value(var_blob, metric_name))
        plt.plot(xs, ys, marker="o", label=var)

    plt.title(f"exp3: {metric_name.upper()} vs regression LR (generation fixed)")
    plt.xlabel("learning rate")
    plt.ylabel(metric_name.lower())
    plt.xscale("log")
    plt.xticks(xs, xlabels)
    plt.legend()
    plt.tight_layout()

    outpath = outdir / f"exp3_{metric_name.lower()}_vs_lr.png"
    plt.savefig(outpath, dpi=200)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expdir", required=True, help="Path to experiments/exp3_lr")
    ap.add_argument("--vars", default="10u,10v", help="Comma-separated variables to plot (default: 10u,10v)")
    args = ap.parse_args()

    expdir = Path(args.expdir).expanduser().resolve()
    outdir = expdir / "figures"
    outdir.mkdir(parents=True, exist_ok=True)

    metrics_path = expdir / "results" / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.json not found at: {metrics_path}")

    metrics = json.loads(metrics_path.read_text())
    vars_list = [v.strip() for v in args.vars.split(",") if v.strip()]

    # sort runs by LR extracted from run key
    run_keys = list(metrics["runs"].keys())
    lr_map = {rk: parse_lr_key(rk) for rk in run_keys}
    run_keys_sorted = sorted(run_keys, key=lambda rk: lr_map[rk])

    # plots
    plot_loss_sweep(expdir, outdir, "regression", run_keys_sorted, lr_map)
    plot_loss_sweep(expdir, outdir, "diffusion", run_keys_sorted, lr_map)
    plot_metric_vs_lr(metrics, outdir, "mae", vars_list, run_keys_sorted, lr_map)
    plot_metric_vs_lr(metrics, outdir, "rmse", vars_list, run_keys_sorted, lr_map)

    print("[OK] Wrote plots to:", outdir)

if __name__ == "__main__":
    main()
