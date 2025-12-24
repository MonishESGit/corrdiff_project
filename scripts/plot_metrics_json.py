import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt


def main():
    """
    Usage:
      python scripts/plot_metrics_json.py <metrics.json> <outdir> <var>
    """
    if len(sys.argv) < 4:
        print("Usage: python scripts/plot_metrics_json.py <metrics.json> <outdir> <var>")
        sys.exit(1)

    metrics_path = Path(sys.argv[1])
    outdir = Path(sys.argv[2])
    var = sys.argv[3]

    payload = json.loads(metrics_path.read_text())

    # Support two shapes:
    # (A) our compute_metrics_nc.py format: payload["metrics"]["variables"][var]...
    # (B) your sweep format: payload["runs"][run_name][var]...
    if "runs" in payload:
        runs = payload["runs"]
        labels = list(runs.keys())
        maes = [runs[k][var]["mae"] for k in labels]
        rmses = [runs[k][var]["rmse"] for k in labels]
        title_prefix = payload.get("experiment", metrics_path.parent.name)
    else:
        # Single file metrics, not a sweep. We'll plot only ensemble_mean if exists.
        m = payload["metrics"]["variables"][var]
        title_prefix = payload.get("experiment", metrics_path.parent.name)
        if "ensemble_mean" in m:
            labels = ["ensemble_mean"]
            maes = [m["ensemble_mean"]["mae"]]
            rmses = [m["ensemble_mean"]["rmse"]]
        else:
            labels = ["single"]
            maes = [m["mae"]]
            rmses = [m["rmse"]]

    outdir.mkdir(parents=True, exist_ok=True)

    # MAE
    plt.figure(figsize=(10, 4))
    plt.bar(labels, maes)
    plt.title(f"{title_prefix} — {var} MAE")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(outdir / f"{var}_mae_bar.png", dpi=200)
    plt.close()

    # RMSE
    plt.figure(figsize=(10, 4))
    plt.bar(labels, rmses)
    plt.title(f"{title_prefix} — {var} RMSE")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(outdir / f"{var}_rmse_bar.png", dpi=200)
    plt.close()

    print(f"Saved: {outdir / (var + '_mae_bar.png')}")
    print(f"Saved: {outdir / (var + '_rmse_bar.png')}")


if __name__ == "__main__":
    main()
