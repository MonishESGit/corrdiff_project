import re
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# matches lines like:
# samples 500.0     training_loss 1363.40 ...
PATTERN = re.compile(
    r"samples\s+(?P<samples>[0-9]+(?:\.[0-9]+)?)\s+training_loss\s+(?P<loss>[0-9]+(?:\.[0-9]+)?)"
)

def parse_loss(log_path: Path):
    xs, ys = [], []
    with log_path.open("r", errors="ignore") as f:
        for line in f:
            m = PATTERN.search(line)
            if m:
                xs.append(float(m.group("samples")))
                ys.append(float(m.group("loss")))
    return xs, ys

def plot(log_file: Path, out_png: Path, title: str):
    xs, ys = parse_loss(log_file)
    if len(xs) < 3:
        raise RuntimeError(f"Not enough loss points found in {log_file}.")
    out_png.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(xs, ys)
    plt.xlabel("samples")
    plt.ylabel("training_loss")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    print(f"Saved: {out_png} (points={len(xs)})")

def main():
    if len(sys.argv) < 4:
        print("Usage: python scripts/plot_training_loss.py <log_file> <out_png> <title>")
        sys.exit(1)
    plot(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3])

if __name__ == "__main__":
    main()
