import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def plot_history(npz_path: Path, out_dir: Path) -> None:
    data = np.load(npz_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    for key in sorted(data.files):
        values = np.asarray(data[key]).flatten()
        if values.size == 0:
            continue

        x = np.arange(1, len(values) + 1)
        plt.figure(figsize=(8, 4))
        plt.plot(x, values, marker="o", linewidth=1)
        plt.title(key)
        plt.xlabel("Epoch")
        plt.ylabel(key)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        out_path = out_dir / f"{key}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()

    # Also save a combined plot with all metrics
    plt.figure(figsize=(10, 6))
    for key in sorted(data.files):
        values = np.asarray(data[key]).flatten()
        if values.size == 0:
            continue
        x = np.arange(1, len(values) + 1)
        plt.plot(x, values, label=key)
    plt.title("Training History")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "training_history_all.png", dpi=150)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot all metrics from training_history.npz")
    parser.add_argument(
        "--npz",
        type=Path,
        default=Path("training_history.npz"),
        help="Path to training_history.npz",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("training_history_plots"),
        help="Output directory for plots",
    )
    args = parser.parse_args()

    if not args.npz.exists():
        raise FileNotFoundError(f"NPZ file not found: {args.npz}")

    plot_history(args.npz, args.out)
    print(f"Saved plots to: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
