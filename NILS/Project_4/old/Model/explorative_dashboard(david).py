import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class RunInfo:
    model: str
    run_id: int
    npz_path: Path
    config_path: Path


def parse_runs(data_dir: Path) -> List[RunInfo]:
    pattern = re.compile(r"^(?P<model>[A-Za-z]+)(?P<run>\d+)training_history\.npz$")
    runs: List[RunInfo] = []
    for path in sorted(data_dir.glob("*training_history.npz")):
        match = pattern.match(path.name)
        if not match:
            continue
        model = match.group("model")
        run_id = int(match.group("run"))
        if run_id < 6 or run_id > 10:
            continue
        config_path = data_dir / f"{model}{run_id}model_config.txt"
        runs.append(RunInfo(model=model, run_id=run_id, npz_path=path, config_path=config_path))
    return runs


def load_metrics(npz_path: Path) -> Dict[str, np.ndarray]:
    data = np.load(npz_path)
    metrics: Dict[str, np.ndarray] = {}
    for key in data.files:
        values = np.asarray(data[key], dtype=float).flatten()
        values[~np.isfinite(values)] = np.nan
        metrics[key] = values
    metrics = convert_confusion_counts_to_percent(metrics)
    return metrics


def convert_confusion_counts_to_percent(metrics: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    count_keys = {"tp", "tn", "fp", "fn"}
    if not count_keys.issubset(metrics.keys()):
        return metrics
    tp = metrics["tp"]
    tn = metrics["tn"]
    fp = metrics["fp"]
    fn = metrics["fn"]
    first_total = (
        first_finite(tp)
        + first_finite(tn)
        + first_finite(fp)
        + first_finite(fn)
    )
    safe_total = np.nan if not np.isfinite(first_total) or first_total == 0 else float(first_total)
    for key in count_keys:
        metrics[f"{key}_count"] = metrics[key]
        metrics[key] = metrics[key] / safe_total
    return metrics


def first_finite(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    finite_mask = np.isfinite(values)
    if not np.any(finite_mask):
        return float("nan")
    return float(values[finite_mask][0])


def aggregate_metric(values: np.ndarray, metric: str, mode: str) -> float:
    if values.size == 0:
        return float("nan")
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return float("nan")
    if mode == "last":
        return float(finite_values[-1])
    minimize = {"loss", "fp", "fn", "fp_count", "fn_count"}
    if metric in minimize:
        return float(np.nanmin(finite_values))
    return float(np.nanmax(finite_values))


def make_boxplots(
    out_dir: Path,
    all_runs: List[RunInfo],
    metrics_by_run: Dict[Path, Dict[str, np.ndarray]],
    aggregate: str,
) -> List[str]:
    metrics = sorted({k for m in metrics_by_run.values() for k in m.keys()})
    created: List[str] = []
    for metric in metrics:
        data_by_model: Dict[str, List[float]] = {}
        for run in all_runs:
            values = metrics_by_run[run.npz_path].get(metric, np.array([]))
            score = aggregate_metric(values, metric, aggregate)
            if np.isnan(score):
                continue
            data_by_model.setdefault(run.model, []).append(score)
        if not data_by_model:
            continue
        labels = sorted(data_by_model.keys())
        data = [data_by_model[label] for label in labels]
        plt.figure(figsize=(8, 4))
        plt.boxplot(data, labels=labels, showmeans=True)
        plt.title(f"{metric} by model ({aggregate})")
        plt.xlabel("Model")
        plt.ylabel(metric)
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        out_path = out_dir / f"boxplot_{metric}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        created.append(out_path.name)
    return created


def make_mean_curves(
    out_dir: Path,
    all_runs: List[RunInfo],
    metrics_by_run: Dict[Path, Dict[str, np.ndarray]],
) -> List[str]:
    metrics = sorted({k for m in metrics_by_run.values() for k in m.keys()})
    created: List[str] = []
    for metric in metrics:
        plt.figure(figsize=(8, 4))
        plotted = False
        for model in sorted({r.model for r in all_runs}):
            series_list: List[np.ndarray] = []
            for run in all_runs:
                if run.model != model:
                    continue
                values = metrics_by_run[run.npz_path].get(metric, np.array([]))
                if values.size == 0:
                    continue
                series_list.append(values)
            if not series_list:
                continue
            max_len = max(len(v) for v in series_list)
            if max_len == 0:
                continue
            stacked = np.full((len(series_list), max_len), np.nan, dtype=float)
            for idx, series in enumerate(series_list):
                stacked[idx, : len(series)] = series
            mean = np.nanmean(stacked, axis=0)
            std = np.nanstd(stacked, axis=0)
            x = np.arange(1, max_len + 1)
            plt.plot(x, mean, label=f"{model} mean")
            plt.fill_between(x, mean - std, mean + std, alpha=0.2)
            plotted = True
        if not plotted:
            plt.close()
            continue
        plt.title(f"{metric} mean +/- std")
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        out_path = out_dir / f"mean_{metric}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        created.append(out_path.name)
    return created


def write_summary(out_dir: Path, all_runs: List[RunInfo], metrics_by_run: Dict[Path, Dict[str, np.ndarray]]) -> str:
    summary_lines = []
    for run in all_runs:
        metrics = metrics_by_run[run.npz_path]
        metric_desc = ", ".join(f"{k}:{len(v)}" for k, v in sorted(metrics.items()))
        summary_lines.append(f"{run.model}{run.run_id} | {run.npz_path.name} | {metric_desc}")
    summary_text = "\n".join(summary_lines)
    summary_path = out_dir / "summary.txt"
    summary_path.write_text(summary_text)
    return summary_path.name


def build_html(
    out_dir: Path,
    all_runs: List[RunInfo],
    boxplots: List[str],
    mean_curves: List[str],
    summary_file: str,
) -> None:
    models = sorted({r.model for r in all_runs})
    runs = sorted(all_runs, key=lambda r: (r.model, r.run_id))
    html_lines = [
        "<!doctype html>",
        "<html lang=\"en\">",
        "<head>",
        "  <meta charset=\"utf-8\">",
        "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">",
        "  <title>Training History Dashboard</title>",
        "  <style>",
        "    body { font-family: Georgia, serif; margin: 24px; color: #1b1b1b; background: #f7f3ee; }",
        "    h1 { margin-bottom: 8px; }",
        "    .section { margin-top: 28px; padding: 16px; background: #ffffff; border: 1px solid #e6dccf; border-radius: 8px; }",
        "    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }",
        "    img { width: 100%; height: auto; border: 1px solid #e6dccf; border-radius: 6px; }",
        "    table { border-collapse: collapse; width: 100%; }",
        "    th, td { border: 1px solid #e6dccf; padding: 6px 8px; text-align: left; }",
        "    th { background: #efe6da; }",
        "    code { background: #efe6da; padding: 1px 4px; border-radius: 4px; }",
        "  </style>",
        "</head>",
        "<body>",
        "  <h1>Training History Dashboard</h1>",
        "  <p>Models: " + ", ".join(models) + "</p>",
        "  <div class=\"section\">",
        "    <h2>Run Overview</h2>",
        "    <table>",
        "      <tr><th>Model</th><th>Run</th><th>NPZ</th><th>Config</th></tr>",
    ]
    for run in runs:
        config = run.config_path.name if run.config_path.exists() else "(missing)"
        html_lines.append(
            f"      <tr><td>{run.model}</td><td>{run.run_id}</td><td>{run.npz_path.name}</td><td>{config}</td></tr>"
        )
    html_lines += [
        "    </table>",
        "  </div>",
        "  <div class=\"section\">",
        "    <h2>NPZ Summary</h2>",
        f"    <p>See <code>{summary_file}</code> for metric lengths per run.</p>",
        "  </div>",
        "  <div class=\"section\">",
        "    <h2>Boxplots (model comparison)</h2>",
        "    <div class=\"grid\">",
    ]
    for name in boxplots:
        html_lines.append(f"      <div><img src=\"{name}\" alt=\"{name}\"></div>")
    html_lines += [
        "    </div>",
        "  </div>",
        "  <div class=\"section\">",
        "    <h2>Mean Curves (per metric)</h2>",
        "    <div class=\"grid\">",
    ]
    for name in mean_curves:
        html_lines.append(f"      <div><img src=\"{name}\" alt=\"{name}\"></div>")
    html_lines += [
        "    </div>",
        "  </div>",
        "</body>",
        "</html>",
    ]
    (out_dir / "index.html").write_text("\n".join(html_lines))


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an exploratory dashboard from training_history.npz files.")
    parser.add_argument("--data-dir", type=Path, default=Path("/root/ASS4/files"))
    parser.add_argument("--out-dir", type=Path, default=Path("/root/ASS4/dashboard"))
    parser.add_argument("--aggregate", choices=["last", "best"], default="last")
    args = parser.parse_args()

    runs = parse_runs(args.data_dir)
    if not runs:
        raise FileNotFoundError("No training_history.npz files found with expected naming.")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    metrics_by_run: Dict[Path, Dict[str, np.ndarray]] = {}
    for run in runs:
        metrics_by_run[run.npz_path] = load_metrics(run.npz_path)

    summary_file = write_summary(args.out_dir, runs, metrics_by_run)
    boxplots = make_boxplots(args.out_dir, runs, metrics_by_run, args.aggregate)
    mean_curves = make_mean_curves(args.out_dir, runs, metrics_by_run)
    build_html(args.out_dir, runs, boxplots, mean_curves, summary_file)

    print(f"Dashboard written to: {args.out_dir / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
