#!/usr/bin/env python3
"""
Combined data creation script that performs the three steps from
make_data_step1.py, make_data_step2.py and make_data_step3.py.

Usage:
    ./makefile all        # run step1, step2, step3 in order
    ./makefile step1 ...  # only convert CSVs -> npz
    ./makefile step2 ...  # only add labels to existing npz files
    ./makefile step3 ...  # only update /npz_dir/index.csv

This script fixes the RuntimeWarning from np.nanmedian when a column is
entirely NaN by silencing the invalid warning and substituting the
specified fill value for such medians.
"""

from __future__ import annotations
import argparse
import json
import os
import glob
import time
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import warnings
import numpy as np
import pandas as pd


def load_labels(labels_path: str = "B/labels.csv") -> dict:
    labels_map = {}
    if not os.path.exists(labels_path):
        print(f"Warning: Labels file {labels_path} not found. Labels will be left blank.")
        return labels_map
    labels_df = pd.read_csv(labels_path, header=0, dtype=str)
    if labels_df.shape[1] < 2:
        return labels_map
    pids = labels_df.iloc[:, 0].astype(str).str.strip().values
    labs = pd.to_numeric(labels_df.iloc[:, 1], errors="coerce")
    valid = ~labs.isna()
    return {pid.strip(): int(lab) for pid, lab in zip(pids[valid], labs[valid].astype(int))}


def median_impute(arr: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    """Fill NaNs using column median. If an entire column is NaN,
    the median becomes `fill_value` (no RuntimeWarning is emitted).
    """
    # np.nanmedian emits a RuntimeWarning when a column is all-NaN. Silence
    # that and replace NaN medians with fill_value.
    with np.errstate(invalid='ignore'):
        med = np.nanmedian(arr, axis=0)
    med = np.where(np.isnan(med), fill_value, med)
    out = np.where(np.isnan(arr), med[None, :], arr)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        med = np.nanmedian(arr, axis=0)
    return out.astype(np.float32)


def convert_one(csv_path: str, out_dir: str, labels_map: dict | None = None,
                expected_cols: int = 40, compress: bool = True) -> dict:
    try:
        base = os.path.basename(csv_path)
        name, _ = os.path.splitext(base)
        pid = name
        pid_short = pid[2:] if pid.startswith("p1") else pid

        # Read CSV robustly: keep empty strings, then convert to NaN
        df = pd.read_csv(csv_path, header=None, dtype=str)
        # strip strings and keep empty strings so we can convert them to NaN
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        df = df.replace("", np.nan)
        df = df.apply(pd.to_numeric, errors="coerce")
        arr = df.values.astype(np.float32)

        # ensure expected columns
        if arr.shape[1] < expected_cols:
            pad = np.full((arr.shape[0], expected_cols - arr.shape[1]), np.nan, dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=1)
        elif arr.shape[1] > expected_cols:
            arr = arr[:, -expected_cols:]

        length = int(arr.shape[0])
        cols = [str(i) for i in range(arr.shape[1])]

        data_median = median_impute(arr, fill_value=0.0)

        label = None
        if labels_map is not None:
            if pid_short in labels_map:
                label = int(labels_map[pid_short])
            elif pid in labels_map:
                label = int(labels_map[pid])

        meta = {
            "source_file": base,
            "pid": pid,
            "pid_short": pid_short,
            "rows": length,
            "cols": arr.shape[1],
            "created": time.time()
        }
        if label is not None:
            meta["label"] = int(label)

        out_name = f"{pid}.npz"
        out_path = os.path.join(out_dir, out_name)

        save_dict = {
            "data_raw": arr,
            "length": np.int32(length),
            "cols": np.array(cols, dtype=object),
            "data_median": data_median,
            "pid": np.array(pid, dtype=object),
            "pid_short": np.array(pid_short, dtype=object),
            "meta": json.dumps(meta)
        }
        if label is not None:
            save_dict["label"] = np.int32(label)

        if compress:
            np.savez_compressed(out_path, **save_dict)
        else:
            np.savez(out_path, **save_dict)

        index_entry = {
            "pid": pid,
            "pid_short": pid_short,
            "npz_path": out_path,
            "rows": length,
            "cols": arr.shape[1],
            "label": label if label is not None else "",
        }
        return {"status": "ok", "entry": index_entry}
    except Exception as e:
        return {"status": "error", "path": csv_path, "error": str(e)}


def convert_all(csv_dir: str = "B/time_series", out_dir: str = "npz_dir", labels_path: str = "B/labels.csv",
                expected_cols: int = 40, workers: int | None = None, compress: bool = True, limit: int | None = None) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    labels_map = load_labels(labels_path)
    csv_files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    if limit is not None:
        csv_files = csv_files[:limit]
    n = len(csv_files)
    print(f"Found {n} CSV files in {csv_dir}. Converting to {out_dir} with {workers or 'auto'} workers.")

    if workers is None:
        workers = max(1, multiprocessing.cpu_count() - 1)

    index_rows = []
    errors = []

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(convert_one, p, out_dir, labels_map, expected_cols, compress): p for p in csv_files}
        for fut in as_completed(futures):
            res = fut.result()
            if res["status"] == "ok":
                index_rows.append(res["entry"])
            else:
                errors.append(res)
                print(f"Error converting {res.get('path')}: {res.get('error')}")

    index_path = os.path.join(out_dir, "index.csv")
    with open(index_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pid", "pid_short", "npz_path", "rows", "cols", "label"])
        writer.writeheader()
        for r in index_rows:
            writer.writerow(r)

    print(f"Conversion finished. {len(index_rows)} files converted, {len(errors)} errors.")
    if errors:
        err_path = os.path.join(out_dir, "errors.json")
        with open(err_path, "w") as ef:
            json.dump(errors, ef, indent=2)
        print(f"Errors written to {err_path}")
    print(f"Index written to {index_path}")
    return {"converted": len(index_rows), "errors": len(errors), "index": index_path}


def add_labels_to_npz(npz_dir: str = "npz_dir", labels_csv: str = "B/labels.csv") -> None:
    labmap = load_labels(labels_csv)
    for path in sorted(glob.glob(os.path.join(npz_dir, "*.npz"))):
        try:
            d = dict(np.load(path, allow_pickle=True))
        except Exception as e:
            print(f"Failed to read {path}: {e}")
            continue

        pid = None
        pid_short = None
        try:
            if "pid" in d:
                pid = d["pid"].item() if hasattr(d["pid"], 'item') else str(d["pid"])
        except Exception:
            pid = str(d.get("pid"))
        try:
            if "pid_short" in d:
                pid_short = d["pid_short"].item() if hasattr(d["pid_short"], 'item') else str(d["pid_short"])
        except Exception:
            pid_short = str(d.get("pid_short"))

        label = None
        candidates = [pid, pid_short, pid.lower() if pid else None, pid_short.lower() if pid_short else None]
        if pid and not pid.startswith("p1"):
            candidates += ["p1" + pid, ("p1" + pid).lower()]
        if pid and pid.startswith("p1"):
            candidates += [pid[2:], pid[2:].lower()]

        for c in candidates:
            if c and c in labmap:
                label = np.int32(labmap[c])
                break

        if label is not None:
            d["label"] = label
            # Overwrite the file (preserve compression)
            try:
                np.savez_compressed(path, **d)
                print("Wrote label", int(label), "to", os.path.basename(path), end = "\r")
            except Exception as e:
                print(f"Failed to write {path}: {e}")
        else:
            print("No label for", os.path.basename(path))


def update_index(npz_dir: str = "npz_dir", labels_csv: str = "B/labels.csv") -> None:
    index_path = os.path.join(npz_dir, "index.csv")
    bak = index_path + ".bak"
    if not os.path.exists(index_path):
        print(f"index.csv not found at {index_path}. Nothing to update.")
        return
    if not os.path.exists(bak):
        os.rename(index_path, bak)
        print("Backup created:", bak)
    else:
        print("Backup already exists:", bak)

    idx = pd.read_csv(bak, dtype=str)
    labmap = load_labels(labels_csv)

    def find_label_for_row(npz_path, pid, pid_short):
        # 1) try reading label from npz
        try:
            with np.load(npz_path, allow_pickle=True) as d:
                if "label" in d:
                    return int(np.array(d["label"]).astype(int).item())
        except Exception:
            pass
        # 2) try labmap candidates
        candidates = []
        if pid: candidates += [pid, pid.lower()]
        if pid_short: candidates += [pid_short, pid_short.lower()]
        if pid and not pid.startswith("p1"):
            candidates += ["p1" + pid, ("p1" + pid).lower()]
        if pid and pid.startswith("p1"):
            candidates += [pid[2:], pid[2:].lower()]
        for c in candidates:
            if c and c in labmap:
                return int(labmap[c])
        return None

    if "label" not in idx.columns:
        idx["label"] = ""

    updated = 0
    for i, row in idx.iterrows():
        npz_path = row.get("npz_path")
        pid = row.get("pid")
        pid_short = row.get("pid_short")
        if not isinstance(npz_path, str) or not os.path.exists(npz_path):
            if pid:
                candidate = os.path.join(npz_dir, f"{pid}.npz")
                if os.path.exists(candidate):
                    npz_path = candidate
        label = find_label_for_row(npz_path, pid, pid_short)
        if label is not None:
            idx.at[i, "label"] = int(label)
            updated += 1
        else:
            idx.at[i, "label"] = ""

    print(f"Labels updated for {updated} entries (rest left blank).")
    idx.to_csv(index_path, index=False)
    print("index.csv updated:", index_path)


def main():
    parser = argparse.ArgumentParser(description="Combined data creation script (steps 1-3).")
    sub = parser.add_subparsers(dest="cmd")

    p1 = sub.add_parser("step1", help="Convert CSVs to NPZ files and create index.csv")
    p1.add_argument("--csv_dir", default="B/time_series")
    p1.add_argument("--out_dir", default="npz_dir")
    p1.add_argument("--labels", default="B/labels.csv")
    p1.add_argument("--cols", type=int, default=40)
    p1.add_argument("--workers", type=int, default=None)
    p1.add_argument("--compress", action="store_true")
    p1.add_argument("--limit", type=int, default=None)

    p2 = sub.add_parser("step2", help="Add labels to existing npz files where possible")
    p2.add_argument("--npz_dir", default="npz_dir")
    p2.add_argument("--labels", default="B/labels.csv")

    p3 = sub.add_parser("step3", help="Update index.csv using npz files and labels csv")
    p3.add_argument("--npz_dir", default="npz_dir")
    p3.add_argument("--labels", default="B/labels.csv")

    pall = sub.add_parser("all", help="Run step1, step2 and step3 in sequence")
    pall.add_argument("--csv_dir", default="B/time_series")
    pall.add_argument("--out_dir", default="npz_dir")
    pall.add_argument("--labels", default="B/labels.csv")
    pall.add_argument("--cols", type=int, default=40)
    pall.add_argument("--workers", type=int, default=None)
    pall.add_argument("--compress", action="store_true")
    pall.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()
    if args.cmd == "step1":
        convert_all(csv_dir=args.csv_dir, out_dir=args.out_dir, labels_path=args.labels,
                    expected_cols=args.cols, workers=args.workers, compress=args.compress, limit=args.limit)
    elif args.cmd == "step2":
        add_labels_to_npz(npz_dir=args.npz_dir, labels_csv=args.labels)
    elif args.cmd == "step3":
        update_index(npz_dir=args.npz_dir, labels_csv=args.labels)
    elif args.cmd == "all":
        convert_all(csv_dir=args.csv_dir, out_dir=args.out_dir, labels_path=args.labels,
                    expected_cols=args.cols, workers=args.workers, compress=args.compress, limit=args.limit)
        add_labels_to_npz(npz_dir=args.out_dir, labels_csv=args.labels)
        update_index(npz_dir=args.out_dir, labels_csv=args.labels)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
