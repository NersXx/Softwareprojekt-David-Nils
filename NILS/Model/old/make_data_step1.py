#!/usr/bin/env python3
"""
convert_csvs_to_npz_simple.py

Konvertiert CSVs in B/time_series/*.csv -> npz_dir/<pid>.npz
- Leere Strings -> NaN
- data_raw mit NaN
- data_median (optionale Median-Imputation)
- pid und pid_short werden gespeichert
- index.csv wird erzeugt mit: pid, pid_short, npz_path, rows, cols, label
"""

import os
import glob
import json
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import time
import csv

import numpy as np
import pandas as pd

def load_labels(labels_path="B/labels.csv"):
    labels_map = {}
    if not os.path.exists(labels_path):
        print(f"Warnung: Labels Datei {labels_path} nicht gefunden. Labels werden leer gelassen.")
        return labels_map
    with open(labels_path, "r", newline="") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            key = parts[0].strip()
            try:
                val = int(parts[1])
            except ValueError:
                continue
            labels_map[key] = val
    return labels_map

def median_impute(arr, fill_value=0.0):
    med = np.nanmedian(arr, axis=0)
    med = np.where(np.isnan(med), fill_value, med)
    out = np.where(np.isnan(arr), med[None, :], arr)
    return out.astype(np.float32)

def convert_one(csv_path, out_dir, labels_map=None, expected_cols=40, compress=True):
    try:
        base = os.path.basename(csv_path)
        name, _ = os.path.splitext(base)
        pid = name
        pid_short = pid[2:] if pid.startswith("p1") else pid

        # Read CSV robustly: keep empty strings, then convert to NaN
        df = pd.read_csv(csv_path, header=None, dtype=str)
        df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
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

def convert_all(csv_dir="B/time_series", out_dir="npz_dir", labels_path="B/labels.csv",
                expected_cols=40, workers=None, compress=True, limit=None):
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CSV time series to NPZ files (simple, no ffill, no observed).")
    parser.add_argument("--csv_dir", type=str, default="B/time_series", help="Directory with CSV files (default B/time_series)")
    parser.add_argument("--out_dir", type=str, default="npz_dir", help="Output directory for NPZ files")
    parser.add_argument("--labels", type=str, default="B/labels.csv", help="Labels CSV path (default B/labels.csv)")
    parser.add_argument("--cols", type=int, default=40, help="Expected number of columns per CSV")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers")
    parser.add_argument("--compress", action="store_true", help="Use compressed NPZ")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of files to convert (for testing)")
    args = parser.parse_args()

    convert_all(csv_dir=args.csv_dir, out_dir=args.out_dir, labels_path=args.labels,
                expected_cols=args.cols, workers=args.workers, compress=args.compress, limit=args.limit)
