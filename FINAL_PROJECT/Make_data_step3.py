#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np

INDEX_PATH = "npz_dir/index.csv"      # Pfad zur index.csv
LABELS_CSV = "B/labels.csv"           # Pfad zu labels.csv (header pid,label)

# 1) Backup
bak = INDEX_PATH + ".bak"
if not os.path.exists(bak):
    os.rename(INDEX_PATH, bak)
    print("Backup created:", bak)
else:
    print("Backup already exists:", bak)

# 2) Lade index und labels
idx = pd.read_csv(bak, dtype=str)
labels_df = pd.read_csv(LABELS_CSV, dtype=str, header=0)
pids = labels_df.iloc[:,0].astype(str).str.strip().values
labs = pd.to_numeric(labels_df.iloc[:,1], errors="coerce")
valid = ~labs.isna()
labmap = {pid.strip(): int(lab) for pid, lab in zip(pids[valid], labs[valid].astype(int))}

# Hilfsfunktion: robustes Matching wie zuvor
def find_label_for_row(npz_path, pid, pid_short):
    # 1) try loading label from npz
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

# 3) Ergänze/aktualisiere label-Spalte
if "label" not in idx.columns:
    idx["label"] = ""

updated = 0
for i, row in idx.iterrows():
    npz_path = row.get("npz_path")
    pid = row.get("pid")
    pid_short = row.get("pid_short")
    if not isinstance(npz_path, str) or not os.path.exists(npz_path):
        # falls npz_path relativ/fehlt, versuche in npz_dir nach pid zu suchen
        if pid:
            candidate = os.path.join(os.path.dirname(INDEX_PATH), f"{pid}.npz")
            if os.path.exists(candidate):
                npz_path = candidate
    label = find_label_for_row(npz_path, pid, pid_short)
    if label is not None:
        idx.at[i, "label"] = int(label)
        updated += 1
    else:
        idx.at[i, "label"] = ""  # leer lassen, falls kein Label

print(f"Labels ergänzt für {updated} Einträge (rest bleibt leer).")

# 4) Schreibe index.csv zurück (erst wenn alles ok)
idx.to_csv(INDEX_PATH, index=False)
print("index.csv aktualisiert:", INDEX_PATH)
