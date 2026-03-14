import numpy as np
import glob
import pandas as pd
import os

# Lade labels.csv mit Header (erste Zeile wird als Header behandelt)
df = pd.read_csv("B/labels.csv", header=0, dtype=str)

# Wir erwarten mindestens zwei Spalten: pid, label
if df.shape[1] < 2:
    raise ValueError("labels.csv must have at least two columns (pid,label)")

# pid und label aus den ersten beiden Spalten
pids = df.iloc[:, 0].astype(str).str.strip().values
labs = pd.to_numeric(df.iloc[:, 1], errors="coerce")

# nur gültige Einträge behalten
valid = ~labs.isna()
labmap = {pid.strip(): int(lab) for pid, lab in zip(pids[valid], labs[valid].astype(int))}

# Durch alle npz-Dateien iterieren und Label ergänzen
for path in glob.glob("npz_dir/*.npz"):
    d = dict(np.load(path, allow_pickle=True))
    pid = None
    pid_short = None
    if "pid" in d:
        try:
            pid = d["pid"].item()
        except Exception:
            pid = str(d["pid"])
    if "pid_short" in d:
        try:
            pid_short = d["pid_short"].item()
        except Exception:
            pid_short = str(d["pid_short"])

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
        np.savez_compressed(path, **d)
        print("Wrote label", int(label), "to", os.path.basename(path))
    else:
        print("No label for", os.path.basename(path))
