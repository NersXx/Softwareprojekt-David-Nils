import os, sys
import numpy as np

# matplotlib ggf. installieren/importe
try:
    import matplotlib.pyplot as plt
except Exception:
    print("matplotlib nicht gefunden. Versuche Installation...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
    import matplotlib.pyplot as plt

npz_path = "training_history.npz"
if not os.path.exists(npz_path):
    print(f"Datei nicht gefunden: {npz_path}")
    sys.exit(1)

data = np.load(npz_path)
keys = list(data.keys())
print("Gefundene Keys in NPZ:", keys)

# Kandidatenlisten
acc_cands = ['accuracy','acc','train_accuracy','accuracy_history','train_acc']
loss_cands = ['loss','train_loss','val_loss']
fp_cands = ['fp','false_positives','false_positive','false_pos']
fn_cands = ['fn','false_negatives','false_negative','false_neg']

def find_best_key(cands):
    # exakte match bevorzugen
    for c in cands:
        for k in keys:
            if k.lower() == c.lower():
                return k
    # substring matches sammeln
    matches = []
    for c in cands:
        for k in keys:
            if c.lower() in k.lower():
                matches.append(k)
    if not matches:
        return None
    # bevorzugen ohne 'val'
    for m in matches:
        if 'val' not in m.lower():
            return m
    return matches[0]

acc_key = find_best_key(acc_cands)
loss_key = find_best_key(loss_cands)
fp_key = find_best_key(fp_cands)
fn_key = find_best_key(fn_cands)

print("Verwendete Keys:")
print("  accuracy:", acc_key)
print("  loss:", loss_key)
print("  fp:", fp_key)
print("  fn:", fn_key)

def load_1d(k):
    if k is None: return None
    arr = np.array(data[k])
    arr = np.squeeze(arr)
    if arr.ndim == 0:
        arr = np.array([float(arr)])
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    return arr.astype(float)

acc = load_1d(acc_key)
loss = load_1d(loss_key)
fp = load_1d(fp_key)
fn = load_1d(fn_key)

lengths = [len(a) for a in [acc, loss, fp, fn] if a is not None]
if not lengths:
    print("Keine der erwarteten Metriken gefunden. Abbruch.")
    sys.exit(1)

epochs = max(lengths)
x = np.arange(1, epochs+1)

def padded(y):
    if y is None:
        return None
    out = np.full(epochs, np.nan)
    out[:len(y)] = y
    return out

acc_p = padded(acc)
loss_p = padded(loss)
fp_p = padded(fp)
fn_p = padded(fn)

plt.figure(figsize=(10,14))

ax1 = plt.subplot(3,1,1)
if acc_p is not None:
    ax1.plot(x, acc_p, marker='o', label='accuracy', color='C0')
    ax1.set_ylim(0,1)
ax1.set_title('Accuracy über Epochen')
ax1.set_xlabel('Epoche')
ax1.set_ylabel('Accuracy')
ax1.grid(True)
ax1.legend()

ax2 = plt.subplot(3,1,2)
if loss_p is not None:
    ax2.plot(x, loss_p, marker='o', label='loss', color='C1')
ax2.set_title('Loss über Epochen')
ax2.set_xlabel('Epoche')
ax2.set_ylabel('Loss')
ax2.grid(True)
ax2.legend()

ax3 = plt.subplot(3,1,3)
if acc_p is not None:
    ax3.plot(x, acc_p, marker='o', label='accuracy', color='C0')
ax3.set_xlabel('Epoche')
ax3.set_ylabel('Accuracy')
ax3.set_ylim(0,1)
ax3.grid(True)

# === BEGIN REPLACEMENT BLOCK ===
ax3b = ax3.twinx()

# Unterstütze beide Namensvarianten: fp_p/fn_p (gepaddet) oder fp/fn (ungepaddet)
fp_arr = fp_p if 'fp_p' in globals() else (fp if 'fp' in globals() else None)
fn_arr = fn_p if 'fn_p' in globals() else (fn if 'fn' in globals() else None)

# sichere Werteliste (ignoriert NaNs)
vals = np.array([], dtype=float)
if fp_arr is not None:
    vals = np.concatenate([vals, fp_arr[~np.isnan(fp_arr)]])
if fn_arr is not None:
    vals = np.concatenate([vals, fn_arr[~np.isnan(fn_arr)]])

# Falls keine FP/FN vorhanden, setze Default
if vals.size == 0:
    clip_percentile = 99
    scale = 1.0
else:
    clip_percentile = 99  # ändere auf 95 wenn du stärker clippen willst
    scale = float(np.nanpercentile(vals, clip_percentile))
    if scale <= 0 or np.isnan(scale):
        scale = 1.0

# Hilfsfunktion: skaliere und markiere Werte, die über dem Clip liegen
def scaled_and_clipped(arr, scale):
    if arr is None:
        return None, None
    scaled = np.full_like(arr, np.nan, dtype=float)
    clipped_mask = np.zeros_like(arr, dtype=bool)
    valid = ~np.isnan(arr)
    scaled[valid] = arr[valid] / scale
    clipped_mask[valid] = arr[valid] > scale
    # setze sichtbare Punkte für clipped auf 1.0 (Rand)
    scaled[clipped_mask] = 1.0
    return scaled, clipped_mask

fp_scaled, fp_clipped = scaled_and_clipped(fp_arr, scale)
fn_scaled, fn_clipped = scaled_and_clipped(fn_arr, scale)

# Plot: Accuracy links (0..1)
if acc_p is not None:
    ax3.plot(x, acc_p, marker='o', label='accuracy', color='C0')
ax3.set_ylim(0, 1)
ax3.set_ylabel('Accuracy')

# Plot skaliert auf rechte Achse (sichtbar neben Accuracy)
if fp_scaled is not None:
    ax3b.plot(x, fp_scaled, marker='x', linestyle='-', label='FP (norm)', color='C2')
    if fp_clipped is not None and fp_clipped.any():
        ax3b.plot(x[fp_clipped], fp_scaled[fp_clipped], 'o', color='red', label='FP clipped')
if fn_scaled is not None:
    ax3b.plot(x, fn_scaled, marker='s', linestyle='-', label='FN (norm)', color='C3')
    if fn_clipped is not None and fn_clipped.any():
        ax3b.plot(x[fn_clipped], fn_scaled[fn_clipped], 'o', color='red', label='FN clipped')

# Sekundäre Achse zeigt die Originalwerte (Rückskalierung)
def to_original(y): return y * scale
def to_scaled(y): return y / scale

secax = ax3.secondary_yaxis('right', functions=(to_original, to_scaled))
secax.set_ylabel(f'Anzahl FP/FN (Original, clipped > {int(scale)})')

# sinnvolle Ticks für rechte Achse (wenige Ticks)
try:
    ticks = np.linspace(0, scale, min(5, int(scale) + 1))
    secax.set_yticks(ticks)
except Exception:
    pass

# kombinierte Legende (Accuracy + FP/FN + clipped)
lines_l, labels_l = ax3.get_legend_handles_labels()
lines_r, labels_r = ax3b.get_legend_handles_labels()
# entferne doppelte Labels
all_lines = lines_l + lines_r
all_labels = labels_l + labels_r
seen = set()
uniq_lines = []
uniq_labels = []
for ln, lb in zip(all_lines, all_labels):
    if lb not in seen:
        uniq_lines.append(ln)
        uniq_labels.append(lb)
        seen.add(lb)
ax3.legend(uniq_lines, uniq_labels, loc='best')
# === END REPLACEMENT BLOCK ===

# kombinierte Legende
lines, labels = ax3.get_legend_handles_labels()
lines2, labels2 = ax3b.get_legend_handles_labels()
if lines2:
    ax3.legend(lines+lines2, labels+labels2, loc='best')
else:
    ax3.legend(loc='best')

plt.tight_layout()
out_png = "training_history_plots.png"
plt.savefig(out_png, dpi=150)
print("Gespeichert:", out_png)
plt.show()
