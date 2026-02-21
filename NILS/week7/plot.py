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

ax3b = ax3.twinx()
plotted = False
if fp_p is not None:
    ax3b.plot(x, fp_p, marker='x', label='FP', color='C2')
    plotted = True
if fn_p is not None:
    ax3b.plot(x, fn_p, marker='s', label='FN', color='C3')
    plotted = True
if plotted:
    ax3b.set_ylabel('Anzahl FP/FN')

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
