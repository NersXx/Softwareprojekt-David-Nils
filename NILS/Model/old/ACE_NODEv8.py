import sys
import time
import os
import glob
from functools import partial
import equinox as eqx
import jax
import jax.numpy as jnp
from jax import random, lax
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optax
import warnings

import ACE_NODEv4 as ace_node
import ASS4.old.norm as norm

warnings.filterwarnings("ignore", category=FutureWarning)

# Anzahl Devices (GPUs)
n_devices = jax.local_device_count()
if n_devices < 2:
    print(f"Warnung: nur {n_devices} Device(s) verfügbar. Dieses Beispiel ist für 2 GPUs gedacht.")

# ---------- Dateiliste / Loader ----------
def list_patient_files(data_dir="files/B/time_series", pattern="*.csv"):
    search_path = os.path.join(data_dir, pattern)
    files = sorted(glob.glob(search_path))
    return files

def load_csv_to_array(path, expected_cols=40, fill_method=None):
    try:
        arr = np.loadtxt(path, delimiter=",", dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
    except Exception:
        import pandas as pd
        df = pd.read_csv(path, dtype=str)
        for col in df.columns:
            df[col] = df[col].astype(str).str.strip().replace('', np.nan)
            df[col] = pd.to_numeric(df[col], errors='coerce')
        arr = df.values.astype(np.float32)

    if arr.shape[1] > expected_cols:
        arr = arr[:, -expected_cols:]
    elif arr.shape[1] < expected_cols:
        pad_width = expected_cols - arr.shape[1]
        pad = np.full((arr.shape[0], pad_width), np.nan, dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=1)

    nan_mask = ~np.isnan(arr)
    if fill_method is not None:
        import pandas as pd
        df = pd.DataFrame(arr)
        if fill_method == "ffill":
            df = df.fillna(method="ffill").fillna(method="bfill")
        elif fill_method == "median":
            df = df.fillna(df.median())
        arr = df.values.astype(np.float32)
        nan_mask = ~np.isnan(arr)

    return arr.astype(np.float32), nan_mask.astype(np.float32)

# ---------- Padding / Sharding ----------
def pad_batch(seqs, masks):
    lengths = [s.shape[0] for s in seqs]
    maxlen = max(lengths)
    C = seqs[0].shape[1]
    B = len(seqs)
    batch = np.zeros((B, maxlen, C), dtype=np.float32)
    feature_mask = np.zeros((B, maxlen, C), dtype=np.float32)
    for i, s in enumerate(seqs):
        L = s.shape[0]
        batch[i, :L, :] = s
        feature_mask[i, :L, :] = masks[i]
    time_mask = (feature_mask.sum(axis=-1) > 0).astype(np.float32)
    return batch, feature_mask, time_mask, np.array(lengths, dtype=np.int32)

def make_sharded_batch(batch_np):
    N = batch_np.shape[0]
    per = N // n_devices
    N_trunc = per * n_devices
    if N_trunc == 0:
        raise ValueError("Batch zu klein für Anzahl Devices.")
    b = batch_np[:N_trunc]
    new_shape = (n_devices, per) + b.shape[1:]
    return b.reshape(new_shape)

def device_put_sharded_batch(sharded_np):
    shards = [sharded_np[i] for i in range(sharded_np.shape[0])]
    return jax.device_put_sharded(shards, jax.local_devices()[:n_devices])

# ---------- Hilfsfunktion für initiale Attention ----------
def generate_initial_attention_from_sample(sample_array, hidden_dim):
    # Beispiel: benutze Korrelationsmatrix der ersten zwei Zielspalten wie im Original
    # sample_array shape (N, C). Wir nehmen zwei Spalten (falls vorhanden) oder fallback zeros.
    if sample_array.shape[1] >= 2:
        corr = np.corrcoef(sample_array[:, :2].T)
        flat = corr.reshape(-1).astype(np.float32)
    else:
        flat = np.zeros((hidden_dim * hidden_dim,), dtype=np.float32)
    # Falls Länge nicht passt, pad/trim
    if flat.size < hidden_dim * hidden_dim:
        pad = np.zeros((hidden_dim * hidden_dim - flat.size,), dtype=np.float32)
        flat = np.concatenate([flat, pad], axis=0)
    elif flat.size > hidden_dim * hidden_dim:
        flat = flat[: hidden_dim * hidden_dim]
    return flat

# ---------- Hauptprogramm ----------
def main() -> int:
    key = random.key(int(time.time()))
    model_key, train_key = random.split(key)

    files = list_patient_files("files/B/time_series", "*.csv")
    if len(files) == 0:
        raise RuntimeError("Keine Dateien in B/time_series gefunden (pattern *.csv).")

    # Labels (optional)
    def load_labels(labels_path="files/B/labels.csv"):
        labels_map = {}
        if not os.path.exists(labels_path):
            print(f"Warnung: Labels Datei {labels_path} nicht gefunden.")
            return labels_map
        import csv
        with open(labels_path, newline='') as f:
            reader = csv.reader(f)
            for r in reader:
                if len(r) < 2:
                    continue
                key = r[0].strip()
                try:
                    val = int(r[1])
                except ValueError:
                    continue
                labels_map[key] = val
        return labels_map

    labels_map = load_labels("files/B/labels.csv")

    # Sample zum Fitten der Normalizer (kleine Stichprobe)
    sample_files = files[:200]
    sample_seqs = []
    for p in sample_files:
        arr, _ = load_csv_to_array(p, expected_cols=40, fill_method=None)
        sample_seqs.append(arr)
    concat = np.concatenate(sample_seqs, axis=0)

    normalizer = norm.MinMaxNorm()
    normalizer2 = norm.ZScoreNorm()
    normalizer2.init(jnp.array(concat), 0)
    normalizer.init(jnp.array(concat[:, :1]), 0)

    # Model initialisieren
    model = ace_node.ACE_NODE(2, 32, 3, key=model_key)
    initial_params = ace_node.get_params(model)
    _, model_static = eqx.partition(model, eqx.is_inexact_array)

    # Optimizer
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(initial_params)

    devices = jax.local_devices()[:n_devices]
    params_repl = jax.device_put_replicated(initial_params, devices)
    opt_state_repl = jax.device_put_replicated(opt_state, devices)

    # WICHTIG: Erzeuge eine einzelne Attention pro Device (keine per-sample Attention)
    hidden_dim = getattr(model, "hidden_dim", 2)  # fallback 2
    # initial attention aus sample (oder zeros)
    initial_attention = generate_initial_attention_from_sample(concat, hidden_dim)
    # repliziere auf alle devices (shape per device: (hidden_dim*hidden_dim,))
    attn_repl = jax.device_put_replicated(jnp.array(initial_attention), devices)

    # Loss-Funktion
    @eqx.filter_value_and_grad
    def loss_fn_params(params, model_static, x, y, time_mask, attn):
        m = eqx.combine(params, model_static)
        # Hier rufen wir m mit batched x,y und einer per-device attn (attn hat keine batch axis)
        # Wenn m nicht batched ist, wir vmap über die lokale Batch-Achse:
        def single_call(xi, yi, tmask):
            y0 = yi[0]
            return m(xi, y0, attn)  # attn ist per-device, nicht per-sample

        # vmap über lokale Batch-Achse
        y_pred = jax.vmap(single_call)(x, y, time_mask)
        if y_pred.ndim == 3:
            se = (y - y_pred) ** 2
            se = jnp.sum(se, axis=-1)
            se = se * time_mask
            loss = jnp.sum(se) / (jnp.sum(time_mask) + 1e-8)
        else:
            loss = jnp.mean((y - y_pred) ** 2)
        return loss

    @partial(jax.pmap, axis_name="i", donate_argnums=(0, 5))
    def train_step_pmap(params_local, x_local, y_local, time_mask_local, attn_local, opt_state_local):
        loss, grads = loss_fn_params(params_local, model_static, x_local, y_local, time_mask_local, attn_local)
        grads = jax.tree.map(lambda g: lax.pmean(g, axis_name="i"), grads)
        loss_mean = lax.pmean(loss, axis_name="i")
        updates, opt_state_local = optimizer.update(grads, opt_state_local, params_local)
        params_local = optax.apply_updates(params_local, updates)
        return params_local, opt_state_local, loss_mean

    # Batch generator (erzeugt x,y,mask; keine per-sample attn)
    def batch_generator(file_list, batch_size_global):
        assert batch_size_global % n_devices == 0
        i = 0
        N = len(file_list)
        while i < N:
            batch_files = file_list[i:i + batch_size_global]
            seqs = []
            masks = []
            ys = []
            for p in batch_files:
                arr, nan_mask = load_csv_to_array(p, expected_cols=40, fill_method=None)
                seqs.append(arr)
                masks.append(nan_mask)
                y = arr[:, :2].astype(np.float32)  # Beispiel: target = erste 2 features
                ys.append(y)
            batch_np, feature_mask_np, time_mask_np, lengths = pad_batch(seqs, masks)
            y_np, _, y_time_mask, _ = pad_batch(ys, [np.ones_like(y) for y in ys])

            x_sharded = make_sharded_batch(batch_np)
            y_sharded = make_sharded_batch(y_np)
            time_mask_sharded = make_sharded_batch(time_mask_np)

            x_dev = device_put_sharded_batch(x_sharded)
            y_dev = device_put_sharded_batch(y_sharded)
            time_mask_dev = device_put_sharded_batch(time_mask_sharded)

            yield x_dev, y_dev, time_mask_dev
            i += batch_size_global

    # Trainingsschleife
    epochs = 10
    batch_size_global = 64
    for epoch in range(epochs):
        gen = batch_generator(files, batch_size_global)
        epoch_losses = []
        for x_dev, y_dev, time_mask_dev in gen:
            # attn_repl ist per-device repliziert und wird an pmap übergeben
            params_repl, opt_state_repl, loss_dev = train_step_pmap(params_repl, x_dev, y_dev, time_mask_dev, attn_repl, opt_state_repl)
            epoch_losses.append(float(jax.device_get(loss_dev)[0]))
        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float('nan')
        print(f"Epoch {epoch}, loss {mean_loss:.6f}")

    # Parameter einmalig auf Host holen
    params_host = jax.device_get(params_repl)[0]
    model = eqx.combine(params_host, model_static)

    print("Training fertig.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
