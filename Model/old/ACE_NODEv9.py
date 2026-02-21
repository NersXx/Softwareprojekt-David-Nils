# ACE_NODEv9_fixed.py
# Trainingsskript, das das gefixte ACE_NODE-Modul verwendet (ACE_NODEv41.py)
# - Lädt files/B/time_series/*.csv
# - Liest labels aus files/B/labels.csv
# - Pad/Batche/Sharde für pmap (2 GPUs)
# - Nutzt eqx + optax + diffrax-basiertes ACE_NODE

import os
import glob
import sys
import time
from functools import partial
import random as pyrandom

import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp
from jax import random as jrandom, lax

import equinox as eqx
import optax

# Importiere dein gefixtes ACE_NODE-Modul
# Speichere die gefixte Datei als ACE_NODEv41.py oder passe den Namen hier an
import ACE_NODEv41 as ace_node

# -------------------------
# Hilfsfunktionen: Dateien, Laden, Padding, Sharding
# -------------------------
def list_patient_files(data_dir="files/B/time_series", pattern="*.csv"):
    search_path = os.path.join(data_dir, pattern)
    files = sorted(glob.glob(search_path))
    return files

def load_csv_to_array(path, expected_cols=40, fill_method="ffill"):
    # robustes Einlesen mit Pandas (um fehlende Werte zu behandeln)
    df = pd.read_csv(path, header=None)
    arr = df.values.astype(np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    # trim/pad columns
    if arr.shape[1] > expected_cols:
        arr = arr[:, -expected_cols:]
    elif arr.shape[1] < expected_cols:
        pad = np.full((arr.shape[0], expected_cols - arr.shape[1]), np.nan, dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=1)
    # forward/backfill then fill remaining with 0
    df2 = pd.DataFrame(arr).fillna(method="ffill").fillna(method="bfill").fillna(0.0)
    return df2.values.astype(np.float32)

def load_labels(labels_path="files/B/labels.csv"):
    labels_map = {}
    if not os.path.exists(labels_path):
        print(f"Warnung: Labels Datei {labels_path} nicht gefunden.")
        return labels_map
    with open(labels_path, newline="") as f:
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

def pad_batch(seqs, expected_cols=40):
    lengths = [s.shape[0] for s in seqs]
    maxlen = max(lengths)
    C = expected_cols
    B = len(seqs)
    batch = np.zeros((B, maxlen, C), dtype=np.float32)
    time_mask = np.zeros((B, maxlen), dtype=np.float32)
    for i, s in enumerate(seqs):
        L = s.shape[0]
        batch[i, :L, :] = s
        time_mask[i, :L] = 1.0
    return batch, time_mask, np.array(lengths, dtype=np.int32)

def shard_array(x, n_shards):
    # x shape: (B, ...) -> returns (n_shards, per, ...)
    N = x.shape[0]
    per = N // n_shards
    if per == 0:
        raise ValueError("Batch zu klein für Anzahl Devices.")
    N_trunc = per * n_shards
    b = x[:N_trunc]
    new_shape = (n_shards, per) + b.shape[1:]
    return b.reshape(new_shape)

def device_put_sharded_from_numpy(sharded_np, devices):
    # sharded_np shape (n_devices, per, ...)
    shards = [sharded_np[i] for i in range(sharded_np.shape[0])]
    return jax.device_put_sharded(shards, devices)

def generate_initial_attention_from_sample(sample_array, hidden_dim):
    # einfache Korrelationsmatrix der ersten zwei Zielspalten (oder zeros)
    if sample_array.shape[1] >= 2:
        corr = np.corrcoef(sample_array[:, :2].T)
        flat = corr.reshape(-1).astype(np.float32)
    else:
        flat = np.zeros((hidden_dim * hidden_dim,), dtype=np.float32)
    if flat.size < hidden_dim * hidden_dim:
        pad = np.zeros((hidden_dim * hidden_dim - flat.size,), dtype=np.float32)
        flat = np.concatenate([flat, pad], axis=0)
    elif flat.size > hidden_dim * hidden_dim:
        flat = flat[: hidden_dim * hidden_dim]
    return flat

# -------------------------
# Training: loss, pmap step
# -------------------------
def make_train_fns(model_static, optimizer):
    # loss_fn: params sind die trainierbaren Arrays (params tree)
    @eqx.filter_value_and_grad
    def loss_fn(params, model_static, x, y, time_mask, attn):
        m = eqx.combine(params, model_static)
        # single sample call: x (T,C), y (T,targets), time_mask (T,)
        def single_call(xi, yi, tmask):
            y0 = yi[0]
            out = m(xi, y0, attn)  # attn ist per-device (kein batch axis)
            return out
        y_pred = jax.vmap(single_call)(x, y, time_mask)
        if y_pred is None:
            # explizite Fehlermeldung, damit pmap nicht mit None abstürzt
            raise RuntimeError("Model returned None for a sample in loss_fn.")
        # y_pred shape: (per, T, targets)
        # compute masked MSE
        se = (y - y_pred) ** 2
        se = jnp.sum(se, axis=-1)  # sum over features
        se = se * time_mask
        loss = jnp.sum(se) / (jnp.sum(time_mask) + 1e-8)
        return loss

    # pmap'ed train step
    @partial(jax.pmap, axis_name="i", donate_argnums=(0, 4))
    def train_step_pmap(params_local, x_local, y_local, time_mask_local, opt_state_local, attn_local):
        loss, grads = loss_fn(params_local, model_static, x_local, y_local, time_mask_local, attn_local)
        # average grads across devices
        grads = jax.tree.map(lambda g: lax.pmean(g, axis_name="i"), grads)
        loss_mean = lax.pmean(loss, axis_name="i")
        updates, opt_state_local = optimizer.update(grads, opt_state_local, params_local)
        params_local = optax.apply_updates(params_local, updates)
        return params_local, opt_state_local, loss_mean

    return loss_fn, train_step_pmap

# -------------------------
# Main
# -------------------------
def main():
    # JAX devices
    n_devices = jax.local_device_count()
    if n_devices < 2:
        print(f"Warnung: nur {n_devices} Device(s) verfügbar. Dieses Beispiel ist für 2 GPUs gedacht.")
    devices = jax.local_devices()[:n_devices]

    # RNG
    key = jrandom.PRNGKey(int(time.time()) & 0xFFFFFFFF)
    model_key, data_key = jrandom.split(key)

    # Dateien / Labels
    files = list_patient_files("files/B/time_series", "*.csv")
    if len(files) == 0:
        raise RuntimeError("Keine Dateien in files/B/time_series gefunden.")
    labels_map = load_labels("files/B/labels.csv")

    # Für schnellen Test: n_samples begrenzen (entferne für komplettes Training)
    # files = files[:200]  # optional

    # Lade eine kleine Stichprobe zum Initialisieren (z.B. erste 500 Dateien)
    sample_files = files[:500]
    sample_seqs = []
    sample_ids = []
    for p in sample_files:
        arr = load_csv_to_array(p, expected_cols=40)
        sample_seqs.append(arr)
        sample_ids.append(os.path.basename(p).split(".")[0])
    concat = np.concatenate(sample_seqs, axis=0)

    # Normalizer (du kannst deine norm-Klasse verwenden; hier einfache Z-Score)
    # Für Demo: einfache Standardisierung
    mean = concat.mean(axis=0)
    std = concat.std(axis=0) + 1e-6

    def normalize_arr(a):
        return (a - mean) / std

    # Model initialisieren (hidden_dim anpassen falls nötig)
    hidden_dim = 2
    model = ace_node.ACE_NODE(hidden_dim=hidden_dim, key=model_key)
    initial_params = ace_node.get_params(model)
    _, model_static = eqx.partition(model, eqx.is_inexact_array)

    # initial attention (einfach aus sample)
    initial_attention = generate_initial_attention_from_sample(concat, hidden_dim).astype(np.float32)
    # per-device attn
    attn_repl = jnp.broadcast_to(jnp.array(initial_attention), (n_devices,) + initial_attention.shape)

    # Optimizer
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(initial_params)

    # Repliziere params und opt_state auf Devices
    params_repl = jax.device_put_replicated(initial_params, devices)
    opt_state_repl = jax.device_put_replicated(opt_state, devices)

    # Erzeuge Trainingsbatches (einfacher epochaler Durchlauf)
    # Wir bauen eine Liste von sequences+labels (nur solche mit Label vorhanden)
    seqs = []
    labs = []
    ids = []
    for p in files:
        keyname = os.path.basename(p).split(".")[0]
        if keyname not in labels_map:
            continue
        arr = load_csv_to_array(p, expected_cols=40)
        seqs.append(arr)
        labs.append(labels_map[keyname])
        ids.append(keyname)

    # Shuffle dataset
    idxs = np.arange(len(seqs))
    pyrandom.shuffle(idxs)
    seqs = [seqs[i] for i in idxs]
    labs = [labs[i] for i in idxs]

    # Batchgröße: per-device * n_devices
    per_device_batch = 4  # anpassen je nach GPU RAM
    batch_size = per_device_batch * n_devices

    # Trainingsfunktionen
    loss_fn, train_step_pmap = make_train_fns(model_static, optimizer)

    # Training loop (ein epoch Beispiel)
    n_epochs = 10
    for epoch in range(n_epochs):
        epoch_losses = []
        # iterate batches
        for i in range(0, len(seqs), batch_size):
            batch_seqs = seqs[i : i + batch_size]
            batch_labs = labs[i : i + batch_size]
            if len(batch_seqs) < batch_size:
                break  # drop last incomplete batch for simplicity

            # pad batch (returns numpy arrays)
            batch_x_np, batch_mask_np, lengths = pad_batch(batch_seqs, expected_cols=40)
            # For targets: here we use the original sequences as targets (auto-regressive fit).
            # If you have labels (binary), adapt loss/targets accordingly.
            batch_y_np = batch_x_np.copy()  # placeholder: predict full sequence
            # normalize
            batch_x_np = (batch_x_np - mean) / std
            batch_y_np = (batch_y_np - mean) / std

            # reshape to (n_devices, per, ...)
            x_sharded = shard_array(batch_x_np, n_devices)        # (n_devices, per, T, C)
            y_sharded = shard_array(batch_y_np, n_devices)
            mask_sharded = shard_array(batch_mask_np, n_devices)

            # put on devices
            x_dev = device_put_sharded_from_numpy(x_sharded, devices)
            y_dev = device_put_sharded_from_numpy(y_sharded, devices)
            mask_dev = device_put_sharded_from_numpy(mask_sharded, devices)

            # attn per-device (already attn_repl)
            # run pmap step
            params_repl, opt_state_repl, loss_dev = train_step_pmap(params_repl, x_dev, y_dev, mask_dev, opt_state_repl, attn_repl)
            # loss_dev shape: (n_devices,)
            loss_val = float(jnp.mean(loss_dev))
            epoch_losses.append(loss_val)

        mean_epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        print(f"Epoch {epoch:03d} mean loss: {mean_epoch_loss:.6f}")

        # optional: parameter averaging across devices to keep replicas in sync
        params_host = jax.device_get(params_repl)
        averaged_params = jax.tree.map(lambda x: jnp.mean(x, axis=0), params_host)
        params_repl = jax.device_put_replicated(averaged_params, devices)

    # Nach Training: kombiniere params + static zu finalem Modell
    final_params = jax.device_get(params_repl)[0]  # alle replicas sind gleich nach averaging
    final_model = eqx.combine(final_params, model_static)

    # Beispielvorhersage (auf Host)
    # Nimm eine Sequenz aus dataset
    test_seq = seqs[0]
    test_x = (test_seq - mean) / std
    y0 = jnp.zeros((hidden_dim,), dtype=jnp.float32)
    # ts: linspace über Länge
    ts = jnp.linspace(0.0, 1.0, num=test_x.shape[0], dtype=jnp.float32)
    pred = final_model(test_x, y0, attn=initial_attention, ts_in=ts)
    print("Prediction shape:", pred.shape)

    return 0

if __name__ == "__main__":
    sys.exit(main())
