#!/usr/bin/env python3
import os
import sys
import time
import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp
from jax import random as jrandom, lax

import equinox as eqx
import optax

# Importieren Ihres ACE_NODE Moduls
import ACE_NODEv41 as ace_node

# ---------------------------------------------------------
# 1. Wrapper-Modell für Klassifikation
# ---------------------------------------------------------
class SepsisClassifier(eqx.Module):
    node: eqx.Module
    readout: eqx.nn.Linear

    def __init__(self, hidden_dim, key):
        k1, k2 = jrandom.split(key)
        # Ihr existierendes ACE_NODE Modell
        self.node = ace_node.ACE_NODE(hidden_dim=hidden_dim, key=k1)
        # Eine Schicht, die von hidden_dim -> 1 projiziert (Logit für Sepsis)
        self.readout = eqx.nn.Linear(hidden_dim, 1, key=k2)

    def __call__(self, x, y0, attn):
        # node gibt (T, hidden_dim) zurück
        output_seq = self.node(x, y0, attn)
        # Wir wenden den Readout auf jeden Zeitschritt an -> (T, 1)
        logits_seq = jax.vmap(self.readout)(output_seq)
        return logits_seq

# ---------------------------------------------------------
# 2. Hilfsfunktionen (Loading & Padding)
# ---------------------------------------------------------
def list_npz_files(index_path="npz_dir/index.csv"):
    # Erwartet CSV mit Spalten: npz_path, label (label optional)
    df = pd.read_csv(index_path)
    return df

def load_npz(path, label_from_csv=None):
    """
    Lädt die Zeitreihe (X) und bestimmt das Label:
    - Wenn in der NPZ ein 'label' Feld vorhanden ist, wird dieses verwendet.
    - Sonst wird label_from_csv (aus index.csv) verwendet.
    Rückgabe: arr (np.float32), label (int 0/1)
    """
    d = np.load(path, allow_pickle=True)
    # Annahme: Daten heißen 'data_raw' oder ähnlich
    if "data_raw" in d:
        arr = d["data_raw"]
    elif "data_median" in d:
        arr = d["data_median"]
    else:
        # fallback: first array-like entry
        first_key = d.files[0]
        arr = d[first_key]
    arr = arr.astype(np.float32)

    # Label aus NPZ falls vorhanden
    label = None
    if "label" in d:
        try:
            label_val = np.array(d["label"])
            # handle scalar arrays and 0.0/1.0 floats
            if label_val.shape == ():
                label = int(label_val.item())
            else:
                # if array, take first element
                label = int(np.array(label_val).astype(int).ravel()[0])
        except Exception:
            label = None

    # fallback auf label_from_csv
    if label is None and label_from_csv is not None:
        try:
            label = int(float(label_from_csv))
        except Exception:
            # if label_from_csv is empty or invalid, default to 0
            label = 0

    # ensure binary 0/1
    label = 1 if label else 0

    return arr, label

def pad_batch_classification(seqs, labels, expected_cols=40):
    """
    Paddet die Sequenzen und gibt Labels als (B, 1) zurück.
    """
    lengths = [s.shape[0] for s in seqs]
    maxlen = max(lengths)
    B = len(seqs)

    batch_x = np.full((B, maxlen, expected_cols), 0.0, dtype=np.float32) # Fill NaN mit 0
    time_mask = np.zeros((B, maxlen), dtype=np.float32)
    observed_mask = np.zeros((B, maxlen, expected_cols), dtype=np.float32)

    # Array für die Indizes des letzten validen Zeitschritts
    last_indices = np.array(lengths, dtype=np.int32) - 1

    for i, s in enumerate(seqs):
        L = s.shape[0]
        s_clean = np.nan_to_num(s, nan=0.0)
        # ensure correct number of columns
        if s_clean.shape[1] < expected_cols:
            pad = np.zeros((L, expected_cols - s_clean.shape[1]), dtype=np.float32)
            s_clean = np.concatenate([s_clean, pad], axis=1)
        elif s_clean.shape[1] > expected_cols:
            s_clean = s_clean[:, :expected_cols]
        batch_x[i, :L, :] = s_clean
        time_mask[i, :L] = 1.0
        observed_mask[i, :L, :] = (~np.isnan(s)).astype(np.float32)

    # Labels zu Shape (B, 1) formen und sicherstellen, dass sie 0/1 sind
    labels_bin = [1 if float(l) else 0 for l in labels]
    batch_y = np.array(labels_bin, dtype=np.float32).reshape((B, 1))

    return batch_x, batch_y, time_mask, observed_mask, last_indices

def shard_array(x, n_shards):
    N = x.shape[0]
    per = N // n_shards
    b = x[:per * n_shards]
    new_shape = (n_shards, per) + b.shape[1:]
    return b.reshape(new_shape)

def device_put_sharded_from_numpy(sharded_np, devices):
    shards = [sharded_np[i] for i in range(sharded_np.shape[0])]
    return jax.device_put_sharded(shards, devices)

# ---------------------------------------------------------
# 3. Training Logic (Loss & Step)
# ---------------------------------------------------------
def make_train_fns(model_static, optimizer):

    @eqx.filter_value_and_grad
    def loss_fn(params, model_static, x, y, time_mask, last_idxs, attn):
        # y hat hier shape (B, 1) -> Wahres Label (0 oder 1)
        model = eqx.combine(params, model_static)

        def single_call(xi):
            # Init Hidden State
            y0 = jnp.zeros((model.node.hidden_dim,), dtype=jnp.float32)
            # Forward pass: gibt (T, 1) Logits zurück
            logits_seq = model(xi, y0, attn)
            return logits_seq

        # (B, T, 1)
        logits_seq_pred = jax.vmap(single_call)(x)

        # WICHTIG: Wir nehmen nur die Vorhersage am letzten echten Zeitschritt
        batch_indices = jnp.arange(x.shape[0])
        final_logits = logits_seq_pred[batch_indices, last_idxs, 0] # Shape (B,)
        final_logits = final_logits.reshape(-1, 1) # Shape (B, 1)

        # Binary Cross Entropy mit Logits (numerisch stabiler)
        loss = optax.sigmoid_binary_cross_entropy(final_logits, y)

        return jnp.mean(loss)

    def train_step(params_local, x_local, y_local, time_mask_local,
                   opt_state_local, last_idxs_local, attn_local):

        loss, grads = loss_fn(params_local, model_static,
                              x_local, y_local, time_mask_local,
                              last_idxs_local, attn_local)

        grads = jax.tree.map(lambda g: lax.pmean(g, axis_name="i"), grads)
        loss_mean = lax.pmean(loss, axis_name="i")

        updates, opt_state_local = optimizer.update(grads, opt_state_local, params_local)
        params_local = optax.apply_updates(params_local, updates)
        return params_local, opt_state_local, loss_mean

    # Donate argnums für Performance (params, opt_state)
    train_step_pmap = jax.pmap(train_step, axis_name="i", donate_argnums=(0, 4))
    return loss_fn, train_step_pmap

# ---------------------------------------------------------
# 4. Main Loop
# ---------------------------------------------------------
def main():
    n_devices = jax.local_device_count()
    devices = jax.local_devices()[:n_devices]
    print(f"Verwende {n_devices} Devices.")

    key = jrandom.PRNGKey(42)
    model_key, _ = jrandom.split(key)

    # Index laden
    df = list_npz_files("npz_dir/index.csv")
    files = df["npz_path"].tolist()
    labels_csv = df["label"].tolist() if "label" in df.columns else [None] * len(files)

    # 1. Neues Klassifikations-Modell initialisieren
    hidden_dim = 2
    model = SepsisClassifier(hidden_dim=hidden_dim, key=model_key)

    params, model_static = eqx.partition(model, eqx.is_array)

    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)

    params_repl = jax.device_put_replicated(params, devices)
    opt_state_repl = jax.device_put_replicated(opt_state, devices)

    loss_fn, train_step_pmap = make_train_fns(model_static, optimizer)

    per_device_batch = 4
    batch_size = per_device_batch * n_devices
    n_epochs = 10

    print("Starte Training...")
    for epoch in range(n_epochs):
        epoch_losses = []

        # Shuffle
        perm = np.random.permutation(len(files))

        for i in range(0, len(files), batch_size):
            batch_idxs = perm[i:i+batch_size]
            if len(batch_idxs) < batch_size: break

            # Batch laden
            seqs, batch_labels = [], []
            for idx in batch_idxs:
                p = files[idx]
                l_csv = labels_csv[idx] if idx < len(labels_csv) else None
                arr, label = load_npz(p, l_csv)  # load_npz nutzt NPZ label falls vorhanden, sonst index.csv
                seqs.append(arr)
                batch_labels.append(label)

            # Pad Batch & Classification Labels
            batch_x, batch_y, mask_time, _, last_idxs = pad_batch_classification(seqs, batch_labels, expected_cols=40)

            # Sharding
            x_shard = shard_array(batch_x, n_devices)
            y_shard = shard_array(batch_y, n_devices) # Shape (Devices, LocalBatch, 1)
            mask_shard = shard_array(mask_time, n_devices)
            last_idxs_shard = shard_array(last_idxs, n_devices)

            # Device Put
            x_dev = device_put_sharded_from_numpy(x_shard, devices)
            y_dev = device_put_sharded_from_numpy(y_shard, devices)
            mask_dev = device_put_sharded_from_numpy(mask_shard, devices)
            last_idxs_dev = device_put_sharded_from_numpy(last_idxs_shard, devices)

            # Attn Platzhalter
            attn = jnp.zeros((hidden_dim*hidden_dim,), dtype=jnp.float32)
            attn_repl = jnp.broadcast_to(attn, (n_devices,) + attn.shape)

            # Step
            params_repl, opt_state_repl, loss_val = train_step_pmap(
                params_repl, x_dev, y_dev, mask_dev, opt_state_repl, last_idxs_dev, attn_repl
            )
            epoch_losses.append(loss_val[0].item())

        print(f"Epoch {epoch+1} | Loss: {np.mean(epoch_losses):.5f}")

    return 0

if __name__ == "__main__":
    main()
