# ACE_NODEv9_fixed.py
# Trainingsskript, das das gefixte ACE_NODE-Modul verwendet (ACE_NODEv41.py)
# - Lädt npz_dir/*.npz über index.csv
# - Liest labels aus index.csv
# - Pad/Batche/Sharde für pmap (2 GPUs)
# - Nutzt eqx + optax + diffrax-basiertes ACE_NODE

import os
import sys
import time
import random as pyrandom
import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp
from jax import random as jrandom, lax

import equinox as eqx
import optax

import ACE_NODEv41 as ace_node

# -------------------------
# Hilfsfunktionen: NPZ laden, Padding, Sharding
# -------------------------
def list_npz_files(index_path="npz_dir/index.csv"):
    df = pd.read_csv(index_path)
    return df

def load_npz(path, use_median=False):
    d = np.load(path, allow_pickle=True)
    arr = d["data_median"] if use_median else d["data_raw"]
    label = d["label"].item() if "label" in d.files else None
    return arr.astype(np.float32), label

def pad_batch(seqs, expected_cols=40):
    lengths = [s.shape[0] for s in seqs]
    maxlen = max(lengths)
    B = len(seqs)
    batch = np.full((B, maxlen, expected_cols), np.nan, dtype=np.float32)
    time_mask = np.zeros((B, maxlen), dtype=np.float32)
    observed_mask = np.zeros((B, maxlen, expected_cols), dtype=np.float32)
    for i, s in enumerate(seqs):
        L = s.shape[0]
        batch[i, :L, :] = s
        time_mask[i, :L] = 1.0
        observed_mask[i, :L, :] = (~np.isnan(s)).astype(np.float32)
    return batch, time_mask, observed_mask, np.array(lengths, dtype=np.int32)

def shard_array(x, n_shards):
    N = x.shape[0]
    per = N // n_shards
    if per == 0:
        raise ValueError("Batch zu klein für Anzahl Devices.")
    N_trunc = per * n_shards
    b = x[:N_trunc]
    new_shape = (n_shards, per) + b.shape[1:]
    return b.reshape(new_shape)

def device_put_sharded_from_numpy(sharded_np, devices):
    shards = [sharded_np[i] for i in range(sharded_np.shape[0])]
    return jax.device_put_sharded(shards, devices)

# -------------------------
# Training: loss, pmap step
# -------------------------
def make_train_fns(model_static, optimizer):
    @eqx.filter_value_and_grad
    def loss_fn(params, model_static, x, y, time_mask, observed_mask, attn):
        m = eqx.combine(params, model_static)

        # m(xi, y0, attn) returns (T,) probabilities per timestep
        def single_call(xi, yi, tmask, omask):
            y0 = jnp.zeros((m.hidden_dim,), dtype=jnp.float32)
            probs_t = m(xi, y0, attn)       # (T,)
            return probs_t

        # y_pred: (B, T)
        y_pred = jax.vmap(single_call)(x, y, time_mask, observed_mask)

        # Ensure y has shape (B, T) and is float
        y = jnp.squeeze(y)
        y = y.astype(jnp.float32)

        # Binary cross entropy per timestep
        eps = 1e-8
        bce = -(y * jnp.log(y_pred + eps) + (1.0 - y) * jnp.log(1.0 - y_pred + eps))  # (B, T)

        # Mask invalid timesteps
        bce = bce * time_mask

        # Average over valid timesteps per batch; avoid division by zero
        denom = jnp.clip(jnp.sum(time_mask, axis=-1), a_min=1.0)  # (B,)
        loss_per_example = jnp.sum(bce, axis=-1) / denom          # (B,)

        # Final loss: mean over batch
        loss = jnp.mean(loss_per_example)
        return loss

    def train_step(params_local, x_local, y_local, time_mask_local,
                   opt_state_local, observed_mask_local, attn_local):
        loss, grads = loss_fn(params_local, model_static,
                              x_local, y_local, time_mask_local,
                              observed_mask_local, attn_local)
        grads = jax.tree.map(lambda g: lax.pmean(g, axis_name="i"), grads)
        loss_mean = lax.pmean(loss, axis_name="i")
        updates, opt_state_local = optimizer.update(grads, opt_state_local, params_local)
        params_local = optax.apply_updates(params_local, updates)
        return params_local, opt_state_local, loss_mean

    train_step_pmap = jax.pmap(train_step, axis_name="i", donate_argnums=(0, 4))
    return loss_fn, train_step_pmap


# -------------------------
# Main
# -------------------------
def main():
    n_devices = jax.local_device_count()
    if n_devices < 2:
        print(f"Warnung: nur {n_devices} Device(s) verfügbar. Dieses Beispiel ist für 2 GPUs gedacht.")
    devices = jax.local_devices()[:n_devices]

    key = jrandom.PRNGKey(int(time.time()) & 0xFFFFFFFF)
    model_key, _ = jrandom.split(key)

    # Index laden
    df = list_npz_files("npz_dir/index.csv")
    files = df["npz_path"].tolist()
    labels = df["label"].tolist()

    # Normalizer über Stichprobe
    sample_arrays = []
    for p in files[:500]:
        arr, _ = load_npz(p, use_median=False)
        sample_arrays.append(arr)
    concat = np.concatenate(sample_arrays, axis=0)
    mean = np.nanmean(concat, axis=0)
    std = np.nanstd(concat, axis=0) + 1e-6

    def normalize_arr(a):
        return (a - mean) / std

    hidden_dim = 2
    model = ace_node.ACE_NODE(hidden_dim=hidden_dim, key=model_key)
    params = ace_node.get_params(model)
    _, model_static = eqx.partition(model, eqx.is_inexact_array)

    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)
    params_repl = jax.device_put_replicated(params, devices)
    opt_state_repl = jax.device_put_replicated(opt_state, devices)

    loss_fn, train_step_pmap = make_train_fns(model_static, optimizer)

    per_device_batch = 4
    batch_size = per_device_batch * n_devices
    n_epochs = 10

    for epoch in range(n_epochs):
        epoch_losses = []
        idxs = np.arange(len(files))
        np.random.shuffle(idxs)
        for i in range(0, len(files), batch_size):
            batch_idxs = idxs[i:i+batch_size]
            if len(batch_idxs) < batch_size:
                break
            seqs, labs = [], []
            for j in batch_idxs:
                arr, lab = load_npz(files[j], use_median=False)
                seqs.append(arr)
                labs.append(lab)
            batch_x, batch_mask, observed_mask, _ = pad_batch(seqs, expected_cols=40)
            batch_y = batch_x.copy()
            batch_x = normalize_arr(batch_x)
            batch_y = normalize_arr(batch_y)

            x_sharded = shard_array(batch_x, n_devices)
            y_sharded = shard_array(batch_y, n_devices)
            mask_sharded = shard_array(batch_mask, n_devices)
            obs_sharded = shard_array(observed_mask, n_devices)

            x_dev = device_put_sharded_from_numpy(x_sharded, devices)
            y_dev = device_put_sharded_from_numpy(y_sharded, devices)
            mask_dev = device_put_sharded_from_numpy(mask_sharded, devices)
            obs_dev = device_put_sharded_from_numpy(obs_sharded, devices)

            attn = jnp.zeros((hidden_dim*hidden_dim,), dtype=jnp.float32)
            attn_repl = jnp.broadcast_to(attn, (n_devices,) + attn.shape)

            params_repl, opt_state_repl, loss_dev = train_step_pmap(params_repl, x_dev, y_dev, mask_dev, opt_state_repl, obs_dev, attn_repl)
            epoch_losses.append(float(jnp.mean(loss_dev)))

        print(f"Epoch {epoch:03d} mean loss: {np.mean(epoch_losses):.6f}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
