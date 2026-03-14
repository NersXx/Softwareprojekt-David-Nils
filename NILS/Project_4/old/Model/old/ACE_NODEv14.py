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
        self.node = ace_node.ACE_NODE(hidden_dim=hidden_dim, key=k1)
        self.readout = eqx.nn.Linear(hidden_dim, 1, key=k2)

    def __call__(self, x, y0, attn):
        output_seq = self.node(x, y0, attn)
        logits_seq = jax.vmap(self.readout)(output_seq)
        return logits_seq

# ---------------------------------------------------------
# 2. Hilfsfunktionen (Loading & Padding)
# ---------------------------------------------------------
def list_npz_files(index_path="npz_dir/index.csv", shuffle=True, seed=42):
    df = pd.read_csv(index_path)
    paths = df["npz_path"].tolist()
    # Labels sicher laden, auch wenn Spalte fehlt
    labels_all = df["label"].tolist() if "label" in df.columns else [0] * len(paths)

    if shuffle:
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(paths))
        paths = [paths[i] for i in perm]
        labels_all = [labels_all[i] for i in perm]

    n = len(paths)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    # Der Rest ist Test
    
    train_files  = paths[:n_train]
    train_labels = labels_all[:n_train]
    
    val_files    = paths[n_train:n_train + n_val]
    val_labels   = labels_all[n_train:n_train + n_val]
    
    test_files   = paths[n_train + n_val:]
    test_labels  = labels_all[n_train + n_val:]

    print(f"Split Info: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
    return train_files, train_labels, val_files, val_labels, test_files, test_labels

def load_npz(path, label_from_csv=None):
    d = np.load(path, allow_pickle=True)
    if "data_raw" in d: arr = d["data_raw"]
    elif "data_median" in d: arr = d["data_median"]
    else: arr = d[d.files[0]]
    arr = arr.astype(np.float32)

    label = None
    if "label" in d:
        try:
            label_val = np.array(d["label"])
            if label_val.shape == (): label = int(label_val.item())
            else: label = int(np.array(label_val).astype(int).ravel()[0])
        except: label = None

    if label is None and label_from_csv is not None:
        try: label = int(float(label_from_csv))
        except: label = 0
    
    label = 1 if label else 0
    return arr, label

def pad_batch_classification(seqs, labels, expected_cols=40):
    lengths = [s.shape[0] for s in seqs]
    maxlen = max(lengths)
    B = len(seqs)

    batch_x = np.full((B, maxlen, expected_cols), 0.0, dtype=np.float32)
    time_mask = np.zeros((B, maxlen), dtype=np.float32)
    last_indices = np.array(lengths, dtype=np.int32) - 1

    for i, s in enumerate(seqs):
        L = s.shape[0]
        s_clean = np.nan_to_num(s, nan=0.0)
        if s_clean.shape[1] < expected_cols:
            pad = np.zeros((L, expected_cols - s_clean.shape[1]), dtype=np.float32)
            s_clean = np.concatenate([s_clean, pad], axis=1)
        elif s_clean.shape[1] > expected_cols:
            s_clean = s_clean[:, :expected_cols]
        batch_x[i, :L, :] = s_clean
        time_mask[i, :L] = 1.0

    labels_bin = [1 if float(l) else 0 for l in labels]
    batch_y = np.array(labels_bin, dtype=np.float32).reshape((B, 1))

    return batch_x, batch_y, time_mask, last_indices

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
# 3. Training & Eval Logic
# ---------------------------------------------------------
def make_fns(model_static, optimizer):

    # --- Common Core Logic ---
    def compute_logits(params, model_static, x, last_idxs, attn):
        model = eqx.combine(params, model_static)
        def single_call(xi):
            y0 = jnp.zeros((model.node.hidden_dim,), dtype=jnp.float32)
            logits_seq = model(xi, y0, attn)
            return logits_seq
        
        logits_seq_pred = jax.vmap(single_call)(x)
        # Extrahiere Logits am letzten Zeitschritt
        batch_indices = jnp.arange(x.shape[0])
        final_logits = logits_seq_pred[batch_indices, last_idxs, 0]
        return final_logits.reshape(-1, 1)

    # --- Loss Function ---
    @eqx.filter_value_and_grad
    def loss_fn(params, model_static, x, y, last_idxs, attn):
        preds = compute_logits(params, model_static, x, last_idxs, attn)
        loss = optax.sigmoid_binary_cross_entropy(preds, y)
        return jnp.mean(loss)

    # --- Train Step ---
    def train_step(params, x, y, opt_state, last_idxs, attn):
        loss, grads = loss_fn(params, model_static, x, y, last_idxs, attn)
        grads = jax.tree.map(lambda g: lax.pmean(g, axis_name="i"), grads)
        loss_mean = lax.pmean(loss, axis_name="i")
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_mean

    # --- Eval Step (NEU) ---
    def eval_step(params, x, last_idxs, attn):
        # Nur Vorhersage, keine Gradienten
        logits = compute_logits(params, model_static, x, last_idxs, attn)
        # Sigmoid für Wahrscheinlichkeiten
        probs = jax.nn.sigmoid(logits)
        return probs

    # PMAP
    train_step_pmap = jax.pmap(train_step, axis_name="i", donate_argnums=(0, 3))
    eval_step_pmap = jax.pmap(eval_step, axis_name="i") # Keine donation hier nötig

    return train_step_pmap, eval_step_pmap

# ---------------------------------------------------------
# 4. Helper: Confusion Matrix
# ---------------------------------------------------------
def print_confusion_matrix(y_true_list, y_prob_list, threshold=0.5):
    """
    Berechnet und druckt eine Confusion Matrix basierend auf Listen von Werten.
    """
    y_true = np.concatenate(y_true_list).flatten()
    y_prob = np.concatenate(y_prob_list).flatten()
    y_pred = (y_prob >= threshold).astype(int)

    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\n" + "="*30)
    print(f" CONFUSION MATRIX (Thresh={threshold})")
    print("="*30)
    print(f"             | Pred NO (0) | Pred YES (1)")
    print(f" True NO (0) | {tn:^11} | {fp:^12}")
    print(f" True YES(1) | {fn:^11} | {tp:^12}")
    print("-" * 30)
    print(f" Accuracy:  {accuracy:.4f}")
    print(f" Precision: {precision:.4f}")
    print(f" Recall:    {recall:.4f}")
    print(f" F1 Score:  {f1:.4f}")
    print("="*30 + "\n")

# ---------------------------------------------------------
# 5. Main
# ---------------------------------------------------------
def run_evaluation(files, labels, batch_size, n_devices, devices, eval_step_pmap, params_repl, attn_repl, desc="Validierung"):
    """
    Führt Inference auf einem Datensatz aus und gibt Metriken aus.
    """
    print(f"--- Starte {desc} ---")
    y_true_all = []
    y_prob_all = []
    
    # Keine Randomisierung nötig bei Validierung/Test
    idxs = np.arange(len(files))
    
    for i in range(0, len(files), batch_size):
        batch_idxs = idxs[i:i+batch_size]
        # Wir müssen sicherstellen, dass wir volle Batches für pmap haben
        # oder Padding verwenden, wenn der letzte Batch zu klein ist.
        # Hier: Einfachster Weg -> Droppen wenn zu klein (für Validierung okay bei großen Daten)
        # Besser: Padding (aber komplexer). Wir droppen hier für Stabilität des Codes.
        if len(batch_idxs) < batch_size:
            continue

        seqs, batch_labels = [], []
        for idx in batch_idxs:
            p = files[idx]
            l_csv = labels[idx] if idx < len(labels) else None
            arr, label = load_npz(p, l_csv)
            seqs.append(arr)
            batch_labels.append(label)

        # Daten vorbereiten
        batch_x, batch_y, _, last_idxs = pad_batch_classification(seqs, batch_labels, expected_cols=40)

        # Auf Devices verteilen
        x_sharded = shard_array(batch_x, n_devices)
        # batch_y brauchen wir nicht auf GPU für Inference, nur für Metrik später
        last_idxs_sharded = shard_array(last_idxs, n_devices)

        x_dev = device_put_sharded_from_numpy(x_sharded, devices)
        last_idxs_dev = device_put_sharded_from_numpy(last_idxs_sharded, devices)

        # Inference Step
        # probs_dev ist sharded (n_devices, local_batch, 1)
        probs_dev = eval_step_pmap(params_repl, x_dev, last_idxs_dev, attn_repl)
        
        # Zurück zu CPU (als Liste von NumPy Arrays)
        probs_np = jax.device_get(probs_dev) # Shape (Devices, LocalBatch, 1)
        
        # Daten sammeln
        y_true_all.append(batch_y) # batch_y ist (GlobalBatch, 1)
        y_prob_all.append(probs_np.reshape(-1, 1)) # Flattens devices dimension

    if len(y_true_all) > 0:
        print_confusion_matrix(y_true_all, y_prob_all)
    else:
        print("Keine Daten evaluiert (Batch Size Mismatch?).")

def main():
    n_devices = jax.local_device_count()
    if n_devices < 2:
        print(f"WARNUNG: Nur {n_devices} Device(s). Sharding erwartet min 2.")
    devices = jax.local_devices()[:n_devices]
    print(f"Verwende {n_devices} Devices.")

    key = jrandom.PRNGKey(42)
    model_key, _ = jrandom.split(key)

    # 1. Daten laden und splitten
    # Die Funktion list_npz_files gibt jetzt Split-Listen zurück
    train_files, train_labels, val_files, val_labels, test_files, test_labels = list_npz_files("npz_dir/index.csv")

    # 2. Modell Init
    hidden_dim = 2
    model = SepsisClassifier(hidden_dim=hidden_dim, key=model_key)
    params, model_static = eqx.partition(model, eqx.is_array)

    optimizer = optax.adam(1e-4) # Etwas konservativer als 1e-3
    opt_state = optimizer.init(params)

    params_repl = jax.device_put_replicated(params, devices)
    opt_state_repl = jax.device_put_replicated(opt_state, devices)

    # 3. Funktionen erstellen
    train_step_pmap, eval_step_pmap = make_fns(model_static, optimizer)

    per_device_batch = 4
    batch_size = per_device_batch * n_devices
    n_epochs = 3

    # Attn Platzhalter (statisch)
    attn = jnp.zeros((hidden_dim*hidden_dim,), dtype=jnp.float32)
    attn_repl = jnp.broadcast_to(attn, (n_devices,) + attn.shape)

    print(f"Starte Training über {n_epochs} Epochen mit Batch-Size {batch_size}...")
    
    for epoch in range(n_epochs):
        start_t = time.time()
        epoch_losses = []
        
        # Training Shuffle
        perm = np.random.permutation(len(train_files))

        # --- TRAINING LOOP ---
        for i in range(0, len(train_files), batch_size):
            batch_idxs = perm[i:i+batch_size]
            if len(batch_idxs) < batch_size: break

            seqs, batch_labels = [], []
            for idx in batch_idxs:
                p = train_files[idx]
                l = train_labels[idx]
                arr, label = load_npz(p, l)
                seqs.append(arr)
                batch_labels.append(label)

            batch_x, batch_y, _, last_idxs = pad_batch_classification(seqs, batch_labels, expected_cols=40)

            x_shard = shard_array(batch_x, n_devices)
            y_shard = shard_array(batch_y, n_devices)
            last_idxs_shard = shard_array(last_idxs, n_devices)

            x_dev = device_put_sharded_from_numpy(x_shard, devices)
            y_dev = device_put_sharded_from_numpy(y_shard, devices)
            last_idxs_dev = device_put_sharded_from_numpy(last_idxs_shard, devices)

            params_repl, opt_state_repl, loss_val = train_step_pmap(
                params_repl, x_dev, y_dev, opt_state_repl, last_idxs_dev, attn_repl
            )
            epoch_losses.append(loss_val[0].item())

        train_dur = time.time() - start_t
        print(f"Epoch {epoch+1:02d} | Loss: {np.mean(epoch_losses):.5f} | Dauer: {train_dur:.1f}s")

        # --- VALIDATION LOOP (Confusion Matrix) ---
        run_evaluation(val_files, val_labels, batch_size, n_devices, devices, 
                       eval_step_pmap, params_repl, attn_repl, desc=f"Validierung (Epoch {epoch+1})")

    # --- FINAL TEST SET EVALUATION ---
    print("\nTraining beendet. Evaluiere Test-Set...")
    run_evaluation(test_files, test_labels, batch_size, n_devices, devices, 
                   eval_step_pmap, params_repl, attn_repl, desc="Final Test Set")

    return 0

if __name__ == "__main__":
    main()