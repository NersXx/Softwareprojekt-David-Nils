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

#Torch nur für DataLoader (Multi-Processing Data Loading)
import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import roc_auc_score, roc_curve

import ACE_NODEv42 as ace_node

if jax.device_count == 1:
    jax.config.update("jax_platform_name", "cpu")
    
# --- SETTINGS ---
MAX_SEQ_LEN = 200  
INPUT_DIM = 40
HIDDEN_DIM = 64     
BATCH_SIZE_PER_GPU = 32
Epochen = 10
Learning_Rate = 1e-4
sepsis_Amp = 18 

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
# 2. Verbesserte Data Loading Pipeline (Torch Dataset)
# ---------------------------------------------------------
class NPZDataset(Dataset):
    def __init__(self, files, labels, max_len=MAX_SEQ_LEN, input_dim=INPUT_DIM):
        self.files = files
        self.labels = labels
        self.max_len = max_len
        self.input_dim = input_dim

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # 1. Laden
        path = self.files[idx]
        try:
            d = np.load(path, allow_pickle=True)
            if "data_raw" in d: arr = d["data_raw"]
            elif "data_median" in d: arr = d["data_median"]
            else: arr = d[d.files[0]]
        except:
            # Fallback bei defekten Files
            arr = np.zeros((10, self.input_dim))
            
        arr = np.nan_to_num(arr.astype(np.float32), nan=0.0)

        # 2. Label
        l_val = self.labels[idx]
        label = 1.0 if float(l_val) > 0 else 0.0

        # 3. FIXES PADDING (Entscheidend für Speed!)
        seq_len = arr.shape[0]
        feat_dim = arr.shape[1]

        # Features anpassen
        if feat_dim < self.input_dim:
            pad = np.zeros((seq_len, self.input_dim - feat_dim), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=1)
        elif feat_dim > self.input_dim:
            arr = arr[:, :self.input_dim]

        # Zeitachse anpassen (Cut or Pad)
        x_out = np.zeros((self.max_len, self.input_dim), dtype=np.float32)
        
        real_len = min(seq_len, self.max_len)
        x_out[:real_len] = arr[:real_len]
        
        # Index des letzten Elements für Prediction
        last_idx = real_len - 1 if real_len > 0 else 0

        return x_out, np.array([label], dtype=np.float32), last_idx

def list_npz_files_split(index_path="npz_dir/index.csv"):
    df = pd.read_csv(index_path)
    # Filtern nach existierenden Files
    valid_mask = [os.path.exists(p) for p in df["npz_path"]]
    df = df[valid_mask]
    
    paths = df["npz_path"].tolist()
    labels = df["label"].tolist() if "label" in df.columns else [0] * len(paths)
    
    # Shuffle
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(paths))
    paths = [paths[i] for i in perm]
    labels = [labels[i] for i in perm]

    n_train = int(len(paths) * 0.8)
    n_val = int(len(paths) * 0.1)
    
    return (paths[:n_train], labels[:n_train],
            paths[n_train:n_train+n_val], labels[n_train:n_train+n_val],
            paths[n_train+n_val:], labels[n_train+n_val:])

# ---------------------------------------------------------
# 3. Training & Eval Logic (PMAP behalten)
# ---------------------------------------------------------
def make_fns(model_static, optimizer):

    # --- Core Logic ---
    def compute_logits(params, model_static, x, last_idxs, attn):
        model = eqx.combine(params, model_static)
        def single_call(xi):
            # y0 Startwert
            y0 = jnp.zeros((model.node.hidden_dim,), dtype=jnp.float32)
            logits_seq = model(xi, y0, attn)
            return logits_seq
        
        logits_seq_pred = jax.vmap(single_call)(x)
        
        # Zugriff auf letzten Zeitschritt
        batch_indices = jnp.arange(x.shape[0])
        final_logits = logits_seq_pred[batch_indices, last_idxs, 0]
        return final_logits.reshape(-1, 1)

    # --- Loss ---
    @eqx.filter_value_and_grad
    def loss_fn(params, model_static, x, y, last_idxs, attn):
        preds = compute_logits(params, model_static, x, last_idxs, attn)
        loss = optax.sigmoid_binary_cross_entropy(preds, y)
        
        # Gewichtung
        pos_weight = sepsis_Amp
        weights = jnp.where(y == 1, pos_weight, 1.0)
        return jnp.mean(loss * weights)

    # --- Train Step ---
    def train_step(params, x, y, opt_state, last_idxs, attn):
        loss, grads = loss_fn(params, model_static, x, y, last_idxs, attn)
        # Gradienten über alle GPUs mitteln (pmean)
        grads = jax.tree.map(lambda g: lax.pmean(g, axis_name="i"), grads)
        loss_mean = lax.pmean(loss, axis_name="i")
        
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_mean

    # --- Eval Step ---
    def eval_step(params, x, last_idxs, attn):
        logits = compute_logits(params, model_static, x, last_idxs, attn)
        probs = jax.nn.sigmoid(logits)
        return probs

    # PMAP definition (Parallelisierung über Devices)
    train_step_pmap = jax.pmap(train_step, axis_name="i", donate_argnums=(0, 3))
    eval_step_pmap = jax.pmap(eval_step, axis_name="i")

    return train_step_pmap, eval_step_pmap

# ---------------------------------------------------------
# 4. Helper
# ---------------------------------------------------------
def print_confusion_matrix(y_true_list, y_prob_list, threshold=0.5):
    y_true = np.concatenate(y_true_list).flatten()
    y_prob = np.concatenate(y_prob_list).flatten()
    y_pred = (y_prob >= threshold).astype(int)

    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)

    print(f"TN={tn} | FP={fp} \nFN={fn}   | TP={tp} | Acc={acc:.2f} F1={f1:.2f} Recall={rec:.2f}")
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "acc": acc, "recall": rec, "f1": f1}

# ---------------------------------------------------------
# 5. Main
# ---------------------------------------------------------
def run_evaluation(dataloader, n_devices, eval_step_pmap, params_repl, attn_repl, desc="Eval"):
    print(f"--- {desc} ---")
    y_true_all, y_prob_all = [], []

    for batch in dataloader:
        bx, by, blast = batch
        # Konvertierung zu Numpy für Sharding
        bx, by, blast = bx.numpy(), by.numpy(), blast.numpy().astype(np.int32)
        
        # Reshape für pmap: (Devices, LocalBatch, ...)
        # Annahme: Dataloader Batchsize ist (Devices * LocalBatch)
        local_bs = bx.shape[0] // n_devices
        
        bx = bx.reshape(n_devices, local_bs, MAX_SEQ_LEN, INPUT_DIM)
        blast = blast.reshape(n_devices, local_bs)

        # Inference
        probs_dev = eval_step_pmap(params_repl, bx, blast, attn_repl)
        probs_np = jax.device_get(probs_dev).reshape(-1, 1) # Flatten GPU dim

        y_true_all.append(by)
        y_prob_all.append(probs_np)

    if len(y_true_all) > 0:
        return print_confusion_matrix(y_true_all, y_prob_all)
    return None

def main():
    # 1. Device Setup
    n_devices = jax.local_device_count()
    devices = jax.local_devices()
    print(f"Verwende {n_devices} Devices: {devices}")
    
    if n_devices < 2:
        print("WARNUNG: Weniger als 2 GPUs gefunden!")

    key = jrandom.PRNGKey(42)
    model_key, _ = jrandom.split(key)

    # 2. Daten laden (jetzt mit Split Funktion)
    files_tr, labs_tr, files_val, labs_val, files_test, labs_test = list_npz_files_split()

    # DataLoader Setup
    # Global Batch Size = Devices * Local Batch Size
    global_batch_size = n_devices * BATCH_SIZE_PER_GPU
    
    # num_workers=4 lädt Daten parallel im Hintergrund -> Viel schneller!
    train_ds = NPZDataset(files_tr, labs_tr)
    train_loader = DataLoader(train_ds, batch_size=global_batch_size, shuffle=True, 
                              num_workers=4, drop_last=True, pin_memory=True)
    
    val_ds = NPZDataset(files_val, labs_val)
    val_loader = DataLoader(val_ds, batch_size=global_batch_size, shuffle=False, 
                            num_workers=2, drop_last=True) # drop_last=True für pmap safety

    # 3. Modell Init
    model = SepsisClassifier(hidden_dim=HIDDEN_DIM, key=model_key)
    params, model_static = eqx.partition(model, eqx.is_array)

    optimizer = optax.adam(Learning_Rate) # LR etwas reduziert wegen größerem Hidden Dim
    opt_state = optimizer.init(params)

    history = {
        "loss": [], "acc": [], "tp": [], "tn": [], "fp": [], "fn": [], "recall": []
    }

    if n_devices == 1:
            # Auf Single-CPU: Wir fügen manuell eine Dimension vorne an (Axis 0)
            # Aus Shape (H,) wird (1, H,). Das simuliert 1 Device für pmap.
            params_repl = jax.tree.map(lambda x: jnp.expand_dims(x, axis=0), params)
            opt_state_repl = jax.tree.map(lambda x: jnp.expand_dims(x, axis=0), opt_state)
            
            attn = jnp.zeros((HIDDEN_DIM*HIDDEN_DIM,), dtype=jnp.float32)
            attn_repl = jnp.expand_dims(attn, axis=0)
    else:
            # Auf Multi-GPU: Wir nutzen die echte Replikation
            params_repl = jax.device_put_replicated(params, devices)
            opt_state_repl = jax.device_put_replicated(opt_state, devices)
            
            attn = jnp.zeros((HIDDEN_DIM*HIDDEN_DIM,), dtype=jnp.float32)
            attn_repl = jax.device_put_replicated(attn, devices)

    # 4. Funktionen
    train_step_pmap, eval_step_pmap = make_fns(model_static, optimizer)

    n_epochs = Epochen
    print(f"Starte Training: {n_epochs} Epochen, Batch={global_batch_size}, MaxLen={MAX_SEQ_LEN}, LR={Learning_Rate}, HiddenDim={HIDDEN_DIM}, Sep_Amp={sepsis_Amp}")
    
    for epoch in range(n_epochs):
        start_t = time.time()
        epoch_losses = []
        
        # --- TRAINING LOOP MIT DATALOADER ---
        for i, batch in enumerate(train_loader):
            # Batch kommt aus Torch DataLoader als Tensor -> nach Numpy wandeln
            bx, by, blast = batch
            bx = bx.numpy()
            by = by.numpy()
            blast = blast.numpy().astype(np.int32)
            
            # Reshape für PMAP: (n_devices, batch_per_device, ...)
            bx_shard = bx.reshape(n_devices, BATCH_SIZE_PER_GPU, MAX_SEQ_LEN, INPUT_DIM)
            by_shard = by.reshape(n_devices, BATCH_SIZE_PER_GPU, 1)
            blast_shard = blast.reshape(n_devices, BATCH_SIZE_PER_GPU)

            # Auf GPU schieben geschieht implizit beim Aufruf von pmap mit numpy arrays
            # (oder man nutzt jax.device_put_sharded, aber pmap handled das gut)
            
            params_repl, opt_state_repl, loss_val = train_step_pmap(
                params_repl, bx_shard, by_shard, opt_state_repl, blast_shard, attn_repl
            )
            
            # Loss ist (n_devices,). Nehmen wir den Durchschnitt.
            # Nur item() holen blockt async execution -> wir machen es nur für Logging ab und zu
            loss_val_cpu = loss_val[0].item() 
            epoch_losses.append(loss_val_cpu)
            
            #Fortschrittsanzeige
            if i % 10 == 0:
                print(f"\rEp {epoch+1} of {n_epochs};Batch {i} Loss {loss_val_cpu:.4f}", end="")

        train_dur = time.time() - start_t
        print(f"\nEpoch {epoch+1:02d} | Loss: {np.mean(epoch_losses):.5f} | Dauer: {train_dur:.1f}s")

        # Durchschnittlichen Loss der Epoche speichern
        history["loss"].append(float(np.mean(epoch_losses)))

        # Validierung durchführen und Ergebnisse abfangen
        res = run_evaluation(val_loader, n_devices, eval_step_pmap, params_repl, attn_repl, desc=f"Val Ep {epoch+1}")
        
        # Falls Metriken zurückgegeben wurden, in history speichern
        if res:
            for key in ["acc", "tp", "tn", "fp", "fn", "recall"]:
                history[key].append(res[key])

        # --- VALIDATION ---
        run_evaluation(val_loader, n_devices, eval_step_pmap, params_repl, attn_repl, desc=f"Val Ep {epoch+1}")

    print("\nTraining beendet.")
    # Finaler Test
    test_ds = NPZDataset(files_test, labs_test)
    test_loader = DataLoader(test_ds, batch_size=global_batch_size, shuffle=False, num_workers=2)
    run_evaluation(test_loader, n_devices, eval_step_pmap, params_repl, attn_repl, desc="FINAL TEST SET")
    # --- SPEICHERN DER GEWICHTE (ROBUSTE METHODE) ---
    print("\nSpeichere finale Gewichte...")
    
    # Parameter von den GPUs holen (nur von der ersten GPU)
    final_params = jax.device_get(jax.tree.map(lambda x: x[0], params_repl))
    
    # Den PyTree in Liste von Arrays umwandeln
    flat_params, treedef = jax.tree_util.tree_flatten(final_params)
    
    # Gesamte Historie als .npz Datei schreiben
    history_path = "training_history.npz"
    np.savez(history_path, **{k: np.array(v) for k, v in history.items()})
    print(f"Trainings-Historie erfolgreich unter '{history_path}' gespeichert!")

    # 3. Als NumPy-Archiv speichern
    # Wir speichern die Liste der Arrays. np.savez packt diese in eine einzige Datei.
    save_path = "sepsis_weights_flat.npz"
    np.savez(save_path, *flat_params)
    print(f"Gewichte erfolgreich als NumPy-Archiv unter '{save_path}' gespeichert!")
    with open("model_config.txt", "w") as f:
        f.write(f"hidden_dim: {HIDDEN_DIM}\n")
        f.write(f"max_seq_len: {MAX_SEQ_LEN}\n")

    return 0

if __name__ == "__main__":
    main()