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
import multiprocessing as mp

"""if __name__ == "__main__":
    # Stellt sicher, dass DataLoader keine Deadlocks mit JAX verursacht
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
"""
if jax.device_count == 1:
    jax.config.update("jax_platform_name", "cpu")
    
# --- SETTINGS ---
MAX_SEQ_LEN = 100
INPUT_DIM = 40
HIDDEN_DIM = 128 
BATCH_SIZE_PER_GPU = 512
Epochen = 50
Learning_Rate = 5e-4
sepsis_Amp = 1
num_workers = 0
seed = int(time.time())

# ---------------------------------------------------------
# 1. Wrapper-Modell für Klassifikation
# ---------------------------------------------------------
class SepsisClassifier(eqx.Module):
    node: ace_node.ACE_NODE
    readout: eqx.nn.Linear
    # NEU: Lernbarer Attention Context
    attn_param: jnp.ndarray 

    def __init__(self, hidden_dim, key):
        k1, k2, k3 = jrandom.split(key, 3)
        self.node = ace_node.ACE_NODE(hidden_dim=hidden_dim, key=k1)
        self.readout = eqx.nn.Linear(hidden_dim, 1, key=k2)
        # Initialisiere zufällig (klein), damit Symmetrie gebrochen wird
        self.attn_param = jrandom.normal(k3, (hidden_dim * hidden_dim,)) * 0.01

    def __call__(self, x, y0, attn=None): # attn Argument ignorieren wir hier, wir nutzen self.attn_param
        # Wir nutzen den gelernten Parameter
        output_seq = self.node(x, y0, self.attn_param) 
        logits_seq = jax.vmap(self.readout)(output_seq)
        return logits_seq

# ---------------------------------------------------------
# 2. Verbesserte Data Loading Pipeline (Torch Dataset)
# ---------------------------------------------------------

def compute_global_stats(file_paths, input_dim, sample_size=500):
    """
    Berechnet approximativen Mean und Std über eine zufällige Auswahl an Files.
    """
    print("Berechne globale Normalisierungs-Statistiken...")
    buffer = []
    # Wir nehmen max 500 Files, das reicht für eine gute Schätzung
    indices = np.random.choice(len(file_paths), size=min(len(file_paths), sample_size), replace=False)
    
    for i in indices:
        try:
            d = np.load(file_paths[i], allow_pickle=True)
            if "data_raw" in d: arr = d["data_raw"]
            elif "data_median" in d: arr = d["data_median"]
            else: arr = d[d.files[0]]
            
            # Features abschneiden falls nötig, damit Dimensionen passen
            if arr.shape[1] >= input_dim:
                 arr = arr[:, :input_dim]
                 buffer.append(arr)
        except:
            continue
            
    all_data = np.concatenate(buffer, axis=0)
    # NaN zu 0 vor der Berechnung, oder ignorieren
    all_data = np.nan_to_num(all_data)
    
    mean = np.mean(all_data, axis=0)
    std = np.std(all_data, axis=0)
    
    # Verhindere Division durch Null bei konstanten Features
    std[std < 1e-5] = 1.0 
    
    print("Stats berechnet.")
    return mean.astype(np.float32), std.astype(np.float32)

class NPZDataset(Dataset):
    def __init__(self, files, labels, mean, std, max_len=MAX_SEQ_LEN, input_dim=INPUT_DIM):
        self.files = files
        self.labels = labels
        self.max_len = max_len
        self.input_dim = input_dim
        self.mean = mean
        self.std = std

    def __len__(self):
        # Gibt die Anzahl der Dateien im Datensatz zurück
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        try:
            d = np.load(path, allow_pickle=True)
            if "data_raw" in d: arr = d["data_raw"]
            elif "data_median" in d: arr = d["data_median"]
            else: arr = d[d.files[0]]
        except:
            arr = np.zeros((10, self.input_dim))
            
        arr = np.nan_to_num(arr.astype(np.float32), nan=0.0)

        seq_len = arr.shape[0]
        feat_dim = arr.shape[1]

        if feat_dim < self.input_dim:
            pad = np.zeros((seq_len, self.input_dim - feat_dim), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=1)
        elif feat_dim > self.input_dim:
            arr = arr[:, :self.input_dim]

        # Normalisierung
        arr = (arr - self.mean) / self.std

        x_out = np.zeros((self.max_len, self.input_dim), dtype=np.float32)
        real_len = min(seq_len, self.max_len)
        x_out[:real_len] = arr[:real_len]
        
        l_val = self.labels[idx]
        label = 1.0 if float(l_val) > 0 else 0.0
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
    rng = np.random.RandomState(seed)
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
    # 'attn' Argument entfernen
    def compute_logits(params, model_static, x, last_idxs):
        model = eqx.combine(params, model_static)
        
        def single_call(xi):
            y0 = jnp.zeros((model.node.hidden_dim,), dtype=jnp.float32)
            # Hier kein 'attn' mehr übergeben, das Modell hat es intern
            logits_seq = model(xi, y0) 
            return logits_seq
        
        logits_seq_pred = jax.vmap(single_call)(x)
        
        batch_indices = jnp.arange(x.shape[0])
        final_logits = logits_seq_pred[batch_indices, last_idxs, 0]
        return final_logits.reshape(-1, 1)

    # --- Loss ---
    @eqx.filter_value_and_grad
    def loss_fn(params, model_static, x, y, last_idxs):
        preds = compute_logits(params, model_static, x, last_idxs)
        loss = optax.sigmoid_binary_cross_entropy(preds, y)
        
        # Gewichtung
        pos_weight = sepsis_Amp
        weights = jnp.where(y == 1, pos_weight, 1.0)
        return jnp.mean(loss * weights)

    # --- Train Step ---
    def train_step(params, x, y, opt_state, last_idxs):
        loss, grads = loss_fn(params, model_static, x, y, last_idxs)
        # Gradienten über alle GPUs mitteln (pmean)
        grads = jax.tree.map(lambda g: lax.pmean(g, axis_name="i"), grads)
        loss_mean = lax.pmean(loss, axis_name="i")
        
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_mean

    # --- Eval Step ---
    def eval_step(params, x, last_idxs):
        logits = compute_logits(params, model_static, x, last_idxs)
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

    print(f"TN={tn}   | FP={fp} \nFN={fn}   | TP={tp} | Acc={acc:.2f} F1={f1:.2f} Recall={rec:.2f}")
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "acc": acc, "recall": rec, "f1": f1}

# ---------------------------------------------------------
# 5. Main
# ---------------------------------------------------------
def run_evaluation(dataloader, n_devices, eval_step_pmap, params_repl, desc="Eval"):
    print(f"--- {desc} ---")
    y_true_all, y_prob_all = [], []

    for batch in dataloader:
        bx, by, blast = batch
        bx_np = bx.numpy()
        blast_np = blast.numpy().astype(np.int32)
        
        local_bs = bx_np.shape[0] // n_devices
        bx_shard = bx_np.reshape(n_devices, local_bs, MAX_SEQ_LEN, INPUT_DIM)
        blast_shard = blast_np.reshape(n_devices, local_bs)

        # Kein attn_repl mehr übergeben
        probs_dev = eval_step_pmap(params_repl, bx_shard, blast_shard)
        probs_np = jax.device_get(probs_dev).reshape(-1, 1)

        y_true_all.append(by.numpy().flatten())
        y_prob_all.append(probs_np.flatten())

    if len(y_true_all) > 0:
        y_true = np.concatenate(y_true_all)
        y_prob = np.concatenate(y_prob_all)
        auc_score = roc_auc_score(y_true, y_prob)
        metrics_dict = print_confusion_matrix([y_true], [y_prob])
        metrics_dict["val_auc"] = auc_score
        return metrics_dict
    return None

def main():
    # 1. Device Setup
    n_devices = jax.local_device_count()
    devices = jax.local_devices()
    print(f"Verwende {n_devices} Devices: {devices}")
    
    # 2. Konstanten & Batch-Berechnung
    # Nutze die Variablen aus deinem Header, stelle aber Konsistenz sicher
    current_lr = Learning_Rate 
    current_epochs = Epochen 
    global_batch_size = BATCH_SIZE_PER_GPU * n_devices
    
    print(f"Setup: {current_epochs} Epochen, Global Batch Size: {global_batch_size}")

    key = jrandom.PRNGKey(seed)
    model_key, _ = jrandom.split(key)

    # 3. Daten laden & Normalisierung
    files_tr, labs_tr, files_val, labs_val, files_test, labs_test = list_npz_files_split()

    # Statistiken nur auf Basis der Trainingsdaten berechnen
    global_mean, global_std = compute_global_stats(files_tr, INPUT_DIM)

    # Datasets mit Normalisierung
    train_ds = NPZDataset(files_tr, labs_tr, mean=global_mean, std=global_std)
    val_ds = NPZDataset(files_val, labs_val, mean=global_mean, std=global_std)
    test_ds = NPZDataset(files_test, labs_test, mean=global_mean, std=global_std)

    # Dataloader Initialisierung
    train_loader = DataLoader(train_ds, batch_size=global_batch_size, shuffle=True, 
                              num_workers=num_workers, drop_last=True)
    
    val_loader = DataLoader(val_ds, batch_size=global_batch_size, shuffle=False, 
                            num_workers=num_workers, drop_last=True)

    # 4. Modell & Optimizer
    model = SepsisClassifier(hidden_dim=HIDDEN_DIM, key=model_key)
    params, model_static = eqx.partition(model, eqx.is_array)

    # Gradient Clipping gegen Exploding Gradients
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0), 
        optax.adam(current_lr)
    )
    opt_state = optimizer.init(params)

    # Replikation für Multi-GPU (pmap)
    if n_devices == 1:
        params_repl = jax.tree.map(lambda x: jnp.expand_dims(x, axis=0), params)
        opt_state_repl = jax.tree.map(lambda x: jnp.expand_dims(x, axis=0), opt_state)
    else:
        params_repl = jax.device_put_replicated(params, devices)
        opt_state_repl = jax.device_put_replicated(opt_state, devices)

    # Kompilierung der JAX-Funktionen
    train_step_pmap, eval_step_pmap = make_fns(model_static, optimizer)

    history = {
        "loss": [], "acc": [], "tp": [], "tn": [], "fp": [], "fn": [], "recall": [], "val_auc": []
    }

    print(f"Start: LR={current_lr}, Hidden={HIDDEN_DIM}, Amp={sepsis_Amp}")
    
    # 5. Training Loop
    for epoch in range(current_epochs):
        start_t = time.time()
        epoch_losses = []
        
        for i, batch in enumerate(train_loader):
            bx, by, blast = batch
            bx_np = bx.numpy()
            by_np = by.numpy()
            blast_np = blast.numpy().astype(np.int32)
            
            # Sharding für die verfügbaren GPUs
            bx_shard = bx_np.reshape(n_devices, BATCH_SIZE_PER_GPU, MAX_SEQ_LEN, INPUT_DIM)
            by_shard = by_np.reshape(n_devices, BATCH_SIZE_PER_GPU, 1)
            blast_shard = blast_np.reshape(n_devices, BATCH_SIZE_PER_GPU)

            # ACE-Modell Training (attn ist nun in params_repl enthalten)
            params_repl, opt_state_repl, loss_val = train_step_pmap(
                params_repl, bx_shard, by_shard, opt_state_repl, blast_shard
            )
            
            epoch_losses.append(loss_val[0].item())
            
            if i % 1 == 0:
                print(f"\rEp {epoch+1}/{current_epochs} | Batch {i} | Loss {loss_val[0]:.4f}", end="")

        # Epochen-Statistik
        avg_loss = float(np.mean(epoch_losses))
        history["loss"].append(avg_loss)
        train_dur = time.time() - start_t
        print(f"\nEnde Epoche {epoch+1:02d} | Avg Loss: {avg_loss:.5f} | Zeit: {train_dur:.1f}s")
                
        # 6. Validation
        res = run_evaluation(val_loader, n_devices, eval_step_pmap, params_repl, desc=f"Val Ep {epoch+1}")
        
        if res:
            history["val_auc"].append(float(res.get("val_auc", 0)))
            history["acc"].append(float(res.get("acc", 0)))
            history["tp"].append(int(res.get("tp", 0)))
            history["tn"].append(int(res.get("tn", 0)))
            history["fp"].append(int(res.get("fp", 0)))
            history["fn"].append(int(res.get("fn", 0)))
            history["recall"].append(float(res.get("recall", 0)))

    # 7. Finaler Test & Speichern
    print("\n--- FINAL TEST ---")
    test_loader = DataLoader(test_ds, batch_size=global_batch_size, shuffle=False, drop_last=True)
    run_evaluation(test_loader, n_devices, eval_step_pmap, params_repl, desc="FINAL TEST SET")

    print("\nSpeichere Ergebnisse...")
    # Parameter von Device 0 zurückholen
    final_params = jax.device_get(jax.tree.map(lambda x: x[0], params_repl))
    flat_params, _ = jax.tree_util.tree_flatten(final_params)
    
    np.savez("training_history.npz", **{k: np.array(v) for k, v in history.items()})
    np.savez("sepsis_weights_flat.npz", *flat_params)
    
    return 0

if __name__ == "__main__":
    main()