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

import renamedv43 as ace_node
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
HIDDEN_DIM = 256 
BATCH_SIZE_PER_GPU = 256
Epochen = 10
Learning_Rate = 1e-3
sepsis_Amp = 18
num_workers = 0
seed = int(time.time())
dropout = 0.4
gamma = 4
VECTOR_FIELD_DEPTH = 3      # Tiefe des MLP im Vector Field (3 = Input → Hidden → Hidden → Output)
VECTOR_FIELD_WIDTH = 64     # Breite der Hidden Layers im Vector Field MLP

# ---------------------------------------------------------
# 1. Wrapper-Modell für Klassifikation
# ---------------------------------------------------------
class SepsisClassifier(eqx.Module):
    node: ace_node.ACE_NODE
    readout: eqx.nn.Linear
    attn_param: jnp.ndarray 
    dropout: eqx.nn.Dropout  # <--- NEU

    # In SepsisClassifier:

    def __init__(self, hidden_dim, key, vector_field_depth=VECTOR_FIELD_DEPTH, vector_field_width=VECTOR_FIELD_WIDTH):
        k1, k2, k3 = jrandom.split(key, 3)
        self.node = ace_node.ACE_NODE(hidden_dim=hidden_dim, key=k1, 
                                       vector_field_depth=vector_field_depth,
                                       vector_field_width=vector_field_width)
        
        # Readout mit Bias-Fix (wie vorher besprochen)
        temp_readout = eqx.nn.Linear(hidden_dim, 1, key=k2)
        initial_bias = jnp.array([-2.94]) # Startet mit niedriger Wahrscheinlichkeit für Sepsis
        self.readout = eqx.tree_at(lambda l: l.bias, temp_readout, initial_bias)
        
        # --- FIX 2: Better Initialization ---
        # Xavier/Glorot-ähnliche Skalierung für die Matrix.
        # Statt 0.01 nehmen wir 1 / sqrt(hidden_dim), aber noch etwas kleiner für Stabilität
        scale = 0.5 / np.sqrt(hidden_dim)
        self.attn_param = jrandom.normal(k3, (hidden_dim * hidden_dim,)) * scale
        
        self.dropout = eqx.nn.Dropout(p=dropout)


    def __call__(self, x, y0, key=None, inference=False):
        output_seq = self.node(x, y0, self.attn_param) 
        
        # Dropout auf den hidden state anwenden, bevor es in den Readout geht
        # Da output_seq eine Sequenz ist, vmappen wir das Dropout nicht zwingend, 
        # aber meist wendet man es auf die Features an.
        
        # Wir wenden Dropout hier auf die gesamte Sequenz gleich an oder pro Step.
        # Einfachheitshalber: Dropout vor dem Readout Call.
        
        def apply_readout(h, k):
            h_drop = self.dropout(h, key=k, inference=inference)
            return self.readout(h_drop)

        # Wir müssen Keys splitten, da wir vmap nutzen
        keys = jrandom.split(key, output_seq.shape[0]) if key is not None else None
        
        if inference:
            # Bei Inference brauchen wir keine Keys
            logits_seq = jax.vmap(lambda h: self.readout(h))(output_seq)
        else:
            # Bei Training mit Dropout
            logits_seq = jax.vmap(apply_readout)(output_seq, keys)
            
        return logits_seq

# ---------------------------------------------------------
# 2. Verbesserte Data Loading Pipeline (Torch Dataset)
# ---------------------------------------------------------

def compute_global_stats(file_paths, input_dim, sample_size=10000):
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
    n_val = int(len(paths) * 0.001)
    
    return (paths[:n_train], labels[:n_train],
            paths[n_train:n_train+n_val], labels[n_train:n_train+n_val],
            paths[n_train+n_val:], labels[n_train+n_val:])

# ---------------------------------------------------------
# 3. Training & Eval Logic (PMAP behalten)
# ---------------------------------------------------------
def make_fns(model_static, optimizer):

    # --- Compute Logits (Korrigiert) ---
    # In make_fns:
    
    def compute_logits(params, model_static, x, last_idxs, key=None, inference=False):
        model = eqx.combine(params, model_static)
        
        if key is not None:
            batch_keys = jrandom.split(key, x.shape[0])
        else:
            dummy_key = jrandom.PRNGKey(0)
            batch_keys = jrandom.split(dummy_key, x.shape[0])

        def single_call(xi, ki):
            y0 = jnp.zeros((model.node.hidden_dim,), dtype=jnp.float32)
            return model(xi, y0, key=ki, inference=inference)
        
        logits_seq_pred = jax.vmap(single_call)(x, batch_keys)
        
        batch_indices = jnp.arange(x.shape[0])
        final_logits = logits_seq_pred[batch_indices, last_idxs, 0]
        
        # --- FIX 1: LOGIT CLAMPING ---
        # Zwingt Werte in den Bereich [-10, 10]. 
        # Sigmoid(-10) ist ~0.000045, Sigmoid(10) ist ~0.99995. Das reicht völlig.
        # Verhindert Loss-Explosionen.
        final_logits = jnp.clip(final_logits, -20.0, 10.0)
        
        return final_logits.reshape(-1, 1)

    # In make_fns:

    # --- Focal Loss Implementation ---
    def sigmoid_focal_loss(logits, labels, alpha=0.25, gamma=gamma):
        """
        Alpha: Gewichtung der positiven Klasse (gegen Imbalance)
        Gamma: Fokus auf harte Beispiele (gegen "einfache" Nullen)
        """
        p = jax.nn.sigmoid(logits)
        # Formel: -alpha * (1-p)^gamma * log(p)   [für y=1]
        #         -(1-alpha) * p^gamma * log(1-p) [für y=0]
        
        ce_loss = optax.sigmoid_binary_cross_entropy(logits, labels)
        p_t = (labels * p) + ((1 - labels) * (1 - p))
        alpha_t = (labels * alpha) + ((1 - labels) * (1 - alpha))
        
        focal_factor = (1 - p_t) ** gamma
        return alpha_t * focal_factor * ce_loss

    @eqx.filter_value_and_grad
    def loss_fn(params, model_static, x, y, last_idxs, key):
        preds = compute_logits(params, model_static, x, last_idxs, key=key, inference=False)
        # Nutze Focal Loss statt gewichtetem BCE
        # alpha=0.8 gibt der Sepsis-Klasse (1) mehr Gewicht, gamma=2 fokussiert auf Fehler
        loss = sigmoid_focal_loss(preds, y, alpha=0.80, gamma=gamma)
        return jnp.mean(loss)

    # --- Train Step ---
    def train_step(params, x, y, opt_state, last_idxs, key): # Key als Argument
        loss, grads = loss_fn(params, model_static, x, y, last_idxs, key)
        
        grads = jax.tree.map(lambda g: lax.pmean(g, axis_name="i"), grads)
        loss_mean = lax.pmean(loss, axis_name="i")
        
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_mean

    # --- Eval Step ---
    def eval_step(params, x, last_idxs):
        # Inference=True -> Dropout ist aus
        logits = compute_logits(params, model_static, x, last_idxs, key=None, inference=True)
        probs = jax.nn.sigmoid(logits)
        return probs

    # PMAP: Key muss auch gesplitet/repliziert werden? 
    # Nein, wir generieren in Main einen Key pro Device und geben ihn rein.
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

    print(f"TN={tn}  | FP={fp} \nFN={fn}   | TP={tp} | Acc={acc:.2f} F1={f1:.2f} Recall={rec:.2f}, Learnungrate: {Learning_Rate}")
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
                            num_workers=num_workers, drop_last=False)

    # 4. Modell & Optimizer
    model = SepsisClassifier(hidden_dim=HIDDEN_DIM, key=model_key, 
                             vector_field_depth=VECTOR_FIELD_DEPTH,
                             vector_field_width=VECTOR_FIELD_WIDTH)
    params, model_static = eqx.partition(model, eqx.is_array)

    # Scheduler: Startet bei Learning_Rate und geht runter
    # Das verhindert das starke Schwanken am Ende
    #total_steps = current_epochs * (len(files_tr) // global_batch_size)
    #schedule = optax.cosine_decay_schedule(init_value=current_lr, decay_steps=total_steps, alpha=0.01)

   # In main():
    
   # In main() deiner ACE_NODEoptimized_v3.py:

    # 1. Schritte berechnen
    steps_per_epoch = len(files_tr) // global_batch_size
    if steps_per_epoch == 0: steps_per_epoch = 1
    total_steps = Epochen * steps_per_epoch
    warmup_steps = 5 * steps_per_epoch  # 3 Epochen Warmup auf die Peak-LR

    # 2. Scheduler definieren
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-7,          # Start fast bei 0
        peak_value=Learning_Rate, # Geht hoch auf z.B. 0.0001
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=1e-8            # Am Ende fast 0
    )

    # 3. Optimizer mit dem Schedule füttern
    optimizer = optax.chain(
        optax.clip_by_global_norm(0.1), # Strenges Clipping für ODE-Stabilität
        optax.adamw(learning_rate=lr_schedule, weight_decay=1e-3)
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

    # In main(), VOR dem Loop:
    # Haupt-Key für Dropout
    dropout_key = jrandom.PRNGKey(seed + 1)

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
            
            bx_shard = bx_np.reshape(n_devices, BATCH_SIZE_PER_GPU, MAX_SEQ_LEN, INPUT_DIM)
            by_shard = by_np.reshape(n_devices, BATCH_SIZE_PER_GPU, 1)
            blast_shard = blast_np.reshape(n_devices, BATCH_SIZE_PER_GPU)

            # --- FIX 3: Correct PRNG Handling ---
            # 1. Splitte den Haupt-Key in (Neuer Haupt-Key, Sub-Key für diesen Step)
            dropout_key, step_key = jrandom.split(dropout_key)
            
            # 2. Splitte den Step-Key auf die Devices
            step_keys_shard = jrandom.split(step_key, n_devices)

            # Train Step Aufruf mit den korrekten Keys
            params_repl, opt_state_repl, loss_val = train_step_pmap(
                params_repl, bx_shard, by_shard, opt_state_repl, blast_shard, step_keys_shard
            )
            
            # Check auf NaN im Loss (Optionaler Not-Halt)
            loss_cpu = loss_val[0].item()
            if np.isnan(loss_cpu) or loss_cpu > 1e6:
                print(f"\nWARNUNG: Loss Explosion in Batch {i}! ({loss_cpu})")
                # Hier könnte man abbrechen, wir loggen erstmal nur
            
            epoch_losses.append(loss_cpu)
            
            if i % 10 == 0: # Nicht jeden Batch printen, spart CPU
                print(f"\rEp {epoch+1}/{current_epochs} | Batch {i} | Loss {loss_cpu:.4f}", end="")

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
    test_loader = DataLoader(test_ds, batch_size=global_batch_size, shuffle=False, drop_last=False)
    run_evaluation(test_loader, n_devices, eval_step_pmap, params_repl, desc="FINAL TEST SET")

    print("\nSpeichere Ergebnisse...")
    # Parameter von Device 0 zurückholen
    final_params = jax.device_get(jax.tree.map(lambda x: x[0], params_repl))
    flat_params, _ = jax.tree_util.tree_flatten(final_params)
    
    np.savez("training_history.npz", **{k: np.array(v) for k, v in history.items()})
    np.savez("sepsis_weights_flat.npz", *flat_params)
    
    return 0

if __name__ == "__main__":
    # Dieser Block verhindert, dass Worker-Prozesse das Skript neu starten
    
    # Optional: Start-Methode für Multiprocessing fixieren (nur nötig wenn num_workers > 0)
    import multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    main()