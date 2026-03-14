import os
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import gc
import psutil
import subprocess
import csv

import jax

# --- DEADLOCK-PREVENTION ---
# Erhöhe den Rendezvous-Timeout für Collective Operations
os.environ['XLA_RENDEZVOUS_WAIT_SECONDS'] = '120'
# Deaktiviere aggressives Optimierungen die Deadlocks auslösen können
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=1'

import jax.numpy as jnp
from jax import random as jrandom, lax

import equinox as eqx
import optax

#Torch nur für DataLoader (Multi-Processing Data Loading)
import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import roc_auc_score, roc_curve

import BASE_model as nodemod
import multiprocessing as mp

if jax.device_count == 1:
    jax.config.update("jax_platform_name", "cpu")
    
# --- SETTINGS ---
MAX_SEQ_LEN = 100
INPUT_DIM = 80
HIDDEN_DIM = 40
BATCH_SIZE_PER_GPU = 256        # wenn cpu, dann gesamte Batch Size
Epochen = 50
Warmup_epochs = 5
Learning_Rate = 1e-3
Final_LR = 1e-9
sepsis_Amp = 9.0      
num_workers = 0
seed = int(time.time())
dropout = 0.4       # Dropout der Neuronen im Readout Layer
gamma = 4           # Focal Loss Gamma (increase for harder focus on difficult samples)
alpha = None        # Focal Loss Alpha (if None, derived from sepsis_Amp)
snapshot = 25          # Speichere alle `snapshot` Epochen ein Modell
VECTOR_FIELD_DEPTH = 3      # Tiefe des MLP im Vector Field (3 = Input → Hidden → Hidden → Output)
VECTOR_FIELD_WIDTH = 64     # Breite der Hidden Layers im Vector Field MLP
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "files", "model_config.txt")  # Nutze skriptrelativen Pfad, damit es unabhängig vom Working Directory funktioniert

# --- Memory/Cache thresholds (configurable via environment variables) ---
# Absolute GB threshold to trigger cache clearing (default: 80 GB)
CACHE_CLEAR_RSS_GB = int(os.environ.get("CACHE_CLEAR_RSS_GB", "80"))
# Fraction of total system memory above which to clear caches (default: 0.8 => 80%)
CACHE_CLEAR_RATIO = float(os.environ.get("CACHE_CLEAR_RATIO", "0.8"))
# Growth factor over previous epoch RSS that triggers extra clearance (default: 1.10 => 10% growth)
CACHE_GROWTH_FACTOR = float(os.environ.get("CACHE_GROWTH_FACTOR", "1.10"))
# Lower threshold to log / be more aggressive (in GB) for dev/debug
CACHE_CLEAR_WARN_GB = int(os.environ.get("CACHE_CLEAR_WARN_GB", "60"))


# Speichere Seed in Config-Datei für Reproduzierbarkeit
try:
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    
    lines = []
    # Datei lesen, falls sie existiert
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r') as f:
            lines = f.readlines()

    # Den alten Seed suchen und entfernen oder direkt filtern
    # Wir behalten alle Zeilen, die NICHT mit "Seed:" anfangen
    new_lines = [line for line in lines if not line.startswith("Seed:")]
    # Füge den neuen Seed hinzu
    new_lines.append(f"Seed: {seed}\n")
    with open(CONFIG_PATH, 'w') as f:
        f.writelines(new_lines)
except Exception as e:
    print(f"Warnung: Konnte Seed nicht in {CONFIG_PATH} speichern: {e}")

# ---------------------------------------------------------
# 1. Wrapper-Modell für Klassifikation
# ---------------------------------------------------------
class SepsisClassifier(eqx.Module):
    node: nodemod.ACE_NODE
    readout: eqx.nn.Linear
    attn_param: jnp.ndarray 
    dropout: eqx.nn.Dropout  

    def __init__(self, hidden_dim, key, vector_field_depth=VECTOR_FIELD_DEPTH, vector_field_width=VECTOR_FIELD_WIDTH):
        k1, k2, k3 = jrandom.split(key, 3)
        self.node = nodemod.ACE_NODE(hidden_dim=hidden_dim, key=k1,
                                       input_dim= INPUT_DIM,
                                       vector_field_depth=vector_field_depth,
                                       vector_field_width=vector_field_width)
        
        # Bias initialisieren, um mit niedriger Sepsis-Wahrscheinlichkeit zu starten
        temp_readout = eqx.nn.Linear(hidden_dim, 1, key=k2)
        initial_bias = jnp.array([-2.94]) # Startet mit niedriger Wahrscheinlichkeit für Sepsis
        self.readout = eqx.tree_at(lambda l: l.bias, temp_readout, initial_bias)
        
        # Better Initialization 
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
    indices = np.random.choice(len(file_paths), size=min(len(file_paths), sample_size), replace=False)
    
    for i in indices:
        try:
            d = np.load(file_paths[i], allow_pickle=True)
            if "data_raw" in d: arr = d["data_raw"]
            elif "data_median" in d: arr = d["data_median"]
            else: arr = d[d.files[0]]
            
            # Include all files regardless of feature count for consistent statistics
            # Pad or trim to match input_dim
            if arr.shape[1] < input_dim:
                pad = np.zeros((arr.shape[0], input_dim - arr.shape[1]), dtype=np.float32)
                arr = np.concatenate([arr, pad], axis=1)
            else:
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


        mask = (~np.isnan(arr)).astype(np.float32) # creates mask where 0.0 indicates a missing value   

        seq_len = arr.shape[0]
        feat_dim = arr.shape[1]


        #in case there is a time_series with inconsistent feature_dim
        if feat_dim < self.input_dim:
            pad = np.zeros((seq_len, self.input_dim - feat_dim), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=1) 
        elif feat_dim > self.input_dim:             
            arr = arr[:, :self.input_dim]
   

        # Normalize BEFORE padding
        arr = (arr - self.mean) / self.std
        # Handle potential NaN from normalization
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        # add mask change***
        arr = arr[:, :feat_dim] #get rid of excess (Because increasiung input dimension changes global norm statistic shape)

        arr_masked = np.concatenate([arr, mask], axis = -1) #concatenate mask feature wise

        x_out = np.zeros((self.max_len, self.input_dim), dtype=np.float32)
        real_len = min(seq_len, self.max_len)

        if seq_len > self.max_len:  #truncate the beginning
            x_out[:real_len] = arr_masked[-self.max_len:]
        else:                       #pad at the end
            x_out[:real_len] = arr_masked[:real_len]
        
        l_val = self.labels[idx]

        #turning label from int to float?
        label = 1.0 if float(l_val) >= 0.5 else 0.0

        last_idx = real_len - 1 if real_len > 0 else 0

        return x_out, np.array([label], dtype=np.float32), last_idx



def make_collate_fn(max_len=MAX_SEQ_LEN, input_dim=INPUT_DIM):
    """Return a collate_fn that enforces a constant shape (batch, max_len, input_dim).

    This prevents variable-length batches which cause repeated JAX recompilations.
    """
    def collate_fn(batch):
        # batch: list of tuples (x_out, label, last_idx)
        xs = [b[0] for b in batch]
        ys = [b[1] for b in batch]
        last_idxs = [b[2] for b in batch]

        # Ensure numpy arrays and fixed shapes
        bx = np.stack([np.asarray(x, dtype=np.float32)[:max_len, :input_dim] if x.shape[0] >= max_len else np.pad(x, ((0, max_len - x.shape[0]), (0,0)), mode='constant') for x in xs], axis=0)
        by = np.stack([np.asarray(y, dtype=np.float32) for y in ys], axis=0)
        blast = np.asarray(last_idxs, dtype=np.int32)

        # Convert to torch tensors for DataLoader compatibility
        import torch
        return torch.from_numpy(bx), torch.from_numpy(by), torch.from_numpy(blast)

    return collate_fn
    

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
    
    return (paths[:n_train], labels[:n_train],
            paths[n_train:], labels[n_train:])

# ---------------------------------------------------------
# 3. Training & Eval Logic (mit pmap für Multi-GPU)
# ---------------------------------------------------------
def make_fns(model_static, optimizer):
    """
    Erstellt die Training- und Evaluierungsfunktionen mit pmap Parallelisierung.
    """
    
    def compute_logits(params, model_static, x, last_idxs, key=None, inference=False):
        """
        Berechnet die finalen Logits für die Sequenzen.
        
        Args:
            params: Trainierbare Parameter
            model_static: Static parts des Models
            x: Input-Sequenzen (batch_size, max_seq_len, input_dim)
            last_idxs: Index des letzten gültigen Timesteps pro Sample (batch_size,)
            key: PRNG Key für Dropout (wird ignoriert, wenn inference=True)
            inference: Bool, ob im Inference-Modus
        
        Returns:
            final_logits: Shape (batch_size, 1)
        """
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
        #model outputs one logit per timestep. so we take it from the last valid timestep of the sample
        final_logits = logits_seq_pred[batch_indices, last_idxs, 0] #[sample, last valid timestep, logit]
        
        # --- FIX 1: LOGIT CLAMPING ---
        # Zwingt Werte in den Bereich [-10, 10]. 
        # Sigmoid(-10) ist ~0.000045, Sigmoid(10) ist ~0.99995. Das reicht völlig.
        # Verhindert Loss-Explosionen.
        final_logits = jnp.clip(final_logits, -20.0, 10.0)
        
        return final_logits.reshape(-1, 1)

    # In make_fns:

    # --- Focal Loss Implementation ---
    def sigmoid_focal_loss(logits, labels, alpha=None, gamma=gamma, sepsis_amp=sepsis_Amp):
        """
        Sigmoid focal loss with optional automatic positive-class weighting.

        If `alpha` is provided it is used as the positive-class weight. Otherwise
        `alpha` is derived from `sepsis_amp` by
            alpha = sepsis_amp / (sepsis_amp + 1.0)
        mapping larger `sepsis_amp` to alpha closer to 1.0 (more weight for the
        positive/sepsis class), which helps reduce false negatives.

        Args:
            logits: raw model logits
            labels: binary labels (0/1)
            alpha: optional manual positive-class weight in (0,1)
            gamma: focal loss gamma
            sepsis_amp: multiplier used to derive alpha when alpha is None
        """
        p = jax.nn.sigmoid(logits)
        # Binary cross-entropy on logits
        ce_loss = optax.sigmoid_binary_cross_entropy(logits, labels)
        p_t = (labels * p) + ((1 - labels) * (1 - p))

        # Derive alpha from sepsis_amp if user didn't pass alpha explicitly
        if alpha is None:
            # Map positive multiplier to (0,1) range
            alpha = sepsis_amp / (sepsis_amp + 1.0)
            alpha = jnp.clip(alpha, 1e-6, 1.0 - 1e-6)

        alpha_t = (labels * alpha) + ((1 - labels) * (1 - alpha))
        focal_factor = (1 - p_t) ** gamma
        return alpha_t * focal_factor * ce_loss

    @eqx.filter_value_and_grad
    def loss_fn(params, model_static, x, y, last_idxs, keys):
        """
        Berechnet den Focal Loss für die aktuelle Batch.
        
        Args:
            params: Trainierbare Parameter
            model_static: Static parts des Models
            x: Input-Sequenzen (local_batch_size, max_seq_len, input_dim)
            y: Labels (local_batch_size, 1)
            last_idxs: Indices der letzten gültigen Timesteps (local_batch_size,)
            keys: PRNG Keys für Dropout (local_batch_size, 2) - ein Key pro Sample
        
        Returns:
            loss: Scalar Focal Loss
        """
        # keys hat Shape (local_batch_size, 2) pro Device da pmap mit in_axes=0 die erste Dim mapped
        # Nutze diese keys für jeden Sample in der Batch
        preds = compute_logits(params, model_static, x, last_idxs, key=keys, inference=False)
        # Nutze Focal Loss statt gewichtetem BCE.
        # Alpha wird standardmäßig aus `sepsis_Amp` abgeleitet (alpha = sepsis_Amp/(sepsis_Amp+1)),
        # sodass `sepsis_Amp` > 1 positives (Sepsis=1) stärker gewichtet und so False Negatives reduziert.
        loss = sigmoid_focal_loss(preds, y, alpha=alpha, gamma=gamma)
        return jnp.mean(loss)

    # --- Train Step ---
    @eqx.filter_jit
    def train_step(params, x, y, opt_state, last_idxs, keys): # Keys (pro Device unterschiedlich)
        loss, grads = loss_fn(params, model_static, x, y, last_idxs, keys)
        
        grads = jax.tree.map(lambda g: lax.pmean(g, axis_name="i"), grads)
        loss_mean = lax.pmean(loss, axis_name="i")
        
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_mean

    # --- Eval Step ---
    @eqx.filter_jit
    def eval_step(params, x, last_idxs):
        # Inference=True -> Dropout ist aus
        logits = compute_logits(params, model_static, x, last_idxs, key=None, inference=True)
        probs = jax.nn.sigmoid(logits)
        return probs

    # PMAP mit expliziten in_axes: Verhindert Shape-Mismatches die zu Deadlocks führen
    # in_axes=(0,0,0,0,0,0) bedeutet: Alle 6 Argumente sind über Device-Axis repliziert
    train_step_pmap = jax.pmap(train_step, axis_name="i", 
                                in_axes=(0, 0, 0, 0, 0, 0),
                                donate_argnums=(0, 3))
    eval_step_pmap = jax.pmap(eval_step, axis_name="i", in_axes=(0, 0, 0))

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

    print(f"TN={tn}  | FP={fp} \nFN={fn}   | TP={tp} | Acc={acc:.2f} F1={f1:.2f} Recall={rec:.2f}")
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "acc": acc, "recall": rec, "f1": f1}

# ---------------------------------------------------------
# 5. Model IO Helpers
# ---------------------------------------------------------

def save_model(model, path):
    """Serialize an Equinox model to `path` (creates parent dirs)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(path, model)


def load_model(model, path):
    """Deserialize leaves into `model` from `path`."""
    return eqx.tree_deserialise_leaves(path, model)

# ---------------------------------------------------------
# 6. Main
# ---------------------------------------------------------
def run_evaluation(dataloader, n_devices, eval_step_pmap, params_repl, desc="Eval", write_preds_path=None, num_bins=1024, threshold=0.5):
    print(f"--- {desc} ---")

    # Streaming accumulators (integers only)
    total_tp = total_tn = total_fp = total_fn = 0
    total_pos = total_neg = 0

    # Histogram bins for approximate AUC
    pos_hist = np.zeros(num_bins, dtype=np.int64)
    neg_hist = np.zeros(num_bins, dtype=np.int64)

    # Prepare output file if requested
    preds_file = None
    if write_preds_path is not None:
        # Start fresh per call
        try:
            preds_file = open(write_preds_path, "w")
            preds_file.write("prob,label\n")
            preds_file.flush()
            os.fsync(preds_file.fileno())
        except Exception as e:
            print(f"Warnung: Konnte Prädiktionsdatei nicht öffnen: {e}")
            preds_file = None

    for batch in dataloader:
        bx, by, blast = batch
        orig_bs = bx.shape[0]
        bx_np = bx.numpy()
        blast_np = blast.numpy().astype(np.int32)

        # Handle remainder samples for multi-device reshape
        local_bs = bx_np.shape[0] // n_devices
        remainder = bx_np.shape[0] % n_devices

        if remainder > 0:
            pad_size = n_devices - remainder
            bx_np = np.pad(bx_np, ((0, pad_size), (0, 0), (0, 0)), mode='constant')
            blast_np = np.pad(blast_np, ((0, pad_size), ), mode='constant')
            local_bs = bx_np.shape[0] // n_devices

        bx_shard = bx_np.reshape(n_devices, local_bs, MAX_SEQ_LEN, INPUT_DIM)
        blast_shard = blast_np.reshape(n_devices, local_bs)

        # Eval (ensure completion before device_get)
        probs_dev = eval_step_pmap(params_repl, bx_shard, blast_shard)
        try:
            probs_dev.block_until_ready()
        except Exception:
            pass
        probs_np = jax.device_get(probs_dev).reshape(-1)[:orig_bs]

        labels_np = by.numpy().flatten()[:orig_bs]

        # Write to CSV if requested (append per batch to limit memory)
        if preds_file is not None:
            for p, l in zip(probs_np, labels_np):
                try:
                    preds_file.write(f"{float(p):.6f},{int(l)}\n")
                except Exception:
                    pass
            try:
                preds_file.flush()
                os.fsync(preds_file.fileno())
            except Exception:
                pass

        # Streaming confusion metrics (thresholded)
        preds_bin = (probs_np >= threshold).astype(np.int32)
        tp = int(np.sum((preds_bin == 1) & (labels_np == 1)))
        tn = int(np.sum((preds_bin == 0) & (labels_np == 0)))
        fp = int(np.sum((preds_bin == 1) & (labels_np == 0)))
        fn = int(np.sum((preds_bin == 0) & (labels_np == 1)))

        total_tp += tp; total_tn += tn; total_fp += fp; total_fn += fn

        # Update histogram for approximate AUC
        bin_idx = np.minimum((probs_np * num_bins).astype(np.int32), num_bins - 1)
        if np.any(labels_np == 1):
            pos_bins = np.bincount(bin_idx[labels_np == 1], minlength=num_bins)
            pos_hist += pos_bins
            total_pos += int(labels_np.sum())
        if np.any(labels_np == 0):
            neg_bins = np.bincount(bin_idx[labels_np == 0], minlength=num_bins)
            neg_hist += neg_bins
            total_neg += int((labels_np == 0).sum())

    # Close file if opened
    if preds_file is not None:
        try:
            preds_file.close()
        except Exception:
            pass

    # Build metrics
    denom = total_tp + total_tn + total_fp + total_fn
    acc = (total_tp + total_tn) / (denom + 1e-8)
    prec = total_tp / (total_tp + total_fp + 1e-8)
    rec = total_tp / (total_tp + total_fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)

    metrics_dict = {
        "tp": int(total_tp), "tn": int(total_tn), "fp": int(total_fp), "fn": int(total_fn),
        "acc": float(acc), "recall": float(rec), "f1": float(f1)
    }

    # Approximate AUC from histogram
    auc_score = float('nan')
    if total_pos > 0 and total_neg > 0:
        cum_pos_rev = np.cumsum(pos_hist[::-1])
        cum_neg_rev = np.cumsum(neg_hist[::-1])
        tpr = cum_pos_rev / float(total_pos)
        fpr = cum_neg_rev / float(total_neg)
        try:
            auc_score = float(np.trapzoid(tpr, fpr))
        except Exception:
            auc_score = float('nan')

    metrics_dict["val_auc"] = auc_score

    # Print a short summary
    print(f"TN={metrics_dict['tn']}  | FP={metrics_dict['fp']} \nFN={metrics_dict['fn']}   | TP={metrics_dict['tp']} | Acc={metrics_dict['acc']:.2f} F1={metrics_dict['f1']:.2f} Recall={metrics_dict['recall']:.2f}")
    print(f"Approx AUC (hist): {auc_score}")

    return metrics_dict

def main():
    # 1. Device Setup
    n_devices = jax.local_device_count()
    devices = jax.local_devices()
    print(f"Verwende {n_devices} Devices: {devices}")
    if n_devices > 1:
        print("Multi-Device Modus aktiviert, die GPUs müssen den gleichen PCIe haben! (gesschwindigkeit)")
    
    # 2. Konstanten & Batch-Berechnung
    # Nutze die Variablen aus deinem Header, stelle aber Konsistenz sicher
    current_lr = Learning_Rate 
    current_epochs = Epochen 
    global_batch_size = BATCH_SIZE_PER_GPU * n_devices
    
    print(f"Setup: {current_epochs} Epochen, Global Batch Size: {global_batch_size}")

    key = jrandom.PRNGKey(seed)
    model_key, _ = jrandom.split(key)

    # 3. Daten laden & Normalisierung
    files_tr, labs_tr, files_test, labs_test = list_npz_files_split()

    # Statistiken nur auf Basis der Trainingsdaten berechnen
    global_mean, global_std = compute_global_stats(files_tr, INPUT_DIM)

    # Datasets mit Normalisierung
    train_ds = NPZDataset(files_tr, labs_tr, mean=global_mean, std=global_std)
    test_ds = NPZDataset(files_test, labs_test, mean=global_mean, std=global_std)

    # Dataloader Initialisierung
    collate_fn = make_collate_fn(max_len=MAX_SEQ_LEN, input_dim=INPUT_DIM)
    train_loader = DataLoader(train_ds, batch_size=global_batch_size, shuffle=True, 
                              num_workers=num_workers, drop_last=True, collate_fn=collate_fn)
    
    test_loader = DataLoader(test_ds, batch_size=global_batch_size, shuffle=False, 
                            num_workers=num_workers, drop_last=True, collate_fn=collate_fn)

    # 4. Modell & Optimizer
    model = SepsisClassifier(hidden_dim=HIDDEN_DIM, key=model_key, 
                             vector_field_depth=VECTOR_FIELD_DEPTH,
                             vector_field_width=VECTOR_FIELD_WIDTH)
    params, model_static = eqx.partition(model, eqx.is_inexact_array)

    # 5. Learning Rate Scheduler
    steps_per_epoch = len(files_tr) // global_batch_size
    if steps_per_epoch == 0: 
        steps_per_epoch = 1
    total_steps = Epochen * steps_per_epoch
    warmup_steps = Warmup_epochs * steps_per_epoch  # 5 Epochen Warmup bis Peak-LR

    # Learning Rate Scheduler - decay über den gesamten Trainingsprozess
    # (warmup ist separate Phase, decay_steps bezieht sich auf komplette Trainingszeit)
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-7,          # Start bei sehr niedriger LR
        peak_value=Learning_Rate, # Peak LR nach Warmup
        warmup_steps=warmup_steps,
        decay_steps=total_steps,  # Decay über gesamte Trainingszeit (nicht nur nach Warmup)
        end_value=Final_LR            # Finale LR am Ende des Trainings
    )

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

    print(f"Start: LR={current_lr}, Hidden={HIDDEN_DIM}, VectorField Depth={VECTOR_FIELD_DEPTH}, Width={VECTOR_FIELD_WIDTH}")

    # Initialisierung für Training
    dropout_key = jrandom.PRNGKey(seed + 1)
    global_step = 0

  
    # 6. Training Loop
    seen_seq_lengths = set()
    prev_rss = None

    for epoch in range(current_epochs):
        start_t = time.time()
        epoch_losses = []
        # Memory snapshot at epoch start
        try:
            proc = psutil.Process(os.getpid())
            rss_mb = proc.memory_info().rss / (1024**2)
            print(f"Start Epoche {epoch+1}: RSS host {rss_mb:.1f} MB")
        except Exception:
            rss_mb = None
            pass
        
        for i, batch in enumerate(train_loader):
            bx, by, blast = batch
            bx_np = bx.numpy()
            by_np = by.numpy()
            blast_np = blast.numpy().astype(np.int32)

            # Track seen sequence lengths (should be constant MAX_SEQ_LEN)
            seq_len_seen = bx_np.shape[1]
            seen_seq_lengths.add(seq_len_seen)
            if len(seen_seq_lengths) > 1:
                print(f"WARNUNG: Gefundene verschiedene Sequenzlängen in Batches: {seen_seq_lengths}. Das kann JAX-Recompiles auslösen.")
                # Prophylaktisch Cache leeren (langsam, aber sicher)
                try:
                    jax.clear_caches()
                except Exception:
                    pass
                gc.collect()

            # BUG FIX #6: Validate reshape compatibility before attempting (prevents silent failures)
            if bx_np.shape[0] != n_devices * BATCH_SIZE_PER_GPU:
                print(f"Warnung: Übersprungene Batch mit Größe {bx_np.shape[0]} (erwartet {n_devices*BATCH_SIZE_PER_GPU}). Überspringe Batch.")
                gc.collect()
                continue
            
            bx_shard = bx_np.reshape(n_devices, BATCH_SIZE_PER_GPU, MAX_SEQ_LEN, INPUT_DIM)
            by_shard = by_np.reshape(n_devices, BATCH_SIZE_PER_GPU, 1)
            blast_shard = blast_np.reshape(n_devices, BATCH_SIZE_PER_GPU)

            # --- FIX 3: Correct PRNG Handling (Deadlock-Prevention) ---
            # Der Deadlock entsteht, wenn pmap asymmetrische Operationen sieht
            # Lösung: Sicherstellen, dass ALLE inputs auf ALLEN devices die gleiche Struktur haben
            
            # 1. Erzeuge einen Key pro Device (Shape wird (n_devices, 2))
            dropout_key, step_key = jrandom.split(dropout_key)
            step_keys_per_device = jrandom.split(step_key, n_devices)
            
            # 2. Validierung: Shape muss konsistent sein
            assert step_keys_per_device.shape[0] == n_devices, \
                f"Key count {step_keys_per_device.shape[0]} != n_devices {n_devices}"
            
            # 3. Übergabe an pmap (Keys haben Shape (n_devices, 2) -> jeder Device bekommt einen Key)
            params_repl, opt_state_repl, loss_val = train_step_pmap(
                params_repl, bx_shard, by_shard, opt_state_repl, blast_shard, step_keys_per_device
            )
            
            # Check auf NaN im Loss (Optionaler Not-Halt)
            loss_cpu = loss_val[0].item()
            if np.isnan(loss_cpu) or loss_cpu > 1e6:
                print(f"\nWARNUNG: Loss Explosion in Batch {i}! ({loss_cpu})")
                # Hier könnte man abbrechen, wir loggen erstmal nur
            
            epoch_losses.append(loss_cpu)
            global_step += 1
            
            if i % 5 == 0: # Nicht jeden Batch printen, spart CPU
                # Berechne aktuelle Learning Rate
                current_lr_val = lr_schedule(global_step)
                print(f"\rEp {epoch+1}/{current_epochs} | Batch {i} | Loss {loss_cpu:.4f} | LR {current_lr_val:.2e}", end="")

        # Epochen-Statistik
        avg_loss = float(np.mean(epoch_losses))
        history["loss"].append(avg_loss)
        train_dur = time.time() - start_t
        print(f"\nEnde Epoche {epoch+1:02d} | Avg Loss: {avg_loss:.5f} | Zeit: {train_dur:.1f}s")
        if (epoch + 1) % snapshot == 0:
            # Safely build a snapshot of the current model from the replicated params
            try:
                params_host_curr = jax.device_get(params_repl)
                params_unrep_curr = jax.tree.map(lambda x: x[0], params_host_curr)
                trained_model_curr = eqx.combine(params_unrep_curr, model_static)
                save_model(trained_model_curr, f"checkpoints/epoch_{epoch+1}_model.eqx")
                print(f"Saved checkpoint for epoch {epoch+1} -> checkpoints/epoch_{epoch+1}_model.eqx")
            except Exception as e:
                print(f"Warning: failed to save checkpoint for epoch {epoch+1}: {e}")

                
        # 6. Test Evaluation nach jedem Epoch (stream to CSV to avoid accumulation)
        res = run_evaluation(test_loader, n_devices, eval_step_pmap, params_repl, desc=f"Test Ep {epoch+1}", write_preds_path=f"preds/preds_epoch_{epoch+1}.csv")
        
        if res:
            history["val_auc"].append(float(res.get("val_auc", 0)))
            history["acc"].append(float(res.get("acc", 0)))
            history["tp"].append(int(res.get("tp", 0)))
            history["tn"].append(int(res.get("tn", 0)))
            history["fp"].append(int(res.get("fp", 0)))
            history["fn"].append(int(res.get("fn", 0)))
            history["recall"].append(float(res.get("recall", 0)))

        # Freigabe nach Evaluation: JAX-Caches leeren und Python-GC erzwingen
        try:
            jax.clear_caches()
        except Exception:
            pass
        gc.collect()
        proc = psutil.Process(os.getpid())
        rss_mb = proc.memory_info().rss / (1024**2)
        rss_gb = rss_mb / 1024.0
        try:
            total_mem_gb = psutil.virtual_memory().total / (1024**3)
        except Exception:
            total_mem_gb = None
        print(f"Nach Evaluation: RSS host {rss_mb:.1f} MB ({rss_gb:.2f} GB)")

        # 1) Growth-based trigger (z.B. > CACHE_GROWTH_FACTOR)
        triggered = False
        if prev_rss is not None and rss_mb is not None and rss_mb > prev_rss * CACHE_GROWTH_FACTOR:
            print(f"WARNUNG: RSS ist stark gewachsen ({prev_rss:.1f} -> {rss_mb:.1f} MB). Versuche Cache-Freigabe.")
            triggered = True

        # 2) Absolute GB threshold trigger
        if rss_gb is not None and rss_gb >= CACHE_CLEAR_RSS_GB:
            print(f"WARNUNG: RSS ({rss_gb:.1f} GB) über dem Grenzwert von {CACHE_CLEAR_RSS_GB} GB. Leere Caches.")
            triggered = True

        # 3) Relative Anteil am Gesamtspeicher
        if total_mem_gb is not None and rss_gb is not None and rss_gb / float(total_mem_gb) >= CACHE_CLEAR_RATIO:
            print(f"WARNUNG: RSS ist {rss_gb:.2f} GB = {rss_gb/float(total_mem_gb):.2%} des Gesamtspeichers. Leere Caches.")
            triggered = True

        # Optional: niedrigere Schwelle nur zur Warnung
        if not triggered and rss_gb is not None and rss_gb >= CACHE_CLEAR_WARN_GB:
            print(f"Hinweis: RSS {rss_gb:.1f} GB überschreitet Warn-Schwelle {CACHE_CLEAR_WARN_GB} GB (keine sofortige Löschung).")

        if triggered:
            try:
                # Versuche JAX-spezifische Caches zu leeren
                jax.clear_caches()
            except Exception:
                pass
            # Versuche zusätzliche Freigaben (Python GC und kurze Pause)
            gc.collect()
            time.sleep(0.5)
            proc = psutil.Process(os.getpid())
            rss_mb_after = proc.memory_info().rss / (1024**2)
            rss_gb_after = rss_mb_after / 1024.0
            print(f"Nach Cache-Freigabe: RSS host {rss_mb_after:.1f} MB ({rss_gb_after:.2f} GB)")

            # Wenn weiterhin sehr hoch, mache einen zweiten Versuch
            if rss_gb_after >= CACHE_CLEAR_RSS_GB or (total_mem_gb is not None and rss_gb_after / float(total_mem_gb) >= CACHE_CLEAR_RATIO):
                print("WARNUNG: RSS bleibt hoch nach erstem Clear, versuche zusätzlichen Clear.")
                try:
                    jax.clear_caches()
                except Exception:
                    pass
                gc.collect()
                time.sleep(0.5)
                proc = psutil.Process(os.getpid())
                rss_mb_after2 = proc.memory_info().rss / (1024**2)
                print(f"Nach 2. Freigabe: RSS host {rss_mb_after2:.1f} MB")

        prev_rss = rss_mb

        try:
            gm = subprocess.check_output(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"])
            print("GPU mem:", gm.decode().strip())
        except Exception:
            pass

    # Speichere auch die Config mit Seed # 7. Finaler Test & Speichern
    print("\n--- FINAL SUMMARY ---")
    run_evaluation(test_loader, n_devices, eval_step_pmap, params_repl, desc="FINAL TEST SET", write_preds_path="preds/preds_final.csv")

    # Letzte saubere Bereinigung vor dem Speichern
    try:
        jax.clear_caches()
    except Exception:
        pass
    gc.collect()
    proc = psutil.Process(os.getpid())
    rss_mb = proc.memory_info().rss / (1024**2)
    print(f"Nach FINAL Evaluation: RSS host {rss_mb:.1f} MB")

    # Optional: Wenn RSS ungewöhnlich hoch ist, schreibe Warnung in Log
    if rss_mb > 100 * 1024:  # 100 GB in MB ~ 102400
        print("WARNUNG: Sehr hoher RSS nach finaler Evaluation - prüfe variable Shapes oder JAX-Cache")
    try:
        gm = subprocess.check_output(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"])
        print("GPU mem:", gm.decode().strip())
    except Exception:
        pass

    print("\nSpeichere Ergebnisse...")
    # Parameter vom pmap zurückholen (Device 0 ist primary device)
    params_host = jax.device_get(params_repl)
    # Extrahiere das erste Replica (funktioniert für n_devices==1 und >1)
    params_unrep = jax.tree.map(lambda x: x[0], params_host)
    # Kombiniere params und static um das vollständige Modell zu rekonstruieren
    trained_model = eqx.combine(params_unrep, model_static)
    # Nutze die helper-Funktion zum Speichern des finalen Modells
    save_model(trained_model, "checkpoints/model_final.eqx")

    # Speichere Trainingshistorie
    np.savez("training_history/training_history.npz", **{k: np.array(v) for k, v in history.items()})


    try:
        with open(CONFIG_PATH, 'a') as f:
            f.write(f"Hidden_Dim,{HIDDEN_DIM}\n")
            f.write(f"Learning_Rate,{Learning_Rate}\n")
            f.write(f"Epochs,{Epochen}\n")
            f.write(f"VectorField_Depth,{VECTOR_FIELD_DEPTH}\n")
            f.write(f"VectorField_Width,{VECTOR_FIELD_WIDTH}\n")
    except:
        pass
    
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