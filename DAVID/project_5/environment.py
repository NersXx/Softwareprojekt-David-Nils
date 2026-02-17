#Group: David, Nils

import sys
import time

import jax
from jax import random
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

import ACE_ODE_RNNv6 as node
import preprocessing as pre
import training
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import re
from pathlib import Path

import gc
import equinox as eqx


def main() -> int:
    
    run_experiment()
    
    return 0

def run_experiment():
    
    key = random.key(int(time.time()))
    model_key, train_key, test_key = random.split(key, 3)
    
    #preparing the datasets
    print("Preparing Dataset")
    data_train, data_test = pre.load_dataset() #y.shape (20000,)     X has keys i [1,...,20000] each key mapping to a filepath
    #split in train and validation set
    
    
    print("creating Model")
    #creating the model
    obs_dim = 34*2 #time is accounted for inside the model
    output_dim = 2 #so = 1 if sepsis 0 if no sepsis after 6 hours
    hidden_dim = 20
    static_feat = 10
    solver_width = 50
    solver_depth = 3
    output_nn_width = 32
    output_nn_depth = 2
    model = node.ACE_ODE_RNN(
        input_dim=obs_dim, output_dim = output_dim, hidden_dim = hidden_dim, static_feat= static_feat, 
        solver_width= solver_width, output_nn_width=output_nn_width, 
        solver_depth=solver_depth, output_nn_depth=output_nn_depth, 
        key = model_key)
    
    #normalizing Data
    
    print("ctreating normalization statistics")
    pre.create_norm_data(data_train)
    
    #Training Model
    model = training.training_loop(data_train = data_train,
                               model = model, epochs = 50, lr = 5e-4,
                               batch_size = 256, key = train_key)
    
    
    #-----------
    #VALIDATION
    #-----------
    
    val_key = random.key(int(time.time()))
    k1, k2 = random.split(key)

    sepsis_data = training.SepsisDataset(data_test)
    
    #Use best version To validate data prediction
    print("Testing best version of the model...")
    auc, y_true, y_score = compute_auc_batched(model, sepsis_data, batch_size=256, key = val_key)
    print("Validation AUC:", auc)

    plot_validation_results(y_true, y_score)
    #DONEZO
    print("EXPERIMENT FINALIZED SUCCESFULLY\n")
    
    
    """
   print("Evaluating checkpoints...")
    #calculate accuracy of different versions of the model
    epochs, aucs, optimal_model = evaluate_checkpoints(
        model_template= model,
        checkpoint_dir= 'checkpoints',
        X = X_test, y = y_test_encoded,
        key = k1
    )
    plot_checkpoints(epochs, aucs) #you also have to make this function
    """
    

def get_checkpoints_sorted(checkpoint_dir):
    paths = list(Path(checkpoint_dir).glob("*.eqx"))

    def extract_epoch(p):
        m = re.search(r"epoch_(\d+)", p.name)
        return int(m.group(1)) if m else -1

    paths = sorted(paths, key=extract_epoch)
    epochs = [extract_epoch(p) for p in paths]
    return paths, epochs

def evaluate_checkpoints(model_template, checkpoint_dir, X, *, key):
    checkpoints, epochs = get_checkpoints_sorted(checkpoint_dir)

    aucs = []
    best_model = model_template
    model = model_template
    max_auc = -1
    
    for ckpt in checkpoints:
        model = eqx.tree_deserialise_leaves(ckpt, model)
        auc, _, _ = compute_auc_batched(model, X, batch_size= 256, key = key)
        aucs.append(auc)

        if auc > max_auc:
            best_model = model
        print("checkpoint Tested\n")

    return epochs, aucs, best_model

def plot_checkpoints(epochs, aucs):
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, aucs, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Validation AUC")
    plt.title("Validation AUC over Training")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def compute_auc_batched(model, data, *, batch_size, key):
    
    all_scores = []
    all_labels = []
    
    total_b = data.X_data.shape[0]//batch_size
    b = 0
    for X_batch, y_batch, ts_batch, Sd_batch in training.data_loader(
        data, batch_size, key=key):

        logits = model(ts_batch, X_batch, Sd_batch)
        probs = jax.nn.softmax(logits, axis=-1)

        # Positive class
        scores = np.array(probs[:, 1])
        labels = np.array(y_batch[:, 1], dtype = np.int32)
        labels = labels.reshape(-1)

        bs = probs.shape[0]
        all_scores.append(scores)
        all_labels.append(labels)
    
        b += 1
        print(f"Computing Prediction: {int((b/total_b)*100)}% ", end = '\r')
        #<-------------- I don't get why it keeps gobling up memory like crazy, The idea of the Data loader is that after the iteration of the loop is done
        #<-------------- The Memory is supposed to become available again, so that it gets reused,
        #<-------------- Maybe the Problem is with how the model computes the prediction? Maybe it doesn't reuse the mempory?    

    all_scores = np.concatenate(all_scores, axis = 0)
    all_labels = np.concatenate(all_labels, axis = 0) 

    auc = roc_auc_score(all_labels, all_scores)
    return auc, all_labels, all_scores



def plot_validation_results(y_true, y_score):
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # ---- ROC curve ----
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    axs[0].plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    axs[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axs[0].set_title("ROC Curve")
    axs[0].set_xlabel("False Positive Rate")
    axs[0].set_ylabel("True Positive Rate")
    axs[0].legend()

    # ---- Score distribution ----
    axs[1].hist(y_score[y_true == 0], bins=30, alpha=0.6, label="Negative")
    axs[1].hist(y_score[y_true == 1], bins=30, alpha=0.6, label="Positive")
    axs[1].set_title("Predicted Probability Distribution")
    axs[1].set_xlabel("P(y=1)")
    axs[1].legend()

    # ---- Confusion matrix (enhanced) ----
    y_pred = (y_score >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm / cm.sum() * 100

    im = axs[2].imshow(cm, cmap="Blues")
    axs[2].set_title("Confusion Matrix (thr = 0.5)")
    axs[2].set_xlabel("Predicted")
    axs[2].set_ylabel("Actual")
    axs[2].set_xticks([0, 1])
    axs[2].set_yticks([0, 1])
    axs[2].set_xticklabels(["0", "1"])
    axs[2].set_yticklabels(["0", "1"])

    # Annotate with counts + percentages
    for i in range(2):
        for j in range(2):
            axs[2].text(
                j, i,
                f"{cm[i, j]}\n({cm_percent[i, j]:.1f}%)",
                ha="center", va="center",
                color="black", fontsize=11
            )

    plt.tight_layout()
    plt.show()

    

if __name__ == "__main__":
    sys.exit(main())