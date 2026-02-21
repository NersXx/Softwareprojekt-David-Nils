import numpy as np
import matplotlib.pyplot as plt
import os

def plot_training_results(history_path="training_history.npz"):
    if not os.path.exists(history_path):
        print(f"Fehler: {history_path} wurde nicht gefunden.")
        return

    data = np.load(history_path, allow_pickle=True)
    
    def extract_values(key):
        raw_data = data[key]
        # Falls die Daten als Liste von Dictionaries gespeichert wurden:
        if raw_data.dtype == object and len(raw_data) > 0 and isinstance(raw_data[0], dict):
            return np.array([d[key] if key in d else 0.0 for d in raw_data])
        # Falls es bereits saubere Zahlen sind:
        return np.array(raw_data)

    try:
        val_auc = extract_values('val_auc')
        loss = extract_values('loss')
    except Exception as e:
        print(f"Fehler beim Verarbeiten der Daten: {e}")
        return

    # Plotting (Backend auf 'Agg' setzen, um X11-Fehler zu vermeiden)
    plt.switch_backend('Agg') 
    fig, ax1 = plt.subplots(figsize=(10, 6))
    epochs = np.arange(1, len(val_auc) + 1)
    
    ax1.set_xlabel('Epoche')
    ax1.set_ylabel('Validation AUC', color='tab:blue')
    ax1.plot(epochs, val_auc, marker='o', color='tab:blue', label='Val AUC', linewidth=2)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Training Loss', color='tab:red')
    ax2.plot(epochs, loss, marker='x', color='tab:red', label='Train Loss', alpha=0.6)
    
    plt.title('ACE-NODE Training: AUC & Loss Verlauf')
    plt.savefig('auc_training_plot.png')
    print("Plot erfolgreich als 'auc_training_plot.png' gespeichert.")

if __name__ == "__main__":
    plot_training_results()