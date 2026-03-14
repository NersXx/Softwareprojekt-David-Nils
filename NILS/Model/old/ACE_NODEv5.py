import sys
import time
import equinox as eqx
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import matplotlib
matplotlib.use("Agg")   # wichtig: nicht-interaktives Backend für Plot
import matplotlib.pyplot as plt

import ACE_NODEv4 as ace_node
import norm
from jax import tree_util

# Anzahl Devices (GPUs) automatisch ermitteln
n_devices = jax.local_device_count()
if n_devices < 2:
    print(f"Warnung: nur {n_devices} Device(s) verfügbar. Dieses Beispiel ist für 2 GPUs gedacht.")

def shard_array(x, n_shards):
    N = x.shape[0]
    per = N // n_shards
    if per == 0:
        raise ValueError("Zu wenige Datenpunkte für die Anzahl Devices.")
    N_trunc = per * n_shards
    x = x[:N_trunc]
    new_shape = (n_shards, per) + x.shape[1:]
    return x.reshape(new_shape)

def main() -> int:
    key = random.key(int(time.time()))
    model_key, train_key = random.split(key)
    
    # Lynx and Hare Dataset
    lh_data = jnp.array(np.load("LH_data.npy"))
    time_steps = lh_data[:, 0:1]
    populations = lh_data[:, 1:3]
    print(f"Years: {time_steps.shape}, Population {populations.shape}")
    
    # scaling
    time_steps_norm = time_steps - time_steps.min()
    eps = 1e-8
    pop_log = jnp.log(populations + eps)
    mean = pop_log.mean(axis=0, keepdims=True)
    std = pop_log.std(axis=0, keepdims=True)
    populations_norm = (pop_log - mean) / std
    
    # creating model
    model = ace_node.ACE_NODE(2, 32, 3, key=model_key)

    # initial attention
    initial_attention = generate_initial_attention(populations_norm).reshape(-1)

    # Shards
    time_shards = shard_array(time_steps_norm.squeeze(), n_devices)
    pop_shards  = shard_array(populations_norm, n_devices)
    attn_repl   = jnp.broadcast_to(initial_attention, (n_devices,) + initial_attention.shape)

    # Parameter extrahieren und replizieren
    params = ace_node.get_params(model)
    devices = jax.local_devices()[:n_devices]
    params_repl = jax.device_put_replicated(params, devices)

    # statischer Teil einmalig extrahieren
    _, model_static = eqx.partition(model, eqx.is_inexact_array)

    # RNG pro Device
    keys = random.split(train_key, n_devices)

    @jax.pmap
    def train_on_device(params_local, t_local, p_local, attn_local, key_local):
        model_local = eqx.combine(params_local, model_static)
        trained_model = ace_node.training_loop(
            t_local, p_local, attn_local, model_local, 50, 1e-3, key=key_local, plot_loss=False
        )
        trained_params = ace_node.get_params(trained_model)
        # Loss pro Device berechnen
        y0_local = p_local[0]
        y_pred_local = trained_model(t_local, y0_local, attn_local)
        loss_local = jnp.mean((p_local - y_pred_local)**2)
        return trained_params, loss_local

    trained_params_repl, losses_repl = train_on_device(params_repl, time_shards, pop_shards, attn_repl, keys)

    # Synchronisieren und auf Host holen
    trained_params_repl = jax.tree_map(lambda x: x.block_until_ready(), trained_params_repl)
    losses_host = jax.device_get(losses_repl)
    trained_params_host = jax.device_get(trained_params_repl)

    # Parameter mitteln und ins Modell setzen
    averaged_params = jax.tree_map(lambda x: jnp.mean(x, axis=0), trained_params_host)
    model = eqx.combine(averaged_params, model_static)

    # Loss anzeigen
    print("Mean loss across devices:", float(jnp.mean(losses_host)))

    # -------------------------
    # Vorhersage / Plot
    # -------------------------
    plt.plot(time_steps, populations[:, 0:1], c="dodgerblue", label="Hares")
    plt.plot(time_steps, populations[:, 1:2], c="green", label="Lynx")
            
    y0 = jnp.array(populations_norm[0])
    t_pred = jnp.concatenate(
        [time_steps_norm.squeeze(), jnp.array([time_steps_norm.max() + i for i in range(20)])],
        axis=0
    )
    hl_predict = jnp.exp(((model(t_pred, y0, initial_attention) * std) + mean) - eps)
    
    hare_predict = hl_predict[:, 0:1]
    lynx_predict = hl_predict[:, 1:2]
    
    plt.plot(t_pred + time_steps.min(), hare_predict, c="red", label="Hares fit")
    plt.plot(t_pred + time_steps.min(), lynx_predict, c="purple", label="Lynx fit")
    plt.legend()
    plt.savefig("plot.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    return 0


def generate_initial_attention(data):
    correlation_matrix = jnp.corrcoef(data.T)
    return correlation_matrix


if __name__ == "__main__":
    sys.exit(main())
