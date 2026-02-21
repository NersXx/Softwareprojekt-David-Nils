import sys
import time
import equinox as eqx
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optax

import ACE_NODEv4 as ace_node
import ASS4.old.norm as norm

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
    
    # Dataset
    lh_data = jnp.array(np.load("LH_data.npy"))
    time_steps = lh_data[:, 0:1]
    populations = lh_data[:, 1:3]
    
    # scaling
    time_steps_norm = time_steps - time_steps.min()
    eps = 1e-8
    pop_log = jnp.log(populations + eps)
    mean = pop_log.mean(axis=0, keepdims=True)
    std = pop_log.std(axis=0, keepdims=True)
    populations_norm = (pop_log - mean) / std
    
    # model
    model = ace_node.ACE_NODE(2, 32, 3, key=model_key)
    initial_attention = generate_initial_attention(populations_norm).reshape(-1)

    # Shards
    time_shards = shard_array(time_steps_norm.squeeze(), n_devices)
    pop_shards  = shard_array(populations_norm, n_devices)
    attn_repl   = jnp.broadcast_to(initial_attention, (n_devices,) + initial_attention.shape)

    # Parameter extrahieren und replizieren
    params = ace_node.get_params(model)
    devices = jax.local_devices()[:n_devices]
    params_repl = jax.device_put_replicated(params, devices)

    # statischer Teil
    _, model_static = eqx.partition(model, eqx.is_inexact_array)

    # Optimizer
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)
    opt_state_repl = jax.device_put_replicated(opt_state, devices)

    # ein einzelner Trainingsschritt
    @eqx.filter_value_and_grad
    def loss_fn(model_train, model_static, x, y, attn):
        m = eqx.combine(model_train, model_static)
        y_pred = m(x, y[0], attn)
        return jnp.mean((y - y_pred)**2)

    def train_step(model, x, y, attn, opt_state):
        model_train, model_static = eqx.partition(model, eqx.is_inexact_array)
        loss, grads = loss_fn(model_train, model_static, x, y, attn)
        updates, opt_state = optimizer.update(grads, opt_state)
        model_train = eqx.apply_updates(model_train, updates)
        model = eqx.combine(model_train, model_static)
        return model, opt_state, loss


    @jax.pmap
    def train_step_pmap(params_local, x_local, y_local, attn_local, opt_state_local):
        model_local = eqx.combine(params_local, model_static)
        model_local, opt_state_local, loss = train_step(model_local, x_local, y_local, attn_local, opt_state_local)
        params_local = ace_node.get_params(model_local)
        return params_local, opt_state_local, loss

    # Epochenschleife auf Host
    for epoch in range(500):
        params_repl, opt_state_repl, losses = train_step_pmap(params_repl, time_shards, pop_shards, attn_repl, opt_state_repl)
        # synchronisieren
        params_host = jax.device_get(params_repl)
        averaged_params = jax.tree.map(lambda x: jnp.mean(x, axis=0), params_host)
        params_repl = jax.device_put_replicated(averaged_params, devices)
        mean_loss = float(jnp.mean(jax.device_get(losses)))
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, loss {mean_loss}")

    # finales Modell zusammensetzen
    model = eqx.combine(averaged_params, model_static)

    # Vorhersage / Plot
    plt.plot(time_steps, populations[:, 0:1], c="dodgerblue", label="Hares")
    plt.plot(time_steps, populations[:, 1:2], c="green", label="Lynx")
            
    y0 = jnp.array(populations_norm[0])
    t_pred = jnp.concatenate(
        [time_steps_norm.squeeze(), jnp.array([time_steps_norm.max() + i for i in range(20)])],
        axis=0
    )
    hl_predict = jnp.exp(((model(t_pred, y0, initial_attention) * std) + mean) - eps)
    
    plt.plot(t_pred + time_steps.min(), hl_predict[:, 0:1], c="red", label="Hares fit")
    plt.plot(t_pred + time_steps.min(), hl_predict[:, 1:2], c="purple", label="Lynx fit")
    plt.legend()
    plt.savefig("plot.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    return 0

def generate_initial_attention(data):
    return jnp.corrcoef(data.T)

if __name__ == "__main__":
    sys.exit(main())
