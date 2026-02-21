# pmap_multigpu_final_fixed_inaxes.py
# Multi-GPU Training mit pmap — nur numerische Parameter-Leaves werden an Devices geschickt.
# CHANGED: separate in_axes für f und g (param_in_axes_f / param_in_axes_g)

import sys
import functools
from collections import Counter

import jax
import jax.numpy as jnp
from jax import random
import jax.tree_util as jtu
import equinox as eqx
import diffrax
import optax

import matplotlib.pyplot as plt
import numpy as np

if jax.device_count() == 1:
    print("Warnung: nur ein Device gefunden. Multi-GPU pmap wird nicht genutzt.", flush=True)

# ----------------
# Model Definition
# ----------------

class OrdinaryDE(eqx.Module):
    output_scale: jax.Array
    mlp: eqx.nn.MLP

    def __init__(self, input_dim, output_dim, layer_width, nn_depth, *, key):
        self.output_scale = jnp.array(1.0)
        self.mlp = eqx.nn.MLP(
            in_size=input_dim + 1,
            out_size=output_dim,
            width_size=layer_width,
            depth=nn_depth,
            activation=jax.nn.relu,
            final_activation=jax.nn.tanh,
            key=key,
        )

    def __call__(self, t, y, args):
        input_vector = jnp.concatenate([y, jnp.array([t])], axis=0)
        return self.output_scale * self.mlp(input_vector)


class ACE_ODE(eqx.Module):
    f_ode: OrdinaryDE
    g_ode: OrdinaryDE
    hidden_dim: int

    def __init__(self, hidden_dim, f_width, g_width, f_depth, g_depth, *, key):
        f_key, g_key = random.split(key)
        self.f_ode = OrdinaryDE(
            input_dim=hidden_dim,
            output_dim=hidden_dim,
            layer_width=f_width,
            nn_depth=f_depth,
            key=f_key,
        )
        self.g_ode = OrdinaryDE(
            input_dim=hidden_dim,
            output_dim=hidden_dim**2,
            layer_width=g_width,
            nn_depth=g_depth,
            key=g_key,
        )
        self.hidden_dim = hidden_dim

    def __call__(self, t, ha, args):
        h_state, a_matrix = jnp.array(ha[0]), jnp.array(ha[1]).reshape(self.hidden_dim, self.hidden_dim)
        h_prime = h_state @ jax.nn.softmax(a_matrix, axis=-1).T
        h_dot = self.f_ode(t, h_prime, args=None)
        g_dot = self.g_ode(t, h_prime, args=None)
        return (h_dot, g_dot)


class ACE_NODE(eqx.Module):
    ace_ode: ACE_ODE

    def __init__(self, hidden_dim, layer_width, depth, *, key):
        ace_key, fe_key, cl_key = random.split(key, 3)
        self.ace_ode = ACE_ODE(
            hidden_dim=hidden_dim,
            f_width=layer_width,
            g_width=layer_width * 2,
            f_depth=depth,
            g_depth=depth + 1,
            key=ace_key,
        )

    def __call__(self, ts, h0, a0_flat):
        ha0 = (h0, a0_flat)
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.ace_ode),
            diffrax.Tsit5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=(ts[1] - ts[0]) * 0.1,
            y0=ha0,
            stepsize_controller=diffrax.PIDController(rtol=1e-5, atol=1e-7),
            saveat=diffrax.SaveAt(ts=ts),
        )
        h_traj, a_traj_flat = solution.ys
        return h_traj[-1]  # final hidden state as example


# -----------------
# Loss / Gradients (Batched)
# -----------------

TS_EXAMPLE = jnp.linspace(0.0, 1.0, 5)  # 5 timepoints example

@eqx.filter_value_and_grad
def grad_loss_h(model_train, model_static, X, y, a0_flat):
    model_single = eqx.combine(model_train, model_static)
    def forward_single(h0, a0):
        return model_single(TS_EXAMPLE, h0, a0)
    y_pred = jax.vmap(forward_single, in_axes=(0, 0))(y, a0_flat)
    loss = jnp.mean((y - y_pred) ** 2)
    return loss

@eqx.filter_value_and_grad
def grad_loss_a(model_train, model_static, X, y, a0_flat, l2_reg=1e-4):
    model_single = eqx.combine(model_train, model_static)
    def forward_single(h0, a0):
        return model_single(TS_EXAMPLE, h0, a0)
    y_pred = jax.vmap(forward_single, in_axes=(0, 0))(y, a0_flat)
    loss = jnp.mean((y - y_pred) ** 2)
    def weights_only(leaf):
        return isinstance(leaf, jax.Array) and leaf.ndim == 2
    model_weights = eqx.filter(eqx.filter(model_train, weights_only), eqx.is_inexact_array)
    l2_loss = l2_reg * sum(jnp.sum(w ** 2) for w in jtu.tree_leaves(model_weights))
    return loss + l2_loss


# -----------------------
# pmapped train step
# -----------------------
def train_step_partitioned(param_leaves_tuple, opt_state, X, y, a0_flat, params_treedef, optimizer, loss_fn, model_static, nonarrays):
    arrays_only = jtu.tree_unflatten(params_treedef, list(param_leaves_tuple))
    model_train_full = eqx.combine(arrays_only, nonarrays)
    loss, grads = loss_fn(model_train_full, model_static, X, y, a0_flat)
    grads_arrays = eqx.filter(grads, eqx.is_inexact_array)
    grads_arrays = jtu.tree_map(lambda g: jax.lax.pmean(g, "devices"), grads_arrays)  # CHANGED
    updates, opt_state = optimizer.update(grads_arrays, opt_state, arrays_only)
    arrays_only = eqx.apply_updates(arrays_only, updates)
    updated_leaves, _ = jtu.tree_flatten(arrays_only)
    loss = jax.lax.pmean(loss, "devices")
    return loss, tuple(updated_leaves), opt_state


# -----------------------
# Hilfsfunktionen
# -----------------------
def _shard_batch(batch):
    n_devices = jax.local_device_count()
    def _reshape(x):
        if not isinstance(x, (jnp.ndarray, np.ndarray)):
            return jax.device_put_replicated(x, jax.local_devices())
        x = jnp.asarray(x)
        assert x.shape[0] % n_devices == 0, "Batch size must be divisible by number of devices"
        per_device = x.shape[0] // n_devices
        new_shape = (n_devices, per_device) + x.shape[1:]
        return x.reshape(new_shape)
    return jtu.tree_map(_reshape, batch)


# -----------------------
# Training loop (pmap)
# -----------------------
def training_loop(X_train, y_train, a0_flat, model, epochs, lr, *, key, plot_loss=True):
    print("training_loop gestartet", flush=True)
    print("jax devices:", jax.device_count(), "local:", jax.local_device_count(), flush=True)

    filter_spec_g = eqx.tree_at(
        lambda m: (m.ace_ode.g_ode, m.ace_ode.f_ode),
        jtu.tree_map(lambda _: False, model),
        replace=(True, False),
    )

    filter_spec_f_other = eqx.tree_at(
        lambda m: (m.ace_ode.g_ode, m.ace_ode.f_ode),
        jtu.tree_map(lambda _: True, model),
        replace=(False, True),
    )

    optimizer_f = optax.adam(lr)
    optimizer_g = optax.adam(lr)

    opt_state_f = optimizer_f.init(eqx.filter(eqx.filter(model, filter_spec_f_other), eqx.is_inexact_array))
    opt_state_g = optimizer_g.init(eqx.filter(eqx.filter(model, filter_spec_g), eqx.is_inexact_array))

    model_train_f, model_static_f = eqx.partition(model, filter_spec_f_other)
    model_train_g, model_static_g = eqx.partition(model, filter_spec_g)

    arrays_only_f = eqx.filter(model_train_f, eqx.is_inexact_array)
    nonarrays_f = eqx.filter(model_train_f, lambda leaf: not eqx.is_inexact_array(leaf))

    arrays_only_g = eqx.filter(model_train_g, eqx.is_inexact_array)
    nonarrays_g = eqx.filter(model_train_g, lambda leaf: not eqx.is_inexact_array(leaf))

    param_leaves_f, params_treedef_f = jtu.tree_flatten(arrays_only_f)
    param_leaves_g, params_treedef_g = jtu.tree_flatten(arrays_only_g)

    types_count = Counter(type(l).__name__ for l in param_leaves_f)
    print("DEBUG f numeric leaf types:", types_count, flush=True)

    devices = jax.local_devices()

    param_leaves_f_repl = tuple(jax.device_put_replicated(l, devices) for l in param_leaves_f)
    param_leaves_g_repl = tuple(jax.device_put_replicated(l, devices) for l in param_leaves_g)

    opt_state_f_repl = jax.device_put_replicated(opt_state_f, devices)
    opt_state_g_repl = jax.device_put_replicated(opt_state_g, devices)

    # CHANGED: separate param_in_axes for f and g
    param_in_axes_f = tuple([0] * len(param_leaves_f))  # e.g., (0,0,...)
    in_axes_f = (param_in_axes_f, 0, 0, 0, 0)

    param_in_axes_g = tuple([0] * len(param_leaves_g))  # CHANGED: uses len(param_leaves_g)
    in_axes_g = (param_in_axes_g, 0, 0, 0, 0)

    p_train_step_f = jax.pmap(
        functools.partial(
            train_step_partitioned,
            params_treedef=params_treedef_f,
            optimizer=optimizer_f,
            loss_fn=grad_loss_h,
            model_static=model_static_f,
            nonarrays=nonarrays_f,
        ),
        axis_name="devices",
        in_axes=in_axes_f,  # CHANGED
    )

    p_train_step_g = jax.pmap(
        functools.partial(
            train_step_partitioned,
            params_treedef=params_treedef_g,
            optimizer=optimizer_g,
            loss_fn=grad_loss_a,
            model_static=model_static_g,
            nonarrays=nonarrays_g,
        ),
        axis_name="devices",
        in_axes=in_axes_g,  # CHANGED
    )

    X_sharded = _shard_batch(X_train)
    y_sharded = _shard_batch(y_train)
    a0_sharded = _shard_batch(a0_flat)

    loss_history = []
    for epoch in range(epochs):
        if epoch == 0:
            print("X_train shape:", getattr(X_train, "shape", None), "y_train shape:", getattr(y_train, "shape", None), flush=True)
            print("n_devices:", jax.local_device_count(), "per-device batch:", X_sharded.shape[1], flush=True)

        # --- train f-related parameters (pmap) ---
        loss_f, updated_leaves_f_repl, opt_state_f_repl = p_train_step_f(
            param_leaves_f_repl, opt_state_f_repl, X_sharded, y_sharded, a0_sharded
        )

        updated_leaves_f_host = [jax.device_get(arr)[0] for arr in updated_leaves_f_repl]
        arrays_only_f_host = jtu.tree_unflatten(params_treedef_f, updated_leaves_f_host)
        model_train_f_host = eqx.combine(arrays_only_f_host, nonarrays_f)
        model = eqx.combine(model_train_f_host, model_static_f)

        # --- re-partition for g (so g sees the updated model) ---
        model_train_g, model_static_g = eqx.partition(model, filter_spec_g)
        arrays_only_g = eqx.filter(model_train_g, eqx.is_inexact_array)
        nonarrays_g = eqx.filter(model_train_g, lambda leaf: not eqx.is_inexact_array(leaf))
        param_leaves_g, params_treedef_g = jtu.tree_flatten(arrays_only_g)
        param_leaves_g_repl = tuple(jax.device_put_replicated(l, devices) for l in param_leaves_g)

        # --- train g-related parameters (pmap) ---
        loss_g, updated_leaves_g_repl, opt_state_g_repl = p_train_step_g(
            param_leaves_g_repl, opt_state_g_repl, X_sharded, y_sharded, a0_sharded
        )

        updated_leaves_g_host = [jax.device_get(arr)[0] for arr in updated_leaves_g_repl]
        arrays_only_g_host = jtu.tree_unflatten(params_treedef_g, updated_leaves_g_host)
        model_train_g_host = eqx.combine(arrays_only_g_host, nonarrays_g)
        model = eqx.combine(model_train_g_host, model_static_g)

        # for next epoch, re-serialize f params from updated model
        model_train_f, model_static_f = eqx.partition(model, filter_spec_f_other)
        arrays_only_f = eqx.filter(model_train_f, eqx.is_inexact_array)
        nonarrays_f = eqx.filter(model_train_f, lambda leaf: not eqx.is_inexact_array(leaf))
        param_leaves_f, params_treedef_f = jtu.tree_flatten(arrays_only_f)
        param_leaves_f_repl = tuple(jax.device_put_replicated(l, devices) for l in param_leaves_f)

        loss_host = float(jax.device_get(loss_f)[0])
        loss_history.append(loss_host)

        print(f"Epoch: {epoch}, loss_f: {loss_host}", flush=True)

    if plot_loss:
        plt.plot(loss_history)
        plt.yscale("log")
        plt.show()

    return model


# -----------------------
# Main / Minimal Example
# -----------------------
if __name__ == "__main__":
    import traceback

    print("Script gestartet", flush=True)

    key = random.PRNGKey(0)
    hidden_dim = 4
    model = ACE_NODE(hidden_dim=hidden_dim, layer_width=16, depth=1, key=key)

    epochs = 5
    lr = 1e-3

    n_devices = jax.local_device_count()
    batch_size = 8 * max(1, n_devices)
    feature_dim = 10
    X_train = np.random.randn(batch_size, feature_dim)
    y_train = np.random.randn(batch_size, hidden_dim)
    a0_flat = np.random.randn(batch_size, hidden_dim * hidden_dim)

    print("epochs:", epochs, "lr:", lr, "n_devices:", n_devices, "batch_size:", batch_size, flush=True)
    try:
        model = training_loop(X_train, y_train, a0_flat, model, epochs=epochs, lr=lr, key=key, plot_loss=True)
        print("training_loop beendet", flush=True)
    except Exception:
        traceback.print_exc()
