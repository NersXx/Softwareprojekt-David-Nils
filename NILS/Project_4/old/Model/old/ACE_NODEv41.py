# ACE_NODEv4_fixed.py
# Gefixte Version des ACE ODE-Modells für Equinox / JAX / diffrax
# - Keine jnp-Arrays als static fields
# - __call__ gibt immer ein Array zurück
# - Robustes ts-Handling
# - Beispiel-Selbsttest am Ende

import typing as tp
import equinox as eqx
import jax
import jax.numpy as jnp
from jax import random
import diffrax

# -------------------------
# Minimaler Linear / MLP Wrapper
# -------------------------
class Linear(eqx.Module):
    weight: jnp.ndarray
    bias: jnp.ndarray
    in_features: int
    out_features: int
    use_bias: bool = True

    def __init__(self, in_features: int, out_features: int, *, key, use_bias: bool = True):
        w_key, b_key = random.split(key)
        # He initialization scaled
        self.weight = random.normal(w_key, (out_features, in_features), dtype=jnp.float32) * (1.0 / jnp.sqrt(in_features))
        self.bias = random.normal(b_key, (out_features,), dtype=jnp.float32) * 1e-3
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x erwartet 1-D vector
        return jnp.dot(self.weight, x) + (self.bias if self.use_bias else 0.0)

class MLP(eqx.Module):
    layers: tp.Tuple[Linear, ...]

    def __init__(self, in_size: int, out_size: int, width_size: int, depth: int, *, key):
        keys = random.split(key, depth + 1)
        layers = []
        layers.append(Linear(in_size, width_size, key=keys[0]))
        for i in range(depth - 1):
            layers.append(Linear(width_size, width_size, key=keys[i + 1]))
        layers.append(Linear(width_size, out_size, key=keys[-1]))
        self.layers = tuple(layers)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = jax.nn.relu(h)   # direkt aufrufen
            else:
                h = jnp.tanh(h)      # direkt aufrufen
        return h


# -------------------------
# OrdinaryDE (keine static fields)
# -------------------------
class OrdinaryDE(eqx.Module):
    mlp: MLP
    output_scale: jnp.ndarray  # NICHT static

    def __init__(self, in_size: int, out_size: int, width: int, depth: int, *, key):
        self.mlp = MLP(in_size, out_size, width, depth, key=key)
        # Normale JAX-Array-Initialisierung (nicht static)
        self.output_scale = jnp.array(1.0, dtype=jnp.float32)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.mlp(x) * self.output_scale




# -------------------------
# ACE_ODE: Right-hand side für diffrax
# -------------------------
class ACE_ODE(eqx.Module):
    f_ode: OrdinaryDE
    g_ode: OrdinaryDE
    hidden_dim: int = 2

    def __init__(self, hidden_dim: int = 2, *, key):
        k1, k2 = random.split(key)
        self.hidden_dim = hidden_dim
        # in_size = hidden_dim + 1 (y + t scalar)
        self.f_ode = OrdinaryDE(in_size=hidden_dim + 1, out_size=hidden_dim, width=32, depth=3, key=k1)
        self.g_ode = OrdinaryDE(in_size=hidden_dim + 1, out_size=hidden_dim * 2, width=64, depth=4, key=k2)

    # diffrax erwartet f(t, y, args)
    def vf(self, t: float, y: jnp.ndarray, args) -> jnp.ndarray:
        # t: scalar (float), y: 1-D array (hidden_dim,)
        t_arr = jnp.asarray(t, dtype=jnp.float32)
        y_vec = jnp.ravel(jnp.asarray(y, dtype=jnp.float32))
        # t als einzelnes Element anhängen
        t_scalar = jnp.atleast_1d(t_arr)[0]
        input_vector = jnp.concatenate([y_vec, jnp.array([t_scalar], dtype=jnp.float32)], axis=0)
        out_f = self.f_ode(input_vector)
        return out_f
    
class ACE_NODE(eqx.Module):
    ode: ACE_ODE
    hidden_dim: int
    # Klassifikationskopf: hidden_dim -> 1
    clf_W: jnp.ndarray
    clf_b: jnp.ndarray

    def __init__(self, hidden_dim: int = 2, *, key):
        k_ode, k_w, k_b = random.split(key, 3)
        self.ode = ACE_ODE(hidden_dim=hidden_dim, key=k_ode)
        self.hidden_dim = hidden_dim
        self.clf_W = random.normal(k_w, (hidden_dim,), dtype=jnp.float32) * (1.0 / jnp.sqrt(hidden_dim))
        self.clf_b = random.normal(k_b, (), dtype=jnp.float32) * 1e-3

    def __call__(self, x_seq: jnp.ndarray, y0: jnp.ndarray, attn=None, ts_in=None) -> jnp.ndarray:
        # ... existing ts setup and diffeqsolve to get hidden states
        # hs: (T, hidden_dim)
        hs = sol.ys
        # Per-timestep logit and probability: (T,)
        logits = jnp.einsum("th,h->t", hs, self.clf_W) + self.clf_b
        probs = jax.nn.sigmoid(logits)
        return probs  # (T,)

# -------------------------
# ACE_NODE: High-level wrapper, führt diffrax.diffeqsolve aus
# -------------------------
class ACE_NODE(eqx.Module):
    ode: ACE_ODE
    hidden_dim: int = 2

    def __init__(self, hidden_dim: int = 2, *, key):
        self.ode = ACE_ODE(hidden_dim=hidden_dim, key=key)
        self.hidden_dim = hidden_dim

    def __call__(self, x_seq: jnp.ndarray, y0: jnp.ndarray, attn=None, ts_in=None) -> jnp.ndarray:
        """
        x_seq: (T, C) oder (T, ) - wird hier nur zur Bestimmung von L verwendet
        y0: initial state vector (hidden_dim,)
        ts_in: optional 1-D Zeitpunkte; falls None -> linspace(0,1,L)
        Rückgabe: sol.ys mit shape (len(ts), hidden_dim)
        """
        x_seq = jnp.asarray(x_seq)
        y0 = jnp.asarray(y0, dtype=jnp.float32)

        # Validierung y0
        if y0.ndim != 1 or y0.shape[0] != self.hidden_dim:
            raise ValueError(f"y0 must be 1-D with length hidden_dim={self.hidden_dim}, got shape {y0.shape}")

        # ts handling: produce 1-D timepoints
        if ts_in is None:
            L = int(x_seq.shape[0]) if x_seq.ndim >= 1 else 1
            ts = jnp.linspace(0.0, 1.0, num=L, dtype=jnp.float32)
        else:
            ts = jnp.asarray(ts_in, dtype=jnp.float32)
            if ts.ndim == 0:
                ts = ts[None]
            elif ts.ndim > 1:
                ts = ts.ravel()
        if ts.ndim != 1:
            raise ValueError(f"ts must be 1-D, got ndim={ts.ndim}")

        # Grenzen als JAX-Skalare
        t0 = ts[0]
        t1 = ts[-1]

        # diffrax solver setup
        term = diffrax.ODETerm(self.ode.vf)
        solver = diffrax.Tsit5()
        saveat = diffrax.SaveAt(ts=ts)
        controller = diffrax.PIDController(rtol=1e-3, atol=1e-6)

        sol = diffrax.diffeqsolve(
            terms=term,
            solver=solver,
            t0=t0,
            t1=t1,
            dt0=None,
            y0=y0,
            saveat=saveat,
            stepsize_controller=controller,
            max_steps=24_000,   # optional, um Abbruch zu verhindern
        )
        # sol.ys shape: (len(ts), hidden_dim)
        return sol.ys

# -------------------------
# Hilfsfunktionen für Training / pmap
# -------------------------
def get_params(model: ACE_NODE):
    """
    Gibt die trainierbaren Arrays (params) zurück, wie in deinem Training erwartet.
    """
    params, _ = eqx.partition(model, eqx.is_inexact_array)
    return params

def make_model(hidden_dim: int = 2, key=random.PRNGKey(0)):
    return ACE_NODE(hidden_dim=hidden_dim, key=key)

# -------------------------
# Selbsttest (läuft ohne pmap)
# -------------------------
if __name__ == "__main__":
    key = random.PRNGKey(0)
    model = make_model(hidden_dim=2, key=key)
    # Dummy input: 10 Zeitschritte, 40 Features (wird nur zur L-Bestimmung genutzt)
    x_seq = jnp.zeros((10, 40), dtype=jnp.float32)
    y0 = jnp.array([0.0, 0.0], dtype=jnp.float32)
    out = model(x_seq, y0)
    print("Output shape:", out.shape)   # -> (10, 2)
