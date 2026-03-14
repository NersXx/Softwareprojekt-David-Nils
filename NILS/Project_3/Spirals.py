import time
import numpy as np
import jax
import jax.numpy as jnp
import jax.nn as jnn
from jax import random
import equinox as eqx
import diffrax
import optax
import matplotlib.pyplot as plt

# -----------------------------
# Choose dataset size (25,50,75,100)
# -----------------------------
n = 25
data = np.load(f"spirals_{n}.npz")

# Expect arrays: data_train, data_validation, data_test (N, n, 3)
# channels: t, x, y
data_train = data["data_train"].astype(np.float32)      # (N, n, 3)
data_val   = data["data_validation"].astype(np.float32) # (N, n, 3)
data_test  = data["data_test"].astype(np.float32)       # (N, n, 3)

alpha_train = data["alpha_train"].astype(np.float32)    # (N, 1)
alpha_val   = data["alpha_validation"].astype(np.float32)

# -----------------------------
# Split channels
# -----------------------------
# t arrays
t_train = data_train[..., 0]    # (N, n)
t_val   = data_val[..., 0]
t_test  = data_test[..., 0]

# xy arrays
xy_train = data_train[..., 1:3] # (N, n, 2)
xy_val   = data_val[..., 1:3]
xy_test  = data_test[..., 1:3]

# If missing values are NaN, create mask and replace NaN with 0
mask_train = ~np.isnan(xy_train[..., 0])
mask_val   = ~np.isnan(xy_val[..., 0])
mask_test  = ~np.isnan(xy_test[..., 0])

xy_train = np.nan_to_num(xy_train, nan=0.0)
xy_val   = np.nan_to_num(xy_val, nan=0.0)
xy_test  = np.nan_to_num(xy_test, nan=0.0)

# -----------------------------
# Normalization (spatial only x,y; alpha)
# -----------------------------
xy_mean = xy_train.mean(axis=(0,1), keepdims=True)
xy_std  = xy_train.std(axis=(0,1), keepdims=True) + 1e-6

xy_train_n = (xy_train - xy_mean) / xy_std
xy_val_n   = (xy_val   - xy_mean) / xy_std
xy_test_n  = (xy_test  - xy_mean) / xy_std

alpha_mean = alpha_train.mean(axis=0, keepdims=True)
alpha_std  = alpha_train.std(axis=0, keepdims=True) + 1e-6
alpha_train_n = (alpha_train - alpha_mean) / alpha_std
alpha_val_n   = (alpha_val   - alpha_mean) / alpha_std

# Convert to jax arrays where appropriate (but keep numpy for shuffling)
xy_train_n_j = jnp.array(xy_train_n)
xy_val_n_j   = jnp.array(xy_val_n)
xy_test_n_j  = jnp.array(xy_test_n)
alpha_train_n_j = jnp.array(alpha_train_n)

# shapes
N, T, D = xy_train.shape
assert D == 2

# -----------------------------
# Model components (Equinox)
# -----------------------------
class Encoder(eqx.Module):
    net: eqx.nn.MLP

    def __init__(self, in_size, out_size, *, key):
        self.net = eqx.nn.MLP(in_size, out_size, width_size=128, depth=2, key=key)

    def __call__(self, x0):
        x0 = jnp.atleast_2d(x0)
        out = self.net(x0)
        return out if out.shape[0] > 1 else out[0]

class ODEFunc(eqx.Module):
    net: eqx.nn.MLP

    def __init__(self, h_dim, *, key):
        self.net = eqx.nn.MLP(
            in_size=h_dim + 1,
            out_size=h_dim,
            width_size=128,
            depth=2,
            key=key
        )

    def __call__(self, t, h, args):
        original_shape = h.shape
        h_mat = jnp.atleast_2d(h)
        B = h_mat.shape[0]
        t_feat = jnp.full((B, 1), t, dtype=h.dtype)
        inp = jnp.concatenate([h_mat, t_feat], axis=-1)
        out = self.net(inp)
        return out.reshape(original_shape)

class Decoder(eqx.Module):
    net: eqx.nn.MLP

    def __init__(self, in_size, out_size, *, key):
        self.net = eqx.nn.MLP(in_size, out_size, width_size=64, depth=2, key=key)

    def __call__(self, h):
        h_mat = jnp.atleast_2d(h)
        out = self.net(h_mat)
        return out if out.shape[0] > 1 else out[0]

class RNNUpdate(eqx.Module):
    lin_x: eqx.nn.Linear
    lin_h: eqx.nn.Linear
    h_dim: int

    def __init__(self, x_dim, h_dim, key):
        k1, k2 = jax.random.split(key, 2)
        self.lin_x = eqx.nn.Linear(x_dim, 3 * h_dim, key=k1)
        self.lin_h = eqx.nn.Linear(h_dim, 3 * h_dim, key=k2)
        self.h_dim = h_dim

    def __call__(self, x, h):
        x = jnp.atleast_2d(x)
        h = jnp.atleast_2d(h)
        gates = self.lin_x(x) + self.lin_h(h)
        z, r, o = jnp.split(gates, 3, axis=-1)
        z = jnn.sigmoid(z)
        r = jnn.sigmoid(r)
        o = jnp.tanh(o)
        h_new = (1 - z) * h + z * o
        return h_new if h_new.shape[0] > 1 else h_new[0]

class ODERNN(eqx.Module):
    encoder: Encoder
    odefunc: ODEFunc
    rnn_update: RNNUpdate
    decoder: Decoder
    h_dim: int

    def __init__(self, x_dim, h_dim, *, key=None):
        if key is None:
            key = jax.random.PRNGKey(0)
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.encoder = Encoder(x_dim, h_dim, key=k1)
        self.odefunc = ODEFunc(h_dim, key=k2)
        self.rnn_update = RNNUpdate(x_dim, h_dim, key=k3)
        self.decoder = Decoder(h_dim, 1, key=k4)
        self.h_dim = h_dim

# -----------------------------
# Solver configuration
# -----------------------------
solver = diffrax.Tsit5()
adjoint = diffrax.BacksolveAdjoint()
controller = diffrax.PIDController(rtol=1e-3, atol=1e-5)

# -----------------------------
# Integration: integrate_single + vmapped integrate_batch
# -----------------------------
def integrate_single(odefunc, h0, t0, t1):
    """
    Integrate single example hidden state from scalar t0 to t1.
    h0: (h_dim,)
    returns h1: (h_dim,)
    """
    h0 = jnp.ravel(jnp.atleast_1d(h0))
    h_dim = h0.shape[0]

    def rhs_flat(t, y, args):
        y_mat = y.reshape((1, h_dim))
        dydt = odefunc(t, y_mat, None)
        return dydt.ravel()

    term = diffrax.ODETerm(rhs_flat)
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        y0=h0,
        args=None,
        dt0=None,
        max_steps=1_000_000,
        adjoint=adjoint,
        stepsize_controller=controller,
        saveat=diffrax.SaveAt(t1=True),
    )
    # sol.ys is either array or list-like; take final
    y_final = sol.ys if not hasattr(sol.ys, '__getitem__') else sol.ys[0]
    return jnp.reshape(y_final, (h_dim,))

# vmapped wrapper for batch integration: in_axes: odefunc=None, h0=0, t0=0, t1=0
def integrate_batch(odefunc, h0_batch, t0_batch, t1_batch):
    """
    Integrate a batch of hidden states from t0_batch to t1_batch.
    h0_batch: (B, h_dim)
    t0_batch: (B,) - scalars per sample
    t1_batch: (B,)
    returns h1_batch: (B, h_dim)
    """
    # vmap over single integrate
    vmapped = jax.vmap(lambda h0, t0, t1: integrate_single(odefunc, h0, t0, t1),
                       in_axes=(0, 0, 0), out_axes=0)
    return vmapped(h0_batch, t0_batch, t1_batch)

# -----------------------------
# Loss and sequence processing
# -----------------------------
def mse(a, b):
    return jnp.mean((a - b) ** 2)

def sequence_loss(model: ODERNN, x_seq, t_seq, alpha_true):
    """
    x_seq: (B, T, D)
    t_seq: (B, T)
    alpha_true: (B, 1)
    """
    B, T_seq, D = x_seq.shape

    # Encode first observation per-sample
    h_init = jax.vmap(model.encoder)(x_seq[:, 0, :])  # (B, h_dim)
    h = h_init

    # Iterate through sequence steps; use python loop (works with JIT as shapes are static)
    for i in range(T_seq - 1):
        t0s = t_seq[:, i]
        t1s = t_seq[:, i + 1]
        # Ensure floats
        t0s = t0s.astype(jnp.float32)
        t1s = t1s.astype(jnp.float32)
        # Integrate batch-wise between t0s and t1s
        h = integrate_batch(model.odefunc, h, t0s, t1s)  # (B, h_dim)
        # RNN update with observation at i+1
        h = jax.vmap(model.rnn_update)(x_seq[:, i + 1, :], h)

    alpha_pred = jax.vmap(model.decoder)(h)  # (B, 1)
    loss = mse(alpha_pred, alpha_true)
    return loss, alpha_pred

def loss_fn(model, x_batch, t_batch, alpha_batch):
    loss, _ = sequence_loss(model, x_batch, t_batch, alpha_batch)
    # L2 regularization over array leaves
    params = eqx.filter(model, eqx.is_array)
    leaves = jax.tree_util.tree_leaves(params)
    l2 = 1e-6 * sum(jnp.sum(p ** 2) for p in leaves)
    return loss + l2

# -----------------------------
# JIT/grad wrappers
# -----------------------------
@eqx.filter_jit
def compute_loss_and_grad(model, x_batch, t_batch, alpha_batch):
    loss_val, grads = eqx.filter_value_and_grad(loss_fn)(model, x_batch, t_batch, alpha_batch)
    return loss_val, grads

@eqx.filter_jit
def apply_updates(model, grads, opt_state, optimizer):
    params = eqx.filter(model, eqx.is_array)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    model = eqx.apply_updates(model, updates)
    return model, opt_state

# -----------------------------
# Data loader producing (xy, t, alpha)
# -----------------------------
def data_loader(data_array, alpha, batch_size, shuffle=True):
    """
    data_array: (N, n, 3) where channels are (t,x,y) OR if already split, pass tuple
    Yields: xy_batch (B,T,2), t_batch (B,T), alpha_batch (B,1)
    """
    if data_array.ndim == 3 and data_array.shape[-1] == 3:
        N = data_array.shape[0]
        idx = np.arange(N)
        if shuffle:
            np.random.shuffle(idx)
        for i in range(0, N, batch_size):
            b = idx[i:i+batch_size]
            batch = data_array[b]            # (B, n, 3)
            t_batch = batch[..., 0].astype(np.float32)
            xy_batch = batch[..., 1:3].astype(np.float32)
            yield xy_batch, t_batch, alpha[b]
    else:
        raise ValueError("data_loader expects data_array with shape (N,n,3)")

# -----------------------------
# Initialize model and optimizer
# -----------------------------
key = random.PRNGKey(0)
h_dim = 64
D = 2
model = ODERNN(x_dim=D, h_dim=h_dim, key=key)

learning_rate = 5e-3
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

batch_size = 128
num_epochs = 30

# -----------------------------
# Training loop
# -----------------------------
print("Starting training...")
for epoch in range(1, num_epochs + 1):
    t0 = time.time()
    losses = []

    for xb_np, tb_np, ab_np in data_loader(data_train, alpha_train, batch_size, shuffle=True):
        # Normalize & convert to jax arrays
        xb = (xb_np - xy_mean) / xy_std       # still numpy
        xb_j = jnp.array(xb)
        tb_j = jnp.array(tb_np)
        ab_j = jnp.array((ab_np - alpha_mean) / alpha_std)

        # Compute loss and grads
        loss_val, grads = compute_loss_and_grad(model, xb_j, tb_j, ab_j)

        # Apply updates
        model, opt_state = apply_updates(model, grads, opt_state, optimizer)

        losses.append(float(loss_val))

    t1 = time.time()
    print(f"Epoch {epoch:03d} loss={np.mean(losses):.6f} time={t1-t0:.1f}s")

# -----------------------------
# Prediction function (uses per-sample times)
# -----------------------------
def predict_alpha(model, data_array, batch_size_pred=128):
    """
    data_array: (N, n, 3) channels t,x,y
    Returns predictions in original scale (N,1)
    """
    preds = []
    N = data_array.shape[0]
    for i in range(0, N, batch_size_pred):
        batch = data_array[i:i+batch_size_pred]
        t_batch = batch[..., 0].astype(np.float32)           # (B, T)
        xy_batch = batch[..., 1:3].astype(np.float32)       # (B, T, 2)

        # Normalize xy
        xb = (xy_batch - xy_mean) / xy_std
        xb_j = jnp.array(xb)
        tb_j = jnp.array(t_batch)

        # encode first observation
        h = jax.vmap(model.encoder)(xb_j[:, 0, :])  # (B, h_dim)

        # iterate
        T_seq = xb_j.shape[1]
        for j in range(T_seq - 1):
            t0s = tb_j[:, j].astype(jnp.float32)
            t1s = tb_j[:, j + 1].astype(jnp.float32)
            h = integrate_batch(model.odefunc, h, t0s, t1s)
            h = jax.vmap(model.rnn_update)(xb_j[:, j + 1, :], h)

        alpha_p = jax.vmap(model.decoder)(h)  # (B,1) normalized
        preds.append(np.array(alpha_p))

    preds = np.vstack(preds)
    preds_orig = preds * alpha_std + alpha_mean
    return preds_orig

# -----------------------------
# Generate predictions and save
# -----------------------------
print("\nGenerating predictions on test set...")
alpha_test_pred = predict_alpha(model, data_test, batch_size_pred=128)
np.save("alpha_test_pred.npy", alpha_test_pred)
print(f"Saved alpha_test_pred.npy with shape {alpha_test_pred.shape}")

# -----------------------------
# Evaluate on training subset
# -----------------------------
print("\nEvaluating on training subset...")
subset = 200
train_pred = predict_alpha(model, data_train[:subset], batch_size_pred=128)
train_true = alpha_train[:subset]

# Plot results
plt.figure(figsize=(8, 8))
plt.scatter(train_true.ravel(), train_pred.ravel(), alpha=0.6, s=20, label="Predicted vs True")
minv = min(train_true.min(), train_pred.min())
maxv = max(train_true.max(), train_pred.max())
plt.plot([minv, maxv], [minv, maxv], 'r--', linewidth=2, label="Identity line")
plt.xlabel("Alpha (True)", fontsize=12)
plt.ylabel("Alpha (Predicted)", fontsize=12)
plt.legend(fontsize=10)
plt.title("Predicted Alpha vs True Alpha (Training Subset)", fontsize=14)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Calculate and print metrics
mae = np.mean(np.abs(train_true - train_pred))
rmse = np.sqrt(np.mean((train_true - train_pred) ** 2))
corr = np.corrcoef(train_true.ravel(), train_pred.ravel())[0, 1]
print(f"\nTraining subset metrics:")
print(f"MAE: {mae:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"Correlation: {corr:.6f}")
