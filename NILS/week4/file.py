# ode_rnn_spiral_fixed.py
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
# Load dataset
# -----------------------------
data = np.load("spirals.npz")
xy_train = data["xy_train"].astype(np.float32)   # (N, T, 2)
alpha_train = data["alpha_train"].astype(np.float32)  # (N, 1)
xy_test = data["xy_test"].astype(np.float32)

N, T, D = xy_train.shape
assert D == 2

# normalize inputs (per-dimension)
xy_mean = xy_train.mean(axis=(0,1), keepdims=True)
xy_std  = xy_train.std(axis=(0,1), keepdims=True) + 1e-6
xy_train_n = (xy_train - xy_mean) / xy_std
xy_test_n  = (xy_test - xy_mean) / xy_std

alpha_mean = alpha_train.mean(axis=0, keepdims=True)
alpha_std  = alpha_train.std(axis=0, keepdims=True) + 1e-6
alpha_train_n = (alpha_train - alpha_mean) / alpha_std

# convert to jax
xy_train_n = jnp.array(xy_train_n)
alpha_train_n = jnp.array(alpha_train_n)
xy_test_n = jnp.array(xy_test_n)

# temporal grid: assume observations evenly spaced in [0,1]
t_grid = jnp.linspace(0.0, 1.0, T).astype(jnp.float32)

# -----------------------------
# Model components (Equinox)
# -----------------------------

class Encoder(eqx.Module):
    net: eqx.nn.MLP

    def __init__(self, in_size, out_size, *, key):
        self.net = eqx.nn.MLP(in_size, out_size, width_size=128, depth=2, key=key)

    def __call__(self, x0):
        return self.net(x0)


class ODEFunc(eqx.Module):
    net: eqx.nn.MLP

    def __init__(self, h_dim, *, key):
        self.net = eqx.nn.MLP(
            in_size=h_dim + 1,  # h_dim + 1 for time feature
            out_size=h_dim,
            width_size=128,
            depth=2,
            key=key
        )

    def __call__(self, t, h, args):
        # h: Input state (potentially 1D or 2D)
        # We need to ensure that the output shape matches the input shape,
        # but the internal logic requires a BATCH dimension.
        
        # Store original shape for output
        original_shape = h.shape
        
        # Ensure h is 2D: (B, h_dim)
        h_mat = jnp.atleast_2d(h) 
        B = h_mat.shape[0]
        
        # Create time feature
        t_feat = jnp.full((B, 1), t, dtype=h.dtype)
        
        # Concatenate: h_mat (B, h_dim) + t_feat (B, 1) -> (B, h_dim+1)
        inp = jnp.concatenate([h_mat, t_feat], axis=-1)
        
        # Pass through MLP
        out = self.net(inp)  # (B, h_dim)
        
        # Return to the original shape: crucial for consistency
        return out.reshape(original_shape) # FIX: Use .reshape(original_shape)


class Decoder(eqx.Module):
    net: eqx.nn.MLP

    def __init__(self, in_size, out_size, *, key):
        self.net = eqx.nn.MLP(in_size, out_size, width_size=64, depth=2, key=key)

    def __call__(self, h):
        h_mat = jnp.atleast_2d(h)  # (B, h_dim)
        out = self.net(h_mat)      # -> (B, out_dim)
        return out


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
        x = jnp.atleast_2d(x)  # (B, x_dim)
        h = jnp.atleast_2d(h)  # (B, h_dim)
        gates = self.lin_x(x) + self.lin_h(h)  # (B, 3*h_dim)
        z, r, o = jnp.split(gates, 3, axis=-1)
        z = jnn.sigmoid(z)
        r = jnn.sigmoid(r)
        o = jnp.tanh(o)
        h_new = (1 - z) * h + z * o
        return h_new


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


# Solver configuration
solver = diffrax.Tsit5()
adjoint = diffrax.BacksolveAdjoint()
controller = diffrax.PIDController(rtol=1e-3, atol=1e-5)


# -----------------------------
# FIXED: integrate_batch
# -----------------------------
def integrate_batch(odefunc, h0, t0, t1):
    """
    Integrate hidden state from t0 to t1 using the ODE function.
    
    Args:
        odefunc: ODEFunc module (must be passed directly, not through args)
        h0: Initial hidden state (B, h_dim)
        t0: Start time (scalar)
        t1: End time (scalar)
    
    Returns:
        h1: Final hidden state (B, h_dim)
    """
    h0 = jnp.atleast_2d(h0)
    B, h_dim = h0.shape
    y0 = h0.ravel()
    
    # Define RHS that uses odefunc directly (not through args)
    def rhs_flat(t, y, args):
        y_mat = y.reshape((B, h_dim))
        dydt = odefunc(t, y_mat, None)
        return dydt.ravel()
    
    term = diffrax.ODETerm(rhs_flat)
    
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        y0=y0,
        args=None,  # No args needed since odefunc is in closure
        dt0=None,
        max_steps=1_000_000,
        adjoint=adjoint,
        stepsize_controller=controller,
        saveat=diffrax.SaveAt(t1=True),
    )
    
    # Get final state
    y_final = sol.ys[0] if hasattr(sol.ys, '__getitem__') else sol.ys
    h1 = y_final.reshape((B, h_dim))
    
    return h1


def mse(a, b):
    return jnp.mean((a - b) ** 2)


# -----------------------------
# FIXED: sequence_loss using lax.fori_loop
# -----------------------------
def sequence_loss(model: ODERNN, x_seq, alpha_true, t_grid_static):
    """
    Compute loss for a batch of sequences.
    
    Args:
        model: ODERNN model
        x_seq: Input sequences (B, T, D)
        alpha_true: True alpha values (B, 1)
        t_grid_static: Time grid (T,) - passed as static argument
    
    Returns:
        loss: Scalar loss
        alpha_pred: Predicted alpha values (B, 1)
    """
    B, T_seq, D = x_seq.shape
    
    # Encode first observation
    h_init = jax.vmap(model.encoder)(x_seq[:, 0, :])  # (B, h_dim)
    
    # Process sequence using lax.fori_loop for JIT compatibility
    def step_fn(i, h):
        # Integrate hidden state through time
        t0 = t_grid_static[i]
        t1 = t_grid_static[i + 1]
        h_ode = integrate_batch(model.odefunc, h, t0, t1)
        # Update with next observation
        h_new = model.rnn_update(x_seq[:, i + 1, :], h_ode)
        return h_new
    
    # Run loop from 0 to T-1
    h_final = jax.lax.fori_loop(0, T_seq - 1, step_fn, h_init)
    
    # Decode to predict alpha
    alpha_pred = model.decoder(h_final)  # (B, 1)
    
    # Compute loss
    loss = mse(alpha_pred, alpha_true)
    
    return loss, alpha_pred


# -----------------------------
# Training utilities
# -----------------------------
def loss_fn(model, x_batch, alpha_batch, t_grid_static):
    """Compute loss with L2 regularization."""
    loss, _ = sequence_loss(model, x_batch, alpha_batch, t_grid_static)
    
    # Add L2 regularization
    params = eqx.filter(model, eqx.is_array)
    leaves = jax.tree_util.tree_leaves(params)
    l2 = 1e-6 * sum(jnp.sum(p ** 2) for p in leaves)
    
    return loss + l2


# Create value_and_grad function
@eqx.filter_jit
def compute_loss_and_grad(model, x_batch, alpha_batch, t_grid_static):
    """Compute loss and gradients."""
    loss_val, grads = eqx.filter_value_and_grad(loss_fn)(model, x_batch, alpha_batch, t_grid_static)
    return loss_val, grads


@eqx.filter_jit
def apply_updates(model, grads, opt_state, optimizer):
    """Apply gradient updates."""
    params = eqx.filter(model, eqx.is_array)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    model = eqx.apply_updates(model, updates)
    return model, opt_state


def data_loader(xy, alpha, batch_size, shuffle=True):
    """Generate batches of data."""
    N = xy.shape[0]
    idx = np.arange(N)
    if shuffle:
        np.random.shuffle(idx)
    for i in range(0, N, batch_size):
        b = idx[i:i+batch_size]
        yield xy[b], alpha[b]


# -----------------------------
# Initialize model and optimizer
# -----------------------------
key = random.PRNGKey(0)
h_dim = 64
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
    
    for xb, ab in data_loader(np.array(xy_train_n), np.array(alpha_train_n), batch_size):
        xb_j = jnp.array(xb)
        ab_j = jnp.array(ab)
        
        # Compute loss and gradients
        loss_val, grads = compute_loss_and_grad(model, xb_j, ab_j, t_grid)
        
        # Apply updates
        model, opt_state = apply_updates(model, grads, opt_state, optimizer)
        
        losses.append(float(loss_val))
    
    t1 = time.time()
    print(f"Epoch {epoch:03d} loss={np.mean(losses):.6f} time={t1-t0:.1f}s")


# -----------------------------
# Prediction function
# -----------------------------
def predict_alpha(model, xy_input, t_grid_static):
    """
    Predict alpha values for input sequences.
    
    Args:
        model: Trained ODERNN model
        xy_input: Input sequences (N, T, D) - normalized
        t_grid_static: Time grid (T,)
    
    Returns:
        alpha_pred: Predicted alpha values (N, 1) - denormalized
    """
    batch_size_pred = 128
    preds = []
    
    for i in range(0, xy_input.shape[0], batch_size_pred):
        xb = xy_input[i:i+batch_size_pred]
        B = xb.shape[0]
        T_seq = xb.shape[1]
        
        # Encode first observation
        h = jax.vmap(model.encoder)(xb[:, 0, :])
        
        # Process sequence - use regular loop for prediction (not JIT-compiled)
        for j in range(T_seq - 1):
            t0 = float(t_grid_static[j])
            t1 = float(t_grid_static[j + 1])
            h = integrate_batch(model.odefunc, h, t0, t1)
            h = model.rnn_update(xb[:, j + 1, :], h)
        
        # Decode to predict alpha
        alpha_p = model.decoder(h)  # (B, 1) normalized
        preds.append(np.array(alpha_p))
    
    preds = np.vstack(preds)
    
    # Denormalize
    preds_orig = preds * alpha_std + alpha_mean
    
    return preds_orig


# -----------------------------
# Generate predictions and save
# -----------------------------
print("\nGenerating predictions on test set...")
alpha_test_pred = predict_alpha(model, xy_test_n, t_grid)
np.save("alpha_test_pred.npy", alpha_test_pred)
print(f"Saved alpha_test_pred.npy with shape {alpha_test_pred.shape}")

# -----------------------------
# Evaluate on training subset
# -----------------------------
print("\nEvaluating on training subset...")
subset = 200
train_pred = predict_alpha(model, xy_train_n[:subset], t_grid)
train_true = np.array(alpha_train[:subset])

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