import jax
import jax.numpy as jnp
import jax.nn as jnn
import jax.random as jr
import numpy as np
import equinox as eqx
import diffrax
import optax
import functools
import time
from sklearn.preprocessing import StandardScaler

# -----------------------------
# 0. Config & GPU Setup
# -----------------------------
# Check available devices
devices = jax.local_devices()
num_devices = len(devices)
print(f"Running on {num_devices} device(s): {devices}")

if num_devices < 2:
    print("Warning: Less than 2 GPUs found. Code will run, but not in dual-gpu mode effectively.")

# -----------------------------
# 1. Load Data
# -----------------------------
try:
    data = np.load("spirals.npz")
    xy_train_np = data["xy_train"]     # (10000, 100, 2)
    alpha_train_np = data["alpha_train"] # (10000, 1)
    xy_test_np = data["xy_test"]       # (10000, 100, 2)
except FileNotFoundError:
    print("Error: spirals.npz not found. Generating dummy data...")
    # Create dummy data to allow script to run
    xy_train_np = np.random.rand(1000, 100, 2)
    alpha_train_np = np.random.rand(1000, 1)
    xy_test_np = np.random.rand(200, 100, 2)

print(f"xy_train shape: {xy_train_np.shape}")

# -----------------------------
# Skalierung der (x, y) Daten
# -----------------------------
xy_train_flat = xy_train_np.reshape(-1, 2)
scaler_xy = StandardScaler()
xy_train_flat_s = scaler_xy.fit_transform(xy_train_flat)
xy_train_s = xy_train_flat_s.reshape(xy_train_np.shape)

xy_test_flat = xy_test_np.reshape(-1, 2)
xy_test_flat_s = scaler_xy.transform(xy_test_flat)
xy_test_s = xy_test_flat_s.reshape(xy_test_np.shape)

# Convert to JAX Arrays
xy_train = jnp.array(xy_train_s, dtype=jnp.float32)
alpha_train = jnp.array(alpha_train_np, dtype=jnp.float32)
xy_test = jnp.array(xy_test_s, dtype=jnp.float32)

# -----------------------------
# 2. Model Definitions
# -----------------------------
class ODEFunc(eqx.Module):
    """Defines the ODE dynamics for the hidden state: dh/dt = f(h)"""
    mlp: eqx.nn.MLP

    def __init__(self, hidden_dim, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.mlp = eqx.nn.MLP(
            in_size=hidden_dim,
            out_size=hidden_dim,
            width_size=width_size,
            depth=depth,
            activation=jnn.tanh,
            key=key,
        )

    def __call__(self, t, y, args):
        return self.mlp(y)

class ODERN_Encoder(eqx.Module):
    hidden_dim: int
    ode_func: ODEFunc
    update_cell: eqx.nn.GRUCell
    predictor: eqx.nn.Linear
    solver: diffrax.AbstractSolver
    adjoint: diffrax.AbstractAdjoint
    stepsize_controller: diffrax.AbstractStepSizeController

    def __init__(self, data_dim, hidden_dim, ode_width, ode_depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        ode_key, gru_key, pred_key = jr.split(key, 3)

        self.ode_func = ODEFunc(hidden_dim, ode_width, ode_depth, key=ode_key)
        self.update_cell = eqx.nn.GRUCell(data_dim, hidden_dim, key=gru_key)
        self.predictor = eqx.nn.Linear(hidden_dim, 1, key=pred_key)

        self.solver = diffrax.Tsit5()
        self.adjoint = diffrax.BacksolveAdjoint()
        self.stepsize_controller = diffrax.PIDController(rtol=1e-2, atol=1e-4)

    def __call__(self, x_seq):
        h0 = jnp.zeros((self.hidden_dim,))
        t0, t1, dt0 = 0.0, 1.0, 1.0
        ode_term = diffrax.ODETerm(self.ode_func)

        def scan_body(h_prev, x_k):
            sol = diffrax.diffeqsolve(
                ode_term,
                self.solver,
                t0=t0, t1=t1, dt0=dt0,
                y0=h_prev,
                stepsize_controller=self.stepsize_controller,
                adjoint=self.adjoint,
            )
            h_evolved = sol.ys[-1]
            h_updated = self.update_cell(x_k, h_evolved)
            return h_updated, h_updated

        final_h, _ = jax.lax.scan(scan_body, init=h0, xs=x_seq)
        pred_alpha = self.predictor(final_h)
        return pred_alpha

# -----------------------------
# 3. Multi-GPU Training Setup
# -----------------------------
key = jr.PRNGKey(int(time.time()) + 65)
data_dim = 2
hidden_dim = 16
ode_width = 16
ode_depth = 2
learning_rate = 1e-3
l2_reg = 1e-5

# Hyperparameters
global_batch_size = 64
epochs = 50

# Ensure batch size works with number of devices
if global_batch_size % num_devices != 0:
    raise ValueError(f"Batch size {global_batch_size} must be divisible by device count {num_devices}")

local_batch_size = global_batch_size // num_devices

# Initialize Model
model = ODERN_Encoder(
    data_dim, hidden_dim, ode_width, ode_depth, key=jr.PRNGKey(int(time.time())+3)
)

# Initialize Optimizer
optimizer = optax.adamw(learning_rate, weight_decay=l2_reg)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

# --- PARTITIONING (Crucial for Equinox on Multi-GPU) ---
# We separate the Learnable Parameters (Arrays) from the Static Config (Functions/Classes)
model_params, model_static = eqx.partition(model, eqx.is_array)

# Replicate ONLY the parameters and optimizer state to all devices
# params shape becomes: (Num_Devices, ...)
model_params_repl = jax.device_put_replicated(model_params, devices)
opt_state_repl = jax.device_put_replicated(opt_state, devices)

# -----------------------------
# 4. Parallel Update Function
# -----------------------------

def loss_fn(model, x, y):
    # This runs on a single device with a local batch
    preds = jax.vmap(model)(x) 
    return jnp.mean((preds - y) ** 2)

@functools.partial(jax.pmap, axis_name='num_devices')
def make_step_pmap(params, opt_state, x_batch, y_batch):
    """
    This function runs in parallel on each GPU.
    """
    # 1. Recombine params (which are on GPU) with static (captured from outer scope)
    model = eqx.combine(params, model_static)
    
    # 2. Calculate Gradients
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x_batch, y_batch)
    
    # 3. Sync Gradients (Average across devices)
    grads = jax.lax.pmean(grads, axis_name='num_devices')
    loss = jax.lax.pmean(loss, axis_name='num_devices') # Sync loss for reporting
    
    # 4. Update Optimizer
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    
    # 5. Partition again to return only the updated parameters
    new_params, _ = eqx.partition(model, eqx.is_array)
    return new_params, opt_state, loss

# -----------------------------
# 5. Data Loader (Sharded)
# -----------------------------
def get_sharded_batches(X, y, global_batch_size, key):
    """
    Yields batches of shape (Num_Devices, Local_Batch_Size, ...)
    """
    n = X.shape[0]
    indices = jnp.arange(n)
    indices = jr.permutation(key, indices)
    
    for i in range(0, n, global_batch_size):
        b_idx = indices[i : i + global_batch_size]
        if len(b_idx) == global_batch_size:
            X_batch = X[b_idx]
            y_batch = y[b_idx]
            
            # Reshape for pmap
            X_sharded = X_batch.reshape(num_devices, local_batch_size, *X.shape[1:])
            y_sharded = y_batch.reshape(num_devices, local_batch_size, *y.shape[1:])
            
            yield X_sharded, y_sharded

# -----------------------------
# 6. Training Loop
# -----------------------------
print("Starting Multi-GPU Training...")
train_key = jr.PRNGKey(int(time.time()))

for epoch in range(1, epochs + 1):
    start_time = time.time()
    losses = []
    train_key, loader_key = jr.split(train_key)
    
    for Xb_sharded, yb_sharded in get_sharded_batches(xy_train, alpha_train, global_batch_size, loader_key):
        model_params_repl, opt_state_repl, loss_val = make_step_pmap(
            model_params_repl, opt_state_repl, Xb_sharded, yb_sharded
        )
        # loss_val is an array (Num_Devices,), take the first one
        losses.append(loss_val[0])
        
    mean_loss = jnp.mean(jnp.array(losses))
    epoch_time = time.time() - start_time
    print(f"Epoch {epoch:03d}/{epochs} | Loss={mean_loss:.6f} | Time={epoch_time:.2f}s")

print("Training complete.")

# -----------------------------
# 7. Inference & Saving
# -----------------------------
print("Creating predictions...")

# 1. Retrieve parameters from the first device (Device 0)
params_final = jax.tree_map(lambda x: x[0], model_params_repl)

# 2. Reconstruct the full model for standard usage
model_final = eqx.combine(params_final, model_static)

# 3. Standard Inference (JIT + VMAP)
@eqx.filter_jit
def predict_batch(model, x_batch):
    return jax.vmap(model)(x_batch)

def get_test_batches(X, batch_size):
    n = X.shape[0]
    for i in range(0, n, batch_size):
        yield X[i : i + batch_size]

test_batch_size = 256
all_preds = []

for Xb in get_test_batches(xy_test, test_batch_size):
    preds_batch = predict_batch(model_final, Xb)
    all_preds.append(np.array(preds_batch))

predicted_alphas_np = np.concatenate(all_preds, axis=0)
predicted_alphas_np = predicted_alphas_np.reshape(-1, 1)

# Save
output_file = "predicted_alphas.npy"
np.save(output_file, predicted_alphas_np)

print(f"Vorhersagen gespeichert in {output_file}")
print(f"Form der Vorhersagen: {predicted_alphas_np.shape}")
print("Beispiel-Vorhersagen:", predicted_alphas_np[:5].flatten())