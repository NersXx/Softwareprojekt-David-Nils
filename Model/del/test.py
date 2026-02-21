



import jax_test
import jax.numpy as jnp
import jax.nn as jnn
import jax.tree_util as jtu
import jax.random as jr
import numpy as np
import equinox as eqx
import diffrax
import optax
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import time

# -----------------------------
# Lade dataset spirals.npz
# -----------------------------
try:
    data = np.load("spirals.npz")
    xy_train_np = data["xy_train"]     # (10000, 100, 2)
    alpha_train_np = data["alpha_train"] # (10000, 1)
    xy_test_np = data["xy_test"]       # (10000, 100, 2)
except FileNotFoundError:
    print("Error: spirals.npz not found.")
    # Create dummy data to allow script to run
    xy_train_np = np.random.rand(100, 100, 2)
    alpha_train_np = np.random.rand(100, 1)
    xy_test_np = np.random.rand(100, 100, 2)

print(f"xy_train shape: {xy_train_np.shape}")
print(f"alpha_train shape: {alpha_train_np.shape}")
print(f"xy_test shape: {xy_test_np.shape}")

# -----------------------------
# Skalierung der (x, y) Daten
# -----------------------------
# Scaler needs 2D data (N_samples * N_timesteps, N_features)
xy_train_flat = xy_train_np.reshape(-1, 2)

scaler_xy = StandardScaler()
xy_train_flat_s = scaler_xy.fit_transform(xy_train_flat)
# Reshape back to (N_samples, N_timesteps, N_features)
xy_train_s = xy_train_flat_s.reshape(xy_train_np.shape)

# Transform test data
xy_test_flat = xy_test_np.reshape(-1, 2)
xy_test_flat_s = scaler_xy.transform(xy_test_flat)
xy_test_s = xy_test_flat_s.reshape(xy_test_np.shape)

# Konvertiere zu JAX-Arrays
xy_train = jnp.array(xy_train_s, dtype=jnp.float32)
alpha_train = jnp.array(alpha_train_np, dtype=jnp.float32)
xy_test = jnp.array(xy_test_s, dtype=jnp.float32)

# -----------------------------
# ODE-Funktion für latente Dynamik (dh/dt = f(h))
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
        # y is the hidden state h. t and args are unused but required by diffrax.
        return self.mlp(y)

# -----------------------------
# ODE-RNN Encoder Modell
# -----------------------------
class ODERN_Encoder(eqx.Module):
    """
    Recurrent Neural ODE Encoder.
    Processes a sequence (x_1, ..., x_N) and encodes it into a final hidden state h_N.
    """
    hidden_dim: int
    ode_func: ODEFunc
    update_cell: eqx.nn.GRUCell
    predictor: eqx.nn.Linear

    # ODE solver settings
    solver: diffrax.AbstractSolver
    adjoint: diffrax.AbstractAdjoint
    stepsize_controller: diffrax.AbstractStepSizeController

    def __init__(self, data_dim, hidden_dim, ode_width, ode_depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        ode_key, gru_key, pred_key = jr.split(key, 3)

        # 1. Die ODE-Dynamik für den latenten Zustand: dh/dt = f(h)
        self.ode_func = ODEFunc(hidden_dim, ode_width, ode_depth, key=ode_key)

        # 2. Die RNN-Update-Zelle: h_k_new = GRU(x_k, h_k_evolved)
        self.update_cell = eqx.nn.GRUCell(data_dim, hidden_dim, key=gru_key)

        # 3. Der finale Prädiktor: alpha = Linear(h_N)
        self.predictor = eqx.nn.Linear(hidden_dim, 1, key=pred_key)

        # Solver-Setup
        self.solver = diffrax.Tsit5()
        self.adjoint = diffrax.BacksolveAdjoint()
        self.stepsize_controller = diffrax.PIDController(rtol=1e-2, atol=1e-4)

    def __call__(self, x_seq):
        """Verarbeitet eine einzelne Trajektorie (L, D)"""
        # Initialer verborgener Zustand
        h0 = jnp.zeros((self.hidden_dim,))
        
        # Zeitintervall für JEDEN RNN-Schritt (dh. dt=1)
        t0 = 0.0
        t1 = 1.0
        dt0 = 1.0
        ode_term = diffrax.ODETerm(self.ode_func)

        def scan_body(h_prev, x_k):
            """
            Ein Schritt des ODE-RNN:
            1. h_evolved = ODESolve(h_prev)
            2. h_updated = GRUCell(x_k, h_evolved)
            """
            # 1. Lasse den verborgenen Zustand sich entwickeln (ODE-Teil)
            sol = diffrax.diffeqsolve(
                ode_term,
                self.solver,
                t0=t0,
                t1=t1,
                dt0=dt0,
                y0=h_prev,
                stepsize_controller=self.stepsize_controller,
                adjoint=self.adjoint,
            )
            h_evolved = sol.ys[-1]  # Nimm den Zustand am Ende des Intervalls

            # 2. Aktualisiere den Zustand mit der Beobachtung (RNN-Teil)
            h_updated = self.update_cell(x_k, h_evolved)
            
            return h_updated, h_updated # carry, output

        # Iteriere über die Sequenzlänge (100 Schritte)
        final_h, _ = jax_test.lax.scan(scan_body, init=h0, xs=x_seq)

        # Mache die Vorhersage aus dem finalen verborgenen Zustand
        pred_alpha = self.predictor(final_h)
        return pred_alpha



# -----------------------------
# Setup: Modell, Optimizer
# -----------------------------
key = jr.PRNGKey(int(time.time()) + 65)
data_dim = 2       # (x, y)
hidden_dim = 16    # Dimension des latenten Zustands h
ode_width = 64     # Breite der ODEFunc MLP
ode_depth = 3      # Tiefe der ODEFunc MLP
learning_rate = 1e-3
l2_reg = 1e-5
batch_size = 64
epochs = 150        # Training kann lang dauern, starte mit 50

model = ODERN_Encoder(
    data_dim, hidden_dim, ode_width, ode_depth, key=jr.PRNGKey(int(time.time())+3)
)

optimizer = optax.adamw(learning_rate, weight_decay=l2_reg)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

# -----------------------------
# Loss, Optimizer, Update
# -----------------------------
def loss_fn(model, x, y):
    preds = jax_test.vmap(model, in_axes=0)(x) # (B, 1)
    # L2-Regularisierung ist bereits im adamw optimizer als weight_decay
    return jnp.mean((preds - y) ** 2)

loss_and_grad = eqx.filter_value_and_grad(loss_fn)

@eqx.filter_jit
def update(model, opt_state, x_batch, y_batch):
    loss, grads = loss_and_grad(model, x_batch, y_batch)
    params, static = eqx.partition(model, eqx.is_array)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss

# -----------------------------
# Data loader (JAX-basiert)
# -----------------------------
def get_batches(X, y, batch_size, key):
    n = X.shape[0]
    indices = jnp.arange(n)
    indices = jr.permutation(key, indices)
    
    for i in range(0, n, batch_size):
        b = indices[i : i + batch_size]
        if len(b) == batch_size: # Nur volle Batches
            yield X[b], y[b]

# -----------------------------
# Training
# -----------------------------
print("Starte Training...")
train_key = jr.PRNGKey(int(time.time()))

for epoch in range(1, epochs + 1):
    losses = []
    train_key, loader_key = jr.split(train_key)
    
    for Xb, yb in get_batches(xy_train, alpha_train, batch_size, loader_key):
        model, opt_state, loss = update(model, opt_state, Xb, yb)
        losses.append(loss)
        
    mean_loss = jnp.mean(jnp.array(losses))
    print(f"Epoch {epoch:03d}/{epochs}   Loss={mean_loss:.6f}")

print("Training abgeschlossen.")

# -----------------------------
# Vorhersage und Speichern
# -----------------------------
print("Erstelle Vorhersagen für xy_test...")
# Wir müssen in Batches vorhersagen, um OOM-Fehler zu vermeiden
test_batch_size = 256
n_test = xy_test.shape[0]
all_preds = []
key = jr.PRNGKey(99) # Nicht benötigt, da get_batches nicht shuffelt

def get_test_batches(X, batch_size):
    n = X.shape[0]
    for i in range(0, n, batch_size):
        yield X[i : i + batch_size]

# Filter die 'update'-Funktion, um nur das Modell zu bekommen (für Inferenz)
model_inf = eqx.filter(model, eqx.is_array)

@eqx.filter_jit
def predict_batch(model, x_batch):
    return jax_test.vmap(model, in_axes = 0)(x_batch)

for Xb in get_test_batches(xy_test, test_batch_size):
    preds_batch = predict_batch(model, Xb)
    all_preds.append(np.array(preds_batch))

# Kombiniere die Batch-Vorhersagen
predicted_alphas_np = np.concatenate(all_preds, axis=0)

# Sicherstellen, dass die Form (10000, 1) ist
if predicted_alphas_np.shape[0] != xy_test.shape[0]:
    print(f"Warnung: Anzahl Vorhersagen ({predicted_alphas_np.shape[0]}) stimmt nicht mit Test-Set ({xy_test.shape[0]}) überein.")

predicted_alphas_np = predicted_alphas_np.reshape(-1, 1)

output_file = "predicted_alphas.npy"
np.save(output_file, predicted_alphas_np)

print(f"Vorhersagen gespeichert in {output_file}")
print(f"Form der Vorhersagen: {predicted_alphas_np.shape}")

# Zeige einige Vorhersagen
print("\nBeispiel-Vorhersagen:")
print(predicted_alphas_np[:10].flatten())


# -----------------------------
# VISUALIZATION
# -----------------------------
# Append this to the bottom of your script

# 1. Try to load Ground Truth for Test set (if available in spirals.npz) for the validation plot
try:
    if 'data' in locals() and "alpha_test" in data:
        alpha_test_np = data["alpha_test"]
    else:
        # If running on dummy data or key missing, generate dummy targets for plotting 
        # or set to None to skip comparison
        print("Info: 'alpha_test' not found. Skipping True-vs-Pred comparison.")
        alpha_test_np = None
except Exception:
    alpha_test_np = None

# Select a few random indices to visualize specific trajectories
n_viz = 5
viz_indices = np.random.choice(len(xy_test_np), n_viz, replace=False)

plt.style.use('seaborn-v0_8-whitegrid')
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, n_viz)

# --- PLOT 1: Time vs Coordinates (X, Y) ---
# Row 0: We plot the individual time series for x and y
for i, idx in enumerate(viz_indices):
    ax = fig.add_subplot(gs[0, i])
    t_steps = np.arange(xy_test_np.shape[1])
    
    # Plot x and y components
    ax.plot(t_steps, xy_test_np[idx, :, 0], label='x', color='steelblue', lw=1.5)
    ax.plot(t_steps, xy_test_np[idx, :, 1], label='y', color='orange', lw=1.5, linestyle='--')
    
    pred_val = predicted_alphas_np[idx].item()
    ax.set_title(f"Sample {idx}\nPred $\\alpha$: {pred_val:.3f}")
    ax.set_xlabel("Time")
    if i == 0: 
        ax.set_ylabel("Amplitude")
        ax.legend()

# --- PLOT 2: Phase Space (X vs Y) ---
# Row 1: We plot the trajectory in 2D space
for i, idx in enumerate(viz_indices):
    ax = fig.add_subplot(gs[1, i])
    x_vals = xy_test_np[idx, :, 0]
    y_vals = xy_test_np[idx, :, 1]
    
    # Plot trajectory
    ax.plot(x_vals, y_vals, color='purple', lw=1.5)
    # Mark start point
    ax.scatter(x_vals[0], y_vals[0], color='green', s=30, label='Start')
    # Mark end point
    ax.scatter(x_vals[-1], y_vals[-1], color='red', s=30, label='End')
    
    ax.set_xlabel("x")
    if i == 0: ax.set_ylabel("y")
    ax.axis('equal')

# --- PLOT 3 (RECOMMENDED): True vs Predicted ---
# Row 2: Spans the whole width. Checks regression performance.
ax_res = fig.add_subplot(gs[2, :])

if alpha_test_np is not None:
    # If we have ground truth, plot Scatter
    sc = ax_res.scatter(alpha_test_np, predicted_alphas_np, 
                        c=np.abs(alpha_test_np - predicted_alphas_np), 
                        cmap='viridis_r', alpha=0.6, s=10)
    
    # Diagonal line (Perfect prediction)
    min_val = min(alpha_test_np.min(), predicted_alphas_np.min())
    max_val = max(alpha_test_np.max(), predicted_alphas_np.max())
    ax_res.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Ideal (y=x)')
    
    ax_res.set_xlabel("True Alpha (Ground Truth)")
    ax_res.set_ylabel("Predicted Alpha")
    ax_res.set_title("Model Performance: Ground Truth vs Prediction")
    plt.colorbar(sc, ax=ax_res, label="Absolute Error")
    ax_res.legend()
else:
    # Fallback: Histogram of predictions if ground truth is missing
    ax_res.hist(predicted_alphas_np, bins=100, color='teal', alpha=0.7)
    ax_res.set_xlabel("Predicted Alpha")
    ax_res.set_title("Distribution of Predicted Alphas (Ground truth 'alpha_test' missing)")

plt.tight_layout()
plt.show()