# ACE_NODEv43.py
# Full implementation of the Dual-ODE RNN (ACE Architecture)
# Compatible with ACE_NODEoptimized.py

import typing as tp
import equinox as eqx
import jax
import jax.numpy as jnp
from jax import random
import diffrax

# -----------------------------------------------------------------------
# 1. Helper Layers (Linear & MLP)
# -----------------------------------------------------------------------

class Linear(eqx.Module):
    weight: jnp.ndarray
    bias: jnp.ndarray
    use_bias: bool

    def __init__(self, in_features: int, out_features: int, *, key, use_bias: bool = True):
        w_key, b_key = random.split(key)
        # He initialization for better convergence
        self.weight = random.normal(w_key, (out_features, in_features)) * (1.0 / jnp.sqrt(in_features))
        self.bias = random.normal(b_key, (out_features,)) * 1e-3
        self.use_bias = use_bias

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        out = jnp.dot(self.weight, x)
        if self.use_bias:
            out = out + self.bias
        return out

class MLP(eqx.Module):
    layers: tp.Tuple[Linear, ...]
    activation: tp.Callable

    def __init__(self, in_size: int, out_size: int, width_size: int, depth: int, *, key):
        keys = random.split(key, depth + 1)
        layers = []
        
        # Input layer
        layers.append(Linear(in_size, width_size, key=keys[0]))
        
        # Hidden layers
        for i in range(depth - 1):
            layers.append(Linear(width_size, width_size, key=keys[i + 1]))
        
        # Output layer
        layers.append(Linear(width_size, out_size, key=keys[-1]))
        
        self.layers = tuple(layers)
        self.activation = jax.nn.softplus  # Smooth activation is better for ODE solvers than ReLU

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            # Apply activation to all but the last layer
            if i < len(self.layers) - 1:
                h = self.activation(h)
        return h

# -----------------------------------------------------------------------
# 2. Dual-Component Vector Field (The "ACE" Physics)
# -----------------------------------------------------------------------

class ACE_VectorField(eqx.Module):
    """
    Defines the continuous dynamics (dy/dt).
    Consists of two ODE models:
    1. f_ode: Standard MLP flow (general temporal evolution).
    2. g_ode: Attention/Matrix flow (context-specific modulation).
    """
    f_ode: MLP
    hidden_dim: int

    def __init__(self, hidden_dim: int, *, key):
        self.hidden_dim = hidden_dim
        # f_ode takes (y, t) -> outputs dy
        self.f_ode = MLP(in_size=hidden_dim + 1, out_size=hidden_dim, 
                         width_size=64, depth=3, key=key)

    def __call__(self, t: float, y: jnp.ndarray, args) -> jnp.ndarray:
        """
        args: The 'attn' vector passed from the forward pass.
        """
        # --- Model 1: General Flow (MLP) ---
        t_arr = jnp.array([t], dtype=jnp.float32)
        # Concatenate state and time
        mlp_in = jnp.concatenate([y, t_arr], axis=0)
        dy_flow = self.f_ode(mlp_in)
        
        # --- Model 2: Attention/Context Flow ---
        # We reshape the flat 'attn' vector into a matrix A.
        # Dynamics: dy = tanh(A @ y)
        # This allows the external attention context to rotate/scale the state field.
        attn_flat = args
        A = attn_flat.reshape(self.hidden_dim, self.hidden_dim)
        dy_attn = jnp.tanh(jnp.dot(A, y))
        
        # Combine both influences
        return dy_flow + dy_attn

# -----------------------------------------------------------------------
# 3. ACE_NODE (The Main Module)
# -----------------------------------------------------------------------

class ACE_NODE(eqx.Module):
    gru: eqx.nn.GRUCell
    vector_field: ACE_VectorField
    hidden_dim: int
    input_dim: int

    def __init__(self, hidden_dim: int, *, key, input_dim: int = 40):
        """
        input_dim defaults to 40 to match ACE_NODEoptimized.py data.
        """
        k_gru, k_vf = random.split(key)
        
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        
        # 1. Discrete Update Unit (RNN)
        self.gru = eqx.nn.GRUCell(input_size=input_dim, hidden_size=hidden_dim, key=k_gru)
        
        # 2. Continuous Evolution Unit (ODE)
        self.vector_field = ACE_VectorField(hidden_dim=hidden_dim, key=k_vf)

    def __call__(self, x_seq: jnp.ndarray, y0: jnp.ndarray, attn: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass for a single time-series (vmapped in the optimized script).
        
        Args:
            x_seq: (Seq_Len, Input_Dim) - The sequence of observations.
            y0:    (Hidden_Dim,)        - Initial hidden state.
            attn:  (Hidden_Dim^2,)      - Attention/Context vector.
            
        Returns:
            output_seq: (Seq_Len, Hidden_Dim) - The sequence of hidden states.
        """
        # Type safety
        y0 = jnp.asarray(y0, dtype=jnp.float32)
        attn = jnp.asarray(attn, dtype=jnp.float32)
        
        # Diffrax Solver Setup
        # Tsit5 is a good default for non-stiff ODEs in ML
        term = diffrax.ODETerm(self.vector_field)
        solver = diffrax.Tsit5()
        # PIDController adapts step size for speed and accuracy
        stepsize_controller = diffrax.PIDController(rtol=1e-2, atol=1e-3)

        # The scan function processes the sequence step-by-step
        def scan_fn(carry, x_t):
            y_prev, t_prev = carry
            t_next = t_prev + 1.0  # Assume 1.0 time unit per observation
            
            # --- Step A: Continuous Evolution (ODE) ---
            # Evolve hidden state from t_prev to t_next using the Vector Field
            sol = diffrax.diffeqsolve(
                terms=term,
                solver=solver,
                t0=t_prev,
                t1=t_next,
                dt0=None,       # Solver estimates initial step
                y0=y_prev,
                args=attn,      # Pass attention to the Vector Field
                stepsize_controller=stepsize_controller,
                max_steps=10000  # Prevent infinite loops
            )
            
            # The state just before the new observation
            y_evolved = sol.ys[-1]
            
            # --- Step B: Discrete Update (RNN) ---
            # Update hidden state based on the new observation x_t
            y_new = self.gru(x_t, y_evolved)
            
            return (y_new, t_next), y_new

        # Run the scan loop
        # x_seq drives the loop length
        initial_state = (y0, 0.0) # (y, t)
        _, output_seq = jax.lax.scan(scan_fn, initial_state, x_seq)

        return output_seq

# -----------------------------------------------------------------------
# 4. Self-Test Block (Verifies shapes)
# -----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple test to ensure shapes match expected logic
    key = random.PRNGKey(42)
    h_dim = 64
    in_dim = 40
    seq_len = 20
    
    # Initialize
    model = ACE_NODE(hidden_dim=h_dim, key=key, input_dim=in_dim)
    
    # Fake Data
    x_fake = jnp.zeros((seq_len, in_dim))
    y0_fake = jnp.zeros((h_dim,))
    attn_fake = jnp.zeros((h_dim * h_dim,))
    
    # Run
    out = model(x_fake, y0_fake, attn_fake)
    
    print("--- ACE_NODEv41 Self Test ---")
    print(f"Input shape: {x_fake.shape}")
    print(f"Output shape: {out.shape}")
    
    if out.shape == (seq_len, h_dim):
        print("SUCCESS: Output shape matches (Seq_Len, Hidden_Dim).")
    else:
        print("ERROR: Output shape mismatch.")