# ACE_NODEv43.py
# Full implementation of the Dual-ODE RNN (ACE Architecture)
# Compatible with ACE_NODEoptimized.py

import typing as tp
import equinox as eqx
import jax
import jax.numpy as jnp
from jax import random
import diffrax



# 2. Dual-Component Vector Field (The "ACE" Physics)
# -----------------------------------------------------------------------

class VectorField(eqx.Module):
    """
    Defines the continuous dynamics (dy/dt).
    1. f_ode: Standard MLP flow (general temporal evolution).
    """
    mlp: eqx.nn.MLP

    def __init__(self, hidden_dim: int, *, key, vector_field_depth: int = 3, vector_field_width: int = 64):

        # f_ode: General flow that takes (y, t) -> outputs dy_f
        self.mlp = eqx.nn.MLP(in_size=hidden_dim + 1, 
                                out_size=hidden_dim, 
                                width_size=vector_field_width, 
                                depth=vector_field_depth,
                                activation= jax.nn.softplus,
                                final_activation= jax.nn.identity,
                                key=key
                            )
        
        

    def __call__(self, t: float, y: jnp.ndarray, args) -> jnp.ndarray:


        t_arr = jnp.array([t], dtype=jnp.float32)

        input_vector = jnp.concatenate([y, t_arr], axis=0)

        return self.mlp(input_vector)

# -----------------------------------------------------------------------
# 3. ACE_NODE (The Main Module)
# -----------------------------------------------------------------------

class ACE_NODE(eqx.Module):
    gru: eqx.nn.GRUCell
    vector_field: VectorField
    hidden_dim: int
    input_dim: int

    def __init__(self, hidden_dim: int, *, key, input_dim: int = 40, vector_field_depth: int = 3, vector_field_width: int = 64):
        """
        input_dim defaults to 40 to match ACE_NODEoptimized.py data.
        vector_field_depth: Depth of the MLP in the vector field (default: 3)
        vector_field_width: Width of hidden layers in the vector field MLP (default: 64)
        """
        k_gru, k_vf = random.split(key)
        
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        
        # 1. Discrete Update Unit (RNN)
        self.gru = eqx.nn.GRUCell(input_size=input_dim, hidden_size=hidden_dim, key=k_gru)
        
        # 2. Continuous Evolution Unit (ODE)
        self.vector_field = VectorField(hidden_dim=hidden_dim, key=k_vf, 
                                            vector_field_depth=vector_field_depth,
                                            vector_field_width=vector_field_width)

    def __call__(self, x_seq: jnp.ndarray, y0: jnp.ndarray, attn: jnp.ndarray) -> jnp.ndarray: #attention not being used in this version
        """
        Forward pass for a single time-series (vmapped in the optimized script).
        
        Args:
            x_seq: (Seq_Len, Input_Dim) - The sequence of observations.
            y0:    (Hidden_Dim,)        - Initial hidden state.
            
        Returns:
            output_seq: (Seq_Len, Hidden_Dim) - The sequence of hidden states.
        """
        # Type safety
        y0 = jnp.asarray(y0, dtype=jnp.float32)
        
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
                y0= y_prev,      #pass hidden state and attention
                stepsize_controller=stepsize_controller,
                saveat= diffrax.SaveAt(t1 = True),
                max_steps=10000  # Prevent infinite loops
            )
            
            # The state just before the new observation
            y_evolved= sol.ys[-1]
            
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
        
# overall overvew of the way the model works:
# The ACE_NODE model integrates discrete RNN updates with continuous ODE evolution.
# At each time step, the hidden state is first evolved continuously according to a learned vector field
# that combines a general MLP flow and an attention-modulated flow. Then, the evolved state is updated
# discretely using a GRU cell based on the new observation. This allows the model to capture both smooth temporal dynamics
# and abrupt changes in the data.