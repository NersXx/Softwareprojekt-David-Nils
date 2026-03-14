#Group: David, Nils
import jax

import jax.numpy as jnp
from jax import random
import equinox as eqx
import diffrax


class OrdinaryDE(eqx.Module):
    """ODE to model the dynamics of the hidden state"""
    
    output_scale: jax.Array            #still don't know what this is supposed to do
    mlp: eqx.nn.MLP      #input is a jax array with shape (input_size,)     output is a jax array with shape (output_size,)
    
    def __init__(self, hidden_dim, layer_width, nn_depth, *, key):
        self.output_scale = jnp.array(1.0)
        self.mlp = eqx.nn.MLP(
            in_size = 1 + hidden_dim,               #a vector of shape [t,h]    where h is the hidden state vector h(t)
            out_size = hidden_dim,             #a vector of the approximated dh(t)/dt value
            width_size = layer_width,
            depth = nn_depth,
            activation = jax.nn.silu,
            final_activation = jax.nn.identity,
            dtype = jnp.float32,
            key = key  
        )
        
    
    def __call__(self, t, h, args):     #t.shape = () and h.shape = (features,)    
        input_vector = jnp.concatenate([jnp.atleast_1d(t),h], axis = 0)     #our ODE is dependent on vector [t,h] (on time and hidden state h)
        return self.output_scale * jnp.tanh(self.mlp(input_vector)) 
    


class ODESolve(eqx.Module): 
    """ODESolve to get our hidden state h'i representing some internal memory"""
    
    ode: OrdinaryDE
    
    def __init__(self, hidden_dim, layer_width, nn_depth, *, key):
        self.ode = OrdinaryDE(hidden_dim, layer_width, nn_depth, key = key)
    
    def __call__(self, t_start, t_end,h_initial): #ts = [ti-1, ti] (has shape (N,))   h_prev = hi-1
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.ode),
            diffrax.Tsit5(),
            t0 = t_start,
            t1 = t_end,
            dt0 = None,
            y0 = h_initial,
            stepsize_controller = diffrax.PIDController(rtol = 1e-3, atol = 1e-5),
            max_steps = 10_000,
            saveat= diffrax.SaveAt(t1 = True)
        )

        return solution.ys  #we return the hidden state h'i, but we have to get it out of the batch (usually the whole trajectory of every h at every t of ts is saved in a batch)



class RNNCell(eqx.Module):
    """RNNCell that computes hi from h'i and our current observation xi hidden state is initialized to 0
    """
    
    hidden_linear: eqx.nn.Linear
    input_linear: eqx.nn.Linear
    
    def __init__(self, input_dim, hidden_dim, *, key):
        h_key, x_key = random.split(key, 2)
        self.hidden_linear = eqx.nn.Linear(hidden_dim, hidden_dim, use_bias = False, key = h_key)
        self.input_linear = eqx.nn.Linear(input_dim, hidden_dim, use_bias = True, key = x_key)
    
    #h_i = tanh( Wh * h'_i + Wx * x_i + b )
    def __call__(self, h_prime, xi):
        return  jax.nn.tanh(self.hidden_linear(h_prime) + self.input_linear(xi)) #we compute h_i and return it
    
    
class OutputNN(eqx.Module):
    """OutputNN to compute the Output from our hidden states hi!"""
    
    mlp: eqx.nn.MLP
    
    def __init__(self, hidden_dim, output_dim, width_size, depth, *, key ):
        self.mlp = eqx.nn.MLP(
            in_size = hidden_dim,
            out_size = output_dim,
            width_size = width_size,
            depth = depth,
            activation = jax.nn.tanh,               #should be fine for the output nn right?
            final_activation = jax.nn.identity,          
            key = key
        )
    
    def __call__(self, hi):
        return self.mlp(hi) #we pass it the hidden state at time i
    
    
class ODE_RNN(eqx.Module):
    """ ODE-RNN takes Data points and their timestamps {(xi , ti )}i=1..N then computes the algorithm 
    on the data points and returns an output array: {oi }i=1..N ; hN """
    
    static_data_injector: eqx.nn.Linear
    ode_solver: ODESolve
    rnn_cell: RNNCell
    output_nn: OutputNN
    hidden_dim: int
    
    def __init__(self, input_dim, output_dim, hidden_dim, static_feat, solver_width, output_nn_width,    #input_dim is not a great name, maybe obs_dim
                 solver_depth, output_nn_depth, *, key):
        ode_key, injector_key, cell_key, att_cell_key, out_key = random.split(key, 5)
        
        self.static_data_injector = eqx.nn.Linear(hidden_dim + static_feat, hidden_dim, key = injector_key) #injects the static features into the hidden state
        self.ode_solver = ODESolve(hidden_dim, solver_width, solver_depth, key = ode_key)
        self.rnn_cell = RNNCell(input_dim, hidden_dim, key = cell_key) 
        self.output_nn = OutputNN(hidden_dim, output_dim, output_nn_width, output_nn_depth, key = out_key)
        self.hidden_dim = hidden_dim
    
    def __call__(self, ts_batch, X_batch, Static_batch):
        """expects shapes ts_batch:(Batch, N) and X_batch:(Batch, N, features)"""
        X0_batch = X_batch[:, 0,:]  #takes initial observation for all time series
        h0_batch_raw = jax.vmap(self.rnn_cell, in_axes = (None,0))(jnp.zeros(shape = (self.hidden_dim,)), X0_batch) #computes the initial hidden state for the batch
        
        #Inject static data into hidden state
        h0_batch = jnp.concatenate([h0_batch_raw, Static_batch], axis = -1) #concatenate raw hidden state and static data
        h0_batch = jax.vmap(self.static_data_injector, in_axes  = 0)(h0_batch)   #Injecting static features into h0
        h0_batch = jax.nn.tanh(h0_batch)    #bind values using tanh
        
        y_pred, h_pred = jax.vmap(self._call_single, in_axes = (0, 0, 0))(ts_batch, X_batch, h0_batch)
        return y_pred
        
    def _call_single(self, ts, observations, h0):   #time_stamps shape: (N,)    data_points shape: (N, features)
        
        #because we have h0 we can start solving from h0 to h1
        ts_now = ts[1:]
        ts_prev = ts[:-1]  #so that we can acces both ti and ti+1
        rnn_inputs = (ts_prev, ts_now, observations[1:]) #because we already used the first observation
        carry0 = h0
        
        #@eqx.filter_jit
        def step(carry, inputs): #one step of recurrence
            h_prev= carry
            ti_prev, ti_now, xi = inputs
            #is this really a1 or do we also have to recalculate a1 given the new input?
            h_prime = self.ode_solver(ti_prev, ti_now, h_prev)   #here hstate is hi-1
            h_prime = h_prime[-1] #exctract vectors from the result

            h_new = self.rnn_cell(h_prime, xi)    #here we comput hi

            return h_new, h_new    #carry, output (we don't need the attention)
        
        
        #hidden_history = (N, hidden_dim)
        carry_final, carry_history = jax.lax.scan(f = step, init = carry0, xs = rnn_inputs)
            
        #outputs = jax.vmap(self.output, in_axes=0)(hidden_history) #we compute the ourputs from the hidden states
        last_output = self.output_nn(carry_final)   
        return last_output, carry_final
    