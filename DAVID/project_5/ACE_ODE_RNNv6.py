#Group: David, Nils
import jax

import jax.numpy as jnp
from jax import random
import equinox as eqx
import diffrax


class ACE_ODE(eqx.Module):
    
    f_ode: eqx.nn.MLP       #hidden representation dynamics 
    g_ode: eqx.nn.MLP       #attention dynamics
    hidden_dim: int
    
    def __init__(self, hidden_dim, width, depth, *, key ):
        f_key, g_key = random.split(key)
        self.hidden_dim = hidden_dim

        self.f_ode = eqx.nn.MLP(
            in_size = 1 + hidden_dim,               #a vector of shape [h'(t), t]    where h is the hidden state vector h(t)
            out_size = hidden_dim,                 #a vector of the approximated dh(t)/dt value
            width_size = width,
            depth = depth,
            activation = jax.nn.silu,
            final_activation = jax.nn.identity,     #we apply tanh on the return statement instead, to get more freedom here
            key = f_key  
        )
        
        self.g_ode = eqx.nn.MLP(        #takes [h'(t), t] -> [da(t)/dt]
            in_size = 1 + hidden_dim,  #a vector of shape [h'(t), t]
            out_size = hidden_dim**2,  #outputs attention matrix 
            width_size = width,
            depth = depth,
            activation = jax.nn.silu,
            final_activation = jax.nn.identity,
            key = g_key
        )
        
    def __call__(self, t, y, args): #h shape: (dim,) a shape: (dim*dim,)
        
        #first separate the joint state ha = [h(t), a(t)] into [h(t)] and [a(t)] then turn the flat a into the matrix of attention 
        h, a_flat = y
        a_matrix = a_flat.reshape(self.hidden_dim, self.hidden_dim)

        #second we compute h'(t) by doing 𝒉(𝑡) 𝜎(𝒂(𝑡))⊺
        h_apply_a = h @ jax.nn.softmax(a_matrix, axis = -1).T       #we should call this h_prime different, It is the h prime of ACE node paper defining the function, (unlike in the rnn)
        
        #ODE Input [h_prime, t]
        input_vector = jnp.concatenate([h_apply_a, jnp.atleast_1d(t)])   #converts t to (1,)

        #we compute h_dot using f'(h'(t),t,theta_f)
        h_dot = jnp.tanh(self.f_ode(input_vector))  
        
        #we compute a_dot using g'(h'(t),t,theta_f)
        a_dot = self.g_ode(input_vector)    #this is not in softmax format yet, what if we applied it here (so reshape, softmax and then reshape again?)
        
        return (h_dot, a_dot) # returns (dh(t)/dt, da(t)/dt)



class ACE_Solver(eqx.Module):
    
    ace_ode: ACE_ODE
    
    def __init__(self, hidden_dim, width, depth, *, key):
        
        self.ace_ode = ACE_ODE(
            hidden_dim = hidden_dim,
            width = width,
            depth = depth,
            key = key
        ) 

           
    def __call__(self, t_start, t_end, y0):  #y0 is a tuple containing a hidden state and its attention (h, a), the attention must be flat
        
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.ace_ode),
            diffrax.Tsit5(),
            t0 = t_start,
            t1 = t_end,
            dt0 = None,     #handled by PID controller
            y0 = y0,
            stepsize_controller = diffrax.PIDController(rtol = 1e-3, atol = 1e-5),
            max_steps = 10000,
            saveat= diffrax.SaveAt(t1 = True)    #t1 = True later
        )
        
        return solution.ys #solution is a tuple of trajectories (h_traj, a_traj_flat)
    


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
    
    
class ACE_ODE_RNN(eqx.Module):
    """ ODE-RNN takes Data points and their timestamps {(xi , ti )}i=1..N then computes the algorithm 
    on the data points and returns an output array: {oi }i=1..N ; hN """
    
    static_data_injector: eqx.nn.Linear
    ode_solver: ACE_Solver
    rnn_cell: RNNCell
    att_rnn_cell: RNNCell
    output_nn: OutputNN
    hidden_dim: int
    
    def __init__(self, input_dim, output_dim, hidden_dim, static_feat, solver_width, output_nn_width,    #input_dim is not a great name, maybe obs_dim
                 solver_depth, output_nn_depth, *, key):
        ode_key, injector_key, cell_key, att_cell_key, out_key = random.split(key, 5)
        
        self.static_data_injector = eqx.nn.Linear(hidden_dim + static_feat, hidden_dim, key = injector_key) #injects the static features into the hidden state
        self.ode_solver = ACE_Solver(hidden_dim, solver_width, solver_depth, key = ode_key)
        self.rnn_cell = RNNCell(input_dim, hidden_dim, key = cell_key) #what if instead of an rnn cell i have 2, or a combined rnn cell for both attention and hidden state?
        self.att_rnn_cell = RNNCell(input_dim= hidden_dim, hidden_dim = hidden_dim**2, key = att_cell_key)   #this one is also trained separatley with attention loss
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
        
        a0_flat = self.gen_initial_attention(h0_batch)
        
        y_pred, h_pred = jax.vmap(self._call_single, in_axes = (0, 0, 0, None))(ts_batch, X_batch, h0_batch, a0_flat)
        return y_pred
        
    def _call_single(self, ts, observations, h0, a0_flat):   #time_stamps shape: (N,)    data_points shape: (N, features)
        
        #because we have h0 we can start solving from h0 to h1
        ts_now = ts[1:]
        ts_prev = ts[:-1]  #so that we can acces both ti and ti+1
        rnn_inputs = (ts_prev, ts_now, observations[1:]) #because we already used the first observation
        carry0 = (h0, a0_flat)
        
        #@eqx.filter_jit
        def step(carry, inputs): #one step of recurrence
            h_prev, a_prev = carry
            ti_prev, ti_now, xi = inputs
            #is this really a1 or do we also have to recalculate a1 given the new input?
            h_prime, a_prime = self.ode_solver(ti_prev, ti_now, (h_prev, a_prev))   #here hstate is hi-1
            h_prime, a_prime = h_prime[-1], a_prime[-1] #exctract vectors from the result

            h_new = self.rnn_cell(h_prime, xi)    #here we comput hi
            #a_new are raw attention values, softmax applied inside of ace_ode
            a_new = self.att_rnn_cell(a_prime, h_new)   #we use the new hidden state alongside a_prime to create a_new
            return (h_new, a_new), h_new    #carry, output (we don't need the attention)
        
        
        #hidden_history = (N, hidden_dim)
        carry_final, carry_history = jax.lax.scan(f = step, init = carry0, xs = rnn_inputs)
            
        #outputs = jax.vmap(self.output, in_axes=0)(hidden_history) #we compute the ourputs from the hidden states
        last_output = self.output_nn(carry_final[0])    #we are only interested in the last alpha value, so that one we return
        return last_output, carry_final[0]
    

    #FUNCTIONS FOR ATTENTION GENERATION
    def gen_initial_attention(self, h_batch):
        """for a batch of hidden vectors (Batch, Features)"""
        
        #correlation between latent dimension with shape (Features, Features)
        att_logits = jnp.corrcoef(h_batch.T) 
        
        #make sure there are no NaN values (when variance is 0)
        att_logits = jnp.nan_to_num(att_logits, nan= 0.0)
        
        #raw correlation values, ODE solver internally applies softmax to att before combining with h
        return att_logits.reshape(-1)

#Normalize att was never needed, because it gets normalized inside the ace_ode
