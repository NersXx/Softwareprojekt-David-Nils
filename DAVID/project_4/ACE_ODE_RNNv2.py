#Group: David, Nils

import sys
import time

import jax

import jax.numpy as jnp
from jax import random
import jax.tree_util as jtu
import equinox as eqx
import diffrax
import optax

import matplotlib.pyplot as plt
import numpy as np



class OrdinaryDE(eqx.Module):
    """ODE to model the dynamics of the hidden state"""
    
    output_scale: jax.Array            #still don't know what this is supposed to do
    mlp: eqx.nn.MLP      #input is a jax array with shape (input_size,)     output is a jax array with shape (output_size,)
    
    def __init__(self, input_dim, output_dim, layer_width, nn_depth, *, key):
        self.output_scale = jnp.array(1.0)
        self.mlp = eqx.nn.MLP(
            in_size = 1 + input_dim,               #a vector of shape [h'(t), t]    where h is the hidden state vector h(t)
            out_size = output_dim,                 #a vector of the approximated dh(t)/dt value
            width_size = layer_width,
            depth = nn_depth,
            activation = jax.nn.silu,
            final_activation = jax.nn.identity,
            key = key  
        )
        
    
    def __call__(self, t, h, args):     #t.shape = () and h.shape = (features,)    
        input_vector = jnp.concatenate([h, jnp.array([t])], axis = 0)     #our ODE is dependent on vector [h'(t), t] (on time and hidden state h'(t))
        return self.output_scale * jnp.tanh(self.mlp(input_vector)) 



class ACE_ODE(eqx.Module):
    
    f_ode: OrdinaryDE       #hidden representation dynamics 
    g_ode: OrdinaryDE       #attention dynamics
    hidden_dim: int
    
    def __init__(self, hidden_dim, f_width, g_width, f_depth, g_depth, *, key ):
        f_key, g_key = random.split(key)
        
        self.f_ode = OrdinaryDE(        #takes [h'(t), t] -> [dh(t)/dt]
            input_dim = hidden_dim,
            output_dim = hidden_dim,
            layer_width = f_width,
            nn_depth = f_depth,
            key = f_key
        )   
        self.g_ode = OrdinaryDE(        #takes [a'(t), t] -> [da(t)/dt]
            input_dim= hidden_dim,
            output_dim= hidden_dim**2,
            layer_width = g_width,
            nn_depth = g_depth,
            key = g_key
        )
        
        self.hidden_dim = hidden_dim

    
    def __call__(self, t, ha, args): #h shape: (dim,) a shape: (dim*dim,)
        
        #first separate the joint state ha = [h(t), a(t)] into [h(t)] and [a(t)] then turn the flat a into the matrix of attention 
        h_state, a_matrix = jnp.array(ha[0]), jnp.array(ha[1]).reshape(self.hidden_dim, self.hidden_dim)
        
        #second we compute h'(t) by doing 𝒉(𝑡) 𝜎(𝒂(𝑡))⊺
        h_prime = h_state @ jax.nn.softmax(a_matrix, axis = -1).T 
        
        #we compute h_dot using f'(h'(t),t,theta_f)
        h_dot = self.f_ode(t, h_prime, args = None)
        
        #we compute a_dot using g'(h'(t),t,theta_f)
        a_dot = self.g_ode(t, h_prime, args = None)
        
        return (h_dot, a_dot) # returns (dh(t)/dt, da(t)/dt)
    

class ACE_ODESolve(eqx.Module):
    
    ace_ode: ACE_ODE
    
    def __init__(self, hidden_dim, layer_width, depth, *, key):
        
        self.ace_ode = ACE_ODE(
            hidden_dim = hidden_dim,
            f_width = layer_width,
            g_width = layer_width,
            f_depth = depth,
            g_depth = depth,
            key = key
        ) 

           
    def __call__(self, ts, h0, a0_flat):  #ts = [ti-1, ti] (has shape (N,))   h_prev = hi-1
        ha0 = (h0, a0_flat)
        
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.ace_ode),
            diffrax.Tsit5(),
            t0 = ts[0],
            t1 = ts[-1],
            dt0 = (ts[1] - ts[0]) * 0.1,
            y0 = ha0,
            stepsize_controller = diffrax.PIDController(rtol = 1e-3, atol = 1e-5),
            max_steps = 10,
            saveat= diffrax.SaveAt(t1 = True)    #t1 = True later
        )

        #extract hidden state from solution from solution
        h_traj, a_traj_flat = solution.ys
        
        return h_traj[0] #solution is a tuple of trajectories (h_traj, a_traj_flat) we return the h_traj

    def gen_initial_attention(self, h):
        h = h / (jnp.linalg.norm(h) + 1e-8)
        #maybe later add some trainable linear transformation?
        return jnp.outer(h, h).reshape(-1)  #compute outer matrix and flatten
    


class RNNCell(eqx.Module):
    """RNNCell that computes hi from h'i and our current observation xi hidden state is initialized to 0"""
    
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
    
    ode_solver: ACE_ODESolve
    rnn_cell: RNNCell
    output_nn: OutputNN
    hidden_dim: int
    
    def __init__(self, input_dim, output_dim, hidden_dim, solver_width, output_nn_width,
                 solver_depth, output_nn_depth, *, key):
        ode_key, cell_key, out_key = random.split(key, 3)
        self.ode_solver = ACE_ODESolve(hidden_dim, solver_width, solver_depth, key = ode_key)
        self.rnn_cell = RNNCell(input_dim, hidden_dim, key = cell_key) #what if instead of an rnn cell i have 2, or a combined rnn cell for both attention and hidden state?
        self.output_nn = OutputNN(hidden_dim, output_dim, output_nn_width, output_nn_depth, key = out_key)
        self.hidden_dim = hidden_dim
    
    def __call__(self, ts, observations):   #time_stamps shape: (N,)    data_points shape: (N, features)

        h0 = self.rnn_cell(jnp.zeros(shape = (self.hidden_dim,)), observations[0]) #rnncell(h'(0), x0)
        a0_flat = self.ode_solver.gen_initial_attention(h0)
        
        #then create initial attention using the new h0!
        
        ts_now = ts[1:]
        ts_prev = ts[:-1] 
        rnn_inputs = (ts_prev, ts_now, observations[1:]) #because we already used the first observation
        carry0 = (h0, a0_flat)
        
        #@eqx.filter_jit
        def step(carry, inputs): #one step of recurrence
            h_prev, a_prev = carry
            ti_prev, ti_now, xi = inputs
            #is this really a1 or do we also have to recalculate a1 given the new input?
            h_prime = self.ode_solver(jnp.array([ti_prev, ti_now]), h_prev, a_prev)   #here hstate is hi-1
            
            h_new = self.rnn_cell(h_prime, xi)    #here we comput hi
            a_new = self.ode_solver.gen_initial_attention(h_new)
            return (h_new, a_new), h_new    #carry, output (we don't need the attention)
        
        
        #hidden_history = (N, hidden_dim)
        carry_final, carry_history = jax.lax.scan(f = step, init = carry0, xs = rnn_inputs)
            
        #outputs = jax.vmap(self.output, in_axes=0)(hidden_history) #we compute the ourputs from the hidden states
        last_output = self.output_nn(carry_final[0])    #we are only interested in the last alpha value, so that one we return
        
        return last_output, carry_final[0]
    
    def batched_call(self, ts_batch, X_batch):
        """expects shapes ts_batch:(Batch, N) and X_batch:(Batch, N, features)"""
        
        y_pred, h_pred = jax.vmap(self.__call__, in_axes = (0, 0))(ts_batch, X_batch)
        return y_pred
    
    

#---------------------------------
# FUNCTIONS FOR TRAINING THE MODEL
#---------------------------------

#The loss function (and gradients of the loss function) 
@eqx.filter_value_and_grad
def grad_loss_h(model_train, model_static, X, y, ts):     #with our spiral data -> y (batch, 1) and X are our batches of [x,y] trajectories with shape (batch, 100, 2)
    
    #we recombine the model
    model = eqx.combine(model_train, model_static)
    #Here we are gonna vectorize to be able to apply the model to the batches and calculate some type of loss across the batches!
    y_pred = model.batched_call(ts, X) 
    #for now let's use L2 loss but let's try different things later
    return jnp.mean((y - y_pred)**2)       # y_train.shape = (batch, 1)

@eqx.filter_value_and_grad
def grad_loss_a(model_train, model_static, X, y, ts, l2_reg = 1e-4):
    model = eqx.combine(model_train, model_static)
    y_pred = model.batched_call(ts, X)
    loss = jnp.mean((y - y_pred)**2)
    
    def weights_only(leaf):
        return isinstance(leaf, jax.Array) and leaf.ndim == 2   #because we want to apply the penalty to weights, not to the biases
    
    model_weights = eqx.filter(eqx.filter(model_train, weights_only), eqx.is_inexact_array)
    leaves = jtu.tree_leaves(model_weights)
    squared_sums = [jnp.sum(w**2) for w in leaves]
    l2_loss = l2_reg * jnp.sum(jnp.stack(squared_sums))
        
    return loss + l2_loss


@eqx.filter_jit
def train_step_partitioned(X, y, ts, model, filter_spec, opt_state, optimizer, loss_fn):
    """the train step partitioned to train f and g"""
    
    #partition model
    model_train, model_static = eqx.partition(model, filter_spec)
    
    #calculating loss and gradients
    loss, grads_train = loss_fn(model_train, model_static, X, y, ts)
    #just in case
    grads_train = eqx.filter(grads_train, eqx.is_inexact_array)
    #updating trainable part of the model
    updates, opt_state = optimizer.update(grads_train, opt_state)
    model_train = eqx.apply_updates(model_train, updates)
    #recombining the model
    model = eqx.combine(model_train, model_static)
    
    return loss, model, opt_state
    

def data_loader(X, y, ts, batch_size, *, key):
    """makes batches for the data"""
    data_size = X.shape[0]   #check what the size of Batch is for (Batch, N, features)
    assert X.shape[0] == y.shape[0]
    indices = jnp.arange(data_size)
    
    perm = random.permutation(key, indices)
    X_shuffled, y_shuffled, ts_shuffled = X[perm], y[perm], ts[perm]
    
    for i in range(0, data_size, batch_size):
        start, end = i, i+batch_size
        yield X_shuffled[start: end], y_shuffled[start: end], ts_shuffled[start: end]
    
    
#the training loop!
def training_loop(X_train, y_train, ts_train, model, epochs, lr, batch_size, *, key, plot_loss = True): # introduce autostop argument (desired loss)
    """training loop, Expects shapes (Batch, N, features), (Batch, 1), (Batch, N), """
    batch_size = batch_size

    #splitting model for training
    
    filter_spec_g = eqx.tree_at(
        lambda m: (m.ode_solver.ace_ode.g_ode, m.ode_solver.ace_ode.f_ode),
        jtu.tree_map(lambda _: False, model),
        replace = (True, False)
    )
    
    filter_spec_f_other = eqx.tree_at(
        lambda m: (m.ode_solver.ace_ode.g_ode, m.ode_solver.ace_ode.f_ode),
        jtu.tree_map(lambda _: True, model),
        replace = (False, True)
    )
    
    #Start Optimizers
    #optimizer_f = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(lr, weight_decay=1e-4))
    optimizer_f = optax.adam(lr)
    opt_state_f = optimizer_f.init(eqx.filter(eqx.filter(model, filter_spec_f_other),eqx.is_inexact_array))
    
    optimizer_g = optax.adam(lr)
    opt_state_g = optimizer_g.init(eqx.filter(eqx.filter(model, filter_spec_g), eqx.is_inexact_array))
    
    
    loss_history = []
    print("Training...\n")
    for epoch in range(epochs):
        key, subkey = random.split(key)
        
        #Maybe use gradient accumulation? optax.MultiStep() ?
        for X_batch, y_batch, ts_batch in data_loader(X_train, y_train, ts_train, batch_size, key = subkey):
            loss, model, opt_state_f = train_step_partitioned(X_batch, y_batch, ts_batch, 
                                                            model, filter_spec_f_other,
                                                            opt_state_f, optimizer_f, grad_loss_h)
            
            _, model, opt_state_g = train_step_partitioned(X_batch, y_batch, ts_batch, 
                                                        model, filter_spec_g,
                                                        opt_state_g, optimizer_g, grad_loss_a)
            
            print("##", end="",flush=True)
            loss_history.append(loss)
        
        print("\n")    
        if epoch % 1 == 0:
            print(f"Epoch: {epoch}, loss: {loss}")  #I was today years old when i found out  python does not create it's own scope for for loops wth
    
        if epoch > (epochs * 50) // 100:    #after we went through 50% of our training time
            if loss < 3e-3: break      #lets make it so this is more customizable
            
    if plot_loss:
        print(len(loss_history))
        plt.plot(loss_history)
        plt.yscale("log")
        plt.show()
        
    return model



#-----------------
# MAIN ENTRY POINT
#-----------------

def main() -> int:
    return 0



if __name__ == "__main__":
    sys.exit(main())