#Group: Nils, David


import sys
import time

import jax
#jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random
import equinox as eqx
import diffrax
import optax

import matplotlib.pyplot as plt
import numpy as np



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
            final_activation = jax.nn.tanh,
            dtype = jnp.float32,
            key = key  
        )
        
    
    def __call__(self, t, h, args):     #t.shape = () and h.shape = (features,)    
        input_vector = jnp.concatenate([jnp.array([t]),h], axis = 0)     #our ODE is dependent on vector [t,h] (on time and hidden state h)
        return self.output_scale * jnp.tanh(self.mlp(input_vector)) 
    

class ODESolve(eqx.Module): 
    """ODESolve to get our hidden state h'i representing some internal memory"""
    
    ode: OrdinaryDE
    
    def __init__(self, hidden_dim, layer_width, nn_depth, *, key):
        self.ode = OrdinaryDE(hidden_dim, layer_width, nn_depth, key = key)
    
    def __call__(self, time_steps, h_initial): #ts = [ti-1, ti] (has shape (N,))   h_prev = hi-1
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.ode),
            diffrax.Tsit5(),
            t0 = time_steps[0],
            t1 = time_steps[-1],
            dt0 = None,
            y0 = h_initial,
            stepsize_controller = diffrax.PIDController(rtol = 1e-3, atol = 1e-5),
            max_steps = 10_000,
            saveat= diffrax.SaveAt(t1 = True)
        )

        return solution.ys[-1]  #we return the hidden state h'i, but we have to get it out of the batch (usually the whole trajectory of every h at every t of ts is saved in a batch)



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
        return self.mlp(hi)
    


    
class ODE_RNN(eqx.Module):
    """ ODE-RNN takes Data points and their timestamps {(xi , ti )}i=1..N then computes the algorithm 
    on the data points and returns an output array: {oi }i=1..N ; hN """
    
    ode_solver: ODESolve
    rnn_cell: RNNCell
    output_nn: OutputNN
    hidden_dim: int
    
    def __init__(self, input_dim, output_dim, hidden_dim, solver_width, output_nn_width,
                 solver_depth, output_nn_depth, *, key):
        ode_key, cell_key, out_key = random.split(key, 3)
        self.ode_solver = ODESolve(hidden_dim, solver_width, solver_depth, key = ode_key)
        self.rnn_cell = RNNCell(input_dim, hidden_dim, key = cell_key)
        self.output_nn = OutputNN(hidden_dim, output_dim, output_nn_width, output_nn_depth, key = out_key)
        self.hidden_dim = hidden_dim
    
    def __call__(self, ts, observations):   #time_stamps shape: (N,)    data_points shape: (N, features)
        N = ts.shape[0]
        h0 = jnp.zeros(shape = (self.hidden_dim,), dtype = jnp.float32)
        ts_now = ts
        ts_prev = jnp.concatenate([ts[0:1],ts[:-1]])  #we create a list where t0 is duplicated to shift everything to the right
        rnn_inputs = (ts_prev, ts_now, observations)
        
        def step(h, inputs): #one step of recurrence
            ti_prev, ti_now, xi = inputs
            
            #at i=0 we have t_prev == t_now
            def integrate(_):
                return self.ode_solver(jnp.array([ti_prev, ti_now]), h)   #here hstate is hi-1
            
            h_prime = jax.lax.cond(jnp.equal(ti_prev, ti_now),   #only true for the first element
                lambda _: h,                      #if condition is true
                integrate,                              #else (so every other time)
                operand=None)
            
            h_state_new = self.rnn_cell(h_prime, xi)    #here we comput hi
            return h_state_new, h_state_new     #carry, output
        
        indices = jnp.arange(N)
        
        
        #hidden_history = (N, hidden_dim)
        h_state_final, hidden_history = jax.lax.scan(f = step, init = h0, xs = rnn_inputs)
            
        #outputs = jax.vmap(self.output, in_axes=0)(hidden_history) #we compute the ourputs from the hidden states
        last_output = self.output_nn(h_state_final)    #we are only interested in the last alpha value, so that one we return
        return last_output, h_state_final
    
    def batched_call(self, ts_batch, X_batch):
        """expects shapes ts_batch:(Batch, N) and X_batch:(Batch, N, features)"""
        y_pred, h_pred = jax.vmap(self.__call__, in_axes = (0, 0))(ts_batch, X_batch)
        return y_pred, h_pred
        

#---------------------------------
# FUNCTION FOR TRAINING THE MODEL
#---------------------------------


#The loss function (and gradients of the loss function) 
@eqx.filter_value_and_grad
def grad_loss(model, X, y, ts):     #with our spiral data -> y (batch, 1) and X are our batches of [x,y] trajectories with shape (batch, 100, 2)
    
    #Here we are gonna vectorize to be able to apply the model to the batches and calculate some type of loss across the batches!
    y_pred, _ = model.batched_call(ts, X) 
    #for now let's use L2 loss but let's try different things later
    return jnp.mean((y - y_pred)**2)       # y_train.shape = (batch, 1)


@eqx.filter_jit
def train_step(X, y, ts, model, opt_state, optimizer):
    """the train step with the optimization"""
    loss, grads = grad_loss(model, X, y, ts)
    grads = eqx.filter(grads, eqx.is_inexact_array)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
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

    #Start Optimizer
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(lr, weight_decay=1e-4))
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    
    loss_history = []
    print("Training...\n")
    for epoch in range(epochs):
        key, subkey = random.split(key)
        
        #Maybe use gradient accumulation? optax.MultiStep() ?
        for X_batch, y_batch, ts_batch in data_loader(X_train, y_train, ts_train, batch_size, key = subkey):
            loss, model, opt_state = train_step(X_batch, y_batch, ts_batch, model, opt_state, optimizer)
            loss_history.append(loss)
            
        if epoch % 1 == 0:
            print(f"Epoch: {epoch}, loss: {loss}")  #I was today years old when i found out  python does not create it's own scope for for loops wth
    
        if epoch > (epochs * 50) // 100:    #after we went through 50% of our training time
            if loss < 1e-4: break      #lets make it so this is more customizable
            
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