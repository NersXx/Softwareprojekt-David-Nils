# For the shape: The last number always describes the size of the innermost dimension (often columns or features)
# the earlier numbers describe groupings of those: (rows, columns) or (batches, features)
# so (2,3) would be [[x,y,z][a,b,c]]
# general rule W.shape == (input_dim, output_dim) for example in the case of the digit_recog_nn
# we have for the connection between the input layer and the first hidden layer (784,16) or (input_dim, hidden_dim1)
# basically 784 rows that represent each pixel, and each pixel has 16 column, which represent each of it's connections to one of the 16 different neurons


import time
import diffrax
import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
import optax
import numpy as np


# import dataset 'spirals.npz' from local directory
data = np.load('spirals.npz')

#ODE
class Func(eqx.Module):
    """ODE"""
    
    #Attributes
    #lets network learn how big the derivatives should be
    out_scale: jax.Array    #learnt scalar to control magnitude of vector field (apparently diffrax expects dy/dt to not be too large or chaotic)
    mlp: eqx.nn.MLP         #Multilayer perceptron
    
    def __init__(self, input_size, output_size, width_size, depth, *, key):
        self.out_scale = jnp.array(1.0) #a scalar to 
        self.mlp = eqx.nn.MLP(
            in_size = input_size,   #in our case the vector [t,y] ]or [t,y1,y2]
            out_size = output_size,     #in our case dy/dt or [dy1/dt, dy2/dt]
            width_size = width_size,            #The dimension of the hidden layers
            depth = depth,                      #amount of hidden layers (including output dimension)
            activation = jnn.relu,             #let's test out different nonlinear activations
            final_activation = jnn.identity,
            key = key
            ) 
    
    #f(t, y) gives the instantaneous rate of change (slope) of y at time t
    def __call__(self, t, y, args):     #diffrax expect the function signature to look like this, where t current tyme and y current system state
        # Best practice is often to use `learnt_scalar * tanh(MLP(...))` for the
        # vector field.
        input_vec = jnp.concatenate([jnp.array([t]), y], axis = 0) #our vector [t, y]
        #we can wrap the derivative in tanh if we want to to avoid big gradients
        return self.out_scale * jnp.tanh(self.mlp(input_vec)) #bind values of mlp to [-1,1] (the output of the mlp is a value of dy/dt) -> prevents exploding gradients
        

#This is really two steps. Defining the ODE and then wrapping it in the numerical solver (which solves form y0 over ts to get y1)
class NeuralODE(eqx.Module):
    """Wrapped ODE Integration"""
    func: Func #our ODE function
    
    def __init__(self, input_size, output_size, width_size, depth, *, key):
        self.func = Func(input_size, output_size, width_size, depth, key= key)  #ODE
        
    #Numerical Integration from y0 over an array of timsteps ts
    def __call__(self, ts, y0):
        solution = diffrax.diffeqsolve( #to integrate the differential equation from t0 to t1
            diffrax.ODETerm(self.func),         #specifies our vector field
            diffrax.Tsit5(),                   #Our specific Differential Equation Solver, In this case 5th order explicit Runge--Kutta method with explicit step seizing, supports backpropagation
            t0 = ts[0],                         #start of integration region
            t1 = ts[-1],                        #end of integration regions
            dt0=ts[1] - ts[0],                  #step size for the first step, after that because we are using adaptive step size it might vary
            y0 = y0,                            #Initial y value
            saveat = diffrax.SaveAt(ts=ts),     #ts is Some array of times at which to save the output. (so we save the output at all points of the array in this case), this field usually defaults SaveAt() to just save t1
            stepsize_controller = diffrax.PIDController(rtol = 1e-4, atol = 1e-7)    #in this case we use an adaptive step size controller, but we can play around with the tolerances and the settings
        )
        return solution.ys          #the value of the solution at each of the times ys shape -> (len(ts), output_dim)

#TOY DATA SET
def toy_dataset(n_samples,* ,key):
    key, xkey, ynoisekey = random.split(key, 3)
    x_samples = random.uniform(xkey,(n_samples, 1), minval = 0, maxval = 2*jnp.pi)         #we have (batch_size, x_value)  
    x_samples = jnp.sort(x_samples, axis=0)                                                            #we sort it so that values are in ascending order
    y_samples = jnp.sin(x_samples) + random.normal(ynoisekey, (n_samples, 1)) * 0.3
    return x_samples, y_samples             #so we have shapes (N,) and (N,1)

"""
def dataloader():
    pass
"""

#GRADIENTS OF OUR LOSS FUNCTION AND SCALAR LOSS
@eqx.filter_value_and_grad  #computes the gradient tree and the scalar loss and ignores non trainable fileds?
def grad_loss(model, ti, yi):   #Model is our NODE, ti our integration paths and y1 our initial y values for the different batches (as we are vectorizing the function to be able to apply it more efficiently)
    """gradients of the loss function"""
    #we basically apply our model to our different "curves" or time series that we want to learn -> so we are computing many full paths in parallel
    # What shape are we makeing yi have?
    #y_pred = jax.vmap(model, in_axes = (None, 1)) (ti, jnp.array([yi[0]])) #vmap lets you apply the function over a batch of inputs in parallel! (aren't the inputs dependant on the output values of previous batches tho?)
    #try to understand this vectorization a lil bit better after, (also the axis thing)
    
    y_pred = model(ti, yi[0])
    return jnp.mean((yi - y_pred)**2)   #Another idea would be to use l_2 regularization and cross entropy loss instead


#y_pred = jax.vmap(model, in_axes = (None, 0)) (ti, yi[:, 0]) This line a bit crazy so:
#vectorizes a function "model" over one or more arguments (applies it to the different time series)
#in_axes tells jax which arguments get batched and which ones stay cobstant, in this case
#   -> None -> same ti is passed to every call of model
#   -> 0 ->     y[:, 0] -> gets batched over the 0th axis, so each element in that dimension is processed separately
# jax basically expands the clal to:
""" 
    y_pred = [model(ti, yi[0, 0]),
            model(ti, yi[1, 0]),
            model(ti, yi[2, 0]),
            ...]
            
    so internally the calls 
    
    model(ti, y0_1)
    model(ti, y0_2)
    model(ti, y0_3)
    
    y_pred.shape = (amt_statrting_values_yi, len(ti)) where len(ti) is the trajectory ??

"""


@eqx.filter_jit
def make_step(ti, yi, model, opt_state, optimizer):
    """train step function"""
    loss, grads = grad_loss(model, ti, yi)  #loss and gradients for our different time series
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state


#we need the optimizer and opt state to be available in the training loop, outside of make step

#expected t_train shape is (N,) and y_train shape is (N,1)
def training_loop(t_train, y_train, model, epochs, learning_rate, *, plot_loss = True):
    """"""
    #Start Optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array)) #what is the exact type we should be checking for?
   
    loss_history = []
    for epoch in range(epochs):
        loss, model, opt_state = make_step(t_train, y_train, model, opt_state, optimizer)
        loss_history.append(loss)
        if epoch % 50 == 0:
            print(f"Epoch: {epoch}, loss: {loss}")
    
    if plot_loss:
        plt.plot(loss_history)
        plt.yscale("log")
        plt.show()
    
    return model
    
    

def make_prediction():
    """Predicts y(t1) from y(t0) and model"""
    pass





#MAIN ENTRY POINT
def main():
    return 0






if __name__ == "__main__":
    sys.exit(main())





