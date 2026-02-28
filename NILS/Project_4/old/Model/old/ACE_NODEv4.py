#The model doesn't work yet...

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

"""ARCHITECTURE:

x -> Feature Extractor -> h(0) -> Main Neural ODE -> h(0) -> Classifier -> Prediction
                            |       | | | | | | 
Initial Attention Generator a(0) ->  Attentive 
                                     Neural ODE
"""

class OrdinaryDE(eqx.Module):
    """Ordinary Differential Equation"""
    output_scale: jax.Array
    mlp: eqx.nn.MLP
    
    def __init__(self, input_dim, output_dim, layer_width, nn_depth, *, key):
        self.output_scale = jnp.array(1.0)
        self.mlp = eqx.nn.MLP(
            in_size = input_dim + 1,       # [h'(t), t]
            out_size = output_dim,
            width_size = layer_width,
            depth = nn_depth,
            activation = jax.nn.relu,
            final_activation = jax.nn.tanh,
            key = key
        )

    def __call__(self, t, y, args):     #changed
        # robustes Zusammenfügen von y und t (t kann Skalar oder Array sein)
        t_arr = jnp.asarray(t)
        # Falls t ein Skalar ist -> (,) -> mache (1,)
        if t_arr.ndim == 0:
            t_arr = t_arr[None]
        # Falls t ein 1-D array mit Länge > 1 ist (z.B. batched time), 
        # wähle das erste Element oder reduziere auf Skalar je nach gewünschtem Verhalten:
        elif t_arr.ndim == 1 and t_arr.size > 1:
            # Option A: nur erstes Element verwenden (häufig korrekt für ODE vf)
            t_arr = t_arr[:1]
            # Option B: falls du stattdessen eine Zusammenfassung brauchst, z.B. mean:
            # t_arr = jnp.array([t_arr.mean()])

        # y sicher in 1-D bringen
        y_vec = jnp.ravel(y)

        # jetzt zusammenfügen
        input_vector = jnp.concatenate([y_vec, t_arr], axis=0)

        return self.output_scale * self.mlp(input_vector)


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
        self.g_ode = OrdinaryDE(        #takes [h'(t), t] -> [da(t)/dt]
            input_dim = hidden_dim,
            output_dim = hidden_dim**2,
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
        g_dot = self.g_ode(t, h_prime, args = None)
        
        return (h_dot, g_dot) # returns (dh(t)/dt, da(t)/dt)
    



class ACE_NODE(eqx.Module):
    
    ace_ode: ACE_ODE
    
    def __init__(self, hidden_dim, layer_width, depth, *, key):
        ace_key, fe_key, cl_key = random.split(key, 3)
        
        self.ace_ode = ACE_ODE(
            hidden_dim = hidden_dim,
            f_width = layer_width,
            g_width = layer_width * 2,
            f_depth = depth,
            g_depth = depth + 1,
            key = ace_key
        ) 

           
    def __call__(self, ts, h0, a0_flat): 
        ha0 = (h0, a0_flat)
        
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.ace_ode),
            diffrax.Tsit5(),
            t0 = ts[0],
            t1 = ts[-1],
            dt0 = (ts[1] - ts[0]) * 0.1,
            y0 = ha0,
            stepsize_controller = diffrax.PIDController(rtol = 1e-5, atol = 1e-7),
            saveat= diffrax.SaveAt(ts = ts)    #t1 = True later
        )

        #extract hidden state from solution from solution
        h_traj, a_traj_flat = solution.ys
        
        return h_traj #solution is a tuple of trajectories (h_traj, a_traj_flat) we return the h_traj



#-----------------
#TRAINING SECTION
#-----------------

@eqx.filter_value_and_grad
def grad_loss_h(model_train, model_static, X, y, a0_flat):  #this loss function is to train Theta_f and Theta_other
    
    model = eqx.combine(model_train, model_static)
    y_pred = model(X, y[0], a0_flat)
    return jnp.mean((y - y_pred)**2)#mean squared error

@eqx.filter_value_and_grad
def grad_loss_a(model_train, model_static, X, y, a0_flat, l2_reg = 1e-4):  #This loss function is to train Theta_g and Theta_other
    
    model = eqx.combine(model_train, model_static)
    y_pred = model(X, y[0], a0_flat)
    loss = jnp.mean((y - y_pred)**2)
    
    def weights_only(leaf):
        return isinstance(leaf, jax.Array) and leaf.ndim == 2
    
    model_weights = eqx.filter(eqx.filter(model_train, weights_only), eqx.is_inexact_array) # we take the weights out of the mlp!
    l2_loss = l2_reg * sum(jnp.sum(w**2) for w in jtu.tree_leaves(model_weights)) 
    
    return loss + l2_loss


@eqx.filter_jit
def train_step_partitioned(X, y, a0_flat, model, filter_spec, opt_state, optimizer, loss_fn):
    
    #partition model
    model_train, model_static = eqx.partition(model, filter_spec)
    #calculating loss and gradients
    loss, grads_train = loss_fn(model_train, model_static, X, y, a0_flat) #the grads are only taken with respect to the first positional argument!!
    
    #updating trainable part of the model
    updates, opt_state = optimizer.update(grads_train, opt_state)  
    
    model_train = eqx.apply_updates(model_train, updates)
    #recombining the model
    model = eqx.combine(model_train, model_static)
    
    return loss, model, opt_state


def data_loader():
    pass


def get_params(model):
    """
    Extrahiert alle inexact (float) Array-Blätter aus dem Equinox-Modell.
    Rückgabe: Pytree mit denselben Strukturen wie die Array-Blätter.
    """
    # eqx.filter(model, eqx.is_inexact_array) gibt ein Pytree mit nur den Float-Arrays
    params = eqx.filter(model, eqx.is_inexact_array)
    # Optional: konvertieren zu jnp.asarray (sicherstellen, dass alles JAX-Arrays sind)
    params = jax.tree.map(lambda x: jnp.asarray(x), params)
    return params


def set_params(model, params):
    """
    Setzt die inexact Array-Blätter des Modells auf die Werte in `params`.
    `params` muss dieselbe Pytree-Struktur wie das Ergebnis von get_params(model) haben.
    Rückgabe: neues Modell mit ersetzten Parametern.
    """
    # Filter-Funktion, die das Sub-Pytree auswählt, das ersetzt werden soll
    filter_fn = lambda m: eqx.filter(m, eqx.is_inexact_array)
    # Ersetze das gefilterte Sub-Pytree durch `params`
    new_model = eqx.tree_at(filter_fn, model, replace=params)
    return new_model


def training_loop(X_train, y_train, a0_flat, model, epochs, lr, *, key, plot_loss = True):
    
    #splitting model into its component
    #model_g_mask = eqx.tree_at()
    #model_f_other_mask = eqx.tree_at()
        
    filter_spec_g = eqx.tree_at(
        lambda m: (m.ace_ode.g_ode, m.ace_ode.f_ode),
        jtu.tree_map(lambda _: False, model),
        replace = (True, False))
    
    filter_spec_f_other = eqx.tree_at(
        lambda m: (m.ace_ode.g_ode, m.ace_ode.f_ode),
        jtu.tree_map(lambda _: True, model),
        replace = (False, True))
    
    optimizer_f = optax.adam(lr)
    optimizer_g = optax.adam(lr)
    
    opt_state_f = optimizer_f.init(eqx.filter(eqx.filter(model, filter_spec_f_other),eqx.is_inexact_array))
    opt_state_g = optimizer_g.init(eqx.filter(eqx.filter(model, filter_spec_g), eqx.is_inexact_array))
    
    loss_history = []
    for epoch in range(epochs):
        
        loss, model, opt_state_f = train_step_partitioned(
            X_train, y_train, a0_flat, 
            model, filter_spec_f_other, opt_state_f, optimizer_f,
            grad_loss_h)
        
        _, model, opt_state_g = train_step_partitioned(
            X_train, y_train, a0_flat, 
            model, filter_spec_g, opt_state_g, optimizer_g,
            grad_loss_a)
        
        loss_history.append(loss) #I think this is the loss that is important to plot right? not the one from the g trainstep
        
        if epoch % 10 == 0:
            print(f"Epoch: {epoch}, loss: {loss}")
     
    if plot_loss:
        plt.plot(loss_history)
        plt.yscale("log")
        plt.show()
    
    return model

"""   
def initialAttentionGenerator(self, h0):    #let us instead pass a data_specific attention to our model
        #compute correlation matrix of h(0)
        # for now make a random matrix
        a0 = jnp.outer(h0, h0)
        a0_flat = a0.reshape(-1)
        return a0_flat
"""