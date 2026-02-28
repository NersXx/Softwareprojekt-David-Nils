import sys
import time
import functools

import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx
import diffrax
import optax

import matplotlib.pyplot as plt
import numpy as np

if jax.device_count() == 1:
    print("you are running on one device, which will crash the code. make sure you have installed jax for gpu and have more than one gpu available")
    sys.exit()

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

    def __call__(self, t, y, args):
        input_vector = jnp.concatenate([y ,jnp.array([t])], axis = 0)
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
        
        h_state, a_matrix = jnp.array(ha[0]), jnp.array(ha[1]).reshape(self.hidden_dim, self.hidden_dim)
        
        h_prime = h_state @ jax.nn.softmax(a_matrix, axis = -1).T 
        
        h_dot = self.f_ode(t, h_prime, args = None)
        g_dot = self.g_ode(t, h_prime, args = None)
        
        return (h_dot, g_dot)
    


class ACE_NODE(eqx.Module):
    
    ace_ode: ACE_ODE
    
    def __init__(self, hidden_dim, layer_width, depth, *, key):
        self.ace_ode = ACE_ODE(
            hidden_dim = hidden_dim,
            f_width = layer_width,
            g_width = layer_width * 2,
            f_depth = depth,
            g_depth = depth + 1,
            key = key
        ) 
        
           
    def __call__(self, ts, h0): 
        a0_flat = self.initialAttentionGenerator(h0)
        ha0 = (h0, a0_flat)
        
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.ace_ode),
            diffrax.Tsit5(),
            t0 = ts[0],
            t1 = ts[-1],
            dt0 = (ts[1] - ts[0]) * 0.1,
            y0 = ha0,
            stepsize_controller = diffrax.PIDController(rtol = 1e-4, atol = 1e-7),
            saveat= diffrax.SaveAt(ts = ts)
        )

        h_traj, a_traj_flat = solution.ys
        
        return h_traj
    
    def initialAttentionGenerator(self, h0):
        a0 = jnp.outer(h0, h0)
        a0_flat = a0.reshape(-1)
        return a0_flat



@eqx.filter_value_and_grad 
def grad_loss(model, X, y):
    y_pred = model(X, y[0])
    return jnp.mean((y - y_pred)**2)  
    

# ---------pmap train_step ----------
@functools.partial(jax.pmap, axis_name="devices")
def train_step(params, opt_state, X, y, optimizer, model):
    # combine host model with device params
    model_with_params = eqx.combine(model, params)
    loss, grads = grad_loss(model_with_params, X, y)
    grads = eqx.filter(grads, eqx.is_inexact_array)
    grads = jax.tree_map(lambda g: jax.lax.pmean(g, "devices"), grads)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = eqx.apply_updates(params, updates)
    loss = jax.lax.pmean(loss, "devices")
    return loss, params, opt_state

# --------------------------------------


def data_loader():
    pass


def training_loop(X_train, y_train, model, epochs, learning_rate, *, plot_loss = True):
    import tree_util as _tu  # falls nicht vorhanden: use jax.tree_util
    def _shapes(tree):
        return jax.tree_map(lambda x: getattr(x, "shape", type(x)), tree)
    print(_shapes(params))

    n_devices = jax.local_device_count()
    assert X_train.shape[0] % n_devices == 0, "Batchgröße muss durch Anzahl Devices teilbar sein"

    def shard(x):
        return x.reshape(n_devices, -1, *x.shape[1:])

    X_shard = shard(X_train)
    y_shard = shard(y_train)

    optimizer = optax.adam(learning_rate)
    # init opt state on params only
    params = eqx.filter(model, eqx.is_inexact_array)
    opt_state = optimizer.init(params)

    # replicate params and opt_state
    params_repl = jax.device_put_replicated(params, jax.local_devices())
    opt_state_repl = jax.device_put_replicated(opt_state, jax.local_devices())

    loss_history = []
    for epoch in range(epochs):
        loss, params_repl, opt_state_repl = train_step(params_repl, opt_state_repl, X_shard, y_shard, optimizer, model)
        loss_val = float(loss[0])
        loss_history.append(loss_val)
        if epoch % 50 == 0:
            print(f"Epoch: {epoch}, loss {loss_val:.6f}")
         
    if plot_loss:
        plt.plot(loss_history)
        plt.yscale("log")
        plt.show()
    
    # pull params from device 0 and reinsert into model
    final_params = jax.device_get(params_repl)[0]
    final_model = eqx.combine(model, final_params)
    return final_model




def toy_dataset(n_samples,* ,key):
    key, xkey, ynoisekey = random.split(key, 3)
    x_samples = random.uniform(xkey,(n_samples, 1), minval = 0, maxval = 2*jnp.pi)  
    x_samples = jnp.sort(x_samples, axis=0)                                                            
    y_samples = jnp.sin(x_samples) + random.normal(ynoisekey, (n_samples, 1)) * 0.3
    return x_samples.reshape(-1), y_samples             


def main() -> int:
    key = random.key(int(time.time()))
    data_key, model_key, model_2key = random.split(key, 3)
    
    X, y = toy_dataset(20, key = data_key)
    
    model = ACE_NODE(1, 16, 3, key = model_key)
    
    model = training_loop(X, y, model, 500, 1e-3)
    
    plt.scatter(X, y)
    plt.scatter(X, model(X, y[0]))
    plt.show()
    
    lh_data = jnp.array(np.load("LH_data.npy"))
    time_steps = lh_data[:, 0:1]
    populations = lh_data[:, 1:3]
    print(f"Years: {time_steps.shape}, Population {populations.shape}")
    
    time_steps_norm = time_steps - time_steps.min()
    eps = 1e-8
    pop_log = jnp.log(populations + eps)
    mean = pop_log.mean(axis=0, keepdims = True)
    std = pop_log.std(axis = 0, keepdims = True)
    populations_norm = (pop_log - mean) / std
    
    model = ACE_NODE(2, 32, 3, key = model_2key)
    model = training_loop(time_steps_norm.squeeze(), populations_norm, model, 4000, 2e-3)
    
    plt.plot(time_steps, populations[:,0:1], c="dodgerblue", label = "Hares")
    plt.plot(time_steps, populations[:,1:2], c="green", label= "Lynx")
            
    y0 = jnp.array(populations_norm[0])
    t_pred = jnp.concatenate([time_steps_norm.squeeze(), jnp.array([time_steps_norm.max() + i for i in range(20)])], axis = 0)
    hl_predict = jnp.exp(((model(t_pred, y0) * std) + mean) - eps)
    
    hare_predict = hl_predict[:, 0:1]
    lynx_predict = hl_predict[:, 1:2]
    
    plt.plot(t_pred + time_steps.min(), hare_predict, c="red", label="Hares fit")
    plt.plot(t_pred + time_steps.min(), lynx_predict, c="purple", label="Lynx fit")
    plt
result = main()
print(result)
