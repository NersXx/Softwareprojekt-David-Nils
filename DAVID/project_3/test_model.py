import sys
import time

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import matplotlib.pyplot as plt

import ACE_NODEv3 as ace_node
import norm


def main() -> int:
    key = random.key(int(time.time()))
    model_key, train_key = random.split(key)
    
    #Lynx and Hare Dataset
    lh_data = jnp.array(np.load("LH_data.npy"))
    time_steps = lh_data[:, 0:1]
    populations = lh_data[:, 1:3]
    print(f"Years: {time_steps.shape}, Population {populations.shape}")
    
    #scaling
    x_normalizer = norm.MinShiftNorm()
    y_normalizer = norm.LogNorm()
    x_normalizer.init(time_steps, axis = 0)
    y_normalizer.init(populations, axis = 0)
    
    time_steps_norm = x_normalizer(time_steps)
    populations_norm = y_normalizer(populations)
    
    #creating model
    model = ace_node.ACE_NODE(2, 32, 2, key = model_key)
    
    #create initial attention matrix and train
    initial_attention = generate_initial_attention(populations_norm).reshape(-1)
    
    model = ace_node.training_loop(
        time_steps_norm.squeeze(), populations_norm, initial_attention,
        model, 2000, 5e-3, key = train_key)
    
    
    #ploting the populations
    plt.plot(time_steps, populations[:,0:1], c="dodgerblue", label = "Hares")
    plt.plot(time_steps, populations[:,1:2], c="green", label= "Lynx")
            
    y0 = jnp.array(populations_norm[0])
    t_pred = x_normalizer(jnp.concatenate([time_steps.squeeze(), jnp.array([time_steps.max() + i for i in range(1,20)])], axis = 0))
    hl_predict = y_normalizer(model(t_pred, y0, initial_attention), denormalize=True) 
    
    hare_predict = hl_predict[:, 0:1]
    lynx_predict = hl_predict[:, 1:2]
    
    plt.plot(x_normalizer(t_pred, denormalize=True), hare_predict, c="red", label="Hares fit")
    plt.plot(x_normalizer(t_pred, denormalize=True), lynx_predict, c="purple", label="Lynx fit")
    plt.legend()
            
    plt.show()
    
    return 0


def generate_initial_attention(data):
    
    correlation_matrix = jnp.corrcoef(data.T)
    correlation_matrix  =jnp.array([[0.6,0.4],[0.4,0.6]])
    return correlation_matrix
    



if __name__ == "__main__":
    sys.exit(main())