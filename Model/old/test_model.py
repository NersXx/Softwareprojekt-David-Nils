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
    time_steps_norm = time_steps - time_steps.min() #better normalize between 0 and 1
    eps = 1e-8
    pop_log = jnp.log(populations + eps)
    mean = pop_log.mean(axis=0, keepdims = True)
    std = pop_log.std(axis = 0, keepdims = True)
    populations_norm = (pop_log - mean) / std
    
    #creating model
    model = ace_node.ACE_NODE(2, 32, 3, key = model_key)
    
    #create initial attention matrix and train
    initial_attention = generate_initial_attention(populations_norm).reshape(-1)
    
    model = ace_node.training_loop(
        time_steps_norm.squeeze(), populations_norm, initial_attention,
        model, 200, 1e-3, key = train_key)
    
    
    #ploting the populations
    plt.plot(time_steps, populations[:,0:1], c="dodgerblue", label = "Hares")
    plt.plot(time_steps, populations[:,1:2], c="green", label= "Lynx")
            
    y0 = jnp.array(populations_norm[0])
    t_pred = jnp.concatenate([time_steps_norm.squeeze(), jnp.array([time_steps_norm.max() + i for i in range(20)])], axis = 0)
    hl_predict = jnp.exp(((model(t_pred, y0, initial_attention) * std) + mean) - eps)   #this is cumbersome, so maybe already the model requires it ina flat state or it reshapes inside?
    
    hare_predict = hl_predict[:, 0:1]
    lynx_predict = hl_predict[:, 1:2]
    
    plt.plot(t_pred + time_steps.min(), hare_predict, c="red", label="Hares fit")
    plt.plot(t_pred + time_steps.min(), lynx_predict, c="purple", label="Lynx fit")
    plt.legend()
            
    plt.savefig("plot.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    return 0


def generate_initial_attention(data):
    
    correlation_matrix = jnp.corrcoef(data.T)
    return correlation_matrix
    



if __name__ == "__main__":
    sys.exit(main())