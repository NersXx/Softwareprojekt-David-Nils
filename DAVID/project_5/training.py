import time
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import random

import equinox as eqx
import optax

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.preprocessing import OneHotEncoder


class SepsisDataset:
    """Pre-loads dataset into memory"""
    def __init__(self, file_paths, max_t = 72):
        self.data_size = file_paths.shape[0]   #check what the size of Batch is for (Batch, N, features)
        
        #load norm_stats
        norm_stats = np.load('data/norm_stats.npz')
        mean_x = jnp.array(norm_stats['mean_x'])
        std_x = jnp.array(norm_stats['std_x'])    
        mean_static = jnp.array(norm_stats['mean_static'])
        std_static = jnp.array(norm_stats['std_static'])
        encoder = OneHotEncoder(sparse_output=False)
        
        i = 0
        X_data, ts_data, Sd_data, y_data = [], [], [], []
        for path in file_paths:
            
            #Build X and ts from data
            patient_data = np.load(path)        #read the files

            x = jnp.array(patient_data['x'])
            mask = jnp.array(patient_data['mask'])
            t = jnp.array(patient_data['t'])
            Sd = jnp.array(patient_data['static'])
            y = jnp.array(patient_data['label'])
            
            #normalize x and Sd
            x = (x - mean_x)/ std_x
            x = x * mask    #just in case
            
            Sd_split = Sd.shape[0]//2
            Sd_vals = Sd[:Sd_split]
            Sd_flags = Sd[Sd_split:]
            
            Sd_vals = (Sd_vals - mean_static)/ std_static
            Sd = jnp.concatenate([Sd_vals, Sd_flags])
            
            assert x.shape[0] == mask.shape[0] == t.shape[0]    #make sure they all have the same length in time dim
            
            #truncate or pad
            if x.shape[0] > max_t: #truncate
                x = x[-max_t:]
                mask = mask[-max_t:]
                t = t[-max_t:]
                            
            elif x.shape[0] < max_t:   #pad, make sure that for x.shape[0] = Max_T it also works
                pad_len = max_t - x.shape[0]
                x = jnp.pad(x, ((0, pad_len),(0, 0)), mode='constant', constant_values = 0.0 )
                mask = jnp.pad(mask, ((0, pad_len),(0, 0)), mode='constant', constant_values = 0.0 ) #we pad with zeroes for not observed
                
                dt = t[1] - t[0] if t.shape[0] > 1 else 1.0
                t_pad = t[-1] + dt*jnp.arange(1, pad_len + 1)   #the padding at the end with dummy values of time
                t = jnp.concatenate([t,t_pad], axis= 0)
            #If the length of the time series happens to be Max_T then we continue
            
            #Now that all t have the same length we can normalize to [0,1]
            t = (t-t[0]) / (t[-1] - t[0] + 1e-8)
            
            #append to correct arary
            x_mask = jnp.concatenate([x, mask], axis = -1) #concatenating feature wise -> shape = (Max_T, Features*2)
            X_data.append(x_mask)
            ts_data.append(t)
            Sd_data.append(Sd)
            y_data.append(y)

            i += 1
            print(f"Data Loading into Main memory {((i)/self.data_size)*100:.3f}% done", end = "\r")
            
                
        print("\n")
        self.X_data = jnp.nan_to_num(jnp.stack(X_data), nan = 0.0)
        self.y_data = jnp.nan_to_num(jnp.stack(y_data), nan = 0.0)
        self.ts_data = jnp.nan_to_num(jnp.stack(ts_data), nan = 0.0)
        self.Sd_data = jnp.nan_to_num(jnp.stack(Sd_data), nan = 0.0)

        #One Hot encode label
        self.y_data = encoder.fit_transform(self.y_data)

    def permutation(self, perm):
        self.X_data = self.X_data[perm]
        self.y_data = self.y_data[perm]
        self.ts_data = self.ts_data[perm]
        self.Sd_data = self.Sd_data[perm]

    def get_batch(self, start, end):
        return self.X_data[start:end], self.y_data[start:end], self.ts_data[start:end], self.Sd_data[start:end]


#The loss function (and gradients of the loss function) 
@eqx.filter_value_and_grad
def grad_loss_h(model_train, model_static, X, y, ts, Sd):     #with our spiral data -> y (batch, 1) and X are our batches of [x,y] trajectories with shape (batch, 100, 2)
    """We'll have to find the appropriate regularization for this later
    For now we use cross entropy loss
    """
    
    #we recombine the model
    model = eqx.combine(model_train, model_static)
    
    #prediction of probabilities
    logits = model(ts, X, Sd) #model takes batches as a standard
    probs = jax.nn.softmax(logits, axis=-1) #turns unbounded numbers into probabilities adding to 1 across each row

    label_weights = jnp.array([1.0, 10.0])   # [neg, pos] --> positives weigh more on the loss
    
    cross_entropy_loss = -jnp.mean(jnp.sum(y*jnp.log(probs + 1e-8)*label_weights, axis = 1))
    
    reg = 0.0
    
    #we use cross entropy loss + some appropriate regularization term
    return cross_entropy_loss + reg       # y_train.shape = (batch, 1)


#<----------------also make a percentage bar for the epochs for sanity bruv

@eqx.filter_value_and_grad
def grad_loss_a(model_train, model_static, X, y, ts, Sd, l2_reg = 1e-4):
    model = eqx.combine(model_train, model_static)
    
    logits = model(ts, X, Sd) #model takes batches as a standard
    probs = jax.nn.softmax(logits, axis=-1) #turns unbounded numbers into probabilities adding to 1 across each row

    label_weights = jnp.array([1.0, 10.0])   # [neg, pos] --> positives weigh more on the loss
    
    cross_entropy_loss = -jnp.mean(jnp.sum(y*jnp.log(probs + 1e-8)*label_weights, axis = 1))
    
    def weights_only(leaf):
        return isinstance(leaf, jax.Array) and leaf.ndim == 2   #because we want to apply the penalty to weights, not to the biases
    
    model_weights = eqx.filter(eqx.filter(model_train, weights_only), eqx.is_inexact_array)
    leaves = jtu.tree_leaves(model_weights)
    squared_sums = [jnp.sum(w**2) for w in leaves]
    l2_loss = l2_reg * jnp.sum(jnp.stack(squared_sums))
        
    return cross_entropy_loss + l2_loss


@eqx.filter_jit
def train_step_partitioned(X, y, ts, Sd, model, filter_spec, opt_state, optimizer, loss_fn):
    """the train step partitioned to train f and g"""
    
    #partition model
    model_train, model_static = eqx.partition(model, filter_spec)
    
    #calculating loss and gradients
    loss, grads_train = loss_fn(model_train, model_static, X, y, ts, Sd)
    #just in case
    grads_train = eqx.filter(grads_train, eqx.is_inexact_array)
    #updating trainable part of the model
    updates, opt_state = optimizer.update(grads_train, opt_state)
    model_train = eqx.apply_updates(model_train, updates)
    #recombining the model
    model = eqx.combine(model_train, model_static)
    
    return loss, model, opt_state
    


def data_loader(data: SepsisDataset, batch_size, *, key):   #data in the form of a filepath array
    """makes batches for the data, In this case data loaded from npz files"""

    indices = jnp.arange(data.data_size)
    perm = random.permutation(key, indices)
    
    #shuffle data_set
    data.permutation(perm)

    num_batches = data.data_size // batch_size #We don't want to allow partial batches, so we allow droping part of the dataset
    for b in range(num_batches):
        
        start = b*batch_size
        end = start+batch_size

        yield data.get_batch(start, end)
    

    
#the training loop!
def training_loop(data_train, model, epochs, lr, batch_size, *, key, plot_loss = True): # introduce autostop argument (desired loss), 
    """training loop, Expects shapes (Batch, N, features), (Batch, 1), (Batch, N), """
    batch_size = batch_size

    #splitting model for training
    
    filter_spec_g = eqx.tree_at(
        lambda m: (m.ode_solver.ace_ode.g_ode, m.att_rnn_cell),
        jtu.tree_map(lambda _: False, model),
        replace = (True, True)
    )
    
    filter_spec_f_other = eqx.tree_at(
        lambda m: (m.ode_solver.ace_ode.g_ode, m.att_rnn_cell),
        jtu.tree_map(lambda _: True, model),
        replace = (False, False)
    )
    
    #Start Optimizers
    #optimizer_f = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(lr, weight_decay=1e-4))
    optimizer_f = optax.adam(lr)
    opt_state_f = optimizer_f.init(eqx.filter(eqx.filter(model, filter_spec_f_other),eqx.is_inexact_array))
    
    optimizer_g = optax.adam(lr)
    opt_state_g = optimizer_g.init(eqx.filter(eqx.filter(model, filter_spec_g), eqx.is_inexact_array))
    
    #create sepsis dataset
    sepsis_data = SepsisDataset(data_train)
    
    #extra variables for training loop
    loss_history = []
    total_b = data_train.shape[0]//batch_size
    b = 0
    print("Training...\n")
    for epoch in range(epochs):
        key, subkey = random.split(key)
        start_t = time.time()
        epoch_loss = []
        
        #Maybe use gradient accumulation? optax.MultiStep() ?
        for X_batch, y_batch, ts_batch, Sd_batch in data_loader(sepsis_data, batch_size, key = subkey): #Here we got rid of ts_train because it's now inside of X!!!
            loss, model, opt_state_f = train_step_partitioned(X_batch, y_batch, ts_batch, Sd_batch, 
                                                            model, filter_spec_f_other,
                                                            opt_state_f, optimizer_f, grad_loss_h)
            
            _, model, opt_state_g = train_step_partitioned(X_batch, y_batch, ts_batch, Sd_batch, 
                                                        model, filter_spec_g,
                                                        opt_state_g, optimizer_g, grad_loss_a)
            
            epoch_loss.append(loss)
            
            b += 1
            print(f"Epoch {epoch} is {int(((b)/total_b)*100)}% done", end = '\r')
            
        
        end_t = time.time() - start_t
        loss_avg = sum(epoch_loss)/len(epoch_loss)
        loss_history.append(loss_avg)   #we plot for each epoch, the average over the batches

        print("")    
        if epoch % 1 == 0:
            print(f"Epoch: {epoch}, loss: {loss_avg}, time: {end_t}\n")  #I was today years old when i found out  python does not create it's own scope for for loops wth
        
        if epoch % 5 == 0: 
            #save model information
            save_model(model, f"checkpoints/model_epoch_{epoch}.eqx")

        if epoch > (epochs * 50) // 100:    #after we went through 50% of our training time
            if loss_avg < 3e-3: break      #lets make it so this is more customizable
        b = 0
            
    if plot_loss:
        print(len(loss_history))
        plt.plot(loss_history)
        plt.yscale("log")
        plt.show()
        
    return model



def save_model(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(path, model)



def load_model(model, path):
    return eqx.tree_deserialise_leaves(path, model)

