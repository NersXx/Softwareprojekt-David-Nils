import sys
import time

import jax
import jax.numpy as jnp
from jax import random
import numpy as np


def main() -> int:
    
    try:
        create_dataset()
        create_path_list()

    except Exception as e:
        print(e)
        return 1

    return 0



def create_dataset():

    #reading labels from csv file
    try:
        labels = np.genfromtxt("B/B/labels.csv", dtype = np.float32, delimiter = ",", skip_header=1, filling_values= None, usecols=(1,))
        labels = labels.reshape((20000,1))
        assert labels.shape == (20000, 1)   #we want this shape (Batch, Features)
    except Exception as e:
        raise e
    
    #don't mix np and jnp during preprocessing, just use np
    print("loading data ...")
    for i in range(1, 20001):
        file_path = f"B/B/time_series/p{100000+i}.csv"
        patient_data_i = {}
        
        #generate x data
        try:
            time_series_i = np.genfromtxt(file_path, dtype = np.float32, delimiter = ",", missing_values="", filling_values= np.nan)
        except Exception as e:
            raise e
            
        ### To be done
        #4) organize in cleaner modules
        #5) Maybe unify X and y into the npz files! so the label and the data in the same file -> data loader then unpacks to X y and ts
        #6) Make sure you are always using float 32??
        #8) Solve memory Issues
        
        #make sure the dimensions for the time series are always (T, Features)!
        if time_series_i.ndim == 1:
            time_series_i = time_series_i[None,:]
        assert time_series_i.ndim == 2
        
        #create static features
        static_features = time_series_i[0][-6:-1]
        static_flags = ~np.isnan(static_features)  #Now we add flags to denote if the value is observed or not, 1 if observed
        static_features = np.concatenate([static_features, static_flags]).astype(np.float32)  #remove nan values
        
        patient_data_i['static'] = static_features #last 5 columns before the last column IUCLOS (time)
        
        
        #we get rid of the first row, which only contains the static data
        time_series_i = time_series_i[1:]

        #generate time_stamps shape (T,)
        time_stamps_i = time_series_i[:,-1:].reshape(-1)
        patient_data_i['t'] = time_stamps_i
        
        #Save time series without the last Six columns (static data)
        time_series_i = time_series_i[:,:-6]    #Now shape (T, 34)
        patient_data_i['x'] = time_series_i     #should I only save after hour 0? <-----------------------
        
        #generate mask, 1 if observed 0 if missing
        obs_mask_i = (~np.isnan(time_series_i)).astype(np.float32) # or obs_mask = jnp.where(jnp.isnan(time_series_i), 0.0, 1.0)
        patient_data_i['mask'] = obs_mask_i

        #add label 1 if sepsis (i-1 because range starts at 1)
        patient_data_i['label'] = labels[i-1]
        
        #Okay so The "hour 0" is always just the static data of the patient, so we should calculate attention after that
        
        try:
            np.savez(f"data/time_series/p{100000+i}", **patient_data_i)
        except Exception as e:
            raise e
        
        if i % 200 == 0:
            print("#", end = '', flush=True)
    print("\nData Created Succesfully")        


def create_path_list():
    #create this as a npy file as well!!!. you can create the split later
    data = []

    for i in range(1, 20001):
        file_path = f'data/time_series/p{100000+i}.npz'
        
        if np.load(file_path)['x'].shape[0] != 0:   #If the time series is not empty
            data.append(file_path)
    
    data = np.array(data)
    np.save("data/path_list.npy", data)
    


def load_dataset(split_ratio = 0.8):

    data = np.load('data/path_list.npy')

    #Shuffle and divide into Train and Test set
    data = np.array(data)
    data_size = len(data)
    indices = np.arange(data_size)
    
    reproducable_key = random.key(1)    #SO THAT WE GET THE SAME SPLIT OF DATASET EVERY TIME, for now
    perm = random.permutation(reproducable_key, indices)
    
    data_shuffled = data[perm]    #X shape (Batch,)

    #Split into Test and Train
    split = int(split_ratio*data_size)
    
    data_train = data_shuffled[:split]
    data_test = data_shuffled[split:]
    
    return data_train, data_test


def create_norm_data(data_X):
    
    sum_x = None    
    sumsq_x = None
    count_x = None
    
    sum_static = None
    sumsq_static = None
    count_static = None
    
    for path in data_X:
        data = np.load(path)
        
        x = data['x']           #(T.F)
        mask = data['mask']     #(T,F)
        static = data['static'] #(S*2,) -> [values | Flags]
        
        #norm statistics for time_series
        observed = mask == 1    #transforming mask of 1 and 00 into true and false
        x_obs = np.where(observed, x, 0.0)

        if sum_x is None:   #Initialize sum vectors
            F = x.shape[1]
            sum_x = np.zeros(F)
            sumsq_x = np.zeros(F)
            count_x = np.zeros(F)
        
        sum_x += x_obs.sum(axis=0)
        sumsq_x += (x_obs ** 2).sum(axis = 0)
        count_x += observed.sum(axis = 0) #how many values where observed in each column
        
        #norm statistics for static
        S = static.shape[0] // 2
        static_vals = static[:S]
        static_flags = static[S:] 
        
        observed_s = static_flags == 1
        static_obs = np.where(observed_s, static_vals, 0.0)
        
        if sum_static is None:
           sum_static = np.zeros(S) 
           sumsq_static = np.zeros(S)
           count_static = np.zeros(S)
        
        sum_static += static_obs           #(S,)
        sumsq_static += static_obs ** 2
        count_static += observed_s
        
        
    #calculate mean and std for datasets    
    mean_x = sum_x / np.maximum(count_x, 1) #maximum to avoid division by 0
    var_x = sumsq_x / np.maximum(count_x, 1) - mean_x**2
    std_x = np.sqrt(np.maximum(var_x, 1e-6)) #In case variance is 0
    
    mean_static = sum_static / np.maximum(count_static, 1)
    var_static = sumsq_static / np.maximum(count_static, 1) - mean_static**2
    std_static = np.sqrt(np.maximum(var_static, 1e-6))
    
    #create file with norm_stats
    np.savez("data/norm_stats.npz", 
        mean_x =mean_x, std_x = std_x,
        mean_static=mean_static,
        std_static=std_static)


if __name__ == "__main__":
    sys.exit(main())


