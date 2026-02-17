#Group: David, Nils

import jax
from jax import random
import jax.numpy as jnp


class Normalizer:
    """Class to apply different normalization strategies to data"""
    def __init__(self):
        pass
    
    def normalize(self, data : jax.Array):
        pass
    
    def denormalize(self, data : jax.Array):
        pass
    
    def init(self, data : jax.Array,*, axis = None):
        pass
     
    def __call__(self, data : jax.Array, *, denormalize = False):
        if not denormalize:
            return self.normalize(data)
        else:
            return self.denormalize(data)



class IdentityNorm(Normalizer):
    
    def __init__(self):
        pass
    
    def normalize(self, data):
        return data
    
    def denormalize(self, data):
        return data
    
    def init(self, data,*, axis = None):
        pass
    
    def __call__(self, data, *, denormalize=False):
        return super().__call__(data, denormalize=denormalize)
    


class MinShiftNorm(Normalizer):
    
    def __init__(self):
        pass
    
    def normalize(self, data):
        data_normalized = data - self.min
        return data_normalized
    
    def denormalize(self, data):
        data_denormalized = data + self.min
        return data_denormalized
    
    def init(self, data,*, axis = None):
        self.min = data.min(axis = axis)
    
    def __call__(self, data, *, denormalize=False):
        return super().__call__(data, denormalize=denormalize)




class ZScoreNorm(Normalizer):
    
    def __init__(self):
        pass
    
    def normalize(self, data):
        data_normalized = (data - self.mean) / self.std
        return data_normalized
    
    def denormalize(self, data):
        data_denormalized = (data * self.std) + self.mean
        return data_denormalized
    
    def init(self, data,*, axis = None):
        self.mean = jnp.mean(data, axis = axis)
        self.std = jnp.std(data, axis = axis)
    
    def __call__(self, data, *, denormalize=False):
        return super().__call__(data, denormalize=denormalize)
    
 
    
class MinMaxNorm(Normalizer):
    
    def __init__(self):
        pass
    
    def normalize(self, data):
        data_normalized = (data - self.min) / (self.max - self.min)
        return data_normalized
    
    def denormalize(self, data):
        data_denormalized = data * (self.max - self.min) + self.min
        return data_denormalized
    
    def init(self, data,*, axis = None):
        self.min = data.min(axis = axis)
        self.max = data.max(axis = axis)
    
    def __call__(self, data, *, denormalize=False):
        return super().__call__(data, denormalize=denormalize)



class LogNorm(Normalizer):
    
    def __init__(self):
        pass
    
    def normalize(self, data):

        data_normalized = jnp.log(data + self.eps)
        
        if self.standardize:
            data_normalized = (data_normalized - self.mean) / self.std
         
        return data_normalized

    def denormalize(self, data):
        
        if self.standardize:
            data = (data * self.std) + self.mean
    
        data_denormalized = jnp.exp(data) - self.eps
        
        return data_denormalized
    
    def init(self, data,*, axis = None, standardize = True):
        
        self.eps = 0
        if data.min() < 0: 
            self.eps -= (data.min())
            
        near_zero = 1e-8
        if jnp.any(data <= near_zero):
            self.eps += 1e-8
        
        if standardize:
            self.mean = jnp.mean(jnp.log(data + self.eps), axis = axis)
            self.std = jnp.std(jnp.log(data + self.eps), axis = axis)
            self.standardize = True
        else: self.standardize = False
    
    def __call__(self, data, *, denormalize=False):
        return super().__call__(data, denormalize=denormalize)

        

