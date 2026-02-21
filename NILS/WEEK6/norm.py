import jax
from jax import random
import jax.numpy as jnp


class Normalizer:
    
    def __init__(self):
        pass
    
    def normalize(self, data : jax.Array):
        pass
    
    def denormalize(self, data : jax.Array):
        pass
    
    def init(self, data : jax.Array, axis = None):
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
    
    def init(self, data, axis = None):
        pass
    
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
    
    def init(self, data, axis = None):
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
    
    def init(self, data, axis = None):
        self.min = data.min(axis = axis)
        self.max = data.max(axis = axis)
    
    def __call__(self, data, *, denormalize=False):
        return super().__call__(data, denormalize=denormalize)

        

