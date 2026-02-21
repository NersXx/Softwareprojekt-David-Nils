#Group: David, Nils

import jax.numpy as jnp
import numpy as np


class SpiralData:

    class InvalidData(Exception):
        def __init__(self):
            pass
            
        def __str__(self):
            return f"Something went wrong\n"
        
        
    def __init__(self, dataset):
        self.dataset = dataset
           
        
    def extract_data(self):
        """returns data_train, alpha_train, data_validation, alpha_validation, data_test"""
        try:
            data_train = jnp.array(self.dataset["data_train"])
            alpha_train = jnp.array(self.dataset["alpha_train"])
            data_validation = jnp.array(self.dataset["data_validation"])
            alpha_validation = jnp.array(self.dataset["alpha_validation"])
            data_test = jnp.array(self.dataset["data_test"])
        except Exception as e:
            raise self.InvalidData(self.dataset)
        
        return data_train, alpha_train, data_validation, alpha_validation, data_test
    
    
    
    
        
try:
    spirals_25 = SpiralData(np.load('spirals_25.npz'))
    spirals_50 = SpiralData(np.load('spirals_50.npz'))
    spirals_75 = SpiralData(np.load('spirals_75.npz'))
    spirals_100 = SpiralData(np.load('spirals_100.npz'))
except Exception as e:
    raise e
    


