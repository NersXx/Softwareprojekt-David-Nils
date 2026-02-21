# %%
import numpy as np

# Load the archive
fname = 'umur.npz' 
archive = np.load(f'{fname}')

# Define the expected keys and shape
expected_keys = {'100', '75', '50', '25'}
expected_shape = (1000, 1)

# Test keys
s_keys = f"Unexpected keys found: {set(archive.keys())}. Keys must be: {expected_keys}."
assert set(archive.keys()) == expected_keys, s_keys

# Test values
for key, value in archive.items():
    s_type = f"Value for key '{key}' must be a numpy array."
    assert isinstance(value, np.ndarray), s_type
    s_shape = f"Value for key '{key}' must have shape {expected_shape}, not shape {value.shape}."
    assert value.shape == expected_shape, s_shape

print("All tests passed, ready for submission")