import jax
print(jax.local_device_count())  # sollte 2 anzeigen
print(jax.devices())
