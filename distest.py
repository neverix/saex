import jax
import jax.numpy as jnp
jax.distributed.initialize()
if jax.process_index() == 0:
    print(len(jax.devices()))
    print(jnp.zeros(10))
# print(len(jax.devices()))