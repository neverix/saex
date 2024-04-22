from functools import partial

from safetensors.flax import save_file
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import jax.sharding as jshard
from jax.sharding import PartitionSpec as P
from typing import List


class ActivationBuffer(eqx.Module):
    # A simple ring buffer for activations

    max_samples: int
    n_dimensions: int
    cache_sharding: jshard.NamedSharding
    mesh: jshard.Mesh
    devices: List[jax.Device]
    _cache: eqx.nn.StateIndex
    _n_valid: eqx.nn.StateIndex
    _index: eqx.nn.StateIndex

    def __init__(self, max_samples, n_dimensions, mesh: jshard.Mesh, dtype=jnp.float16):
        self.max_samples = max_samples
        self.n_dimensions = n_dimensions
        self.cache_sharding = jshard.NamedSharding(mesh, P("dp", None, None))
        self._cache = eqx.nn.StateIndex(jnp.empty((mesh.shape["dp"], max_samples, n_dimensions), dtype=dtype,
                                                  device=self.cache_sharding))
        devices = np.ravel(mesh.devices).tolist()
        self._n_valid = eqx.nn.StateIndex(jnp.array(0))
        self._index = eqx.nn.StateIndex(jnp.array(0))
        self.devices = devices
        self.mesh = mesh

    @partial(eqx.filter_jit, donate="all-except-first")
    def __call__(self, activations, state):
        cache, n_valid, index = state.get(self._cache), state.get(self._n_valid), state.get(self._index)
        cache = jax.lax.with_sharding_constraint(cache, self.cache_sharding)
        activations = activations.reshape((self.mesh.shape["dp"], -1, self.n_dimensions))
        activations = jax.lax.with_sharding_constraint(activations, jshard.NamedSharding(self.mesh, P("dp", None, None)))
        offsets = jnp.arange(activations.shape[1])
        new_n_valid = jnp.minimum(n_valid + activations.shape[1], self.max_samples)
        # if n_valid == max_samples, we want to overwrite the oldest samples
        indices = (index + offsets) % self.max_samples
        new_index = (index + activations.shape[1]) % self.max_samples
        # # completely unintentional variable naming
        # # ...essentially pmap.
        # new_cache = jax.vmap(lambda c, i, a: c.at[i].set(a.astype(cache.dtype)), in_axes=(0, 0, 0))(cache, indices, activations)
        new_cache = cache.at[:, indices].set(activations.astype(cache.dtype))
        new_cache = jax.lax.with_sharding_constraint(new_cache, self.cache_sharding)
        return (state
                .set(self._cache, new_cache)
                .set(self._n_valid, new_n_valid)
                .set(self._index, new_index))

    @partial(eqx.filter_jit, donate="first")
    def sample_batch(self, state, key):
        cache, n_valid = state.get(self._cache), state.get(self._n_valid)
        index = jax.vmap(lambda k: jax.random.randint(k, (1,), 0, n_valid), in_axes=(0,))(key)[:, :, None]
        return state, jnp.take_along_axis(cache, index, axis=1)

    def save(self, state, save_path):
        assert state.get(self._n_valid) == self.max_samples
        save_file({"cache": state.get(self._cache), "n_valid": state.get(self._n_valid)}, save_path)


# https://github.com/google/jax/issues/8487#issuecomment-963693106
def scatter(input, dim, index, src):
    idx = jnp.meshgrid(*(jnp.arange(n) for n in input.shape), sparse=True)
    idx[dim] = index
    return input.at[tuple(idx)].set(src)
