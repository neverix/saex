from functools import partial

from safetensors.flax import save_file
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import jax.sharding as jshard
from jax.sharding import PartitionSpec as P

from . import utils


class ActivationBuffer(eqx.Module):
    # A simple ring buffer for activations

    max_samples: int
    n_dimensions: int
    cache_sharding: jshard.NamedSharding
    mesh: jshard.Mesh
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
        self._n_valid = eqx.nn.StateIndex(jax.device_put_replicated(jnp.array(0), devices))
        self._index = eqx.nn.StateIndex(jax.device_put_replicated(jnp.array(0), devices))
        self.mesh = mesh

    @partial(eqx.filter_jit, donate="all-except-first")
    def __call__(self, activations, state):
        cache, n_valid, index = state.get(self._cache), state.get(self._n_valid), state.get(self._index)
        
        cache = jax.lax.with_sharding_constraint(cache, self.cache_sharding)
        activations = activations.reshape(self.mesh.shape["dp"], -1, self.n_dimensions)
        activations = jax.lax.with_sharding_constraint(activations, self.mesh.shape["dp"])
        offsets = jnp.arange(activations.shape[1])
        new_n_valid = jnp.minimum(n_valid + activations.shape[1], self.max_samples)
        # if n_valid == max_samples, we want to overwrite the oldest samples
        indices = (index + offsets) % self.max_samples
        new_index = (index + activations.shape[1]) % self.max_samples
        new_cache = cache.at[:, indices].set(activations.astype(cache.dtype))
        new_cache = jax.lax.with_sharding_constraint(new_cache, self.cache_sharding)
        return (state
                .set(self._cache, new_cache)
                .set(self._n_valid, new_n_valid)
                .set(self._index, new_index))

    @partial(eqx.filter_jit, donate="first")
    def sample_batch(self, state, key):
        key = jax.lax.with_sharding_constraint(key, jshard.NamedSharding(self.mesh, P("dp")))
        cache, n_valid = state.get(self._cache), state.get(self._n_valid)
        index = jax.random.randint(key, (self.mesh.shape["dp"], 1,), 0, n_valid)[:, :, None]
        return state, jnp.take_along_axis(cache, index, axis=1)[:, 0, :]

    def save(self, state, save_path):
        assert state.get(self._n_valid) == self.max_samples
        save_file({"cache": state.get(self._cache), "n_valid": state.get(self._n_valid)}, save_path)
