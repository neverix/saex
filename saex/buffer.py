from functools import partial
from typing import List

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.sharding as jshard
import numpy as np
from jax.sharding import PartitionSpec as P
from safetensors import safe_open
from safetensors.flax import save_file


class ActivationBuffer(eqx.Module):
    # A simple ring buffer for activations

    max_samples: int
    n_dimensions: int
    cache_sharding: jshard.NamedSharding
    stat_sharding: jshard.NamedSharding
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
        self.stat_sharding = jshard.NamedSharding(mesh, P("dp"))
        self._n_valid = eqx.nn.StateIndex(jnp.zeros((mesh.shape["dp"],), device=self.stat_sharding, dtype=jnp.int32))
        self._index = eqx.nn.StateIndex(jnp.zeros((mesh.shape["dp"],), device=self.stat_sharding, dtype=jnp.int32))
        self.devices = devices
        self.mesh = mesh

    @partial(eqx.filter_jit, donate="all-except-first")
    def __call__(self, activations, mask, state):
        cache, n_valid, index = state.get(self._cache), state.get(self._n_valid), state.get(self._index)
        cache = jax.lax.with_sharding_constraint(cache, self.cache_sharding)
        n_valid = jax.lax.with_sharding_constraint(n_valid, self.stat_sharding)
        index = jax.lax.with_sharding_constraint(index, self.stat_sharding)
    
        activations = activations.reshape((self.mesh.shape["dp"], -1, self.n_dimensions)).astype(cache.dtype)
        activations = jax.lax.with_sharding_constraint(activations, jshard.NamedSharding(self.mesh, P("dp", None, None)))
        mask = mask.reshape((self.mesh.shape["dp"], -1))
        mask = jax.lax.with_sharding_constraint(mask, jshard.NamedSharding(self.mesh, P("dp", None)))

        offsets = mask.astype(jnp.int32).cumsum(axis=1) - 1
        accumulated = offsets[:, -1] + 1
        new_n_valid = jnp.minimum(n_valid + accumulated, self.max_samples)
        # if n_valid == max_samples, we want to overwrite the oldest samples
        indices = (index[:, None] + offsets) % self.max_samples
        new_index = (index + accumulated) % self.max_samples
        # completely unintentional variable naming
        # ...essentially pmap.
        new_cache = jax.vmap(lambda c, i, a, m: c.at[i].set(0).at[i].add((a * m[..., None]).astype(cache.dtype)),
                             in_axes=(0, 0, 0, 0))(cache, indices, activations, mask)
        
        new_cache = jax.lax.with_sharding_constraint(new_cache, self.cache_sharding)
        new_n_valid = jax.lax.with_sharding_constraint(new_n_valid, self.stat_sharding)
        new_index = jax.lax.with_sharding_constraint(new_index, self.stat_sharding)
        
        return (state
                .set(self._cache, new_cache)
                .set(self._n_valid, new_n_valid)
                .set(self._index, new_index))

    @partial(eqx.filter_jit, donate="first")
    def sample_batch(self, state, key):
        cache, n_valid = state.get(self._cache), state.get(self._n_valid)
        index = jax.vmap(lambda k, nv: jax.random.randint(k, (1,), 0, nv), in_axes=(0, 0))(key, n_valid)[:, :, None]
        return state, jnp.take_along_axis(cache, index, axis=1)

    def save(self, state, save_path):
        assert state.get(self._n_valid) == self.max_samples
        cache = state.get(self._cache)
        save_file({"cache": cache.reshape(-1, cache.shape[-1]), "n_valid": state.get(self._n_valid)}, save_path)

    def restore(self, state, restore_path):
        with safe_open(restore_path, framework="flax") as f:
            state = state.set(self._cache, jax.device_put(f.get_tensor("cache")
                                                          .reshape(state.get(self._cache).shape),
                                                          self.cache_sharding))
            state = state.set(self._n_valid, f.get_tensor("n_valid"))
            state = state.set(self._index, jnp.array(0))
        return state


# https://github.com/google/jax/issues/8487#issuecomment-963693106
def scatter(input, dim, index, src):
    idx = jnp.meshgrid(*(jnp.arange(n) for n in input.shape), sparse=True)
    idx[dim] = index
    return input.at[tuple(idx)].set(src)
