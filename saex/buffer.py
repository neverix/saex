import equinox as eqx
import jax
import jax.numpy as jnp

from functools import partial
from . import utils


class ActivationBuffer(eqx.Module):
    # A simple ring buffer for activations

    max_samples: int
    n_dimensions: int
    _cache: eqx.nn.StateIndex
    _n_valid: eqx.nn.StateIndex
    _index: eqx.nn.StateIndex

    def __init__(self, max_samples, n_dimensions, dtype=jnp.float16):
        self.max_samples = max_samples
        self.n_dimensions = n_dimensions
        self._cache = eqx.nn.StateIndex(jnp.empty((max_samples, n_dimensions), dtype=dtype))
        self._n_valid = eqx.nn.StateIndex(jnp.array(0))
        self._index = eqx.nn.StateIndex(jnp.array(0))

    @partial(eqx.filter_jit, donate="all-except-first")
    def __call__(self, activations, state, mask=None):
        cache, n_valid, index = state.get(self._cache), state.get(self._n_valid), state.get(self._index)
        if mask is None:
            mask = jnp.ones(len(activations), dtype=jnp.bool)
        offsets = jnp.cumsum(mask.astype(jnp.int32)) - 1
        new_n_valid = jnp.minimum(n_valid + offsets[-1], self.max_samples)
        # if n_valid == max_samples, we want to overwrite the oldest samples
        indices = (index + offsets) % self.max_samples
        new_index = (index + offsets[-1]) % self.max_samples
        return (state
                .set(self._cache,
                     cache
                     # TODO properly order indices so one .set() does the job
                     .at[indices].set(0)
                     .at[indices].add(activations.astype(cache.dtype) * mask[:, None]))
                .set(self._n_valid, new_n_valid)
                .set(self._index, new_index))

    @partial(eqx.filter_jit, donate="first")
    def sample_batch(self, state, batch_size, key=None):
        if key is None:
            key = utils.get_key()
        cache, n_valid = state.get(self._cache), state.get(self._n_valid)
        index = jax.random.randint(key, (batch_size,), 0, n_valid)
        return state, cache[index]
