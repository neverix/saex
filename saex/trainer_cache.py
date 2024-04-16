from dataclasses import dataclass
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
from tqdm.auto import trange
import optax

from . import utils
from .iterable_dataset import IterableDatasetConfig
from .iterable_dataset import create_iterable_dataset
from .transformers_model import TransformersModelConfig
from .sae import SAE, SAEConfig


class ActivationBuffer(eqx.Module):
    # A simple ring buffer for activations

    max_samples: int
    n_dimensions: int
    _cache: eqx.nn.StateIndex
    _n_valid: eqx.nn.StateIndex
    
    def __init__(self, max_samples, n_dimensions, dtype=jnp.float16):
        self.max_samples = max_samples
        self.n_dimensions = n_dimensions
        self._cache = eqx.nn.StateIndex(jnp.empty((max_samples, n_dimensions), dtype=dtype))
        self._n_valid = eqx.nn.StateIndex(0)

    @jax.jit(donate_argnums=(1,))
    def __call__(self, activations, state, mask=None):
        cache, n_valid = state.get(self._cache), state.get(self._n_valid)
        if mask is None:
            mask = jnp.ones(len(activations), dtype=bool)
        offsets = jnp.cumsum(mask) - 1
        new_n_valid = min(n_valid + offsets[-1], self.max_samples)
        # if n_valid == max_samples, we want to overwrite the oldest samples
        indices = (n_valid + offsets) % self.max_samples
        return (state
                .set(self._cache,
                     cache.at[indices].set(activations))
                .set(self._n_valid, new_n_valid))

    def sample_batch(self, state, key=None):
        if key is None:
            key = utils.get_key()
        cache, n_valid = state.get(self._cache), state.get(self._n_valid)
        index = jax.random.randint(key, 0, n_valid)
        return cache[index]


@dataclass
class BufferTrainerConfig:
    n_dimensions: int

    lr: float
    scheduler_warmup: int
    
    train_iterations: int
    sae_config: SAEConfig
    
    cache_batch_size: int
    model_config: TransformersModelConfig
    dataset_config: IterableDatasetConfig
    
    buffer_max_samples: int
    buffer_dtype: jnp.dtype


class BufferTrainer(object):
    def __init__(self, config: BufferTrainerConfig, sae=None, model=None, create_dataset=None):
        self.config = config
        self.buffer, self.buffer_state = eqx.nn.make_with_state(ActivationBuffer)(
            config.buffer_max_samples, config.n_dimensions, dtype=config.buffer_dtype)
        if model is None:
            model = config.model_config.model_class(config.model_config)
        self.model = model
        if create_dataset is None:
            create_dataset = create_iterable_dataset(config.dataset_config)
        self.create_dataset = create_dataset
        if sae is not None:
            self.sae, self.sae_state = utils.unstatify(sae)
        else:
            self.sae, self.sae_state = eqx.nn.make_with_state(SAE)(config.sae_config)

    def train(self):
        bar = trange(self.config.train_iterations)
        dataset_iterator = iter(self.create_dataset())
        
        sae_params, sae_static = eqx.partition(self.sae, self.sae.is_trainable)
        key = utils.get_key()
        
        @partial(jax.jit, static_argnums=1)
        @partial(jax.value_and_grad, has_aux=True)
        def loss(sae_params, sae_static, batch):
            sae = eqx.combine(sae_params, sae_static)
            sae_output, sae_state = sae(batch, self.sae_state)
            return sae_output.loss, (sae_output, sae_state)

        for iteration in bar:
            # cache more activations
            texts = []
            for _ in range(self.config.cache_batch_size):
                texts.append(next(dataset_iterator))
            activations, model_misc = self.model(texts)
            cache_kwargs = {k: v for k, v in model_misc.items()
                            if k in ["mask"]}
            self.buffer_state = self.buffer(activations, self.buffer_state, **cache_kwargs)
            
            # train SAE
            key, *subkeys = jax.random.split(key, self.config.sae_config.batch_size + 1)
            batch = jax.vmap(self.buffer.sample_batch, in_axes=(None, 0))(
                self.buffer_state, subkeys)
            (loss, (sae_output, self.sae_state)), grad = loss(sae_params, sae_static, batch)
            
            
            self.sae_state = self.sae(batch, self.sae_state)


if __name__ == "__main__":
    config = BufferTrainerConfig(
        n_dimensions=768,
        train_iterations=1000,
        train_batch_size=32,
        cache_batch_size=32,
        model_config=TransformersModelConfig(
            model_name_or_path="gpt2",
            layer=11,
            cache_n=1,
            cache_hidden_states=True,
        ),
        dataset_config=IterableDatasetConfig(
            dataset_name="Skylion007/openwebtext"
        ),
        buffer_max_samples=256,
        buffer_dtype=jnp.float16,
    )
    trainer = BufferTrainer(config)
    trainer.train()
