from dataclasses import dataclass
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from tqdm.auto import trange

from . import utils
from .iterable_dataset import IterableDatasetConfig, create_iterable_dataset
from .sae import SAE, SAEConfig
from .transformers_model import TransformersModelConfig


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

    def sample_batch(self, state, key=None):
        if key is None:
            key = utils.get_key()
        cache, n_valid = state.get(self._cache), state.get(self._n_valid)
        index = jax.random.randint(key, (1,), 0, n_valid)[0]
        return cache[index]


@dataclass
class BufferTrainerConfig:
    n_dimensions: int

    lr: float
    scheduler_warmup: int
    scheduler_cycle: int

    train_iterations: int
    dry_run_steps: int
    sae_config: SAEConfig
    
    cache_batch_size: int
    cache_every_steps: int
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
        bar = trange(self.config.train_iterations + self.config.dry_run_steps)
        dataset_iterator = iter(self.create_dataset())
        
        is_trainable = lambda value: eqx.is_array(value) and value.dtype.kind in "f"
        sae_params, sae_static = eqx.partition(self.sae, is_trainable)
        key = utils.get_key()
        
        scheduler_cycle = self.config.scheduler_cycle
        if not scheduler_cycle:
            scheduler_cycle = self.config.train_iterations - self.config.scheduler_warmup
        n_cycles = int(self.config.train_iterations / scheduler_cycle)
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(self.config.lr),
            optax.scale_by_schedule(
                optax.join_schedules(
                    [optax.linear_schedule(0, 1, self.config.scheduler_warmup)]
                    + [optax.cosine_decay_schedule(1, scheduler_cycle) for _ in range(n_cycles)],
                    boundaries=[self.config.scheduler_warmup + i * scheduler_cycle for i in range(n_cycles)]),
            ),
            optax.zero_nans(),
        )
        opt_state = optimizer.init(sae_params)
        
        @partial(jax.jit, static_argnums=1)
        @partial(jax.value_and_grad, has_aux=True)
        def loss_fn(sae_params, sae_static, sae_state, batch):
            sae = eqx.combine(sae_params, sae_static)
            sae_output, sae_state = sae(batch, sae_state)
            return sae_output.loss, (sae_output, sae_state)

        for iteration in bar:
            if iteration % self.config.cache_every_steps == 0 or iteration < self.config.dry_run_steps:
                # cache more activations
                texts = []
                for _ in range(self.config.cache_batch_size):
                    texts.append(next(dataset_iterator))
                activations, model_misc = self.model(texts)
                self.buffer_state = self.buffer(activations, self.buffer_state, mask=model_misc.get("mask"))
            
            if iteration < self.config.dry_run_steps:
                bar.set_description("Caching activations")
                continue
            bar.set_description("Training SAE")

            # train SAE
            key, subkeys = jax.random.split(key, 2)
            subkeys = jax.random.split(subkeys, self.config.sae_config.batch_size)
            batch = jax.vmap(self.buffer.sample_batch, in_axes=(None, 0))(
                self.buffer_state, subkeys).astype(jnp.float32)
            batch = jnp.nan_to_num(batch)
            (_, (sae_output, self.sae_state)), grad = loss_fn(sae_params, sae_static, self.sae_state, batch)
            updates, opt_state = optimizer.update(grad, opt_state, sae_params)
            self.sae = self.sae.apply_updates(updates, self.sae_state,
                                              batch, sae_output,
                                              int(self.sae_state.get(self.sae.num_steps)))
            sae_params, sae_static = eqx.partition(self.sae, is_trainable)
            
            stats = self.sae.get_stats(self.sae_state, batch, sae_output)
            bar.set_postfix(stats)
            # TODO: track in wandb or other logger:
            # - L0
            # - reconstruction loss
            # - loss
            # - number of dead features (tracked using own window)
            # - % variance explained
            # - learning rate
            # - norm ratio


if __name__ == "__main__":
    config = BufferTrainerConfig(
        n_dimensions=768,
        lr=1e-2,
        scheduler_warmup=20,
        scheduler_cycle=10_000,
        train_iterations=100_000,
        dry_run_steps=0,
        sae_config=SAEConfig(
            n_dimensions=768,
            sparsity_coefficient=1e-3,
            batch_size=2**14,
            expansion_factor=32,
            use_encoder_bias=False,
            decoder_init_method="pseudoinverse",
            decoder_bias_init_method="zeros",
            sparsity_loss_type="l1",
            reconstruction_loss_type="mse",
            project_updates_from_dec=True,
            use_ghost_grads=True,
            dead_after=50,
            restrict_dec_norm="exact",
            stat_tracking_epsilon=0.05,
        ),
        cache_every_steps=8,
        cache_batch_size=32,
        model_config=TransformersModelConfig(
            model_name_or_path="gpt2",
            layer=1,
        ),
        dataset_config=IterableDatasetConfig(
            dataset_name="Skylion007/openwebtext"
        ),
        buffer_max_samples=2 ** 19,
        buffer_dtype=jnp.float16,
    )
    trainer = BufferTrainer(config)
    trainer.train()
