from dataclasses import dataclass
from functools import partial
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from tqdm.auto import trange

from . import utils
from .buffer import ActivationBuffer
from .iterable_dataset import IterableDatasetConfig, create_iterable_dataset
from .sae import SAE, SAEConfig
from .transformers_model import TransformersModelConfig


@dataclass
class BufferTrainerConfig:
    n_dimensions: int

    lr: float
    scheduler_warmup: int
    scheduler_cycle: int

    train_iterations: int
    save_steps: Optional[int]
    save_path: Optional[str]
    dry_run_steps: int
    no_update: bool
    
    sae_config: SAEConfig
    sae_restore: Optional[str]
    
    cache_batch_size: int
    cache_every_steps: int
    cache_acc: int
    model_config: TransformersModelConfig
    dataset_config: IterableDatasetConfig
    
    buffer_max_samples: int
    buffer_dtype: jnp.dtype


class BufferTrainer(object):
    def __init__(self, config: BufferTrainerConfig, sae=None, model=None, create_dataset=None):
        self.config = config
        if self.config.buffer_max_samples < self.config.sae_config.batch_size:
            print("Skipping buffer creation because buffer_max_samples < sae_config.batch_size")
            self.buffer = None
        else:
            self.buffer, self.buffer_state = eqx.nn.make_with_state(ActivationBuffer)(
                config.buffer_max_samples, config.n_dimensions, dtype=config.buffer_dtype)
        if sae is not None:
            self.sae, self.sae_state = utils.unstatify(sae)
        else:
            print("Creating SAE...")
            self.sae, self.sae_state = eqx.nn.make_with_state(SAE)(config.sae_config)
            if config.sae_restore:
                print(f"Loading checkpoint ({config.sae_restore})...")
                self.sae = self.sae.restore(config.sae_restore)
        if model is None:
            print("Loading model...")
            model = config.model_config.model_class(config.model_config)
        self.model = model
        if create_dataset is None:
            print("Loading dataset...")
            create_dataset = create_iterable_dataset(config.dataset_config)
        self.create_dataset = create_dataset

    def train(self):
        print("Training for", self.config.train_iterations, "iterations")
        
        dataset_iterator = iter(self.create_dataset())
        
        is_trainable = lambda value: eqx.is_array(value) and value.dtype.kind in "f"
        sae_params, sae_static = eqx.partition(self.sae, is_trainable)
        key = utils.get_key()
        
        scheduler_cycle = self.config.scheduler_cycle
        if not scheduler_cycle:
            scheduler_cycle = self.config.train_iterations
        print("Learning rate:", self.config.lr, "warmed up for", self.config.scheduler_warmup, "iterations",
              "and cycled every", scheduler_cycle, "iterations")
        n_cycles = int(self.config.train_iterations / scheduler_cycle)
        optimizer = optax.chain(
            # optax.clip_by_global_norm(1.0),
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

        @partial(jax.jit, static_argnums=(2,), donate_argnums=(1, 3, 4))
        def train_step(
            batch, sae_params, sae_static, sae_state, opt_state, step
        ):
            batch = jnp.nan_to_num(batch)
            (_, (sae_output, sae_state)), grad = loss_fn(sae_params, sae_static, sae_state, batch)
            sae = eqx.combine(sae_params, sae_static)
            if not self.config.no_update:
                updates, opt_state = optimizer.update(grad, opt_state, sae_params)
                sae = sae.apply_updates(updates, sae_state,
                                        batch, sae_output, step)
                sae_params, _ = eqx.partition(sae, is_trainable)
            sae = eqx.combine(sae_params, sae_static)
            stats = sae.get_stats(sae_state, batch, sae_output)
            return sae_params, sae_state, opt_state, stats

        bar = trange(self.config.train_iterations + self.config.dry_run_steps)
        for iteration in bar:
            if (iteration % self.config.cache_every_steps == 0
                or iteration < self.config.dry_run_steps
                or self.buffer is None):
                for _ in range(self.config.cache_acc):
                    # cache more activations
                    texts = []
                    for _ in range(self.config.cache_batch_size):
                        texts.append(next(dataset_iterator))
                    activations, model_misc = self.model(texts)
                    if self.buffer is not None:
                        self.buffer_state = self.buffer(activations, self.buffer_state, mask=model_misc.get("mask"))
            
            if iteration < self.config.dry_run_steps:
                bar.set_description("Caching activations")
                continue
            bar.set_description("Training SAE")

            # train SAE
            if self.buffer is None:
                batch = activations
            else:
                key, subkey = jax.random.split(key, 2)
                self.buffer_state, batch = self.buffer.sample_batch(
                    self.buffer_state, self.config.sae_config.batch_size, subkey)
            
            # TODO put update into 1 step
            sae_params, self.sae_state, opt_state, stats = train_step(
                batch, sae_params, sae_static, self.sae_state, opt_state, iteration)

            bar.set_postfix(stats)
            # TODO: track in wandb or other logger:
            # - L0
            # - reconstruction loss
            # - loss
            # - number of dead features (tracked using own window)
            # - % variance explained
            # - learning rate
            # - norm ratio
            
            if self.config.save_steps is not None and iteration % self.config.save_steps == 0:
                self.sae = eqx.combine(sae_params, sae_static)
                self.sae.save(self.config.save_path)


def main():
    layer = 1
    cache_size = 524288  # 2**19
    cache_batch_size = 512
    batch_size = 1024
    max_seq_len = 128
    config = BufferTrainerConfig(
        n_dimensions=768,
        lr=1e-3,
        scheduler_warmup=128,
        scheduler_cycle=None,
        train_iterations=10_000,
        save_steps=1_000,
        # save_steps=None,
        save_path=f"weights/gpt2-{layer}.safetensors",
        # save_path=f"weights/gpt2s-{layer}-tuned.safetensors",
        dry_run_steps=0,
        no_update=False,
        sae_config=SAEConfig(
            n_dimensions=768,
            # sparsity_loss_type="l1_sqrt",
            sparsity_loss_type=("recip", 0.1),
            # sparsity_loss_type="l1",
            sparsity_coefficient=3e-5,
            batch_size=batch_size,
            expansion_factor=32,
            use_encoder_bias=True,
            remove_decoder_bias=True,
            encoder_init_method="orthogonal",
            # https://tenor.com/view/gun-tears-cat-point-gun-crying-cat-gif-17741904
            decoder_init_method="pseudoinverse",
            # decoder_bias_init_method="zeros",
            decoder_bias_init_method="geom_median",
            reconstruction_loss_type="mse_batchnorm",
            project_updates_from_dec=True,
            # death_loss_type="sparsity_threshold",
            death_loss_type="ghost_grads",
            # death_loss_type="none",
            death_penalty_threshold=1e-5,
            dead_after=8_000,
            restrict_dec_norm="exact",
            sparsity_tracking_epsilon=0.05,
        ),
        sae_restore=None,
        # sae_restore=f"weights/jb-gpt2s-{layer}.safetensors",
        cache_every_steps=int(cache_size / batch_size / 2),
        cache_batch_size=cache_batch_size,
        cache_acc=int(cache_size / cache_batch_size / max_seq_len),
        buffer_max_samples=cache_size,
        model_config=TransformersModelConfig(
            model_name_or_path="gpt2",
            # model_name_or_path="MBZUAI/LaMini-GPT-124M",
            from_pt=True,
            layer=layer,
            max_seq_len=max_seq_len,
            add_prefix="<|endoftext|>",
            concat_all=False,
            
            # cache_n=25,
            return_real_mask=False,
        ),
        dataset_config=IterableDatasetConfig(
            dataset_name="Skylion007/openwebtext",
            # dataset_name="nev/lamini-dataset-text",
        ),
        buffer_dtype=jnp.float16,
    )
    trainer = BufferTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
