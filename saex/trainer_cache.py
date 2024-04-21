import json
from dataclasses import dataclass, is_dataclass
from functools import partial
from typing import Optional, Tuple

import equinox as eqx
import jax
import jax.experimental.mesh_utils as mesh_utils
import jax.numpy as jnp
import jax.sharding as jshard
import jax_smi
import numpy as np
import optax
from tqdm.auto import trange

import wandb

from . import utils
from .buffer import ActivationBuffer
from .iterable_dataset import IterableDatasetConfig, create_iterable_dataset
from .sae import SAE, SAEConfig
from .transformers_model import TransformersModelConfig


@dataclass
class BufferTrainerConfig:
    n_dimensions: int
    
    loss_batch_size: int

    use_wandb: Optional[Tuple[str, str]]
    log_every: int
    hist_every: int
    eval_loss_every: int

    lr: float
    beta1: float
    beta2: float
    scheduler_warmup: int
    scheduler_cycle: int
    scheduler_multiply: float

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
    save_buffer: Optional[str]
    
    use_devices: int


class BufferTrainer(object):
    def __init__(self, config: BufferTrainerConfig, sae=None, model=None, create_dataset=None):
        self.config = config

        try:
            self.mesh = mesh_utils.create_device_mesh((self.config.use_devices, 1))
            self.sharding = jshard.PositionalSharding(self.mesh)
            self.sharding_dp = self.sharding.replicate(0)
        except ValueError:
            print("Warning: mesh size mismatch, falling back to single device")
            device = jax.devices()[0]
            self.sharding = device
            self.sharding_dp = device
        
        if self.config.buffer_max_samples < self.config.sae_config.batch_size:
            print("Skipping buffer creation because buffer_max_samples < sae_config.batch_size")
            self.buffer = None
        else:
            self.buffer, self.buffer_state = eqx.nn.make_with_state(ActivationBuffer)(
                config.buffer_max_samples, config.n_dimensions,
                dtype=config.buffer_dtype)
            self.buffer_state = eqx.filter_shard(self.buffer_state, self.sharding_dp)
        if sae is not None:
            self.sae, self.sae_state = utils.unstatify(sae)
        else:
            print("Creating SAE...")
            self.sae, self.sae_state = eqx.nn.make_with_state(SAE)(config.sae_config)
            if config.sae_restore:
                print(f"Loading checkpoint ({config.sae_restore})...")
                self.sae = self.sae.restore(config.sae_restore)
            self.sae = eqx.filter_shard(self.sae, self.sharding_dp)
            self.sae_state = eqx.filter_shard(self.sae_state, self.sharding_dp)
        
        if model is None:
            print("Loading model...")
            model = config.model_config.model_class(config.model_config, sharding=self.sharding_dp)
        self.model = model
        
        if create_dataset is None:
            print("Loading dataset...")
            create_dataset = create_iterable_dataset(config.dataset_config)
        self.create_dataset = create_dataset

    def train(self):
        print("Training for", self.config.train_iterations, "iterations")
        
        if self.config.use_wandb:
            entity, project = self.config.use_wandb
            run = wandb.init(entity=entity, project=project)
            
            def save_config(config, prefix=""):
                for k, v in config.items():
                    if isinstance(v, dict):
                        save_config(v, prefix + k + ".")
                    elif is_dataclass(v):
                        save_config(v.__dict__, prefix + k + ".")
                    else:
                        try:
                            run.config[prefix + k] = json.dumps(v)
                        except TypeError:
                            pass
            save_config(self.config.__dict__)
        
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
            optax.adam(self.config.lr, b1=self.config.beta1, b2=self.config.beta2),
            optax.scale_by_schedule(
                optax.join_schedules(
                    [optax.linear_schedule(0, 1, self.config.scheduler_warmup)]
                    + [optax.cosine_decay_schedule(1, scheduler_cycle, self.config.scheduler_multiply)
                       for _ in range(n_cycles)],
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

        @partial(jax.jit, donate_argnums=(1, 2, 3), static_argnums=(4, 5))
        def train_step(
            batch, sae_params, sae_state, opt_state, sae_static, optimizer, step, key
        ):
            batch = jnp.nan_to_num(batch)
            sae_params, sae_state, opt_state = eqx.filter_shard(
                (sae_params, sae_state, opt_state), self.sharding_dp)
            
            batch = eqx.filter_shard(batch, self.sharding)
            (_, (sae_output, sae_state)), grad = loss_fn(sae_params, sae_static, sae_state, batch)
            sae_params = eqx.filter_shard(sae_params, self.sharding_dp)
            sae = eqx.combine(sae_params, sae_static)
            if not self.config.no_update:
                updates, opt_state = optimizer.update(grad, opt_state, sae_params)
                sae, sae_state, opt_state = sae.apply_updates(updates, sae_state, opt_state,
                                                   batch, sae_output, step, key)
                sae_params, _ = eqx.partition(sae, is_trainable)
            sae = eqx.combine(sae_params, sae_static)
            stats = sae.get_stats(sae_state, batch, sae_output)
            sae_params, sae_state, opt_state = eqx.filter_shard(
                (sae_params, sae_state, opt_state), self.sharding_dp)
            return sae_params, sae_state, opt_state, stats

        bar = trange(self.config.train_iterations + self.config.dry_run_steps)
        tokens_processed = 0
        try:
            for iteration in bar:
                if (iteration % self.config.cache_every_steps == 0
                    or iteration < self.config.dry_run_steps
                    or self.buffer is None):
                    mid_buffer = jnp.empty((
                        self.config.cache_acc * self.config.cache_batch_size * self.config.model_config.max_seq_len,
                        self.config.n_dimensions), dtype=self.config.buffer_dtype)
                    accumulated = 0
                    while accumulated < mid_buffer.shape[0]:
                        # cache more activations
                        texts = []
                        for _ in range(self.config.cache_batch_size):
                            texts.append(next(dataset_iterator))
                        activations, model_misc = self.model(texts)
                        mask = model_misc.get("mask")
                        if mask is not None:
                            n_tokens = mask.sum()
                            raw_tokens = activations.reshape(-1, activations.shape[-1])[mask.flatten()]
                            mid_buffer = mid_buffer.at[accumulated:accumulated + n_tokens].set(raw_tokens)  # ..
                            accumulated += n_tokens
                        else:
                            n_tokens = np.prod(activations.shape[:-1])
                        tokens_processed += n_tokens
                    if self.buffer is not None:
                        self.buffer_state = self.buffer(mid_buffer, self.buffer_state)
                
                if iteration < self.config.dry_run_steps:
                    bar.set_description("Caching activations")
                    continue
                bar.set_description("Training SAE")

                # train SAE
                if self.buffer is None:
                    batch = mid_buffer
                else:
                    key, subkey = jax.random.split(key, 2)
                    # self.buffer_state, batch = self.buffer.sample_batch(
                    #     self.buffer_state, self.config.sae_config.batch_size, subkey)
                    subkeys = jax.random.split(subkey, self.config.sae_config.batch_size)
                    subkeys = eqx.filter_shard(subkeys, self.sharding)
                    self.buffer_state, batch = jax.vmap(self.buffer.sample_batch,
                                                        in_axes=(None, 0), out_axes=(None, 0))(
                        self.buffer_state,
                        subkeys)
                
                batch = eqx.filter_shard(batch, self.sharding)
                
                key, subkey = jax.random.split(key)
                sae_params, self.sae_state, opt_state, stats = train_step(
                    batch, sae_params, self.sae_state, opt_state, sae_static, optimizer,
                    iteration, subkey)
                stats["tokens_processed"] = tokens_processed

                bar.set_postfix(stats)
                
                if self.config.use_wandb:
                    if iteration % self.config.log_every == 0:
                        run.log(stats, step=iteration)
                    if iteration % self.config.hist_every == self.config.hist_every - 1:
                        # look at this graph
                        run.log({"histogram": wandb.Histogram(self.sae.get_log_sparsity(self.sae_state))}, step=iteration)
                    if iteration % self.config.eval_loss_every == self.config.eval_loss_every - 1:
                        texts = []
                        for _ in range(self.config.loss_batch_size):
                            texts.append(next(dataset_iterator))
                        self.sae = eqx.combine(sae_params, sae_static)
                        loss_clean, loss_reconstructed = self.model.eval_loss(texts, self.sae)
                        run.log({"loss_clean": loss_clean, "loss_reconstructed": loss_reconstructed,
                                 "recon_loss_diff": loss_reconstructed - loss_clean}, step=iteration)
                
                # TODO: track in wandb or other logger:
                # - learning rate
                # - norm ratio
                
                if self.config.save_steps is not None and iteration % self.config.save_steps == 0:
                    self.sae = eqx.combine(sae_params, sae_static)
                    self.sae.save(self.config.save_path)
        except KeyboardInterrupt:
            print("Exiting early...")
            if self.config.save_buffer:
                save_buffer = input("Save buffer? (y/N)")
                if save_buffer.lower() in ("y", "yes"):
                    self.buffer.save(self.buffer_state, self.config.save_buffer)
        run.finish()


def main():
    # jax_smi.initialise_tracking()
 
    n_devices = len(jax.devices())
    # n_devices = 1
    layer = 9
    cache_size = 2**19
    cache_batch_size = 1024
    batch_size = 1024
    max_seq_len = 128
    restore = False
    n_features = 768
    # n_features = 1600
    config = BufferTrainerConfig(
        n_dimensions=n_features,
        lr=4e-4,
        # lr=6e-5,
        beta1=0.0,
        # beta1=0.99,
        # beta2=0.99,
        beta2=0.999,
        scheduler_warmup=128,
        scheduler_cycle=100_000,
        scheduler_multiply=0.1,
        train_iterations=20_000,
        # save_steps=1_000,
        save_steps=None,
        use_wandb=("neverix", "saex"),
        log_every=10,
        hist_every=100,
        save_path=f"weights/gpt2-{layer}.safetensors" if not restore else f"weights/gpt2s-{layer}-tuned.safetensors",
        dry_run_steps=0,
        no_update=False,
        sae_config=SAEConfig(
            n_dimensions=n_features,
            # sparsity_loss_type="l1_sqrt",
            # sparsity_loss_type=("recip", 0.2),
            # sparsity_loss_type="hoyer",
            sparsity_loss_type="l1",
            # sparsity_coefficient=2e-4,
            sparsity_coefficient=1.6e-4,
            # sparsity_coefficient=7.5e-5,
            # sparsity_coefficient=1e-5,
            # sparsity_coefficient=3e-5,
            # sparsity_coefficient=2e-3,
            batch_size=batch_size,
            expansion_factor=32,
            use_encoder_bias=True,
            remove_decoder_bias=restore,
            encoder_init_method="kaiming",
            decoder_init_method="pseudoinverse",
            decoder_bias_init_method="zeros",
            # decoder_bias_init_method="geom_median" if not restore else "zeros",
            reconstruction_loss_type="mse_batchnorm",
            # project_updates_from_dec=False,
            project_updates_from_dec=True,
            # death_loss_type="sparsity_threshold",
            # death_loss_type="ghost_grads",
            death_loss_type="dm_ghost_grads",
            # death_loss_type="none",
            death_penalty_threshold=1e-5,
            death_penalty_coefficient=1,
            dead_after=500,
            # resample_every=2000,
            # resample_type="sample_inputs",
            restrict_dec_norm="exact",
            sparsity_tracking_epsilon=0.05,
        ),
        sae_restore=None if not restore else f"weights/jb-gpt2s-{layer}.safetensors",
        cache_every_steps=int(cache_size / batch_size / 2),
        cache_batch_size=cache_batch_size,
        cache_acc=int(cache_size / cache_batch_size / max_seq_len),
        buffer_max_samples=cache_size,
        # buffer_max_samples=0,
        model_config=TransformersModelConfig(
            # model_name_or_path="openai-community/gpt2-xl",
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
        loss_batch_size=16,
        eval_loss_every=512,
        buffer_dtype=jnp.float32,
        save_buffer="weights/buffer.safetensors",
        use_devices=n_devices
    )
    trainer = BufferTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
