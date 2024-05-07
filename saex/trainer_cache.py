import json
from dataclasses import dataclass, is_dataclass
from functools import partial
from typing import Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.sharding as jshard
import jax_smi
import numpy as np
import optax
from jax.sharding import PartitionSpec as P
from tqdm.auto import trange

import wandb

from . import utils
from .buffer import ActivationBuffer
from .iterable_dataset import IterableDatasetConfig, create_iterable_dataset
from .models.transformers_model import TransformersModelConfig
from .sae import SAE, SAEConfig


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
    restore_buffer: Optional[str]
    
    use_devices: int
    mp_devices: int


class BufferTrainer(object):
    def __init__(self, config: BufferTrainerConfig, sae=None, model=None, create_dataset=None):
        self.config = config

        mesh = None

        if model is None:
            print("Loading model...")
            if config.model_config.model_class.has_mesh:
                model = config.model_config.model_class(config.model_config)
                mesh = model.mesh
            else:
                mesh = jshard.Mesh(np.array(jax.devices())[:config.use_devices].reshape(
                    -1, config.mp_devices), axis_names=("dp", "mp"))
                model = config.model_config.model_class(config.model_config, mesh=mesh)
        self.model = model
        self.mesh = mesh

        if self.config.cache_acc == 0:
            print("Skipping buffer creation because there's no cache accumulation")
            self.buffer = None
        else:
            print("Creating buffer...")
            self.buffer, self.buffer_state = eqx.nn.make_with_state(ActivationBuffer)(
                config.buffer_max_samples, config.n_dimensions,
                dtype=config.buffer_dtype, mesh=self.mesh)
            if config.restore_buffer:
                print(f"Loading buffer ({config.restore_buffer})...")
                self.buffer_state = self.buffer.restore(
                    self.buffer_state, config.restore_buffer)
        if sae is not None:
            self.sae, self.sae_state = utils.unstatify(sae)
        else:
            print("Creating SAE...")
            self.sae, self.sae_state = eqx.nn.make_with_state(SAE)(config.sae_config, self.mesh)
            if config.sae_restore:
                print(f"Loading checkpoint ({config.sae_restore})...")
                self.sae = self.sae.restore(config.sae_restore)
            sharding = {k: jshard.NamedSharding(self.mesh, v) for k, v in self.sae.get_partition_spec()[0].items()}
            sae_params, _ = eqx.partition(self.sae, lambda x: eqx.is_array(x))
            self.sharding_sae = jax.tree_util.tree_map_with_path(lambda path, x: sharding.get(path[0].name), sae_params)
        
        if create_dataset is None:
            print("Loading dataset...")
            create_dataset = create_iterable_dataset(config.dataset_config)
        self.create_dataset = create_dataset

    def train(self):
        print("Training for", self.config.train_iterations, "iterations")
        print("Sparsity coefficient:", self.config.sae_config.sparsity_coefficient)
        
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
        
        is_trainable = lambda value: eqx.is_array(value) and value.dtype.kind in ("f", "V")
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
            sae_params = eqx.filter_shard(sae_params, self.sharding_sae)
            # SAE state is pretty small and there's no quadratic scaling, so we don't need to shard it as hard
            # (I don't think equinox state can be sharded... StateIndex is not ordered, so tree_flatten won't work)
            # sae_state = eqx.filter_shard(sae_state, self.sharding_sae_state)
            opt_state = eqx.tree_at(lambda o: o[0][0].mu, opt_state, replace_fn=lambda x: eqx.filter_shard(x, self.sharding_sae))
            opt_state = eqx.tree_at(lambda o: o[0][0].nu, opt_state, replace_fn=lambda x: eqx.filter_shard(x, self.sharding_sae))
            
            batch = jax.lax.with_sharding_constraint(batch, jshard.NamedSharding(self.mesh, P("dp", None)))
            (_, (sae_output, sae_state)), grad = loss_fn(sae_params, sae_static, sae_state, batch)
            sae_params = eqx.filter_shard(sae_params, self.sharding_sae)
            sae = eqx.combine(sae_params, sae_static)
            stats = sae.get_stats(sae_state, batch, sae_output)
            if not self.config.no_update:
                updates, opt_state = optimizer.update(grad, opt_state, sae_params)
                sae, sae_state, opt_state = sae.apply_updates(updates, sae_state, opt_state,
                                                   batch, sae_output, step, key)
                sae_params, _ = eqx.partition(sae, is_trainable)
            sae_params = eqx.filter_shard(sae_params, self.sharding_sae)
            # sae_state = eqx.filter_shard(sae_state, self.sharding_sae_state)
            opt_state = eqx.tree_at(lambda o: o[0][0].mu, opt_state, replace_fn=lambda x: eqx.filter_shard(x, self.sharding_sae))
            opt_state = eqx.tree_at(lambda o: o[0][0].nu, opt_state, replace_fn=lambda x: eqx.filter_shard(x, self.sharding_sae))
            return sae_params, sae_state, opt_state, stats

        @partial(jax.jit, donate_argnums=(0,))
        def update_buffer(mid_buffer, activations, mask, accumulated):
            n_tokens = mask.sum()
            raw_tokens = activations.reshape(-1, activations.shape[-1]).astype(mid_buffer.dtype)
            mask = mask.flatten()
            indices = jnp.nonzero(mask, size=mask.size, fill_value=mask.size)
            update = raw_tokens[indices]
            update = jnp.where(mask[indices][:, None],
                                update,
                                jax.lax.dynamic_slice(raw_tokens, (accumulated, 0), (mask.size, raw_tokens.shape[-1])))
            mid_buffer = jax.lax.dynamic_update_slice(mid_buffer, update, (accumulated, 0))
            return mid_buffer, n_tokens

        bar = trange(self.config.train_iterations + self.config.dry_run_steps)
        tokens_processed = 0
        try:
            for iteration in bar:
                if (iteration % self.config.cache_every_steps == 0
                    or iteration < self.config.dry_run_steps
                    or self.buffer is None) and self.config.cache_acc > 0:
                    accumulated = 0
                    while accumulated < self.config.cache_acc * self.config.cache_batch_size * self.config.model_config.max_seq_len:
                        # cache more activations
                        texts = []
                        for _ in range(self.config.cache_batch_size):
                            texts.append(next(dataset_iterator))
                        activations, model_misc = self.model(texts)
                        raw_tokens = activations.reshape(-1, activations.shape[-1])
                        mask = model_misc.get("mask")
                        if self.buffer is None:
                            assert mask is None
                            n_tokens = raw_tokens.shape[0]
                        else:
                            if mask is None:
                                mask = jnp.ones(raw_tokens.shape[0], dtype=jnp.bool)
                            n_tokens = mask.sum()
                            self.buffer_state = self.buffer(raw_tokens, mask, self.buffer_state)

                        accumulated += n_tokens
                        tokens_processed += n_tokens
                
                if iteration < self.config.dry_run_steps:
                    bar.set_description("Caching activations")
                    continue
                bar.set_description("Training SAE")

                # train SAE
                if self.buffer is None:
                    batch = raw_tokens
                else:
                    key, subkey = jax.random.split(key, 2)
                    # self.buffer_state, batch = self.buffer.sample_batch(
                    #     self.buffer_state, self.config.sae_config.batch_size, subkey)
                    subkeys = jax.random.split(subkey, self.config.sae_config.batch_size)
                    subkeys = jax.device_put(subkeys.reshape(self.mesh.shape["dp"], -1, subkeys.shape[-1]),
                                             jshard.NamedSharding(self.mesh, P("dp", None, None)))
                    self.buffer_state, batch = jax.vmap(self.buffer.sample_batch,
                                                        in_axes=(None, 1), out_axes=(None, 1))(
                                                            self.buffer_state, subkeys)
                    batch = batch.reshape(-1, batch.shape[-1])
                
                batch = jax.device_put(batch, jshard.NamedSharding(self.mesh, P("dp", None)))
                
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

                # TODO: track in wandb:
                # - learning rate
                # - norm ratio
                
                if self.config.save_steps and iteration % self.config.save_steps == 0:
                    self.sae = eqx.combine(sae_params, sae_static)
                    self.sae.save(self.config.save_path)
        except KeyboardInterrupt:
            print("Exiting early...")
        if self.config.save_buffer:
            save_buffer = input("Save buffer? (y/N)")
            if save_buffer.lower() in ("y", "yes"):
                self.buffer.save(self.buffer_state, self.config.save_buffer)
        run.finish()

