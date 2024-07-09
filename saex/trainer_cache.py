import json
from dataclasses import dataclass, is_dataclass
from functools import partial
from typing import Literal, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.sharding as jshard
import jax_smi
import numpy as np
import optax
from jax.sharding import PartitionSpec as P
from tqdm.auto import tqdm, trange
from micrlhf.adam_8bit import scale_by_adam_8bit

import wandb

from . import utils
from .buffer import ActivationBuffer
from .haver import ModelHaver, SAEHaver
from .iterable_dataset import IterableDatasetConfig, create_iterable_dataset
from .models.transformers_model import TransformersModelConfig
from .sae import SAE, SAEConfig


@dataclass
class BufferTrainerConfig:
    n_dimensions: int
    
    sae_config: SAEConfig
    model_config: TransformersModelConfig
    dataset_config: IterableDatasetConfig

    loss_batch_size: int = 16
    use_wandb: Optional[Tuple[str, str]] = None
    log_every: int = 1
    hist_every: int = 1
    eval_loss_every: int = 1
    log_texts: bool = False

    lr: float = 1e-3
    beta1: float = 0.0
    beta2: float = 0.99
    scheduler_warmup: int = 0
    scheduler_cycle: int = 1e6
    scheduler_multiply: float = 0.0
    ema: Optional[float] = None

    train_iterations: int = 1e6
    save_steps: Optional[int] = None
    save_path: Optional[str] = None
    dry_run_steps: int = 0
    no_update: bool = False

    sae_restore: Optional[str] = None
    restore_buffer: Optional[str] = None
    
    cache_batch_size: int = 16
    cache_every_steps: int = 1
    cache_acc: int = 1
    
    optimizer: Literal["adam", "adafactor", "adam8"] = "adam"

    buffer_max_samples: int = 0
    buffer_dtype: str = "float32"
    save_buffer: Optional[str] = None
    
    use_devices: int = 1
    mp_devices: int = 1
    is_distributed: bool = False
    
    push_to_hub: Optional[Tuple[str, str]] = None


class BufferCacher(ModelHaver):
    def __init__(self, config: BufferTrainerConfig, model=None, create_dataset=None):
        super().__init__(model_config=config.model_config,
                         model=model, create_dataset=create_dataset,
                         dataset_config=config.dataset_config,
                         use_devices=config.use_devices, mp_devices=config.mp_devices)
        self.config = config
        
        if self.config.cache_acc == 0:
            print("Skipping buffer creation because there's no cache accumulation")
            self.buffer = None
        else:
            print("Creating buffer...")
            self.buffer, self.buffer_state = eqx.nn.make_with_state(ActivationBuffer)(
                config.buffer_max_samples, config.n_dimensions,
                dtype=getattr(jnp, config.buffer_dtype), mesh=self.mesh)
            if config.restore_buffer:
                print(f"Loading buffer ({config.restore_buffer})...")
                self.buffer_state = self.buffer.restore(
                    self.buffer_state, config.restore_buffer)

    def __iter__(self):
        sampler = jax.jit(jax.vmap(self.buffer.sample_batch, in_axes=(None, 1), out_axes=(None, 1)), donate_argnums=(0,))

        dataset_iterator = iter(self.create_dataset())
        self.dataset_iterator = dataset_iterator
        bar = trange(self.config.train_iterations + self.config.dry_run_steps)
        tokens_processed = 0
        key = utils.get_key()
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
                    mask = model_misc.get("mask")
                    if self.buffer is None:
                        assert mask is None
                        n_tokens = np.prod(activations.shape[:-1])
                    else:
                        if mask is None:
                            mask = jnp.ones(np.prod(activations.shape[:-1]), dtype=jnp.bool,
                                            device=jshard.NamedSharding(self.mesh, P("dp")))
                        self.buffer_state, n_tokens = self.buffer(activations, mask, self.buffer_state)

                    accumulated += n_tokens
                    tokens_processed += n_tokens

                    if self.config.use_wandb and self.config.log_texts:
                        wandb.log({"texts": wandb.Table(columns=["text"], data=[[x] for x in texts])}, step=iteration)
            
            if iteration < self.config.dry_run_steps:
                bar.set_description("Caching activations")
                continue
            bar.set_description("Training SAE")

            # train SAE
            if self.buffer is None:
                batch = activations.reshape(-1, activations.shape[-1])
            else:
                key, subkey = jax.random.split(key, 2)
                subkeys = jax.random.split(subkey, self.config.sae_config.batch_size)
                subkeys = jax.device_put(subkeys.reshape(self.mesh.shape["dp"], -1, subkeys.shape[-1]),
                                        jshard.NamedSharding(self.mesh, P("dp", None, None)))
                self.buffer_state, batch = sampler(self.buffer_state, subkeys)
                batch = batch.reshape(-1, batch.shape[-1])

            bar.set_postfix({"tokens_processed": tokens_processed})
            if self.config.use_wandb:
                wandb.log({"tokens_processed": tokens_processed}, step=iteration)
            do_stop = yield batch
            if do_stop:
                break
        
        if self.config.save_buffer:
            save_buffer = input("Save buffer? (y/N)")
            if save_buffer.lower() in ("y", "yes"):
                self.buffer.save(self.buffer_state, self.config.save_buffer)

    def evaluate(self, sae):
        texts = []
        for _ in range(self.config.loss_batch_size):
            texts.append(next(self.dataset_iterator))
        return self.model.eval_loss(texts, sae)


class BufferTrainer(SAEHaver):
    def __init__(self, config: BufferTrainerConfig, mesh: Optional[jshard.Mesh] = None,
                 sae=None, evaluator=None):
        super().__init__(sae=sae, sae_restore=config.sae_restore, sae_config=config.sae_config, mesh=mesh)
        self.config = config
        self.evaluator = evaluator

    def train(self, wandb_suffix=0):
        print("Training for", self.config.train_iterations, "iterations")
        print("Sparsity coefficient:", self.config.sae_config.sparsity_coefficient)
        
        if self.config.use_wandb:
            entity, project = self.config.use_wandb

            def save_config(config, prefix=""):
                for k, v in config.items():
                    if isinstance(v, dict):
                        save_config(v, prefix + k + ".")
                    elif is_dataclass(v):
                        save_config(v.__dict__, prefix + k + ".")
                    else:
                        try:
                            wandb.config[prefix + k] = json.dumps(v)
                        except TypeError:
                            pass
            save_config(self.config.__dict__, prefix=f"{wandb_suffix}.")
        
        is_trainable = lambda value: eqx.is_array(value) and value.dtype.kind in ("f", "V")
        sae_params, sae_static = eqx.partition(self.sae, is_trainable)
        if self.config.ema:
            ema_params = jax.tree_map(lambda x: jnp.empty_like(x), sae_params)
        else:
            ema_params = None
        def get_final_params():
            if self.config.ema:
                return eqx.combine(ema_params, sae_static)
            else:
                return eqx.combine(sae_params, sae_static)
        
        key = utils.get_key()
        
        scheduler_cycle = self.config.scheduler_cycle
        if not scheduler_cycle:
            scheduler_cycle = self.config.train_iterations
        print("Learning rate:", self.config.lr, "warmed up for", self.config.scheduler_warmup, "iterations",
              "and cycled every", scheduler_cycle, "iterations")
        n_cycles = int(self.config.train_iterations / scheduler_cycle)
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),

            optax.adam(self.config.lr, b1=self.config.beta1, b2=self.config.beta2) if self.config.optimizer == "adam" else
            optax.adafactor(self.config.lr) if self.config.optimizer == "adafactor" else
            scale_by_adam_8bit(b1=self.config.beta1, b2=self.config.beta2) if self.config.optimizer == "adam8" else 1/0,

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
        def loss_fn(sae_params, sae_static, sae_state, batch, targets):
            sae = eqx.combine(sae_params, sae_static)
            sae_output, sae_state = sae(batch, targets, sae_state)
            return sae_output.loss, (sae_output, sae_state)

        @partial(jax.jit, donate_argnums=(2, 3, 4, 5), static_argnums=(6, 7))
        def train_step(
            batch, targets, sae_params, ema_params, sae_state, opt_state, sae_static, optimizer, step, key
        ):
            batch = jnp.nan_to_num(batch)
            targets = jnp.nan_to_num(targets)
            sae_params = eqx.filter_shard(sae_params, self.sharding_sae)
            if self.config.optimizer == "adam":
                # SAE state is pretty small and there's no quadratic scaling, so we don't need to shard it as hard
                # (I don't think equinox state can be sharded... StateIndex is not ordered, so tree_flatten won't work)
                # sae_state = eqx.filter_shard(sae_state, self.sharding_sae_state)
                opt_state = eqx.tree_at(lambda o: o[1][0].mu, opt_state, replace_fn=lambda x: eqx.filter_shard(x, self.sharding_sae))
                opt_state = eqx.tree_at(lambda o: o[1][0].nu, opt_state, replace_fn=lambda x: eqx.filter_shard(x, self.sharding_sae))
            
            batch = jax.lax.with_sharding_constraint(batch, jshard.NamedSharding(self.mesh, P("dp", None)))
            targets = jax.lax.with_sharding_constraint(targets, jshard.NamedSharding(self.mesh, P("dp", None)))
            (_, (sae_output, sae_state)), grad = loss_fn(sae_params, sae_static, sae_state, batch, targets)
            sae_params = eqx.filter_shard(sae_params, self.sharding_sae)
            sae = eqx.combine(sae_params, sae_static)
            stats = sae.get_stats(sae_state, batch, targets, sae_output)
            if not self.config.no_update:
                k1, k2 = jax.random.split(key)
                grad = sae.update_gradients(grad, sae_state, k1)
                updates, opt_state = optimizer.update(grad, opt_state, sae_params)
                sae, sae_state, opt_state = sae.apply_updates(updates, sae_state, opt_state,
                                                              batch, targets, sae_output, step, k2)
                sae_params, _ = eqx.partition(sae, is_trainable)
            sae_params = eqx.filter_shard(sae_params, self.sharding_sae)
            # sae_state = eqx.filter_shard(sae_state, self.sharding_sae_state)
            if self.config.optimizer == "adam":
                opt_state = eqx.tree_at(lambda o: o[1][0].mu, opt_state, replace_fn=lambda x: eqx.filter_shard(x, self.sharding_sae))
                opt_state = eqx.tree_at(lambda o: o[1][0].nu, opt_state, replace_fn=lambda x: eqx.filter_shard(x, self.sharding_sae))
            if self.config.ema:
                def ema_update(ema_params, sae_params):
                    return jax.tree.map(lambda ema, sae: ema * self.config.ema + sae * (1 - self.config.ema),
                                        ema_params, sae_params)
                ema_params = jax.lax.switch((step == 0).astype(np.int32),
                                            (ema_update, lambda _, y: y),
                                            ema_params, sae_params)
            return sae_params, ema_params, sae_state, opt_state, stats

        bar = tqdm()

        iteration = 0  # no clue why it has to start from 1 but it does
        while True:
            iteration += 1
            
            batch = yield
            if batch is None:
                print("stopped", wandb_suffix)
                break
            
            key, subkey = jax.random.split(key)
            sae_params, ema_params, self.sae_state, opt_state, stats = train_step(
                batch, batch, sae_params, ema_params, self.sae_state, opt_state,
                sae_static, optimizer, iteration, subkey)

            if iteration % self.config.log_every == 0:
                bar.set_postfix(stats)
            if self.config.use_wandb:
                if iteration % self.config.log_every == 0:
                    bar.set_postfix(stats)
                    stats = {f"{k}/{wandb_suffix}": v for k, v in stats.items()}
                    wandb.log(stats, step=iteration)
                if iteration % self.config.hist_every == self.config.hist_every - 1:
                    # look at this graph
                    wandb.log({f"histogram/{wandb_suffix}": wandb.Histogram(self.sae.get_log_sparsity(self.sae_state))}, step=iteration)
                if iteration % self.config.eval_loss_every == self.config.eval_loss_every - 1:
                    final_params = get_final_params()
                    self.sae = eqx.combine(final_params, sae_static)
                    loss_clean, loss_reconstructed = self.evaluator.evaluate(self.sae)
                    wandb.log({f"loss_clean/{wandb_suffix}": loss_clean, "loss_reconstructed": loss_reconstructed,
                                f"recon_loss_diff/{wandb_suffix}": loss_reconstructed - loss_clean,
                                f"log_recon_loss_ratio/{wandb_suffix}": np.log(loss_reconstructed / loss_clean)}, step=iteration)

            # TODO: track in wandb:
            # - learning rate
            # - norm ratio
            
            if self.config.save_steps and iteration % self.config.save_steps == 0:
                final_params = get_final_params()
                self.sae = eqx.combine(final_params, sae_static)
                self.sae.save(self.config.save_path)

            bar.update()

        if self.config.push_to_hub:
            final_params = get_final_params()
            self.sae = eqx.combine(final_params, sae_static)
            self.sae.push_to_hub(*self.config.push_to_hub)
