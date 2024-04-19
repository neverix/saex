import os
from dataclasses import dataclass
from typing import Dict, Literal, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import safetensors
from safetensors.flax import save_file
from jax.experimental.checkify import checkify
from jaxtyping import Array, Float, PyTree

from . import utils
from .geometric_median import geometric_median


@dataclass(frozen=True)
class SAEConfig:
    n_dimensions: int
    sparsity_coefficient: float
    batch_size: int    

    expansion_factor: int = 32
    
    encoder_bias_init_mean: float = 0.0
    use_encoder_bias: bool = False
    remove_decoder_bias: bool = False
    decoder_init_method: Literal["kaiming", "orthogonal", "pseudoinverse"] = "kaiming"
    decoder_bias_init_method: Literal["zeros", "mean", "geom_median"] = "geom_median"
    
    project_updates_from_dec: bool = True
    restrict_dec_norm: Literal["none", "exact", "lte"] = "exact"
    
    sparsity_loss_type: Literal["l1", "l1_sqrt", "tanh"] = "l1"
    reconstruction_loss_type: Literal[None, "mse", "mse_batchnorm", "l1"] = "mse"
    
    use_ghost_grads: bool = False
    dead_after: int = 50
    
    stat_tracking_epsilon: float = 0.05

class SAEOutput(NamedTuple):
    losses: Dict[str, jax.Array]
    output: jax.Array
    loss: jax.Array
    activations: Dict[str, jax.Array]

class SAE(eqx.Module):
    config: SAEConfig
    d_hidden: int
    b_enc: jax.Array
    
    W_enc: jax.Array
    s: jax.Array
    W_dec: jax.Array
    b_dec: jax.Array
    
    time_since_fired: eqx.nn.StateIndex
    num_steps: eqx.nn.StateIndex
    avg_loss_sparsity: eqx.nn.StateIndex
    avg_l0: eqx.nn.StateIndex

    def __init__(self, config, key=None):
        if key is None:
            key = utils.get_key()
        key, w_enc_subkey, w_dec_subkey = jax.random.split(key, 3)
        self.config = config
        self.d_hidden = config.n_dimensions * config.expansion_factor
        self.W_enc = jax.random.normal(w_enc_subkey, (config.n_dimensions, self.d_hidden))
        self.b_enc = jnp.full((self.d_hidden,), config.encoder_bias_init_mean)
        self.s = jnp.ones((self.d_hidden,))
        self.b_dec = jnp.zeros((config.n_dimensions,))

        if config.decoder_init_method == "kaiming":
            self.W_dec = jax.nn.initializers.kaiming_uniform()(w_dec_subkey, (self.d_hidden, config.n_dimensions))
            self.W_dec = self.normalize_w_dec(self.W_dec)
        elif config.decoder_init_method == "orthogonal":
            self.W_dec = jax.nn.initializers.orthogonal()(w_dec_subkey, (self.d_hidden, config.n_dimensions))
            self.W_dec = self.normalize_w_dec(self.W_dec)
        elif config.decoder_init_method == "pseudoinverse":
            if config.restrict_dec_norm == "none":
                self.W_dec = jnp.linalg.pinv(self.W_enc)
            else:
                W_enc = self.W_enc
                for _ in range(5):
                    W_dec = jnp.linalg.pinv(W_enc)
                    W_dec = self.normalize_w_dec(W_dec)
                    W_enc = jnp.linalg.pinv(W_dec)
                self.W_enc, self.W_dec = W_enc, W_dec
        else:
            raise ValueError(f"Unknown decoder init method: {config.decoder_init_method}")

        self.time_since_fired = eqx.nn.StateIndex(jnp.zeros((self.d_hidden,)))
        self.num_steps = eqx.nn.StateIndex(jnp.array(0))
        self.avg_loss_sparsity = eqx.nn.StateIndex(jnp.zeros((self.d_hidden,)))
        self.avg_l0 = eqx.nn.StateIndex(jnp.zeros((self.d_hidden,)))

    def __call__(self, activations: jax.Array, state=None):
        activations = activations.astype(jnp.float32)
        inputs = activations
        if self.config.remove_decoder_bias:
            inputs = inputs - self.b_dec
        pre_relu = inputs @ self.W_enc
        if self.config.use_encoder_bias:
            pre_relu = pre_relu + self.b_enc
        active = pre_relu > 0
        if state is not None:
            state = state.set(self.time_since_fired,
                              jnp.where(active.any(axis=0), 0, state.get(self.time_since_fired) + 1))
            state = state.set(self.num_steps, state.get(self.num_steps) + 1)
        post_relu = jax.nn.relu(pre_relu)
        sparsity_loss = self.sparsity_loss(post_relu)
        hidden = post_relu * self.s
        out = hidden @ self.W_dec + self.b_dec
        reconstruction_loss = self.reconstruction_loss(out, activations)
        if state is not None:
            state = state.set(self.avg_loss_sparsity,
                                sparsity_loss.mean(0) * self.config.stat_tracking_epsilon
                                + state.get(self.avg_loss_sparsity) * (1 - self.config.stat_tracking_epsilon))
            state = state.set(self.avg_l0,
                                active.mean(0) * self.config.stat_tracking_epsilon
                                + state.get(self.avg_l0) * (1 - self.config.stat_tracking_epsilon))
        loss = reconstruction_loss.mean() + self.config.sparsity_coefficient * sparsity_loss.sum(-1).mean()
        losses = {"reconstruction": reconstruction_loss, "sparsity": sparsity_loss}
        if state is not None and self.config.use_ghost_grads:
            ghost_losses = self.compute_ghost_losses(state, activations, out, pre_relu)
            losses = {**losses, "ghost": ghost_losses}
            loss = loss + ghost_losses.mean()
        output = SAEOutput(
            output=out,
            loss=loss,
            losses=losses,
            activations={"pre_relu": pre_relu, "post_relu": post_relu, "hidden": hidden},
        )
        if state is None:
            return output 
        else:
            return output, state

    def sparsity_loss(self, activations):
        if self.config.sparsity_loss_type == "l1":
            return jnp.abs(activations)
        elif self.config.sparsity_loss_type == "l1_sqrt":
            return jnp.sqrt(jnp.abs(activations))
        elif self.config.sparsity_loss_type == "tanh":
            return jnp.tanh(activations)
        else:
            raise ValueError(f"Unknown sparsity_loss_type {self.config.sparsity_loss_type}")
    
    def reconstruction_loss(self, output, target, eps=1e-6):
        if self.config.reconstruction_loss_type == "mse":
            return (output - target) ** 2
        elif self.config.reconstruction_loss_type == "mse_batchnorm":
            return ((output - target) ** 2) / ((target - target.mean(0, keepdims=True)).norm(axis=-1, keepdims=True) + eps)
        elif self.config.reconstruction_loss_type == "l1":
            return jnp.abs(output - target)

    def apply_updates(self, updates: PyTree[Float],
                      state: eqx.nn.State,
                      last_input: Float[Array, "b f"],
                      last_output: SAEOutput,
                      step: int):
        if self.config.project_updates_from_dec:
            def project_away(W_dec_grad):
                return W_dec_grad - jnp.einsum("h f, h -> h f",
                                               self.W_dec,
                                               jnp.einsum("h f, h f -> h",
                                                          self.W_dec,
                                                          W_dec_grad))
            updates = eqx.tree_at(lambda update: update.W_dec, updates,
                                  replace_fn=project_away)
        updated = eqx.apply_updates(self, updates)
        
        w_dec_selector = lambda x: x.W_dec
        updated = eqx.tree_at(w_dec_selector, updated, replace_fn=self.normalize_w_dec)
        
        # at the end of our first step, compute mean
        def adjust_mean(b_dec):
            if self.config.decoder_bias_init_method == "mean":
                b_dec = b_dec + jnp.mean(last_input - last_output.output, axis=0)
            elif self.config.decoder_bias_init_method == "geom_median":
                b_dec = b_dec + geometric_median(last_input - last_output.output)
            return b_dec
        updated = eqx.tree_at(lambda s: s.b_dec, updated,
                              jax.lax.switch(jnp.astype(step == 1, jnp.int32), (adjust_mean, lambda x: x), updated.b_dec))
        
        return updated
    
    def normalize_w_dec(self, w_dec, eps=1e-6):
        if self.config.restrict_dec_norm == "exact":
            return w_dec / (eps + jnp.linalg.norm(w_dec, axis=-1, keepdims=True))
        elif self.config.restrict_dec_norm == "lte":
            return w_dec / jnp.maximum(1, eps + jnp.linalg.norm(w_dec, axis=-1, keepdims=True))
        return w_dec
    
    def compute_ghost_losses(self, state, activations, reconstructions, pre_relu, eps=1e-3):
        dead = state.get(self.time_since_fired) > self.config.dead_after
        post_exp = jnp.where(dead, 0, jnp.exp(pre_relu) * self.s)
        ghost_recon = post_exp @ self.W_dec
        
        ghost_norm = jnp.linalg.norm(ghost_recon, axis=-1)
        diff_norm = jnp.linalg.norm(activations - reconstructions, axis=-1)
        ghost_recon = ghost_recon * jax.lax.stop_gradient(diff_norm / (ghost_norm * 2 + eps))[:, None]
        
        ghost_loss = self.reconstruction_loss(ghost_recon, activations)
        recon_loss = self.reconstruction_loss(reconstructions, activations)
        ghost_loss = ghost_loss * jax.lax.stop_gradient(recon_loss / (ghost_loss + eps))
        
        return jax.lax.select(dead.any(), ghost_loss, jnp.zeros_like(ghost_loss))
        
    
    def get_stats(self, state: eqx.nn.State, last_input: jax.Array, last_output: SAEOutput):
        num_steps = state.get(self.num_steps)
        checkify(num_steps > 0, "num_steps must be positive")
        bias_corr = 1 - (1 - self.config.stat_tracking_epsilon) ** num_steps
        time_since_fired = state.get(self.time_since_fired)
        return dict(
            loss=last_output.loss,
            loss_sparsity=last_output.losses["sparsity"].sum(-1).mean(),
            loss_reconstruction=last_output.losses["reconstruction"].mean(),
            l0=(last_output.activations["pre_relu"] > 0).sum(-1).mean(),
            # l0=(state.get(self.avg_l0) / bias_corr).sum(),
            dead=(time_since_fired > self.config.dead_after).mean(),
            var_explained=jnp.square(((last_input - last_input.mean(axis=0)) / last_input.std(axis=0)
                                      * (last_output.output - last_output.output.mean(axis=0)) / last_output.output.std(axis=0)
                                      ).mean(0)).mean(),
            max_time_since_fired=time_since_fired.max(),
        )

    def restore(self, weights_path: os.PathLike):
        with safetensors.safe_open(weights_path, "flax") as f:
            self = eqx.tree_at(lambda s: s.W_enc, self, f.get_tensor("W_enc"))
            self = eqx.tree_at(lambda s: s.b_enc, self, f.get_tensor("b_enc"))
            self = eqx.tree_at(lambda s: s.W_dec, self, f.get_tensor("W_dec"))
            self = eqx.tree_at(lambda s: s.b_dec, self, f.get_tensor("b_dec"))
        return self

    def save(self, weights_path: os.PathLike):
        save_file({
            "W_enc": self.W_enc,
            "b_enc": self.b_enc,
            "W_dec": self.W_dec * self.s[:, None],
            "b_dec": self.b_dec,
        }, weights_path)
