from collections import namedtuple
from dataclasses import dataclass
from typing import Dict, Literal, NamedTuple

from functools import partial
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
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
    decoder_init_method: Literal["kaiming", "orthogonal", "pseudoinverse"] = "kaiming"
    decoder_bias_init_method: Literal["zeros", "mean", "geom_median"] = "geom_median"
    sparsity_loss_type: Literal["l1", "l1_sqrt", "tanh"] = "l1"
    reconstruction_loss_type: Literal[None, "mse", "mse_batchnorm", "l1"] = "mse"
    project_updates_from_dec: bool = True
    restrict_dec_norm: Literal["none", "exact", "lte"] = "exact"
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
                for _ in range(100):
                    self.W_dec = jnp.linalg.pinv(self.W_enc)
                    self.W_dec = self.normalize_w_dec(self.W_dec)
                    self.W_enc = jnp.linalg.pinv(self.W_dec)
        else:
            raise ValueError(f"Unknown decoder init method: {config.decoder_init_method}")
    
        self.time_since_fired = eqx.nn.StateIndex(jnp.zeros((self.d_hidden,)))
        self.num_steps = eqx.nn.StateIndex(jnp.array(0))
        self.avg_loss_sparsity = eqx.nn.StateIndex(jnp.zeros((self.d_hidden,)))
        self.avg_l0 = eqx.nn.StateIndex(jnp.zeros((self.d_hidden,)))

    def __call__(self, activations, state=None):
        pre_relu = activations @ self.W_enc
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
        output = SAEOutput(
            output=out,
            loss=loss,
            losses={"reconstruction": reconstruction_loss, "sparsity": sparsity_loss},
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
    
    def reconstruction_loss(self, output, target):
        if self.config.reconstruction_loss_type == "mse":
            return (output - target) ** 2
        elif self.config.reconstruction_loss_type == "mse_batchnorm":
            return ((output - target) ** 2) / (target - target.mean(0, keepdims=True)).norm(axis=-1).mean()
        elif self.config.reconstruction_loss_type == "l1":
            return jnp.abs(output - target)

    def apply_updates(self, updates: PyTree[Float], state: eqx.nn.State, last_input: Float[Array, "b f"], last_output: SAEOutput, step: int):
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
        
        # at the end of our first step
        if step == 1:
            if self.config.decoder_bias_init_method == "mean":
                updated = eqx.tree_at(lambda self: self.b_dec, updated, jnp.mean(last_input - last_output.output, axis=0))
            elif self.config.decoder_bias_init_method == "geom_median":
                updated = eqx.tree_at(lambda self: self.b_dec, updated, geometric_median(last_input - last_output.output))
        
        return updated
    
    def normalize_w_dec(self, w_dec):
        if self.config.restrict_dec_norm == "exact":
            return w_dec / jnp.linalg.norm(w_dec, axis=-1, keepdims=True)
        elif self.config.restrict_dec_norm == "lte":
            return w_dec / jnp.maximum(1, jnp.linalg.norm(w_dec, axis=-1, keepdims=True))
        return w_dec
        
    
    def get_stats(self, state: eqx.nn.State, last_input: jax.Array, last_output: SAEOutput):
        num_steps = state.get(self.num_steps)
        assert num_steps > 0
        bias_corr = 1 - (1 - self.config.stat_tracking_epsilon) ** num_steps
        time_since_fired = np.asarray(state.get(self.time_since_fired))
        return dict(
            loss=last_output.loss,
            loss_sparsity=float(last_output.losses["sparsity"].sum(-1).mean()),
            loss_reconstruction=float(last_output.losses["reconstruction"].mean()),
            l0=(state.get(self.avg_l0) / bias_corr).sum(),
            max_time_since_fired=int(time_since_fired.max()),
            pct_dead=float((time_since_fired > 100).mean() * 100),
            pct_var_explained=float(((last_output.output - last_input).var(0) / last_input.var(0)).mean() * 100)
        )
