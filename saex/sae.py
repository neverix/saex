import random

import equinox as eqx
import jax
import jax.numpy as jnp

from typing import Literal, Dict
from dataclasses import dataclass
from collections import namedtuple


@dataclass
class SAEConfig:
    n_dimensions: int
    sparsity_coefficient: float
    batch_size: int    

    expansion_factor: int = 32
    decoder_init_method: Literal["random", "orthogonal", "pseudoinverse"] = "random"
    dec_bias_init_steps: int = 100
    sparsity_loss_type: Literal["l1", "l1_sqrt", "tanh"] = "l1"
    reconstruction_loss_type: Literal[None, "mse", "mse_batchnorm", "l1"] = "mse"
    project_updates_from_dec: bool = True
    restrict_dec_norm: Literal["none", "exact", "lte"] = "exact"

class SAEOutput(namedtuple):
    output: jax.Array
    loss: jax.Array
    losses: Dict[str, jax.Array]
    activations: Dict[str, jax.Array]
    state: eqx.nn.State


class SAE(eqx.Module):
    config: SAEConfig
    W_enc: jax.Array
    b_enc: jax.Array
    s: jax.Array
    W_dec: jax.Array
    b_dec: jax.Array
    time_since_fired: eqx.nn.StateIndex
    num_steps: eqx.nn.StateIndex
    last_input: eqx.nn.StateIndex

    def __init__(self, config, key=None):
        if key is None:
            key = jax.random.PRNGKey(0)
        key, w_enc_subkey, w_dec_subkey = jax.random.split(key, 3)
        self.config = config
        self.d_hidden = config.n_dimensions * config.expansion_factor
        self.W_enc = jax.random.normal(w_enc_subkey, (self.d_hidden, config.n_dimensions))
        self.b_enc = jnp.zeros((self.d_hidden,))
        self.s = jnp.ones((self.d_hidden,))
        self.b_dec = jnp.zeros((config.n_dimensions,))

        if config.decoder_init_method == "random":
            self.W_dec = jax.random.normal(w_dec_subkey, (self.d_hidden, config.n_dimensions))
        elif config.decoder.init_method == "orthogonal":
            self.W_dec = jax.nn.initializers.orthogonal()(w_dec_subkey, (self.d_hidden, config.n_dimensions))
        else:
            self.W_dec = jnp.linalg.pinv(self.W_enc)
        
        self.time_since_fired = eqx.nn.StateIndex(jnp.zeros((config.expansion_factor,)))
        self.num_steps = eqx.nn.StateIndex(0)
        self.last_input = eqx.nn.StateIndex(jnp.empty((config.batch_size, config.n_dimensions,)))

    def __call__(self, activations, state=None):
        pre_relu = activations @ self.W_enc + self.b_enc
        active = pre_relu > 0
        if state is not None:
            state = state.set(self.time_since_fired,
                              jnp.where(active, 0, state.get(self.time_since_fired) + 1))
            state = state.set(self.num_steps, state.get(self.num_steps) + 1)
        post_relu = jax.nn.relu(pre_relu)
        sparsity_loss = self.sparsity_loss(post_relu)
        hidden = post_relu * self.s
        out = hidden @ self.W_dec + self.b_dec
        reconstruction_loss = self.reconstruction_loss(out, activations)
        loss = reconstruction_loss.mean() + self.config.sparsity_coefficient * sparsity_loss.mean()
        output = SAEOutput(
            output=out,
            loss=loss,
            losses={"reconstruction": reconstruction_loss, "sparsity": sparsity_loss},
            activations={"pre_relu": pre_relu, "post_relu": post_relu, "hidden": hidden},
        )
        if state is None:
            return output, 
        else:
            return output, sparsity_loss, state

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

    def apply_updates(self, state, updates):
        if self.state.num_steps % self.config.dec_bias_init_steps == 0:
            updates = eqx.tree_at(lambda update: update.b_dec, updates,
                                  replace_fn=lambda b_dec_grad: b_dec_grad - b_dec_grad.mean())
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
        if self.config.restrict_dec_norm == "exact":
            updated = eqx.tree_at(w_dec_selector, updated, updated.W_dec / jnp.linalg.norm(updated.W_dec, axis=-1, keepdims=True))
        elif self.config.restrict_dec_norm == "lte":
            updated = eqx.tree_at(w_dec_selector, updated, updated.W_dec / jnp.maximum(1, jnp.linalg.norm(updated.W_dec, axis=-1, keepdims=True)))
        return updated

    def is_trainable(self, value):
        return eqx.is_array(value) and value.dtype.kind in "f"
