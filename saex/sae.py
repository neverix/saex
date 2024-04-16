import random

import equinox as eqx
import jax
import jax.numpy as jnp

from typing import Literal
from dataclasses import dataclass


@dataclass
class SAEConfig:
    n_features: int
    expansion_factor: int = 32
    decoder_init_method: Literal["random", "orthogonal", "pseudoinverse"] = "random"
    project_updates_from_dec: bool = True
    restrict_dec_norm: Literal["none", "exact", "lte"] = "exact"

class SAE(eqx.Module):
    config: SAEConfig
    W_enc: jax.Array
    b_enc: jax.Array
    s: jax.Array
    W_dec: jax.Array
    b_dec: jax.Array
    time_since_fired: jax.Array
        
    def __init__(self, config, key=None):
        if key is None:
            key = jax.random.PRNGKey(0)
        key, w_enc_subkey, w_dec_subkey = jax.random.split(key, 3)
        self.config = config
        self.d_hidden = config.n_features * config.expansion_factor
        self.W_enc = jax.random.normal(w_enc_subkey, (config.n_features, self.d_hidden))
        self.b_enc = jnp.zeros((self.d_hidden,))
        self.s = jnp.ones((self.d_hidden,))
        self.b_dec = jnp.zeros((config.n_features,))

        if config.decoder_init_method == "random":
            self.W_dec = jax.random.normal(w_dec_subkey, (self.d_hidden, config.n_features))
        elif config.decoder.init_method == "orthogonal":
            self.W_dec = jax.nn.initializers.orthogonal()(w_dec_subkey, (self.d_hidden, config.n_features))
        else:
            self.W_dec = jnp.linalg.pinv(self.W_enc)
        
        self.time_since_fired = eqx.nn.StateIndex(jnp.zeros((config.expansion_factor,)))

    def __call__(self, activations, state=None):
        pre_relu = activations @ self.W_enc + self.b_enc
        active = pre_relu > 0
        if state is not None:
            state = state.set(self.time_since_fired,
                              jnp.where(active, 0, state.get(self.time_since_fired) + 1))
        hidden = jax.nn.relu(pre_relu) * self.s
        output = hidden @ self.W_dec + self.b_dec
        if state is None:
            return output
        else:
            return output, state

    def apply_updates(self, updates):
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
        if self.config.restrict_dec_norm == "exact":
            updated.W_dec = updated.W_dec / jnp.linalg.norm(updated.W_dec, axis=-1, keepdims=True)
        elif self.config.restrict_dec_norm == "lte":
            updated.W_dec = updated.W_dec / jnp.maximum(1, jnp.linalg.norm(updated.W_dec, axis=-1, keepdims=True))
        return updated
