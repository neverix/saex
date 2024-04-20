import os
from dataclasses import dataclass
from typing import Dict, Literal, NamedTuple, Tuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import safetensors
from jax.experimental.checkify import checkify
from jaxtyping import Array, Float, PyTree
from safetensors.flax import save_file

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
    encoder_init_method: Literal["kaiming", "orthogonal"] = "kaiming"
    decoder_init_method: Literal["kaiming", "orthogonal", "pseudoinverse"] = "kaiming"
    decoder_bias_init_method: Literal["zeros", "mean", "geom_median"] = "geom_median"
    
    project_updates_from_dec: bool = True
    restrict_dec_norm: Literal["none", "exact", "lte"] = "exact"
    
    sparsity_loss_type: Union[Literal["l1", "l1_sqrt", "tanh", "hoyer"], Tuple[Literal["recip"], float]] = "l1"
    reconstruction_loss_type: Literal[None, "mse", "mse_batchnorm", "l1"] = "mse"
    
    death_loss_type: Literal["none", "ghost_grads", "sparsity_threshold", "dm_ghost_grads"] = "none"
    dead_after: int = 50
    death_penalty_threshold: float = 1e-5
    death_penalty_coefficient: float = 1.0
    resample_every: int = 1e10
    resample_type: Literal["boom", "sample_inputs"] = "sample_inputs"
    
    sparsity_tracking_epsilon: float = 0.05

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
        if config.encoder_init_method == "kaiming":
            self.W_enc = jax.nn.initializers.kaiming_uniform()(w_enc_subkey, (config.n_dimensions, self.d_hidden))
        elif config.encoder_init_method == "orthogonal":
            self.W_enc = jax.nn.initializers.orthogonal()(w_enc_subkey, (config.n_dimensions, self.d_hidden))
        else:
            raise ValueError(f"Unknown encoder init method: \"{config.encoder_init_method}\"")
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
            raise ValueError(f"Unknown decoder init method: \"{config.decoder_init_method}\"")

        self.time_since_fired = eqx.nn.StateIndex(jnp.zeros((self.d_hidden,)))
        self.num_steps = eqx.nn.StateIndex(jnp.array(0))
        self.avg_loss_sparsity = eqx.nn.StateIndex(jnp.zeros((self.d_hidden,)))
        self.avg_l0 = eqx.nn.StateIndex(jnp.zeros((self.d_hidden,)))

    def forward(self, activations: jax.Array):
        activations = activations.astype(jnp.float32)
        inputs = activations
        if self.config.remove_decoder_bias:
            inputs = inputs - self.b_dec
        pre_relu = inputs @ self.W_enc
        if self.config.use_encoder_bias:
            pre_relu = pre_relu + self.b_enc
        post_relu = jax.nn.relu(pre_relu)
        hidden = post_relu * self.s
        out = hidden @ self.W_dec + self.b_dec
        return out

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
                                sparsity_loss.mean(0) * self.config.sparsity_tracking_epsilon
                                + state.get(self.avg_loss_sparsity) * (1 - self.config.sparsity_tracking_epsilon))
            state = state.set(self.avg_l0,
                                active.mean(0) * self.config.sparsity_tracking_epsilon
                                + state.get(self.avg_l0) * (1 - self.config.sparsity_tracking_epsilon))
        loss = reconstruction_loss.mean() + self.config.sparsity_coefficient * sparsity_loss.sum(-1).mean()
        losses = {"reconstruction": reconstruction_loss, "sparsity": sparsity_loss}
        if state is not None:  # we can only tell if a neuron is dead if we know if it was alive in the first place
            death_loss = self.compute_death_loss(state, activations, out, pre_relu)
            losses = {**losses, "death": death_loss}
            loss = loss + death_loss.mean() * self.config.death_penalty_coefficient
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
        if isinstance(self.config.sparsity_loss_type, tuple):
            assert self.config.sparsity_loss_type[0] == "recip"
            return activations / (activations + self.config.sparsity_loss_type[1])
        if self.config.sparsity_loss_type == "l1":
            return jnp.abs(activations)
        elif self.config.sparsity_loss_type == "hoyer":
            # every other loss is per-dimension
            # can't use sparsity_threshold
            return (jnp.square(jnp.abs(activations).sum(-1)) / jnp.square(activations).sum(-1))[:, None]
        elif self.config.sparsity_loss_type == "l1_sqrt":
            return jnp.sqrt(jnp.abs(activations))
        elif self.config.sparsity_loss_type == "tanh":
            return jnp.tanh(activations)
        else:
            raise ValueError(f"Unknown sparsity loss type: \"{self.config.sparsity_loss_type}\"")
    
    def reconstruction_loss(self, output, target, eps=1e-6):
        if self.config.reconstruction_loss_type == "mse":
            return (output - target) ** 2
        elif self.config.reconstruction_loss_type == "mse_batchnorm":
            return ((output - target) ** 2) / (
                jnp.linalg.norm((target - target.mean(0, keepdims=True)), axis=-1, keepdims=True) + eps)
        elif self.config.reconstruction_loss_type == "l1":
            return jnp.abs(output - target)
        else:
            raise ValueError(f"Unknown reconstruction loss type: \"{self.config.reconstruction_loss_type}\"")
    
    def compute_death_loss(self, state, activations, reconstructions, pre_relu, eps=1e-3):
        if self.config.death_loss_type == "none":
            return jnp.zeros(reconstructions.shape[:-1])
        elif self.config.death_loss_type == "sparsity_threshold":
            # eps = 1e-5
            post_relu = jax.nn.relu(pre_relu)
            sparsity = self.sparsity_loss(post_relu).mean(0)
            # log_sparsity = jnp.nan_to_num(jnp.log10(sparsity + eps))
            # offset = jax.lax.stop_gradient(jnp.log10(state.get(self.avg_loss_sparsity) + eps) - log_sparsity)
            offset = 0
            # return (jax.nn.relu(jnp.log10(self.config.death_penalty_threshold + eps)) - (log_sparsity + offset)).sum(-1)
            return (jax.nn.relu(self.config.death_penalty_threshold - (sparsity + offset)).sum(-1) / self.config.death_penalty_threshold)
        elif self.config.death_loss_type == "ghost_grads":
            dead = state.get(self.time_since_fired) > self.config.dead_after
            post_exp = jnp.where(dead, jnp.exp(pre_relu) * self.s, 0)
            ghost_recon = post_exp @ self.W_dec
            
            residual = activations - reconstructions
            ghost_norm = jnp.linalg.norm(ghost_recon, axis=-1)
            diff_norm = jnp.linalg.norm(residual, axis=-1)
            ghost_recon = ghost_recon * jax.lax.stop_gradient(diff_norm / (ghost_norm * 2 + eps))[:, None]
            
            ghost_loss = self.reconstruction_loss(ghost_recon, jax.lax.stop_gradient(residual))
            recon_loss = self.reconstruction_loss(reconstructions, activations)
            ghost_loss = ghost_loss * jax.lax.stop_gradient(recon_loss / (ghost_loss + eps))
            
            return jax.lax.select(dead.any(), ghost_loss, jnp.zeros_like(ghost_loss)).mean(-1)
        elif self.config.death_loss_type == "dm_ghost_grads":
            dead = state.get(self.time_since_fired) > self.config.dead_after
            post_exp = jnp.exp(pre_relu) * self.s
            ghost_recon = jnp.nan_to_num(post_exp @ self.W_dec)
            
            residual = jax.lax.stop_gradient(activations - reconstructions)
            ghost_norm = jnp.linalg.norm(ghost_recon, axis=-1)
            diff_norm = jnp.linalg.norm(residual, axis=-1)
            ghost_recon = ghost_recon * jnp.nan_to_num(jax.lax.stop_gradient(diff_norm / (ghost_norm * 2 + eps))[:, None])
            
            # ghost_loss = self.reconstruction_loss(ghost_recon, residual)
            ghost_loss = (ghost_recon - residual) ** 2
            # recon_loss = self.reconstruction_loss(reconstructions, activations)
            # ghost_loss = ghost_loss * jax.lax.stop_gradient(recon_loss / (ghost_loss + eps))
            
            return (ghost_loss).mean(-1) * jnp.mean(dead.astype(jnp.float32))
        else:
            raise ValueError(f"Unknown death loss type: \"{self.config.death_loss_type}\"")

    def apply_updates(self, updates: PyTree[Float],
                      state: eqx.nn.State,
                      opt_state,
                      last_input: Float[Array, "b f"],
                      last_output: SAEOutput,
                      step: int,
                      key: jax.random.PRNGKey):
        # assumes adam is the 0th gradient processor
        get_adam = lambda s: s[0][0]
        
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
        def adjust_mean(b_dec, opt_state):
            if self.config.decoder_bias_init_method == "mean":
                b_dec = b_dec + jnp.mean(last_input - last_output.output, axis=0)
            elif self.config.decoder_bias_init_method == "geom_median":
                b_dec = b_dec + geometric_median(last_input - last_output.output)
            opt_state = eqx.tree_at(lambda s: get_adam(s).mu.b_dec, opt_state, jnp.zeros_like(get_adam(opt_state).mu.b_dec))            
            opt_state = eqx.tree_at(lambda s: get_adam(s).nu.b_dec, opt_state, jnp.zeros_like(get_adam(opt_state).nu.b_dec))           
            return b_dec, opt_state
        updated_b_dec, opt_state = jax.lax.switch(jnp.astype(step == 1, jnp.int32), (lambda *a: a, adjust_mean), updated.b_dec, opt_state)
        updated = eqx.tree_at(lambda s: s.b_dec, updated, updated_b_dec)
        
        def resample(updated, state, opt_state):
            dead = state.get(self.time_since_fired) > self.config.dead_after
            if self.config.resample_type == "boom":
                W_enc = jnp.where(dead[None, :],
                                jax.nn.initializers.orthogonal()(utils.get_key(), updated.W_enc.shape),
                                updated.W_enc)
                b_enc = jnp.where(dead, self.config.encoder_bias_init_mean, updated.b_enc)
                W_dec = jnp.where(dead[:, None],
                                W_enc.T,
                                updated.W_dec)
            elif self.config.resample_type == "sample_inputs":
                # https://github.com/saprmarks/dictionary_learning/blob/main/training.py#L105
                alive_norm = jnp.linalg.norm(updated.W_enc * (~dead[None, :]), axis=-1).mean()
                sampled_vecs = jax.random.choice(key, last_input, (self.d_hidden,), replace=True, axis=0)
                norm_vecs = sampled_vecs / jnp.linalg.norm(sampled_vecs, axis=-1, keepdims=True)
                W_enc = jnp.where(dead[None, :],
                                  (sampled_vecs * alive_norm * 0.2).T,
                                  updated.W_enc)
                W_dec = jnp.where(dead[:, None],
                                  norm_vecs,
                                  updated.W_dec)
                b_enc = jnp.where(dead, 0, updated.b_enc)
            updated = eqx.tree_at(lambda s: s.W_enc, updated, W_enc)
            if self.config.use_encoder_bias:
                updated = eqx.tree_at(lambda s: s.b_enc, updated, b_enc)
            updated = eqx.tree_at(lambda s: s.W_dec, updated, W_dec)
            updated = eqx.tree_at(lambda s: s.s, updated, jnp.where(dead, 1, updated.s))
            
            # reset momentum and variance
            adam = get_adam(opt_state)
    
            opt_state = eqx.tree_at(lambda s: get_adam(s).mu.W_enc, opt_state, jnp.where(dead[None, :], 0, adam.mu.W_enc))
            opt_state = eqx.tree_at(lambda s: get_adam(s).mu.b_enc, opt_state, jnp.where(dead, 0, adam.mu.b_enc))
            # opt_state = eqx.tree_at(lambda s: s[adam_idx].mu.W_dec, opt_state, jnp.where(dead[:, None], 0, opt_state[adam_idx].mu.W_dec))

            opt_state = eqx.tree_at(lambda s: get_adam(s).nu.W_enc, opt_state, jnp.where(dead[None, :], 0, adam.nu.W_enc))
            opt_state = eqx.tree_at(lambda s: get_adam(s).nu.b_enc, opt_state, jnp.where(dead, 0, adam.nu.b_enc))
            # opt_state = eqx.tree_at(lambda s: s[adam_idx].nu.W_dec, opt_state, jnp.where(dead[:, None], 0, opt_state[adam_idx].nu.W_dec))

            state = state.set(self.time_since_fired, jnp.where(dead, 0, state.get(self.time_since_fired)))
            return updated, state, opt_state
        updated_params, updated_static = eqx.partition(updated, eqx.is_array)
        updated_params, state, opt_state = jax.lax.switch(
            jnp.astype(step % self.config.resample_every == self.config.resample_every - 1, jnp.int32),
            (lambda *a: a, resample),
            updated_params, state, opt_state)
        updated = eqx.combine(updated_params, updated_static)

        return updated, state, opt_state
    
    def normalize_w_dec(self, w_dec, eps=1e-6):
        if self.config.restrict_dec_norm == "exact":
            return w_dec / (eps + jnp.linalg.norm(w_dec, axis=-1, keepdims=True))
        elif self.config.restrict_dec_norm == "lte":
            return w_dec / jnp.maximum(1, eps + jnp.linalg.norm(w_dec, axis=-1, keepdims=True))
        return w_dec
        
    
    def get_stats(self, state: eqx.nn.State, last_input: jax.Array, last_output: SAEOutput, eps=1e-12):
        num_steps = state.get(self.num_steps)
        checkify(num_steps > 0, "num_steps must be positive")
        bias_corr = 1 - (1 - self.config.sparsity_tracking_epsilon) ** num_steps
        time_since_fired = state.get(self.time_since_fired)
        return dict(
            loss=last_output.loss,
            loss_sparsity=last_output.losses["sparsity"].sum(-1).mean(),
            loss_reconstruction=last_output.losses["reconstruction"].mean(),
            loss_death=jnp.mean(last_output.losses.get("death", 0)),
            # l0=(last_output.activations["pre_relu"] > 0).sum(-1).mean(),
            l0=(state.get(self.avg_l0) / bias_corr).sum(),
            dead=(time_since_fired > self.config.dead_after).mean(),
            var_explained=jnp.square(((last_input - last_input.mean(axis=0)) / (last_input.std(axis=0) + eps)
                                      * (last_output.output - last_output.output.mean(axis=0)) / (last_output.output.std(axis=0) + eps)
                                      ).mean(0)).mean(),
            max_time_since_fired=time_since_fired.max(),
        )

    def restore(self, weights_path: os.PathLike):
        with safetensors.safe_open(weights_path, "flax") as f:
            self = eqx.tree_at(lambda s: s.W_enc, self, f.get_tensor("W_enc"))
            self = eqx.tree_at(lambda s: s.b_enc, self, f.get_tensor("b_enc"))
            try:
                self = eqx.tree_at(lambda s: s.s, self, f.get_tensor("scaling_factor"))
            except safetensors.SafetensorError:
                pass
            self = eqx.tree_at(lambda s: s.W_dec, self, f.get_tensor("W_dec"))
            self = eqx.tree_at(lambda s: s.b_dec, self, f.get_tensor("b_dec"))
        return self

    def save(self, weights_path: os.PathLike, save_dtype: jax.typing.DTypeLike = jnp.float16):
        state_dict = {
            "W_enc": self.W_enc,
            "b_enc": self.b_enc,
            "scaling_factor": self.s,
            "W_dec": self.W_dec,
            "b_dec": self.b_dec,
        }
        state_dict = {k: v.astype(save_dtype) for k, v in state_dict.items()}
        save_file(state_dict, weights_path)
