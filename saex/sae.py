import json
import os
from dataclasses import dataclass
from tempfile import NamedTemporaryFile
from typing import Dict, Literal, NamedTuple, Optional, Tuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.sharding as jshard
import jmp
import numpy as np
import safetensors
from jax.experimental.checkify import checkify
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, PyTree
from safetensors.flax import save_file
import aqt.jax.v2.config as aqt_config
import aqt.jax.v2.aqt_dot_general as aqt

from . import utils
from .utils.geometric_median import geometric_median


@dataclass(frozen=True)
class SAEConfig:
    n_dimensions: int
    batch_size: int
    sparsity_coefficient: float = 1
    buffer_size: int = 100

    expansion_factor: float = 32
    topk_k: Optional[int] = None
    topk_approx: Optional[float] = None
 
    encoder_bias_init_mean: float = 0.0
    use_encoder_bias: bool = False
    remove_decoder_bias: bool = False
    encoder_init_method: Literal["kaiming", "orthogonal"] = "kaiming"
    decoder_init_method: Literal["kaiming", "orthogonal", "pseudoinverse"] = "kaiming"
    decoder_bias_init_method: Literal["zeros", "mean", "geom_median"] = "geom_median"
    
    norm_input: Literal[None, "wes", "wes-mean", "wes-clip", "wes-mean-fixed"] = None
    wes_clip: tuple[float, float] = (1e-2, 100)
    wes_ema: float = 0.99
    min_norm: float = 1.0
    anthropic_norm: bool = False
    
    project_updates_from_dec: bool = True
    project_grads_from_dec: bool = False
    restrict_dec_norm: Literal["none", "exact", "lte"] = "exact"
    
    is_gated: bool = False
    
    sparsity_loss_type: Union[Literal["l1", "l1_sqrt", "tanh", "hoyer", "recip"]] = "l1"
    recip_schedule: Tuple[Tuple[int, float]] = ((10_000, 0.2), (20_000, 0.1), (100_000, 0.05))
    reconstruction_loss_type: Literal[None, "mse", "mse_batchnorm", "l1"] = "mse"
    
    death_loss_type: Literal["none", "ghost_grads", "sparsity_threshold", "dm_ghost_grads"] = "none"
    scale_ghost_by_death: bool = True
    dead_after: int = 50
    death_penalty_threshold: Optional[Union[Literal["auto"], float]] = None
    death_penalty_coefficient: float = 1.0
    resample_every: int = 1e10
    resample_type: Literal["boom", "sample_inputs"] = "sample_inputs"
    
    sparsity_tracking_epsilon: float = 0.05
    use_model_parallel: bool = True
    loss_scaling: float = 100.0
    param_dtype: str = "float32"
    bias_dtype: str = "float32"
    misc_dtype: str = "float32"
    weights_8bit: bool = False
    use_aqt: bool = False

class SAEOutput(NamedTuple):
    losses: Dict[str, jax.Array]
    output: jax.Array
    loss: jax.Array
    activations: Dict[str, jax.Array]

class SAE(eqx.Module):
    config: SAEConfig
    d_hidden: int
    
    sharding: Dict[str, jshard.NamedSharding]
    state_sharding: Dict[str, jshard.NamedSharding]
    
    b_enc: jax.Array
    
    W_enc: jax.Array
    s: jax.Array
    s_gate: jax.Array
    b_gate: jax.Array
    W_dec: jax.Array
    b_dec: jax.Array
    mean_norm: jax.Array
    tgt_mean_norm: jax.Array
    
    time_since_fired: eqx.nn.StateIndex
    num_steps: eqx.nn.StateIndex
    avg_loss_sparsity: eqx.nn.StateIndex
    avg_l0: eqx.nn.StateIndex
    activated_buffer: eqx.nn.StateIndex
    ds_mean_norm: eqx.nn.StateIndex
    ds_tgt_mean_norm: eqx.nn.StateIndex

    enc_config: Optional[aqt_config.DotGeneral]
    dec_config: Optional[aqt_config.DotGeneral]

    def __init__(self, config, mesh: jshard.Mesh, key=None):        
        if key is None:
            key = utils.get_key()
        key, w_enc_subkey, w_dec_subkey = jax.random.split(key, 3)
        self.config = config
        self.d_hidden = int(config.n_dimensions * config.expansion_factor)
        
        spec, state_spec = self.get_partition_spec()
        sharding = {k: jshard.NamedSharding(mesh, v) for k, v in spec.items()}
        state_sharding = {k: jshard.NamedSharding(mesh, v) for k, v in state_spec.items()}
        self.sharding = sharding
        self.state_sharding = state_sharding
        
        if config.encoder_init_method == "kaiming":
            self.W_enc = jax.nn.initializers.kaiming_uniform()(w_enc_subkey,
                                                               (config.n_dimensions, self.d_hidden),
                                                               dtype=self.param_dtype)
        elif config.encoder_init_method == "orthogonal":
            # self.W_enc = jax.nn.initializers.orthogonal()(w_enc_subkey,
            #                                               (config.n_dimensions, self.d_hidden),
            #                                               dtype=self.param_dtype)
            # # idk why, but it stopped working
            self.W_enc = jax.nn.initializers.orthogonal()(w_enc_subkey,
                                                          (config.n_dimensions, self.d_hidden)
                                                          ).astype(self.param_dtype)
        else:
            raise ValueError(f"Unknown encoder init method: \"{config.encoder_init_method}\"")
        self.W_enc = jax.device_put(self.W_enc, sharding["W_enc"])
        
        self.b_enc = jnp.full((self.d_hidden,), config.encoder_bias_init_mean,
                              device=sharding["b_enc"], dtype=self.bias_dtype)
        self.s = jnp.ones((self.d_hidden,), device=sharding["s"], dtype=self.misc_dtype)
        self.s_gate = jnp.zeros((self.d_hidden,), device=sharding["s"], dtype=self.misc_dtype)
        self.b_gate = jnp.zeros((self.d_hidden,), device=sharding["b_enc"], dtype=self.bias_dtype)
        self.b_dec = jnp.zeros((config.n_dimensions,), device=sharding["b_dec"], dtype=self.bias_dtype)

        if config.decoder_init_method == "kaiming":
            self.W_dec = jax.nn.initializers.kaiming_uniform()(w_dec_subkey,
                                                               (self.d_hidden, config.n_dimensions),
                                                               dtype=self.param_dtype)
            self.W_dec = jax.device_put(self.W_dec, sharding["W_dec"])
            self.W_dec = self.normalize_w_dec(self.W_dec)
        elif config.decoder_init_method == "orthogonal":
            self.W_dec = jax.nn.initializers.orthogonal()(w_dec_subkey,
                                                          (self.d_hidden, config.n_dimensions),
                                                          dtype=self.param_dtype)
            self.W_dec = jax.device_put(self.W_dec, sharding["W_dec"])
            self.W_dec = self.normalize_w_dec(self.W_dec)
        elif config.decoder_init_method == "pseudoinverse":
            if config.encoder_init_method == "orthogonal":
                self.W_dec = jax.device_put(self.normalize_w_dec(self.W_enc.T), sharding["W_dec"])
            else:
                if config.restrict_dec_norm == "none":
                    self.W_dec = jnp.linalg.pinv(self.W_enc)
                else:
                    W_enc = self.W_enc
                    # TODO parallelize?
                    for _ in range(5):
                        W_dec = jnp.linalg.pinv(W_enc)
                        W_dec = self.normalize_w_dec(W_dec)
                        W_enc = jnp.linalg.pinv(W_dec)
                    W_enc = jax.device_put(W_enc, sharding["W_enc"])
                    W_dec = jax.device_put(W_dec, sharding["W_dec"])
                    self.W_enc, self.W_dec = W_enc, W_dec
        else:
            raise ValueError(f"Unknown decoder init method: \"{config.decoder_init_method}\"")

        self.time_since_fired = eqx.nn.StateIndex(jnp.zeros((self.d_hidden,),
                                                            device=state_sharding["time_since_fired"]))
        self.num_steps = eqx.nn.StateIndex(jnp.array(0))
        self.avg_loss_sparsity = eqx.nn.StateIndex(jnp.zeros((self.d_hidden,),
                                                             device=state_sharding["avg_loss_sparsity"]))
        self.avg_l0 = eqx.nn.StateIndex(jnp.zeros((self.d_hidden,),
                                                  device=state_sharding["avg_l0"]))
        self.activated_buffer = eqx.nn.StateIndex(jnp.zeros((self.config.buffer_size, self.d_hidden),
                                                            device=state_sharding["activated_buffer"],
                                                            dtype=jnp.float32
                                                            ))
        self.ds_mean_norm = eqx.nn.StateIndex(jnp.array(1.0, dtype=self.mean_norm_dtype))
        self.ds_tgt_mean_norm = eqx.nn.StateIndex(jnp.array(1.0, dtype=self.mean_norm_dtype))
        self.mean_norm = jnp.array(1.0, dtype=self.mean_norm_dtype)
        self.tgt_mean_norm = jnp.array(1.0, dtype=self.mean_norm_dtype)
        
        if self.config.use_aqt:
            self.enc_config = aqt_config.fully_quantized(fwd_bits=8, bwd_bits=8)
            self.dec_config = aqt_config.fully_quantized(fwd_bits=8, bwd_bits=8)
        else:
            self.enc_config = None
            self.dec_config = None
    
    @property
    def param_dtype(self):
        return getattr(jnp, self.config.param_dtype)
    
    @property
    def mean_norm_dtype(self):
        # return self.param_dtype
        return jnp.float32
        # return jnp.float16
        # return jnp.bfloat16

    @property
    def misc_dtype(self):
        return getattr(jnp, self.config.misc_dtype)
    
    @property
    def bias_dtype(self):
        return getattr(jnp, self.config.bias_dtype)

    @property
    def scaler(self):
        return jmp.StaticLossScale(self.config.loss_scaling)
    
    @property
    def is_gated(self):
        return self.config.is_gated

    def dot_general(self, key):
        if key is None or not self.config.use_aqt:
            return None, None
        key1, key2 = jax.random.split(key)
        return aqt_config.set_context(self.enc_config, key1, None), aqt_config.set_context(self.dec_config, key2, None)

    def norm_input(self, x: jax.typing.ArrayLike):
        if self.config.norm_input == "wes":
            # Wes-style normalization
            norm_factor = (x.shape[-1] ** 0.5) / jnp.linalg.norm(x, axis=-1, keepdims=True)
        elif self.config.norm_input == "wes-mean":
            # batchnorm (bad)
            # jax.debug.print("{}", jnp.linalg.norm(x, axis=-1, keepdims=True).mean())
            norm_factor = (x.shape[-1] ** 0.5) / jnp.linalg.norm(x, axis=-1, keepdims=True).mean()
        elif self.config.norm_input == "wes-clip":
            norm_factor = (x.shape[-1] ** 0.5) / jnp.clip(jnp.linalg.norm(x, axis=-1, keepdims=True), *self.config.wes_clip)
        elif self.config.norm_input == "wes-mean-fixed":
            norm_factor = (x.shape[-1] ** 0.5) / self.mean_norm
        else:
            norm_factor = jnp.ones_like(x)
        return x * norm_factor, norm_factor

    def norm_target(self, x: jax.typing.ArrayLike):
        assert self.config.norm_input in (None, "wes-mean-fixed")
        if self.config.norm_input == "wes-mean-fixed":
            norm_factor = (x.shape[-1] ** 0.5) / self.tgt_mean_norm
        else:
            norm_factor = jnp.ones_like(x)
        return x * norm_factor, norm_factor

    def encode(self, activations: jax.Array, dot_enc=None):
        inputs, norm_factor = self.norm_input(activations.astype(jnp.float32))
        if self.config.remove_decoder_bias:
            inputs = inputs - self.b_dec
        if dot_enc is not None:
            pre_relu = aqt.einsum("ab,bc->ac", inputs.astype(self.W_enc), self.W_enc, dot_enc)
        else:
            pre_relu = inputs @ self.W_enc
        if self.config.use_encoder_bias:
            pre_relu = pre_relu + self.b_enc
        post_relu = jax.nn.relu(pre_relu)
        if self.config.topk_k is not None:
            og_shape = post_relu.shape
            post_relu = post_relu.reshape(-1, post_relu.shape[-1])
            if self.config.topk_approx is not None:
                values, indices = jax.lax.approx_max_k(post_relu, self.config.topk_k, aggregate_to_topk=True, recall_target=self.config.topk_approx)
            else:
                values, indices = jax.lax.top_k(post_relu, self.config.topk_k)
            post_relu = jax.vmap(lambda a, v, i: jnp.zeros_like(a).at[i].set(v))(post_relu, values, indices)
            post_relu = post_relu.reshape(og_shape)
        if self.config.is_gated:
            # hidden = (post_relu > 0) * ((inputs @ self.W_enc) * jax.nn.softplus(self.s_gate) * self.s + self.b_gate)
            hidden = (post_relu > 0) * jax.nn.relu((inputs @ self.W_enc) * jax.nn.softplus(self.s_gate) * self.s + self.b_gate)
        else:
            hidden = post_relu * self.s
        return inputs, pre_relu, post_relu, hidden, norm_factor

    def forward(self, activations: jax.Array):
        _, _, _, hidden, norm_factor = self.encode(activations)
        out = hidden @ self.W_dec + self.b_dec
        return out / (norm_factor if self.config.norm_input != "wes-mean-fixed" else ((out.shape[-1] ** 0.5) / (self.tgt_mean_norm)))

    def __call__(self, activations: jax.Array, targets: jax.typing.ArrayLike, key: jax.random.PRNGKey = None, state=None):
        targets, target_norm_factor = self.norm_target(targets)
        if self.config.norm_input == "wes-mean-fixed":
            mean_norm = jnp.maximum(self.config.min_norm, jnp.linalg.norm(activations.astype(self.mean_norm_dtype), axis=-1, keepdims=True).mean())
            target_mean_norm = jnp.maximum(self.config.min_norm, jnp.linalg.norm(targets.astype(self.mean_norm_dtype), axis=-1, keepdims=True).mean())
            if state is not None:
                # t = state.get(self.num_steps)
                # state = state.set(self.ds_mean_norm,
                #                   state.get(self.ds_mean_norm) * (t / (t + 1)) + mean_norm / (t + 1))
                state = state.set(self.ds_mean_norm, jnp.maximum(self.config.min_norm, state.get(self.ds_mean_norm) * self.config.wes_ema + mean_norm * (1 - self.config.wes_ema)).astype(self.mean_norm_dtype))
                state = state.set(self.ds_tgt_mean_norm, jnp.maximum(self.config.min_norm, state.get(self.ds_tgt_mean_norm) * self.config.wes_ema + target_mean_norm * (1 - self.config.wes_ema)).astype(self.mean_norm_dtype))

        dot_enc, dot_dec = self.dot_general(key)

        activations, pre_relu, post_relu, hidden, norm_factor = self.encode(activations, dot_enc=dot_enc)
        active = hidden != 0
        if state is not None:
            state = state.set(self.time_since_fired,
                              jnp.where(active.any(axis=0), 0, state.get(self.time_since_fired) + 1))
            state = state.set(self.num_steps, state.get(self.num_steps) + 1)
            state = state.set(self.avg_l0,
                                active.mean(0) * self.config.sparsity_tracking_epsilon
                                + state.get(self.avg_l0) * (1 - self.config.sparsity_tracking_epsilon))
            buffer = state.get(self.activated_buffer)
            buffer = jnp.roll(buffer, -1, 0)
            buffer = buffer.at[-1].set(active.astype(buffer.dtype).mean(0))
            state = state.set(self.activated_buffer, buffer)
        sparsity_loss = self.sparsity_loss(post_relu, state=state).astype(jnp.float32)
        if dot_dec is not None:
            decoded = aqt.einsum("ab,bc->ac", hidden.astype(self.W_dec), self.W_dec, dot_dec)
        else:
            decoded = hidden @ self.W_dec
        out = decoded + self.b_dec
        reconstruction_loss = self.reconstruction_loss(out, targets).astype(jnp.float32)
        if state is not None:
            state = state.set(self.avg_loss_sparsity,
                                sparsity_loss.mean(0) * self.config.sparsity_tracking_epsilon
                                + state.get(self.avg_loss_sparsity) * (1 - self.config.sparsity_tracking_epsilon))
        loss = reconstruction_loss.mean() + (
            self.config.sparsity_coefficient * (1 if not self.is_gated else 2)
            ) * sparsity_loss.sum(-1).mean()
        losses = {"reconstruction": reconstruction_loss, "sparsity": sparsity_loss}
        if self.is_gated:
            sg_gated = (lambda x: x) if self.config.anthropic_norm else jax.lax.stop_gradient
            g_out = ((pre_relu * active) * self.s) @ sg_gated(self.W_dec) + sg_gated(self.b_dec)
            gated_loss = self.reconstruction_loss(g_out, targets).astype(jnp.float32)
            losses = {**losses, "gated": gated_loss}
            loss = loss + gated_loss.mean()
        if state is not None:  # we can only tell if a neuron is dead if we know if it was alive in the first place
            death_loss = self.compute_death_loss(state, targets, out, pre_relu).astype(jnp.float32)
            losses = {**losses, "death": death_loss}
            loss = loss + death_loss.mean() * self.config.death_penalty_coefficient
        loss = self.scaler.scale(loss)
        output = SAEOutput(
            output=out / target_norm_factor,
            loss=loss,
            losses=losses,
            activations={"pre_relu": pre_relu, "hidden": hidden},
        )
        if state is None:
            return output
        else:
            return output, state

    def sparsity_loss_base(self, activations, state):
        if self.config.sparsity_loss_type == "recip":
            step = state.get(self.num_steps)
            steps, values = map(jnp.asarray, zip(*self.config.recip_schedule))
            idx = jnp.searchsorted(steps, step, side="right")
            return activations / (activations + values[idx])
        elif self.config.sparsity_loss_type == "l1":
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

    def sparsity_loss(self, activations, state):
        losses = self.sparsity_loss_base(activations, state)
        if self.config.anthropic_norm:
            return losses * jnp.linalg.norm(self.W_dec, axis=-1)
        return losses

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
    
    def compute_death_loss(self, state, targets, reconstructions, pre_relu, eps=1e-10):
        dead = state.get(self.time_since_fired) > self.config.dead_after
        if self.get_death_penalty_threshold(state) is not None:
            dead = dead | (
                (state.get(self.activated_buffer).mean(0) < self.get_death_penalty_threshold(state))
                * (state.get(self.num_steps) > self.config.buffer_size))
        if self.config.death_loss_type == "none":
            return jnp.zeros(reconstructions.shape[:-1])
        elif self.config.death_loss_type == "sparsity_threshold":
            # eps = 1e-5
            post_relu = jax.nn.relu(pre_relu)
            sparsity = self.sparsity_loss(post_relu, state=state).mean(0)
            # log_sparsity = jnp.nan_to_num(jnp.log10(sparsity + eps))
            # offset = jax.lax.stop_gradient(jnp.log10(state.get(self.avg_loss_sparsity) + eps) - log_sparsity)
            offset = jax.lax.stop_gradient(state.get(self.activated_buffer).mean(0) - sparsity)
            # offset = jax.lax.stop_gradient(jnp.log10(state.get(self.activated_buffer).mean(0) + eps) - log_sparsity)
            # offset = 0
            return jax.nn.relu(jnp.log10(self.death_penalty_threshold + eps)) - jnp.log10(sparsity + offset + eps).sum(-1) * (state.get(self.num_steps) > self.config.buffer_size)
            # return (jax.nn.relu(jnp.log10(self.config.death_penalty_threshold + eps)) - (log_sparsity + offset)).sum(-1)
            # return (jax.nn.relu(self.config.death_penalty_threshold - (sparsity + offset)).sum(-1))
            # return (jax.nn.relu(self.config.death_penalty_threshold - (sparsity + offset)).sum(-1) / self.config.death_penalty_threshold)
        elif self.config.death_loss_type == "ghost_grads":
            post_exp = jnp.where(dead, jnp.exp(pre_relu) * self.s, 0)
            ghost_recon = post_exp @ self.W_dec
            
            residual = targets - reconstructions
            ghost_norm = jnp.linalg.norm(ghost_recon, axis=-1)
            diff_norm = jnp.linalg.norm(residual, axis=-1)
            ghost_recon = ghost_recon * jax.lax.stop_gradient(diff_norm / (ghost_norm * 2 + eps))[:, None]
            
            # same thing
            ghost_loss = self.reconstruction_loss(ghost_recon, jax.lax.stop_gradient(residual)).astype(jnp.float32)
            # ghost_loss = jnp.square(ghost_recon - jax.lax.stop_gradient(residual))
            recon_loss = self.reconstruction_loss(reconstructions, targets).astype(jnp.float32)
            ghost_loss = ghost_loss * jax.lax.stop_gradient(recon_loss / (ghost_loss + eps))
            
            loss = jax.lax.select(dead.any(), ghost_loss, jnp.zeros_like(ghost_loss)).mean(-1)
            if self.config.scale_ghost_by_death:
                loss = loss * jnp.mean(dead.astype(jnp.float32))
            return loss
        elif self.config.death_loss_type == "dm_ghost_grads":
            # if self.is_gated:
                # reconstructions = jax.lax.stop_gradient((jax.nn.relu(pre_relu) * self.s) @ self.W_dec + self.b_dec)
            # post_exp = jnp.exp(pre_relu)
            # post_exp = jnp.where(pre_relu > 0, 2 - jnp.exp(-pre_relu), jnp.exp(pre_relu))
            post_exp = jax.nn.softplus(pre_relu)
            # post_exp = jnp.where(pre_relu > 0, pre_relu + 1, jnp.exp(pre_relu))
            # post_exp = post_exp * jax.lax.stop_gradient(self.s)
            post_exp = post_exp * self.s
            if self.is_gated:
                # post_exp = post_exp * jax.lax.stop_gradient(jax.nn.softplus(self.s_gate))
                post_exp = post_exp * jax.nn.softplus(self.s_gate)
            post_exp = jnp.where(dead, post_exp, 0)
            # ghost_recon = post_exp @ self.W_dec
            # ghost_recon = post_exp @ jax.lax.stop_gradient(self.W_dec)
            ghost_recon = post_exp @ self.W_dec
            
            residual = jax.lax.stop_gradient(targets - reconstructions)
            ghost_norm = jnp.linalg.norm(ghost_recon, axis=-1, keepdims=True)
            diff_norm = jnp.linalg.norm(residual, axis=-1, keepdims=True)
            ghost_recon = ghost_recon * (jax.lax.stop_gradient(diff_norm / (ghost_norm * 2 + eps)))
            # ghost_recon = ghost_recon * jax.lax.stop_gradient(diff_norm) / jax.lax.stop_gradient(ghost_norm * 2 + eps)
            # ghost_recon = (ghost_recon / jax.lax.stop_gradient(ghost_norm * 2 + eps)) * jax.lax.stop_gradient(diff_norm)
            # ghost_recon = ghost_recon * jnp.nan_to_num(jax.lax.stop_gradient(diff_norm / (ghost_norm * 2 + eps)), nan=1.0)
            
            # ghost_loss = self.reconstruction_loss(ghost_recon, residual)
            ghost_loss = (ghost_recon - residual) ** 2
            recon_loss = self.reconstruction_loss(reconstructions, targets)
            ghost_loss = ghost_loss * jax.lax.stop_gradient(recon_loss / (ghost_loss + eps))
            
            return (ghost_loss).mean(-1) * jnp.mean(dead.astype(jnp.float32))
        else:
            raise ValueError(f"Unknown death loss type: \"{self.config.death_loss_type}\"")

    def project(self, updates: PyTree[Float]):
        def project_away(W_dec_grad):
            return W_dec_grad - jnp.einsum("h f, h -> h f",
                                            self.W_dec,
                                            jnp.einsum("h f, h f -> h",
                                                        self.W_dec,
                                                        W_dec_grad))
        updates = eqx.tree_at(lambda update: update.W_dec, updates,
                                replace_fn=project_away)
        return updates

    def update_gradients(self, grads: PyTree[Float], state: eqx.nn.State, key: jax.random.PRNGKey):
        grads = self.scaler.unscale(grads)
        if self.config.project_grads_from_dec and not self.config.anthropic_norm:
            grads = self.project(grads)
        return grads

    def apply_updates(self, updates: PyTree[Float],
                      state: eqx.nn.State,
                      opt_state,
                      last_input: Float[Array, "b f"],
                      last_target: Float[Array, "b f"],
                      last_output: SAEOutput,
                      step: int,
                      key: jax.random.PRNGKey):
        # assumes adam is the 1st gradient processor
        get_adam = lambda s: s[1][0]
        
        if self.config.project_updates_from_dec and not self.config.anthropic_norm:
            updates = self.project(updates)
        updated = eqx.apply_updates(self, updates)

        w_dec_selector = lambda x: x.W_dec
        updated = eqx.tree_at(w_dec_selector, updated, replace_fn=self.normalize_w_dec)
        
        if self.config.norm_input == "wes-mean-fixed":
            updated = eqx.tree_at(lambda x: x.mean_norm, updated, state.get(self.ds_mean_norm))
        
        # at the end of our first step, compute mean
        def adjust_mean(b_dec, opt_state):
            if self.config.decoder_bias_init_method == "mean":
                b_dec = b_dec + jnp.mean(last_target - last_output.output, axis=0)
            elif self.config.decoder_bias_init_method == "geom_median":
                b_dec = b_dec + geometric_median(last_target - last_output.output)
            try:
                try:
                    opt_state = eqx.tree_at(lambda s: get_adam(s).mu.b_dec, opt_state, jnp.zeros_like(get_adam(opt_state).mu.b_dec))
                except AttributeError:
                    pass
                opt_state = eqx.tree_at(lambda s: get_adam(s).nu.b_dec, opt_state, jnp.zeros_like(get_adam(opt_state).nu.b_dec))
            except AttributeError:
                pass
            return b_dec, opt_state
        updated_b_dec, opt_state = jax.lax.switch(jnp.astype(step == 1, jnp.int32), (lambda *a: a, adjust_mean), updated.b_dec, opt_state)
        updated = eqx.tree_at(lambda s: s.b_dec, updated, updated_b_dec)
        
        def resample(updated, state, opt_state):
            dead = state.get(self.time_since_fired) > self.config.dead_after
            if self.config.resample_type == "boom":
                W_enc = jnp.where(dead[None, :],
                                jax.nn.initializers.orthogonal()(utils.get_key(),
                                                                 updated.W_enc.shape,
                                                                 dtype=self.param_dtype),
                                updated.W_enc)
                b_enc = jnp.where(dead, self.config.encoder_bias_init_mean, updated.b_enc)
                W_dec = jnp.where(dead[:, None],
                                W_enc.T,
                                updated.W_dec)
            elif self.config.resample_type == "sample_inputs":
                # https://github.com/saprmarks/dictionary_learning/blob/main/training.py#L105
                alive_norm = jnp.linalg.norm(updated.W_enc * (~dead[None, :]), axis=-1).mean()
                sampled_vecs = jax.random.choice(key, last_input,
                                                 (self.d_hidden,), replace=True, axis=0).astype(self.param_dtype)
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
            
            try:
                # reset momentum and variance
                adam = get_adam(opt_state)

                try:
                    opt_state = eqx.tree_at(lambda s: get_adam(s).mu.W_enc, opt_state, jnp.where(dead[None, :], 0, adam.mu.W_enc))
                    opt_state = eqx.tree_at(lambda s: get_adam(s).mu.b_enc, opt_state, jnp.where(dead, 0, adam.mu.b_enc))
                    # opt_state = eqx.tree_at(lambda s: s[adam_idx].mu.W_dec, opt_state, jnp.where(dead[:, None], 0, opt_state[adam_idx].mu.W_dec))
                except AttributeError:
                    pass

                opt_state = eqx.tree_at(lambda s: get_adam(s).nu.W_enc, opt_state, jnp.where(dead[None, :], 0, adam.nu.W_enc))
                opt_state = eqx.tree_at(lambda s: get_adam(s).nu.b_enc, opt_state, jnp.where(dead, 0, adam.nu.b_enc))
                # opt_state = eqx.tree_at(lambda s: s[adam_idx].nu.W_dec, opt_state, jnp.where(dead[:, None], 0, opt_state[adam_idx].nu.W_dec))
            except AttributeError:
                pass

            state = state.set(self.time_since_fired, jnp.where(dead, 0, state.get(self.time_since_fired)))
            return updated, state, opt_state
        updated_params, updated_static = eqx.partition(updated, eqx.is_array)
        updated_params, state, opt_state = jax.lax.switch(
            jnp.astype(step % self.config.resample_every == self.config.resample_every - 1, jnp.int32),
            (lambda *a: a, resample),
            updated_params, state, opt_state)
        updated = eqx.combine(updated_params, updated_static)
        
        def requantize(x):
            # simulating 8-bit quantization
            og_shape = x.shape
            is_transpose = x.shape[0] < x.shape[1]
            if is_transpose:
                x = x.T
                og_shape = x.shape
            x = x.reshape(-1, 16).astype(jnp.bfloat16)
            zero = x.min(axis=1, keepdims=True)
            x = x - zero
            mx = 255
            scale = x.max(axis=1, keepdims=True) / mx
            quants = x / scale
            quants = quants.clip(0, mx).round()
            x = (quants * scale + zero).reshape(og_shape)
            if is_transpose:
                x = x.T
            return x


        if self.config.weights_8bit:
            for selector in (lambda s: s.W_enc, lambda s: s.W_dec):
                updated = eqx.tree_at(selector, updated, replace_fn=requantize)

        return updated, state, opt_state
    
    def normalize_w_dec(self, w_dec, eps=1e-6):
        if self.config.anthropic_norm:
            return w_dec
        if self.config.restrict_dec_norm == "exact":
            return w_dec / (eps + jnp.linalg.norm(w_dec, axis=-1, keepdims=True))
        elif self.config.restrict_dec_norm == "lte":
            return w_dec / jnp.maximum(1, eps + jnp.linalg.norm(w_dec, axis=-1, keepdims=True))
        return w_dec
        
    
    def get_stats(self, state: eqx.nn.State,
                  last_input: jax.Array, last_target: jax.typing.ArrayLike, last_output: SAEOutput,
                  eps=1e-12):
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
            var_explained=jnp.square(((last_target - last_target.mean(axis=0)) / (last_target.std(axis=0) + eps)
                                      * (last_output.output - last_output.output.mean(axis=0)) / (last_output.output.std(axis=0) + eps)
                                      ).mean(0)).mean(),
            max_time_since_fired=time_since_fired.max(),
            mean_norm=state.get(self.ds_mean_norm),
            target_mean_norm=state.get(self.ds_tgt_mean_norm),
            **{f"{param_name}_norm": jnp.linalg.norm(getattr(self, param_name)) for param_name in ("W_enc", "W_dec", "b_enc", "b_dec")},
        )
    
    def get_death_penalty_threshold(self, state):
        if self.config.death_penalty_threshold == "auto":
            buf = state.get(self.activated_buffer)
            l0_avg = buf.mean(0).sum()
            thresh_calculated = (l0_avg * 0.1) / self.d_hidden
            return jax.lax.select(state.get(self.num_steps) >= self.config.buffer_size,
                                thresh_calculated, 0.0)
        else:
            return self.config.death_penalty_threshold

    def get_log_sparsity(self, state: eqx.nn.State):
        return np.log10(1e-13 + state.get(self.activated_buffer).mean(0))
        # return np.log10(1e-13 + state.get(self.avg_l0))
    
    def get_partition_spec(self):
        if not self.config.use_model_parallel:
            spec, state_spec = {
                "W_enc": P(None, None),
                "b_enc": P(None),
                "s": P(None),
                "W_dec": P(None, None),
                "b_dec": P(None),
                "s_gate": P(None),
                "b_gate": P(None),
                "mean_norm": P(),
                "tgt_mean_norm": P(),
            }, {
                "time_since_fired": P(None),
                "num_steps": P(),
                "avg_loss_sparsity": P(None),
                "avg_l0": P(None),
                "activated_buffer": P(None, None),
                "ds_mean_norm": P(),
                "ds_tgt_mean_norm": P(),
            }
        else:
            spec, state_spec = {
                "W_enc": P(None, "mp"),
                "b_enc": P("mp"),
                "s": P("mp"),
                "W_dec": P("mp", None),
                "b_dec": P(None),
                "s_gate": P("mp"),
                "b_gate": P("mp"),
                "mean_norm": P(),
                "tgt_mean_norm": P(),
            }, {
                "time_since_fired": P("mp"),
                "num_steps": P(),
                "avg_loss_sparsity": P("mp"),
                "avg_l0": P("mp"),
                "activated_buffer": P(None, "mp"),
                "ds_mean_norm": P(),
                "ds_tgt_mean_norm": P(),
            }
        return spec, state_spec

    def restore(self, weights_path: os.PathLike):
        with safetensors.safe_open(weights_path, "flax") as f:
            for param in ("W_enc", "b_enc", "W_dec", "b_dec", "s", "s_gate", "b_gate", "mean_norm", "tgt_mean_norm"):
                match param:
                    case "s":
                        load_param = "scaling_factor"
                    case _:
                        load_param = param
                try:
                    self = eqx.tree_at(lambda s: getattr(s, param), self,
                                    jax.device_put(f.get_tensor(load_param).astype(self.param_dtype if param.startswith("W") else
                                                                                   self.bias_dtype if param.startswith("b") else
                                                                                   self.mean_norm_dtype if param.endswith("norm") else
                                                                                   self.misc_dtype),
                                                    self.sharding[param]))
                except safetensors.SafetensorError:
                    print("Can't load parameter", param)
            print("Weights restored.")
        return self

    def save(self, weights_path: os.PathLike, save_dtype: jax.typing.DTypeLike = jnp.float32):
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        state_dict = {
            "W_enc": self.W_enc,
            "b_enc": self.b_enc,
            "scaling_factor": self.s,
            "W_dec": self.W_dec,
            "b_dec": self.b_dec,
            "s_gate": self.s_gate,
            "b_gate": self.b_gate,
            "mean_norm": self.mean_norm,
            "tgt_mean_norm": self.tgt_mean_norm,
        }
        state_dict = {k: v.astype(save_dtype) for k, v in state_dict.items()}
        save_file(state_dict, weights_path)

    def push_to_hub(self, repo: str, sae_name: str = ""):
        prefix = f"{sae_name}/" if sae_name else ""
        import huggingface_hub as hf_hub
        api = hf_hub.HfApi()
        print("Uploading config...")
        config = dict(
            d_in=self.config.n_dimensions,
            dtype=self.config.param_dtype,
            expansion_factor=self.config.expansion_factor,
            l1_coefficient=self.config.sparsity_coefficient,
            train_batch_size=self.config.batch_size,
            dead_feature_window=self.config.dead_after,
            use_ghost_grads="ghost" in self.config.death_loss_type,
            d_sae=self.d_hidden,
        )
        with NamedTemporaryFile("w", suffix="cfg.json") as cfg_file:
            json.dump(config, cfg_file)
            cfg_file.flush()
            cfg_file.seek(0)
            api.upload_file(
                path_or_fileobj=cfg_file.name,
                path_in_repo=f"{prefix}/cfg.json",
                repo_id=repo,
                repo_type="model"
            )
        with NamedTemporaryFile("wb", suffix="weights.safetensors") as weights_file:
            print("Saving weights...")
            self.save(weights_file.name)
            print("Uploading weights...")
            api.upload_file(
                path_or_fileobj=weights_file.name,
                path_in_repo=f"{prefix}/sae_weights.safetensors",
                repo_id=repo,
                repo_type="model"
            )
