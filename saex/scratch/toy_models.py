"""

cribbed from https://github.com/jbloomAus/SAELens/blob/main/sae_lens/training/toy_models.py

https://www.lesswrong.com/posts/LnHowHgmrMbWtpkxx/intro-to-superposition-and-sparse-autoencoders-colab?fbclid=IwAR04OCGu_unvxezvDWkys9_6MJPEnXuu6GSqU6ScrMkAb1bvdSYFOeS35AY
https://github.com/callummcdougall/sae-exercises-mats?fbclid=IwAR3qYAELbyD_x5IAYN4yCDFQzxXHeuH6CwMi_E7g4Qg6G1QXRNAYabQ4xGs

"""


from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import optax
from jax import jit
from jaxtyping import Array, Float
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from tqdm import tqdm


def linear_lr(step: int, steps: int):
    return 1 - (step / steps)


def constant_lr(*_):
    return 1.0


def cosine_decay_lr(step: int, steps: int):
    return jnp.cos(0.5 * jnp.pi * step / (steps - 1))


@dataclass
class Config:
    n_instances: int
    n_features: int = 5
    n_hidden: int = 2
    n_correlated_pairs: int = 0
    n_anticorrelated_pairs: int = 0


class Model:
    W: jnp.ndarray
    b_final: jnp.ndarray

    def __init__(
        self,
        cfg: Config,
        feature_probability: Optional[Union[float, jnp.ndarray]] = None,
        importance: Optional[Union[float, jnp.ndarray]] = None,
    ):
        self.cfg = cfg
        
        if feature_probability is None:
            feature_probability = jnp.ones(())
        if isinstance(feature_probability, float):
            feature_probability = jnp.array(feature_probability)
        self.feature_probability = jnp.broadcast_to(
            feature_probability, (cfg.n_instances, cfg.n_features)
        )
        if importance is None:
            importance = jnp.ones(())
        if isinstance(importance, float):
            importance = jnp.array(importance)
        self.importance = jnp.broadcast_to(
            importance, (cfg.n_instances, cfg.n_features)
        )

    def forward(
        self, W: Float[Array, "i h f"], b_final: Float[Array, "i f"], features: Float[Array, "b i f"]
    ) -> Float[Array, "b i f"]:
        hidden = jnp.einsum("...if,ihf->...ih", features, W)
        out = jnp.einsum("...ih,ihf->...if", hidden, W)
        return jax.nn.relu(out + b_final)

    def generate_correlated_features(
        self, batch_size: int, n_correlated_pairs: int
    ) -> jnp.ndarray:
        feat = jax.random.uniform(
            jax.random.PRNGKey(0), (batch_size, self.cfg.n_instances, 2 * n_correlated_pairs)
        )
        feat_set_seeds = jax.random.uniform(
            jax.random.PRNGKey(0), (batch_size, self.cfg.n_instances, n_correlated_pairs)
        )
        feat_set_is_present = feat_set_seeds <= self.feature_probability[:, [0]]
        feat_is_present = jnp.repeat(
            feat_set_is_present, 2, axis=-1
        )
        return jnp.where(feat_is_present, feat, 0.0)

    def generate_anticorrelated_features(
        self, batch_size: int, n_anticorrelated_pairs: int
    ) -> jnp.ndarray:
        feat = jax.random.uniform(
            jax.random.PRNGKey(0), (batch_size, self.cfg.n_instances, 2 * n_anticorrelated_pairs)
        )
        feat_set_seeds = jax.random.uniform(
            jax.random.PRNGKey(0), (batch_size, self.cfg.n_instances, n_anticorrelated_pairs)
        )
        first_feat_seeds = jax.random.uniform(
            jax.random.PRNGKey(0), (batch_size, self.cfg.n_instances, n_anticorrelated_pairs)
        )
        feat_set_is_present = feat_set_seeds <= 2 * self.feature_probability[:, [0]]
        first_feat_is_present = first_feat_seeds <= 0.5
        first_feats = jnp.where(
            feat_set_is_present & first_feat_is_present,
            feat[:, :, :n_anticorrelated_pairs],
            0.0,
        )
        second_feats = jnp.where(
            feat_set_is_present & (~first_feat_is_present),
            feat[:, :, n_anticorrelated_pairs:],
            0.0,
        )
        return jnp.concatenate([first_feats, second_feats], axis=-1)

    def generate_uncorrelated_features(
        self, batch_size: int, n_uncorrelated: int
    ) -> jnp.ndarray:
        feat = jax.random.uniform(
            jax.random.PRNGKey(0), (batch_size, self.cfg.n_instances, n_uncorrelated)
        )
        feat_seeds = jax.random.uniform(
            jax.random.PRNGKey(0), (batch_size, self.cfg.n_instances, n_uncorrelated)
        )
        feat_is_present = feat_seeds <= self.feature_probability[:, [0]]
        return jnp.where(feat_is_present, feat, 0.0)

    def generate_batch(
        self, batch_size: int
    ) -> jnp.ndarray:
        n_uncorrelated = (
            self.cfg.n_features
            - 2 * self.cfg.n_correlated_pairs
            - 2 * self.cfg.n_anticorrelated_pairs
        )
        data = []
        if self.cfg.n_correlated_pairs > 0:
            data.append(
                self.generate_correlated_features(
                    batch_size, self.cfg.n_correlated_pairs
                )
            )
        if self.cfg.n_anticorrelated_pairs > 0:
            data.append(
                self.generate_anticorrelated_features(
                    batch_size, self.cfg.n_anticorrelated_pairs
                )
            )
        if n_uncorrelated > 0:
            data.append(self.generate_uncorrelated_features(batch_size, n_uncorrelated))
        batch = jnp.concatenate(data, axis=-1)
        return batch

    def calculate_loss(
        self,
        out: jnp.ndarray,
        batch: jnp.ndarray,
    ) -> jnp.ndarray:
        error = self.importance * ((batch - out) ** 2)
        loss = jnp.mean(error, axis=(1, 2)).sum()
        return loss

    def optimize(
        self,
        batch_size: int = 1024,
        steps: int = 10_000,
        log_freq: int = 100,
        lr: float = 1e-3,
        lr_scale: Callable[[int, int], float] = constant_lr,
    ):
        optimizer = optax.adam(lr)
        
        W = jax.random.normal(
            jax.random.PRNGKey(0), (self.cfg.n_instances, self.cfg.n_hidden, self.cfg.n_features)
        )
        b_final = jnp.zeros((self.cfg.n_instances, self.cfg.n_features))
        
        params = (W, b_final)
        opt_state = optimizer.init(params)
        lwg = jax.value_and_grad(lambda params, batch: self.calculate_loss(self.forward(*params, batch), batch))

        progress_bar = tqdm(range(steps), desc="Training Toy Model")

        for step in progress_bar:
            step_lr = lr * lr_scale(step, steps)
            loss, grad = lwg(params, self.generate_batch(batch_size))
            updates, opt_state = optimizer.update(grad, opt_state, params)
            params = optax.apply_updates(params, updates)

            if step % log_freq == 0 or (step + 1 == steps):
                batch = self.generate_batch(batch_size)
                loss = self.calculate_loss(self.forward(*params, batch), batch) / self.cfg.n_instances
                progress_bar.set_postfix(loss=loss, lr=step_lr)
