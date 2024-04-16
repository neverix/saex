"""

cribbed from https://github.com/jbloomAus/SAELens/blob/main/sae_lens/training/toy_models.py

https://www.lesswrong.com/posts/LnHowHgmrMbWtpkxx/intro-to-superposition-and-sparse-autoencoders-colab?fbclid=IwAR04OCGu_unvxezvDWkys9_6MJPEnXuu6GSqU6ScrMkAb1bvdSYFOeS35AY
https://github.com/callummcdougall/sae-exercises-mats?fbclid=IwAR3qYAELbyD_x5IAYN4yCDFQzxXHeuH6CwMi_E7g4Qg6G1QXRNAYabQ4xGs

"""


from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple, Union

from jaxtyping import Array, Float

import jax
import jax.numpy as jnp
from jax import jit
import optax

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


Arr = jnp.ndarray


def plot_features_in_2d(
    values: jnp.ndarray,
    colors: Optional[List[Any]] = None,
    title: Optional[List[str] | str] = None,
    subplot_titles: Optional[List[str] | List[List[str]]] = None,
    save: Optional[str] = None,
    colab: bool = False,
):
    # Convert values to 4D for consistency
    if values.ndim == 3:
        values = values[jnp.newaxis]
    values = values.transpose(0, 2, 1, 3)

    n_timesteps, n_features, n_instances, _ = values.shape

    # Convert colors to 3D, if it's 2D (i.e. same colors for all instances)
    if isinstance(colors, list) and isinstance(colors[0], str):
        colors = [colors for _ in range(n_instances)]
    # Convert colors to something which has 4D, if it is 3D (i.e. same colors for all timesteps)
    if any(
        [
            colors is None,
            isinstance(colors, list)
            and isinstance(colors[0], list)
            and isinstance(colors[0][0], str),
            (isinstance(colors, jnp.ndarray) or isinstance(colors, Arr))
            and colors.ndim == 3,
        ]
    ):
        colors = [colors for _ in range(values.shape[0])]
    # Now that colors has length `timesteps` in some sense, we can convert it to lists of strings
    assert colors is not None
    colors = [
        parse_colors_for_superposition_plot(c, n_instances, n_features) for c in colors
    ]

    # Same for subplot titles & titles
    if subplot_titles is not None:
        if isinstance(subplot_titles, list) and isinstance(subplot_titles[0], str):
            subplot_titles = [subplot_titles for _ in range(values.shape[0])]
    if title is not None:
        if isinstance(title, str):
            title = [title for _ in range(values.shape[0])]

    # Create a figure and axes
    fig, axs = plt.subplots(1, n_instances, figsize=(5 * n_instances, 5))
    if n_instances == 1:
        axs = [axs]

    # If there are titles, add more spacing for them
    fig.subplots_adjust(bottom=0.2, top=0.9, left=0.1, right=0.9)
    if title:
        fig.subplots_adjust(top=0.8)
    # Initialize lines and markers
    lines = []
    markers = []
    for instance_idx, ax in enumerate(axs):
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal", adjustable="box")
        instance_lines = []
        instance_markers = []
        for feature_idx in range(n_features):
            (line,) = ax.plot([], [], color=colors[0][instance_idx][feature_idx], lw=1)
            (marker,) = ax.plot([], [], color=colors[0][instance_idx][feature_idx], marker="o", markersize=4)
            instance_lines.append(line)
            instance_markers.append(marker)
        lines.append(instance_lines)
        markers.append(instance_markers)

    def update(val: float):
        if n_timesteps > 1:
            _ = slider.val
        t = int(val)
        for instance_idx in range(n_instances):
            for feature_idx in range(n_features):
                x, y = values[t, feature_idx, instance_idx].tolist()
                lines[instance_idx][feature_idx].set_data([0, x], [0, y])
                markers[instance_idx][feature_idx].set_data(x, y)
                lines[instance_idx][feature_idx].set_color(colors[t][instance_idx][feature_idx])
                markers[instance_idx][feature_idx].set_color(colors[t][instance_idx][feature_idx])
            if title:
                fig.suptitle(title[t], fontsize=15)
            if subplot_titles:
                axs[instance_idx].set_title(subplot_titles[t][instance_idx], fontsize=12)
        fig.canvas.draw_idle()

    def play(event: Any):
        _ = slider.val
        for i in range(n_timesteps):
            update(i)
            slider.set_val(i)
            plt.pause(0.05)
        fig.canvas.draw_idle()

    if n_timesteps > 1:
        # Create the slider
        ax_slider = plt.axes((0.15, 0.05, 0.7, 0.05), facecolor="lightgray")
        slider = Slider(
            ax_slider, "Time", 0, n_timesteps - 1, valinit=0, valfmt="%1.0f"
        )

        # Call the update function when the slider value is changed
        slider.on_changed(update)

        # Initialize the plot
        play(0)
    else:
        update(0)

    if isinstance(save, str):
        ani = FuncAnimation(
            fig, cast(Any, update), frames=n_timesteps, interval=0.04, repeat=False
        )
        ani.save(save, writer="pillow", fps=25)
    elif colab:
        ani = FuncAnimation(
            fig, cast(Any, update), frames=n_timesteps, interval=0.04, repeat=False
        )
        clear_output()
        return ani


def parse_colors_for_superposition_plot(
    colors: Optional[
        Union[Tuple[int, int], List[List[str]], jnp.ndarray]
    ],
    n_instances: int,
    n_feats: int,
) -> List[List[str]]:
    if isinstance(colors, tuple):
        n_corr, n_anti = colors
        n_indep = n_feats - 2 * (n_corr - n_anti)
        return [
            ["blue", "blue", "limegreen", "limegreen", "purple", "purple"][: n_corr * 2]
            + ["red", "red", "orange", "orange", "brown", "brown"][: n_anti * 2]
            + ["black"] * n_indep
            for _ in range(n_instances)
        ]
    elif isinstance(colors, str):
        return [[colors] * n_feats] * n_instances
    elif colors is None:
        return [["black"] * n_feats] * n_instances
    assert isinstance(colors, list)
    return colors
