from dataclasses import dataclass
from typing import Any, cast

import einops
import jax.numpy as jnp

import wandb
from saex.toy_models import Config as ToyConfig
from saex.toy_models import Model as ToyModel


@dataclass
class SAEToyModelRunnerConfig:
    # ReLu Model Parameters
    n_features: int = 5
    n_hidden: int = 2
    n_correlated_pairs: int = 0
    n_anticorrelated_pairs: int = 0
    feature_probability: float = 0.025
    model_training_steps: int = 10_000

    def __post_init__(self):
        self.d_in = self.n_hidden  # hidden for the ReLu model is the input for the SAE


def toy_model_sae_runner(cfg: SAEToyModelRunnerConfig):
    """ 
    A runner for training an SAE on a toy model.
    """
    # Toy Model Config
    toy_model_cfg = ToyConfig(
        n_instances=1,  # Not set up to train > 1 SAE so shouldn't do > 1 model.
        n_features=cfg.n_features,
        n_hidden=cfg.n_hidden,
        n_correlated_pairs=cfg.n_correlated_pairs,
        n_anticorrelated_pairs=cfg.n_anticorrelated_pairs,
    )

    # Initialize Toy Model
    model = ToyModel(
        cfg=toy_model_cfg,
        feature_probability=cfg.feature_probability,
    )

    # Train the Toy Model
    model.optimize(steps=cfg.model_training_steps)

    # Generate Training Data
    batch = model.generate_batch(cfg.total_training_tokens)
    hidden = einops.einsum(
        batch,
        model.W,
        "batch_size instances features, instances hidden features -> batch_size instances hidden",
    )


if __name__ == "__main__":
    cfg = SAEToyModelRunnerConfig()
    toy_model_sae_runner(cfg)
