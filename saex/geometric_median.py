# cribbed from https://github.com/jbloomAus/SAELens/blob/main/sae_lens/training/geometric_median.py
import warnings

import jax.numpy as jnp
import jax


def weighted_average(points, weights):
    return jnp.sum(points * weights[:, None], axis=0) / jnp.sum(weights)


def geometric_median_objective(
    median, points, weights
):
    norms = jnp.linalg.norm(points - median[None, :], axis=-1)
    return jnp.sum(weights * norms)


def geometric_median(points,
                     eps=1e-6,
                     maxiter=200):
    weights = jnp.ones(points.shape[0])
    median = weighted_average(points, weights)
    def scanner(state, _):
        median, weights = state
        norms = jnp.linalg.norm(points - median[None, :], axis=-1)
        new_weights = weights / jnp.maximum(norms, eps)
        median = weighted_average(points, new_weights)
        return (median, new_weights), None
    (median, _), _ = jax.lax.scan(scanner, (median, weights), None, length=maxiter)
    
    return median
