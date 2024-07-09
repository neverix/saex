import functools
from typing import NamedTuple, Optional, Any

import chex
import jax
from jax import tree_util as jtu
import jax.numpy as jnp

from optax import tree_utils as otu
from optax._src import base
from optax._src import numerics
from optax._src import utils
from optax._src import transform
from optax._src import combine


_abs_sq = numerics.abs_sq


class ScaleByMomentumlessAdamState(NamedTuple):
  """State for the Adam algorithm."""
  count: chex.Array  # shape=(), dtype=jnp.int32.
  nu: base.Updates


def update_moment_per_elem_norm(updates, moments, decay, order):
  """Compute the EMA of the `order`-th moment of the element-wise norm."""

  def orderth_norm(g):
    if jnp.isrealobj(g):
      return g ** order
    else:
      half_order = order / 2
      # JAX generates different HLO for int and float `order`
      if half_order.is_integer():
        half_order = int(half_order)
      return _abs_sq(g) ** half_order

  return jax.tree_util.tree_map(
      lambda g, t: (1 - decay) * orderth_norm(g) + decay * t, updates, moments)


@functools.partial(jax.jit, inline=True)
def bias_correction(moment, decay, count):
  """Performs bias correction. It becomes a no-op as count goes to infinity."""
  # The conversion to the data type of the moment ensures that bfloat16 remains
  # bfloat16 in the optimizer state. This conversion has to be done after
  # `bias_correction_` is calculated as calculating `decay**count` in low
  # precision can result in it being rounded to 1 and subsequently a
  # "division by zero" error.
  bias_correction_ = 1 - decay**count

  # Perform division in the original precision.
  return jax.tree_util.tree_map(
      lambda t: t / bias_correction_.astype(t.dtype), moment)
  

# https://github.com/google-deepmind/optax/blob/d58899db86d2b37a60f19067ea2e4994c795b663/optax/_src/transform.py#L172
def scale_by_momentumless_adam(
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
) -> base.GradientTransformation:
  def init_fn(params):
    nu = otu.tree_zeros_like(params)  # Second moment
    return ScaleByMomentumlessAdamState(count=jnp.zeros([], jnp.int32), nu=nu)

  def update_fn(updates, state, params=None):
    del params
    nu = update_moment_per_elem_norm(updates, state.nu, b2, 2)
    count_inc = numerics.safe_int32_increment(state.count)
    nu_hat = bias_correction(nu, b2, count_inc)
    updates = jtu.tree_map(
        lambda m, v: m / (jnp.sqrt(v + eps_root) + eps), updates, nu_hat)
    return updates, ScaleByMomentumlessAdamState(count=count_inc, nu=nu)

  return base.GradientTransformation(init_fn, update_fn)


def momentumless_adam(
    learning_rate: base.ScalarOrSchedule,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
) -> base.GradientTransformation:
  return combine.chain(
      scale_by_momentumless_adam(
          b2=b2,
          eps=eps,
          eps_root=eps_root,
      ),
      transform.scale_by_learning_rate(learning_rate),
  )
