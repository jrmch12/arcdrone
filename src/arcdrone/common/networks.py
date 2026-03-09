"""Shared neural-network building blocks for all ARC-Drone tasks."""

from typing import Any, Callable, Mapping, Sequence, Tuple

from brax.training import types
import flax
from flax import linen as nn
import jax
import jax.numpy as jnp


ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]


class MLP(nn.Module):
  layer_sizes: Sequence[int]
  activation: ActivationFn = nn.tanh
  kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
  activate_final: bool = False

  @nn.compact
  def __call__(self, x: jnp.ndarray):
    for i, size in enumerate(self.layer_sizes):
      x = nn.Dense(
          size,
          kernel_init=self.kernel_init,
          name=f'hidden_{i}',
      )(x)
      if i != len(self.layer_sizes) - 1 or self.activate_final:
        x = self.activation(x)
    return x


class CNN(nn.Module):
  """CNN module.  Inputs are expected in Batch * HWC format."""

  num_filters: Sequence[int]
  kernel_sizes: Sequence[Tuple]
  strides: Sequence[Tuple]
  activation: ActivationFn = nn.relu
  use_bias: bool = True

  @nn.compact
  def __call__(self, data: jnp.ndarray):
    hidden = data
    for num_filter, kernel_size, stride in zip(
        self.num_filters, self.kernel_sizes, self.strides
    ):
      hidden = nn.Conv(
          num_filter,
          kernel_size=kernel_size,
          strides=stride,
          use_bias=self.use_bias,
      )(hidden)
      hidden = self.activation(hidden)
    return hidden
  
class Decoder(nn.Module):
  """Plain MLP — a named alias used when the role of the network should be
  explicit (e.g. state decoder, proxy decoder)."""

  layer_sizes: Sequence[int]
  activation: ActivationFn = nn.relu
  kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
  activate_final: bool = False

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    return MLP(
        layer_sizes=self.layer_sizes,
        activation=self.activation,
        kernel_init=self.kernel_init,
        activate_final=self.activate_final,
    )(x)


def _get_obs_size(obs_size: types.ObservationSize, obs_key: str) -> int:
  obs_size = obs_size[obs_key] if isinstance(obs_size, Mapping) else obs_size
  return jax.tree_util.tree_flatten(obs_size)[0][-1]
