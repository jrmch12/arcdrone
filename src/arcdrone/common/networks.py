"""Shared neural-network building blocks for all ARC-Drone tasks."""

import functools
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
  

class VisionDecoder(nn.Module):
  """CNN + MLP decoder for the vision-based student.

  1. Runs NatureCNN on the pixels key(s) in the obs dict.
     - If ``pixels_obs_key`` is set, only that key is used.
     - Otherwise every key starting with ``pixels/`` is used.
  2. Spatial average-pools each CNN output.
  3. Concatenates with the flat ``state_obs_key`` vector (e.g. action-buffer)
     when ``state_obs_key`` is non-empty and present in the obs dict.
  4. Passes the combined vector through an MLP to produce a feature vector
     compatible with the shared ``action_head``.
  """

  layer_sizes: Sequence[int]
  activation: ActivationFn = nn.relu
  kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
  state_obs_key: str = 'state'
  # When set, use this single key for the CNN instead of the ``pixels/`` prefix scan.
  pixels_obs_key: str = ''

  @nn.compact
  def __call__(self, data: dict):
    # --- CNN on pixel stream(s) ---
    if self.pixels_obs_key:
      pixels_keys = [self.pixels_obs_key]
    else:
      pixels_keys = sorted(k for k in data if k.startswith('pixels/'))
    nature_cnn = functools.partial(
        CNN,
        num_filters=[32, 64, 64],
        kernel_sizes=[(8, 8), (4, 4), (3, 3)],
        strides=[(4, 4), (2, 2), (1, 1)],
        activation=nn.relu,
        use_bias=False,
    )
    cnn_outs = [nature_cnn()(data[key]) for key in pixels_keys]
    cnn_outs = [jnp.mean(out, axis=(-2, -3)) for out in cnn_outs]

    # --- Concat with flat state (action-buffer) if present ---
    parts = cnn_outs
    if self.state_obs_key and self.state_obs_key in data:
      parts.append(data[self.state_obs_key])

    hidden = jnp.concatenate(parts, axis=-1)

    # --- MLP → feature vector (activate_final=False to match policy_decoder) ---
    return MLP(
        layer_sizes=self.layer_sizes,
        activation=self.activation,
        kernel_init=self.kernel_init,
        activate_final=False,
    )(hidden)
  
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
