# Copyright 2025 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PPO networks."""

import functools
from typing import Any, Callable, Literal, Mapping, Sequence, Tuple, Optional

from brax.training import distribution
# from brax.training import networks
from brax.training import types
from brax.training.networks import normalizer_select
from brax.training.types import PRNGKey
import flax
from flax import linen as nn
import jax
import jax.numpy as jnp



ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]

@flax.struct.dataclass
class FeedForwardNetwork:
  init: Callable[..., Any]
  apply: Callable[..., Any]


@flax.struct.dataclass
class SITTNetworks:
  # PPO-compatible 
  policy_network: FeedForwardNetwork
  value_network: FeedForwardNetwork
  parametric_action_distribution: distribution.ParametricDistribution

  # sitt
  policy_decoder: Optional[FeedForwardNetwork] = None
  student_decoder: Optional[FeedForwardNetwork] = None
  proxy_decoder: Optional[FeedForwardNetwork] = None
  action_head: Optional[FeedForwardNetwork] = None


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


class StudentVisionDecoder(nn.Module):
  """CNN + MLP decoder for the vision-based student.

  1. Runs NatureCNN on every ``pixels/*`` key in the obs dict.
  2. Spatial average-pools each CNN output.
  3. Concatenates with the flat ``state_obs_key`` vector (action-buffer).
  4. Passes the combined vector through an MLP to produce a feature vector
     compatible with the shared ``action_head``.
  """

  layer_sizes: Sequence[int]
  activation: ActivationFn = nn.relu
  kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
  state_obs_key: str = 'state'

  @nn.compact
  def __call__(self, data: dict):
    # --- CNN on every pixel stream ---
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


def _get_obs_size(obs_size: types.ObservationSize, obs_key: str) -> int:
  obs_size = obs_size[obs_key] if isinstance(obs_size, Mapping) else obs_size
  return jax.tree_util.tree_flatten(obs_size)[0][-1]

# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------

def make_sitt_networks(
    observation_size: types.ObservationSize,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = (
        types.identity_observation_preprocessor
    ),
    policy_hidden_layer_sizes: Sequence[int] = (32,) * 4,
    student_hidden_layer_sizes: Sequence[int] = (32,) * 4,
    proxy_hidden_layer_sizes: Sequence[int] = (32,) * 4,
    action_hidden_layer_sizes: Sequence[int] = (64,),
    value_hidden_layer_sizes: Sequence[int] = (256,) * 5,
    activation: ActivationFn = nn.tanh,
    policy_obs_key: str = "state",
    value_obs_key: str = "state",
    distribution_type: Literal["tanh_normal"] = "tanh_normal",
    use_sitt: bool = False,
    student_observation_size: Optional[Mapping[str, Tuple]] = None,
    student_obs_key: str = "state",
) -> SITTNetworks:

    if distribution_type != "tanh_normal":
        raise ValueError("Only tanh_normal supported")

    kernel_init = jax.nn.initializers.lecun_uniform()
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )

    obs_size = _get_obs_size(observation_size, policy_obs_key)
    dummy_obs = jnp.zeros((1, obs_size))

    # ------------------------------------------------------------------
    # Modules (defined ONCE)
    # ------------------------------------------------------------------

    policy_decoder = MLP(
        layer_sizes=policy_hidden_layer_sizes,
        activation=activation,
        kernel_init=kernel_init,
    )

    action_head = MLP(
        layer_sizes=list(action_hidden_layer_sizes)
        + [parametric_action_distribution.param_size],
        activation=activation,
        kernel_init=kernel_init,
    )

    value_mlp = MLP(
        layer_sizes=list(value_hidden_layer_sizes) + [1],
        activation=activation,
        kernel_init=kernel_init,
    )

    student_decoder = (
        StudentVisionDecoder(
            layer_sizes=student_hidden_layer_sizes,
            activation=activation,
            kernel_init=kernel_init,
            state_obs_key=student_obs_key,
        )
        if use_sitt
        else None
    )

    # Dummy student obs dict for init (pixels + flat state)
    dummy_student_obs = None
    if use_sitt and student_observation_size is not None:
        dummy_student_obs = {
            key: jnp.zeros((1,) + tuple(shape))
            for key, shape in student_observation_size.items()
        }
    proxy_decoder = (
        MLP(proxy_hidden_layer_sizes, activation, kernel_init)
        if use_sitt
        else None
    )

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _preprocess(obs, pparams, key):
        if isinstance(obs, Mapping):
            return preprocess_observations_fn(
                obs[key], normalizer_select(pparams, key)
            )
        return preprocess_observations_fn(obs, pparams)

    def _apply_policy(decoder, dec_params, head_params, pparams, obs):
        feats = decoder.apply(dec_params, _preprocess(obs, pparams, policy_obs_key))
        return action_head.apply(head_params, feats)

    # ------------------------------------------------------------------
    # PPO-compatible policy network (UNCHANGED interface)
    # ------------------------------------------------------------------

    def policy_init(key):
        k1, k2 = jax.random.split(key)
        dec_params = policy_decoder.init(k1, dummy_obs)
        feats = policy_decoder.apply(dec_params, dummy_obs)
        head_params = action_head.init(k2, feats)
        return dec_params, head_params

    policy_network = FeedForwardNetwork(
        init=policy_init,
        apply=lambda pparams, params, obs: _apply_policy(
            policy_decoder, params[0], params[1], pparams, obs
        ),
    )

    # ------------------------------------------------------------------
    # Value network (UNCHANGED)
    # ------------------------------------------------------------------

    obs_size_value = _get_obs_size(observation_size, value_obs_key)
    dummy_obs_value = jnp.zeros((1, obs_size_value))

    value_network = FeedForwardNetwork(
        init=lambda key: value_mlp.init(key, dummy_obs_value),
        apply=lambda pparams, params, obs: jnp.squeeze(
            value_mlp.apply(
                params,
                _preprocess(obs, pparams, value_obs_key),
            ),
            axis=-1,
        ),
    )

    # ------------------------------------------------------------------
    # Decoder-only adapters (feature access for SITT)
    # ------------------------------------------------------------------

    policy_decoder_net = FeedForwardNetwork(
        init=lambda key: policy_decoder.init(key, dummy_obs),
        apply=lambda pparams, params, obs: policy_decoder.apply(
            params,
            _preprocess(obs, pparams, policy_obs_key),
        ),
    )

    student_decoder_net = (
        FeedForwardNetwork(
            init=lambda key: student_decoder.init(key, dummy_student_obs),
            apply=lambda pparams, params, obs: student_decoder.apply(
                params, obs,
            ),
        )
        if use_sitt
        else None
    )

    proxy_decoder_net = (
        FeedForwardNetwork(
            init=lambda key: proxy_decoder.init(key, dummy_obs),
            apply=lambda pparams, params, obs: proxy_decoder.apply(
                params,
                _preprocess(obs, pparams, policy_obs_key),
            ),
        )
        if use_sitt
        else None
    )

    action_head_net = FeedForwardNetwork(
        init=lambda key: action_head.init(
            key,
            policy_decoder.apply(
                policy_decoder.init(key, dummy_obs), dummy_obs
            ),
        ),
        apply=lambda _pparams, params, feats: action_head.apply(params, feats),
    )

    return SITTNetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
        policy_decoder=policy_decoder_net,
        student_decoder=student_decoder_net,
        proxy_decoder=proxy_decoder_net,
        action_head=action_head_net,
    )
# ---------------------------------------------------------------------
# Inference function
# ---------------------------------------------------------------------

def make_inference_fn(sitt_networks: SITTNetworks, compute_value: bool = False):
  """Creates params and inference function for the PPO agent.

  Args:
    sitt_networks: The SITT networks.
    compute_value: If True, compute value during rollouts.
  """

  def make_policy(
      params: types.Params, deterministic: bool = False
  ) -> types.Policy:
    policy_network = sitt_networks.policy_network
    parametric_action_distribution = sitt_networks.parametric_action_distribution

    def policy(
        observations: types.Observation, key_sample: PRNGKey
    ) -> Tuple[types.Action, types.Extra]:
      param_subset = (params[0], params[1])  # normalizer and policy params
      logits = policy_network.apply(*param_subset, observations)
      if deterministic:
        return sitt_networks.parametric_action_distribution.mode(logits), {}
      raw_actions = parametric_action_distribution.sample_no_postprocessing(
          logits, key_sample
      )
      log_prob = parametric_action_distribution.log_prob(logits, raw_actions)
      postprocessed_actions = parametric_action_distribution.postprocess(
          raw_actions
      )
      extras = {
          'log_prob': log_prob,
          'raw_action': raw_actions,
          'distribution_params': logits,
      }
      if compute_value:
        extras['value'] = sitt_networks.value_network.apply(
            params[0], params[2], observations
        )
      return postprocessed_actions, extras

    return policy

  return make_policy


def make_student_inference_fn(
        sitt_networks: SITTNetworks, compute_value: bool = False
):
    """Creates student-policy inference function.

    Expected params layout:
        (normalizer_params, policy_params, value_params, student_dec_params, ...)
    """

    def make_policy(
            params: types.Params, deterministic: bool = False
    ) -> types.Policy:
        parametric_action_distribution = sitt_networks.parametric_action_distribution

        def policy(
                observations: types.Observation, key_sample: PRNGKey
        ) -> Tuple[types.Action, types.Extra]:
            if len(params) < 4 or params[3] is None:
                raise ValueError(
                        'Student inference requires `student_dec_params` at params[3].'
                )

            student_feats = sitt_networks.student_decoder.apply(
                    params[0], params[3], observations
            )
            logits = sitt_networks.action_head.apply(None, params[1][1], student_feats)

            if deterministic:
                return parametric_action_distribution.mode(logits), {}

            raw_actions = parametric_action_distribution.sample_no_postprocessing(
                    logits, key_sample
            )
            log_prob = parametric_action_distribution.log_prob(logits, raw_actions)
            postprocessed_actions = parametric_action_distribution.postprocess(
                    raw_actions
            )

            extras = {
                    'log_prob': log_prob,
                    'raw_action': raw_actions,
                    'distribution_params': logits,
            }
            if compute_value:
                extras['value'] = sitt_networks.value_network.apply(
                        params[0], params[2], observations
                )
            return postprocessed_actions, extras

        return policy

    return make_policy


# def make_decoder_inference(decoder_network: FeedForwardNetwork):
#   """
#   Returns a pure inference function for a decoder.
#   """

#   def infer(
#       norm_params,
#       decoder_params,
#       observations,
#   ):
#     return decoder_network.apply(
#         norm_params,
#         decoder_params,
#         observations,
#     )

#   return infer

# def make_action_head_inference(action_head: FeedForwardNetwork):
#   """
#   Returns a pure inference function for the action head.
#   """

#   def infer(action_head_params, features):
#     return action_head.apply(None, action_head_params, features)

#   return infer