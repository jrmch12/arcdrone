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

from typing import Any, Callable, Literal, Mapping, Sequence, Tuple

from brax.training import distribution
# from brax.training import networks
from brax.training import types
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
class PPONetworks:
  policy_network: FeedForwardNetwork
  value_network: FeedForwardNetwork
  parametric_action_distribution: distribution.ParametricDistribution

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

def _get_obs_size(
    observation_size: types.ObservationSize,
    obs_key: str,
) -> int:
  if isinstance(observation_size, Mapping):
    obs_size = observation_size[obs_key]
  else:
    obs_size = observation_size
  return jax.tree_util.tree_flatten(obs_size)[0][-1]



def make_inference_fn(ppo_networks: PPONetworks, compute_value: bool = False):
  """Creates params and inference function for the PPO agent.

  Args:
    ppo_networks: The PPO networks.
    compute_value: If True, compute value during rollouts.
  """

  def make_policy(
      params: types.Params, deterministic: bool = False
  ) -> types.Policy:
    policy_network = ppo_networks.policy_network
    parametric_action_distribution = ppo_networks.parametric_action_distribution

    def policy(
        observations: types.Observation, key_sample: PRNGKey
    ) -> Tuple[types.Action, types.Extra]:
      param_subset = (params[0], params[1])  # normalizer and policy params
      logits = policy_network.apply(*param_subset, observations)
      if deterministic:
        return ppo_networks.parametric_action_distribution.mode(logits), {}
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
        extras['value'] = ppo_networks.value_network.apply(
            params[0], params[2], observations
        )
      return postprocessed_actions, extras

    return policy

  return make_policy


def make_ppo_networks(
    observation_size: types.ObservationSize,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    policy_hidden_layer_sizes: Sequence[int] = (32,) * 4,
    value_hidden_layer_sizes: Sequence[int] = (256,) * 5,
    activation: ActivationFn = nn.tanh,
    policy_obs_key: str = 'state',
    value_obs_key: str = 'state',
    distribution_type: Literal['normal', 'tanh_normal'] = 'tanh_normal',
    noise_std_type: Literal['scalar', 'log'] = 'scalar',
    init_noise_std: float = 1.0,
    state_dependent_std: bool = False,
    policy_network_kernel_init_fn: Initializer = (jax.nn.initializers.lecun_uniform),
    policy_network_kernel_init_kwargs: Mapping[str, Any] | None = None,
    value_network_kernel_init_fn: Initializer = (jax.nn.initializers.lecun_uniform),
    value_network_kernel_init_kwargs: Mapping[str, Any] | None = None,
) -> PPONetworks:
  """Make PPO networks with preprocessor."""
  parametric_action_distribution: distribution.ParametricDistribution
  if distribution_type == 'tanh_normal':
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )
  else:
    raise ValueError(
        f'Unsupported distribution type: {distribution_type}. Must be one'
        ' of "normal" or "tanh_normal".'
    )
  # -------------------------------------------------------------------------
  # Policy network (MLP -> param_size)
  # -------------------------------------------------------------------------

  policy_kernel_init = policy_network_kernel_init_fn(
      **(policy_network_kernel_init_kwargs or {})
  )

  policy_module = MLP(
      layer_sizes=list(policy_hidden_layer_sizes)
      + [parametric_action_distribution.param_size],
      activation=activation,
      kernel_init=policy_kernel_init,
  )
  
  obs_size_policy = _get_obs_size(observation_size, policy_obs_key)
  dummy_obs_policy = jnp.zeros((1, obs_size_policy))

  def policy_apply(processor_params, policy_params, obs):
    if isinstance(obs, Mapping):
      obs = preprocess_observations_fn(
          obs[policy_obs_key], processor_params
      )
    else:
      obs = preprocess_observations_fn(obs, processor_params)
    return policy_module.apply(policy_params, obs)

  def policy_init(key):
    return policy_module.init(key, dummy_obs_policy)

  policy_network = FeedForwardNetwork(
      init=policy_init,
      apply=policy_apply,
  )

  # -------------------------------------------------------------------------
  # Value network (MLP -> 1)
  # -------------------------------------------------------------------------
  obs_size_value = _get_obs_size(observation_size, value_obs_key)
  dummy_obs_value = jnp.zeros((1, obs_size_value))

  value_kernel_init = value_network_kernel_init_fn(
      **(value_network_kernel_init_kwargs or {})
  )

  value_module = MLP(
      layer_sizes=list(value_hidden_layer_sizes) + [1],
      activation=activation,
      kernel_init=value_kernel_init,
  )

  def value_apply(processor_params, value_params, obs):
    if isinstance(obs, Mapping):
      obs = preprocess_observations_fn(
          obs[value_obs_key], processor_params
      )
    else:
      obs = preprocess_observations_fn(obs, processor_params)
    return jnp.squeeze(
        value_module.apply(value_params, obs), axis=-1
    )

  def value_init(key):
    return value_module.init(key, dummy_obs_value)

  value_network = FeedForwardNetwork(
      init=value_init,
      apply=value_apply,
  )
  return PPONetworks(
      policy_network=policy_network,
      value_network=value_network,
      parametric_action_distribution=parametric_action_distribution,
  )
