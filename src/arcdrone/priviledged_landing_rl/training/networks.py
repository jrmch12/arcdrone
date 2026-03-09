"""PPO networks for privileged-state landing RL.

Observation dict (flat structure, no pixels):
    obs = {
        "policy_obs": flat_vector,  # policy input  (privileged state)
        "value_obs":  flat_vector,  # critic input  (privileged state)
    }

Architecture:
    Policy : policy_obs → MLP decoder → features → MLP action head → logits
    Value  : value_obs  → MLP → scalar
"""

from typing import Any, Callable, Mapping, Sequence, Tuple

from brax.training import distribution
from brax.training import types
from brax.training.networks import normalizer_select
from brax.training.types import PRNGKey
import flax
from flax import linen as nn
import jax
import jax.numpy as jnp

from arcdrone.common.networks import MLP


ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]


@flax.struct.dataclass
class FeedForwardNetwork:
    init: Callable[..., Any]
    apply: Callable[..., Any]


@flax.struct.dataclass
class PPONetworks:
    """Brax-PPO-compatible network container."""
    policy_network: FeedForwardNetwork
    value_network: FeedForwardNetwork
    parametric_action_distribution: distribution.ParametricDistribution


def _shape_last_dim(shape_spec: Any) -> int:
    if isinstance(shape_spec, int):
        return shape_spec
    return int(jax.tree_util.tree_flatten(shape_spec)[0][-1])


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_ppo_networks(
    observation_size: types.ObservationSize,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = (
        types.identity_observation_preprocessor
    ),
    # MLP decoder: maps policy_obs → features
    policy_dec_hidden_layers: Sequence[int] = (512, 512, 256, 128),
    # MLP action head: maps features → distribution params
    action_hidden_layer_sizes: Sequence[int] = (512, 512, 256, 128),
    # MLP critic: maps value_obs → scalar
    value_hidden_layer_sizes: Sequence[int] = (512, 512, 256, 128),
    activation: ActivationFn = nn.tanh,
    # Top-level obs dict key for policy input
    policy_obs_key: str = "policy_obs",
    # Top-level obs dict key for critic input
    value_obs_key: str = "value_obs",
) -> PPONetworks:
    """Build flat-MLP policy + value PPO networks for privileged-state obs."""

    kernel_init = jax.nn.initializers.lecun_uniform()

    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )

    # ------------------------------------------------------------------
    # Modules
    # ------------------------------------------------------------------

    policy_decoder = MLP(
        layer_sizes=list(policy_dec_hidden_layers),
        activation=activation,
        kernel_init=kernel_init,
        activate_final=True,        # features passed into action head
    )

    policy_action_head = MLP(
        layer_sizes=list(action_hidden_layer_sizes)
            + [parametric_action_distribution.param_size],
        activation=activation,
        kernel_init=kernel_init,
        activate_final=False,
    )

    value_mlp = MLP(
        layer_sizes=list(value_hidden_layer_sizes) + [1],
        activation=activation,
        kernel_init=kernel_init,
        activate_final=False,
    )

    # ------------------------------------------------------------------
    # Dummy inputs for init
    # ------------------------------------------------------------------

    if isinstance(observation_size, Mapping):
        policy_obs_dim = _shape_last_dim(observation_size[policy_obs_key])
        value_obs_dim  = _shape_last_dim(observation_size[value_obs_key])
    else:
        policy_obs_dim = value_obs_dim = _shape_last_dim(observation_size)

    dummy_policy_obs = jnp.zeros((1, policy_obs_dim))
    dummy_value_obs  = jnp.zeros((1, value_obs_dim))

    # ------------------------------------------------------------------
    # Preprocessing helpers
    # ------------------------------------------------------------------

    def _preprocess_policy(obs, pparams):
        if isinstance(obs, Mapping):
            policy_obs = obs[policy_obs_key]
            norm = normalizer_select(pparams, policy_obs_key)
            return preprocess_observations_fn(policy_obs, norm)
        return preprocess_observations_fn(obs, pparams)

    def _preprocess_value(obs, pparams):
        if isinstance(obs, Mapping):
            value_obs = obs[value_obs_key]
            norm = normalizer_select(pparams, value_obs_key)
            return preprocess_observations_fn(value_obs, norm)
        return preprocess_observations_fn(obs, pparams)

    # ------------------------------------------------------------------
    # Value network
    # ------------------------------------------------------------------

    def _value_init(key):
        return value_mlp.init(key, dummy_value_obs)

    def _value_apply(pparams, params, obs):
        return jnp.squeeze(
            value_mlp.apply(params, _preprocess_value(obs, pparams)),
            axis=-1,
        )

    value_network = FeedForwardNetwork(init=_value_init, apply=_value_apply)

    # ------------------------------------------------------------------
    # Policy network
    # params = (decoder_params, action_head_params)
    # apply  : (normalizer_params, policy_params, obs_dict) → logits
    # ------------------------------------------------------------------

    def _policy_init(key):
        k1, k2 = jax.random.split(key)
        dec_params = policy_decoder.init(k1, dummy_policy_obs)
        feats = policy_decoder.apply(dec_params, dummy_policy_obs)
        head_params = policy_action_head.init(k2, feats)
        return dec_params, head_params

    def _policy_apply(pparams, params, obs):
        policy_obs = _preprocess_policy(obs, pparams)
        feats = policy_decoder.apply(params[0], policy_obs)
        return policy_action_head.apply(params[1], feats)

    policy_network = FeedForwardNetwork(init=_policy_init, apply=_policy_apply)

    # ------------------------------------------------------------------

    return PPONetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def make_inference_fn(ppo_networks: PPONetworks, compute_value: bool = False):
    """Returns a policy-factory compatible with brax PPO.

    Expected params layout passed by brax.ppo.train:
        params[0] = normalizer_params
        params[1] = policy_params  → (decoder_params, action_head_params)
        params[2] = value_params
    """

    def make_policy(
        params: types.Params, deterministic: bool = False
    ) -> types.Policy:

        def policy(
            observations: types.Observation, key_sample: PRNGKey
        ) -> Tuple[types.Action, types.Extra]:
            logits = ppo_networks.policy_network.apply(
                params[0], params[1], observations
            )
            if deterministic:
                return ppo_networks.parametric_action_distribution.mode(logits), {}

            raw_actions = (
                ppo_networks.parametric_action_distribution
                .sample_no_postprocessing(logits, key_sample)
            )
            log_prob = ppo_networks.parametric_action_distribution.log_prob(
                logits, raw_actions
            )
            postprocessed_actions = (
                ppo_networks.parametric_action_distribution.postprocess(raw_actions)
            )
            extras = {
                "log_prob": log_prob,
                "raw_action": raw_actions,
                "distribution_params": logits,
            }
            if compute_value:
                extras["value"] = ppo_networks.value_network.apply(
                    params[0], params[2], observations
                )
            return postprocessed_actions, extras

        return policy

    return make_policy
