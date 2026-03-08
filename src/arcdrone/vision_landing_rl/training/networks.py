"""PPO networks for vision-based landing RL.

Architecture
------------
Policy  : NatureCNN on ``policy_state_pixels``
          + concat flat ``policy_state_propio`` (action history)
          → MLP body (policy_hidden_layers)
          → action_head MLP (action_hidden_layers)
          → logits
Value   : plain MLP on ``value_state`` (privileged flat obs) → scalar

Both networks are Brax-PPO-compatible via the FeedForwardNetwork / PPONetworks
dataclasses.  Building blocks (MLP, CNN, VisionDecoder) live in
arrowdrone.common.networks so they can be reused by other tasks.
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

from arcdrone.common.networks import MLP, VisionDecoder, _get_obs_size


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


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_ppo_networks_vision(
    observation_size: types.ObservationSize,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = (
        types.identity_observation_preprocessor
    ),
    # Vision policy backbone (MLP layers *after* the CNN)
    policy_hidden_layer_sizes: Sequence[int] = (256, 256),
    # Action head on top of the policy backbone features
    action_hidden_layer_sizes: Sequence[int] = (64,),
    # Value network. When value_obs_key starts with "pixels/" a separate
    # VisionDecoder + linear head is used; otherwise a plain MLP on a flat
    # state vector is used (e.g. when privileged_state is in the obs dict).
    value_hidden_layer_sizes: Sequence[int] = (256, 256, 256, 256, 256),
    activation: ActivationFn = nn.relu,
    # Key for the pixel observation fed to the CNN.
    policy_obs_key: str = "policy_state_pixels",
    # Key for the proprioceptive state (e.g. action history) concatenated
    # with CNN features before the MLP body.  Set to "" to disable.
    policy_propio_key: str = "policy_state_propio",
    value_obs_key: str = "value_state",
) -> PPONetworks:
    """Build two-stream vision-policy + privileged-state-value PPO networks.

    Policy params layout  : (vision_decoder_params, action_head_params)
    Value params layout   : value_mlp_params  (plain MLP, no vision)
    Brax PPO params order : (normalizer_params, policy_params, value_params)
    """

    kernel_init = jax.nn.initializers.lecun_uniform()

    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )

    # ------------------------------------------------------------------
    # Modules
    # ------------------------------------------------------------------

    vision_decoder = VisionDecoder(
        layer_sizes=list(policy_hidden_layer_sizes),
        activation=activation,
        kernel_init=kernel_init,
        pixels_obs_key=policy_obs_key,     # explicit pixel key
        state_obs_key=policy_propio_key,   # propio concat; "" = disabled
    )

    action_head = MLP(
        layer_sizes=list(action_hidden_layer_sizes)
            + [parametric_action_distribution.param_size],
        activation=activation,
        kernel_init=kernel_init,
    )

    # Value backbone: VisionDecoder when value uses pixels, plain MLP otherwise.
    # This lets you switch to privileged state later just by changing value_obs_key.
    _value_uses_vision = value_obs_key.startswith("pixels/")

    if _value_uses_vision:
        value_backbone = VisionDecoder(
            layer_sizes=list(value_hidden_layer_sizes),
            activation=activation,
            kernel_init=kernel_init,
            state_obs_key="",
        )
        # Linear head → scalar
        value_head = MLP(layer_sizes=[1], activation=activation, kernel_init=kernel_init)
    else:
        value_backbone = MLP(
            layer_sizes=list(value_hidden_layer_sizes) + [1],
            activation=activation,
            kernel_init=kernel_init,
        )
        value_head = None

    # ------------------------------------------------------------------
    # Dummy inputs for init
    # ------------------------------------------------------------------

    # Policy dummy obs: pixels + propio (if enabled)
    pixel_shape = observation_size[policy_obs_key]   # e.g. (64, 64, 5)
    dummy_policy_obs = {policy_obs_key: jnp.zeros((1,) + tuple(pixel_shape))}
    if policy_propio_key:
        propio_size = _get_obs_size(observation_size, policy_propio_key)
        dummy_policy_obs[policy_propio_key] = jnp.zeros((1, propio_size))

    # Value dummy input
    if _value_uses_vision:
        val_shape = observation_size[value_obs_key]  # (H, W, C)
        dummy_value_obs = {value_obs_key: jnp.zeros((1,) + tuple(val_shape))}
    else:
        value_obs_size = _get_obs_size(observation_size, value_obs_key)
        dummy_value_obs = jnp.zeros((1, value_obs_size))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _preprocess_state(obs, pparams):
        """Normalise the flat state slice used by the value network (state-only path)."""
        if isinstance(obs, Mapping):
            return preprocess_observations_fn(
                obs[value_obs_key], normalizer_select(pparams, value_obs_key)
            )
        return preprocess_observations_fn(obs, pparams)

    # ------------------------------------------------------------------
    # Value network init / apply
    # ------------------------------------------------------------------

    if _value_uses_vision:
        def _value_init(key):
            k1, k2 = jax.random.split(key)
            bb_params = value_backbone.init(k1, dummy_value_obs)
            feats = value_backbone.apply(bb_params, dummy_value_obs)
            head_params = value_head.init(k2, feats)
            return bb_params, head_params

        def _value_apply(pparams, params, obs):
            if not isinstance(obs, Mapping):
                obs = {value_obs_key: obs}
            feats = value_backbone.apply(params[0], obs)
            return jnp.squeeze(value_head.apply(params[1], feats), axis=-1)
    else:
        def _value_init(key):
            return value_backbone.init(key, dummy_value_obs)

        def _value_apply(pparams, params, obs):
            return jnp.squeeze(
                value_backbone.apply(params, _preprocess_state(obs, pparams)),
                axis=-1,
            )

    # ------------------------------------------------------------------
    # Policy network
    # Params = (vision_decoder_params, action_head_params)
    # apply : (normalizer_params, policy_params, obs_dict) → logits
    # ------------------------------------------------------------------

    def _policy_init(key):
        k1, k2 = jax.random.split(key)
        dec_params = vision_decoder.init(k1, dummy_policy_obs)
        feats = vision_decoder.apply(dec_params, dummy_policy_obs)
        head_params = action_head.init(k2, feats)
        return dec_params, head_params

    def _policy_apply(pparams, params, obs):
        # Pixels are already in [-0.5, 0.5] from the env — no normalisation.
        # obs must be a dict with policy_obs_key (pixels) and optionally
        # policy_propio_key (action history).
        feats = vision_decoder.apply(params[0], obs)
        return action_head.apply(params[1], feats)

    policy_network = FeedForwardNetwork(init=_policy_init, apply=_policy_apply)

    value_network = FeedForwardNetwork(init=_value_init, apply=_value_apply)

    # ------------------------------------------------------------------

    return PPONetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )


# ---------------------------------------------------------------------------
# Inference helper (standard PPO, no student/proxy)
# ---------------------------------------------------------------------------

def make_inference_fn(ppo_networks: PPONetworks, compute_value: bool = False):
    """Returns a policy-factory compatible with brax PPO.

    Expected params layout passed by brax.ppo.train:
        params[0] = normalizer_params
        params[1] = policy_params   → (vision_decoder_params, action_head_params)
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
