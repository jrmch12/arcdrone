"""PPO networks for vision-based landing RL.

**CNN feature-caching changes (arcdrone-specific)**

1. ``extract_cnn_features``  — runs the frozen CNN + global-avg-pool on an
   obs dict, returning a flat feature vector per sample.  Called *once* per
   rollout scan iteration inside the forked ``train.py``.
2. ``compact_obs`` / ``strip_pixels`` — replace pixel tensors in an obs dict
   with the pre-computed ``cnn_feats`` key so that the Transition accumulated
   across scan iterations is compact.
3. ``_policy_apply`` dual-path — if obs contains ``cnn_feats`` (SGD path)
   skip the CNN and feed features directly into fusion+head.  If obs contains
   ``pixels/*`` (rollout acting path) run the full encoder as before.
4. ``stop_gradient`` on the CNN is removed: the CNN is structurally excluded
   from the differentiable SGD graph since it never runs during loss
   computation.
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

from arcdrone.common.networks import CNN, MLP, _get_obs_size


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


class PolicyVisionProprioEncoder(nn.Module):
    """Two-branch policy encoder: [pixels->CNN and proprio->MLP projector]->fusion MLP."""

    cnn_num_filters: Sequence[int]
    cnn_kernel_sizes: Sequence[Tuple[int, int]]
    cnn_strides: Sequence[Tuple[int, int]]
    proprio_proj_hidden_layers: Sequence[int]
    fusion_hidden_layers: Sequence[int]
    activation: ActivationFn = nn.relu
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform()

    @nn.compact
    def __call__(self, pixels: jnp.ndarray, proprio: jnp.ndarray) -> jnp.ndarray:
        cnn_out = CNN(
            num_filters=self.cnn_num_filters,
            kernel_sizes=self.cnn_kernel_sizes,
            strides=self.cnn_strides,
            activation=self.activation,
            use_bias=False,
        )(pixels)
        
        cnn_feats = jnp.mean(cnn_out, axis=(-2, -3))

        proprio_feats = MLP(
            layer_sizes=list(self.proprio_proj_hidden_layers),
            activation=self.activation,
            kernel_init=self.kernel_init,
            activate_final=True,
        )(proprio)

        fused = jnp.concatenate([cnn_feats, proprio_feats], axis=-1)
        return MLP(
            layer_sizes=list(self.fusion_hidden_layers),
            activation=self.activation,
            kernel_init=self.kernel_init,
            activate_final=False,
        )(fused)


def _split_path(path: str) -> Sequence[str]:
    return tuple(k for k in path.split("/") if k)


def _get_by_path(tree: Mapping[str, Any], path: str):
    value = tree
    for key in _split_path(path):
        value = value[key]
    return value


def _select_normalizer_by_path(pparams: Any, path: str):
    # Backwards-compat: older checkpoints may store a flat RunningStatisticsState
    # instead of a dict keyed by obs name. In that case, just return as-is.
    mean = getattr(pparams, "mean", None)
    if not isinstance(mean, Mapping):
        return pparams
    selected = pparams
    for key in _split_path(path):
        selected = normalizer_select(selected, key)
    return selected


def _shape_last_dim(shape_spec: Any) -> int:
    if isinstance(shape_spec, int):
        return shape_spec
    return int(jax.tree_util.tree_flatten(shape_spec)[0][-1])


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_ppo_networks_vision(
    observation_size: types.ObservationSize,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = (
        types.identity_observation_preprocessor
    ),
    # Fusion decoder layers after concatenating CNN and proprio projector features.
    policy_dec_hidden_layers: Sequence[int] = (256, 256),
    # Proprio projection branch (MLP) before fusion.
    policy_proprio_proj_hidden_layers: Sequence[int] = (64,),
    # Action head on top of the policy backbone features
    action_hidden_layer_sizes: Sequence[int] = (64,),
    value_hidden_layer_sizes: Sequence[int] = (256, 256, 256),
    cnn_num_filters: Sequence[int] = (32, 64, 64),
    cnn_kernel_sizes: Sequence[Tuple[int, int]] = ((8, 8), (4, 4), (3, 3)),
    cnn_strides: Sequence[Tuple[int, int]] = ((4, 4), (2, 2), (1, 1)),
    activation: ActivationFn = nn.tanh,
    # Top-level key for pixel tensor (must start with 'pixels/' for Brax normalizer compat).
    policy_pixels_key: str = "pixels/view_0",
    policy_pixels_key_1: str = "pixels/view_1",
    policy_pixels_key_2: str = "pixels/view_2",
    # Top-level key for proprio tensor.
    policy_proprio_key: str = "proprio_obs",
    # Top-level key for value vector.
    value_obs_key: str = "value_obs",
) -> PPONetworks:
    """Build two-stream vision-policy + privileged-state-value PPO networks."""

    kernel_init = jax.nn.initializers.lecun_uniform()

    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )

    # ------------------------------------------------------------------
    # Modules
    # ------------------------------------------------------------------

    policy_encoder = PolicyVisionProprioEncoder(
        cnn_num_filters=list(cnn_num_filters),
        cnn_kernel_sizes=list(cnn_kernel_sizes),
        cnn_strides=list(cnn_strides),
        proprio_proj_hidden_layers=list(policy_proprio_proj_hidden_layers),
        fusion_hidden_layers=list(policy_dec_hidden_layers),
        activation=activation,
        kernel_init=kernel_init,
    )

    policy_action_head = MLP(
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


    # ------------------------------------------------------------------
    # Dummy inputs for init
    # ------------------------------------------------------------------

    # Flat obs structure: pixels/view_* and proprio_obs are top-level keys
    pixel_shape_0 = tuple(observation_size[policy_pixels_key])
    pixel_shape_1 = tuple(observation_size[policy_pixels_key_1])
    pixel_shape_2 = tuple(observation_size[policy_pixels_key_2])
    pixel_shape = pixel_shape_0[:-1] + (
        pixel_shape_0[-1] + pixel_shape_1[-1] + pixel_shape_2[-1],
    )
    proprio_obs_size = _shape_last_dim(observation_size[policy_proprio_key])
    dummy_pixels = jnp.zeros((1,) + pixel_shape)
    dummy_proprio_obs = jnp.zeros((1, proprio_obs_size))

    # Value dummy input
    value_obs_size = _get_obs_size(_get_by_path(observation_size, value_obs_key), "")
    dummy_value_obs = jnp.zeros((1, value_obs_size))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _preprocess_value(obs, pparams):
        if isinstance(obs, Mapping):
            value_obs = _get_by_path(obs, value_obs_key)
            return preprocess_observations_fn(
                value_obs,
                _select_normalizer_by_path(pparams, value_obs_key),
            )
        return preprocess_observations_fn(obs, pparams)

    def _extract_policy_obs(obs):
        # Flat structure: concatenate pixels/view_* along channel axis
        pixels = jnp.concatenate([
            obs[policy_pixels_key],
            obs[policy_pixels_key_1],
            obs[policy_pixels_key_2],
        ], axis=-1)
        return pixels, obs[policy_proprio_key]

    def _preprocess_policy_proprio_obs(obs, pparams):
        _, proprio = _extract_policy_obs(obs)
        return preprocess_observations_fn(
            proprio,
            _select_normalizer_by_path(pparams, policy_proprio_key),
        )

    # ------------------------------------------------------------------
    # Value network init / apply
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
    # Params = (vision_decoder_params, action_head_params)
    # apply : (normalizer_params, policy_params, obs_dict) → logits
    # ------------------------------------------------------------------

    def _policy_init(key):
        k1, k2 = jax.random.split(key)
        dec_params = policy_encoder.init(k1, dummy_pixels, dummy_proprio_obs)
        feats = policy_encoder.apply(dec_params, dummy_pixels, dummy_proprio_obs)
        head_params = policy_action_head.init(k2, feats)
        return dec_params, head_params

    # Previous _policy_apply (always runs CNN on pixels):
    # def _policy_apply(pparams, params, obs):
    #     pixels, _ = _extract_policy_obs(obs)
    #     proprio = _preprocess_policy_proprio_obs(obs, pparams)
    #     feats = policy_encoder.apply(params[0], pixels, proprio)
    #     return policy_action_head.apply(params[1], feats)

    def _preprocess_proprio_only(obs, pparams):
        """Normalize proprio without touching pixel keys."""
        proprio = obs[policy_proprio_key]
        return preprocess_observations_fn(
            proprio,
            _select_normalizer_by_path(pparams, policy_proprio_key),
        )

    def _policy_apply(pparams, params, obs):
        """Dual-path policy forward.

        * **Rollout (acting)** — obs contains ``pixels/view_*`` keys.
          Runs the full CNN encoder.
        * **SGD (loss)** — obs contains ``cnn_feats`` key (pre-computed).
          Skips CNN, feeds cached features directly into fusion+head.
        """
        if isinstance(obs, Mapping) and "cnn_feats" in obs:
            # ── Cached-feature path (SGD) ──
            proprio = _preprocess_proprio_only(obs, pparams)
            cnn_feats = obs["cnn_feats"]
            # Fuse with proprio and run through fusion MLP + action head.
            # This mirrors the tail of PolicyVisionProprioEncoder.__call__.
            proprio_feats = MLP(
                layer_sizes=list(policy_encoder.proprio_proj_hidden_layers),
                activation=policy_encoder.activation,
                kernel_init=policy_encoder.kernel_init,
                activate_final=True,
            ).apply({"params": params[0]["params"]["MLP_0"]}, proprio)  # MLP_0 = proprio proj
            fused = jnp.concatenate([cnn_feats, proprio_feats], axis=-1)
            encoder_out = MLP(
                layer_sizes=list(policy_encoder.fusion_hidden_layers),
                activation=policy_encoder.activation,
                kernel_init=policy_encoder.kernel_init,
                activate_final=False,
            ).apply({"params": params[0]["params"]["MLP_1"]}, fused)    # MLP_1 = fusion
        else:
            # ── Full-encoder path (rollout acting) ──
            proprio = _preprocess_policy_proprio_obs(obs, pparams)
            pixels, _ = _extract_policy_obs(obs)
            encoder_out = policy_encoder.apply(params[0], pixels, proprio)

        return policy_action_head.apply(params[1], encoder_out)

    policy_network = FeedForwardNetwork(init=_policy_init, apply=_policy_apply)

    # ------------------------------------------------------------------

    return PPONetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )


# ---------------------------------------------------------------------------
# CNN feature-caching helpers (used by the forked train loop)
# ---------------------------------------------------------------------------

def extract_cnn_features(
    encoder_params,
    obs: Mapping[str, jnp.ndarray],
    policy_pixels_key: str = "pixels/view_0",
    policy_pixels_key_1: str = "pixels/view_1",
    policy_pixels_key_2: str = "pixels/view_2",
    cnn_num_filters: Sequence[int] = (32, 64, 64),
    cnn_kernel_sizes: Sequence[Tuple[int, int]] = ((8, 8), (4, 4), (3, 3)),
    cnn_strides: Sequence[Tuple[int, int]] = ((4, 4), (2, 2), (1, 1)),
    activation: ActivationFn = nn.tanh,
) -> jnp.ndarray:
    """Run the frozen CNN on pixel observations and return pooled features.

    This is called *once* per rollout scan iteration — never inside the
    differentiable loss computation.

    Args:
        encoder_params: The ``params[0]`` policy-encoder Flax params
            (same tree used by ``PolicyVisionProprioEncoder``).
        obs: Observation dict containing ``pixels/view_*`` keys.

    Returns:
        cnn_feats with shape ``(..., num_filters[-1])``.
    """
    pixels = jnp.concatenate([
        obs[policy_pixels_key],
        obs[policy_pixels_key_1],
        obs[policy_pixels_key_2],
    ], axis=-1)
    cnn_module = CNN(
        num_filters=list(cnn_num_filters),
        kernel_sizes=list(cnn_kernel_sizes),
        strides=list(cnn_strides),
        activation=activation,
        use_bias=False,
    )
    cnn_out = cnn_module.apply({"params": encoder_params["params"]["CNN_0"]}, pixels)
    return jnp.mean(cnn_out, axis=(-2, -3))  # global average pool


def compact_obs(
    obs: Mapping[str, jnp.ndarray],
    cnn_feats: jnp.ndarray,
) -> Mapping[str, jnp.ndarray]:
    """Replace pixel tensors with pre-computed CNN features.

    Drops all ``pixels/*`` keys and adds a ``cnn_feats`` key.
    """
    return {
        k: v for k, v in obs.items() if not k.startswith("pixels/")
    } | {"cnn_feats": cnn_feats}


def strip_pixels(
    obs: Mapping[str, jnp.ndarray],
) -> Mapping[str, jnp.ndarray]:
    """Drop pixel keys from the obs dict (used for next_observation)."""
    return {k: v for k, v in obs.items() if not k.startswith("pixels/")}


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
