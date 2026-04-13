"""SITT networks for vision-based landing RL + alignment.

Provides networks for both:
  - PPO training on teacher env (flat privileged obs)
  - Alignment on student env (pixels + proprio → teacher feature matching)

The underlying Flax modules are shared; PPO-side and align-side
FeedForwardNetworks differ only in their obs-key extraction and
normalizer handling.
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


# ── Encoder modules ─────────────────────────────────────────────────

class PolicyVisionProprioEncoder(nn.Module):
    """Two-branch policy encoder: [pixels→CNN and proprio→MLP projector]→fusion MLP."""

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


class PolicyProprioEncoder(nn.Module):
    """Debug encoder: ignores pixels, uses only proprio."""

    proprio_proj_hidden_layers: Sequence[int]
    fusion_hidden_layers: Sequence[int]
    activation: ActivationFn = nn.relu
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform()

    @nn.compact
    def __call__(self, pixels: jnp.ndarray, proprio: jnp.ndarray) -> jnp.ndarray:
        _ = pixels

        proprio_feats = MLP(
            layer_sizes=list(self.proprio_proj_hidden_layers),
            activation=self.activation,
            kernel_init=self.kernel_init,
            activate_final=True,
        )(proprio)

        return MLP(
            layer_sizes=list(self.fusion_hidden_layers),
            activation=self.activation,
            kernel_init=self.kernel_init,
            activate_final=False,
        )(proprio_feats)


# ── Helpers ──────────────────────────────────────────────────────────

def _split_path(path: str) -> Sequence[str]:
    return tuple(k for k in path.split("/") if k)


def _get_by_path(tree: Mapping[str, Any], path: str):
    value = tree
    for key in _split_path(path):
        value = value[key]
    return value


def _select_normalizer_by_path(pparams: Any, path: str):
    selected = pparams
    for key in _split_path(path):
        selected = normalizer_select(selected, key)
    return selected


def _shape_last_dim(shape_spec: Any) -> int:
    if isinstance(shape_spec, int):
        return shape_spec
    return int(jax.tree_util.tree_flatten(shape_spec)[0][-1])


# ── SITT Networks dataclass ──────────────────────────────────────────

@flax.struct.dataclass
class SITTNetworks:
    """All networks needed for SITT (RL + alignment).

    PPO side — operates on teacher env obs (policy_obs, value_obs):
        policy_network:     teacher_dec + action_head → logits
        value_network:      value MLP → scalar
        policy_decoder:     teacher_dec → features (for rl_align_loss)
        ppo_proxy_decoder:  proxy_dec → features (for KL reward shaping)

    Align side — operates on student env obs (pixels, proprio, teacher_obs):
        teacher_network:    teacher_dec + action_head → logits
        teacher_decoder:    teacher_dec → features
        student_network:    student_enc + action_head → logits
        student_encoder:    student_enc → features
        proxy_decoder:      proxy_dec → features (from teacher_obs key)

    Shared:
        action_head:        features → logits
        parametric_action_distribution
    """
    # PPO (teacher env, normalizer_select inside apply)
    policy_network: FeedForwardNetwork
    value_network: FeedForwardNetwork
    policy_decoder: FeedForwardNetwork
    ppo_proxy_decoder: FeedForwardNetwork

    # Align (student env, pre-extracted normalizer)
    teacher_network: FeedForwardNetwork
    teacher_decoder: FeedForwardNetwork
    student_network: FeedForwardNetwork
    student_encoder: FeedForwardNetwork
    proxy_decoder: FeedForwardNetwork

    # Shared
    action_head: FeedForwardNetwork
    parametric_action_distribution: distribution.ParametricDistribution


# ── Factory ──────────────────────────────────────────────────────────

def make_sitt_networks(
    observation_size: types.ObservationSize,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = (
        types.identity_observation_preprocessor
    ),
    # Student env obs shapes (needed for student encoder init)
    student_observation_size: Any = None,
    # Teacher MLP decoder (frozen from checkpoint, but trained further by PPO)
    teacher_dec_hidden_layers: Sequence[int] = (512, 512, 256, 128),
    # Student (vision) encoder
    policy_dec_hidden_layers: Sequence[int] = (256, 256),
    policy_proprio_proj_hidden_layers: Sequence[int] = (64,),
    cnn_num_filters: Sequence[int] = (32, 64, 64),
    cnn_kernel_sizes: Sequence[Tuple[int, int]] = ((8, 8), (4, 4), (3, 3)),
    cnn_strides: Sequence[Tuple[int, int]] = ((4, 4), (2, 2), (1, 1)),
    # Proxy decoder
    proxy_hidden_layers: Sequence[int] = (512, 512, 256, 128),
    # Shared action head
    action_hidden_layer_sizes: Sequence[int] = (64,),
    # Value MLP
    value_hidden_layer_sizes: Sequence[int] = (256, 256, 256),
    activation: ActivationFn = nn.tanh,
    # PPO obs keys (teacher env: {policy_obs, value_obs})
    policy_obs_key: str = "policy_obs",
    value_obs_key: str = "value_obs",
    # Align obs keys (student env: {pixels/view_*, proprio, teacher_obs, value_obs})
    teacher_obs_key: str = "teacher_obs",
    policy_pixels_key: str = "pixels/view_0",
    policy_proprio_key: str = "proprio_obs",
) -> SITTNetworks:
    """Build all networks for SITT training.

    Args:
        observation_size: obs shape dict from *teacher* env (for PPO init).
        student_observation_size: obs shape dict from *student* env (for
            student encoder init).  If None, student/align networks are
            created with dummy shapes.
    """

    kernel_init = jax.nn.initializers.lecun_uniform()
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )

    # ==================================================================
    # Flax modules (defined ONCE, shared by PPO and align apply fns)
    # ==================================================================

    teacher_decoder_mlp = MLP(
        layer_sizes=list(teacher_dec_hidden_layers),
        activation=activation,
        kernel_init=kernel_init,
        activate_final=True,
    )

    proxy_decoder_mlp = MLP(
        layer_sizes=list(proxy_hidden_layers),
        activation=activation,
        kernel_init=kernel_init,
        activate_final=True,
    )

    student_encoder_module = PolicyVisionProprioEncoder(
        cnn_num_filters=list(cnn_num_filters),
        cnn_kernel_sizes=list(cnn_kernel_sizes),
        cnn_strides=list(cnn_strides),
        proprio_proj_hidden_layers=list(policy_proprio_proj_hidden_layers),
        fusion_hidden_layers=list(policy_dec_hidden_layers),
        activation=activation,
        kernel_init=kernel_init,
    )

    # student_encoder_module = PolicyProprioEncoder(
    #     proprio_proj_hidden_layers=list(policy_proprio_proj_hidden_layers),
    #     fusion_hidden_layers=list(policy_dec_hidden_layers),
    #     activation=activation,
    #     kernel_init=kernel_init,
    # )

    action_head_mlp = MLP(
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

    # ==================================================================
    # Dummy inputs for init
    # ==================================================================

    # Teacher / PPO obs (flat privileged state)
    teacher_obs_size = _shape_last_dim(
        _get_by_path(observation_size, policy_obs_key)
        if isinstance(observation_size, Mapping) else observation_size
    )
    dummy_teacher_obs = jnp.zeros((1, teacher_obs_size))

    # Value obs
    value_obs_size = _shape_last_dim(
        _get_by_path(observation_size, value_obs_key)
        if isinstance(observation_size, Mapping) else observation_size
    )
    dummy_value_obs = jnp.zeros((1, value_obs_size))

    # Student encoder obs (from student env or inferred)
    if student_observation_size is not None:
        pixel_shape = tuple(student_observation_size[policy_pixels_key])
        proprio_size = _shape_last_dim(student_observation_size[policy_proprio_key])
    else:
        # Fallback: infer from teacher obs size (proprio == policy_obs dimension)
        pixel_shape = (64, 64, 3)  # placeholder
        proprio_size = teacher_obs_size

    dummy_pixels = jnp.zeros((1,) + pixel_shape)
    dummy_proprio = jnp.zeros((1, proprio_size))

    # Feature dummy (for action head init)
    _tmp_params = teacher_decoder_mlp.init(jax.random.PRNGKey(0), dummy_teacher_obs)
    _dummy_feats = teacher_decoder_mlp.apply(_tmp_params, dummy_teacher_obs)

    # ==================================================================
    # PPO-side preprocessing (normalizer_select inside)
    # ==================================================================

    def _ppo_preprocess(obs, pparams, key):
        """Extract obs by key and normalise using normalizer_select."""
        if isinstance(obs, Mapping):
            return preprocess_observations_fn(
                obs[key], normalizer_select(pparams, key)
            )
        return preprocess_observations_fn(obs, pparams)

    # ==================================================================
    # Align-side preprocessing (pre-extracted normalizer)
    # ==================================================================

    def _preprocess_teacher(obs, pparams):
        """pparams is already the per-key sub-state (pre-extracted)."""
        teacher_obs = obs[teacher_obs_key] if isinstance(obs, Mapping) else obs
        return preprocess_observations_fn(teacher_obs, pparams)

    def _preprocess_value(obs, pparams):
        if isinstance(obs, Mapping):
            return preprocess_observations_fn(
                obs[value_obs_key], normalizer_select(pparams, value_obs_key)
            )
        return preprocess_observations_fn(obs, pparams)

    def _extract_student_obs(obs):
        return obs[policy_pixels_key], obs[policy_proprio_key]

    def _preprocess_student_proprio(obs, pparams):
        """pparams is the proprio RunningStatisticsState directly."""
        proprio = obs[policy_proprio_key] if isinstance(obs, Mapping) else obs
        return preprocess_observations_fn(proprio, pparams)

    # ==================================================================
    # PPO-side FeedForwardNetworks
    # ==================================================================

    # ─ Policy network: teacher_dec + action_head (uses policy_obs_key) ─
    def _ppo_policy_init(key):
        k1, k2 = jax.random.split(key)
        dec_params = teacher_decoder_mlp.init(k1, dummy_teacher_obs)
        feats = teacher_decoder_mlp.apply(dec_params, dummy_teacher_obs)
        head_params = action_head_mlp.init(k2, feats)
        return dec_params, head_params

    def _ppo_policy_apply(pparams, params, obs):
        feats = teacher_decoder_mlp.apply(
            params[0], _ppo_preprocess(obs, pparams, policy_obs_key)
        )
        return action_head_mlp.apply(params[1], feats)

    ppo_policy_network = FeedForwardNetwork(
        init=_ppo_policy_init, apply=_ppo_policy_apply
    )

    # ─ Value network (uses value_obs_key, normalizer_select inside) ─
    ppo_value_network = FeedForwardNetwork(
        init=lambda key: value_mlp.init(key, dummy_value_obs),
        apply=lambda pparams, params, obs: jnp.squeeze(
            value_mlp.apply(params, _ppo_preprocess(obs, pparams, value_obs_key)),
            axis=-1,
        ),
    )

    # ─ Policy decoder only → features (uses policy_obs_key) ─
    ppo_policy_decoder = FeedForwardNetwork(
        init=lambda key: teacher_decoder_mlp.init(key, dummy_teacher_obs),
        apply=lambda pparams, params, obs: teacher_decoder_mlp.apply(
            params, _ppo_preprocess(obs, pparams, policy_obs_key)
        ),
    )

    # ─ Proxy decoder → features (uses policy_obs_key) ─
    ppo_proxy_decoder_net = FeedForwardNetwork(
        init=lambda key: proxy_decoder_mlp.init(key, dummy_teacher_obs),
        apply=lambda pparams, params, obs: proxy_decoder_mlp.apply(
            params, _ppo_preprocess(obs, pparams, policy_obs_key)
        ),
    )

    # ==================================================================
    # Align-side FeedForwardNetworks
    # ==================================================================

    # ─ Teacher network (dec + head, uses teacher_obs_key) ─
    def _align_teacher_init(key):
        k1, k2 = jax.random.split(key)
        dec_params = teacher_decoder_mlp.init(k1, dummy_teacher_obs)
        feats = teacher_decoder_mlp.apply(dec_params, dummy_teacher_obs)
        head_params = action_head_mlp.init(k2, feats)
        return dec_params, head_params

    def _align_teacher_apply(pparams, params, obs):
        feats = teacher_decoder_mlp.apply(
            params[0], _preprocess_teacher(obs, pparams)
        )
        return action_head_mlp.apply(params[1], feats)

    align_teacher_network = FeedForwardNetwork(
        init=_align_teacher_init, apply=_align_teacher_apply
    )

    # ─ Teacher decoder only → features (uses teacher_obs_key) ─
    align_teacher_decoder = FeedForwardNetwork(
        init=lambda key: teacher_decoder_mlp.init(key, dummy_teacher_obs),
        apply=lambda pparams, params, obs: teacher_decoder_mlp.apply(
            params, _preprocess_teacher(obs, pparams)
        ),
    )

    # ─ Student encoder → features ─
    align_student_encoder = FeedForwardNetwork(
        init=lambda key: student_encoder_module.init(key, dummy_pixels, dummy_proprio),
        apply=lambda pparams, params, obs: student_encoder_module.apply(
            params,
            _extract_student_obs(obs)[0],
            _preprocess_student_proprio(obs, pparams),
        ),
    )

    # ─ Student network (enc + head) → logits ─
    def _student_net_init(key):
        k1, k2 = jax.random.split(key)
        enc_params = student_encoder_module.init(k1, dummy_pixels, dummy_proprio)
        feats = student_encoder_module.apply(enc_params, dummy_pixels, dummy_proprio)
        head_params = action_head_mlp.init(k2, feats)
        return enc_params, head_params

    def _student_net_apply(pparams, params, obs):
        pixels, _ = _extract_student_obs(obs)
        proprio = _preprocess_student_proprio(obs, pparams)
        feats = student_encoder_module.apply(params[0], pixels, proprio)
        return action_head_mlp.apply(params[1], feats)

    align_student_network = FeedForwardNetwork(
        init=_student_net_init, apply=_student_net_apply
    )

    # ─ Proxy decoder → features (uses teacher_obs_key, pre-extracted norm) ─
    align_proxy_decoder = FeedForwardNetwork(
        init=lambda key: proxy_decoder_mlp.init(key, dummy_teacher_obs),
        apply=lambda pparams, params, obs: proxy_decoder_mlp.apply(
            params, _preprocess_teacher(obs, pparams)
        ),
    )

    # ─ Action head (shared, no normalizer) ─
    action_head_net = FeedForwardNetwork(
        init=lambda key: action_head_mlp.init(key, _dummy_feats),
        apply=lambda pparams, params, feats: action_head_mlp.apply(params, feats),
    )

    # ==================================================================
    return SITTNetworks(
        # PPO side
        policy_network=ppo_policy_network,
        value_network=ppo_value_network,
        policy_decoder=ppo_policy_decoder,
        ppo_proxy_decoder=ppo_proxy_decoder_net,
        # Align side
        teacher_network=align_teacher_network,
        teacher_decoder=align_teacher_decoder,
        student_network=align_student_network,
        student_encoder=align_student_encoder,
        proxy_decoder=align_proxy_decoder,
        # Shared
        action_head=action_head_net,
        parametric_action_distribution=parametric_action_distribution,
    )


# Backward compat alias
ILNetworks = SITTNetworks
make_il_networks = make_sitt_networks


# ── Inference helpers ────────────────────────────────────────────────

def make_inference_fn(sitt_networks: SITTNetworks, compute_value: bool = False):
    """PPO-compatible teacher policy inference.

    Expected params: ``(normalizer_params, policy_params, value_params)``.
    """

    def make_policy(
        params: types.Params, deterministic: bool = False
    ) -> types.Policy:

        def policy(
            observations: types.Observation, key_sample: PRNGKey
        ) -> Tuple[types.Action, types.Extra]:
            logits = sitt_networks.policy_network.apply(
                params[0], params[1], observations
            )
            if deterministic:
                return sitt_networks.parametric_action_distribution.mode(logits), {}
            raw_actions = (
                sitt_networks.parametric_action_distribution
                .sample_no_postprocessing(logits, key_sample)
            )
            log_prob = sitt_networks.parametric_action_distribution.log_prob(
                logits, raw_actions
            )
            postprocessed = sitt_networks.parametric_action_distribution.postprocess(
                raw_actions
            )
            extras = {
                "log_prob": log_prob,
                "raw_action": raw_actions,
                "distribution_params": logits,
            }
            if compute_value:
                extras["value"] = sitt_networks.value_network.apply(
                    params[0], params[2], observations
                )
            return postprocessed, extras

        return policy

    return make_policy


def make_frozen_teacher_policy(
    sitt_networks: SITTNetworks,
    teacher_norm_params: Any,
    teacher_policy_params: Any,
    deterministic: bool = False,
) -> types.Policy:
    """Returns ``policy(obs, key)`` with all teacher params baked in.

    Uses the *align-side* teacher_network (teacher_obs_key) so that
    rollouts on the student env work correctly.
    ``teacher_norm_params`` should be pre-extracted via normalizer_select.
    """

    def policy(
        observations: types.Observation, key_sample: PRNGKey
    ) -> Tuple[types.Action, types.Extra]:
        logits = sitt_networks.teacher_network.apply(
            teacher_norm_params, teacher_policy_params, observations
        )
        if deterministic:
            return sitt_networks.parametric_action_distribution.mode(logits), {}
        raw_actions = (
            sitt_networks.parametric_action_distribution
            .sample_no_postprocessing(logits, key_sample)
        )
        log_prob = sitt_networks.parametric_action_distribution.log_prob(
            logits, raw_actions
        )
        postprocessed = sitt_networks.parametric_action_distribution.postprocess(
            raw_actions
        )
        return postprocessed, {
            "log_prob": log_prob,
            "raw_action": raw_actions,
            "distribution_params": logits,
        }

    return policy


def make_student_inference_fn(
    sitt_networks: SITTNetworks, action_head_params: Any
):
    """Student (vision) policy factory.

    ``action_head_params`` is a frozen constant closed over here.
    Expected params at call time: ``(proprio_norm, student_enc_params)``.
    """

    def make_policy(
        params: types.Params, deterministic: bool = False
    ) -> types.Policy:
        proprio_norm, student_enc = params

        def policy(
            observations: types.Observation, key_sample: PRNGKey
        ):
            logits = sitt_networks.student_network.apply(
                proprio_norm, (student_enc, action_head_params), observations
            )
            if deterministic:
                return sitt_networks.parametric_action_distribution.mode(logits), {}
            raw_actions = (
                sitt_networks.parametric_action_distribution
                .sample_no_postprocessing(logits, key_sample)
            )
            log_prob = sitt_networks.parametric_action_distribution.log_prob(
                logits, raw_actions
            )
            postprocessed = sitt_networks.parametric_action_distribution.postprocess(
                raw_actions
            )
            return postprocessed, {
                "log_prob": log_prob,
                "raw_action": raw_actions,
                "distribution_params": logits,
            }

        return policy

    return make_policy
