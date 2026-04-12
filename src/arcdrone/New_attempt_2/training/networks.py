"""Fully self-contained networks for DAgger vision-landing training.

All shared code (CNN, MLP, IL networks) is inlined here so that
New_attempt_2 does NOT depend on the arcdrone package at all.

Architecture (vel-in-the-loop):

    pixels → CNN → cnn_flat
    proprio → MLP → proprio_feats
    [cnn_flat, proprio_feats] → fusion MLP → encoder_feats

    vel_estimator: [encoder_feats, aux_tilt] → MLP → pred_linvel(3)
                   (supervised by ground-truth linvel during training)

    action_head:   [encoder_feats, pred_linvel] → MLP → actions
                   (pred_linvel flows into the action pipeline at inference!)

Checkpoint layout:
    (proprio_norm, (student_enc, action_head, vel_estimator))
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


ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]


# ===================================================================
# Common network building blocks
# ===================================================================

class MLP(nn.Module):
    layer_sizes: Sequence[int]
    activation: ActivationFn = nn.tanh
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
    activate_final: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        for i, size in enumerate(self.layer_sizes):
            x = nn.Dense(size, kernel_init=self.kernel_init, name=f'hidden_{i}')(x)
            if i != len(self.layer_sizes) - 1 or self.activate_final:
                x = self.activation(x)
        return x


class CNN(nn.Module):
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
                num_filter, kernel_size=kernel_size, strides=stride, use_bias=self.use_bias,
            )(hidden)
            hidden = self.activation(hidden)
        return hidden


def _get_obs_size(obs_size: types.ObservationSize, obs_key: str) -> int:
    obs_size = obs_size[obs_key] if isinstance(obs_size, Mapping) else obs_size
    return jax.tree_util.tree_flatten(obs_size)[0][-1]


# ===================================================================
# IL Networks
# ===================================================================

@flax.struct.dataclass
class FeedForwardNetwork:
    init: Callable[..., Any]
    apply: Callable[..., Any]


class PolicyVisionProprioEncoder(nn.Module):
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
    selected = pparams
    for key in _split_path(path):
        selected = normalizer_select(selected, key)
    return selected


def _shape_last_dim(shape_spec: Any) -> int:
    if isinstance(shape_spec, int):
        return shape_spec
    return int(jax.tree_util.tree_flatten(shape_spec)[0][-1])


@flax.struct.dataclass
class ILNetworks:
    teacher_network: FeedForwardNetwork
    student_network: FeedForwardNetwork
    value_network: FeedForwardNetwork
    teacher_decoder: FeedForwardNetwork
    student_encoder: FeedForwardNetwork
    action_head: FeedForwardNetwork
    teacher_action_head: FeedForwardNetwork
    vel_estimator: FeedForwardNetwork
    parametric_action_distribution: distribution.ParametricDistribution


def make_il_networks(
    observation_size: types.ObservationSize,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = (
        types.identity_observation_preprocessor
    ),
    teacher_dec_hidden_layers: Sequence[int] = (512, 512, 256, 128),
    policy_dec_hidden_layers: Sequence[int] = (256, 256),
    policy_proprio_proj_hidden_layers: Sequence[int] = (64,),
    cnn_num_filters: Sequence[int] = (32, 64, 64),
    cnn_kernel_sizes: Sequence[Tuple[int, int]] = ((8, 8), (4, 4), (3, 3)),
    cnn_strides: Sequence[Tuple[int, int]] = ((4, 4), (2, 2), (1, 1)),
    action_hidden_layer_sizes: Sequence[int] = (64,),
    value_hidden_layer_sizes: Sequence[int] = (256, 256, 256),
    activation: ActivationFn = nn.tanh,
    policy_pixels_key: str = "pixels/view_0",
    policy_proprio_key: str = "proprio_obs",
    teacher_obs_key: str = "teacher_obs",
    value_obs_key: str = "value_obs",
) -> ILNetworks:
    kernel_init = jax.nn.initializers.lecun_uniform()
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )

    teacher_decoder_mlp = MLP(
        layer_sizes=list(teacher_dec_hidden_layers),
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
    value_mlp = MLP(
        layer_sizes=list(value_hidden_layer_sizes) + [1],
        activation=activation,
        kernel_init=kernel_init,
    )

    # ── Vel estimator: [encoder_feats, aux_tilt] → pred_linvel(3) ──
    tilt_size = _shape_last_dim(observation_size.get("aux_tilt", (10,)))
    vel_estimator_mlp = MLP(
        layer_sizes=[64, 32, 3],
        activation=nn.relu,
        kernel_init=kernel_init,
    )

    # ── Dummy inputs ──
    teacher_obs_raw = (
        _get_by_path(observation_size, teacher_obs_key)
        if isinstance(observation_size, Mapping)
        else observation_size
    )
    teacher_obs_size = _shape_last_dim(teacher_obs_raw)
    dummy_teacher_obs = jnp.zeros((1, teacher_obs_size))

    value_obs_raw = (
        _get_by_path(observation_size, value_obs_key)
        if isinstance(observation_size, Mapping)
        else observation_size
    )
    value_obs_size = _shape_last_dim(value_obs_raw)
    dummy_value_obs = jnp.zeros((1, value_obs_size))

    pixel_shape = tuple(observation_size[policy_pixels_key])
    proprio_size = _shape_last_dim(observation_size[policy_proprio_key])
    dummy_pixels = jnp.zeros((1,) + pixel_shape)
    dummy_proprio = jnp.zeros((1, proprio_size))

    _tmp_enc_params = student_encoder_module.init(jax.random.PRNGKey(0), dummy_pixels, dummy_proprio)
    _dummy_enc_feats = student_encoder_module.apply(_tmp_enc_params, dummy_pixels, dummy_proprio)
    encoder_feat_dim = _dummy_enc_feats.shape[-1]

    # Teacher features have the same role as encoder features for action head sizing
    _tmp_params = teacher_decoder_mlp.init(jax.random.PRNGKey(0), dummy_teacher_obs)
    _dummy_teacher_feats = teacher_decoder_mlp.apply(_tmp_params, dummy_teacher_obs)

    # Vel estimator input: encoder_feats + tilt history
    _dummy_vel_input = jnp.zeros((1, encoder_feat_dim + tilt_size))
    # Action head input: encoder_feats + predicted linvel(3)
    action_head_mlp = MLP(
        layer_sizes=list(action_hidden_layer_sizes)
            + [parametric_action_distribution.param_size],
        activation=activation,
        kernel_init=kernel_init,
    )
    _dummy_action_input = jnp.zeros((1, encoder_feat_dim + 3))
    # Teacher action head: same architecture but takes teacher_feats (no vel concat)
    teacher_action_head_mlp = MLP(
        layer_sizes=list(action_hidden_layer_sizes)
            + [parametric_action_distribution.param_size],
        activation=activation,
        kernel_init=kernel_init,
    )

    # ── Preprocessing helpers ──

    def _preprocess_teacher(obs, pparams):
        teacher_obs = obs[teacher_obs_key] if isinstance(obs, Mapping) else obs
        return preprocess_observations_fn(teacher_obs, pparams)

    def _preprocess_value(obs, pparams):
        if isinstance(obs, Mapping):
            value_obs = obs[value_obs_key]
            return preprocess_observations_fn(
                value_obs, normalizer_select(pparams, value_obs_key)
            )
        return preprocess_observations_fn(obs, pparams)

    def _extract_student_obs(obs):
        pixels = obs[policy_pixels_key]
        return pixels, obs[policy_proprio_key]

    def _preprocess_student_proprio(obs, pparams):
        proprio = obs[policy_proprio_key] if isinstance(obs, Mapping) else obs
        return preprocess_observations_fn(proprio, pparams)

    def _get_aux_tilt(obs):
        return obs.get("aux_tilt", jnp.zeros(tilt_size))

    # ── FeedForwardNetwork wrappers ──

    teacher_decoder_net = FeedForwardNetwork(
        init=lambda key: teacher_decoder_mlp.init(key, dummy_teacher_obs),
        apply=lambda pparams, params, obs: teacher_decoder_mlp.apply(
            params, _preprocess_teacher(obs, pparams)
        ),
    )

    student_encoder_net = FeedForwardNetwork(
        init=lambda key: student_encoder_module.init(key, dummy_pixels, dummy_proprio),
        apply=lambda pparams, params, obs: student_encoder_module.apply(
            params,
            _extract_student_obs(obs)[0],
            _preprocess_student_proprio(obs, pparams),
        ),
    )

    vel_estimator_net = FeedForwardNetwork(
        init=lambda key: vel_estimator_mlp.init(key, _dummy_vel_input),
        apply=lambda pparams, params, feats_and_tilt: vel_estimator_mlp.apply(
            params, feats_and_tilt
        ),
    )

    action_head_net = FeedForwardNetwork(
        init=lambda key: action_head_mlp.init(key, _dummy_action_input),
        apply=lambda pparams, params, feats_and_vel: action_head_mlp.apply(
            params, feats_and_vel
        ),
    )

    # Teacher action head: takes teacher_feat_dim (no vel concat)
    teacher_action_head_net = FeedForwardNetwork(
        init=lambda key: teacher_action_head_mlp.init(key, _dummy_teacher_feats),
        apply=lambda pparams, params, feats: teacher_action_head_mlp.apply(
            params, feats
        ),
    )

    # ── Teacher network (unchanged: decoder → teacher action head) ──

    def _teacher_net_init(key):
        k1, k2 = jax.random.split(key)
        dec_params = teacher_decoder_mlp.init(k1, dummy_teacher_obs)
        feats = teacher_decoder_mlp.apply(dec_params, dummy_teacher_obs)
        head_params = teacher_action_head_mlp.init(k2, feats)
        return dec_params, head_params

    def _teacher_net_apply(pparams, params, obs):
        feats = teacher_decoder_mlp.apply(params[0], _preprocess_teacher(obs, pparams))
        return teacher_action_head_mlp.apply(params[1], feats)

    teacher_network = FeedForwardNetwork(init=_teacher_net_init, apply=_teacher_net_apply)

    # ── Student network (vel-in-the-loop) ──
    # params = (enc_params, action_head_params, vel_estimator_params)

    def _student_net_init(key):
        k1, k2, k3 = jax.random.split(key, 3)
        enc_params = student_encoder_module.init(k1, dummy_pixels, dummy_proprio)
        head_params = action_head_mlp.init(k2, _dummy_action_input)
        vel_params = vel_estimator_mlp.init(k3, _dummy_vel_input)
        return enc_params, head_params, vel_params

    def _student_net_apply(pparams, params, obs):
        enc_params, head_params, vel_params = params
        pixels, _ = _extract_student_obs(obs)
        proprio = _preprocess_student_proprio(obs, pparams)
        encoder_feats = student_encoder_module.apply(enc_params, pixels, proprio)

        # Stage 1: estimate velocity from encoder features + tilt history
        tilt = _get_aux_tilt(obs)
        vel_input = jnp.concatenate([encoder_feats, tilt], axis=-1)
        pred_linvel = vel_estimator_mlp.apply(vel_params, vel_input)

        # Stage 2: action from encoder features + predicted velocity
        action_input = jnp.concatenate([encoder_feats, pred_linvel], axis=-1)
        return action_head_mlp.apply(head_params, action_input)

    student_network = FeedForwardNetwork(init=_student_net_init, apply=_student_net_apply)

    value_network = FeedForwardNetwork(
        init=lambda key: value_mlp.init(key, dummy_value_obs),
        apply=lambda pparams, params, obs: jnp.squeeze(
            value_mlp.apply(params, _preprocess_value(obs, pparams)), axis=-1,
        ),
    )

    return ILNetworks(
        teacher_network=teacher_network,
        student_network=student_network,
        value_network=value_network,
        teacher_decoder=teacher_decoder_net,
        student_encoder=student_encoder_net,
        action_head=action_head_net,
        teacher_action_head=teacher_action_head_net,
        vel_estimator=vel_estimator_net,
        parametric_action_distribution=parametric_action_distribution,
    )


def make_frozen_teacher_policy(
    il_networks: ILNetworks,
    teacher_norm_params: Any,
    teacher_policy_params: Any,
    deterministic: bool = False,
) -> types.Policy:
    def policy(observations: types.Observation, key_sample: PRNGKey):
        logits = il_networks.teacher_network.apply(
            teacher_norm_params, teacher_policy_params, observations
        )
        if deterministic:
            return il_networks.parametric_action_distribution.mode(logits), {}
        raw_actions = (
            il_networks.parametric_action_distribution
            .sample_no_postprocessing(logits, key_sample)
        )
        log_prob = il_networks.parametric_action_distribution.log_prob(
            logits, raw_actions
        )
        postprocessed = il_networks.parametric_action_distribution.postprocess(raw_actions)
        return postprocessed, {
            "log_prob": log_prob,
            "raw_action": raw_actions,
            "distribution_params": logits,
        }
    return policy


# ---------------------------------------------------------------------------
# DAgger-specific inference helper — vel-in-the-loop
# ---------------------------------------------------------------------------

def make_student_inference_fn(il_networks: ILNetworks):
    """Student (vision) policy factory — vel-in-the-loop.

    The student_network.apply runs encoder → vel_estimator → action_head
    as a single forward pass.  The predicted linvel flows into the action
    pipeline at both training AND inference time.

    Expected params at call time (from _pack_student_params):
        params[0] = proprio_norm
        params[1] = student_enc_params
        params[2] = student_action_head_params
        params[3] = vel_estimator_params

    Checkpoint layout (saved by train.py):
        (proprio_norm, (student_enc, action_head, vel_estimator))
    """

    def make_policy(params: types.Params, deterministic: bool = False) -> types.Policy:
        proprio_norm, student_enc, action_head, vel_estimator = params

        def policy(observations: types.Observation, key_sample: PRNGKey):
            logits = il_networks.student_network.apply(
                proprio_norm, (student_enc, action_head, vel_estimator), observations
            )
            if deterministic:
                return il_networks.parametric_action_distribution.mode(logits), {}
            raw_actions = (
                il_networks.parametric_action_distribution
                .sample_no_postprocessing(logits, key_sample)
            )
            log_prob = il_networks.parametric_action_distribution.log_prob(
                logits, raw_actions
            )
            postprocessed = il_networks.parametric_action_distribution.postprocess(
                raw_actions
            )
            return postprocessed, {
                "log_prob": log_prob,
                "raw_action": raw_actions,
                "distribution_params": logits,
            }

        return policy

    return make_policy
