"""Fully self-contained networks for DAgger vision-landing training.

All shared code (CNN, MLP, IL networks) is inlined here so that
New_attempt_2 does NOT depend on the arcdrone package at all.

DAgger-specific: make_student_inference_fn — the action head is fully
trainable, passed as part of params (not frozen from teacher).
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
# Common network building blocks (from arcdrone.common.networks)
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
# IL Networks (from arcdrone.vision_landing_il.training.networks)
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


class PolicyProprioEncoder(nn.Module):
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

    # Dummy inputs for init
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

    pixel_shape_0 = tuple(observation_size[policy_pixels_key])
    pixel_shape = pixel_shape_0
    proprio_size = _shape_last_dim(observation_size[policy_proprio_key])
    dummy_pixels = jnp.zeros((1,) + pixel_shape)
    dummy_proprio = jnp.zeros((1, proprio_size))

    _tmp_params = teacher_decoder_mlp.init(jax.random.PRNGKey(0), dummy_teacher_obs)
    _dummy_feats = teacher_decoder_mlp.apply(_tmp_params, dummy_teacher_obs)

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

    action_head_net = FeedForwardNetwork(
        init=lambda key: action_head_mlp.init(key, _dummy_feats),
        apply=lambda pparams, params, feats: action_head_mlp.apply(params, feats),
    )

    def _teacher_net_init(key):
        k1, k2 = jax.random.split(key)
        dec_params = teacher_decoder_mlp.init(k1, dummy_teacher_obs)
        feats = teacher_decoder_mlp.apply(dec_params, dummy_teacher_obs)
        head_params = action_head_mlp.init(k2, feats)
        return dec_params, head_params

    def _teacher_net_apply(pparams, params, obs):
        feats = teacher_decoder_mlp.apply(params[0], _preprocess_teacher(obs, pparams))
        return action_head_mlp.apply(params[1], feats)

    teacher_network = FeedForwardNetwork(init=_teacher_net_init, apply=_teacher_net_apply)

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
# DAgger-specific inference helper
# ---------------------------------------------------------------------------

def make_student_inference_fn(il_networks: ILNetworks):
    """Student (vision) policy factory for DAgger — trainable action head.

    Unlike the IL version, the action head is NOT closed over as a constant.
    It is part of the params tuple so the training loop can update it.

    Expected params at call time (from _pack_student_params):
        params[0] = proprio_norm              (RunningStatisticsState)
        params[1] = student_enc_params        (trainable)
        params[2] = student_action_head_params (trainable)

    Checkpoint layout (saved by train.py):
        (proprio_norm, (student_enc_params, student_action_head_params))
    """

    def make_policy(params: types.Params, deterministic: bool = False) -> types.Policy:
        proprio_norm, student_enc, student_action_head = params

        def policy(observations: types.Observation, key_sample: PRNGKey):
            logits = il_networks.student_network.apply(
                proprio_norm, (student_enc, student_action_head), observations
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
