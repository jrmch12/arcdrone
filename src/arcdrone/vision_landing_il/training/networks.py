"""PPO networks for vision-based landing RL/IL."""

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
    selected = pparams
    for key in _split_path(path):
        selected = normalizer_select(selected, key)
    return selected


def _shape_last_dim(shape_spec: Any) -> int:
    if isinstance(shape_spec, int):
        return shape_spec
    return int(jax.tree_util.tree_flatten(shape_spec)[0][-1])


# # ---------------------------------------------------------------------------
# # Factory
# # ---------------------------------------------------------------------------

# def make_ppo_networks_vision(
#     observation_size: types.ObservationSize,
#     action_size: int,
#     preprocess_observations_fn: types.PreprocessObservationFn = (
#         types.identity_observation_preprocessor
#     ),
#     # Fusion decoder layers after concatenating CNN and proprio projector features.
#     policy_dec_hidden_layers: Sequence[int] = (256, 256),
#     # Proprio projection branch (MLP) before fusion.
#     policy_propio_proj_hidden_layers: Sequence[int] = (64,),
#     # Action head on top of the policy backbone features
#     action_hidden_layer_sizes: Sequence[int] = (64,),
#     value_hidden_layer_sizes: Sequence[int] = (256, 256, 256),
#     cnn_num_filters: Sequence[int] = (32, 64, 64),
#     cnn_kernel_sizes: Sequence[Tuple[int, int]] = ((8, 8), (4, 4), (3, 3)),
#     cnn_strides: Sequence[Tuple[int, int]] = ((4, 4), (2, 2), (1, 1)),
#     activation: ActivationFn = nn.relu,
#     # Path to policy group in observation dict (e.g. "policy_obs").
#     policy_obs_key: str = "policy_obs",
#     # Key inside policy_obs group for pixel tensor.
#     policy_pixels_key: str = "pixels",
#     # Key inside policy_obs group for proprio tensor.
#     policy_propio_key: str = "propio",
#     # Path to value vector in observation dict (e.g. "value_obs").
#     value_obs_key: str = "value_obs",
# ) -> PPONetworks:
#     """Build two-stream vision-policy + privileged-state-value PPO networks."""

#     kernel_init = jax.nn.initializers.lecun_uniform()

#     parametric_action_distribution = distribution.NormalTanhDistribution(
#         event_size=action_size
#     )

#     # ------------------------------------------------------------------
#     # Modules
#     # ------------------------------------------------------------------

#     policy_encoder = PolicyVisionProprioEncoder(
#         cnn_num_filters=list(cnn_num_filters),
#         cnn_kernel_sizes=list(cnn_kernel_sizes),
#         cnn_strides=list(cnn_strides),
#         proprio_proj_hidden_layers=list(policy_propio_proj_hidden_layers),
#         fusion_hidden_layers=list(policy_dec_hidden_layers),
#         activation=activation,
#         kernel_init=kernel_init,
#     )

#     policy_action_head = MLP(
#         layer_sizes=list(action_hidden_layer_sizes)
#             + [parametric_action_distribution.param_size],
#         activation=activation,
#         kernel_init=kernel_init,
#     )


#     value_mlp = MLP(
#         layer_sizes=list(value_hidden_layer_sizes) + [1],
#         activation=activation,
#         kernel_init=kernel_init,
#     )


#     # ------------------------------------------------------------------
#     # Dummy inputs for init
#     # ------------------------------------------------------------------

#     policy_obs_size = _get_by_path(observation_size, policy_obs_key)
#     pixel_shape = tuple(policy_obs_size[policy_pixels_key])
#     propio_size = _shape_last_dim(policy_obs_size[policy_propio_key])
#     dummy_pixels = jnp.zeros((1,) + pixel_shape)
#     dummy_propio = jnp.zeros((1, propio_size))

#     # Value dummy input
#     value_obs_size = _get_obs_size(_get_by_path(observation_size, value_obs_key), "")
#     dummy_value_obs = jnp.zeros((1, value_obs_size))

#     # ------------------------------------------------------------------
#     # Helpers
#     # ------------------------------------------------------------------

#     def _preprocess_value(obs, pparams):
#         if isinstance(obs, Mapping):
#             value_obs = _get_by_path(obs, value_obs_key)
#             return preprocess_observations_fn(
#                 value_obs,
#                 _select_normalizer_by_path(pparams, value_obs_key),
#             )
#         return preprocess_observations_fn(obs, pparams)

#     def _extract_policy_obs(obs):
#         policy_group = _get_by_path(obs, policy_obs_key)
#         return policy_group[policy_pixels_key], policy_group[policy_propio_key]

#     def _preprocess_policy_propio(obs, pparams):
#         _, proprio = _extract_policy_obs(obs)
#         return preprocess_observations_fn(
#             proprio,
#             _select_normalizer_by_path(
#                 pparams, f"{policy_obs_key}/{policy_propio_key}"
#             ),
#         )

#     # ------------------------------------------------------------------
#     # Value network init / apply
#     # ------------------------------------------------------------------

#     def _value_init(key):
#         return value_mlp.init(key, dummy_value_obs)

#     def _value_apply(pparams, params, obs):
#         return jnp.squeeze(
#             value_mlp.apply(params, _preprocess_value(obs, pparams)),
#             axis=-1,
#         )
    
#     value_network = FeedForwardNetwork(init=_value_init, apply=_value_apply)

#     # ------------------------------------------------------------------
#     # Policy network
#     # Params = (vision_decoder_params, action_head_params)
#     # apply : (normalizer_params, policy_params, obs_dict) → logits
#     # ------------------------------------------------------------------

#     def _policy_init(key):
#         k1, k2 = jax.random.split(key)
#         dec_params = policy_encoder.init(k1, dummy_pixels, dummy_propio)
#         feats = policy_encoder.apply(dec_params, dummy_pixels, dummy_propio)
#         head_params = policy_action_head.init(k2, feats)
#         return dec_params, head_params

#     def _policy_apply(pparams, params, obs):
#         pixels, _ = _extract_policy_obs(obs)
#         proprio = _preprocess_policy_propio(obs, pparams)
#         feats = policy_encoder.apply(params[0], pixels, proprio)
#         return policy_action_head.apply(params[1], feats)

#     policy_network = FeedForwardNetwork(init=_policy_init, apply=_policy_apply)

#     # ------------------------------------------------------------------

#     return PPONetworks(
#         policy_network=policy_network,
#         value_network=value_network,
#         parametric_action_distribution=parametric_action_distribution,
#     )


# # ---------------------------------------------------------------------------
# # Inference helper (standard PPO, no student/proxy)
# # ---------------------------------------------------------------------------

# def make_inference_fn(ppo_networks: PPONetworks, compute_value: bool = False):
#     """Returns a policy-factory compatible with brax PPO.

#     Expected params layout passed by brax.ppo.train:
#         params[0] = normalizer_params
#         params[1] = policy_params   → (vision_decoder_params, action_head_params)
#         params[2] = value_params
#     """

#     def make_policy(
#         params: types.Params, deterministic: bool = False
#     ) -> types.Policy:

#         def policy(
#             observations: types.Observation, key_sample: PRNGKey
#         ) -> Tuple[types.Action, types.Extra]:
#             logits = ppo_networks.policy_network.apply(
#                 params[0], params[1], observations
#             )
#             if deterministic:
#                 return ppo_networks.parametric_action_distribution.mode(logits), {}

#             raw_actions = (
#                 ppo_networks.parametric_action_distribution
#                 .sample_no_postprocessing(logits, key_sample)
#             )
#             log_prob = ppo_networks.parametric_action_distribution.log_prob(
#                 logits, raw_actions
#             )
#             postprocessed_actions = (
#                 ppo_networks.parametric_action_distribution.postprocess(raw_actions)
#             )
#             extras = {
#                 "log_prob": log_prob,
#                 "raw_action": raw_actions,
#                 "distribution_params": logits,
#             }
#             if compute_value:
#                 extras["value"] = ppo_networks.value_network.apply(
#                     params[0], params[2], observations
#                 )
#             return postprocessed_actions, extras

#         return policy

#     return make_policy


# ===========================================================================
# IL (Imitation Learning) data structures and factory
# ===========================================================================

@flax.struct.dataclass
class ILNetworks:
    """All networks needed for vision IL (student–teacher alignment).

    Teacher : flat privileged state  → MLP decoder → features → action head → logits
    Student : pixels + propio        → CNN+MLP encoder → features → action head → logits
    (action head is shared / frozen from teacher checkpoint)
    """
    # Full rollout networks
    teacher_network: FeedForwardNetwork        # flat state  → logits  (frozen, for rollouts)
    student_network: FeedForwardNetwork        # vision obs  → logits  (student inference)
    value_network: FeedForwardNetwork          # flat state  → scalar  (frozen)
    # Adapters used in alignment
    teacher_decoder: FeedForwardNetwork        # flat state  → features
    student_encoder: FeedForwardNetwork        # vision obs  → features
    action_head: FeedForwardNetwork            # features    → logits
    parametric_action_distribution: distribution.ParametricDistribution


# ---------------------------------------------------------------------------
# IL factory
# ---------------------------------------------------------------------------

def make_il_networks(
    observation_size: types.ObservationSize,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = (
        types.identity_observation_preprocessor
    ),
    # Teacher MLP decoder (frozen from checkpoint)
    teacher_dec_hidden_layers: Sequence[int] = (512, 512, 256, 128),
    # Student (vision) encoder — same architecture as vision RL student
    policy_dec_hidden_layers: Sequence[int] = (256, 256),
    policy_propio_proj_hidden_layers: Sequence[int] = (64,),
    cnn_num_filters: Sequence[int] = (32, 64, 64),
    cnn_kernel_sizes: Sequence[Tuple[int, int]] = ((8, 8), (4, 4), (3, 3)),
    cnn_strides: Sequence[Tuple[int, int]] = ((4, 4), (2, 2), (1, 1)),
    # Shared action head
    action_hidden_layer_sizes: Sequence[int] = (64,),
    # Value MLP (frozen from teacher ckpt)
    value_hidden_layer_sizes: Sequence[int] = (256, 256, 256),
    activation: ActivationFn = nn.relu,
    # Student vision obs keys (flat dict: e.g. 'pixels/view_0', 'propio')
    policy_pixels_key: str = "pixels/view_0",
    policy_pixels_key_1: str = "pixels/view_1",
    policy_propio_key: str = "propio",
    # Teacher / value obs key (flat privileged state)
    teacher_obs_key: str = "teacher_obs",
    value_obs_key: str = "value_obs",
) -> ILNetworks:
    """Build teacher-state + student-vision IL networks for alignment training."""

    kernel_init = jax.nn.initializers.lecun_uniform()
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )

    # ------------------------------------------------------------------
    # Flax modules
    # ------------------------------------------------------------------

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
        proprio_proj_hidden_layers=list(policy_propio_proj_hidden_layers),
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

    # ------------------------------------------------------------------
    # Dummy inputs for init
    # ------------------------------------------------------------------

    # Teacher (flat privileged state)
    teacher_obs_raw = (
        _get_by_path(observation_size, teacher_obs_key)
        if isinstance(observation_size, Mapping)
        else observation_size
    )
    teacher_obs_size = _shape_last_dim(teacher_obs_raw)
    dummy_teacher_obs = jnp.zeros((1, teacher_obs_size))

    # Value obs — may differ in size from teacher obs
    value_obs_raw = (
        _get_by_path(observation_size, value_obs_key)
        if isinstance(observation_size, Mapping)
        else observation_size
    )
    value_obs_size = _shape_last_dim(value_obs_raw)
    dummy_value_obs = jnp.zeros((1, value_obs_size))

    # Student vision obs — flat keys: obs["pixels/view_0"], obs["pixels/view_1"], obs["propio"]
    # Both camera frame stacks are concatenated along the channel axis before the CNN.
    pixel_shape_0 = tuple(observation_size[policy_pixels_key])
    pixel_shape_1 = tuple(observation_size[policy_pixels_key_1])
    # Concatenate: (H, W, history) + (H, W, history) → (H, W, 2*history)
    pixel_shape = pixel_shape_0[:-1] + (pixel_shape_0[-1] + pixel_shape_1[-1],)
    propio_size = _shape_last_dim(observation_size[policy_propio_key])
    dummy_pixels = jnp.zeros((1,) + pixel_shape)
    dummy_propio = jnp.zeros((1, propio_size))

    # Feature dummy (needed to init action head)
    _tmp_params = teacher_decoder_mlp.init(jax.random.PRNGKey(0), dummy_teacher_obs)
    _dummy_feats = teacher_decoder_mlp.apply(_tmp_params, dummy_teacher_obs)

    # ------------------------------------------------------------------
    # Shared preprocessing helpers
    # ------------------------------------------------------------------

    def _preprocess_teacher(obs, pparams):
        """Normalise teacher (policy) obs.

        ``pparams`` is already the per-key sub-state (extracted from the full
        teacher checkpoint normalizer via normalizer_select before training),
        so we do NOT call normalizer_select here — just apply it directly.
        """
        teacher_obs = obs[teacher_obs_key] if isinstance(obs, Mapping) else obs
        return preprocess_observations_fn(teacher_obs, pparams)

    def _preprocess_value(obs, pparams):
        """Normalise value obs using its own normalizer sub-state (may differ from teacher)."""
        if isinstance(obs, Mapping):
            value_obs = obs[value_obs_key]
            return preprocess_observations_fn(
                value_obs, normalizer_select(pparams, value_obs_key)
            )
        return preprocess_observations_fn(obs, pparams)

    def _extract_student_obs(obs):
        # Concatenate both camera frame stacks along channel axis: (H, W, 2*history)
        pixels = jnp.concatenate([obs[policy_pixels_key], obs[policy_pixels_key_1]], axis=-1)
        return pixels, obs[policy_propio_key]

    def _preprocess_student_propio(obs, pparams):
        """Normalise proprio obs.

        ``pparams`` is the propio RunningStatisticsState directly — not a dict.
        """
        proprio = obs[policy_propio_key] if isinstance(obs, Mapping) else obs
        return preprocess_observations_fn(proprio, pparams)

    # ------------------------------------------------------------------
    # Decoder-only FeedForwardNetworks (used in align.py)
    # Each apply returns *features*, not logits.
    # ------------------------------------------------------------------

    teacher_decoder_net = FeedForwardNetwork(
        init=lambda key: teacher_decoder_mlp.init(key, dummy_teacher_obs),
        apply=lambda pparams, params, obs: teacher_decoder_mlp.apply(
            params, _preprocess_teacher(obs, pparams)
        ),
    )

    student_encoder_net = FeedForwardNetwork(
        init=lambda key: student_encoder_module.init(key, dummy_pixels, dummy_propio),
        apply=lambda pparams, params, obs: student_encoder_module.apply(
            params,
            _extract_student_obs(obs)[0],                    # pixels  — no normalisation needed
            _preprocess_student_propio(obs, pparams),        # normalised propio
        ),
    )

    action_head_net = FeedForwardNetwork(
        init=lambda key: action_head_mlp.init(key, _dummy_feats),
        apply=lambda pparams, params, feats: action_head_mlp.apply(params, feats),
    )

    # ------------------------------------------------------------------
    # Full rollout networks (decoder/encoder + action head)
    # Params layout: (decoder_or_encoder_params, action_head_params)
    # ------------------------------------------------------------------

    # Teacher network
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

    # Student network
    # params[0] = student_encoder_module params
    # params[1] = action_head_mlp params  (shared from teacher, frozen at inference)
    def _student_net_init(key):
        k1, k2 = jax.random.split(key)
        enc_params = student_encoder_module.init(k1, dummy_pixels, dummy_propio)
        feats = student_encoder_module.apply(enc_params, dummy_pixels, dummy_propio)
        head_params = action_head_mlp.init(k2, feats)
        return enc_params, head_params

    def _student_net_apply(pparams, params, obs):
        pixels, _ = _extract_student_obs(obs)
        proprio = _preprocess_student_propio(obs, pparams)
        feats = student_encoder_module.apply(params[0], pixels, proprio)
        return action_head_mlp.apply(params[1], feats)

    student_network = FeedForwardNetwork(init=_student_net_init, apply=_student_net_apply)

    # Value network
    value_network = FeedForwardNetwork(
        init=lambda key: value_mlp.init(key, dummy_value_obs),
        apply=lambda pparams, params, obs: jnp.squeeze(
            value_mlp.apply(params, _preprocess_value(obs, pparams)),
            axis=-1,
        ),
    )

    # ------------------------------------------------------------------

    return ILNetworks(
        teacher_network=teacher_network,
        student_network=student_network,
        value_network=value_network,
        teacher_decoder=teacher_decoder_net,
        student_encoder=student_encoder_net,
        action_head=action_head_net,
        parametric_action_distribution=parametric_action_distribution,
    )


# ---------------------------------------------------------------------------
# IL inference helpers
# ---------------------------------------------------------------------------

def make_frozen_teacher_policy(
    il_networks: ILNetworks,
    teacher_norm_params: Any,
    teacher_policy_params: Any,
    deterministic: bool = False,
) -> types.Policy:
    """Returns a ``policy(obs, key)`` with all teacher params baked in as constants.

    Nothing is passed at call time — teacher norm and policy params are closed
    over here so they never appear in the JAX training state.
    """

    def policy(
        observations: types.Observation, key_sample: PRNGKey
    ) -> Tuple[types.Action, types.Extra]:
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


def make_student_inference_fn(il_networks: ILNetworks, action_head_params: Any):
    """Student (vision) policy factory.

    ``action_head_params`` is a frozen constant from the teacher checkpoint,
    closed over here so it is never part of the JAX training state.

    Expected params at call time (from _pack_student_params):
        params[0] = propio_norm        (RunningStatisticsState)
        params[1] = student_enc_params
    """

    def make_policy(params: types.Params, deterministic: bool = False) -> types.Policy:
        propio_norm, student_enc = params

        def policy(observations: types.Observation, key_sample: PRNGKey):
            # Pass propio_norm directly — _preprocess_student_propio uses it as-is.
            logits = il_networks.student_network.apply(
                propio_norm, (student_enc, action_head_params), observations
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

    return make_policy
