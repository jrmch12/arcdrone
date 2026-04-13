"""SITT training: PPO on teacher env + independent alignment on student env.

PPO trains the teacher policy (teacher_dec + action_head), value network,
and proxy decoder via reward shaping.  Alignment runs independently on the
student env: the student encoder and proxy decoder are trained to match the
teacher decoder features.

The two phases use **separate environments**:
  - teacher env: ARCDroneRL_Landing (flat privileged obs)
  - student env: ARCDroneRL_VisionLanding_StudentTeacher (pixels + proprio)

Parameters are shared: the teacher_dec_params and action_head_params trained
by PPO are the same ones used as frozen targets during alignment.
"""

import functools
import time
from typing import Any, Callable, Mapping, Optional, Tuple, Union

from absl import logging
from brax import base
from brax import envs
from brax.training import acting
from brax.training import gradients
from brax.training import pmap
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.acme import specs
from brax.training.agents.ppo import optimizer as ppo_optimizer
from brax.training.networks import normalizer_select
from brax.training.types import PRNGKey

from . import losses as sitt_losses
from . import networks as sitt_networks
from .align import align

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax

logging.set_verbosity(logging.INFO)
logging.set_stderrthreshold("info")


InferenceParams = Any
Metrics = types.Metrics

_PMAP_AXIS_NAME = "i"


# ── Training state ─────────────────────────────────────────────────

@flax.struct.dataclass
class TrainingState:
    """Mutable state for SITT training."""
    # PPO
    optimizer_state: optax.OptState
    params: sitt_losses.PPONetworkParams   # policy=(dec,head), value, proxy_dec
    normalizer_params: running_statistics.RunningStatisticsState
    env_steps: jnp.int32                   # PPO env steps
    # Align
    align_opt_state: optax.OptState        # for (student_enc, proxy_dec)
    student_enc_params: Any
    student_proprio_norm: running_statistics.RunningStatisticsState
    align_env_steps: jnp.int32             # align env steps (separate counter)


# ── Helpers ────────────────────────────────────────────────────────

def _unpmap(v):
    def _first_or_self(x):
        if x is None:
            return None
        shape = getattr(x, "shape", None)
        if shape and len(shape) > 0:
            return x[0]
        return x

    return jax.tree_util.tree_map(_first_or_self, v)


def _strip_weak_type(tree):
    def f(leaf):
        leaf = jnp.asarray(leaf)
        return jnp.astype(leaf, leaf.dtype)
    return jax.tree_util.tree_map(f, tree)


def _remove_pixels(obs):
    if not isinstance(obs, Mapping):
        return obs
    return {k: v for k, v in obs.items() if not k.startswith("pixels/")}


def _pack_ppo_inference_params(ts: TrainingState) -> InferenceParams:
    """For PPO evaluator: (normalizer, policy, value)."""
    return (ts.normalizer_params, ts.params.policy, ts.params.value)


def _pack_student_params(ts: TrainingState) -> Tuple:
    """For student evaluator: (proprio_norm, student_enc_params)."""
    return (ts.student_proprio_norm, ts.student_enc_params)


# ── Main ───────────────────────────────────────────────────────────

def train(
    teacher_env: envs.Env,
    student_env: envs.Env,
    teacher_params: Any,
    *,
    # PPO schedule
    num_timesteps: int = 20_000_000,
    num_evals: int = 10,
    unroll_length: int = 10,
    batch_size: int = 32,
    num_minibatches: int = 16,
    num_updates_per_batch: int = 2,
    num_resets_per_eval: int = 0,
    # PPO hyperparams
    learning_rate: float = 3e-4,
    entropy_cost: float = 1e-3,
    discounting: float = 0.97,
    reward_scaling: float = 0.1,
    clipping_epsilon: float = 0.3,
    clipping_epsilon_value: float | None = None,
    gae_lambda: float = 0.95,
    max_grad_norm: Optional[float] = None,
    normalize_advantage: bool = True,
    vf_loss_coefficient: float = 0.5,
    bootstrap_on_timeout: bool = False,
    desired_kl: float = 0.01,
    learning_rate_schedule: Optional[Union[str, ppo_optimizer.LRSchedule]] = None,
    # SITT alignment schedule
    align_num_epochs: int = 200,
    align_num_splits: int = 10,
    align_num_envs: int = 256,
    align_batch_size: int = 256,
    align_num_minibatches: int = 4,
    align_updates_per_trigger: int = 4,
    align_learning_rate: float = 3e-4,
    proxy_kl_coef: float = 1.9,
    sitt_align_coef: float = 0.01,
    align_unroll_length: int = 10,
    align_embed_coef: float = 1.0,
    align_action_coef: float = 1.0,
    policy_proprio_key: str = "proprio_obs",
    policy_obs_key: str = "policy_obs",
    # Networks
    network_factory = sitt_networks.make_sitt_networks,
    # Environment
    num_envs: int = 256,
    episode_length: Optional[int] = None,
    action_repeat: int = 1,
    num_eval_envs: int = 128,
    deterministic_eval: bool = True,
    # Normalizer
    normalize_observations: bool = True,
    normalize_observations_std_eps: float = 0.0,
    normalize_observations_mode: str = "welford",
    # Misc
    seed: int = 0,
    max_devices_per_host: Optional[int] = None,
    log_training_metrics: bool = False,
    ppo_off: bool = False,
    # Callbacks
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    policy_params_fn: Callable[..., None] = lambda *args: None,
    # Restore
    restore_params: Optional[Any] = None,
    restore_value_fn: bool = True,
    run_evals: bool = True,
    wrap_env: bool = False,
):
    """SITT training loop.

    Args:
        teacher_env: wrapped teacher env (flat obs: policy_obs, value_obs).
        student_env: wrapped student env (pixels, proprio, teacher_obs, value_obs).
        teacher_params: 3-tuple ``(normalizer, policy, value)`` from the
            privileged_landing_rl PPO checkpoint.  Used to initialise the
            teacher decoder, action head, and value network.
        align_num_epochs: total number of alignment epochs across the whole
            training run, spread evenly across PPO eval periods.
        align_batch_size: env trajectories per alignment minibatch.
        align_num_minibatches: minibatches per alignment epoch.
        align_updates_per_trigger: gradient passes per alignment epoch.
        align_learning_rate: Adam LR for alignment optimizer.
        proxy_kl_coef: coefficient for proxy KL reward shaping during PPO.
        sitt_align_coef: coefficient for the auxiliary rl_align_loss in PPO loss.

    Returns:
        ``(make_policy, params, metrics)`` where params contains both
        PPO and student/proxy parameters.
    """

    xt = time.time()

    # ======================== device setup ============================

    process_count = jax.process_count()
    process_id = jax.process_index()
    local_device_count = jax.local_device_count()
    local_devices_to_use = local_device_count
    if max_devices_per_host:
        local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
    device_count = local_devices_to_use * process_count

    logging.info(
        "SITT — devices: %d, processes: %d (id %d), local: %d, used: %d",
        jax.device_count(), process_count, process_id,
        local_device_count, local_devices_to_use,
    )

    # ======================== PPO schedule ============================

    env_step_per_training_step = (
        batch_size * unroll_length * num_minibatches * action_repeat
    )
    num_evals_after_init = max(num_evals - 1, 1)
    num_training_steps_per_epoch = np.ceil(
        num_timesteps
        / (
            num_evals_after_init
            * env_step_per_training_step
            * max(num_resets_per_eval, 1)
        )
    ).astype(int)
    ppo_unrolls_per_step = batch_size * num_minibatches // num_envs
    num_envs_per_device = num_envs // device_count

    # ======================== align schedule ===========================

    if align_num_splits <= 0:
        raise ValueError("align_num_splits must be > 0")
    num_align_splits = min(align_num_splits, num_evals_after_init)
    align_split_indices = np.linspace(
        0, num_evals_after_init - 1, num=num_align_splits, dtype=int
    )
    align_split_indices = np.unique(align_split_indices)
    align_base = align_num_epochs // len(align_split_indices) if align_num_epochs > 0 else 0
    align_remainder = align_num_epochs % len(align_split_indices) if align_num_epochs > 0 else 0
    align_epochs_by_iter = {
        int(idx): align_base + (1 if i < align_remainder else 0)
        for i, idx in enumerate(align_split_indices)
    }

    # Env steps per alignment epoch (student env).
    assert align_num_envs % device_count == 0
    align_num_envs_per_device = align_num_envs // device_count
    align_unrolls_per_epoch = (
        align_batch_size * align_num_minibatches // align_num_envs_per_device
    )
    align_env_steps_per_epoch = (
        align_batch_size * align_unroll_length * align_num_minibatches * action_repeat
    )

    # ======================== keys ====================================

    key = jax.random.PRNGKey(seed)
    global_key, local_key = jax.random.split(key)
    del key
    local_key = jax.random.fold_in(local_key, process_id)
    local_key, key_teacher_env, key_student_env, eval_key = jax.random.split(
        local_key, 4
    )
    key_policy, key_value, key_sitt = jax.random.split(global_key, 3)
    del global_key

    assert num_envs % device_count == 0
    assert batch_size * num_minibatches % num_envs == 0
    assert align_num_envs % device_count == 0

    # ======================== reset teacher env =======================

    key_envs_t = jax.random.split(key_teacher_env, num_envs // process_count)
    key_envs_t = jnp.reshape(
        key_envs_t, (local_devices_to_use, -1) + key_envs_t.shape[1:]
    )
    reset_teacher = jax.pmap(teacher_env.reset, axis_name=_PMAP_AXIS_NAME)
    teacher_env_state = reset_teacher(key_envs_t)

    # ======================== reset student env =======================

    key_envs_s = jax.random.split(key_student_env, align_num_envs // process_count)
    key_envs_s = jnp.reshape(
        key_envs_s, (local_devices_to_use, -1) + key_envs_s.shape[1:]
    )
    reset_student = jax.pmap(student_env.reset, axis_name=_PMAP_AXIS_NAME)
    student_env_state = reset_student(key_envs_s)

    # ======================== obs shapes ==============================

    teacher_obs_shape = jax.tree_util.tree_map(
        lambda x: x.shape[2:], teacher_env_state.obs
    )
    student_obs_shape = jax.tree_util.tree_map(
        lambda x: x.shape[2:], student_env_state.obs
    )

    normalize = lambda x, y: x
    if normalize_observations:
        normalize = running_statistics.normalize

    # ======================== networks ================================

    sitt_network = network_factory(
        teacher_obs_shape,
        teacher_env.action_size,
        preprocess_observations_fn=normalize,
        student_observation_size=student_obs_shape,
    )

    make_ppo_policy = sitt_networks.make_inference_fn(
        sitt_network,
        compute_value=bootstrap_on_timeout or clipping_epsilon_value is not None,
    )

    # ======================== optimizers ==============================

    base_optimizer = optax.adam(learning_rate=learning_rate)
    lr_schedule = learning_rate_schedule or ppo_optimizer.LRSchedule.NONE
    lr_schedule = ppo_optimizer.LRSchedule(lr_schedule)
    lr_is_adaptive_kl = lr_schedule == ppo_optimizer.LRSchedule.ADAPTIVE_KL
    if lr_is_adaptive_kl:
        base_optimizer = optax.inject_hyperparams(optax.adam)(
            learning_rate=learning_rate
        )
    if max_grad_norm is not None:
        ppo_optimizer_obj = optax.chain(
            optax.clip_by_global_norm(max_grad_norm), base_optimizer
        )
    else:
        ppo_optimizer_obj = base_optimizer

    align_optimizer = optax.adam(align_learning_rate)

    # ======================== PPO loss ================================

    loss_fn = functools.partial(
        sitt_losses.compute_ppo_loss,
        ppo_network=sitt_network,
        entropy_cost=entropy_cost,
        discounting=discounting,
        reward_scaling=reward_scaling,
        gae_lambda=gae_lambda,
        clipping_epsilon=clipping_epsilon,
        normalize_advantage=normalize_advantage,
        vf_coefficient=vf_loss_coefficient,
        clipping_epsilon_value=clipping_epsilon_value,
        use_sitt=True,
        sitt_align_coef=sitt_align_coef,
    )
    loss_and_pgrad_fn = gradients.loss_and_pgrad(
        loss_fn, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True
    )

    # ======================== align fn ================================

    align_fn = functools.partial(
        align,
        sitt_network=sitt_network,
        optimizer=align_optimizer,
        align_updates_per_trigger=align_updates_per_trigger,
        num_minibatches=align_num_minibatches,
        embed_coef=align_embed_coef,
        action_coef=align_action_coef,
    )

    # ======================== init params =============================

    key_policy_init, key_value_init, key_student, key_proxy = jax.random.split(
        key_sitt, 4
    )

    # Student proprio normalizer (separate from teacher PPO normalizer).
    student_proprio_spec = specs.Array(
        student_env_state.obs[policy_proprio_key].shape[2:], jnp.dtype("float32")
    )
    student_proprio_norm = running_statistics.init_state(
        student_proprio_spec,
        std_eps=normalize_observations_std_eps,
        mode=normalize_observations_mode,
    )

    if ppo_off and teacher_params is None:
        logging.warning(
            "ppo_off=True but no teacher checkpoint provided; PPO params will stay random."
        )

    if teacher_params is not None:
        # Warm-start: decompose teacher checkpoint (norm, policy=(dec,head), value)
        teacher_norm_params = teacher_params[0]
        teacher_policy_params = teacher_params[1]
        teacher_value_params = teacher_params[2]
        policy_params = teacher_policy_params
        value_params = teacher_value_params
        normalizer_params = teacher_norm_params
        action_head_params = teacher_policy_params[1]
        logging.info("Initialised PPO params from teacher checkpoint.")
    else:
        # Cold-start: random init for teacher policy + value + normalizer
        policy_params = sitt_network.policy_network.init(key_policy_init)
        value_params = sitt_network.value_network.init(key_value_init)
        # Build a fresh normalizer from teacher env obs spec
        obs_spec = jax.tree_util.tree_map(
            lambda x: specs.Array(x.shape[-1:], jnp.dtype("float32")),
            teacher_env_state.obs,
        )
        normalizer_params = running_statistics.init_state(
            _remove_pixels(obs_spec),
            std_eps=normalize_observations_std_eps,
            mode=normalize_observations_mode,
        )
        # action_head params live inside policy_params[1] for student inference
        action_head_params = policy_params[1]
        logging.info("Initialised PPO params from scratch (no teacher checkpoint).")

    # Student inference factory (action_head baked in as closure)
    make_student_policy = sitt_networks.make_student_inference_fn(
        sitt_network, action_head_params=action_head_params,
    )

    # Fresh student encoder + proxy decoder
    student_enc_params = sitt_network.student_encoder.init(key_student)
    proxy_dec_params = sitt_network.ppo_proxy_decoder.init(key_proxy)

    init_ppo_params = sitt_losses.PPONetworkParams(
        policy=policy_params,
        value=value_params,
        proxy_dec_params=proxy_dec_params,
    )

    # Align optimizer: (student_enc, proxy_dec)
    align_opt_state = align_optimizer.init((student_enc_params, proxy_dec_params))

    training_state = TrainingState(
        optimizer_state=ppo_optimizer_obj.init(init_ppo_params),
        params=init_ppo_params,
        normalizer_params=normalizer_params,
        env_steps=jnp.int32(0),
        align_opt_state=align_opt_state,
        student_enc_params=student_enc_params,
        student_proprio_norm=student_proprio_norm,
        align_env_steps=jnp.int32(0),
    )

    # Optionally restore from a previous SITT checkpoint
    if restore_params is not None:
        logging.info("Restoring SITT TrainingState from restore_params.")
        # Expected layout: (norm, policy, value, proxy_dec, student_enc, student_proprio_norm)
        value_p = restore_params[2] if restore_value_fn else init_ppo_params.value
        proxy_p = restore_params[3] if len(restore_params) > 3 else proxy_dec_params
        student_p = restore_params[4] if len(restore_params) > 4 else student_enc_params
        student_norm_p = (
            restore_params[5] if len(restore_params) > 5 else student_proprio_norm
        )
        training_state = training_state.replace(
            normalizer_params=restore_params[0],
            params=training_state.params.replace(
                policy=restore_params[1],
                value=value_p,
                proxy_dec_params=proxy_p,
            ),
            student_enc_params=student_p,
            student_proprio_norm=student_norm_p,
        )

    # ======================== PPO training fns =========================

    def minibatch_step(carry, data, normalizer_params):
        optimizer_state, params, key = carry
        key, key_loss = jax.random.split(key)
        (_, metrics), grads = loss_and_pgrad_fn(
            params, normalizer_params, data, key_loss
        )
        if lr_is_adaptive_kl:
            kl_mean = jax.lax.pmean(
                metrics["kl_mean"], axis_name=_PMAP_AXIS_NAME
            )
            optimizer_state, lr = ppo_optimizer.adaptive_kl_learning_rate(
                optimizer_state, kl_mean, desired_kl
            )
        else:
            lr = jnp.array(learning_rate)
        metrics["learning_rate"] = lr
        params_update, optimizer_state = ppo_optimizer_obj.update(
            grads, optimizer_state
        )
        params = optax.apply_updates(params, params_update)
        return (optimizer_state, params, key), metrics

    def sgd_step(carry, unused_t, data, normalizer_params):
        optimizer_state, params, key = carry
        key, key_perm, key_grad = jax.random.split(key, 3)

        def convert_data(x):
            x = jax.random.permutation(key_perm, x)
            return jnp.reshape(x, (num_minibatches, -1) + x.shape[1:])

        shuffled = jax.tree_util.tree_map(convert_data, data)
        (optimizer_state, params, _), metrics = jax.lax.scan(
            functools.partial(minibatch_step, normalizer_params=normalizer_params),
            (optimizer_state, params, key_grad),
            shuffled,
            length=num_minibatches,
        )
        return (optimizer_state, params, key), metrics

    def _generate_teacher_rollout(state, key, policy, num_unrolls):
        def _scan(carry, _):
            s, k = carry
            k, nk = jax.random.split(k)
            extra = ["truncation", "episode_metrics", "episode_done"]
            if bootstrap_on_timeout:
                extra.append("time_out")
            ns, data = acting.generate_unroll(
                teacher_env, s, policy, k, unroll_length,
                extra_fields=tuple(extra),
            )
            return (ns, nk), data

        (next_state, _), data = jax.lax.scan(
            _scan, (state, key), (), length=num_unrolls
        )
        data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
        data = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data
        )
        return next_state, data

    def training_step(carry, unused_t):
        training_state, state, key = carry
        key_sgd, key_unroll, new_key = jax.random.split(key, 3)

        policy = make_ppo_policy((
            training_state.normalizer_params,
            training_state.params.policy,
            training_state.params.value,
        ))
        state, data = _generate_teacher_rollout(
            state, key_unroll, policy, ppo_unrolls_per_step,
        )

        # ── Proxy KL reward shaping ──────────────────────────────────
        teacher_logits = sitt_network.policy_network.apply(
            training_state.normalizer_params,
            training_state.params.policy,
            data.observation,
        )
        proxy_feats = sitt_network.ppo_proxy_decoder.apply(
            training_state.normalizer_params,
            training_state.params.proxy_dec_params,
            data.observation,
        )
        proxy_logits = sitt_network.action_head.apply(
            training_state.normalizer_params,
            training_state.params.policy[1],
            proxy_feats,
        )
        teacher_probs = jax.nn.softmax(teacher_logits)
        proxy_probs = jax.nn.softmax(proxy_logits)
        kl = jnp.sum(
            teacher_probs * (
                jnp.log(teacher_probs + 1e-8)
                - jnp.log(proxy_probs + 1e-8)
            ),
            axis=-1,
        )
        # ── reward_align as episode-return metric ────────────────────
        kl_per_step = proxy_kl_coef * kl  # (num_envs_per_device * ppo_unrolls_per_step, unroll_length)
        total_steps = ppo_unrolls_per_step * unroll_length
        kl_reshaped = jnp.reshape(kl_per_step, (num_envs_per_device, total_steps))
        metric_reward_align = -jnp.mean(jnp.sum(kl_reshaped, axis=-1))  # mean over envs, sum over steps
     
        data = types.Transition(
            observation=data.observation,
            action=data.action,
            reward=data.reward - proxy_kl_coef * kl,
            # reward=data.reward,
            discount=data.discount,
            next_observation=data.next_observation,
            extras=data.extras,
        )

        # ── Bootstrap on timeout ─────────────────────────────────────
        if bootstrap_on_timeout:
            time_out = data.extras["state_extras"]["time_out"]
            value = data.extras["policy_extras"]["value"]
            data = types.Transition(
                observation=data.observation,
                action=data.action,
                reward=data.reward + discounting * time_out * value,
                discount=data.discount,
                next_observation=data.next_observation,
                extras=data.extras,
            )

        # ── Normalizer update ────────────────────────────────────────
        normalizer_params = training_state.normalizer_params
        if not lr_is_adaptive_kl:
            normalizer_params = running_statistics.update(
                normalizer_params,
                _remove_pixels(data.observation),
                pmap_axis_name=_PMAP_AXIS_NAME,
            )

        # ── PPO SGD ──────────────────────────────────────────────────
        (optimizer_state, params, _), metrics = jax.lax.scan(
            functools.partial(
                sgd_step, data=data, normalizer_params=normalizer_params,
            ),
            (training_state.optimizer_state, training_state.params, key_sgd),
            (),
            length=num_updates_per_batch,
        )

        if lr_is_adaptive_kl:
            normalizer_params = running_statistics.update(
                normalizer_params,
                _remove_pixels(data.observation),
                pmap_axis_name=_PMAP_AXIS_NAME,
            )

        metrics = {
            **metrics,
            "reward_align": metric_reward_align,
        }

        new_ts = TrainingState(
            optimizer_state=optimizer_state,
            params=params,
            normalizer_params=normalizer_params,
            env_steps=training_state.env_steps + jnp.int32(env_step_per_training_step),
            align_opt_state=training_state.align_opt_state,
            student_enc_params=training_state.student_enc_params,
            student_proprio_norm=training_state.student_proprio_norm,
            align_env_steps=training_state.align_env_steps,
        )
        return (new_ts, state, new_key), metrics

    def training_epoch(ts, state, key):
        (ts, state, _), loss_metrics = jax.lax.scan(
            training_step, (ts, state, key), (), length=num_training_steps_per_epoch,
        )
        loss_metrics = jax.tree_util.tree_map(jnp.mean, loss_metrics)
        return ts, state, loss_metrics

    training_epoch_pmap = jax.pmap(
        training_epoch, axis_name=_PMAP_AXIS_NAME, donate_argnums=(0, 1),
    )

    # ======================== align epoch ==============================

    def _generate_student_rollout(state, key, policy, num_unrolls):
        def _scan(carry, _):
            s, k = carry
            k, nk = jax.random.split(k)
            ns, data = acting.generate_unroll(
                student_env, s, policy, k, align_unroll_length,
                extra_fields=("truncation", "episode_metrics", "episode_done"),
            )
            return (ns, nk), data

        (next_state, _), data = jax.lax.scan(
            _scan, (state, key), (), length=num_unrolls,
        )
        data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
        data = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data
        )
        return next_state, data

    def align_epoch(ts: TrainingState, student_state, key):
        key_rollout, key_next = jax.random.split(key)

        # Build teacher policy from current PPO params (align-side teacher_network)
        teacher_norm = normalizer_select(ts.normalizer_params, policy_obs_key)
        student_proprio_norm = ts.student_proprio_norm

        def _teacher_policy(observations, key_sample):
            logits = sitt_network.teacher_network.apply(
                teacher_norm, ts.params.policy, observations
            )
            raw = sitt_network.parametric_action_distribution.sample_no_postprocessing(
                logits, key_sample
            )
            log_prob = sitt_network.parametric_action_distribution.log_prob(logits, raw)
            post = sitt_network.parametric_action_distribution.postprocess(raw)
            return post, {
                "log_prob": log_prob,
                "raw_action": raw,
                "distribution_params": logits,
            }

        # Collect rollouts on student env with teacher policy
        student_state, data = _generate_student_rollout(
            student_state, key_rollout, _teacher_policy, align_unrolls_per_epoch,
        )

        # Extract student env episode metrics from rollout.
        # episode_done lives in state_extras; reward_total is the key used by
        # the student env's _initialize_metrics (NOT 'episode_reward').
        ep_done = data.extras["state_extras"]["episode_done"]
        ep_reward = data.extras["state_extras"]["episode_metrics"]["reward_total"]
        completed_mask = ep_done.astype(jnp.float32)
        n_completed = jnp.maximum(jnp.sum(completed_mask), 1.0)
        student_ep_reward = jnp.sum(ep_reward * completed_mask) / n_completed

        # Update student proprio normalizer with latest rollout data.
        student_proprio_norm = running_statistics.update(
            student_proprio_norm,
            data.observation[policy_proprio_key],
            pmap_axis_name=_PMAP_AXIS_NAME,
        )

        # Run alignment: train student_enc and proxy_dec
        (
            student_enc, proxy_dec, align_opt_state,
            align_loss, embed_loss, action_loss,
        ) = align_fn(
            ts.student_enc_params,
            ts.params.proxy_dec_params,
            ts.align_opt_state,
            data.observation,            # teacher_obs (teacher_obs_key extracted inside)
            data.observation,            # student_obs (proprio + pixels extracted inside)
            student_proprio_norm,        # proprio_norm (student-specific stats)
            teacher_dec_params=ts.params.policy[0],   # frozen during align
            action_head_params=ts.params.policy[1],   # frozen during align
            teacher_norm=teacher_norm,
        )

        new_ts = ts.replace(
            student_enc_params=student_enc,
            params=ts.params.replace(proxy_dec_params=proxy_dec),
            align_opt_state=align_opt_state,
            student_proprio_norm=student_proprio_norm,
            align_env_steps=ts.align_env_steps + jnp.int32(align_env_steps_per_epoch),
        )

        metrics = {
            "align_loss": align_loss,
            "embed_loss": embed_loss,
            "action_loss": action_loss,
            "student_episode_reward": student_ep_reward,
        }
        return new_ts, student_state, metrics

    # Batch all alignment epochs into a single jax.lax.scan inside pmap
    # to avoid the massive overhead of a Python for-loop calling pmap
    # repeatedly (e.g. 158 pmap dispatch+sync round-trips per iteration).
    max_align_epochs = max(align_epochs_by_iter.values()) if align_epochs_by_iter else 0

    def align_n_epochs(ts, student_state, key):
        def _body(carry, _):
            ts, ss, k = carry
            k, nk = jax.random.split(k)
            ts, ss, m = align_epoch(ts, ss, k)
            return (ts, ss, nk), m

        (ts, ss, _), all_metrics = jax.lax.scan(
            _body, (ts, student_state, key), (), length=max_align_epochs,
        )
        mean_metrics = jax.tree_util.tree_map(jnp.mean, all_metrics)
        return ts, ss, mean_metrics

    if max_align_epochs > 0:
        align_n_epochs_pmap = jax.pmap(
            align_n_epochs, axis_name=_PMAP_AXIS_NAME, donate_argnums=(0, 1),
        )

    # ======================== evaluators ==============================

    teacher_evaluator = acting.Evaluator(
        teacher_env,
        functools.partial(make_ppo_policy, deterministic=deterministic_eval),
        num_eval_envs=num_eval_envs,
        episode_length=episode_length,
        action_repeat=action_repeat,
        key=eval_key,
    )

    student_evaluator = acting.Evaluator(
        student_env,
        make_student_policy,
        num_eval_envs=num_eval_envs,
        episode_length=episode_length,
        action_repeat=action_repeat,
        key=jax.random.fold_in(eval_key, 1),
    )

    # ======================== replicate ================================

    training_state = jax.device_put_replicated(
        training_state, jax.local_devices()[:local_devices_to_use]
    )

    # ======================== main loop ================================

    training_walltime = 0.0
    current_step = 0
    metrics: Metrics = {}

    # Initial eval
    if process_id == 0 and num_evals > 1 and run_evals:
        ppo_params = _unpmap(_pack_ppo_inference_params(training_state))
        metrics = teacher_evaluator.run_evaluation(ppo_params, training_metrics={})
        logging.info(metrics)
        progress_fn(0, metrics)

    params = _unpmap(_pack_ppo_inference_params(training_state))
    policy_params_fn(current_step, make_ppo_policy, params)

    for it in range(num_evals_after_init):
        logging.info("SITT iteration %d/%d  (%.1fs)", it + 1, num_evals_after_init, time.time() - xt)

        # ── PPO epoch ─────────────────────────────────────────────────
        ppo_metrics = {}
        ppo_epoch_time = 0.0
        if not ppo_off:
            for _ in range(max(num_resets_per_eval, 1)):
                t0 = time.time()
                epoch_key, local_key = jax.random.split(local_key)
                epoch_keys = jax.random.split(epoch_key, local_devices_to_use)

                training_state, teacher_env_state = _strip_weak_type(
                    (training_state, teacher_env_state)
                )
                training_state, teacher_env_state, ppo_metrics = training_epoch_pmap(
                    training_state, teacher_env_state, epoch_keys,
                )
                ppo_metrics = jax.tree_util.tree_map(jnp.mean, ppo_metrics)
                jax.tree_util.tree_map(lambda x: x.block_until_ready(), ppo_metrics)

                ppo_epoch_time = time.time() - t0
                training_walltime += ppo_epoch_time
                current_step = int(_unpmap(training_state.env_steps))

            ppo_sps = (
                num_training_steps_per_epoch
                * env_step_per_training_step
                * max(num_resets_per_eval, 1)
            ) / max(ppo_epoch_time, 1e-9)
        else:
            ppo_sps = 0.0
            current_step = int(_unpmap(training_state.env_steps))

        # ── Align epochs (batched into single XLA scan) ───────────────
        align_metrics_agg = {}
        align_epochs_now = align_epochs_by_iter.get(it, 0)
        if align_epochs_now > 0:
            t0 = time.time()
            align_key, local_key = jax.random.split(local_key)
            align_keys = jax.random.split(align_key, local_devices_to_use)

            training_state, student_env_state = _strip_weak_type(
                (training_state, student_env_state)
            )
            training_state, student_env_state, a_metrics = align_n_epochs_pmap(
                training_state, student_env_state, align_keys,
            )
            a_metrics = jax.tree_util.tree_map(jnp.mean, a_metrics)
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), a_metrics)
            training_walltime += time.time() - t0
            align_metrics_agg = {k: float(v) for k, v in a_metrics.items()}

        current_align_step = int(_unpmap(training_state.align_env_steps))

        # ── Eval & progress ───────────────────────────────────────────
        if process_id == 0:
            ppo_params = _unpmap(_pack_ppo_inference_params(training_state))
            policy_params_fn(current_step, make_ppo_policy, ppo_params)

            training_metrics_out = {
                "training/sps": ppo_sps,
                "training/walltime": training_walltime,
                **{f"training/{k}": float(v) for k, v in ppo_metrics.items()},
            }

            if run_evals:
                metrics = teacher_evaluator.run_evaluation(
                    ppo_params, training_metrics_out,
                )
                # Also evaluate student
                student_params = _unpmap(_pack_student_params(training_state))
                student_metrics = student_evaluator.run_evaluation(
                    student_params, {},
                )
                for k, v in student_metrics.items():
                    if k.startswith("eval/"):
                        metrics[f"student_{k}"] = v
            else:
                metrics = training_metrics_out

            # Add align metrics with separate step counter
            metrics["rl_env_steps"] = current_step
            metrics["align_env_steps"] = current_align_step
            for k, v in align_metrics_agg.items():
                metrics[f"align/{k}"] = v
                # metrics[f"training/{k}"] = v
                # metrics[f"training_align/{k}"] = v

            logging.info(
                "SITT iter %d  ppo_steps=%d  align_steps=%d  "
                "ppo_loss=%.4f  align_loss=%.4f  sps=%.0f",
                it + 1, current_step, current_align_step,
                float(ppo_metrics.get("total_loss", 0.0)),
                align_metrics_agg.get("align_loss", 0.0),
                ppo_sps,
            )
            log_step = current_align_step if ppo_off else current_step
            progress_fn(log_step, metrics)

    # ======================== done ====================================

    pmap.assert_is_replicated(training_state)
    ppo_params = _unpmap(_pack_ppo_inference_params(training_state))
    student_params = _unpmap(_pack_student_params(training_state))

    # Teacher checkpoint: (norm, policy=(dec, head), value)
    teacher_checkpoint_params = (
        ppo_params[0],   # normalizer
        ppo_params[1],   # policy = (teacher_dec_params, action_head_params)
        ppo_params[2],   # value
    )

    # Student checkpoint: (proprio_norm, (student_enc_params, action_head_params))
    student_checkpoint_params = (
        student_params[0],                                 # student proprio_norm
        (student_params[1], action_head_params),           # (student_enc, action_head)
    )

    # Proxy checkpoint: proxy decoder params only.
    proxy_checkpoint_params = _unpmap(training_state.params.proxy_dec_params)

    logging.info(
        "SITT training complete — ppo_steps: %d, align_steps: %d",
        current_step, current_align_step,
    )
    pmap.synchronize_hosts()
    return (
        make_ppo_policy,
        make_student_policy,
        teacher_checkpoint_params,
        student_checkpoint_params,
        proxy_checkpoint_params,
        metrics,
    )
