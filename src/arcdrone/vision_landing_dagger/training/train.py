"""DAgger (Dataset Aggregation) training loop — trainable action head.

Unlike vision_landing_il, DAgger:
  1. Rolls out a β-mixture of teacher and student to collect states under the
     student's own distribution (addressing distribution shift).
  2. Trains BOTH the student encoder AND the student action head from scratch.
     The teacher action head is only used to generate action labels.

β decays linearly from ``beta_start`` to ``beta_end`` across epochs.

Imports:
  - Networks / align from arcdrone.vision_landing_dagger.training (own copies)
  - Task/env  from arcdrone.vision_landing_il.task  (shared, unchanged)

Checkpoint layout (compatible with evaluate.py):
    (proprio_norm, (student_enc_params, student_action_head_params))
"""

import functools
import time
from typing import Any, Callable, Optional, Tuple

from absl import logging
from brax import envs
from brax.training import acting
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.acme import specs
from brax.training.types import PRNGKey

from arcdrone.vision_landing_dagger.training import networks as dagger_networks
from arcdrone.vision_landing_dagger.training.align import align

import flax
import jax
import jax.numpy as jnp
import optax


Metrics = types.Metrics
InferenceParams = Any

_PMAP_AXIS_NAME = "i"


# ====================== dataclass ======================

@flax.struct.dataclass
class DAggerTrainingState:
    """Mutable state for the DAgger learner.

    Both student_enc_params and action_head_params are trainable.
    Teacher params are Python-level constants, never in JAX state.
    """
    align_opt_state: optax.OptState   # covers (student_enc, action_head) jointly
    student_enc_params: Any
    action_head_params: Any           # student's own head, trained from scratch
    normalizer_params: Any
    env_steps: Any


# ====================== helpers ======================

def _unpmap(v):
    return jax.tree_util.tree_map(lambda x: x[0] if x is not None else None, v)


def _strip_weak_type(tree):
    def f(leaf):
        leaf = jnp.asarray(leaf)
        return jnp.astype(leaf, leaf.dtype)
    return jax.tree_util.tree_map(f, tree)


# ====================== main ======================

def train(
    env: envs.Env,
    teacher_params: Any,
    *,
    # --- DAgger schedule ---
    num_dagger_epochs: int = 200,
    beta_start: float = 1.0,
    beta_end: float = 0.0,
    # --- data collection ---
    num_evals: int = 10,
    unroll_length: int = 10,
    batch_size: int = 32,
    num_minibatches: int = 16,
    align_updates_per_trigger: int = 4,
    # --- architecture ---
    network_factory: types.NetworkFactory[
        dagger_networks.ILNetworks
    ] = dagger_networks.make_il_networks,
    # --- environment ---
    num_envs: int = 1,
    episode_length: Optional[int] = None,
    action_repeat: int = 1,
    num_eval_envs: int = 128,
    deterministic_eval: bool = True,
    # --- normalizer ---
    normalize_observations: bool = False,
    # --- optimizer ---
    learning_rate: float = 1e-4,
    # --- misc ---
    seed: int = 0,
    max_devices_per_host: Optional[int] = None,
    # --- callbacks ---
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    policy_params_fn: Callable[..., None] = lambda *args: None,
    run_evals: bool = True,
    wrap_env: bool = False,
    wrap_env_fn: Optional[Callable[[Any], Any]] = None,
    # --- restore ---
    restore_params: Optional[Any] = None,
):
    """DAgger training — β-mixture rollout + joint encoder/action-head training.

    Returns:
        ``(make_policy, checkpoint_params, metrics)``

        ``checkpoint_params`` layout:
            ``(proprio_norm, (student_enc_params, student_action_head_params))``
    """

    xt = time.time()

    # ========================= device setup =========================

    process_count = jax.process_count()
    process_id = jax.process_index()
    local_device_count = jax.local_device_count()
    local_devices_to_use = local_device_count
    if max_devices_per_host:
        local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
    device_count = local_devices_to_use * process_count

    logging.info(
        "DAgger — Device count: %d, process count: %d (id %d), "
        "local device count: %d, devices to be used count: %d",
        jax.device_count(), process_count, process_id,
        local_device_count, local_devices_to_use,
    )

    # ========================= env setup =========================

    if wrap_env:
        if episode_length is None:
            raise ValueError("episode_length required when wrap_env=True")
        wrap_fn = wrap_env_fn or envs.training.wrap
        env = wrap_fn(env, episode_length=episode_length, action_repeat=action_repeat)

    env_steps_per_epoch = batch_size * unroll_length * num_minibatches * action_repeat

    # ========================= keys =========================

    key = jax.random.PRNGKey(seed)
    global_key, local_key = jax.random.split(key)
    del key
    local_key = jax.random.fold_in(local_key, process_id)
    local_key, key_env, eval_key = jax.random.split(local_key, 3)
    key_policy, key_init = jax.random.split(global_key, 2)
    key_enc, key_head = jax.random.split(key_init)
    del global_key

    # ========================= reset envs =========================

    assert num_envs % device_count == 0
    assert batch_size * num_minibatches % num_envs == 0, (
        f"batch_size ({batch_size}) * num_minibatches ({num_minibatches}) "
        f"must be divisible by num_envs ({num_envs})"
    )

    num_envs_per_device = num_envs // device_count
    num_unrolls_per_epoch = batch_size * num_minibatches // num_envs_per_device

    key_envs = jax.random.split(key_env, num_envs // process_count)
    key_envs = jnp.reshape(
        key_envs, (local_devices_to_use, -1) + key_envs.shape[1:]
    )

    reset_fn = jax.pmap(env.reset, axis_name=_PMAP_AXIS_NAME)
    env_state = reset_fn(key_envs)

    # ========================= observation shapes =========================

    obs_shape = jax.tree_util.tree_map(lambda x: x.shape[2:], env_state.obs)

    normalize = lambda x, y: x
    if normalize_observations:
        normalize = running_statistics.normalize

    # ========================= networks =========================

    il_network = network_factory(
        obs_shape,
        env.action_size,
        preprocess_observations_fn=normalize,
    )

    # ========================= optimizer =========================

    align_optimizer = optax.adam(learning_rate)

    # ========================= decompose teacher checkpoint =========================

    from brax.training.networks import normalizer_select as _nsel
    teacher_norm_params         = teacher_params[0]
    teacher_policy_params       = teacher_params[1]
    teacher_dec_params          = teacher_policy_params[0]
    teacher_action_head_params  = teacher_policy_params[1]   # frozen label source

    # TODO: Update privileged RL keys — checkpoint uses "policy_obs" / "value_obs"
    teacher_obs_norm = _nsel(teacher_norm_params, "policy_obs")

    # ── Frozen teacher policy for the mixture rollout ──
    frozen_teacher_policy = dagger_networks.make_frozen_teacher_policy(
        il_network,
        teacher_norm_params=teacher_obs_norm,
        teacher_policy_params=teacher_policy_params,
        deterministic=False,
    )

    # ── Student inference factory (action head comes from params at call time) ──
    make_student_policy = dagger_networks.make_student_inference_fn(il_network)

    def _pack_student_params(ts: DAggerTrainingState):
        """Returns (proprio_norm, student_enc, student_action_head)."""
        return (ts.normalizer_params, ts.student_enc_params, ts.action_head_params)

    # ── Align function — teacher_action_head_params pre-bound as frozen labels ──
    align_fn = functools.partial(
        align,
        il_network=il_network,
        optimizer=align_optimizer,
        align_updates_per_trigger=align_updates_per_trigger,
        num_minibatches=num_minibatches,
        teacher_dec_params=teacher_dec_params,
        teacher_action_head_params=teacher_action_head_params,  # frozen
        teacher_norm=teacher_obs_norm,
    )

    # ========================= init params =========================

    # Fresh student encoder
    student_enc_params  = il_network.student_encoder.init(key_enc)
    # Fresh student action head (trained from scratch, NOT copied from teacher)
    action_head_params  = il_network.action_head.init(key_head)

    # Normalizer: proprio running stats only
    proprio_spec = specs.Array(
        env_state.obs["proprio_obs"].shape[2:], jnp.dtype("float32")
    )
    proprio_norm = running_statistics.init_state(proprio_spec)

    if restore_params is not None:
        # Support two checkpoint layouts:
        #   NEW: (proprio_norm, (student_enc, student_action_head))
        #   OLD: (proprio_norm, student_enc)   — from IL or pre-trainable-head DAgger
        restored_norm = restore_params[0]
        restored_policy = restore_params[1]
        if isinstance(restored_policy, (tuple, list)) and len(restored_policy) == 2:
            restored_enc, restored_head = restored_policy
            student_enc_params = restored_enc
            action_head_params = restored_head
            logging.info("Restored student encoder + action head (new format).")
        else:
            student_enc_params = restored_policy
            logging.info("Restored student encoder only (old format, action head stays fresh).")

        # Only restore normalizer if it's a flat array state matching current
        # proprio shape.  Old checkpoints may have a dict-shaped normalizer or
        # a different size (e.g. before linvel was added to proprio).
        try:
            restored_mean_shape = restored_norm.mean.shape
            if restored_mean_shape == proprio_norm.mean.shape:
                proprio_norm = restored_norm
                logging.info("Restored proprio normalizer.")
            else:
                logging.warning(
                    "Skipping normalizer restore: shape mismatch "
                    "(checkpoint %s vs current %s).",
                    restored_mean_shape, proprio_norm.mean.shape,
                )
        except AttributeError:
            logging.warning(
                "Skipping normalizer restore: checkpoint normalizer has "
                "incompatible structure (dict vs array)."
            )

    # Optimizer covers (student_enc_params, action_head_params) as a tuple tree
    training_state = DAggerTrainingState(
        align_opt_state=align_optimizer.init((student_enc_params, action_head_params)),
        student_enc_params=student_enc_params,
        action_head_params=action_head_params,
        normalizer_params=proprio_norm,
        env_steps=jnp.uint32(0),
    )

    # ========================= DAgger mixture policy =========================

    def _make_dagger_policy(student_policy_fn, beta):
        """β-mixture: Bernoulli(β)=1 → teacher action, 0 → student action."""

        def policy(observations, key_sample):
            key_teacher, key_student, key_coin = jax.random.split(key_sample, 3)
            teacher_action, _ = frozen_teacher_policy(observations, key_teacher)
            student_action, student_extras = student_policy_fn(observations, key_student)
            use_teacher = jax.random.bernoulli(key_coin, beta)
            action = jnp.where(use_teacher, teacher_action, student_action)
            extras = {
                "log_prob": student_extras.get("log_prob", jnp.zeros(())),
                "raw_action": student_extras.get("raw_action", action),
            }
            return action, extras

        return policy

    # ========================= rollout helper =========================

    def _generate_rollout_data(state, key_unroll, policy, num_unrolls):
        def _scan_unroll(carry, _):
            current_state, current_key = carry
            current_key, next_key = jax.random.split(current_key)
            next_state, data = acting.generate_unroll(
                env, current_state, policy, current_key, unroll_length,
                extra_fields=("truncation", "episode_metrics", "episode_done"),
            )
            return (next_state, next_key), data

        (next_state, _), data = jax.lax.scan(
            _scan_unroll, (state, key_unroll), (), length=num_unrolls,
        )
        data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
        data = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data
        )
        return next_state, data

    # ========================= one DAgger epoch =========================

    def dagger_epoch(training_state, env_state, key, beta):
        key_rollout, _ = jax.random.split(key)

        # 1. Build current student policy (live params)
        student_params = _pack_student_params(training_state)
        student_policy_fn = make_student_policy(student_params, deterministic=False)

        # 2. Build β-mixture rollout policy
        mixture_policy = _make_dagger_policy(student_policy_fn, beta)

        # 3. Collect data under mixture distribution
        env_state, data = _generate_rollout_data(
            env_state, key_rollout, mixture_policy, num_unrolls_per_epoch,
        )

        # 4. Joint alignment: train (student_enc, student_action_head) together
        trainable_in = (training_state.student_enc_params, training_state.action_head_params)
        (new_student_enc, new_action_head), align_opt_state, align_loss, embed_loss, action_loss = align_fn(
            trainable_in,
            training_state.align_opt_state,
            data.observation,
            data.observation,
            training_state.normalizer_params,
        )

        # 5. Update proprio running stats
        normalizer_params = training_state.normalizer_params
        if normalize_observations:
            normalizer_params = running_statistics.update(
                normalizer_params, data.observation["proprio_obs"],
            )

        new_state = DAggerTrainingState(
            align_opt_state=align_opt_state,
            student_enc_params=new_student_enc,
            action_head_params=new_action_head,
            normalizer_params=normalizer_params,
            env_steps=training_state.env_steps + jnp.uint32(env_steps_per_epoch),
        )

        metrics = {
            "align_loss": align_loss,
            "embed_loss": embed_loss,
            "action_loss": action_loss,
            "beta": beta,
        }
        return new_state, env_state, metrics

    # ========================= pmap epoch =========================

    dagger_epoch_jit = jax.pmap(
        dagger_epoch,
        axis_name=_PMAP_AXIS_NAME,
        in_axes=(0, 0, 0, None),   # beta is a scalar broadcast to all devices
        donate_argnums=(0, 1),
    )

    # ========================= eval =========================

    evaluator = acting.Evaluator(
        env,
        make_student_policy,
        num_eval_envs=num_eval_envs,
        episode_length=episode_length,
        action_repeat=action_repeat,
        key=eval_key,
    )

    # ========================= replicate =========================

    training_state = jax.tree_util.tree_map(
        lambda x: jnp.expand_dims(jnp.asarray(x), 0),
        training_state,
    )

    # ========================= main loop =========================

    training_walltime = 0.0
    eval_every = max(num_dagger_epochs // max(num_evals, 1), 1)

    metrics: Metrics = {}
    if process_id == 0 and run_evals:
        metrics = evaluator.run_evaluation(
            _unpmap(_pack_student_params(training_state)),
            training_metrics={},
        )
        logging.info(metrics)
        progress_fn(0, metrics)

    params = _unpmap(_pack_student_params(training_state))
    policy_params_fn(0, make_student_policy, params)

    current_step = 0
    for epoch in range(num_dagger_epochs):
        t0 = time.time()

        # Linear β decay
        progress = epoch / (num_dagger_epochs - 1) if num_dagger_epochs > 1 else 1.0
        beta = jnp.float32(beta_start + (beta_end - beta_start) * progress)

        epoch_key, local_key = jax.random.split(local_key)
        epoch_keys = jax.random.split(epoch_key, local_devices_to_use)

        training_state, env_state = _strip_weak_type((training_state, env_state))
        training_state, env_state, train_metrics = dagger_epoch_jit(
            training_state, env_state, epoch_keys, beta
        )
        train_metrics = jax.tree_util.tree_map(jnp.mean, train_metrics)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), train_metrics)

        epoch_time = time.time() - t0
        training_walltime += epoch_time
        current_step = int(_unpmap(training_state.env_steps))
        sps = env_steps_per_epoch / epoch_time

        if process_id == 0:
            student_params = _unpmap(_pack_student_params(training_state))
            policy_params_fn(current_step, make_student_policy, student_params)

            should_eval = (
                run_evals
                and ((epoch + 1) % eval_every == 0 or epoch == num_dagger_epochs - 1)
            )
            training_metrics_out = {
                "training/align_loss":  float(train_metrics["align_loss"]),
                "training/embed_loss":  float(train_metrics["embed_loss"]),
                "training/action_loss": float(train_metrics["action_loss"]),
                "training/beta":        float(train_metrics["beta"]),
                "training/sps":         sps,
                "training/walltime":    training_walltime,
            }

            if should_eval:
                metrics = evaluator.run_evaluation(
                    _unpmap(_pack_student_params(training_state)),
                    training_metrics_out,
                )
            else:
                metrics = training_metrics_out

            logging.info(
                "DAgger epoch %d/%d  β=%.3f  env_steps=%d  "
                "align_loss=%.4f  action_loss=%.4f  sps=%.0f",
                epoch + 1, num_dagger_epochs, float(beta), current_step,
                float(train_metrics["align_loss"]),
                float(train_metrics["action_loss"]), sps,
            )
            progress_fn(current_step, metrics)

    # ========================= done =========================

    params = _unpmap(_pack_student_params(training_state))
    logging.info("DAgger training complete — total env steps: %d", current_step)

    # Checkpoint: (proprio_norm, (student_enc_params, student_action_head_params))
    # NOTE: student_action_head_params here is the TRAINED student head,
    # not the teacher's. evaluate.py loads [1][1] as action_head_params.
    checkpoint_params = (params[0], (params[1], params[2]))
    return (make_student_policy, checkpoint_params, metrics)
