"""Pure Imitation Learning training.

Given a pre-trained RL teacher checkpoint, trains a student network to
reproduce the teacher's behaviour using the SITT alignment loss.

No PPO, no value function, no GAE — just supervised feature/action matching.

Usage from the hydra entry-point::

    arcdrone-train-il task=landing

Or programmatically::

    from arcdrone.vision_landing_il.training.train import train
    make_policy, params, metrics = train(
        teacher_env=env,
        teacher_params=teacher_params,
        ...
    )
"""

import functools
import time
from typing import Any, Callable, Mapping, Optional, Tuple

from absl import logging
from brax import envs
from brax.training import acting
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.acme import specs
from brax.training.types import PRNGKey

from arcdrone.vision_landing_il.training import networks as il_networks
from arcdrone.vision_landing_il.training.align import align

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax


Metrics = types.Metrics
InferenceParams = Any


_PMAP_AXIS_NAME = "i"


# ====================== dataclass ======================

@flax.struct.dataclass
class ILTrainingState:
    """Minimal mutable state for the IL learner.

    Only student_enc_params is updated during training.
    Teacher policy/value params are Python-level constants (closed over),
    not JAX arrays in the training state.
    """
    align_opt_state: optax.OptState
    student_enc_params: Any   # PolicyVisionProprioEncoder — the only trainable params
    normalizer_params: Any    # running stats for non-pixel obs (proprio updated; others frozen)
    env_steps: Any            # jnp scalar (uint32)


# ====================== helpers ======================

def _unpmap(v):
    return jax.tree_util.tree_map(lambda x: x[0] if x is not None else None, v)


def _strip_weak_type(tree):
    def f(leaf):
        leaf = jnp.asarray(leaf)
        return jnp.astype(leaf, leaf.dtype)
    return jax.tree_util.tree_map(f, tree)


# _pack_inference_params is defined as a closure inside train() once teacher
# params are known (see below).


# ====================== main ======================

def train(
    env: envs.Env,
    teacher_params: Any,
    *,
    # --- schedule ---
    num_il_epochs: int = 200,
    num_evals: int = 10,
    unroll_length: int = 10,
    batch_size: int = 32,
    num_minibatches: int = 16,
    align_updates_per_trigger: int = 4,
    # --- architecture ---
    network_factory: types.NetworkFactory[
        il_networks.ILNetworks
    ] = il_networks.make_il_networks,
    # --- environment ---
    num_envs: int = 1,
    episode_length: Optional[int] = None,
    action_repeat: int = 1,
    num_eval_envs: int = 128,
    deterministic_eval: bool = True,
    # --- normalizer ---
    normalize_observations: bool = False,
    normalize_observations_std_eps: float = 0.0,
    normalize_observations_mode: str = "welford",
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
    """Pure IL training loop.

    Args:
        teacher_params: 3-tuple ``(normalizer, policy, value)`` loaded from a
            priviledged_landing_rl PPO checkpoint.  Only the teacher policy
            weights are used for rollouts; student encoder is re-initialised.
        restore_params: optional student checkpoint ``(proprio_norm, (student_enc, action_head))``
            to resume training from a previous IL run.
        num_il_epochs: total number of collect-then-align cycles.
        num_evals: how often to evaluate (spread across ``num_il_epochs``).
        unroll_length: number of env steps per unroll segment.
        batch_size: number of environment trajectories per minibatch (matches PPO
            semantics).  The CNN sees ``batch_size * unroll_length`` images per
            gradient step.
        num_minibatches: number of minibatches to split the collected data into
            per epoch.  ``num_unrolls_per_epoch`` is inferred as
            ``batch_size * num_minibatches // num_envs``.
        align_updates_per_trigger: how many full passes over all minibatches to
            run per epoch (≈ ``num_updates_per_batch`` in PPO).  Total gradient
            steps = ``align_updates_per_trigger * num_minibatches``.
        network_factory: factory that builds the ``ILNetworks``.
        num_envs: parallel environments.
        episode_length: max episode length (used by the wrapper).
        action_repeat: action repeat for the env wrapper.
        num_eval_envs: envs used for evaluation rollouts.
        deterministic_eval: if ``True``, use deterministic (mode) policy for eval.
        normalize_observations: whether to use running-statistics normalizer.
        learning_rate: Adam LR for the alignment optimizer.
        seed: random seed.
        max_devices_per_host: cap on devices per host process.
        progress_fn: callback ``(step, metrics) -> None``.
        policy_params_fn: callback ``(step, make_policy, params) -> None``.
        run_evals: if ``False`` skip evaluation rollouts.
        wrap_env: if ``True``, wrap envs with the Brax training wrapper.
        wrap_env_fn: custom wrapping function.

    Returns:
        ``(make_policy, params, metrics)`` where ``make_policy`` is the
        **student** inference function and ``params`` is ``(norm, (student_enc, action_head))``.
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
        "IL — Device count: %d, process count: %d (id %d), "
        "local device count: %d, devices to be used count: %d",
        jax.device_count(), process_count, process_id,
        local_device_count, local_devices_to_use,
    )

    # ========================= env setup =========================
    # The caller (scripts/train.py) is expected to wrap the env via
    # mujoco_playground's wrapper BEFORE passing it here, so wrap_env
    # defaults to False.  Optionally, we support the Brax-style wrapper.

    if wrap_env:
        if episode_length is None:
            raise ValueError("episode_length required when wrap_env=True")
        wrap_fn = wrap_env_fn or envs.training.wrap
        teacher_env = wrap_fn(
            teacher_env,
            episode_length=episode_length,
            action_repeat=action_repeat,
        )

    env = env

    # env-steps per epoch — matches PPO formula exactly:
    #   env_step_per_training_step = batch_size * unroll_length * num_minibatches * action_repeat
    env_steps_per_epoch = batch_size * unroll_length * num_minibatches * action_repeat

    # ========================= keys =========================

    key = jax.random.PRNGKey(seed)
    global_key, local_key = jax.random.split(key)
    del key
    local_key = jax.random.fold_in(local_key, process_id)
    local_key, key_env, eval_key = jax.random.split(local_key, 3)
    key_policy, key_il = jax.random.split(global_key, 2)
    del global_key

    # ========================= reset envs =========================

    assert num_envs % device_count == 0
    assert batch_size * num_minibatches % num_envs == 0, (
        f"batch_size ({batch_size}) * num_minibatches ({num_minibatches}) "
        f"must be divisible by num_envs ({num_envs})"
    )

    # Number of rollout segments collected per epoch per device.
    # Mirrors PPO: batch_size * num_minibatches // num_envs
    num_envs_per_device = num_envs // device_count
    num_unrolls_per_epoch = batch_size * num_minibatches // num_envs_per_device

    key_envs = jax.random.split(key_env, num_envs // process_count)
    key_envs = jnp.reshape(
        key_envs, (local_devices_to_use, -1) + key_envs.shape[1:]
    )

    # Always pmap (even 1 device) to avoid nested-vmap issues with warp rendering.
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

    # ========================= init params =========================

    # ── Decompose teacher checkpoint ──────────────────────────────────────
    # Layout from priviledged_landing_rl PPO: (norm, policy, value)
    #   norm   = RunningStatisticsState with dict leaves keyed by obs name
    #   policy = (teacher_dec_params, action_head_params)
    #   value  = value_params
    from brax.training.networks import normalizer_select as _nsel
    teacher_norm_params   = teacher_params[0]   # frozen constant
    teacher_policy_params = teacher_params[1]   # (teacher_dec, action_head) — frozen constant
    teacher_value_params  = teacher_params[2]   # frozen constant
    teacher_dec_params    = teacher_policy_params[0]
    action_head_params    = teacher_policy_params[1]

    # Extract per-key sub-states (Python constants — never enter JAX training state).
    # TODO: Update priviledged RL keys... Teacher checkpoint was trained with keys "policy_obs" and "value_obs".
    teacher_obs_norm = _nsel(teacher_norm_params, "policy_obs")
    value_obs_norm   = _nsel(teacher_norm_params, "value_obs")  # noqa: F841

    # ── Frozen teacher policy for rollouts (stochastic, all constants baked in) ──
    frozen_teacher_policy = il_networks.make_frozen_teacher_policy(
        il_network,
        teacher_norm_params=teacher_obs_norm,
        teacher_policy_params=teacher_policy_params,
        deterministic=False,
    )

    # # Optionally wrap the frozen teacher policy to inject Gaussian action noise.
    # def _make_noisy_teacher_policy(base_policy, sigma: float = None, teacher_action_noise_sigma: float = None):
    #     # Accept either positional `sigma` or keyword `teacher_action_noise_sigma`.
    #     actual_sigma = sigma if sigma is not None else teacher_action_noise_sigma
    #     if actual_sigma is None or actual_sigma <= 0.0:
    #         return base_policy

    #     def policy_with_noise(observations, key_sample):
    #         action, extras = base_policy(observations, key_sample)
    #         key_sample, key_noise = jax.random.split(key_sample)
    #         noise = actual_sigma * jax.random.normal(key_noise, shape=action.shape)
    #         noisy_action = jnp.clip(action + noise, -1.0, 1.0)
    #         extras = {**extras, "teacher_noise": noise}
    #         return noisy_action, extras

    #     return policy_with_noise

    # frozen_teacher_policy = _make_noisy_teacher_policy(frozen_teacher_policy, teacher_action_noise_sigma=0.4)
    

    # ── Student inference factory (action_head baked in as closure) ──────────
    make_student_policy = il_networks.make_student_inference_fn(
        il_network,
        action_head_params=action_head_params,
    )

    def _pack_student_params(ts: ILTrainingState):
        """Pack replicated-only tensors for evaluator + _unpmap.

        Returns (proprio_norm, student_enc_params) — both are part of
        ILTrainingState and are correctly replicated across devices, so
        _unpmap (x[0]) is safe on their leaves.

        action_head_params is NOT included here because it is a Python-level
        constant that was never replicated; _unpmap would corrupt it by
        indexing [0] on non-replicated arrays.
        """
        return (ts.normalizer_params, ts.student_enc_params)

    # ── Align function: teacher constants pre-bound via partial ───────────
    align_fn = functools.partial(
        align,
        il_network=il_network,
        optimizer=align_optimizer,
        align_updates_per_trigger=align_updates_per_trigger,
        num_minibatches=num_minibatches,         # for the minibatch inner loop
        teacher_dec_params=teacher_dec_params,   # constant — pre-bound here
        action_head_params=action_head_params,   # constant — pre-bound here
        teacher_norm=teacher_obs_norm,           # constant — pre-bound here
    )

    # ── Fresh student encoder ──────────────────────────────────────────────
    student_enc_params = il_network.student_encoder.init(key_il)

    # ── Normalizer: proprio only ────────────────────────────────────────────
    # teacher_obs_norm and value_obs_norm are Python-level constants closed
    # over by the network apply functions — they never go into ILTrainingState.
    # Only proprio stats are dynamic and need to be tracked.
    proprio_spec = specs.Array(
        env_state.obs["proprio_obs"].shape[2:], jnp.dtype("float32")
    )
    proprio_norm = running_statistics.init_state(proprio_spec)

    if restore_params is not None:
        restored_norm, (restored_student_enc, _) = restore_params
        student_enc_params = restored_student_enc
        proprio_norm = restored_norm
        logging.info("Restored student encoder + proprio norm from checkpoint.")

    training_state = ILTrainingState(
        align_opt_state=align_optimizer.init(student_enc_params),
        student_enc_params=student_enc_params,
        normalizer_params=proprio_norm,
        env_steps=jnp.uint32(0),
    )

    # ========================= rollout helper =========================

    def _generate_rollout_data(
        state: envs.State,
        key_unroll: PRNGKey,
        policy: Callable[..., Any],
        num_unrolls: int,
    ) -> Tuple[envs.State, types.Transition]:
        """Collect ``num_unrolls * unroll_length`` env transitions."""

        def _scan_unroll(carry, unused_t):
            current_state, current_key = carry
            current_key, next_key = jax.random.split(current_key)
            next_state, data = acting.generate_unroll(
                env,
                current_state,
                policy,
                current_key,
                unroll_length,
                extra_fields=("truncation", "episode_metrics", "episode_done"),
            )
            return (next_state, next_key), data

        (next_state, _), data = jax.lax.scan(
            _scan_unroll,
            (state, key_unroll),
            (),
            length=num_unrolls,
        )
        # [num_unrolls, envs_per_device, T, ...] -> [num_unrolls*envs_per_device, T, ...]
        data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
        data = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data
        )
        return next_state, data

    # ========================= one IL epoch =========================

    def il_epoch(
        training_state: ILTrainingState,
        env_state: envs.State,
        key: PRNGKey,
    ) -> Tuple[ILTrainingState, envs.State, Metrics]:
        key_rollout, key_next = jax.random.split(key)

        # 1. Roll out the FROZEN teacher policy to collect observations.
        # frozen_teacher_policy is a direct policy(obs, key) function with all
        # teacher constants baked in — no params needed at call time.
        env_state, data = _generate_rollout_data(
            env_state, key_rollout, frozen_teacher_policy, num_unrolls_per_epoch,
        )

        # 2. Alignment: student encoder learns to match frozen teacher decoder.
        # All teacher constants (norm, dec_params, action_head) are pre-bound
        # in align_fn via functools.partial — not in the training state.
        student_enc, align_opt_state, align_loss, embed_loss, action_loss = align_fn(
            training_state.student_enc_params,
            training_state.align_opt_state,
            data.observation,              # teacher_obs (teacher_obs_key extracted inside)
            data.observation,              # student_obs (proprio + pixels extracted inside)
            training_state.normalizer_params,  # live proprio_norm
        )

        # 3. Update proprio running stats from collected batch.
        normalizer_params = training_state.normalizer_params
        if normalize_observations:
            normalizer_params = running_statistics.update(
                normalizer_params, data.observation["proprio_obs"],
            )

        new_state = ILTrainingState(
            align_opt_state=align_opt_state,
            student_enc_params=student_enc,
            normalizer_params=normalizer_params,
            env_steps=training_state.env_steps + jnp.uint32(env_steps_per_epoch),
        )

        metrics = {
            "align_loss": align_loss,
            "embed_loss": embed_loss,
            "action_loss": action_loss,
        }
        return new_state, env_state, metrics

    # ========================= jit / pmap epoch ==============================
    # Always use pmap, even for 1 device.
    # jit(vmap(il_epoch)) would wrap render calls in a JAX vmap context,
    # which combines with the training-wrapper's own jax.vmap(env.step) into
    # a nested vmap.  Warp's render kernels cannot handle nested vmap and
    # produce wrong-rank outputs.  pmap on 1 device is effectively jit with
    # a device-partition axis — no extra vmap transform.
    # donate_argnums lets XLA reuse buffer memory for training_state and
    # env_state across epochs.
    il_epoch_jit = jax.pmap(
        il_epoch,
        axis_name=_PMAP_AXIS_NAME,
        donate_argnums=(0, 1),
    )

    # ========================= eval =========================

    eval_env_resolved = env  
    if wrap_env:
        wrap_fn = wrap_env_fn or envs.training.wrap
        eval_env_resolved = wrap_fn(
            eval_env_resolved,
            episode_length=episode_length,
            action_repeat=action_repeat,
        )

    evaluator = acting.Evaluator(
        eval_env_resolved,
        make_student_policy,   # factory: params=(proprio_norm, student_enc_params)
        num_eval_envs=num_eval_envs,
        episode_length=episode_length,
        action_repeat=action_repeat,
        key=eval_key,
    )

    # ========================= replicate =========================
    # Expand a leading dim of 1 to match the vmap'd training loop
    # (equivalent to device_put_replicated with 1 device).

    training_state = jax.tree_util.tree_map(
        lambda x: jnp.expand_dims(jnp.asarray(x), 0),
        training_state,
    )

    # ========================= main loop =========================

    training_walltime = 0.0
    eval_every = max(num_il_epochs // max(num_evals, 1), 1)

    # initial eval
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

    for epoch in range(num_il_epochs):
        t0 = time.time()

        epoch_key, local_key = jax.random.split(local_key)
        epoch_keys = jax.random.split(epoch_key, local_devices_to_use)

        training_state, env_state = _strip_weak_type(
            (training_state, env_state)
        )
        training_state, env_state, train_metrics = il_epoch_jit(
            training_state, env_state, epoch_keys
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
                and ((epoch + 1) % eval_every == 0 or epoch == num_il_epochs - 1)
            )
            training_metrics_out = {
                "training/align_loss": float(train_metrics["align_loss"]),
                "training/embed_loss": float(train_metrics["embed_loss"]),
                "training/action_loss": float(train_metrics["action_loss"]),
                "training/sps": sps,
                "training/walltime": training_walltime,
            }

            if should_eval:
                metrics = evaluator.run_evaluation(
                    _unpmap(_pack_student_params(training_state)),
                    training_metrics_out,
                )
            else:
                metrics = training_metrics_out

            logging.info(
                "IL epoch %d/%d  env_steps=%d  align_loss=%.4f  sps=%.0f",
                epoch + 1, num_il_epochs, current_step,
                float(train_metrics["align_loss"]), sps,
            )
            progress_fn(current_step, metrics)

    # ========================= done =========================

    params = _unpmap(_pack_student_params(training_state))
    logging.info("IL training complete — total env steps: %d", current_step)
    # Build a self-contained checkpoint: include action_head_params so the
    # saved file does not require the teacher checkpoint at evaluation time.
    # Layout: (proprio_norm, (student_enc_params, action_head_params))
    checkpoint_params = (params[0], (params[1], action_head_params))
    return (make_student_policy, checkpoint_params, metrics)
