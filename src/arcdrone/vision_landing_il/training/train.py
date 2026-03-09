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
    """Minimal state for the IL learner (no PPO optimizer / value fn)."""

    align_opt_state: optax.OptState
    # ILNetworkParams: policy and value are frozen from the teacher checkpoint;
    # student_enc_params and proxy_dec_params are updated by alignment.
    params: il_networks.ILNetworkParams
    # Flat normalizer covering propio (updated during IL) and
    # value_obs / teacher_obs (frozen from teacher checkpoint).
    normalizer_params: Any
    env_steps: Any  # jnp scalar (uint32)


# ====================== helpers ======================

def _unpmap(v):
    return jax.tree_util.tree_map(lambda x: x[0] if x is not None else None, v)


def _strip_weak_type(tree):
    def f(leaf):
        leaf = jnp.asarray(leaf)
        return jnp.astype(leaf, leaf.dtype)
    return jax.tree_util.tree_map(f, tree)


def _pack_inference_params(ts: ILTrainingState) -> InferenceParams:
    """Pack into the 4-tuple expected by make_student_inference_fn.

    Layout:
        [0] normalizer_params
        [1] teacher policy params  (decoder, action_head) — frozen
        [2] value params           — frozen
        [3] student network params (student_enc, action_head)
    """
    # The student network uses (student_enc_params, shared_action_head_params)
    student_net_params = (ts.params.student_enc_params, ts.params.policy[1])
    return (
        ts.normalizer_params,
        ts.params.policy,
        ts.params.value,
        student_net_params,
    )


# ====================== main ======================

def train(
    env: envs.Env,
    teacher_params: Any,
    *,
    # --- schedule ---
    num_il_epochs: int = 200,
    num_evals: int = 10,
    unroll_length: int = 10,
    num_unrolls_per_epoch: int = 50,
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
):
    """Pure IL training loop.

    Args:
        teacher_env: teacher (state-based) environment, already wrapped for
            Brax training if ``wrap_env=False``.
        student_env: student (vision) environment — only used when the student
            decoder is a CNN; ignored when student obs == teacher obs.
        teacher_params: 5-tuple ``(normalizer, policy, value, student_dec,
            proxy_dec)`` loaded from a SITT/RL checkpoint.  Only the teacher
            policy weights are used for rollouts; student/proxy are
            re-initialised.
        num_il_epochs: total number of collect-then-align cycles.
        num_evals: how often to evaluate (spread across ``num_il_epochs``).
        unroll_length: number of env steps per unroll segment.
        num_unrolls_per_epoch: how many unroll segments to collect per epoch.
        align_updates_per_trigger: gradient steps on alignment loss per epoch.
        network_factory: factory that builds the ``SITTNetworks``.
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
        student_observation_size: dict ``{key: shape}`` for student obs init.
        run_evals: if ``False`` skip evaluation rollouts.
        wrap_env: if ``True``, wrap envs with the Brax training wrapper.
        wrap_env_fn: custom wrapping function.

    Returns:
        ``(make_policy, params, metrics)`` where ``make_policy`` is the
        **student** inference function and ``params`` is the 5-tuple.
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

    # env-steps per epoch
    env_steps_per_epoch = num_envs * unroll_length * num_unrolls_per_epoch * action_repeat

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

    key_envs = jax.random.split(key_env, num_envs // process_count)
    key_envs = jnp.reshape(
        key_envs, (local_devices_to_use, -1) + key_envs.shape[1:]
    )

    # Follow the Brax PPO convention: pmap when multiple devices are available,
    # jit+vmap for the single-device (Warp) case.  Both produce a leading
    # device-count dimension that the vmap'd training loop expects.
    if local_devices_to_use > 1:
        reset_fn = jax.pmap(env.reset, axis_name=_PMAP_AXIS_NAME)
    else:
        reset_fn = jax.jit(jax.vmap(env.reset))
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

    # Teacher inference (frozen — drives rollouts)
    make_teacher_policy = il_networks.make_teacher_inference_fn(il_network)

    # Student inference (what we export / evaluate at the end)
    make_student_policy = il_networks.make_student_inference_fn(il_network)

    # ========================= optimizer =========================

    align_optimizer = optax.adam(learning_rate)

    align_fn = functools.partial(
        align,
        il_network=il_network,      # ILNetworks — static, hashable
        optimizer=align_optimizer,
        align_updates_per_trigger=align_updates_per_trigger,
    )

    # ========================= init params =========================


    # ── Decompose teacher checkpoint ──────────────────────────────────────
    # teacher_params layout (from vision_landing_rl PPO): (norm, policy, value)
    # where policy = (teacher_dec_params, action_head_params)
    teacher_norm_params = teacher_params[0]
    teacher_policy_params = teacher_params[1]  # (teacher_dec, action_head)
    teacher_value_params = teacher_params[2]

    # ── Fresh student encoder + proxy decoder ─────────────────────────────
    key_student, key_proxy = jax.random.split(key_il)
    student_enc_params = il_network.student_encoder.init(key_student)
    proxy_dec_params = il_network.proxy_decoder.init(key_proxy)

    init_params = il_networks.ILNetworkParams(
        policy=teacher_policy_params,
        value=teacher_value_params,
        student_enc_params=student_enc_params,
        proxy_dec_params=proxy_dec_params,
    )

    # ── Normalizer ────────────────────────────────────────────────────────
    # The teacher checkpoint was trained on flat privileged state only, so its
    # RunningStatisticsState has array leaves (not dict-structured).  It has NO
    # propio stats — we must supply those fresh.
    #
    # Strategy: build a fresh RS for the full non-pixel obs structure, then
    # transplant the teacher's per-field arrays into value_obs / teacher_obs,
    # leaving propio at its fresh-init values (updated from rollouts).

    propio_spec = specs.Array(
        env_state.obs["propio"].shape[2:], jnp.dtype("float32")
    )
    fresh_propio_norm = running_statistics.init_state(propio_spec)

    # Build a combined RS whose leaves are dicts keyed by obs name.
    # running_statistics.RunningStatisticsState is a flax struct with
    #   .count, .mean, .std, .summed_variance
    # normalizer_select(rs, key) does tree_map(lambda x: x[key], rs) so each leaf
    # must be a dict at the top level.
    init_normalizer_params = running_statistics.RunningStatisticsState(
        count=teacher_norm_params.count,
        mean={
            "value_obs":   teacher_norm_params.mean,
            "teacher_obs": teacher_norm_params.mean,
            "propio":      fresh_propio_norm.mean,
        },
        std={
            "value_obs":   teacher_norm_params.std,
            "teacher_obs": teacher_norm_params.std,
            "propio":      fresh_propio_norm.std,
        },
        summed_variance={
            "value_obs":   teacher_norm_params.summed_variance,
            "teacher_obs": teacher_norm_params.summed_variance,
            "propio":      fresh_propio_norm.summed_variance,
        },
        std_eps=teacher_norm_params.std_eps,
        mode=teacher_norm_params.mode,
    )

    training_state = ILTrainingState(
        align_opt_state=align_optimizer.init(
            (student_enc_params, proxy_dec_params)
        ),
        params=init_params,
        normalizer_params=init_normalizer_params,
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
        # Use params from training_state so this function is self-contained
        # and compatible with jit/vmap tracing (no closure over outer tensors).
        teacher_policy = make_teacher_policy((
            training_state.normalizer_params,
            training_state.params.policy,
            training_state.params.value,
        ))

        env_state, data = _generate_rollout_data(
            env_state, key_rollout, teacher_policy, num_unrolls_per_epoch,
        )

        # 2. Alignment update: student encoder + proxy learn to match teacher decoder
        teacher_dec = training_state.params.policy[0]   # teacher decoder params (frozen)
        action_head = training_state.params.policy[1]   # shared action head (frozen)
        norm_params = training_state.normalizer_params

        # The env returns a flat obs dict:
        #   data.observation["teacher_obs"]   → teacher (privileged) state
        #   data.observation["value_obs"]     → critic state
        #   data.observation["pixels/view_0"] → student vision frames
        #   data.observation["propio"]        → student propio
        # We pass the same dict for both; each network helper extracts its own keys.
        student_obs = data.observation

        (
            (_, student_enc, proxy_dec, _, _),
            align_opt_state,
            align_loss,
        ) = align_fn(
            (
                teacher_dec,
                training_state.params.student_enc_params,
                training_state.params.proxy_dec_params,
                action_head,
                norm_params,
            ),
            training_state.align_opt_state,
            data.observation,   # teacher (privileged) obs → teacher_decoder & proxy
            student_obs,        # vision obs → student_encoder
        )

        # 3. Update running normalizer with non-pixel obs from the collected batch.
        # teacher_norm_params is a single RunningStatisticsState for the dict obs;
        # running_statistics.update expects a batch matching its structure.
        # Pixels are excluded (normalised by env to [-0.5, 0.5]).
        normalizer_params = training_state.normalizer_params
        if normalize_observations:
            non_pixel_obs = {k: v for k, v in data.observation.items()
                             if not k.startswith("pixels/")}
            normalizer_params = running_statistics.update(
                normalizer_params, non_pixel_obs,
            )

        new_params = training_state.params.replace(
            student_enc_params=student_enc,
            proxy_dec_params=proxy_dec,
        )
        new_state = ILTrainingState(
            align_opt_state=align_opt_state,
            params=new_params,
            normalizer_params=normalizer_params,
            env_steps=training_state.env_steps + jnp.uint32(env_steps_per_epoch),
        )

        metrics = {
            "align_loss": align_loss,
        }
        return new_state, env_state, metrics

    # ========================= jit / pmap epoch ==============================
    # Mirror the Brax PPO convention: pmap for multi-device, jit+vmap for
    # single-device (Warp).  donate_argnums lets XLA reuse buffer memory for
    # training_state and env_state across epochs.
    if local_devices_to_use > 1:
        il_epoch_jit = jax.pmap(
            il_epoch,
            axis_name=_PMAP_AXIS_NAME,
            donate_argnums=(0, 1),
        )
    else:
        il_epoch_jit = jax.jit(
            jax.vmap(il_epoch),
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
        functools.partial(
            make_teacher_policy, deterministic=deterministic_eval
        ),
        num_eval_envs=num_envs,
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
            _unpmap(_pack_inference_params(training_state)),
            training_metrics={},
        )
        logging.info(metrics)
        progress_fn(0, metrics)

    params = _unpmap(_pack_inference_params(training_state))
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
            params = _unpmap(_pack_inference_params(training_state))
            policy_params_fn(current_step, make_student_policy, params)

            should_eval = (
                run_evals
                and ((epoch + 1) % eval_every == 0 or epoch == num_il_epochs - 1)
            )
            training_metrics_out = {
                "training/align_loss": float(train_metrics["align_loss"]),
                "training/sps": sps,
                "training/walltime": training_walltime,
            }

            if should_eval:
                metrics = evaluator.run_evaluation(
                    params, training_metrics_out,
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

    params = _unpmap(_pack_inference_params(training_state))
    logging.info("IL training complete — total env steps: %d", current_step)
    return (make_student_policy, params, metrics)
