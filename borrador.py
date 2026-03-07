"""Pure Imitation Learning training.

Given a pre-trained RL teacher checkpoint, trains a student network to
reproduce the teacher's behaviour using the SITT alignment loss.

No PPO, no value function, no GAE — just supervised feature/action matching.

Usage from the hydra entry-point::

    arcdrone-train-il task=landing

Or programmatically::

    from arcdrone.controller.il.il.train import train
    make_policy, params, metrics = train(
        teacher_env=env,
        student_env=student_env,
        teacher_params=teacher_params,
        ...
    )
"""

import functools
import time
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

from absl import logging
from brax import envs
from brax.training import acting
from brax.training import pmap
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.acme import specs
from brax.training.types import PRNGKey

# Reuse SITT building blocks — no duplication
from arcdrone.controller.sitt.sitt import networks as sitt_networks
from arcdrone.controller.sitt.sitt import losses as sitt_losses
from arcdrone.controller.sitt.sitt.align import align

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
    # Full PPONetworkParams so we can reuse align() as-is.
    # Only student_dec_params and proxy_dec_params are updated;
    # policy and value are frozen from the teacher checkpoint.
    params: sitt_losses.PPONetworkParams
    normalizer_params: running_statistics.RunningStatisticsState
    env_steps: jnp.uint32


# ====================== helpers ======================

def _unpmap(v):
    return jax.tree_util.tree_map(lambda x: x[0] if x is not None else None, v)


def _strip_weak_type(tree):
    def f(leaf):
        leaf = jnp.asarray(leaf)
        return jnp.astype(leaf, leaf.dtype)
    return jax.tree_util.tree_map(f, tree)


def _pack_inference_params(ts: ILTrainingState) -> InferenceParams:
    """Pack into the 5-tuple expected by make_student_inference_fn."""
    return (
        ts.normalizer_params,
        ts.params.policy,
        ts.params.value,
        ts.params.student_dec_params,
        ts.params.proxy_dec_params,
    )


# ====================== main ======================

def train(
    teacher_env: envs.Env,
    student_env: envs.Env,
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
        sitt_networks.SITTNetworks
    ] = sitt_networks.make_sitt_networks,
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
    # --- student observation size ---
    student_observation_size: Optional[Mapping[str, Tuple]] = None,
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

    env = teacher_env  # alias — rollouts happen in the teacher env

    # env-steps per epoch
    env_steps_per_epoch = num_envs * unroll_length * num_unrolls_per_epoch * action_repeat

    # ========================= keys =========================

    key = jax.random.PRNGKey(seed)
    global_key, local_key = jax.random.split(key)
    del key
    local_key = jax.random.fold_in(local_key, process_id)
    local_key, key_env, eval_key = jax.random.split(local_key, 3)
    key_policy, key_sitt = jax.random.split(global_key, 2)
    del global_key

    # ========================= reset envs =========================

    assert num_envs % device_count == 0

    key_envs = jax.random.split(key_env, num_envs // process_count)
    key_envs = jnp.reshape(
        key_envs, (local_devices_to_use, -1) + key_envs.shape[1:]
    )

    reset_fn = jax.pmap(env.reset, axis_name=_PMAP_AXIS_NAME)
    env_state = reset_fn(key_envs)c

    # ========================= observation shapes =========================

    obs_shape = jax.tree_util.tree_map(lambda x: x.shape[2:], env_state.obs)

    normalize = lambda x, y: x
    if normalize_observations:
        normalize = running_statistics.normalize

    # ========================= networks =========================

    sitt_network = network_factory(
        obs_shape,
        env.action_size,
        preprocess_observations_fn=normalize,
    )

    # Teacher inference (frozen)
    make_teacher_policy = sitt_networks.make_inference_fn(sitt_network)

    # Student inference (what we train)
    make_student_policy = sitt_networks.make_student_inference_fn(sitt_network)

    # ========================= optimizer =========================

    align_optimizer = optax.adam(learning_rate)

    align_fn = functools.partial(
        align,
        sitt_network=sitt_network,
        optimizer=align_optimizer,
        align_updates_per_trigger=align_updates_per_trigger,
    )

    # ========================= init params =========================

    # Teacher weights from checkpoint (frozen throughout training)
    teacher_normalizer = teacher_params[0]
    teacher_policy = teacher_params[1]
    teacher_value = teacher_params[2]

    # Fresh student + proxy decoder params
    key_student, key_proxy = jax.random.split(key_sitt)
    student_dec_params = sitt_network.student_decoder.init(key_student)
    proxy_dec_params = sitt_network.proxy_decoder.init(key_proxy)

    init_params = sitt_losses.PPONetworkParams(
        policy=teacher_policy,     # frozen
        value=teacher_value,       # frozen (unused, kept for compat)
        student_dec_params=student_dec_params,
        proxy_dec_params=proxy_dec_params,
    )

    obs_shape_spec = jax.tree_util.tree_map(
        lambda x: specs.Array(x.shape[-1:], jnp.dtype("float32")),
        env_state.obs,
    )

    training_state = ILTrainingState(
        align_opt_state=align_optimizer.init(
            (student_dec_params, proxy_dec_params)
        ),
        params=init_params,
        normalizer_params=running_statistics.init_state(
            {k: v for k, v in obs_shape_spec.items()
             if not k.startswith("pixels/")},
            std_eps=normalize_observations_std_eps,
            mode=normalize_observations_mode,
        ) if normalize_observations else teacher_normalizer,
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
                extra_fields=("truncation", "episode_metrics", "episode_done",
                              "student_obs"),
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

        # 1. Roll out the FROZEN teacher policy to collect observations
        teacher_policy = make_teacher_policy(
            (
                training_state.normalizer_params,
                training_state.params.policy,
                training_state.params.value,
            )
        )
        env_state, data = _generate_rollout_data(
            env_state, key_rollout, teacher_policy, num_unrolls_per_epoch,
        )

        # 2. Alignment update: student & proxy decoders learn to match teacher
        policy_dec = training_state.params.policy[0]   # teacher decoder params
        action_head = training_state.params.policy[1]   # shared action head
        norm_params = training_state.normalizer_params

        # Student (vision) obs collected by StudentWrapper during rollouts
        student_obs = data.extras['state_extras']['student_obs']

        (
            (_, student_dec, proxy_dec, _, _),
            align_opt_state,
            align_loss,
        ) = align_fn(
            (
                policy_dec,
                training_state.params.student_dec_params,
                training_state.params.proxy_dec_params,
                action_head,
                norm_params,
            ),
            training_state.align_opt_state,
            data.observation,      # teacher obs for policy_decoder & proxy
            student_obs,           # student (vision) obs for student_decoder
        )

        # 3. Update normalizer (if enabled)
        normalizer_params = training_state.normalizer_params
        if normalize_observations:
            normalizer_params = running_statistics.update(
                normalizer_params,
                {k: v for k, v in data.observation.items()
                 if not k.startswith("pixels/")}
                if isinstance(data.observation, Mapping)
                else data.observation,
                pmap_axis_name=_PMAP_AXIS_NAME,
            )

        new_params = training_state.params.replace(
            student_dec_params=student_dec,
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

    # ========================= pmap =========================

    il_epoch_pmap = jax.pmap(
        il_epoch,
        axis_name=_PMAP_AXIS_NAME,
        donate_argnums=(0, 1),
    )

    # ========================= eval =========================

    eval_env_resolved = teacher_env  # eval in the teacher env
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
        num_eval_envs=num_eval_envs,
        episode_length=episode_length,
        action_repeat=action_repeat,
        key=eval_key,
    )

    # ========================= replicate =========================

    training_state = jax.device_put_replicated(
        training_state, jax.local_devices()[:local_devices_to_use]
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
        training_state, env_state, train_metrics = il_epoch_pmap(
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

    pmap.assert_is_replicated(training_state)
    params = _unpmap(_pack_inference_params(training_state))
    logging.info("IL training complete — total env steps: %d", current_step)
    pmap.synchronize_hosts()
    return (make_student_policy, params, metrics)
