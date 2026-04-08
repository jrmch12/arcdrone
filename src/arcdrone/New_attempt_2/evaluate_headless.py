"""Headless evaluator for ARCDrone DAgger policies.

Runs batched rollouts without launching the MuJoCo viewer and reports policy
quality metrics that are useful for iteration:
  - success rate
  - ground-violation/crash proxy rate
  - timeout rate
  - reward and episode-length statistics
  - final/min XY distance to the landing target
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
from pathlib import Path
from typing import Any

# Must be set before importing JAX runtime users.
_xla_flags = os.environ.get("XLA_FLAGS", "")
if "--xla_gpu_triton_gemm_any=True" not in _xla_flags:
    _xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = _xla_flags
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")

import jax
import jax.numpy as jnp
import numpy as np
from brax.io import model
from brax.training.acme import running_statistics
from brax.training.acme import specs
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import sys as _sys
from pathlib import Path as _Path
_NA2_ROOT = str(_Path(__file__).resolve().parent)
if _NA2_ROOT not in _sys.path:
    _sys.path.insert(0, _NA2_ROOT)
from task.arcdrone import ARCDroneVisionLandingIL
from training import networks as dagger_networks

OmegaConf.register_new_resolver("mul", lambda a, b: int(a) * int(b), replace=True)

CFG_DIR = Path(__file__).resolve().parent / "cfg"
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _find_latest_checkpoint(outputs_dir: Path) -> str:
    """Prefers latest best_model.pkl; falls back to trained_model.pkl."""
    candidates: list[Path] = []
    candidates.extend(outputs_dir.rglob("best_model.pkl"))
    candidates.extend(outputs_dir.rglob("trained_model.pkl"))
    if not candidates:
        raise FileNotFoundError(f"No model checkpoint found under: {outputs_dir}")
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return str(latest)


def _build_network_factory(cfg_train):
    return functools.partial(
        dagger_networks.make_il_networks,
        preprocess_observations_fn=(
            running_statistics.normalize
            if cfg_train.normalize_observations
            else (lambda x, y: x)
        ),
        teacher_dec_hidden_layers=cfg_train.teacher_dec_hidden_layers,
        policy_dec_hidden_layers=cfg_train.policy_dec_hidden_layers,
        policy_proprio_proj_hidden_layers=cfg_train.policy_proprio_proj_hidden_layers,
        action_hidden_layer_sizes=cfg_train.action_hidden_layers,
        value_hidden_layer_sizes=cfg_train.value_hidden_layers,
        cnn_num_filters=cfg_train.cnn_num_filters,
        cnn_kernel_sizes=cfg_train.cnn_kernel_sizes,
        cnn_strides=cfg_train.cnn_strides,
        policy_pixels_key=cfg_train.policy_pixels_key,
        policy_pixels_key_1=cfg_train.policy_pixels_key_1,
        policy_pixels_key_2=cfg_train.policy_pixels_key_2,
        policy_proprio_key=cfg_train.policy_proprio_key,
        teacher_obs_key=cfg_train.teacher_obs_key,
        value_obs_key=cfg_train.value_obs_key,
    )


def _init_env_and_network(cfg, batch_envs: int):
    cfg_env = OmegaConf.to_container(cfg.env, resolve=True)
    cfg_env["vision_config"]["nworld"] = int(batch_envs)
    cfg_env["naconmax"] = int(cfg_env["njmax"]) * int(batch_envs)

    env = ARCDroneVisionLandingIL(cfg=cfg_env)
    reset_fn = jax.jit(jax.vmap(env.reset))
    init_state = reset_fn(jax.random.split(jax.random.PRNGKey(123), batch_envs))
    obs_shape = jax.tree_util.tree_map(lambda x: x.shape[1:], init_state.obs)
    action_size = env._mj_model.nu
    il_network = _build_network_factory(cfg.train)(obs_shape, action_size)
    return env, il_network, init_state


def _coerce_proprio_norm(norm, obs_shape):
    """Normaliser compatibility guard for older checkpoints."""
    proprio_size = int(obs_shape["proprio_obs"][-1])
    target_shape = (proprio_size,)
    try:
        if tuple(norm.mean.shape) == target_shape:
            return norm
    except AttributeError:
        pass
    return running_statistics.init_state(
        specs.Array(target_shape, jnp.dtype("float32"))
    )


def _make_student_policy(
    il_network,
    cfg_train,
    obs_shape,
    model_path: str,
    deterministic: bool,
    teacher_ckpt: Any | None = None,
):
    ckpt = model.load_params(model_path)
    proprio_norm = _coerce_proprio_norm(ckpt[0], obs_shape)
    restored_policy = ckpt[1]

    if isinstance(restored_policy, (tuple, list)) and len(restored_policy) == 2:
        student_enc, action_head = restored_policy
    else:
        if teacher_ckpt is None:
            raise ValueError(
                "Student checkpoint appears to be old format (missing action head). "
                "Provide a teacher checkpoint for fallback via --teacher_checkpoint_path."
            )
        student_enc = restored_policy
        action_head = teacher_ckpt[1][1]

    make_student = dagger_networks.make_student_inference_fn(il_network)
    return make_student((proprio_norm, student_enc, action_head), deterministic=deterministic)


def _make_teacher_policy(il_network, cfg_train, teacher_ckpt_path: str, deterministic: bool):
    teacher_ckpt = model.load_params(teacher_ckpt_path)
    try:
        teacher_norm = dagger_networks._select_normalizer_by_path(
            teacher_ckpt[0], cfg_train.teacher_normalizer_key
        )
    except KeyError:
        teacher_norm = dagger_networks._select_normalizer_by_path(
            teacher_ckpt[0], "policy_obs"
        )
    policy = dagger_networks.make_frozen_teacher_policy(
        il_network,
        teacher_norm_params=teacher_norm,
        teacher_policy_params=teacher_ckpt[1],
        deterministic=deterministic,
    )
    return policy, teacher_ckpt


def _rollout_batch(
    env,
    policy_fn,
    *,
    batch_envs: int,
    episode_length: int,
    success_steps_required: int,
    landing_radius: float,
    seed: int,
):
    v_reset = jax.vmap(env.reset)
    v_step = jax.vmap(env.step)
    jit_reset = jax.jit(v_reset)
    jit_step = jax.jit(v_step)
    jit_policy = jax.jit(policy_fn)

    rng = jax.random.PRNGKey(seed)
    rng, key_reset = jax.random.split(rng)
    state = jit_reset(jax.random.split(key_reset, batch_envs))

    done = jnp.zeros((batch_envs,), dtype=bool)
    success = jnp.zeros((batch_envs,), dtype=bool)
    ground_event = jnp.zeros((batch_envs,), dtype=bool)
    reached_landing_zone = jnp.zeros((batch_envs,), dtype=bool)

    episode_return = jnp.zeros((batch_envs,), dtype=jnp.float32)
    episode_len = jnp.zeros((batch_envs,), dtype=jnp.int32)
    min_xy = jnp.full((batch_envs,), jnp.float32(jnp.inf))
    final_xy = jnp.zeros((batch_envs,), dtype=jnp.float32)
    final_z = jnp.zeros((batch_envs,), dtype=jnp.float32)

    for _ in range(episode_length):
        pos = state.info["pos_buffer"][:, 0, :]
        xy = jnp.linalg.norm(pos[:, :2], axis=-1)
        min_xy = jnp.minimum(min_xy, xy)
        reached_landing_zone = reached_landing_zone | (xy <= landing_radius)

        active = ~done
        episode_len = episode_len + active.astype(jnp.int32)

        rng, key_action = jax.random.split(rng)
        action, _ = jit_policy(state.obs, key_action)
        state = jit_step(state, action)

        episode_return = episode_return + state.reward * active.astype(jnp.float32)

        done_now = state.done > 0.5
        newly_done = active & done_now
        done = done | done_now

        success_now = state.info["steps_within_success"] >= success_steps_required
        success = success | success_now

        ground_now = state.info["ground_violation"] > 1e-6
        ground_event = ground_event | (ground_now & active)

        pos_next = state.info["pos_buffer"][:, 0, :]
        xy_next = jnp.linalg.norm(pos_next[:, :2], axis=-1)
        z_next = state.data.qpos[:, 2]
        final_xy = jnp.where(newly_done, xy_next, final_xy)
        final_z = jnp.where(newly_done, z_next, final_z)

    pos_last = state.info["pos_buffer"][:, 0, :]
    xy_last = jnp.linalg.norm(pos_last[:, :2], axis=-1)
    z_last = state.data.qpos[:, 2]
    final_xy = jnp.where(done, final_xy, xy_last)
    final_z = jnp.where(done, final_z, z_last)

    timeout = (~success) & (~ground_event)
    return {
        "episode_return": np.asarray(episode_return),
        "episode_len": np.asarray(episode_len),
        "success": np.asarray(success),
        "ground_event": np.asarray(ground_event),
        "timeout": np.asarray(timeout),
        "reached_landing_zone": np.asarray(reached_landing_zone),
        "min_xy": np.asarray(min_xy),
        "final_xy": np.asarray(final_xy),
        "final_z": np.asarray(final_z),
    }


def _summarise(stacked: dict[str, np.ndarray], num_episodes: int):
    returns = stacked["episode_return"]
    lens = stacked["episode_len"].astype(np.float32)
    success = stacked["success"].astype(np.float32)
    ground = stacked["ground_event"].astype(np.float32)
    timeout = stacked["timeout"].astype(np.float32)
    reach = stacked["reached_landing_zone"].astype(np.float32)
    min_xy = stacked["min_xy"]
    final_xy = stacked["final_xy"]
    final_z = stacked["final_z"]

    return {
        "num_episodes": int(num_episodes),
        "episode_reward_mean": float(np.mean(returns)),
        "episode_reward_std": float(np.std(returns)),
        "episode_len_mean": float(np.mean(lens)),
        "episode_len_std": float(np.std(lens)),
        "success_rate": float(np.mean(success)),
        "ground_event_rate": float(np.mean(ground)),
        "timeout_rate": float(np.mean(timeout)),
        "reached_landing_zone_rate": float(np.mean(reach)),
        "min_xy_mean": float(np.mean(min_xy)),
        "min_xy_p50": float(np.percentile(min_xy, 50)),
        "min_xy_p90": float(np.percentile(min_xy, 90)),
        "final_xy_mean": float(np.mean(final_xy)),
        "final_xy_p50": float(np.percentile(final_xy, 50)),
        "final_xy_p90": float(np.percentile(final_xy, 90)),
        "final_z_mean": float(np.mean(final_z)),
        "score_success_minus_ground": float(np.mean(success) - np.mean(ground)),
    }


def evaluate(
    *,
    policy: str,
    model_path: str | None,
    teacher_checkpoint_path: str | None,
    num_episodes: int,
    batch_envs: int,
    steps: int | None,
    seed: int,
    deterministic: bool,
):
    initialize_config_dir(
        config_dir=str(CFG_DIR), job_name="dagger_evaluate_headless", version_base=None
    )
    cfg = compose(config_name="config")
    cfg_train = cfg.train

    episode_length = int(steps or cfg_train.episode_length)
    success_steps_required = int(cfg.env.success_steps_required)
    landing_radius = float(getattr(cfg.env, "landing_radius", 0.4))

    env, il_network, init_state = _init_env_and_network(cfg, batch_envs)
    obs_shape = jax.tree_util.tree_map(lambda x: x.shape[1:], init_state.obs)

    policy = policy.lower()
    if policy not in {"student", "teacher"}:
        raise ValueError("policy must be 'student' or 'teacher'")

    teacher_ckpt = None
    if teacher_checkpoint_path:
        teacher_ckpt = model.load_params(teacher_checkpoint_path)

    if policy == "teacher":
        ckpt_path = teacher_checkpoint_path or str(cfg_train.teacher_checkpoint_path or "")
        if not ckpt_path:
            raise ValueError(
                "Teacher evaluation requires --teacher_checkpoint_path or "
                "train.teacher_checkpoint_path in config."
            )
        policy_fn, _ = _make_teacher_policy(
            il_network, cfg_train, ckpt_path, deterministic=deterministic
        )
        selected_checkpoint = ckpt_path
    else:
        if model_path is None:
            model_path = _find_latest_checkpoint(PROJECT_ROOT / "outputs")
        if teacher_ckpt is None and teacher_checkpoint_path:
            teacher_ckpt = model.load_params(teacher_checkpoint_path)
        policy_fn = _make_student_policy(
            il_network,
            cfg_train,
            obs_shape,
            model_path,
            deterministic=deterministic,
            teacher_ckpt=teacher_ckpt,
        )
        selected_checkpoint = model_path

    num_batches = int(math.ceil(num_episodes / batch_envs))
    buffer: dict[str, list[np.ndarray]] = {}
    written = 0

    for batch_idx in range(num_batches):
        batch = _rollout_batch(
            env,
            policy_fn,
            batch_envs=batch_envs,
            episode_length=episode_length,
            success_steps_required=success_steps_required,
            landing_radius=landing_radius,
            seed=seed + batch_idx,
        )
        keep = min(batch_envs, num_episodes - written)
        for key, value in batch.items():
            buffer.setdefault(key, []).append(value[:keep])
        written += keep

    stacked = {k: np.concatenate(v, axis=0) for k, v in buffer.items()}
    summary = _summarise(stacked, num_episodes)
    summary.update(
        {
            "policy": policy,
            "checkpoint": str(selected_checkpoint),
            "episode_length": int(episode_length),
            "batch_envs": int(batch_envs),
            "deterministic": bool(deterministic),
            "seed": int(seed),
        }
    )
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Headless evaluator for ARCDrone New_attempt_2 DAgger policies."
    )
    parser.add_argument(
        "--policy",
        choices=["student", "teacher"],
        default="student",
        help="Which policy to evaluate.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to student checkpoint. Defaults to latest best_model/trained_model.",
    )
    parser.add_argument(
        "--teacher_checkpoint_path",
        type=str,
        default=None,
        help="Path to teacher checkpoint (required for --policy teacher).",
    )
    parser.add_argument("--episodes", type=int, default=256, help="Number of episodes.")
    parser.add_argument(
        "--batch_envs",
        type=int,
        default=64,
        help="Number of parallel environments per rollout batch.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override max episode steps; defaults to config train.episode_length.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Evaluation RNG seed.")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions from policy instead of deterministic mode.",
    )
    parser.add_argument(
        "--json_out",
        type=str,
        default=None,
        help="Optional path to save summary JSON.",
    )
    args = parser.parse_args()

    summary = evaluate(
        policy=args.policy,
        model_path=args.model_path,
        teacher_checkpoint_path=args.teacher_checkpoint_path,
        num_episodes=int(args.episodes),
        batch_envs=int(args.batch_envs),
        steps=args.steps,
        seed=int(args.seed),
        deterministic=not args.stochastic,
    )

    print("=" * 64)
    print("ARCDrone New_attempt_2 Headless Evaluation")
    print("=" * 64)
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Saved JSON summary to: {out_path}")


if __name__ == "__main__":
    main()
