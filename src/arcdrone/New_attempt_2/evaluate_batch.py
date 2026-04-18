#!/usr/bin/env python3
"""Batch evaluation: run many episodes, report distance-to-target statistics.

Focuses on the metric that matters most: how close the drone gets to the landing
target.  Runs episodes in parallel batches for speed.

Usage:
    python src/arcdrone/New_attempt_2/evaluate_batch.py \
        --checkpoint outputs/2026-04-14/00-59-34/trained_model.pkl \
        --teacher_checkpoint outputs/2026-04-13/17-10-10/teacher_model.pkl \
        --episodes 100 --batch 32

    # Compare multiple models at once:
    python src/arcdrone/New_attempt_2/evaluate_batch.py \
        --checkpoint model_A.pkl model_B.pkl model_C.pkl \
        --teacher_checkpoint outputs/.../teacher_model.pkl \
        --episodes 100 --batch 32

    # Teacher baseline:
    python ... --policy teacher --teacher_checkpoint ... --episodes 100
"""
from __future__ import annotations

import argparse
import functools
import os
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")

import jax
import jax.numpy as jnp
import numpy as np
from brax.io import model as brax_model
from brax.training.acme import running_statistics, specs
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from task.arcdrone import ARCDroneVisionLandingIL
from training import networks as dagger_networks

OmegaConf.register_new_resolver("mul", lambda a, b: int(a) * int(b), replace=True)
CFG_DIR = _THIS_DIR / "cfg"


def _build_env(batch_envs: int):
    initialize_config_dir(config_dir=str(CFG_DIR), job_name="batch_eval", version_base=None)
    cfg = compose(config_name="config")
    cfg_env = OmegaConf.to_container(cfg.env, resolve=True)
    cfg_train = cfg.train
    cfg_env["vision_config"]["nworld"] = batch_envs
    cfg_env["naconmax"] = int(cfg_env["njmax"]) * batch_envs
    env = ARCDroneVisionLandingIL(cfg=cfg_env)
    return env, cfg_train


def _build_student_inference(env, cfg_train, checkpoint_path: str, batch_envs: int):
    """Build JIT-compiled student inference function."""
    v_reset = jax.jit(jax.vmap(env.reset))
    rng = jax.random.PRNGKey(0)
    state = v_reset(jax.random.split(rng, batch_envs))
    obs_shape = jax.tree_util.tree_map(lambda x: x.shape[1:], state.obs)
    action_size = env._mj_model.nu

    net_factory = functools.partial(
        dagger_networks.make_il_networks,
        preprocess_observations_fn=(
            running_statistics.normalize if cfg_train.normalize_observations else (lambda x, y: x)
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
        policy_proprio_key=cfg_train.policy_proprio_key,
        teacher_obs_key=cfg_train.teacher_obs_key,
        value_obs_key=cfg_train.value_obs_key,
    )
    il_net = net_factory(obs_shape, action_size)

    ckpt = brax_model.load_params(checkpoint_path)
    proprio_norm = ckpt[0]
    student_enc, action_head, vel_estimator = ckpt[1]

    proprio_size = int(obs_shape["proprio_obs"][-1])
    try:
        if tuple(proprio_norm.mean.shape) != (proprio_size,):
            raise AttributeError
    except AttributeError:
        proprio_norm = running_statistics.init_state(
            specs.Array((proprio_size,), jnp.dtype("float32"))
        )

    make_policy = dagger_networks.make_student_inference_fn(il_net)
    inference_fn = make_policy(
        (proprio_norm, student_enc, action_head, vel_estimator), deterministic=True
    )
    return jax.jit(inference_fn), il_net


def _build_teacher_inference(env, cfg_train, teacher_checkpoint: str, batch_envs: int):
    """Build JIT-compiled teacher inference function."""
    v_reset = jax.jit(jax.vmap(env.reset))
    rng = jax.random.PRNGKey(0)
    state = v_reset(jax.random.split(rng, batch_envs))
    obs_shape = jax.tree_util.tree_map(lambda x: x.shape[1:], state.obs)
    action_size = env._mj_model.nu

    net_factory = functools.partial(
        dagger_networks.make_il_networks,
        preprocess_observations_fn=(
            running_statistics.normalize if cfg_train.normalize_observations else (lambda x, y: x)
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
        policy_proprio_key=cfg_train.policy_proprio_key,
        teacher_obs_key=cfg_train.teacher_obs_key,
        value_obs_key=cfg_train.value_obs_key,
    )
    il_net = net_factory(obs_shape, action_size)

    teacher_ckpt = brax_model.load_params(teacher_checkpoint)
    try:
        teacher_norm = dagger_networks._select_normalizer_by_path(
            teacher_ckpt[0], cfg_train.teacher_normalizer_key
        )
    except KeyError:
        teacher_norm = dagger_networks._select_normalizer_by_path(
            teacher_ckpt[0], "policy_obs"
        )
    inference_fn = dagger_networks.make_frozen_teacher_policy(
        il_net,
        teacher_norm_params=teacher_norm,
        teacher_policy_params=teacher_ckpt[1],
        deterministic=True,
    )
    return jax.jit(inference_fn), il_net


def run_eval(env, inference_fn, num_episodes: int, batch_envs: int,
             max_steps: int = 200, seed: int = 42):
    """Run episodes and collect final distance to target for each.

    Returns dict with per-episode results.
    """
    v_reset = jax.jit(jax.vmap(env.reset))
    v_step = jax.jit(jax.vmap(env.step))

    rng = jax.random.PRNGKey(seed)

    # JIT warmup
    rng, key = jax.random.split(rng)
    state = v_reset(jax.random.split(key, batch_envs))
    obs_0 = jax.tree.map(lambda x: x[0], state.obs)
    act_0, _ = inference_fn(obs_0, rng)
    actions = jnp.broadcast_to(act_0, (batch_envs,) + act_0.shape)
    state = v_step(state, actions)

    results = []
    episodes_done = 0
    batch_idx = 0

    while episodes_done < num_episodes:
        batch_idx += 1
        # How many episodes to run in this batch
        remaining = num_episodes - episodes_done
        this_batch = min(batch_envs, remaining)

        rng, reset_key = jax.random.split(rng)
        reset_keys = jax.random.split(reset_key, batch_envs)
        state = v_reset(reset_keys)

        # Track per-env state
        start_pos = np.array(state.data.qpos[:, 0:3])
        target = np.array(state.info["target_buffer"][:, 0, :])
        done_mask = np.zeros(batch_envs, dtype=bool)
        final_pos = np.array(state.data.qpos[:, 0:3])
        final_step = np.full(batch_envs, max_steps, dtype=int)
        ep_reward = np.zeros(batch_envs)

        for step in range(max_steps):
            rng, action_key = jax.random.split(rng)

            # Run inference for ALL envs (batched via vmap)
            # We need per-env obs, per-env inference
            # Since inference_fn expects single-env obs, we loop or vmap
            # The simplest: process each env (inference is fast on GPU)
            all_actions = []
            for e in range(batch_envs):
                obs_e = jax.tree.map(lambda x: x[e], state.obs)
                rng, k = jax.random.split(rng)
                act_e, _ = inference_fn(obs_e, k)
                all_actions.append(act_e)
            actions = jnp.stack(all_actions)

            state = v_step(state, actions)

            # Record final positions for envs that just finished
            current_done = np.array(state.done) > 0.5
            newly_done = current_done & ~done_mask
            for e in range(batch_envs):
                if not done_mask[e]:
                    ep_reward[e] += float(state.reward[e])
                    final_pos[e] = np.array(state.data.qpos[e, 0:3])
                    final_step[e] = step + 1
                if newly_done[e]:
                    done_mask[e] = True

            if done_mask[:this_batch].all():
                break

        # Collect results for this batch
        for e in range(this_batch):
            xy_dist = float(np.linalg.norm(final_pos[e, :2] - target[e, :2]))
            xyz_dist = float(np.linalg.norm(final_pos[e, :3] - target[e, :3]))
            results.append({
                "episode": episodes_done + e + 1,
                "final_xy_dist": xy_dist,
                "final_xyz_dist": xyz_dist,
                "final_z": float(final_pos[e, 2]),
                "start_dist": float(np.linalg.norm(start_pos[e, :2] - target[e, :2])),
                "steps": int(final_step[e]),
                "total_reward": float(ep_reward[e]),
                "landed": float(final_pos[e, 2]) < 0.2 and xy_dist < 0.5,
                "close": xy_dist < 1.0,
            })

        episodes_done += this_batch
        print(f"  batch {batch_idx}: {episodes_done}/{num_episodes} episodes done")

    return results


def print_report(results: list, label: str):
    """Print distance statistics."""
    xy_dists = np.array([r["final_xy_dist"] for r in results])
    z_vals = np.array([r["final_z"] for r in results])
    landed = sum(1 for r in results if r["landed"])
    close = sum(1 for r in results if r["close"])
    n = len(results)

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"  Episodes:        {n}")
    print(f"  Final XY distance to target:")
    print(f"    Mean:          {np.mean(xy_dists):.3f} m")
    print(f"    Median:        {np.median(xy_dists):.3f} m")
    print(f"    Std:           {np.std(xy_dists):.3f} m")
    print(f"    Min:           {np.min(xy_dists):.3f} m")
    print(f"    Max:           {np.max(xy_dists):.3f} m")
    print(f"    P25:           {np.percentile(xy_dists, 25):.3f} m")
    print(f"    P75:           {np.percentile(xy_dists, 75):.3f} m")
    print(f"    P90:           {np.percentile(xy_dists, 90):.3f} m")
    print(f"  Final height (z):")
    print(f"    Mean:          {np.mean(z_vals):.3f} m")
    print(f"    Median:        {np.median(z_vals):.3f} m")
    print(f"  Landed (<0.2m z, <0.5m xy): {landed}/{n} ({100*landed/n:.1f}%)")
    print(f"  Close  (<1.0m xy):          {close}/{n} ({100*close/n:.1f}%)")
    print(f"{'='*70}")

    # Per-episode detail (sorted by distance)
    sorted_results = sorted(results, key=lambda r: r["final_xy_dist"])
    print(f"\n  Per-episode (sorted by distance):")
    print(f"  {'ep':>4s}  {'xy_dist':>8s}  {'z':>6s}  {'start_d':>8s}  {'steps':>5s}  {'reward':>8s}  {'status'}")
    print(f"  {'-'*60}")
    for r in sorted_results:
        status = "LANDED" if r["landed"] else ("CLOSE" if r["close"] else "FAR")
        print(f"  {r['episode']:4d}  {r['final_xy_dist']:8.3f}  {r['final_z']:6.3f}  "
              f"{r['start_dist']:8.3f}  {r['steps']:5d}  {r['total_reward']:8.2f}  {status}")


def main():
    parser = argparse.ArgumentParser(description="Batch distance evaluation")
    parser.add_argument("--checkpoint", nargs="+", default=None,
                        help="One or more student checkpoint paths")
    parser.add_argument("--teacher_checkpoint", type=str, required=True)
    parser.add_argument("--policy", default="student", choices=["student", "teacher"])
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--batch", type=int, default=16,
                        help="Parallel envs per batch (balance speed vs GPU memory)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    batch_envs = args.batch
    env, cfg_train = _build_env(batch_envs)
    print(f"Environment ready. Batch size={batch_envs}, episodes={args.episodes}")

    if args.policy == "teacher":
        print(f"\nEvaluating TEACHER: {args.teacher_checkpoint}")
        inference_fn, _ = _build_teacher_inference(
            env, cfg_train, args.teacher_checkpoint, batch_envs
        )
        results = run_eval(env, inference_fn, args.episodes, batch_envs,
                          args.max_steps, args.seed)
        print_report(results, f"TEACHER: {args.teacher_checkpoint}")
    else:
        if not args.checkpoint:
            raise ValueError("--checkpoint required for student evaluation")
        for ckpt_path in args.checkpoint:
            print(f"\nEvaluating: {ckpt_path}")
            inference_fn, _ = _build_student_inference(
                env, cfg_train, ckpt_path, batch_envs
            )
            results = run_eval(env, inference_fn, args.episodes, batch_envs,
                              args.max_steps, args.seed)
            print_report(results, ckpt_path)


if __name__ == "__main__":
    main()
