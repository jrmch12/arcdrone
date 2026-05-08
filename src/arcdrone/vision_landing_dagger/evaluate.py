#!/usr/bin/env python3
"""Unified evaluator for ARCDrone DAgger policies.

Three evaluation modes selectable via ``--mode``:

  gui         Launch the MuJoCo viewer and watch the drone fly in real time.
              Reports per-episode reward breakdown and real-time rate.

  batch       Headless batched rollouts. Reports distance-to-target statistics
              (mean/median/P90), success/crash/timeout rates, and optionally
              exports results as JSON. Supports multi-checkpoint comparison.

  diagnostic  Step-by-step state dump (position, velocity, orientation, reward
              components, camera-visibility analysis) for a few episodes.

Usage examples:

    # Watch the student in the GUI viewer
    python evaluate.py --mode gui --checkpoint best_model.pkl

    # Headless benchmark with 256 episodes, export JSON
    python evaluate.py --mode batch --checkpoint best_model.pkl \\
        --episodes 256 --batch_envs 64 --json_out results.json

    # Step-by-step debug for 3 episodes
    python evaluate.py --mode diagnostic --checkpoint best_model.pkl --episodes 3

    # Evaluate teacher baseline
    python evaluate.py --mode batch --policy teacher \\
        --teacher_checkpoint teacher_model.pkl --episodes 100
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np
from mujoco import mjx

from eval_utils import (
    MUJOCO_XML,
    build_env,
    build_network,
    compute_camera_visibility,
    load_config,
    resolve_policy,
)


# ═══════════════════════════════════════════════════════════════════════════════
# GUI mode
# ═══════════════════════════════════════════════════════════════════════════════

def _run_gui(args):
    cfg = load_config()
    env, cfg_env = build_env(cfg, batch_envs=1)

    jit_reset = jax.jit(jax.vmap(env.reset))
    jit_step = jax.jit(jax.vmap(env.step))

    rng = jax.random.PRNGKey(args.seed)
    rng, key = jax.random.split(rng)
    state = jit_reset(jax.random.split(key, 1))

    obs_shape = jax.tree_util.tree_map(lambda x: x.shape[1:], state.obs)
    action_size = env._mj_model.nu
    il_net = build_network(cfg.train, obs_shape, action_size)

    inference_fn, ckpt_path = resolve_policy(
        il_net, cfg.train, obs_shape,
        policy=args.policy,
        checkpoint=args.checkpoint,
        teacher_checkpoint=args.teacher_checkpoint,
        deterministic=True,
    )

    def _squeeze(tree):
        return jax.tree_util.tree_map(lambda x: x[0], tree)

    # JIT warmup
    act, _ = inference_fn(_squeeze(state.obs), rng)
    state = jit_step(state, act[None])

    physics_model = mujoco.MjModel.from_xml_path(MUJOCO_XML)
    physics_data = mujoco.MjData(physics_model)
    viewer = mujoco.viewer.launch_passive(physics_model, physics_data)

    sim_dt_per_step = float(cfg.env["ctrl_dt"]) * int(cfg.train["action_repeat"])

    print("=" * 60)
    print(f"ARCDrone Evaluation — GUI mode")
    print(f"Policy: {args.policy}  |  Checkpoint: {ckpt_path}")
    print("=" * 60)

    for episode in range(args.episodes):
        print(f"\nEpisode {episode + 1}/{args.episodes}")
        print("-" * 60)
        rng, key = jax.random.split(rng)
        state = jit_reset(jax.random.split(key, 1))
        mjx.get_data_into([physics_data], physics_model, state.data)

        ep_reward = 0.0
        ep_steps = 0
        ep_metrics: dict[str, float] = {}
        wall_start = time.perf_counter()

        for _ in range(args.steps):
            rng, key = jax.random.split(rng)
            act, _ = inference_fn(_squeeze(state.obs), key)
            state = jit_step(state, act[None])
            mjx.get_data_into([physics_data], physics_model, state.data)
            viewer.sync()

            ep_reward += float(_squeeze(state.reward))
            ep_steps += 1

            for k, v in state.metrics.items():
                leaf = jax.tree_util.tree_leaves(v)[0]
                val = float(leaf[0]) if leaf.ndim > 0 else float(leaf)
                ep_metrics[k] = ep_metrics.get(k, 0.0) + val

            if bool(_squeeze(state.done)):
                break

        wall_elapsed = time.perf_counter() - wall_start
        sim_time = ep_steps * sim_dt_per_step
        rt_rate = sim_time / wall_elapsed if wall_elapsed > 0 else float("inf")

        print(
            f"  steps={ep_steps}  reward={ep_reward:.3f}  "
            f"sim={sim_time:.2f}s  wall={wall_elapsed:.2f}s  "
            f"realtime={rt_rate:.2f}x"
        )
        for k in sorted(k for k in ep_metrics if k.startswith("reward_")):
            print(f"    {k:<35s} {ep_metrics[k]:+.3f}")

    viewer.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Batch mode
# ═══════════════════════════════════════════════════════════════════════════════

def _rollout_batch(env, policy_fn, *, batch_envs, episode_length,
                   success_steps_required, landing_radius, seed):
    """Run one batch of parallel episodes and return per-env result arrays."""
    jit_reset = jax.jit(jax.vmap(env.reset))
    jit_step = jax.jit(jax.vmap(env.step))
    jit_policy = policy_fn  # already JIT-compiled by resolve_policy

    rng = jax.random.PRNGKey(seed)
    rng, key = jax.random.split(rng)
    state = jit_reset(jax.random.split(key, batch_envs))

    done = jnp.zeros(batch_envs, dtype=bool)
    success = jnp.zeros(batch_envs, dtype=bool)
    ground_event = jnp.zeros(batch_envs, dtype=bool)
    reached_zone = jnp.zeros(batch_envs, dtype=bool)

    ep_return = jnp.zeros(batch_envs, dtype=jnp.float32)
    ep_len = jnp.zeros(batch_envs, dtype=jnp.int32)
    min_xy = jnp.full(batch_envs, jnp.float32(jnp.inf))
    final_xy = jnp.zeros(batch_envs, dtype=jnp.float32)
    final_z = jnp.zeros(batch_envs, dtype=jnp.float32)
    start_xy = jnp.linalg.norm(state.data.qpos[:, :2], axis=-1)

    for _ in range(episode_length):
        pos = state.info["pos_buffer"][:, 0, :]
        xy = jnp.linalg.norm(pos[:, :2], axis=-1)
        min_xy = jnp.minimum(min_xy, xy)
        reached_zone = reached_zone | (xy <= landing_radius)

        active = ~done
        ep_len = ep_len + active.astype(jnp.int32)

        rng, key = jax.random.split(rng)
        action, _ = jit_policy(state.obs, key)
        state = jit_step(state, action)

        ep_return = ep_return + state.reward * active.astype(jnp.float32)

        done_now = state.done > 0.5
        newly_done = active & done_now
        done = done | done_now

        success = success | (state.info["steps_within_success"] >= success_steps_required)
        ground_event = ground_event | ((state.info["ground_violation"] > 1e-6) & active)

        pos_next = state.info["pos_buffer"][:, 0, :]
        final_xy = jnp.where(newly_done, jnp.linalg.norm(pos_next[:, :2], axis=-1), final_xy)
        final_z = jnp.where(newly_done, state.data.qpos[:, 2], final_z)

    # Fill results for episodes that didn't terminate
    pos_last = state.info["pos_buffer"][:, 0, :]
    xy_last = jnp.linalg.norm(pos_last[:, :2], axis=-1)
    final_xy = jnp.where(done, final_xy, xy_last)
    final_z = jnp.where(done, final_z, state.data.qpos[:, 2])

    return {
        "episode_return": np.asarray(ep_return),
        "episode_len": np.asarray(ep_len),
        "success": np.asarray(success),
        "ground_event": np.asarray(ground_event),
        "timeout": np.asarray((~success) & (~ground_event)),
        "reached_landing_zone": np.asarray(reached_zone),
        "min_xy": np.asarray(min_xy),
        "final_xy": np.asarray(final_xy),
        "final_z": np.asarray(final_z),
        "start_xy": np.asarray(start_xy),
    }


def _summarise(stacked, n):
    """Compute aggregate statistics from stacked per-episode arrays."""
    s = lambda k: stacked[k].astype(np.float32)
    xy = stacked["final_xy"]
    return {
        "num_episodes": int(n),
        "episode_reward_mean": float(np.mean(s("episode_return"))),
        "episode_reward_std": float(np.std(s("episode_return"))),
        "episode_len_mean": float(np.mean(s("episode_len"))),
        "success_rate": float(np.mean(s("success"))),
        "ground_event_rate": float(np.mean(s("ground_event"))),
        "timeout_rate": float(np.mean(s("timeout"))),
        "reached_landing_zone_rate": float(np.mean(s("reached_landing_zone"))),
        "final_xy_mean": float(np.mean(xy)),
        "final_xy_median": float(np.median(xy)),
        "final_xy_p90": float(np.percentile(xy, 90)),
        "final_xy_std": float(np.std(xy)),
        "final_xy_min": float(np.min(xy)),
        "final_xy_max": float(np.max(xy)),
        "min_xy_mean": float(np.mean(stacked["min_xy"])),
        "final_z_mean": float(np.mean(stacked["final_z"])),
        "landed_rate": float(np.mean(
            (stacked["final_z"] < 0.2) & (xy < 0.5)
        )),
        "close_rate": float(np.mean(xy < 1.0)),
    }


def _print_batch_report(summary, label):
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")
    print(f"  Episodes:           {summary['num_episodes']}")
    print(f"  Final XY distance to target:")
    print(f"    Mean:             {summary['final_xy_mean']:.3f} m")
    print(f"    Median:           {summary['final_xy_median']:.3f} m")
    print(f"    Std:              {summary['final_xy_std']:.3f} m")
    print(f"    Min / Max:        {summary['final_xy_min']:.3f} / {summary['final_xy_max']:.3f} m")
    print(f"    P90:              {summary['final_xy_p90']:.3f} m")
    print(f"  Rates:")
    print(f"    Success:          {summary['success_rate']*100:.1f}%")
    print(f"    Landed:           {summary['landed_rate']*100:.1f}%")
    print(f"    Close (<1m):      {summary['close_rate']*100:.1f}%")
    print(f"    Ground crash:     {summary['ground_event_rate']*100:.1f}%")
    print(f"    Timeout:          {summary['timeout_rate']*100:.1f}%")
    print(f"  Reward:             {summary['episode_reward_mean']:.2f} +/- {summary['episode_reward_std']:.2f}")
    print(f"  Avg episode len:    {summary['episode_len_mean']:.0f}")
    print(f"{'=' * 70}")


def _run_batch(args):
    cfg = load_config()
    batch_envs = args.batch_envs
    env, cfg_env = build_env(cfg, batch_envs)

    jit_reset = jax.jit(jax.vmap(env.reset))
    rng = jax.random.PRNGKey(args.seed)
    state = jit_reset(jax.random.split(rng, batch_envs))
    obs_shape = jax.tree_util.tree_map(lambda x: x.shape[1:], state.obs)
    action_size = env._mj_model.nu
    il_net = build_network(cfg.train, obs_shape, action_size)

    checkpoints = args.checkpoint if isinstance(args.checkpoint, list) else [args.checkpoint]
    if args.policy == "teacher":
        checkpoints = [args.teacher_checkpoint]

    episode_length = args.steps or int(cfg.train.episode_length)
    success_steps = int(cfg.env.success_steps_required)
    landing_radius = float(getattr(cfg.env, "landing_radius", 0.4))

    for ckpt_path in checkpoints:
        inference_fn, resolved_path = resolve_policy(
            il_net, cfg.train, obs_shape,
            policy=args.policy,
            checkpoint=ckpt_path,
            teacher_checkpoint=args.teacher_checkpoint,
            deterministic=not args.stochastic,
        )

        print(f"\nEvaluating: {resolved_path}")
        print(f"  batch_envs={batch_envs}  episodes={args.episodes}  "
              f"steps={episode_length}")

        num_batches = math.ceil(args.episodes / batch_envs)
        buffer: dict[str, list[np.ndarray]] = {}
        written = 0

        for bi in range(num_batches):
            batch = _rollout_batch(
                env, inference_fn,
                batch_envs=batch_envs,
                episode_length=episode_length,
                success_steps_required=success_steps,
                landing_radius=landing_radius,
                seed=args.seed + bi,
            )
            keep = min(batch_envs, args.episodes - written)
            for k, v in batch.items():
                buffer.setdefault(k, []).append(v[:keep])
            written += keep
            print(f"  batch {bi+1}/{num_batches}: {written}/{args.episodes} episodes")

        stacked = {k: np.concatenate(v) for k, v in buffer.items()}
        summary = _summarise(stacked, args.episodes)
        summary["checkpoint"] = str(resolved_path)
        summary["policy"] = args.policy

        _print_batch_report(summary, f"{args.policy.upper()}: {resolved_path}")

        if args.json_out:
            out = Path(args.json_out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(summary, indent=2, sort_keys=True))
            print(f"Saved JSON: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Diagnostic mode
# ═══════════════════════════════════════════════════════════════════════════════

def _run_diagnostic(args):
    cfg = load_config()
    batch_envs = max(args.batch_envs, 1)
    env, cfg_env = build_env(cfg, batch_envs)

    jit_reset = jax.jit(jax.vmap(env.reset))
    jit_step = jax.jit(jax.vmap(env.step))

    rng = jax.random.PRNGKey(args.seed)
    rng, key = jax.random.split(rng)
    state = jit_reset(jax.random.split(key, batch_envs))

    obs_shape = jax.tree_util.tree_map(lambda x: x.shape[1:], state.obs)
    action_size = env._mj_model.nu
    il_net = build_network(cfg.train, obs_shape, action_size)

    inference_fn, ckpt_path = resolve_policy(
        il_net, cfg.train, obs_shape,
        policy=args.policy,
        checkpoint=args.checkpoint,
        teacher_checkpoint=args.teacher_checkpoint,
        deterministic=True,
    )

    # JIT warmup
    print("JIT compiling (this takes a moment with vision)...")
    obs_0 = jax.tree.map(lambda x: x[0], state.obs)
    act_0, _ = inference_fn(obs_0, rng)
    actions = jnp.broadcast_to(act_0, (batch_envs,) + act_0.shape)
    state = jit_step(state, actions)
    print("JIT done.\n")

    header = (
        f"{'step':>4s}  {'x':>7s} {'y':>7s} {'z':>7s}  "
        f"{'vx':>7s} {'vy':>7s} {'vz':>7s}  "
        f"{'wx':>7s} {'wy':>7s} {'wz':>7s}  "
        f"{'reward':>8s} {'done':>4s}  {'dist':>6s}  "
        f"{'tilt':>6s} {'ang':>6s} {'FOV':>3s} {'pad_px':>6s}  actions"
    )
    print("=" * 160)
    print(header)
    print("=" * 160)

    all_fov_pcts = []

    for ep in range(args.episodes):
        rng, key = jax.random.split(rng)
        state = jit_reset(jax.random.split(key, batch_envs))

        pos0 = np.array(state.data.qpos[0, 0:3])
        tgt0 = np.array(state.info["target_buffer"][0, 0, :])
        print(
            f"\n--- Episode {ep+1}/{args.episodes} --- "
            f"start=({pos0[0]:.2f}, {pos0[1]:.2f}, {pos0[2]:.2f}) "
            f"target=({tgt0[0]:.2f}, {tgt0[1]:.2f}, {tgt0[2]:.2f})"
        )

        total_reward = 0.0
        ep_angles, ep_in_fov, ep_tilt, ep_pad_px = [], [], [], []
        ep_rw_accum: dict[str, float] = {}

        for step in range(args.steps):
            rng, key = jax.random.split(rng)
            obs_0 = jax.tree.map(lambda x: x[0], state.obs)
            act_0, _ = inference_fn(obs_0, key)
            actions = jnp.broadcast_to(act_0, (batch_envs,) + act_0.shape)
            state = jit_step(state, actions)

            pos = np.array(state.data.qpos[0, 0:3])
            vel = np.array(state.data.qvel[0, 0:3])
            angvel = np.array(state.data.qvel[0, 3:6])
            r = float(state.reward[0])
            d = float(state.done[0])
            act = np.array(act_0)
            total_reward += r

            for mk, mv in state.metrics.items():
                leaf = jax.tree_util.tree_leaves(mv)[0]
                val = float(leaf[0]) if leaf.ndim > 0 else float(leaf)
                ep_rw_accum[mk] = ep_rw_accum.get(mk, 0.0) + val

            target = np.array(state.info["target_buffer"][0, 0, :])
            dist = np.linalg.norm(pos[:2] - target[:2])

            vis = compute_camera_visibility(np.array(state.data.qpos[0]), target)
            ep_angles.append(vis["angle_deg"])
            ep_in_fov.append(vis["in_fov"])
            ep_tilt.append(vis["tilt_deg"])
            ep_pad_px.append(vis["pad_px"])

            if step % 5 == 0 or step < 3 or d > 0.5 or step == args.steps - 1:
                act_str = "[" + ", ".join(f"{a:+.2f}" for a in act) + "]"
                fov_str = " Y " if vis["in_fov"] else " N "
                print(
                    f"{step:4d}  {pos[0]:7.3f} {pos[1]:7.3f} {pos[2]:7.3f}  "
                    f"{vel[0]:7.3f} {vel[1]:7.3f} {vel[2]:7.3f}  "
                    f"{angvel[0]:7.3f} {angvel[1]:7.3f} {angvel[2]:7.3f}  "
                    f"{r:8.3f} {d:4.0f}  {dist:6.3f}  "
                    f"{vis['tilt_deg']:6.1f} {vis['angle_deg']:6.1f} {fov_str} "
                    f"{vis['pad_px']:6.1f}  {act_str}"
                )

            if d > 0.5:
                break

        outcome = "SUCCESS" if step < args.steps - 1 and pos[2] < 0.1 else "TIMEOUT/CRASH"
        fov_pct = 100.0 * np.mean(ep_in_fov)
        all_fov_pcts.append(fov_pct)
        print(
            f"    --> {outcome}  total_reward={total_reward:.2f}  "
            f"final=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})  "
            f"final_dist={dist:.3f}  steps={step+1}"
        )
        print(
            f"    [CAMERA] FOV%={fov_pct:.1f}  "
            f"mean_angle={np.mean(ep_angles):.1f} deg  "
            f"mean_tilt={np.mean(ep_tilt):.1f} deg  "
            f"mean_pad_px={np.mean(ep_pad_px):.1f}"
        )
        rw_keys = sorted(k for k in ep_rw_accum if k.startswith("reward_"))
        if rw_keys:
            print("    [REWARDS]")
            for rk in rw_keys:
                print(f"      {rk:<35s} {ep_rw_accum[rk]:+.3f}")

    print("\n" + "=" * 80)
    print("CAMERA VISIBILITY SUMMARY")
    print("=" * 80)
    for i, pct in enumerate(all_fov_pcts):
        bar = "#" * int(pct / 2) + "." * (50 - int(pct / 2))
        print(f"  Episode {i+1:2d}: [{bar}] {pct:5.1f}% in FOV")
    print(f"  Overall mean FOV%: {np.mean(all_fov_pcts):.1f}%")
    print("=" * 80)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Unified evaluator for ARCDrone DAgger policies.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Common arguments
    parser.add_argument(
        "--mode", choices=["gui", "batch", "diagnostic"],
        default="batch",
        help="Evaluation mode (default: batch).",
    )
    parser.add_argument(
        "--policy", choices=["student", "teacher"], default="student",
        help="Which policy to evaluate.",
    )
    parser.add_argument(
        "--checkpoint", nargs="+", default=None,
        help="Path(s) to student checkpoint .pkl. Defaults to latest in outputs/.",
    )
    parser.add_argument(
        "--teacher_checkpoint", type=str, default=None,
        help="Path to teacher checkpoint (required for --policy teacher).",
    )
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--steps", type=int, default=None,
                        help="Max steps per episode (defaults to config episode_length).")
    parser.add_argument("--seed", type=int, default=42)

    # Batch-specific
    parser.add_argument("--batch_envs", type=int, default=32,
                        help="Parallel envs per batch (batch/diagnostic modes).")
    parser.add_argument("--json_out", type=str, default=None,
                        help="Path to save JSON summary (batch mode).")
    parser.add_argument("--stochastic", action="store_true",
                        help="Sample actions stochastically (batch mode).")

    args = parser.parse_args()

    # Default steps from config if not provided
    if args.steps is None:
        cfg = load_config()
        args.steps = int(cfg.train.episode_length)

    # For single-checkpoint modes, unwrap the list
    if args.checkpoint and len(args.checkpoint) == 1 and args.mode != "batch":
        args.checkpoint = args.checkpoint[0]

    dispatch = {
        "gui": _run_gui,
        "batch": _run_batch,
        "diagnostic": _run_diagnostic,
    }
    dispatch[args.mode](args)


if __name__ == "__main__":
    main()
