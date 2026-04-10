#!/usr/bin/env python3
"""Diagnostic evaluator: prints full drone state each step to understand policy behavior.

Usage (from repo root):
    source /home/jrmch12f/Documents/code/mujoco_playground/.venv/bin/activate
    python src/arcdrone/New_attempt_2/evaluate_diagnostic.py \
        --checkpoint outputs/2026-04-08/10-03-03/best_model.pkl \
        --episodes 3 --max_steps 200

  Or to evaluate the teacher:
    python src/arcdrone/New_attempt_2/evaluate_diagnostic.py \
        --policy teacher \
        --teacher_checkpoint outputs/2026-03-21/18-28-17/teacher_model.pkl \
        --episodes 3

  Camera-target visibility analysis is always printed.
"""
from __future__ import annotations

import argparse
import functools
import os
import sys
from pathlib import Path

# Self-contained imports
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")

import jax
import jax.numpy as jnp
import numpy as np
from brax.io import model
from brax.training.acme import running_statistics, specs
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from task.arcdrone import ARCDroneVisionLandingIL
from training import networks as dagger_networks

OmegaConf.register_new_resolver("mul", lambda a, b: int(a) * int(b), replace=True)

CFG_DIR = _THIS_DIR / "cfg"
PROJECT_ROOT = _THIS_DIR.parents[2]


# ── Camera-target visibility helpers ─────────────────────────────────────────

def _quat_to_rotmat(quat):
    """MuJoCo quaternion (w, x, y, z) -> 3x3 rotation matrix."""
    w, x, y, z = quat
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])


def compute_camera_visibility(qpos, target_pos, cam_fovy_deg=70.0, cam_res_h=64,
                               landing_radius=0.4):
    """Compute camera-target visibility metrics from drone state.

    The x2_camera is on a tilt joint (hinge around body-Y, range [-π/2, 0]).
    Default look direction in body frame at tilt=0: (-1, 0, 0) (forward).
    At tilt = -π/2: (0, 0, -1) (straight down).

    Args:
        qpos:  full qpos array [x,y,z, qw,qx,qy,qz, tilt_joint, ...]
        target_pos: (3,) world-frame landing target
        cam_fovy_deg: vertical FOV in degrees (MuJoCo default 45)
        cam_res_h: camera resolution height (pixels)
        landing_radius: radius of the landing pad (metres)

    Returns:  dict with tilt_deg, angle_to_target_deg, in_fov, pad_pixels, ...
    """
    drone_pos = qpos[0:3]
    drone_quat = qpos[3:7]
    tilt = float(qpos[7])  # camera tilt joint value (radians)

    R_body = _quat_to_rotmat(drone_quat)

    # Camera look direction in body frame after tilt rotation around Y
    look_body = np.array([-np.cos(tilt), 0.0, np.sin(tilt)])
    look_world = R_body @ look_body

    # Camera world position (mount offset in body frame)
    cam_offset_body = np.array([-0.15, 0.0, 0.05])
    cam_pos = drone_pos + R_body @ cam_offset_body

    # Vector from camera to target
    to_target = target_pos - cam_pos
    dist = np.linalg.norm(to_target)
    to_target_n = to_target / (dist + 1e-8)

    # Angle between camera look direction and direction to target
    cos_a = np.clip(np.dot(look_world, to_target_n), -1.0, 1.0)
    angle_deg = float(np.degrees(np.arccos(cos_a)))

    fov_half = cam_fovy_deg / 2.0
    in_fov = angle_deg < fov_half

    # Approx pad size in pixels (angular subtense mapped to image height)
    pad_ang = np.degrees(2 * np.arctan(landing_radius / (dist + 1e-8)))
    pad_px = pad_ang / cam_fovy_deg * cam_res_h

    return {
        "tilt_deg": float(np.degrees(tilt)),
        "angle_deg": angle_deg,
        "in_fov": in_fov,
        "dist": dist,
        "pad_px": pad_px,
        "look_world": look_world,
    }


def _find_latest_checkpoint(outputs_dir: Path) -> str:
    candidates = list(outputs_dir.rglob("best_model.pkl"))
    candidates.extend(outputs_dir.rglob("trained_model.pkl"))
    if not candidates:
        raise FileNotFoundError(f"No model checkpoint found under: {outputs_dir}")
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return str(latest)


def main():
    parser = argparse.ArgumentParser(description="Diagnostic drone state evaluator")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to student checkpoint .pkl")
    parser.add_argument("--teacher_checkpoint", type=str, default=None, help="Path to teacher checkpoint .pkl")
    parser.add_argument("--policy", type=str, default="student", choices=["student", "teacher"])
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_envs", type=int, default=1, help="Number of parallel envs (only env 0 is logged)")
    args = parser.parse_args()

    # Load config
    initialize_config_dir(config_dir=str(CFG_DIR), job_name="diag_eval", version_base=None)
    cfg = compose(config_name="config")
    cfg_env = OmegaConf.to_container(cfg.env, resolve=True)
    cfg_train = cfg.train
    batch_envs = max(args.batch_envs, 1)
    cfg_env["vision_config"]["nworld"] = batch_envs
    cfg_env["naconmax"] = int(cfg_env["njmax"]) * batch_envs

    print("Creating environment...")
    env = ARCDroneVisionLandingIL(cfg=cfg_env)

    # Build network
    v_reset = jax.vmap(env.reset)
    v_step = jax.vmap(env.step)
    jit_reset = jax.jit(v_reset)
    jit_step = jax.jit(v_step)

    rng = jax.random.PRNGKey(args.seed)
    rng, key_reset = jax.random.split(rng)
    state = jit_reset(jax.random.split(key_reset, batch_envs))
    obs_shape = jax.tree_util.tree_map(lambda x: x.shape[1:], state.obs)
    action_size = env._mj_model.nu

    network_factory = functools.partial(
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
    il_net = network_factory(obs_shape, action_size)

    # Build policy
    if args.policy == "student":
        ckpt_path = args.checkpoint
        if ckpt_path is None:
            ckpt_path = _find_latest_checkpoint(PROJECT_ROOT / "outputs")
        print(f"Loading student checkpoint: {ckpt_path}")
        ckpt = model.load_params(ckpt_path)
        proprio_norm = ckpt[0]
        restored = ckpt[1]
        if isinstance(restored, (tuple, list)) and len(restored) == 2:
            student_enc, action_head = restored
        else:
            raise ValueError("Old checkpoint format — needs teacher for action head fallback")

        # Coerce normalizer
        proprio_size = int(obs_shape["proprio_obs"][-1])
        try:
            if tuple(proprio_norm.mean.shape) != (proprio_size,):
                raise AttributeError
        except AttributeError:
            proprio_norm = running_statistics.init_state(
                specs.Array((proprio_size,), jnp.dtype("float32"))
            )

        make_policy = dagger_networks.make_student_inference_fn(il_net)
        inference_fn = make_policy((proprio_norm, student_enc, action_head), deterministic=True)
    else:
        if not args.teacher_checkpoint:
            raise ValueError("--teacher_checkpoint required for teacher policy")
        print(f"Loading teacher checkpoint: {args.teacher_checkpoint}")
        teacher_ckpt = model.load_params(args.teacher_checkpoint)
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

    jit_inference = jax.jit(inference_fn)

    # JIT warmup
    print("JIT compiling (this takes a moment with vision)...")
    obs_0 = jax.tree.map(lambda x: x[0], state.obs)
    act_0, _ = jit_inference(obs_0, rng)
    actions = jnp.broadcast_to(act_0, (batch_envs,) + act_0.shape)
    state = jit_step(state, actions)
    print("JIT done.\n")

    # Header
    print("=" * 160)
    print(f"{'step':>4s}  {'x':>7s} {'y':>7s} {'z':>7s}  {'vx':>7s} {'vy':>7s} {'vz':>7s}  "
          f"{'wx':>7s} {'wy':>7s} {'wz':>7s}  {'reward':>8s} {'done':>4s}  {'dist':>6s}  "
          f"{'tilt°':>6s} {'ang°':>6s} {'FOV':>3s} {'pad_px':>6s}  actions")
    print("=" * 160)

    all_episodes_fov = []  # collect per-episode FOV %

    for ep in range(args.episodes):
        rng, reset_key = jax.random.split(rng)
        reset_keys = jax.random.split(reset_key, batch_envs)
        state = jit_reset(reset_keys)

        # Extract env 0 initial state
        pos0 = np.array(state.data.qpos[0, 0:3])
        target0 = np.array(state.info["target_buffer"][0, 0, :])
        print(f"\n--- Episode {ep+1}/{args.episodes} --- "
              f"start=({pos0[0]:.2f}, {pos0[1]:.2f}, {pos0[2]:.2f}) "
              f"target=({target0[0]:.2f}, {target0[1]:.2f}, {target0[2]:.2f})")

        total_reward = 0.0
        # Camera visibility tracking for this episode
        ep_angles = []
        ep_in_fov = []
        ep_tilt = []
        ep_pad_px = []

        for step in range(args.max_steps):
            rng, action_key = jax.random.split(rng)
            obs_0 = jax.tree.map(lambda x: x[0], state.obs)
            act_0, _ = jit_inference(obs_0, action_key)
            actions = jnp.broadcast_to(act_0, (batch_envs,) + act_0.shape)
            state = jit_step(state, actions)

            # Extract env 0 data
            pos = np.array(state.data.qpos[0, 0:3])
            vel = np.array(state.data.qvel[0, 0:3])
            angvel = np.array(state.data.qvel[0, 3:6])
            r = float(state.reward[0])
            d = float(state.done[0])
            act = np.array(act_0)
            total_reward += r

            target = np.array(state.info["target_buffer"][0, 0, :])
            dist = np.linalg.norm(pos[:2] - target[:2])

            # Camera visibility
            qpos_full = np.array(state.data.qpos[0])
            vis = compute_camera_visibility(qpos_full, target)
            ep_angles.append(vis["angle_deg"])
            ep_in_fov.append(vis["in_fov"])
            ep_tilt.append(vis["tilt_deg"])
            ep_pad_px.append(vis["pad_px"])

            # Print every 5 steps + first 3 + last + done
            if step % 5 == 0 or step < 3 or d > 0.5 or step == args.max_steps - 1:
                act_str = "[" + ", ".join(f"{a:+.2f}" for a in act) + "]"
                fov_str = " Y " if vis["in_fov"] else " N "
                print(f"{step:4d}  {pos[0]:7.3f} {pos[1]:7.3f} {pos[2]:7.3f}  "
                      f"{vel[0]:7.3f} {vel[1]:7.3f} {vel[2]:7.3f}  "
                      f"{angvel[0]:7.3f} {angvel[1]:7.3f} {angvel[2]:7.3f}  "
                      f"{r:8.3f} {d:4.0f}  {dist:6.3f}  "
                      f"{vis['tilt_deg']:6.1f} {vis['angle_deg']:6.1f} {fov_str} {vis['pad_px']:6.1f}  "
                      f"{act_str}")

            if d > 0.5:
                break

        outcome = "SUCCESS" if step < args.max_steps - 1 and pos[2] < 0.1 else "TIMEOUT/CRASH"
        fov_pct = 100.0 * np.mean(ep_in_fov)
        all_episodes_fov.append(fov_pct)
        print(f"    --> {outcome}  total_reward={total_reward:.2f}  "
              f"final=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})  "
              f"final_dist={dist:.3f}  steps={step+1}")
        print(f"    [CAMERA] FOV%={fov_pct:.1f}  "
              f"mean_angle={np.mean(ep_angles):.1f}°  "
              f"mean_tilt={np.mean(ep_tilt):.1f}°  "
              f"mean_pad_px={np.mean(ep_pad_px):.1f}  "
              f"min_angle={np.min(ep_angles):.1f}°  "
              f"max_angle={np.max(ep_angles):.1f}°")

    # Global summary
    print("\n" + "=" * 80)
    print("CAMERA VISIBILITY SUMMARY ACROSS ALL EPISODES")
    print("=" * 80)
    for i, pct in enumerate(all_episodes_fov):
        bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
        print(f"  Episode {i+1:2d}: {bar} {pct:5.1f}% in FOV")
    print(f"  Overall mean FOV%: {np.mean(all_episodes_fov):.1f}%")
    print("=" * 80)
    print("\nDone.")


if __name__ == "__main__":
    main()
