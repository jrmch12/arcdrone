#!/usr/bin/env python3
"""Diagnostic evaluator for SITT mounted-camera drone landing.

Prints full drone state each step: position, velocity, angular velocity,
camera visibility, reward, actions.

Usage (from repo root):
    source /home/jrmch12f/Documents/code/mujoco_playground/.venv/bin/activate

    # Student policy:
    python src/arcdrone/vision_landing_sitt/evaluate_diagnostic.py \
        --checkpoint outputs/<run>/student_model.pkl --episodes 3

    # Teacher policy:
    python src/arcdrone/vision_landing_sitt/evaluate_diagnostic.py \
        --policy teacher \
        --checkpoint outputs/<run>/teacher_model.pkl --episodes 3
"""
from __future__ import annotations

import argparse
import functools
import os
from copy import deepcopy
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")

import jax
import jax.numpy as jnp
import numpy as np
from brax.io import model
from brax.training.acme import running_statistics
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("mul", lambda a, b: int(a) * int(b), replace=True)

from arcdrone.New_attempt_2.task.arcdrone import ARCDroneVisionLandingIL
from arcdrone.vision_landing_sitt.training import networks as sitt_networks

_THIS_DIR = Path(__file__).resolve().parent
CFG_DIR = _THIS_DIR / "cfg"
PROJECT_ROOT = _THIS_DIR.parents[2]


# ── Camera-target visibility helpers ─────────────────────────────────────────

def _quat_to_rotmat(quat):
    w, x, y, z = quat
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])


def compute_camera_visibility(qpos, target_pos, cam_fovy_deg=70.0, cam_res_h=64,
                               landing_radius=0.4):
    drone_pos = qpos[0:3]
    drone_quat = qpos[3:7]
    tilt = float(qpos[7])

    R_body = _quat_to_rotmat(drone_quat)
    look_body = np.array([-np.cos(tilt), 0.0, np.sin(tilt)])
    look_world = R_body @ look_body

    cam_offset_body = np.array([-0.15, 0.0, 0.05])
    cam_pos = drone_pos + R_body @ cam_offset_body

    to_target = target_pos - cam_pos
    dist = np.linalg.norm(to_target)
    to_target_n = to_target / (dist + 1e-8)

    cos_a = np.clip(np.dot(look_world, to_target_n), -1.0, 1.0)
    angle_deg = float(np.degrees(np.arccos(cos_a)))

    fov_half = cam_fovy_deg / 2.0
    in_fov = angle_deg < fov_half

    pad_ang = np.degrees(2 * np.arctan(landing_radius / (dist + 1e-8)))
    pad_px = pad_ang / cam_fovy_deg * cam_res_h

    return {
        "tilt_deg": float(np.degrees(tilt)),
        "angle_deg": angle_deg,
        "in_fov": in_fov,
        "dist": dist,
        "pad_px": pad_px,
    }


def _find_latest_checkpoint(outputs_dir: Path, filename: str) -> str:
    candidates = list(outputs_dir.rglob(filename))
    if not candidates:
        raise FileNotFoundError(f"No {filename} found under: {outputs_dir}")
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return str(latest)


def main():
    parser = argparse.ArgumentParser(description="Diagnostic SITT evaluator (mounted camera)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pkl")
    parser.add_argument("--policy", type=str, default="student", choices=["student", "teacher"])
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ── Config ──
    initialize_config_dir(config_dir=str(CFG_DIR), job_name="sitt_diag", version_base=None)
    cfg = compose(config_name="config")
    cfg_env = OmegaConf.to_container(cfg.env, resolve=True)
    cfg_train = cfg.train

    cfg_env["vision_config"]["nworld"] = 1
    cfg_env["naconmax"] = cfg_env["njmax"]
    cfg_env["naccdmax"] = cfg_env["njmax"] #TODO

    # ── Envs ──
    teacher_cfg = deepcopy(cfg_env)
    teacher_cfg["enable_vision_obs"] = False
    teacher_cfg["vision"] = False
    teacher_env = ARCDroneVisionLandingIL(cfg=teacher_cfg)

    student_env = ARCDroneVisionLandingIL(cfg=deepcopy(cfg_env))

    active_env = student_env if args.policy == "student" else teacher_env

    v_reset = jax.vmap(active_env.reset)
    v_step  = jax.vmap(active_env.step)
    jit_reset = jax.jit(v_reset)
    jit_step  = jax.jit(v_step)

    rng = jax.random.PRNGKey(args.seed)
    rng, key_reset = jax.random.split(rng)
    state = jit_reset(jax.random.split(key_reset, 1))

    # obs shapes
    teacher_state = teacher_env.reset(jax.random.PRNGKey(0))
    teacher_obs_shape = jax.tree_util.tree_map(lambda x: x.shape, teacher_state.obs)

    student_vmap_rst = jax.vmap(student_env.reset)
    student_state = jax.jit(student_vmap_rst)(jax.random.split(jax.random.PRNGKey(1), 1))
    student_obs_shape = jax.tree_util.tree_map(lambda x: x.shape, student_state.obs)

    # ── Networks ──
    preprocess_obs_fn = (
        running_statistics.normalize
        if cfg_train.normalize_observations
        else (lambda x, y: x)
    )
    network_factory = functools.partial(
        sitt_networks.make_sitt_networks,
        preprocess_observations_fn=preprocess_obs_fn,
        teacher_dec_hidden_layers=cfg_train.teacher_dec_hidden_layers,
        policy_dec_hidden_layers=cfg_train.policy_dec_hidden_layers,
        policy_proprio_proj_hidden_layers=cfg_train.policy_proprio_proj_hidden_layers,
        proxy_hidden_layers=cfg_train.proxy_hidden_layers,
        action_hidden_layer_sizes=cfg_train.action_hidden_layers,
        value_hidden_layer_sizes=cfg_train.value_hidden_layers,
        cnn_num_filters=cfg_train.cnn_num_filters,
        cnn_kernel_sizes=cfg_train.cnn_kernel_sizes,
        cnn_strides=cfg_train.cnn_strides,
        policy_obs_key=cfg_train.policy_obs_key,
        value_obs_key=cfg_train.value_obs_key,
        teacher_obs_key=cfg_train.teacher_obs_key,
        policy_pixels_key=cfg_train.policy_pixels_key,
        policy_proprio_key=cfg_train.policy_proprio_key,
    )
    sitt_net = network_factory(
        teacher_obs_shape,
        teacher_env.action_size,
        student_observation_size=student_obs_shape,
    )

    # ── Load policy ──
    if args.checkpoint is None:
        fn = "student_model.pkl" if args.policy == "student" else "teacher_model.pkl"
        ckpt_path = _find_latest_checkpoint(PROJECT_ROOT / "outputs", fn)
    else:
        ckpt_path = args.checkpoint
    print(f"Loading {args.policy} checkpoint: {ckpt_path}")

    if args.policy == "student":
        ckpt = model.load_params(ckpt_path)
        proprio_norm, (student_enc, action_head) = ckpt
        make_policy = sitt_networks.make_student_inference_fn(
            sitt_net, action_head_params=action_head
        )
        inference_fn = make_policy((proprio_norm, student_enc), deterministic=True)
    else:
        ckpt = model.load_params(ckpt_path)
        norm_whole, policy_params, _ = ckpt
        policy_obs_key = getattr(cfg_train, "policy_obs_key", "teacher_obs")
        teacher_norm = sitt_networks._select_normalizer_by_path(norm_whole, policy_obs_key)
        inference_fn = sitt_networks.make_frozen_teacher_policy(
            sitt_net,
            teacher_norm_params=teacher_norm,
            teacher_policy_params=policy_params,
            deterministic=True,
        )

    jit_inference = jax.jit(inference_fn)

    # ── JIT warmup ──
    print("JIT compiling...")
    obs_0 = jax.tree.map(lambda x: x[0], state.obs)
    act_0, _ = jit_inference(obs_0, rng)
    actions = jnp.broadcast_to(act_0, (1,) + act_0.shape)
    state = jit_step(state, actions)
    print("JIT done.\n")

    # ── Header ──
    print("=" * 160)
    print(f"{'step':>4s}  {'x':>7s} {'y':>7s} {'z':>7s}  {'vx':>7s} {'vy':>7s} {'vz':>7s}  "
          f"{'wx':>7s} {'wy':>7s} {'wz':>7s}  {'reward':>8s} {'done':>4s}  {'dist':>6s}  "
          f"{'tilt°':>6s} {'ang°':>6s} {'FOV':>3s} {'pad_px':>6s}  actions")
    print("=" * 160)

    all_fov = []

    for ep in range(args.episodes):
        rng, reset_key = jax.random.split(rng)
        state = jit_reset(jax.random.split(reset_key, 1))

        pos0 = np.array(state.data.qpos[0, 0:3])
        target0 = np.array(state.info["target_buffer"][0, 0, :])
        print(f"\n--- Episode {ep+1}/{args.episodes} --- "
              f"start=({pos0[0]:.2f}, {pos0[1]:.2f}, {pos0[2]:.2f}) "
              f"target=({target0[0]:.2f}, {target0[1]:.2f}, {target0[2]:.2f})")

        total_reward = 0.0
        ep_angles, ep_in_fov, ep_tilt, ep_pad_px = [], [], [], []

        for step in range(args.max_steps):
            rng, action_key = jax.random.split(rng)
            obs_0 = jax.tree.map(lambda x: x[0], state.obs)
            act_0, _ = jit_inference(obs_0, action_key)
            actions = jnp.broadcast_to(act_0, (1,) + act_0.shape)
            state = jit_step(state, actions)

            pos = np.array(state.data.qpos[0, 0:3])
            vel = np.array(state.data.qvel[0, 0:3])
            angvel = np.array(state.data.qvel[0, 3:6])
            r = float(state.reward[0])
            d = float(state.done[0])
            act = np.array(act_0)
            total_reward += r

            target = np.array(state.info["target_buffer"][0, 0, :])
            dist_xy = np.linalg.norm(pos[:2] - target[:2])

            qpos_full = np.array(state.data.qpos[0])
            vis = compute_camera_visibility(qpos_full, target)
            ep_angles.append(vis["angle_deg"])
            ep_in_fov.append(vis["in_fov"])
            ep_tilt.append(vis["tilt_deg"])
            ep_pad_px.append(vis["pad_px"])

            if step % 5 == 0 or step < 3 or d > 0.5 or step == args.max_steps - 1:
                act_str = "[" + ", ".join(f"{a:+.2f}" for a in act) + "]"
                fov_str = " Y " if vis["in_fov"] else " N "
                print(f"{step:4d}  {pos[0]:7.3f} {pos[1]:7.3f} {pos[2]:7.3f}  "
                      f"{vel[0]:7.3f} {vel[1]:7.3f} {vel[2]:7.3f}  "
                      f"{angvel[0]:7.3f} {angvel[1]:7.3f} {angvel[2]:7.3f}  "
                      f"{r:8.3f} {d:4.0f}  {dist_xy:6.3f}  "
                      f"{vis['tilt_deg']:6.1f} {vis['angle_deg']:6.1f} {fov_str} {vis['pad_px']:6.1f}  "
                      f"{act_str}")

            if d > 0.5:
                break

        outcome = "SUCCESS" if step < args.max_steps - 1 and pos[2] < 0.1 else "TIMEOUT/CRASH"
        fov_pct = 100.0 * np.mean(ep_in_fov)
        all_fov.append(fov_pct)
        print(f"    --> {outcome}  total_reward={total_reward:.2f}  "
              f"final=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})  "
              f"final_dist={dist_xy:.3f}  steps={step+1}")
        print(f"    [CAMERA] FOV%={fov_pct:.1f}  "
              f"mean_angle={np.mean(ep_angles):.1f} deg  "
              f"mean_tilt={np.mean(ep_tilt):.1f} deg  "
              f"mean_pad_px={np.mean(ep_pad_px):.1f}  "
              f"min_angle={np.min(ep_angles):.1f} deg  "
              f"max_angle={np.max(ep_angles):.1f} deg")

    # ── Summary ──
    print("\n" + "=" * 80)
    print("CAMERA VISIBILITY SUMMARY")
    print("=" * 80)
    for i, pct in enumerate(all_fov):
        bar = "#" * int(pct / 2) + "." * (50 - int(pct / 2))
        print(f"  Episode {i+1:2d}: {bar} {pct:5.1f}% in FOV")
    print(f"  Overall mean FOV%: {np.mean(all_fov):.1f}%")
    print("=" * 80)
    print("\nDone.")


if __name__ == "__main__":
    main()
