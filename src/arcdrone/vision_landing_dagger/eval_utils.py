"""Shared evaluation utilities for ARCDrone DAgger policies.

Provides environment setup, network construction, checkpoint loading,
and policy building used across all evaluation modes (gui, batch, diagnostic).
"""
from __future__ import annotations

import functools
import os
import sys
from pathlib import Path

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
from brax.training.acme import running_statistics, specs
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
from task.arcdrone import ARCDroneVisionLandingIL
from training import networks as dagger_networks

OmegaConf.register_new_resolver("mul", lambda a, b: int(a) * int(b), replace=True)

CFG_DIR = _THIS_DIR / "cfg"
PROJECT_ROOT = _THIS_DIR.parents[2]
MUJOCO_XML = str(PROJECT_ROOT / "assets" / "skydio_x2" / "scene_mounted_cam.xml")


# ── Config & environment ─────────────────────────────────────────────────────

def load_config():
    """Load Hydra config from the cfg/ directory. Returns the full cfg object.

    Safe to call multiple times in the same process — Hydra is only
    initialized on the first call.
    """
    if not GlobalHydra.instance().is_initialized():
        initialize_config_dir(
            config_dir=str(CFG_DIR), job_name="evaluate", version_base=None
        )
    return compose(config_name="config")


def build_env(cfg, batch_envs: int):
    """Create the ARCDrone environment with the correct nworld setting."""
    cfg_env = OmegaConf.to_container(cfg.env, resolve=True)
    cfg_env["vision_config"]["nworld"] = int(batch_envs)
    cfg_env["naconmax"] = int(cfg_env["njmax"]) * int(batch_envs)
    return ARCDroneVisionLandingIL(cfg=cfg_env), cfg_env


# ── Network factory ──────────────────────────────────────────────────────────

def build_network_factory(cfg_train):
    """Return a functools.partial that builds the IL network from config."""
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
        policy_proprio_key=cfg_train.policy_proprio_key,
        teacher_obs_key=cfg_train.teacher_obs_key,
        value_obs_key=cfg_train.value_obs_key,
    )


def build_network(cfg_train, obs_shape, action_size):
    """Instantiate the IL network from config and observation/action shapes."""
    return build_network_factory(cfg_train)(obs_shape, action_size)


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def find_latest_checkpoint(outputs_dir: Path | str | None = None) -> str:
    """Find the most recent best_model.pkl or trained_model.pkl under *outputs_dir*."""
    if outputs_dir is None:
        outputs_dir = PROJECT_ROOT / "outputs"
    outputs_dir = Path(outputs_dir)
    candidates = list(outputs_dir.rglob("best_model.pkl"))
    candidates.extend(outputs_dir.rglob("trained_model.pkl"))
    if not candidates:
        raise FileNotFoundError(f"No model checkpoint found under: {outputs_dir}")
    return str(max(candidates, key=lambda p: p.stat().st_mtime))


def coerce_proprio_norm(norm, obs_shape):
    """Normalizer compatibility guard for older checkpoints."""
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


# ── Policy builders ──────────────────────────────────────────────────────────

def build_student_policy(
    il_network,
    obs_shape,
    checkpoint_path: str,
    deterministic: bool = True,
):
    """Load a student checkpoint and return a JIT-compiled inference function."""
    ckpt = model.load_params(checkpoint_path)
    proprio_norm = coerce_proprio_norm(ckpt[0], obs_shape)
    restored = ckpt[1]

    if isinstance(restored, (tuple, list)) and len(restored) == 3:
        student_enc, action_head, vel_estimator = restored
    else:
        raise ValueError(
            "Incompatible checkpoint format — expected 3-element tuple "
            "(student_enc, action_head, vel_estimator)."
        )

    make_policy = dagger_networks.make_student_inference_fn(il_network)
    inference_fn = make_policy(
        (proprio_norm, student_enc, action_head, vel_estimator),
        deterministic=deterministic,
    )
    return jax.jit(inference_fn)


def build_teacher_policy(
    il_network,
    cfg_train,
    teacher_checkpoint_path: str,
    deterministic: bool = True,
):
    """Load a teacher checkpoint and return a JIT-compiled inference function."""
    teacher_ckpt = model.load_params(teacher_checkpoint_path)
    try:
        teacher_norm = dagger_networks._select_normalizer_by_path(
            teacher_ckpt[0], cfg_train.teacher_normalizer_key
        )
    except (KeyError, TypeError):
        try:
            teacher_norm = dagger_networks._select_normalizer_by_path(
                teacher_ckpt[0], "policy_obs"
            )
        except (KeyError, TypeError):
            teacher_norm = teacher_ckpt[0]
    inference_fn = dagger_networks.make_frozen_teacher_policy(
        il_network,
        teacher_norm_params=teacher_norm,
        teacher_policy_params=teacher_ckpt[1],
        deterministic=deterministic,
    )
    return jax.jit(inference_fn)


def resolve_policy(
    il_network,
    cfg_train,
    obs_shape,
    *,
    policy: str,
    checkpoint: str | None,
    teacher_checkpoint: str | None,
    deterministic: bool = True,
):
    """High-level helper: resolve student or teacher policy from CLI args.

    Returns (jit_inference_fn, selected_checkpoint_path).
    """
    policy = policy.lower()
    if policy == "teacher":
        if not teacher_checkpoint:
            raise ValueError("--teacher_checkpoint is required for teacher policy")
        fn = build_teacher_policy(
            il_network, cfg_train, teacher_checkpoint, deterministic=deterministic
        )
        return fn, teacher_checkpoint
    else:
        if checkpoint is None:
            checkpoint = find_latest_checkpoint()
        fn = build_student_policy(
            il_network, obs_shape, checkpoint, deterministic=deterministic
        )
        return fn, checkpoint


# ── Camera-target visibility helpers ─────────────────────────────────────────

def _quat_to_rotmat(quat):
    """MuJoCo quaternion (w, x, y, z) -> 3x3 rotation matrix."""
    w, x, y, z = quat
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])


def compute_camera_visibility(
    qpos, target_pos, cam_fovy_deg=70.0, cam_res_h=64, landing_radius=0.4
):
    """Compute camera-target visibility metrics from drone state.

    The x2_camera is on a tilt joint (hinge around body-Y, range [-pi/2, 0]).
    Default look direction in body frame at tilt=0: (-1, 0, 0) (forward).
    At tilt = -pi/2: (0, 0, -1) (straight down).
    """
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
        "look_world": look_world,
    }
