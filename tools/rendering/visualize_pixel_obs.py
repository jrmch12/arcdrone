"""
Visualize pixel observations from the ARCDroneIL_VisionLanding environment.

Saves one video per camera (pixels/view_0, pixels/view_1, pixels/view_2, ...)
showing exactly what the agent sees at every step of a rollout.
Also renders an external "outer_camera" view side by side.

Usage:
    python tools/rendering/visualize_pixel_obs.py --teacher_checkpoint outputs/.../trained_model.pkl
    python tools/rendering/visualize_pixel_obs.py --checkpoint outputs/.../best_model.pkl --policy student
"""

import argparse
import functools
import os

# Enable Triton GEMM for better GPU utilisation (recommended for warp impl)
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
# Let warp manage its own GPU memory — do NOT pre-allocate JAX memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"

import sys
from pathlib import Path
import jax
from jax import numpy as jp
import jax.numpy as jnp
import mediapy as media
import numpy as np

from brax.io import model
from brax.training.acme import running_statistics, specs
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
OmegaConf.register_new_resolver("mul", lambda a, b: int(a) * int(b), replace=True)

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "src" / "arcdrone" / "New_attempt_2"))
from arcdrone.New_attempt_2.task.arcdrone import ARCDroneVisionLandingIL
from arcdrone.New_attempt_2.training import networks as dagger_networks

# ===========================================================================
# Args
parser = argparse.ArgumentParser()
parser.add_argument("--teacher_checkpoint", type=str, default=None)
parser.add_argument("--checkpoint", type=str, default=None, help="Student checkpoint")
parser.add_argument("--policy", type=str, default="teacher", choices=["student", "teacher"])
parser.add_argument("--episode_length", type=int, default=300)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

# ===========================================================================
# Config

CFG_DIR = _REPO_ROOT / "src" / "arcdrone" / "New_attempt_2" / "cfg"

initialize_config_dir(config_dir=str(CFG_DIR), job_name="visualize", version_base=None)
cfg = compose(config_name="config")
cfg_env = OmegaConf.to_container(cfg.env, resolve=True)
cfg_train = cfg.train

# Single-world overrides
cfg_env["vision_config"]["nworld"] = 1
cfg_env["naconmax"] = cfg_env["njmax"]  # 1 world

# ===========================================================================
# Env setup

env = ARCDroneVisionLandingIL(cfg=cfg_env)

# NOTE: The warp renderer requires vmap even for nworld=1.
# mjx.get_rgb expects a batched (nworld, ...) layout that vmap provides.
jit_reset = jax.jit(jax.vmap(env.reset))
jit_step  = jax.jit(jax.vmap(env.step))

# ===========================================================================
# Build policy

rng = jax.random.PRNGKey(args.seed)
rng, key_reset = jax.random.split(rng)
state = jit_reset(jax.random.split(key_reset, 1))  # vmap expects batch dim
obs_shape = jax.tree_util.tree_map(lambda x: x.shape[1:], state.obs)  # strip batch dim
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

if args.policy == "teacher":
    ckpt_path = args.teacher_checkpoint
    if ckpt_path is None:
        # Try to find one
        candidates = list((_REPO_ROOT / "outputs").rglob("trained_model.pkl"))
        if not candidates:
            raise ValueError("--teacher_checkpoint required")
        ckpt_path = str(max(candidates, key=lambda p: p.stat().st_mtime))
        print(f"Auto-detected teacher: {ckpt_path}")
    teacher_ckpt = model.load_params(ckpt_path)
    try:
        teacher_norm = dagger_networks._select_normalizer_by_path(
            teacher_ckpt[0], cfg_train.teacher_normalizer_key
        )
    except (KeyError, TypeError):
        try:
            teacher_norm = dagger_networks._select_normalizer_by_path(teacher_ckpt[0], "policy_obs")
        except (KeyError, TypeError):
            # Teacher checkpoint has flat normalizer (e.g., plain PPO model)
            teacher_norm = teacher_ckpt[0]
    inference_fn = dagger_networks.make_frozen_teacher_policy(
        il_net,
        teacher_norm_params=teacher_norm,
        teacher_policy_params=teacher_ckpt[1],
        deterministic=True,
    )
    print(f"Loaded TEACHER policy from: {ckpt_path}")
else:
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        candidates = list((_REPO_ROOT / "outputs").rglob("best_model.pkl"))
        if not candidates:
            raise ValueError("--checkpoint required for student policy")
        ckpt_path = str(max(candidates, key=lambda p: p.stat().st_mtime))
        print(f"Auto-detected student: {ckpt_path}")
    ckpt = model.load_params(ckpt_path)
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
    inference_fn = make_policy((proprio_norm, student_enc, action_head, vel_estimator), deterministic=True)
    print(f"Loaded STUDENT policy from: {ckpt_path}")

jit_inference = jax.jit(inference_fn)

# ===========================================================================
# Rollout with policy

rng, reset_key = jax.random.split(rng)
state = jit_reset(jax.random.split(reset_key, 1))

# JIT warmup
obs_0 = jax.tree_util.tree_map(lambda x: x[0], state.obs)  # unbatch for inference
act_0, _ = jit_inference(obs_0, rng)
state = jit_step(state, act_0[None])  # batch action

# Fresh reset for the actual rollout
rng, reset_key = jax.random.split(rng)
state = jit_reset(jax.random.split(reset_key, 1))
rollout = [state]

print(f"Rolling out {args.episode_length} steps...")
for step in range(args.episode_length):
    rng, action_key = jax.random.split(rng)
    obs_0 = jax.tree_util.tree_map(lambda x: x[0], state.obs)  # unbatch
    act_0, _ = jit_inference(obs_0, action_key)
    state = jit_step(state, act_0[None])  # rebatch
    rollout.append(state)

    # Per-step pixel diagnostics (first 10 + every 50)
    if step < 10 or step % 50 == 0:
        pix = np.array(state.obs["pixels/view_0"][0])  # (H, W, C) — unbatch
        pos = np.array(state.data.qpos[0, 0:3])
        tilt = float(state.data.qpos[0, 7]) if state.data.qpos.shape[-1] > 7 else 0.0
        act_cam = float(act_0[4]) if act_0.shape[-1] > 4 else 0.0
        print(f"  step={step:3d}  pix: mean={pix.mean():.4f}  min={pix.min():.4f}  max={pix.max():.4f}  "
              f"nonzero%={(np.count_nonzero(pix > -0.49) / pix.size * 100):.1f}  "
              f"pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f})  tilt={np.degrees(tilt):.1f}°  "
              f"act_cam={act_cam:.2f}")

    if float(state.done[0]) > 0.5:
        print(f"  Episode done at step {step+1}")
        break

# ===========================================================================
# Save one video per camera (agent's pixel view)

camera_keys = sorted(k for k in rollout[0].obs.keys() if k.startswith("pixels/view_"))
print(f"Found camera keys: {camera_keys}")

for cam_key in camera_keys:
    raw = np.stack([np.array(r.obs[cam_key][0]) for r in rollout])  # (T, H, W, C) — [0] unbatch
    raw = raw + 0.5  # shift from [-0.5, 0.5] back to [0, 1]
    raw = np.clip(raw, 0.0, 1.0)
    
    is_grayscale = bool(cfg_env.get("grayscale_obs", True))
    if is_grayscale:
        # Stack is (H, W, buffer_size) with 1 grayscale channel per frame
        # Take the last frame and broadcast to RGB for visualization
        frames = np.repeat(raw[..., -1:], 3, axis=-1)  # (T, H, W, 3)
    else:
        # Stack is (H, W, buffer_size*3) — take last 3 channels (latest RGB frame)
        frames = raw[..., -3:]
    
    # Per-channel diagnostics
    print(f"\n--- {cam_key} diagnostics ---")
    print(f"  raw stack shape: {raw.shape}  (T, H, W, C)")
    for ch in range(raw.shape[-1]):
        ch_data = raw[..., ch]
        print(f"  channel {ch}: mean={ch_data.mean():.4f}  min={ch_data.min():.4f}  max={ch_data.max():.4f}")
    
    # Save diagnostic PNGs for individual frames
    import PIL.Image
    os.makedirs("debug_frames", exist_ok=True)
    for fidx in [0, 1, 5, 10, 50, 100, min(200, len(frames)-1)]:
        if fidx < len(frames):
            img = (frames[fidx] * 255).astype(np.uint8)
            PIL.Image.fromarray(img).save(f"debug_frames/frame_{fidx:04d}.png")
            print(f"  Saved debug_frames/frame_{fidx:04d}.png  mean={frames[fidx].mean():.4f}")

    out_name = f"obs_{cam_key.replace('/', '_')}.mp4"
    media.write_video(out_name, frames, fps=10)
    print(f"Saved: {out_name}  shape={frames.shape}  min={frames.min():.3f}  max={frames.max():.3f}")

# External render from outer_camera (MuJoCo CPU renderer, not warp)
# Unbatch the rollout states (remove vmap batch dim) for CPU rendering
rollout_unbatched = jax.tree_util.tree_map(lambda x: x[0] if x.ndim > 0 else x, rollout)
try:
    render_frames = env.render(rollout_unbatched, camera="outer_camera", width=480, height=480)
    media.write_video("render_outer.mp4", render_frames, fps=1.0 / env.dt)
    print(f"Saved: render_outer.mp4  ({len(render_frames)} frames)")
except (ValueError, AttributeError, TypeError) as e:
    print(f"Skipping outer render: {e}")
    try:
        render_frames = env.render(rollout_unbatched, camera="x2_camera", width=480, height=480)
        media.write_video("render_x2cam.mp4", render_frames, fps=1.0 / env.dt)
        print(f"Saved: render_x2cam.mp4  ({len(render_frames)} frames)")
    except (ValueError, AttributeError, TypeError) as e2:
        print(f"Skipping all renders: {e2}")
