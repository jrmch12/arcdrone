"""
Visualize pixel observations from the ARCDroneIL_VisionLanding environment.

Saves one video per camera (pixels/view_0, pixels/view_1, pixels/view_2, ...)
showing exactly what the agent sees at every step of a rollout.
"""

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
import mediapy as media
import numpy as np

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
OmegaConf.register_new_resolver("mul", lambda a, b: int(a) * int(b), replace=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from arcdrone.vision_landing_il.task.arcdrone import ARCDroneRL_VisionLanding_StudentTeacher

# ===========================================================================
# Config

CFG_DIR = Path(__file__).resolve().parent.parent / "src" / "arcdrone" / "vision_landing_il" / "cfg"
episode_length = 100

# Load config the same way as evaluate.py
initialize_config_dir(config_dir=str(CFG_DIR), job_name="visualize", version_base=None)
cfg = compose(config_name="config")
cfg_env = OmegaConf.to_container(cfg.env, resolve=True)
cfg_env["vision_config"]["nworld"] = 1
cfg_env["naconmax"] = cfg_env["njmax"]  # 1 world

# ===========================================================================
# Env setup

env = ARCDroneRL_VisionLanding_StudentTeacher(cfg=cfg_env)

# from mujoco_playground import wrapper

# env = wrapper.wrap_for_brax_training(
#     env,
#     action_repeat=cfg.train.action_repeat,
#     episode_length=cfg.train.episode_length,
# )

vmap_reset = jax.vmap(env.reset)
vmap_step  = jax.vmap(env.step)
# jit_reset  = jax.jit(vmap_reset)
# jit_step   = jax.jit(vmap_step)
jit_reset  = vmap_reset # For vscode debugger
jit_step   = vmap_step


def squeeze(tree):
    """Remove the leading batch-1 axis."""
    return jax.tree_util.tree_map(lambda x: x[0], tree)

# ===========================================================================
# Rollout

rng = jax.random.PRNGKey(42)
keys = jax.random.split(rng, 1)
state = jit_reset(keys)
rollout = [squeeze(state)]

for _ in range(episode_length):
    action = jp.zeros((1, env._mj_model.nu))
    state = jit_step(state, action)
    rollout.append(squeeze(state))

# ===========================================================================
# Save one video per camera

camera_keys = sorted(k for k in rollout[0].obs.keys() if k.startswith("pixels/view_"))
print(f"Found camera keys: {camera_keys}")

for cam_key in camera_keys:
    # frames = [np.array(r.obs[cam_key][..., 0]) for r in rollout]  # (H, W) per step
    frames = [np.array(r.obs[cam_key][..., -3:]) for r in rollout] # (H, W, 3) per step — take the last 3 frames for RGB visualization
    frames = np.stack(frames)
    # Shift from [-0.5, 0.5] back to [0, 1]
    frames = frames + 0.5
    frames = np.clip(frames, 0.0, 1.0)
    out_name = f"obs_{cam_key.replace('/', '_')}.mp4"
    media.write_video(out_name, frames, fps=5)
    print(f"Saved: {out_name}  shape={frames.shape}  min={frames.min():.3f}  max={frames.max():.3f}")

# Normal render from outer_camera (same style as the original script)
render_frames = env.render(rollout, camera="outer_camera", width=256, height=256)
media.write_video("render.mp4", render_frames, fps=1.0 / env.dt)
print(f"Saved: render.mp4")