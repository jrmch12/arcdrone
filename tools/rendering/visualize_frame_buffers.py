"""Visualize frame buffer contents for each camera.

For each camera (view_0, view_1, view_2) saves a PNG grid showing all
`buffer_size` frames at several time-steps during a rollout. This lets
you quickly check whether the buffer frames are actually different from
each other (i.e. whether the temporal history is informative at the
current sim/ctrl frequency and camera resolution).

Layout of each saved PNG:
  Rows    = time-steps sampled during the episode
  Columns = buffer frames  (newest on the left, oldest on the right)

Usage (from repo root):
    python tools/rendering/visualize_frame_buffers.py

Outputs written to tools/rendering/  (or CWD if run from elsewhere):
    frame_buffer_pixels_view_0.png
    frame_buffer_pixels_view_1.png
    frame_buffer_pixels_view_2.png
    frame_buffer_diff_pixels_view_0.png   ← absolute pixel diff between consecutive frames
    frame_buffer_diff_pixels_view_1.png
    frame_buffer_diff_pixels_view_2.png
"""

import os
import sys
from pathlib import Path

os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"

import jax
import jax.numpy as jp
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("mul", lambda a, b: int(a) * int(b), replace=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from arcdrone.vision_landing_il.task.arcdrone import ARCDroneRL_VisionLanding_StudentTeacher

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CFG_DIR = Path(__file__).resolve().parents[2] / "src" / "arcdrone" / "vision_landing_il" / "cfg"
OUT_DIR  = Path(__file__).resolve().parent

EPISODE_LENGTH = 120   # total steps to simulate
# Time-steps at which we snapshot the buffer (evenly spaced + first & last)
NUM_SNAPSHOTS  = 8
BUFFER_SIZE    = 3     # must match env.cfg.buffer_size

initialize_config_dir(config_dir=str(CFG_DIR), job_name="buf_vis", version_base=None)
cfg     = compose(config_name="config")
cfg_env = OmegaConf.to_container(cfg.env, resolve=True)
cfg_env["vision_config"]["nworld"] = 1
cfg_env["naconmax"]  = cfg_env["njmax"]
cfg_env["buffer_size"] = BUFFER_SIZE

# ---------------------------------------------------------------------------
# Build env
# ---------------------------------------------------------------------------

env = ARCDroneRL_VisionLanding_StudentTeacher(cfg=cfg_env)

vmap_reset = jax.vmap(env.reset)
vmap_step  = jax.vmap(env.step)

rng  = jax.random.PRNGKey(0)
keys = jax.random.split(rng, 1)

print("Compiling reset + step …")
state = vmap_reset(keys)

# ---------------------------------------------------------------------------
# Rollout — collect snapshots
# ---------------------------------------------------------------------------

snapshot_steps = sorted(set(
    [0]
    + list(np.linspace(0, EPISODE_LENGTH - 1, NUM_SNAPSHOTS, dtype=int))
    + [EPISODE_LENGTH - 1]
))
print(f"Snapshotting at steps: {snapshot_steps}")

def squeeze(tree):
    return jax.tree_util.tree_map(lambda x: x[0], tree)

snapshots = {}  # step -> obs dict (squeezed)

for step in range(EPISODE_LENGTH):
    if step in snapshot_steps:
        snapshots[step] = jax.tree_util.tree_map(np.array, squeeze(state).obs)
    action = jp.zeros((1, env._mj_model.nu))
    state  = vmap_step(state, action)

print(f"Collected {len(snapshots)} snapshots.")

# ---------------------------------------------------------------------------
# Build grids
# ---------------------------------------------------------------------------

camera_keys = sorted(k for k in snapshots[0].keys() if k.startswith("pixels/view_"))
print(f"Camera obs keys: {camera_keys}")


def frames_from_stack(stack_hwc, buffer_size):
    """Extract each buffered RGB frame from the stacked (H, W, buffer_size*3) array.

    Returns list of (H, W, 3) uint8 images, newest first.
    `stack_hwc` has channels ordered: [oldest_R, ..., newest_R, newest_G, ..., newest_B]
    but our concatenation in obs.py prepends newest each time, so:
      channels 0..2   → newest frame
      channels 3..5   → one step older
      …
    """
    H, W, C = stack_hwc.shape
    assert C == buffer_size * 3, f"Expected {buffer_size*3} channels, got {C}"
    imgs = []
    for i in range(buffer_size):
        ch_start = i * 3
        frame = stack_hwc[..., ch_start:ch_start + 3]   # (H, W, 3) in [-0.5, 0.5]
        frame = np.clip(frame + 0.5, 0.0, 1.0)
        frame = (frame * 255).astype(np.uint8)
        imgs.append(frame)
    return imgs   # [newest, ..., oldest]


PAD   = 4      # pixels between cells
LABEL_H = 18   # height reserved for text labels

try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
except Exception:
    font = ImageFont.load_default()


for cam_key in camera_keys:
    sorted_steps = sorted(snapshots.keys())
    n_rows = len(sorted_steps)
    n_cols = BUFFER_SIZE

    # Compute cell size from first snapshot
    sample_stack = snapshots[sorted_steps[0]][cam_key]
    H, W = sample_stack.shape[:2]

    cell_w = W + PAD
    cell_h = H + LABEL_H + PAD

    grid_w = n_cols * cell_w + PAD
    grid_h = n_rows * cell_h + PAD

    grid_img  = Image.new("RGB", (grid_w, grid_h), color=(30, 30, 30))
    diff_grid = Image.new("RGB", (grid_w, grid_h), color=(30, 30, 30))
    draw      = ImageDraw.Draw(grid_img)
    draw_d    = ImageDraw.Draw(diff_grid)

    for row, step in enumerate(sorted_steps):
        stack = snapshots[step][cam_key]
        frames = frames_from_stack(stack, BUFFER_SIZE)

        for col, frame in enumerate(frames):
            x = PAD + col * cell_w
            y = PAD + row * cell_h

            # ── RGB frame ──
            img_tile = Image.fromarray(frame)
            grid_img.paste(img_tile, (x, y))
            label = f"t={step:3d} buf[{col}]{'(new)' if col==0 else ''}"
            draw.text((x + 1, y + H + 1), label, fill=(200, 200, 200), font=font)

            # ── Diff vs previous buffer frame (amplified 4×) ──
            if col < BUFFER_SIZE - 1:
                diff = np.abs(frame.astype(np.int16) - frames[col + 1].astype(np.int16))
                diff = np.clip(diff * 4, 0, 255).astype(np.uint8)
            else:
                diff = np.zeros_like(frame)  # no older frame to compare to
            diff_tile = Image.fromarray(diff)
            diff_grid.paste(diff_tile, (x, y))
            draw_d.text((x + 1, y + H + 1), label, fill=(200, 200, 200), font=font)

    out_path  = OUT_DIR / f"frame_buffer_{cam_key.replace('/', '_')}.png"
    out_d_path = OUT_DIR / f"frame_buffer_diff_{cam_key.replace('/', '_')}.png"
    grid_img.save(out_path)
    diff_grid.save(out_d_path)
    print(f"Saved: {out_path}")
    print(f"Saved: {out_d_path}")

    # ── Per-camera summary: mean absolute diff between consecutive buffer frames ──
    diffs = []
    for step in sorted_steps:
        stack = snapshots[step][cam_key]
        frames = frames_from_stack(stack, BUFFER_SIZE)
        for i in range(len(frames) - 1):
            d = np.abs(frames[i].astype(np.float32) - frames[i + 1].astype(np.float32)).mean()
            diffs.append(d)
    print(f"  {cam_key} — mean |buf[i] - buf[i+1]|: {np.mean(diffs):.3f}  "
          f"(0 = identical, 255 = max diff)")

print("\nDone. Open the PNGs to inspect buffer diversity.")
print("  Rows   = episode time-steps")
print("  Cols   = buffer slots (left=newest, right=oldest)")
print("  _diff_ = amplified absolute pixel diff vs. previous buffer slot")
