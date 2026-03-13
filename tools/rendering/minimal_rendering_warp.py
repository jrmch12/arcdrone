import os
import numpy as np
import warp as wp
import mujoco
import mujoco_warp as mjw
from PIL import Image

# ── config ────────────────────────────────────────────────────────────────────
XML_PATH = "./assets/skydio_x2/scene.xml"   # <-- point this to your xml
NWORLD   = 16            # number of parallel worlds (must be a perfect square)
CAM_RES  = (256, 256) # (width, height) per world
GRID     = 4             # sqrt(NWORLD) — grid is GRID x GRID
OUTPUT   = "render.png"
# ─────────────────────────────────────────────────────────────────────────────

# EGL setup — needed on headless machines (no monitor attached)
os.environ["MUJOCO_GL"] = "egl"
wp.config.quiet = True

# Load model from your xml file
mj_model = mujoco.MjModel.from_xml_path(XML_PATH)

# Put model and data on GPU via Warp
model = mjw.put_model(mj_model)
data  = mjw.make_data(mj_model, nworld=NWORLD)

# Create render context — allocates GPU buffers for RGB and depth output
rc = mjw.create_render_context(
    mj_model,
    nworld=NWORLD,
    cam_res=CAM_RES,
    render_rgb=True,
    render_depth=False,
)

# Override background color: warp doesn't support MuJoCo skybox textures,
# so rays that miss geometry get filled with this color instead.
from mujoco_warp._src.render_util import pack_rgba_to_uint32
rc.background_color = pack_rgba_to_uint32(
    0.5 * 255.0, 0.7 * 255.0, 0.95 * 255.0, 1.0 * 255.0  # sky blue matching skybox gradient  # The warp function needs float arguments
)

# Randomize qpos across worlds so each world looks different
qpos = data.qpos.numpy()
for i in range(qpos.shape[1]):          # randomize every joint
    qpos[:, i] = np.random.uniform(-1, 1, size=NWORLD)
data.qpos = wp.array(qpos, dtype=wp.float32)

# Forward kinematics — populate all derived data fields from qpos
mjw.forward(model, data)

# Refit BVH — update collision tree to match new positions, required before render
mjw.refit_bvh(model, data, rc)

# Render all worlds in parallel on GPU
mjw.render(model, data, rc)

# Allocate output buffer and pull RGB from render context
rgb_data = wp.zeros((NWORLD, CAM_RES[1], CAM_RES[0]), dtype=wp.vec3)
mjw.get_rgb(rc, camera_index=1, rgb_out=rgb_data)

# Copy from GPU → CPU
pixels = rgb_data.numpy()               # (NWORLD, H, W, 3)

# Tile into a GRID x GRID image
pixels = pixels.reshape(GRID, GRID, CAM_RES[1], CAM_RES[0], 3)
pixels = pixels.transpose(0, 2, 1, 3, 4)
pixels = pixels.reshape(GRID * CAM_RES[1], GRID * CAM_RES[0], 3)

# Convert float [0,1] → uint8 [0,255] and save
pixels_uint8 = (np.clip(pixels, 0, 1) * 255).astype(np.uint8)
Image.fromarray(pixels_uint8).save(OUTPUT)
print(f"Saved {GRID}x{GRID} grid of worlds to: {OUTPUT}")