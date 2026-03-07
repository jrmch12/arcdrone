"""
Minimal reset-only diagnostic — no PPO, no double-vmap, no wrapper chain.

Steps it covers:
  1. Instantiates ARCDroneRL_VisionLanding raw (no brax wrapping)
  2. Runs a vmapped reset over num_envs worlds
  3. Checks obs keys & shapes vs what brax PPO expects
  4. Tries the exact ops that fail in PPO (jnp.zeros from obs shapes)

Run with:
    python debug_vision_reset.py
"""

import os

os.environ["MADRONA_MWGPU_KERNEL_CACHE"] = (
    "/home/jrmch12f/Documents/code/borrador_braxenvs/madrona_cache"
)
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.60"
os.environ["MADRONA_MWGPU_DEVICE_HEAP_SIZE"] = "1073741824"

import jax
import jax.numpy as jnp
from ml_collections import config_dict

print("JAX devices:", jax.devices())

# ── Minimal env config ────────────────────────────────────────────────────────
# render_batch_size MUST equal NUM_ENVS — Madrona is compiled for exactly this many worlds.
# When the vmapped reset runs, the inner vmap batches exactly NUM_ENVS renderer.init calls
# which Madrona collapses into one call. Size mismatch → CUDA invalid argument.
NUM_ENVS = 16  # keep small but MATCH render_batch_size below

cfg = config_dict.create(
    xml_path_rel="assets/skydio_x2/scene.xml",
    impl="jax",
    sim_dt=0.004,
    ctrl_dt=0.02,
    action_repeat=1,
    nconmax=0,
    njmax=2,
    vision=True,
    buffer_size=5,
    max_episode_steps=200,
    success_threshold=0.02,
    success_steps_required=10,
    distance_weight=1,
    distance_scale=0.3,
    overshoot_penalty=5,
    oscillation_penalty=2,
    action_chattering_weight=0.001,
    action_penalty_weight=0.0001,
    time_penalty=0.01,
    success_bonus=500.0,
    ground_penalty_weight=10.0,
    ground_threshold_event=0.15,
    ground_threshold_penalty=0.5,
    vision_config=config_dict.create(
        gpu_id=0,
        render_batch_size=NUM_ENVS,  # MUST equal NUM_ENVS
        render_width=64,
        render_height=64,
        enabled_geom_groups=[0, 1, 2],
        use_rasterizer=False,
        history=5,
    ),
)

print(f"\n[1] Instantiating environment (num_worlds={NUM_ENVS}) ...")
from arcdrone import ARCDroneRL_VisionLanding
env = ARCDroneRL_VisionLanding(cfg=cfg)
print(f"    action_size = {env.action_size}")

# ── STEP 1: bare single reset (no vmap) ───────────────────────────────────────
print("\n[2] Running single (non-vmapped) reset ...")
rng = jax.random.PRNGKey(0)
jit_reset = jax.jit(env.reset)
state = jit_reset(rng)
jax.block_until_ready(state)   # force GPU sync to surface any deferred errors
print("    OK")
print("    obs keys:", list(state.obs.keys()))
for k, v in state.obs.items():
    print(f"    obs[{k!r}].shape = {v.shape}")

# ── STEP 2: vmapped reset (mimics BraxDomainRandomizationVmapWrapper) ─────────
print(f"\n[3] Running vmapped reset over {NUM_ENVS} envs ...")
rngs = jax.random.split(jax.random.PRNGKey(1), NUM_ENVS)
vmapped_reset = jax.jit(jax.vmap(env.reset))
batch_state = vmapped_reset(rngs)
jax.block_until_ready(batch_state)  # surface deferred CUDA errors HERE
print("    OK")
for k, v in batch_state.obs.items():
    print(f"    batch obs[{k!r}].shape = {v.shape}")

# ── STEP 3: test through the mujoco_playground wrapper chain ──────────────────
# This is the ACTUAL path used during PPO training.
print("\n[4] Testing wrapped env (wrap_for_brax_training) as PPO does it ...")
from mujoco_playground import wrapper

# Re-create env so render_batch_size is fresh (renderer is stateful)
print("    Re-instantiating env for wrapper test ...")
env_wrapped_raw = ARCDroneRL_VisionLanding(cfg=cfg)
env_w = wrapper.wrap_for_brax_training(
    env_wrapped_raw,
    vision=True,
    num_vision_envs=NUM_ENVS,
    action_repeat=1,
    episode_length=200,
)
print("    Wrapped env ready")

# PPO key_envs: (local_devices_to_use=1, num_envs, 2)
key_envs = jax.random.split(jax.random.PRNGKey(2), NUM_ENVS)
key_envs = jnp.reshape(key_envs, (1, NUM_ENVS) + key_envs.shape[1:])  # (1, N, 2)

print(f"    Calling jax.jit(jax.vmap(env_w.reset))(key_envs) with key_envs.shape={key_envs.shape} ...")
reset_fn = jax.jit(jax.vmap(env_w.reset))
env_state = reset_fn(key_envs)
jax.block_until_ready(env_state)  # surface any deferred CUDA errors
print("    OK — env_state.obs shapes:")
for k, v in env_state.obs.items():
    print(f"    env_state.obs[{k!r}].shape = {v.shape}")

obs_shape = jax.tree_util.tree_map(lambda x: x.shape[2:], env_state.obs)
print("    obs_shape (after stripping 2 batch axes):")
for k, shape in obs_shape.items():
    print(f"    obs_shape[{k!r}] = {shape}")

# ── STEP 4: replicate the exact line that crashes in PPO ──────────────────────
print("\n[5] Creating dummy_obs exactly as brax networks.py does ...")
for k, shape in obs_shape.items():
    print(f"    jnp.zeros((1,) + {shape}) for key={k!r} ...")
    arr = jnp.zeros((1,) + shape)
    jax.block_until_ready(arr)
    print(f"    -> shape {arr.shape}  OK")

print("\n=== All steps passed ===")
