"""
Docstring for tutorial

Minimal env rendering. Visualize the camera observations. Check proper Madrona GPU allocation.

"""

# @title Import MuJoCo, MJX, and Brax
import os
from pathlib import Path
import time
import jax
from jax import numpy as jp
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=100)
from mujoco_playground import wrapper
from hydra import compose, initialize_config_dir



CFG_DIR = '/home/jrmch12f/Documents/code/borrador_braxenvs/src/arcdrone/controller/cfg'
# Env import:
from arcdrone.controller.rl.task.vision_mode.arcdrone import ARCDroneRL_VisionLanding
task_name = "vision"


# ===========================================================================


# On your second reading, load the compiled rendering backend to save time!
# os.environ["MADRONA_MWGPU_KERNEL_CACHE"] = "<YOUR_PATH>/madrona_mjx/build/cache"
os.environ["MADRONA_MWGPU_KERNEL_CACHE"] = "/home/jrmch12f/Documents/code/borrador_braxenvs/madrona_cache"

# Coordinate between Jax and the Madrona rendering backend
def limit_jax_mem(limit):
  os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"{limit:.2f}"


limit_jax_mem(0.6)
# Reduce madrona memory allocation to 1GB as cartpole doesn't need much
os.environ["MADRONA_MWGPU_DEVICE_HEAP_SIZE"] = "1073741824"

# ===========================================================================

# For dm envs
num_envs = 1024
ctrl_dt = 0.04
episode_length = int(3 / ctrl_dt)

config_overrides = {
    "vision": True,
    "vision_config.render_batch_size": num_envs,
    "action_repeat": 1,
    "ctrl_dt": ctrl_dt,
    "episode_length": episode_length,
    "vision_config.use_rasterizer": False,  # MJO it was on default (False)
    "vision_config.render_width": 64,       # MJO it was on default (64)
    "vision_config.render_height": 64,      # MJO it was on default (64)
}

from mujoco_playground import dm_control_suite
env_name = "CartpoleBalance"
env = dm_control_suite.load(
    env_name, config_overrides=config_overrides
)
env = wrapper.wrap_for_brax_training(
    env,
    vision=True,
    num_vision_envs=num_envs,
    action_repeat=1,
    episode_length=episode_length,
)

# ===========================================================================

# # For our hydra pipeline
# initialize_config_dir(
#     config_dir=str(CFG_DIR), job_name="sitt_evaluate", version_base=None
# )
# cfg = compose(config_name="config", overrides=[f"task={task_name}"])
# cfg_env = cfg.env
# cfg_train = cfg.train

# env = ARCDroneRL_VisionLanding(cfg=cfg_env)
# env = wrapper.wrap_for_brax_training(
#     env,
#     vision=True,
#     num_vision_envs=cfg_train.num_envs,
#     action_repeat=cfg_train.action_repeat,
#     episode_length=cfg_train.episode_length,
# )
# num_envs = cfg_train.num_envs

# ===========================================================================


jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)

# ===========================================================================

key_reset, key_act = jax.random.split(jax.random.PRNGKey(0))
state = jit_reset(jax.random.split(key_reset, num_envs)) # to debug: state = env.reset(jax.random.split(key_reset, num_envs))

# Pre-compile
jit_step = jit_step.lower(
    state, jp.zeros((num_envs, env.action_size))
).compile()

t0 = time.time()

N = 1000
for i in range(N):
  act = jax.random.uniform(
      key_act, (num_envs, env.action_size), minval=-1.0, maxval=1.0
  )
  state = jit_step(state, act)

jax.tree_util.tree_map(
    lambda x: x.block_until_ready(), state
)  # Await device completion
dt = time.time() - t0

print("Madrona MJX: {:d} transitions per second".format(int(N * num_envs / dt)))


