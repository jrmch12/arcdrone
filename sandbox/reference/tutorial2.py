"""
Docstring for tutorial

This tutorial follows training_vision_1.ipynb from mujoco playground.
We just visualize the env, using only the first agent's state for visualisation. We can see the rgb camera,
as well as the grayscale version, which will be use for training.

"""
import os

# On your second reading, load the compiled rendering backend to save time!
# os.environ["MADRONA_MWGPU_KERNEL_CACHE"] = "<YOUR_PATH>/madrona_mjx/build/cache"
os.environ["MADRONA_MWGPU_KERNEL_CACHE"] = "/home/jrmch12f/Documents/code/madmjx_projects/madrona_cache"

# Coordinate between Jax and the Madrona rendering backend
def limit_jax_mem(limit):
  os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"{limit:.2f}"


limit_jax_mem(0.6)
# Reduce madrona memory allocation to 1GB as cartpole doesn't need much
os.environ["MADRONA_MWGPU_DEVICE_HEAP_SIZE"] = "1073741824"

import jax
from jax import numpy as jp
import mediapy as media
from mujoco_playground import dm_control_suite
from mujoco_playground import wrapper




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

jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)



# ===========================================================================
# Visualize env 

def unvmap(x):
  return jax.tree.map(lambda y: y[0], x)


state = jit_reset(jax.random.split(jax.random.PRNGKey(0), num_envs))
rollout = [unvmap(state)]

f = 0.2
for i in range(episode_length):
  action = []
  for j in range(env.action_size):
    action.append(
        jp.sin(
            unvmap(state).data.time * 2 * jp.pi * f
            + j * 2 * jp.pi / env.action_size
        )
    )
  action = jp.tile(jp.array(action), (num_envs, 1))
  state = jit_step(state, action)
  rollout.append(unvmap(state))

frames = env.render(rollout, camera="fixed", width=256, height=256)
k = next(iter(rollout[0].obs.items()), None)[0]  # ex: pixels/view_0
obs = [r.obs[k][..., 0] for r in rollout]  # visualise first channel
# media.show_videos([frames, obs], fps=1.0 / env.dt, height=256)
media.write_video("render.mp4", frames, fps=1.0 / env.dt)
media.write_video("obs.mp4", obs, fps=1.0 / env.dt)