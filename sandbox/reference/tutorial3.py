"""
Docstring for tutorial

This tutorial follows training_vision_1.ipynb from mujoco playground.

Training!

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
from mujoco_playground import dm_control_suite
from mujoco_playground import wrapper
# @title Import MuJoCo, MJX, and Brax
from datetime import datetime
import functools
import os
from brax.training.agents.ppo import networks_vision as ppo_networks_vision
from brax.training.agents.ppo import train as ppo
from IPython.display import clear_output
from matplotlib import pyplot as plt
import numpy as np
from mujoco_playground import wrapper
np.set_printoptions(precision=3, suppress=True, linewidth=100)
from mujoco_playground import dm_control_suite


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

# jit_reset = jax.jit(env.reset)
# jit_step = jax.jit(env.step)



from mujoco_playground.config import dm_control_suite_params

# Load vision-specific PPO configuration tuned for CartpoleBalance
ppo_params = dm_control_suite_params.brax_vision_ppo_config(env_name)
ppo_params.episode_length = episode_length
ppo_params.network_factory = ppo_networks_vision.make_ppo_networks_vision

x_data, y_data, y_dataerr = [], [], []
times = [datetime.now()]


def progress(num_steps, metrics):
  clear_output(wait=True)

  times.append(datetime.now())
  x_data.append(num_steps)
  y_data.append(metrics["eval/episode_reward"])
  y_dataerr.append(metrics["eval/episode_reward_std"])

  plt.xlim([0, ppo_params["num_timesteps"] * 1.25])
  plt.ylim([0, 100])
  plt.xlabel("# environment steps")
  plt.ylabel("reward per episode")
  plt.title(f"y={y_data[-1]:.3f}")
  plt.errorbar(x_data, y_data, yerr=y_dataerr, color="blue")

  plt.savefig("training_progress.png")
  plt.close()


train_fn = functools.partial(
    ppo.train, **dict(ppo_params), progress_fn=progress
)
make_inference_fn, params, metrics = train_fn(environment=env)
print(f"time to jit: {times[1] - times[0]}")
print(f"time to train: {times[-1] - times[1]}")

# Save trained model parameters
import pickle
model_path = "trained_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(params, f)
print(f"Model parameters saved to {model_path}") 