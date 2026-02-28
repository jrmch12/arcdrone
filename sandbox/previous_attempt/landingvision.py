# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Cartpole environment."""

from typing import Any, Dict, Optional, Union
import warnings
from pathlib import Path

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np
from omegaconf import DictConfig, OmegaConf

from mujoco_playground._src import mjx_env
from mujoco_playground._src import reward
from mujoco_playground._src.dm_control_suite import common
from .obs import _get_obs_impl
from .reward import _get_reward_impl
from .target import _get_target_impl

import arcdrone.paths as paths

_XML_PATH = paths.ASSETS_DIR / "arc_tinywhoop" / "scene.xml"
_TASK_CONFIG_PATH = paths.CONFIGS_DIR / "landing_vision.yaml"


def default_config() -> config_dict.ConfigDict:
  cfg = OmegaConf.load(_TASK_CONFIG_PATH)
  env_cfg = OmegaConf.to_container(cfg.env, resolve=True)
  return config_dict.ConfigDict(env_cfg)


def _rgba_to_grayscale(rgba: jax.Array) -> jax.Array:
  """
  Intensity-weigh the colors.
  This expects the input to have the channels in the last dim.
  """
  r, g, b = rgba[..., 0], rgba[..., 1], rgba[..., 2]
  gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
  return gray


class LandingVision(mjx_env.MjxEnv):
  """Landing vision environment."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    # Config overrides are managed by Hydra flags, so this is intentionally unused.
    super().__init__(config, config_overrides=config_overrides)
    self._vision = self._config.vision

    self._xml_path = _XML_PATH.as_posix()
    # self._model_assets = common.get_assets()        
    # self._mj_model = mujoco.MjModel.from_xml_string(
    #     _XML_PATH.read_text(), self._model_assets
    # )     # I dont think there is an advantage of doing it this way.
    self._mj_model = mujoco.MjModel.from_xml_path(self._xml_path)
    self._mj_model.opt.timestep = self._config.opt.timestep
    solver_map = {
        "cg": mujoco.mjtSolver.mjSOL_CG,
        "newton": mujoco.mjtSolver.mjSOL_NEWTON,
    } # add more if needed
    solver_name = str(self._config.opt.mj_solver).lower()
    self._mj_model.opt.solver = solver_map[solver_name]
    self._mj_model.opt.iterations = int(self._config.opt.mj_iterations)
    self._mj_model.opt.ls_iterations = int(self._config.opt.mj_ls_iterations)
    self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)

    if self._vision:
      try:
        # pylint: disable=import-outside-toplevel
        from madrona_mjx.renderer import BatchRenderer  # pytype: disable=import-error
      except ImportError:
        warnings.warn("Madrona MJX not installed. Cannot use vision with.")
        return
      self.renderer = BatchRenderer(
          m=self._mjx_model,
          gpu_id=self._config.vision_config.gpu_id,
          num_worlds=self._config.vision_config.render_batch_size,
          batch_render_view_width=self._config.vision_config.render_width,
          batch_render_view_height=self._config.vision_config.render_height,
          enabled_geom_groups=np.asarray(
              self._config.vision_config.enabled_geom_groups
          ),
          enabled_cameras=np.asarray([
              0,
          ]),
          add_cam_debug_geo=False,
          use_rasterizer=self._config.vision_config.use_rasterizer,
          viz_gpu_hdls=None,
      )

  def reset(self, rng: jp.ndarray) -> mjx_env.State:
      """Resets the environment to an initial state and samples a new goal."""

      rng, key = jax.random.split(rng)
      qpos, qvel = self._sample_initial_state(key) 
      data = mjx_env.make_data(
          self.mj_model,
          qpos=qpos,
          qvel=qvel,
          impl=self.mjx_model.impl.value,
          nconmax=self._config.nconmax,
          njmax=self._config.njmax,
      )
      data = mjx.forward(self.mjx_model, data)

      state = mjx_env.State(
          data= data, 
          obs= jp.array(0.0), # overwritten to the right shape bellow
          reward=jp.array(0.0),
          done=jp.array(0.0),
          metrics=self._initialize_metrics(),
          info=self._initialize_info(data, rng)
      )

      state = self._get_target(state)
      zero_action = jp.zeros(self.sys.nu)         # initialize with zero action
      state = self._get_obs(state, zero_action)
      
      if self._vision:
        render_token, rgb, _ = self.renderer.init(data, self._mjx_model)
        state.info.update({"render_token": render_token})
        obs = _rgba_to_grayscale(rgb[0].astype(jp.float32)) / 255.0
        obs_history = jp.tile(obs, (self._config.vision_config.history, 1, 1))
        state.info.update({"obs_history": obs_history})
        obs = {"pixels/view_0": obs_history.transpose(1, 2, 0)}

      return state
  

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:

      # 0. Scale the action
      action_normalized = action
      action = self.scale_action(action)
      
      # 1. Update physics with scaled action
      data = mjx_env.step(self.mjx_model, state.data, action, self.n_substeps)
      state = state.replace(data=data)
      
      # 2. Update target
      state = self._get_target(state)
      
      # 3. Compute observations
      state = self._get_obs(state, action)
      
      # 4. Compute reward
      state = self._get_reward(state, action_normalized)
      
      # 5. Check termination
      state = self._check_termination(state)

      if self._vision:
        _, rgb, _ = self.renderer.render(state.info["render_token"], data)
        # Update observation buffer
        obs_history = state.info["obs_history"]
        obs_history = jp.roll(obs_history, 1, axis=0)
        obs_history = obs_history.at[0].set(
            _rgba_to_grayscale(rgb[0].astype(jp.float32)) / 255.0
        )
        state.info["obs_history"] = obs_history
        obs = {"pixels/view_0": obs_history.transpose(1, 2, 0)}
      
      return state


  def _get_obs(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    return _get_obs_impl(self, state, action)

  def _get_reward(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    """Calculate reward from current state."""
    return _get_reward_impl(self, state, action)

  def _get_target(self, state: mjx_env.State) -> mjx_env.State:
    """Update target velocity."""
    return _get_target_impl(self, state)
  
  def scale_action(self, action_normalized):
      """Scale action from [-1, 1] to actuator control range [ctrl_min, ctrl_max]."""
      action_normalized = jp.array(action_normalized)
      action_normalized = jp.clip(action_normalized, -1.0, 1.0)
      return (action_normalized + 1.0) / 2.0 * (self.ctrl_max - self.ctrl_min) + self.ctrl_min

  def _initialize_info(self, data, rng):
      """Initialize state variables """

      # TODO: maybe is better not to train while the buffer is not valid? 
      # or do some soft masking adding one observation to let know the buffer is not full yet?

      quat_init = data.sensordata[0:4]     
      angvel_init = data.sensordata[4:7]  
      linacc_init = data.sensordata[7:10] 
      linvel_init = data.sensordata[10:13]  
      
      return {
          'step': 0,
          'rng': rng,
          'goal_achieved': jp.array(0.0),
          'steps_within_success': jp.array(0),
          
          # Buffers - initialized with current state repeated
          'action_buffer': jp.tile(jp.zeros(self.sys.nu), (self.cfg.buffer_size, 1)),
          'target_vel_buffer': jp.tile(jp.zeros(3), (self.cfg.buffer_size, 1)),
          'linacc_buffer': jp.tile(linacc_init, (self.cfg.buffer_size, 1)),
          'quat_buffer': jp.tile(quat_init, (self.cfg.buffer_size, 1)),
          'angvel_buffer': jp.tile(angvel_init, (self.cfg.buffer_size, 1)),
          'linvel_buffer': jp.tile(linvel_init, (self.cfg.buffer_size, 1)),
          'linacc_buffer_noisy': jp.tile(linacc_init, (self.cfg.buffer_size, 1)),
          'quat_buffer_noisy': jp.tile(quat_init, (self.cfg.buffer_size, 1)),
          'angvel_buffer_noisy': jp.tile(angvel_init, (self.cfg.buffer_size, 1)),
          'linvel_buffer_noisy': jp.tile(linvel_init, (self.cfg.buffer_size, 1)),
      }

  def _initialize_metrics(self):

      return {
          'reward_distance': jp.float32(0.0),
          'reward_time_penalty': jp.float32(0.0),
          'reward_overshoot': jp.float32(0.0),
          'reward_oscillation': jp.float32(0.0),
          'reward_action_chattering': jp.float32(0.0),
          'reward_action_penalty': jp.float32(0.0),
          'reward_success_bonus': jp.float32(0.0),
          'reward_ground_penalty': jp.float32(0.0),
          'reward_total': jp.float32(0.0),
      }

  @property
  def xml_path(self) -> str:
    return self._xml_path

  @property
  def action_size(self) -> int:
    return self.mjx_model.nu

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model
