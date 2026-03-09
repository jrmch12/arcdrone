import jax
from jax import numpy as jp
import mujoco
from mujoco import mjx
from ml_collections import config_dict
from omegaconf import DictConfig, OmegaConf
from flax import struct

# MjxEnv base
from mujoco_playground._src import mjx_env

# RL helper functions
from .obs import _get_obs_impl
from .reward import _get_reward_impl  
from .reward import _check_episode_events_impl  
from .target import _get_target_impl




# @struct.dataclass
# class CustomState:
#     # Compatibility placeholder: we will store extra mutable variables inside state.info
#     pass



#======== Main class ================

class ARCDroneRL_Landing(mjx_env.MjxEnv):

    """ARC Drone Controller RL Environment."""

    def __init__(self, *, cfg: DictConfig = None, config_overrides=None, **kwargs):
        # Accept OmegaConf DictConfig or plain dict and convert to ml_collections.ConfigDict
        if isinstance(cfg, DictConfig):
            cfg = OmegaConf.to_container(cfg, resolve=True)
        if isinstance(cfg, dict):
            cfg = config_dict.ConfigDict(cfg)

        # Nested env-specific settings under `env`; extract if present.
        env_cfg = cfg.env if hasattr(cfg, "env") else cfg

        # Call parent initializer (MjxEnv expects the env ConfigDict)
        super().__init__(env_cfg, config_overrides=config_overrides)

        # store environment config for convenience (keys like max_episode_steps, buffer_size)
        self.cfg = env_cfg # TODO: use self._config comming from super.init instead?

        # Build MJ model and put mjx model
        self._mj_model = mujoco.MjModel.from_xml_path(self.cfg.xml_path_rel)
        self._mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        self._mj_model.opt.iterations = int(self.cfg.get('mj_iterations', 6))
        self._mj_model.opt.ls_iterations = int(self.cfg.get('mj_ls_iterations', 6))
        self._mj_model.opt.timestep = self.sim_dt
        # Create mjx model for fast simulation
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)

        # Control ranges
        self.ctrl_min = jp.array(self._mj_model.actuator_ctrlrange[:, 0])
        self.ctrl_max = jp.array(self._mj_model.actuator_ctrlrange[:, 1])



    # Main methods =================================================================

    def reset(self, rng: jp.ndarray) -> mjx_env.State:
        """Resets the environment to an initial state and samples a new goal."""

        rng, key = jax.random.split(rng)
        qpos, qvel = self._sample_initial_state(key)

        data = mjx_env.make_data(
            self._mj_model,
            qpos=qpos,
            qvel=qvel,
            impl=self._mjx_model.impl.value,
            nconmax=self._config.nconmax,
            njmax=self._config.njmax,
        )
        data = mjx.forward(self._mjx_model, data)

        info = self._initialize_state_vars(data, rng)

        state = mjx_env.State(
            data=data,
            obs=jp.array(0.0),
            reward=jp.array(0.0),
            done=jp.array(0.0),
            metrics=self._initialize_metrics(),
            info=info,
        )

        state = self._get_target(state)
        zero_action = jp.zeros(self._mjx_model.nu)
        state = self._get_obs(state, zero_action)

        return state
    

    def step(self, state: mjx_env.State, action: jp.ndarray) -> mjx_env.State:
        """Runs one timestep of the environment's dynamics."""

        # 0. Scale the action   # TODO: check if mjxenvs does not handle this already?
        action_normalized = action
        action = self.scale_action(action)

        # 1. Update physics with scaled action
        data = mjx_env.step(self._mjx_model, state.data, action, self.n_substeps)
        state = state.replace(data=data)

        # 2. Update target
        state = self._get_target(state)

        # 3. Compute observations
        state = self._get_obs(state, action)

        # 5. Check termination
        state = self._check_episode_events(state)

        # 4. Compute reward
        state = self._get_reward(state, action_normalized)

        

        return state

    # Helper functions =============================================================
    
    def _sample_initial_state(self, rng: jp.ndarray):
        """Sample initial position and velocity for the drone.
        
        Drone starts at 1.5m above ground with zero velocity.
        Position can be slightly randomized if configured.
        
        Args:
            rng: Random key for sampling
            
        Returns:
            qpos: (7,) array [x, y, z, qw, qx, qy, qz] - position and orientation quaternion
            qvel: (6,) array [vx, vy, vz, wx, wy, wz] - linear and angular velocity
        """
        # Initial position: 1.5m above ground (z=1.5)
        # Base qpos layout for a free joint + quaternion: [x, y, z, qw, qx, qy, qz]
        position = jp.array([0.0, 3.0, 1.5])
        quaternion = jp.array([1.0, 0.0, 0.0, 0.0])

        # Build qpos with correct length (model may define additional joints, e.g. camera tilt)
        nq = int(self._mjx_model.nq)
        qpos = jp.zeros(nq)
        # fill base pose (first 7 entries)
        qpos = qpos.at[0:3].set(position)
        qpos = qpos.at[3:7].set(quaternion)

        # If the model has extra joint coordinates (e.g. hinge for camera), leave them at zero

        # Build qvel with correct length
        nv = int(self._mjx_model.nv)
        qvel = jp.zeros(nv)

        return qpos, qvel


    def _initialize_state_vars(self, data, rng):
        """Initialize state variables and info dictionary"""

        # TODO: maybe is better not to train while the buffer is not valid? 
        # or do some soft masking adding one observation to let know the buffer is not full yet?

        quat_init = data.sensordata[0:4]
        angvel_init = data.sensordata[4:7]
        linacc_init = data.sensordata[7:10]
        linvel_init = data.sensordata[10:13]
        pos_init = data.qpos[0:3]
        
        return {
            'step': 0,
            'rng': rng,
            'goal_achieved': jp.array(0.0),
            'steps_within_success': jp.array(0),
            
            # Buffers - initialized with current state repeated
            'action_buffer': jp.tile(jp.zeros(self._mjx_model.nu), (self.cfg.buffer_size, 1)),
            'target_buffer': jp.tile(jp.zeros(3), (self.cfg.buffer_size, 1)),
            'linacc_buffer': jp.tile(linacc_init, (self.cfg.buffer_size, 1)),
            'quat_buffer': jp.tile(quat_init, (self.cfg.buffer_size, 1)),
            'angvel_buffer': jp.tile(angvel_init, (self.cfg.buffer_size, 1)),
            'linvel_buffer': jp.tile(linvel_init, (self.cfg.buffer_size, 1)),
            'pos_buffer': jp.tile(pos_init, (self.cfg.buffer_size, 1)),
            'linacc_buffer_noisy': jp.tile(linacc_init, (self.cfg.buffer_size, 1)),
            'quat_buffer_noisy': jp.tile(quat_init, (self.cfg.buffer_size, 1)),
            'angvel_buffer_noisy': jp.tile(angvel_init, (self.cfg.buffer_size, 1)),
            'linvel_buffer_noisy': jp.tile(linvel_init, (self.cfg.buffer_size, 1)),
            # initialize ground_violation to match keys expected by reward/termination
            'ground_violation': jp.array(0.0),
        }

    def _initialize_metrics(self):

        return {
            'reward_distance': jp.float32(0.0),
            'reward_time_penalty': jp.float32(0.0),
            'reward_overshoot': jp.float32(0.0),
            'reward_oscillation': jp.float32(0.0),
            'reward_action_chattering': jp.float32(0.0),
            'reward_action_penalty': jp.float32(0.0),
            'reward_attitude_level': jp.float32(0.0),
            'reward_low_angvel': jp.float32(0.0),
            'reward_low_linvel': jp.float32(0.0),
            'reward_soft_landing': jp.float32(0.0),
            'reward_success_bonus': jp.float32(0.0),
            'reward_ground_penalty': jp.float32(0.0),
            'reward_total': jp.float32(0.0),
        }

    def _get_obs(self, state: mjx_env.State, action: jp.ndarray) -> mjx_env.State:
        return _get_obs_impl(self, state, action)
    
    def _check_episode_events(self, state: mjx_env.State) -> mjx_env.State:
        """Check episode events from current state."""
        return _check_episode_events_impl(self, state)
        
    def _get_reward(self, state: mjx_env.State, action: jp.ndarray) -> mjx_env.State:
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
