import jax
from jax import numpy as jp
import mujoco
from mujoco import mjx
from ml_collections import config_dict
from omegaconf import DictConfig, OmegaConf

# MjxEnv base
from mujoco_playground._src import mjx_env

# RL helper functions
from .obs import _get_obs_impl
from .reward import _get_reward_impl  
from .reward import _check_episode_events_impl  
from .target import _get_target_impl

import sys as _sys
from pathlib import Path as _Path
_NA2_ROOT = str(_Path(__file__).resolve().parents[1])
if _NA2_ROOT not in _sys.path:
    _sys.path.insert(0, _NA2_ROOT)
from utils.math_utils import euler_to_quaternion



# @struct.dataclass
# class CustomState:
#     # Compatibility placeholder: we will store extra mutable variables inside state.info
#     pass



#======== Main class ================

class ARCDroneVisionLandingIL(mjx_env.MjxEnv):

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

        # Vision setup (warp-based rendering pipeline).
        # This flag is static for a given run, so python-side branching is safe.
        self._vision_enabled = bool(getattr(self.cfg, "enable_vision_obs", self.cfg.vision))
        self._grayscale_obs = bool(getattr(self.cfg, "grayscale_obs", True))
        
        if self._vision_enabled:
            vision_kwargs = self.cfg.vision_config.to_dict()
            # OmegaConf/YAML always deserializes as lists; warp API needs tuples
            for k in ('cam_res', 'render_rgb', 'render_depth', 'cam_active'):
                if k in vision_kwargs and isinstance(vision_kwargs[k], list):
                    vision_kwargs[k] = tuple(vision_kwargs[k])
            self._rc = mjx.create_render_context(
                mjm=self._mj_model,
                **vision_kwargs,
            )
            # Warp doesn't support MuJoCo skybox textures — rays that miss geometry
            # are filled with background_color. Override to match the skybox gradient.
            from mujoco_warp._src.render_util import pack_rgba_to_uint32
            _sky_bg = pack_rgba_to_uint32(0.5 * 255.0, 0.7 * 255.0, 0.95 * 255.0, 1.0 * 255.0)
            for _warp_ctx in self._rc._contexts.values():
                _warp_ctx.background_color = _sky_bg
            self._rc_pytree = self._rc.pytree()
        else:
            self._rc = None
            self._rc_pytree = None

        cam_res = self.cfg.vision_config.cam_res
        if not self._vision_enabled:
            cam_res = (8, 8)
        cam_h, cam_w = int(cam_res[0]), int(cam_res[1])
        self._pixel_channels_per_frame = 1 if self._grayscale_obs else 3
        self._pixel_stack_shape = (
            cam_h,
            cam_w,
            self._pixel_channels_per_frame * int(self.cfg.buffer_size_pixels),
        )
        self._dummy_frame_stack = jp.zeros(self._pixel_stack_shape, dtype=jp.float32)

    def _format_frame(self, rgb):
        if self._grayscale_obs:
            return jp.mean(rgb, axis=-1, keepdims=True) - 0.5
        return rgb - 0.5


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
            naconmax=self._config.naconmax,
            njmax=self._config.njmax,
        )
        data = mjx.forward(self._mjx_model, data)

        info = self._initialize_state_vars(data, rng)

        if self._vision_enabled:
            # Vision: render initial frame from the mounted camera
            render_data = mjx.refit_bvh(self.mjx_model, data, self._rc_pytree)
            out = mjx.render(self.mjx_model, render_data, self._rc_pytree)

            frame0 = self._format_frame(mjx.get_rgb(self._rc_pytree, 0, out[0]))
            frame_stack_0 = jp.tile(frame0, (1, 1, self.cfg.buffer_size_pixels))
        else:
            frame_stack_0 = self._dummy_frame_stack

        info = {
            **info,
            "frame_stack_0": frame_stack_0,
            "frame_skip_counter": jp.int32(0),
        }

        # Build initial obs dict (flat structure, pixels/view_* keys)
        priviledged_state = jp.concatenate([
            info["linacc_buffer"].flatten(),
            info["linvel_buffer"].flatten(),
            info["quat_buffer"].flatten(),
            info["angvel_buffer"].flatten(),
            info["target_buffer"].flatten(),
            info["action_buffer"].flatten(),
            info["pos_buffer"].flatten(),
        ])
        # Proprio: IMU-like sensors + velocity + altitude + actions
        # altitude = jp.array([data.qpos[2]])
        proprio = jp.concatenate([
            info["action_buffer"].flatten(),
            info["linacc_buffer"].flatten(),
            info["angvel_buffer"].flatten(),
            info["quat_buffer"].flatten(),
            #altitude, 
            # info["linvel_buffer"].flatten(),
        ])
        obs = {
            "pixels/view_0": frame_stack_0,     # (H, W, history) — mounted camera
            "proprio_obs": proprio,  # (history * (nu+3+3+4),)
            "value_obs": priviledged_state,           # critic obs
            "teacher_obs": priviledged_state,
        }

        state = mjx_env.State(
            data=data,
            obs=obs,
            reward=jp.array(0.0),
            done=jp.array(0.0),
            metrics=self._initialize_metrics(),
            info=info,
        )

        state = self._get_target(state)

        # Build the real initial privileged_state from sensor buffers now that
        # target is set.  Mirrors what _get_obs_impl does each step.
        # NOTE: keep in sync with obs.py — uncomment both together to re-enable.
        # _i = state.info
        # privileged_state = jp.concatenate([
        #     _i["linacc_buffer"].flatten(),
        #     _i["linvel_buffer"].flatten(),
        #     _i["quat_buffer"].flatten(),
        #     _i["angvel_buffer"].flatten(),
        #     _i["target_buffer"].flatten(),
        #     _i["action_buffer"].flatten(),
        #     _i["pos_buffer"].flatten(),
        # ])
        # state = state.replace(obs={**state.obs, "privileged_state": privileged_state})

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
    
    # def _sample_initial_state(self, rng: jp.ndarray):
    #     """Sample initial position and velocity for the drone.
        
    #     Drone starts at ~1.5m above ground with small random perturbation.
        
    #     NOTE: qpos MUST depend on `rng` so that JAX traces it through
    #     jax.vmap.  If qpos were a pure constant the warp render output
    #     would not carry a batch dimension and mjx.get_rgb would crash
    #     with a shape mismatch.
        
    #     Args:
    #         rng: Random key for sampling
            
    #     Returns:
    #         qpos: (nq,) array — position and orientation quaternion (+ any extra joints)
    #         qvel: (nv,) array — linear and angular velocity
    #     """
    #     rng, rng_pos, rng_vel = jax.random.split(rng, 3)

    #     # Nominal pose
    #     position = jp.array([0.0, 3.0, 1.5])
    #     quaternion = jp.array([1.0, 0.0, 0.0, 0.0])

    #     # Small random perturbation on xyz (keeps trace alive under vmap
    #     # and helps exploration).
    #     position = position + 0.01 * jax.random.normal(rng_pos, (3,))

    #     # Build qpos with correct length (model may define additional joints)
    #     nq = int(self._mjx_model.nq)
    #     qpos = jp.zeros(nq)
    #     qpos = qpos.at[0:3].set(position)
    #     qpos = qpos.at[3:7].set(quaternion)

    #     # Build qvel with a tiny random kick (also keeps trace alive)
    #     nv = int(self._mjx_model.nv)
    #     qvel = 0.001 * jax.random.normal(rng_vel, (nv,))

    #     return qpos, qvel


    # def _sample_initial_state(self, rng: jp.ndarray):
    #     rng, rng_pos, rng_vel, rng_ang = jax.random.split(rng, 4)

    #     # Position (start somewhere above pad)
    #     position = jp.array([
    #         jax.random.uniform(rng_pos, (), minval=-1.0, maxval=1.0),
    #         jax.random.uniform(rng_pos, (), minval= -1.0, maxval=1.0),
    #         jax.random.uniform(rng_pos, (), minval= 0.5, maxval=2.0),
    #     ])

    #     # Random orientation (small tilt)
    #     roll  = jax.random.uniform(rng_ang, (), minval=-0.2, maxval=0.2)
    #     pitch = jax.random.uniform(rng_ang, (), minval=-0.2, maxval=0.2)
    #     yaw   = jax.random.uniform(rng_ang, (), minval=-jp.pi, maxval=jp.pi)

    #     quaternion = euler_to_quaternion(roll, pitch, yaw)

    #     # Build qpos
    #     nq = int(self._mjx_model.nq)
    #     qpos = jp.zeros(nq)
    #     qpos = qpos.at[0:3].set(position)
    #     qpos = qpos.at[3:7].set(quaternion)

    #     # Random velocities
    #     nv = int(self._mjx_model.nv)
    #     qvel = jp.zeros(nv)

    #     # linear velocity
    #     qvel = qvel.at[0:3].set(
    #         jax.random.normal(rng_vel, (3,)) * 0.5
    #     )

    #     # angular velocity
    #     qvel = qvel.at[3:6].set(
    #         jax.random.normal(rng_vel, (3,)) * 1
    #     )

    #     return qpos, qvel
    

    def _sample_initial_state(self, rng: jp.ndarray):
        rng, rng_pos, rng_vel, rng_ang, rng_yaw = jax.random.split(rng, 5)

        xy_range = float(getattr(self.cfg, "init_xy_range", 2))
        z_min = float(getattr(self.cfg, "init_z_min", 1))
        z_max = float(getattr(self.cfg, "init_z_max", 2.5))
        tilt_range = float(getattr(self.cfg, "init_tilt_range", 2.4))
        linvel_std = float(getattr(self.cfg, "init_linvel_std", 1.5))
        angvel_std = float(getattr(self.cfg, "init_angvel_std", 3))

        # Position (start above the pad with moderate XY randomization)
        position = jp.array([
            jax.random.uniform(rng_pos, (), minval=-xy_range, maxval=xy_range),
            jax.random.uniform(rng_pos, (), minval=-xy_range, maxval=xy_range),
            jax.random.uniform(rng_pos, (), minval=z_min, maxval=z_max),
        ])

        # Random orientation (mild tilt)
        roll = jax.random.uniform(rng_ang, (), minval=-tilt_range, maxval=tilt_range)
        pitch = jax.random.uniform(rng_ang, (), minval=-tilt_range, maxval=tilt_range)
        yaw = jax.random.uniform(rng_yaw, (), minval=-jp.pi, maxval=jp.pi)

        quaternion = euler_to_quaternion(roll, pitch, yaw)

        # Build qpos
        nq = int(self._mjx_model.nq)
        qpos = jp.zeros(nq)
        qpos = qpos.at[0:3].set(position)
        qpos = qpos.at[3:7].set(quaternion)
        # Camera tilt joint (qpos[7]): start pointing straight down (-π/2)
        # so the mounted camera sees the landing pad from the first frame.
        qpos = qpos.at[7].set(-jp.pi / 2)

        # Random velocities
        nv = int(self._mjx_model.nv)
        qvel = jp.zeros(nv)

        # linear velocity
        qvel = qvel.at[0:3].set(
            jax.random.normal(rng_vel, (3,)) * linvel_std
        )

        # angular velocity
        qvel = qvel.at[3:6].set(
            jax.random.normal(rng_vel, (3,)) * angvel_std
        )

        return qpos, qvel


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
            'reward_crash_penalty': jp.zeros(()),
            'reward_camera_alignment': jp.float32(0.0),
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
    
    def _initialize_state_vars(self, data, rng):
        """Initialize state variables and info dictionary."""
        return _initialize_state_vars_impl(self, data, rng)
    
    def scale_action(self, action_normalized):
        """Scale action from [-1, 1] to actuator control range [ctrl_min, ctrl_max]."""
        action_normalized = jp.array(action_normalized)
        action_normalized = jp.clip(action_normalized, -1.0, 1.0)
        return (action_normalized + 1.0) / 2.0 * (self.ctrl_max - self.ctrl_min) + self.ctrl_min
    

    @property
    def xml_path(self) -> str:
        return self.cfg.xml_path_rel

    @property
    def action_size(self) -> int:
        return self.mjx_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model


def _initialize_state_vars_impl(self, data, rng):
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
