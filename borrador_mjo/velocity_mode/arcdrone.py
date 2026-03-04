import jax
from jax import numpy as jp
import mujoco
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from omegaconf import DictConfig,OmegaConf
from flax import struct

# RL helper functions
from .obs import _get_obs_impl
from .reward import _get_reward_impl  
from .target import _get_target_impl



@struct.dataclass
class CustomState(State):
    #     Quick solution to still leverage on the Brax PipelineEnv Class.
    #     Every extra  mutable object that is not define in Brax State parent class, should be added here.
    #     State Parent Class attributes:
    #     pipeline_state: Optional[base.State]
    #     obs: Observation
    #     reward: jax.Array
    #     done: jax.Array
    #     metrics: Dict[str, jax.Array] = struct.field(default_factory=dict)
    #     info: Dict[str, Any] = struct.field(default_factory=dict)
    
    state_vars: dict = struct.field(default_factory=dict)  



#======== Main class ================

class ARCDroneRL_Vel(PipelineEnv):

    """ARC Drone Velocity Level Controller RL Environment."""

    def __init__(self, *, cfg: DictConfig = None, **kwargs):


        # TODO: make environment initialization faster!


        # ======== Constants ========
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
        self.cfg = cfg

        # ======== Super Init ========
        mj_model = mujoco.MjModel.from_xml_path(cfg.xml_path_rel)
        mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG  # Conjugate gradient for large systems this is faster than Newton
        mj_model.opt.iterations = 6
        mj_model.opt.ls_iterations = 6
        sys = mjcf.load_model(mj_model)
        sys = sys.tree_replace({'opt.timestep': self.cfg.get('mj_timestep', 0.004)})
        super().__init__(sys, self.cfg.backend, self.cfg.n_frames, **kwargs)

        # ======== Constant Variables ========

        # Constant variables can be define here, and not interferred with jax jit.
        # Most are define in the yaml config file.

        self.ctrl_min = jp.array(mj_model.actuator_ctrlrange[:, 0])
        self.ctrl_max = jp.array(mj_model.actuator_ctrlrange[:, 1])


        # ======= Dynamic Variables ========

        # Jax need stateless functions, so we assign this dynamic variables to an external dataclass.
        # All dynamic variables are initialize in reset() and update in step().
        # see above CustomState class.



    # Main methods =================================================================

    def reset(self, rng: jp.ndarray) -> CustomState:
        """Resets the environment to an initial state and samples a new goal."""

        rng, key = jax.random.split(rng)
        qpos, qvel = self._sample_initial_state(key) 
        data = self.pipeline_init(qpos, qvel) 
        state_vars = self._initialize_state_vars(data, rng)

        state = CustomState(
            pipeline_state=data,
            obs= jp.array(0.0),   # overwritten to the right shape bellow
            reward=jp.array(0.0),
            done=jp.array(0.0),
            state_vars=state_vars,
            metrics=self._initialize_metrics()
        )

        state = self._get_target(state)
        zero_action = jp.zeros(self.sys.nu)         # initialize with zero action
        state = self._get_obs(state, zero_action)

        return state
    

    def step(self, state: CustomState, action: jp.ndarray) -> CustomState:
        """Runs one timestep of the environment's dynamics."""
        
        # 0. Scale the action
        action_normalized = action
        action = self.scale_action(action)
        
        # 1. Update physics with scaled action
        data = self.pipeline_step(state.pipeline_state, action)
        state = state.replace(pipeline_state=data)
        
        # 2. Update target
        state = self._get_target(state)
        
        # 3. Compute observations
        state = self._get_obs(state, action)
        
        # 4. Compute reward
        state = self._get_reward(state, action_normalized)
        
        # 5. Check termination
        state = self._check_termination(state)
        
        return state

    # Helper functions =============================================================

    def _check_termination(self, state: CustomState) -> CustomState:
        """Check if episode should terminate."""
        
        # Get z position
        z_position = state.pipeline_state.qpos[2]
        
        # Check termination conditions
        done = jp.logical_or(
            state.state_vars['steps_within_success'] >= self.cfg.success_steps_required,
            state.info.get('step', 0) >= self.cfg.max_episode_steps
        )
        
        # Add ground collision termination (z <= threshold means drone is too close to ground)
        ground_collision = z_position <= self.cfg.ground_threshold
        done = jp.logical_or(done, ground_collision)
        
        return state.replace(done=done.astype(jp.float32))
    
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
        # qpos: [x, y, z, qw, qx, qy, qz]
        position = jp.array([0.0, 0.0, 1.5])
        
        # Initial orientation: level (identity quaternion: w=1, x=0, y=0, z=0)
        quaternion = jp.array([1.0, 0.0, 0.0, 0.0])
        
        # Combine into qpos
        qpos = jp.concatenate([position, quaternion])
        
        # Initial velocity: zero (at rest)
        # qvel: [vx, vy, vz, wx, wy, wz]
        qvel = jp.zeros(6)
        
        return qpos, qvel


    def _initialize_state_vars(self, data, rng):
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

    def _get_obs(self, state: CustomState, action: jp.ndarray) -> CustomState:
        return _get_obs_impl(self, state, action)
        
    def _get_reward(self, state: CustomState, action: jp.ndarray) -> CustomState:
        """Calculate reward from current state."""
        return _get_reward_impl(self, state, action)
    
    def _get_target(self, state: CustomState) -> CustomState:
        """Update target velocity."""
        return _get_target_impl(self, state)
    
    def scale_action(self, action_normalized):
        """Scale action from [-1, 1] to actuator control range [ctrl_min, ctrl_max]."""
        action_normalized = jp.array(action_normalized)
        action_normalized = jp.clip(action_normalized, -1.0, 1.0)
        return (action_normalized + 1.0) / 2.0 * (self.ctrl_max - self.ctrl_min) + self.ctrl_min