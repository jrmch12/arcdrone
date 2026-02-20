import jax
from jax import numpy as jp
import mujoco
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from omegaconf import DictConfig, OmegaConf
from flax import struct

from .obs import _get_obs_impl
from .reward import _get_reward_impl
from .target import _get_target_impl


@struct.dataclass
class CustomState(State):
    state_vars: dict = struct.field(default_factory=dict)


class ARCDroneRL_Landing(PipelineEnv):
    """ARC Drone Landing Controller RL Environment."""

    def __init__(self, *, cfg: DictConfig = None, **kwargs):
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
        self.cfg = cfg

        mj_model = mujoco.MjModel.from_xml_path(cfg.xml_path_rel)
        mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        mj_model.opt.iterations = 6
        mj_model.opt.ls_iterations = 6
        sys = mjcf.load_model(mj_model)
        sys = sys.tree_replace({'opt.timestep': self.cfg.get('mj_timestep', 0.004)})
        super().__init__(sys, self.cfg.backend, self.cfg.n_frames, **kwargs)

        self.ctrl_min = jp.array(mj_model.actuator_ctrlrange[:, 0])
        self.ctrl_max = jp.array(mj_model.actuator_ctrlrange[:, 1])

    def reset(self, rng: jp.ndarray) -> CustomState:
        rng, key = jax.random.split(rng)
        qpos, qvel = self._sample_initial_state(key)
        data = self.pipeline_init(qpos, qvel)
        state_vars = self._initialize_state_vars(data, rng)

        state = CustomState(
            pipeline_state=data,
            obs=jp.array(0.0),
            reward=jp.array(0.0),
            done=jp.array(0.0),
            state_vars=state_vars,
            metrics=self._initialize_metrics(),
        )

        state = self._get_target(state)
        zero_action = jp.zeros(self.sys.nu)
        state = self._get_obs(state, zero_action)
        return state

    def step(self, state: CustomState, action: jp.ndarray) -> CustomState:
        action_normalized = action
        action = self.scale_action(action)

        data = self.pipeline_step(state.pipeline_state, action)
        state = state.replace(pipeline_state=data)

        state = self._get_target(state)
        state = self._get_obs(state, action)
        state = self._get_reward(state, action_normalized)
        state = self._check_termination(state)

        state_vars = state.state_vars.copy()
        state_vars.update({'step': state.state_vars['step'] + 1})
        return state.replace(state_vars=state_vars)

    def _check_termination(self, state: CustomState) -> CustomState:
        z_position = state.pipeline_state.qpos[2]
        done = jp.logical_or(
            state.state_vars['steps_within_success'] >= self.cfg.success_steps_required,
            state.state_vars['step'] >= self.cfg.max_episode_steps,
        )

        touchdown_active = state.state_vars['goal_achieved'] > 0.5
        crash = jp.logical_and(z_position <= self.cfg.crash_height, jp.logical_not(touchdown_active))
        done = jp.logical_or(done, crash)
        return state.replace(done=done.astype(jp.float32))

    def _sample_initial_state(self, rng: jp.ndarray):
        rng_xy, _ = jax.random.split(rng)
        xy = jax.random.uniform(rng_xy, (2,), minval=-self.cfg.reset_xy_range, maxval=self.cfg.reset_xy_range)
        z = jp.array([self.cfg.reset_z])
        position = jp.concatenate([xy, z])

        quaternion = jp.array([1.0, 0.0, 0.0, 0.0])
        qpos = jp.concatenate([position, quaternion])
        qvel = jp.zeros(6)
        return qpos, qvel

    def _initialize_state_vars(self, data, rng):
        initial_xy_error = jp.linalg.norm(data.qpos[0:2])
        initial_z_error = jp.abs(data.qpos[2])
        initial_hover_z_error = jp.abs(data.qpos[2] - self.cfg.hover_target_z)
        return {
            'step': 0,
            'rng': rng,
            'goal_achieved': jp.array(0.0),
            'steps_within_success': jp.array(0),
            'target_pos': jp.array([0.0, 0.0, 0.0]),
            'prev_xy_error': initial_xy_error,
            'prev_z_error': initial_z_error,
            'prev_hover_z_error': initial_hover_z_error,
        }

    def _initialize_metrics(self):
        return {
            'reward_xy_alignment': jp.float32(0.0),
            'reward_z_alignment': jp.float32(0.0),
            'reward_xy_progress': jp.float32(0.0),
            'reward_z_progress': jp.float32(0.0),
            'reward_upright': jp.float32(0.0),
            'reward_linvel_penalty': jp.float32(0.0),
            'reward_angvel_penalty': jp.float32(0.0),
            'reward_action_penalty': jp.float32(0.0),
            'reward_time_penalty': jp.float32(0.0),
            'reward_touchdown_progress': jp.float32(0.0),
            'reward_touchdown_bonus': jp.float32(0.0),
            'reward_crash_penalty': jp.float32(0.0),
            'reward_total': jp.float32(0.0),
        }

    def _get_obs(self, state: CustomState, action: jp.ndarray) -> CustomState:
        return _get_obs_impl(self, state, action)

    def _get_reward(self, state: CustomState, action: jp.ndarray) -> CustomState:
        return _get_reward_impl(self, state, action)

    def _get_target(self, state: CustomState) -> CustomState:
        return _get_target_impl(self, state)

    def scale_action(self, action_normalized):
        action_normalized = jp.array(action_normalized)
        action_normalized = jp.clip(action_normalized, -1.0, 1.0)
        return (action_normalized + 1.0) / 2.0 * (self.ctrl_max - self.ctrl_min) + self.ctrl_min
