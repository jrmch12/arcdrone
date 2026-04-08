"""Vision-based drone landing environment.

Follows the CartpoleBalance vision pattern exactly:
- Single fixed side camera, grayscale, 64x64, 3-frame stack
- Observation is ONLY pixels: {"pixels/view_0": (64,64,3)}
- NO state observations — forces CNN to learn from pixels
- Dense additive reward (alive + penalties)
- Early termination with done penalty
- Compatible with Brax's standard vision PPO training
"""

from typing import Any, Dict, Optional, Union
from pathlib import Path

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env

# Resolve scene path relative to this file
_WORKSPACE = Path(__file__).resolve().parents[3]  # borrador_braxenvs/
_SCENE_PATH = str(_WORKSPACE / "assets" / "skydio_x2" / "scene.xml")


def _euler_to_quaternion(roll, pitch, yaw):
    """Convert Euler angles to quaternion [w, x, y, z]."""
    cy = jp.cos(yaw * 0.5)
    sy = jp.sin(yaw * 0.5)
    cp = jp.cos(pitch * 0.5)
    sp = jp.sin(pitch * 0.5)
    cr = jp.cos(roll * 0.5)
    sr = jp.sin(roll * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return jp.array([w, x, y, z])


def default_vision_config() -> config_dict.ConfigDict:
    return config_dict.create(
        nworld=2048,
        cam_res=(64, 64),
        use_textures=True,
        use_shadows=False,
        render_rgb=(True,),
        render_depth=(False,),
        enabled_geom_groups=[0, 1, 2],
        cam_active=(False, True, False),  # only outer_camera_side
    )


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.004,
        episode_length=200,
        action_repeat=1,
        vision=True,
        vision_config=default_vision_config(),
        impl="warp",
        naconmax=0,
        njmax=250,
        blind=False,  # zero out pixels (test state-only learning)
    )


class DroneLanding(mjx_env.MjxEnv):
    """Vision-based drone landing on a fiducial plate.

    The drone must land on a checkerboard pad at the origin.
    Observation is ONLY a top-down camera view (pixels).
    Reward is additive and dense, following the CartpoleBalance pattern.
    """

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(config, config_overrides=config_overrides)
        self._vision = self._config.vision
        self._blind = getattr(self._config, 'blind', False)

        self._mj_model = mujoco.MjModel.from_xml_path(_SCENE_PATH)
        self._mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        self._mj_model.opt.iterations = 6
        self._mj_model.opt.ls_iterations = 6
        self._mj_model.opt.timestep = self.sim_dt
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)

        # Control ranges for action scaling
        self._ctrl_min = jp.array(self._mj_model.actuator_ctrlrange[:, 0])
        self._ctrl_max = jp.array(self._mj_model.actuator_ctrlrange[:, 1])
        # Number of thrust actuators (exclude camera servo for the policy)
        self._n_thrusters = 4

        if self._vision:
            vision_kwargs = self._config.vision_config.to_dict()
            self._rc = mjx.create_render_context(
                mjm=self._mj_model, **vision_kwargs
            )
            self._rc_pytree = self._rc.pytree()

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, rng_pos, rng_vel, rng_att = jax.random.split(rng, 4)

        # Position: start above pad, moderate randomization
        pos_rand = jax.random.uniform(rng_pos, (3,))
        position = jp.array([
            pos_rand[0] * 2.0 - 1.0,   # x in [-1.0, 1.0]
            pos_rand[1] * 2.0 - 1.0,   # y in [-1.0, 1.0]
            pos_rand[2] * 1.2 + 0.8,   # z in [0.8, 2.0]
        ])

        # Orientation: moderate tilt randomization
        att_rand = jax.random.uniform(rng_att, (3,))
        roll = att_rand[0] * 1.0 - 0.5    # [-0.5, 0.5]
        pitch = att_rand[1] * 1.0 - 0.5   # [-0.5, 0.5]
        yaw = att_rand[2] * 2.0 * jp.pi - jp.pi  # [-pi, pi]
        quaternion = _euler_to_quaternion(roll, pitch, yaw)

        nq = int(self._mjx_model.nq)
        qpos = jp.zeros(nq)
        qpos = qpos.at[0:3].set(position)
        qpos = qpos.at[3:7].set(quaternion)

        nv = int(self._mjx_model.nv)
        qvel = jp.zeros(nv)
        qvel = qvel.at[0:3].set(jax.random.normal(rng_vel, (3,)) * 0.5)
        qvel = qvel.at[3:6].set(jax.random.normal(rng_vel, (3,)) * 1.0)

        data = mjx_env.make_data(
            self._mj_model,
            qpos=qpos,
            qvel=qvel,
            impl=self._mjx_model.impl.value,
            naconmax=self._config.naconmax,
            njmax=self._config.njmax,
        )
        data = mjx.forward(self._mjx_model, data)

        metrics = {
            "reward/alive": jp.zeros(()),
            "reward/height": jp.zeros(()),
            "reward/xy_penalty": jp.zeros(()),
            "reward/vel_penalty": jp.zeros(()),
            "reward/attitude_penalty": jp.zeros(()),
            "reward/proximity": jp.zeros(()),
        }
        info = {"rng": rng, "step": jp.array(0)}

        if self._vision:
            render_data = mjx.refit_bvh(self.mjx_model, data, self._rc_pytree)
            out = mjx.render(self.mjx_model, render_data, self._rc_pytree)
            rgb = mjx.get_rgb(self._rc_pytree, 0, out[0])
            gray = jp.mean(rgb, axis=-1, keepdims=True) - 0.5
            frame_stack = jp.repeat(gray, 3, axis=-1)
            info["frame_stack"] = frame_stack
            pixels = jp.where(self._blind, jp.zeros_like(frame_stack), frame_stack)
            obs = {"pixels/view_0": pixels}
        else:
            obs = self._get_state_obs(data)

        return mjx_env.State(data, obs, jp.zeros(()), jp.zeros(()), metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        # Scale action: policy outputs [-1, 1], scale to actuator range
        # Only use first 4 dims for thrusters, set camera servo to 0
        action_scaled = self._scale_action(action)
        data = mjx_env.step(self.mjx_model, state.data, action_scaled, self.n_substeps)

        # Compute reward
        reward, metrics = self._compute_reward(data, action, state.metrics)

        # Check termination
        pos = data.qpos[0:3]
        xy_radius = jp.linalg.norm(pos[0:2])
        z = pos[2]
        in_landing_zone = xy_radius <= 0.4

        done = (
            jp.isnan(data.qpos).any()
            | jp.isnan(data.qvel).any()
            | (z <= 0.05) & (~in_landing_zone)  # crash outside landing zone
            | (jp.abs(pos[0]) > 4.0)  # too far away
            | (jp.abs(pos[1]) > 4.0)
            | (z > 5.0)  # too high
        )
        done = done.astype(float)
        reward = reward + -2.0 * done  # death penalty

        info = dict(state.info)
        info["step"] = info["step"] + 1

        if self._vision:
            render_data = mjx.refit_bvh(self.mjx_model, data, self._rc_pytree)
            out = mjx.render(self.mjx_model, render_data, self._rc_pytree)
            rgb = mjx.get_rgb(self._rc_pytree, 0, out[0])
            gray = jp.mean(rgb, axis=-1, keepdims=True) - 0.5
            prev_stack = state.info["frame_stack"]
            frame_stack = jp.concatenate([prev_stack[..., 1:], gray], axis=-1)
            info["frame_stack"] = frame_stack
            pixels = jp.where(self._blind, jp.zeros_like(frame_stack), frame_stack)
            obs = {"pixels/view_0": pixels}
        else:
            obs = self._get_state_obs(data)

        return mjx_env.State(data, obs, reward, done, metrics, info)

    def _scale_action(self, action: jax.Array) -> jax.Array:
        """Scale policy output [-1, 1] to actuator control range."""
        action = jp.clip(action, -1.0, 1.0)
        # Build full control: 4 thrusters + camera servo (fixed at 0)
        ctrl = (action + 1.0) / 2.0 * (self._ctrl_max[:4] - self._ctrl_min[:4]) + self._ctrl_min[:4]
        # Append camera servo control (neutral)
        full_ctrl = jp.concatenate([ctrl, jp.zeros(1)])
        return full_ctrl

    def _compute_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        metrics: Dict[str, Any],
    ) -> tuple:
        """Dense additive reward following CartpoleBalance pattern.

        All terms are smooth, bounded, and have useful gradients everywhere.
        Structure: alive(+1) + proximity bonus + small penalties.
        """
        pos = data.qpos[0:3]
        quat = data.qpos[3:7]
        linvel = data.qvel[0:3]
        angvel = data.qvel[3:6]
        z = pos[2]
        xy_dist_sq = pos[0] ** 2 + pos[1] ** 2

        # 1. Alive bonus: +1 per step (backbone, like CartpoleBalance)
        alive = jp.float32(1.0)

        # 2. Height penalty: linear, gentle
        height_penalty = -0.5 * z

        # 3. XY position penalty: quadratic centering (strong — incentivize positioning)
        xy_penalty = -0.2 * xy_dist_sq

        # 4. Velocity penalty: penalize fast movement
        speed_sq = jp.sum(linvel ** 2)
        angspeed_sq = jp.sum(angvel ** 2)
        vel_penalty = -0.01 * speed_sq - 0.005 * angspeed_sq

        # 5. Attitude penalty: stay level (penalize tilt)
        quat_norm = quat / jp.maximum(jp.linalg.norm(quat), 1e-8)
        cos_tilt = jp.clip(1.0 - 2.0 * (quat_norm[1] ** 2 + quat_norm[2] ** 2), -1.0, 1.0)
        attitude_penalty = -0.1 * (1.0 - cos_tilt)

        # 6. Proximity bonus: continuous reward that increases as drone
        #    gets closer to target (origin at ground level).
        #    3D distance to origin: sqrt(x² + y² + z²)
        #    Reward: 1/(1 + dist) so it's [0, 1], smooth everywhere
        dist_3d = jp.sqrt(xy_dist_sq + z ** 2 + 1e-6)
        proximity = 3.0 / (1.0 + dist_3d)

        metrics = dict(metrics)
        metrics["reward/alive"] = alive
        metrics["reward/height"] = height_penalty
        metrics["reward/xy_penalty"] = xy_penalty
        metrics["reward/vel_penalty"] = vel_penalty
        metrics["reward/attitude_penalty"] = attitude_penalty
        metrics["reward/proximity"] = proximity

        total = alive + height_penalty + xy_penalty + vel_penalty + attitude_penalty + proximity
        return total, metrics

    def _get_state_obs(self, data: mjx.Data) -> jax.Array:
        """Proprioceptive obs for non-vision mode (debugging only)."""
        return jp.concatenate([
            data.qpos[0:3],   # position
            data.qpos[3:7],   # quaternion
            data.qvel[0:6],   # linear + angular velocity
        ])

    @property
    def xml_path(self) -> str:
        return _SCENE_PATH

    @property
    def action_size(self) -> int:
        return self._n_thrusters  # 4 thrusters only (not camera servo)

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model
