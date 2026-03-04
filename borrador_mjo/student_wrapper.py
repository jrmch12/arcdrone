from collections.abc import Mapping
from typing import Any

from flax import struct
from jax import numpy as jp
from mujoco_playground._src import mjx_env

from arcdrone.controller.rl.task.vision_mode.obs import _get_obs_impl as _default_student_obs_fn


@struct.dataclass
class StudentData:
    """Student observation data carried alongside the teacher state."""
    obs: Any                  # flat student observation vector
    shouldObs: jp.ndarray     # scalar bool — True when obs is valid
    info: Any = None          # student-specific buffers (same keys as teacher info)


@struct.dataclass
class StudentState(mjx_env.State):
    """MJX state extended with a ``student`` field.  Teacher fields are untouched."""
    student: StudentData | None = None


class StudentWrapper:
    """Wraps a teacher env so student obs live in ``state.student``.

    * ``state.obs`` / ``state.info`` are **never modified** — PPO works as usual.
    * The student obs function (same one used later for RL fine-tuning) is
      called with a *fake* state whose ``info`` carries the student buffers,
      and only its outputs are stored back in ``state.student``.
    """

    def __init__(self, teacher_env, student_obs_fn=_default_student_obs_fn):
        self.teacher_env = teacher_env
        self.student_obs_fn = student_obs_fn

    def __getattr__(self, name):
        return getattr(self.teacher_env, name)

    # ------------------------------------------------------------------
    # Core: call student_obs_fn with student buffers, return (obs, info)
    # ------------------------------------------------------------------

    def _run_student_obs_fn(self, state, action, student_info):
        """Run the student obs function on a fake state built from *student_info*.

        The student_obs_fn signature is ``fn(env, state, action) -> state``
        where the returned ``state.obs`` contains the student observation dict
        and ``state.info`` contains the updated student buffers.
        """
        fake_state = state.replace(info=student_info)
        out = self.student_obs_fn(self.teacher_env, fake_state, action)
        # Extract flat obs from the dict returned by the student obs function
        obs = out.obs
        if isinstance(obs, Mapping):
            obs = obs.get("state", next(iter(obs.values())))
        return obs, out.info

    def _init_student_info(self, state, rng):
        """Initialise the student-specific info dict that vision_mode/obs.py needs.

        Mirrors landing_mode._initialize_state_vars for student-side buffers.
        """
        data = state.data
        buf  = self.teacher_env.cfg.buffer_size
        nu   = self.teacher_env.action_size

        quat   = data.sensordata[0:4]
        angvel = data.sensordata[4:7]
        linacc = data.sensordata[7:10]
        linvel = data.sensordata[10:13]
        pos    = data.qpos[0:3]
        target = jp.array([0.0, 0.0, 1.5])

        return {
            'step':              0,
            'rng':               rng,
            'goal_achieved':     jp.array(0.0),
            'steps_within_success': jp.array(0),
            'target':            target,
            # Buffers - initialized with current state repeated
            'action_buffer':     jp.tile(jp.zeros(nu), (buf, 1)),
            'target_buffer':     jp.tile(jp.zeros(3), (buf, 1)),
            'linacc_buffer':     jp.tile(linacc, (buf, 1)),
            'quat_buffer':       jp.tile(quat,   (buf, 1)),
            'angvel_buffer':     jp.tile(angvel, (buf, 1)),
            'linvel_buffer':     jp.tile(linvel, (buf, 1)),
            'pos_buffer':        jp.tile(pos,    (buf, 1)),
            'linacc_buffer_noisy': jp.tile(linacc, (buf, 1)),
            'quat_buffer_noisy':   jp.tile(quat,   (buf, 1)),
            'angvel_buffer_noisy': jp.tile(angvel, (buf, 1)),
            'linvel_buffer_noisy': jp.tile(linvel, (buf, 1)),
            'ground_violation':  jp.array(0.0),
        }

    # ------------------------------------------------------------------

    def reset(self, rng):
        state = self.teacher_env.reset(rng)
        student_info = self._init_student_info(state, rng)
        zero_action = jp.zeros(self.teacher_env.action_size)
        obs, student_info = self._run_student_obs_fn(state, zero_action, student_info)

        student = StudentData(
            obs=jp.zeros_like(obs),
            shouldObs=jp.array(False),
            info=student_info,
        )
        return StudentState(
            data=state.data, obs=state.obs, reward=state.reward,
            done=state.done, metrics=state.metrics, info=state.info,
            student=student,
        )

    def step(self, state, action):
        state = self.teacher_env.step(state, action)

        # Always run the obs fn to keep student buffers (FIFO history) up to date.
        # shouldObs is read from the state — training sets it before align rollouts.
        shouldObs = state.student.shouldObs
        obs, info = self._run_student_obs_fn(state, action, state.student.info)

        student = StudentData(
            obs=jp.where(shouldObs, obs, jp.zeros_like(obs)),
            shouldObs=shouldObs,  # preserved — caller controls this via state
            info=info,
        )
        return state.replace(student=student)