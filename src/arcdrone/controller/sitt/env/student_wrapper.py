import jax
from typing import Any

from flax import struct
from jax import numpy as jp
from mujoco_playground._src import mjx_env


@struct.dataclass
class StudentData:
    """Student observation data carried alongside the teacher state."""
    obs: Any                  # pixel + action obs dict
    shouldObs: jp.ndarray     # scalar bool — True when rendering is requested
    validObs: jp.ndarray      # scalar int  — 1 if this step was actually rendered, 0 if skipped
    info: Any = None          # student-specific buffers (render_token, obs_history, …)


@struct.dataclass
class StudentState(mjx_env.State):
    """MJX state extended with a ``student`` field.  Teacher fields are untouched."""
    student: StudentData | None = None


class StudentWrapper:
    """Wraps a teacher env so student obs live in ``state.student``.

    * ``state.obs`` / ``state.info`` are **never modified** — PPO works as usual.
    * reset: runs teacher_env.reset then student_env.reset; student info is kept
      in ``state.student.info`` alongside the render token and frame history.
    * step: runs teacher_env.step then calls _get_obs_impl(student_env, ...) to
      update the pixel frame-history buffer.
    """

    def __init__(self, teacher_env, student_env):
        self.teacher_env = teacher_env
        self.student_env = student_env

    def __getattr__(self, name):
        return getattr(self.teacher_env, name)

    # ------------------------------------------------------------------

    def reset(self, rng):
        # 1. Full teacher reset
        teacher_state = self.teacher_env.reset(rng)

        # 2. Full student reset (initialises render_token + obs_history)
        student_state = self.student_env.reset(rng)

        # 3. Carry student's info (render_token, obs_history, buffers…) in StudentData
        student = StudentData(
            obs=student_state.obs,
            shouldObs=jp.array(False),
            validObs=jp.array(0),
            info=student_state.info,
        )

        return StudentState(
            data=teacher_state.data,
            obs=teacher_state.obs,
            reward=teacher_state.reward,
            done=teacher_state.done,
            metrics=teacher_state.metrics,
            info=teacher_state.info,
            student=student,
        )

    def step(self, state, action):
        # 1. Teacher step
        state = self.teacher_env.step(state, action)

        # 2. Student obs: only render when shouldObs=True (rendering is expensive)
        shouldObs = state.student.shouldObs

        def do_render(_):
            state_with_student_info = state.replace(info=state.student.info)
            out = self.student_env._get_obs(state_with_student_info, action)
            return out.obs, out.info

        def skip_render(_):
            zero_obs = jax.tree_util.tree_map(jp.zeros_like, state.student.obs)
            return zero_obs, state.student.info

        student_obs, student_info = jax.lax.cond(shouldObs, do_render, skip_render, None)

        student = StudentData(
            obs=student_obs,
            shouldObs=shouldObs,
            validObs=jp.where(shouldObs, jp.array(1), jp.array(0)),
            info=student_info,
        )
        return state.replace(student=student)