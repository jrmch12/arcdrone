import jax
import mujoco
from mujoco import mjx
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

    Designed to be wrapped by ``MadronaWrapper`` (``vision=True``):

    * Exposes the student env's MJX model and ``unwrapped`` property so that
      ``BraxDomainRandomizationVmapWrapper`` tiles the visual fields on the
      correct (student/vision) model rather than the teacher's.
    * ``state.obs`` / ``state.info`` are **never modified** — PPO works as usual.
    * Student pixel obs are also stored in ``state.info['student_obs']`` so
      that ``acting.generate_unroll`` can capture them via ``extra_fields``.
    * reset: runs teacher_env.reset then student_env.reset; student info is kept
      in ``state.student.info`` alongside the render token and frame history.
    * step: runs teacher_env.step then calls _get_obs_impl(student_env, ...) to
      update the pixel frame-history buffer.
    """

    def __init__(self, teacher_env, student_env):
        self.teacher_env = teacher_env
        self.student_env = student_env

    # ------------------------------------------------------------------
    # Attributes delegated to the *student* env so that MadronaWrapper /
    # BraxDomainRandomizationVmapWrapper can tile the vision model fields
    # and temporarily replace _mjx_model during vmapped reset/step.
    # ------------------------------------------------------------------

    @property
    def unwrapped(self):
        """Return self so MadronaWrapper's v_env_fn yields *this* wrapper
        (calling StudentWrapper.reset/step), while still allowing
        _mjx_model replacement on the student env via the property below."""
        return self

    @property
    def _mjx_model(self):
        return self.student_env._mjx_model

    @_mjx_model.setter
    def _mjx_model(self, value):
        self.student_env._mjx_model = value

    @property
    def mjx_model(self) -> mjx.Model:
        return self.student_env.mjx_model

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self.student_env.mj_model

    # Everything else (observation_size, action_size, …) → teacher env.
    def __getattr__(self, name):
        if name == '__setstate__':
            raise AttributeError(name)
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
            shouldObs=jp.array(True),
            validObs=jp.array(0),
            info=student_state.info,
        )

        # 4. Also store student obs in state.info so generate_unroll can
        #    capture them via extra_fields=('student_obs',).
        info = {**teacher_state.info, 'student_obs': student_state.obs}

        return StudentState(
            data=teacher_state.data,
            obs=teacher_state.obs,
            reward=teacher_state.reward,
            done=teacher_state.done,
            metrics=teacher_state.metrics,
            info=info,
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

        new_obs, new_info = jax.lax.cond(shouldObs, do_render, skip_render, None)

        new_student = StudentData(
            obs=new_obs,
            shouldObs=state.student.shouldObs,
            validObs=jp.where(shouldObs, jp.array(1), jp.array(0)),
            info=new_info,
        )

        # Store student obs in state.info so generate_unroll captures them
        info = {**state.info, 'student_obs': new_obs}

        return state.replace(student=new_student, info=info)

