from collections.abc import Mapping
from typing import Any, Callable

from jax import numpy as jp

from arcdrone.controller.rl.task.vision_mode.obs import _get_obs_impl as _default_student_obs_fn


class StudentWrapper:
    """Wraps a teacher env and appends student observations.

    The wrapper keeps student-only recurrent buffers in ``state.info[student_info_key]``
    so teacher buffers are never modified by student observation computation.
    """

    def __init__(
        self,
        teacher_env,
        student_obs_fn: Callable[[Any, Any, jp.ndarray], Any] = _default_student_obs_fn,
        *,
        get_student_obs: bool = True,
        student_obs_key: str = "state",
        student_obs_name: str = "student_state",
        student_info_key: str = "student_info",
        valid_student_obs_key: str = "valid_student_obs",
    ):
        self.teacher_env = teacher_env
        self.student_obs_fn = student_obs_fn
        self.get_student_obs = get_student_obs
        self.student_obs_key = student_obs_key
        self.student_obs_name = student_obs_name
        self.student_info_key = student_info_key
        self.valid_student_obs_key = valid_student_obs_key

    def __getattr__(self, name):
        return getattr(self.teacher_env, name)

    def _extract_student_obs(self, obs: Any):
        if isinstance(obs, Mapping):
            if self.student_obs_key not in obs:
                raise KeyError(
                    f"student_obs_key='{self.student_obs_key}' not found in student obs keys: {list(obs.keys())}"
                )
            return obs[self.student_obs_key]
        return obs

    def _compute_student_obs(self, state, action):
        student_info = state.info.get(self.student_info_key, state.info)
        student_state_in = state.replace(info=student_info)
        student_state_out = self.student_obs_fn(self.teacher_env, student_state_in, action)
        student_obs = self._extract_student_obs(student_state_out.obs)
        return student_obs, student_state_out.info

    def _attach_student_fields(self, state, action, *, compute_student_obs: bool):
        if not isinstance(state.obs, Mapping):
            raise TypeError("StudentWrapper expects env observations to be a mapping/dict.")

        obs_out = dict(state.obs)
        info_out = dict(state.info)

        if compute_student_obs:
            student_obs, student_info = self._compute_student_obs(state, action)
            valid_student_obs = jp.array(True)
        else:
            if self.student_obs_name in obs_out:
                student_obs = jp.zeros_like(obs_out[self.student_obs_name])
            else:
                student_obs = jp.zeros_like(obs_out[self.student_obs_key])
            student_info = info_out.get(self.student_info_key, info_out)
            valid_student_obs = jp.array(False)

        obs_out[self.student_obs_name] = student_obs
        info_out[self.student_info_key] = student_info
        info_out[self.valid_student_obs_key] = valid_student_obs

        return state.replace(obs=obs_out, info=info_out)

    def reset(self, rng):
        state = self.teacher_env.reset(rng)

        zero_action = jp.zeros(self.teacher_env.action_size)
        student_obs, student_info = self._compute_student_obs(state, zero_action)

        obs_out = dict(state.obs)
        info_out = dict(state.info)

        if self.get_student_obs:
            obs_out[self.student_obs_name] = student_obs
            valid_student_obs = jp.array(True)
        else:
            obs_out[self.student_obs_name] = jp.zeros_like(student_obs)
            valid_student_obs = jp.array(False)

        info_out[self.student_info_key] = student_info
        info_out[self.valid_student_obs_key] = valid_student_obs

        return state.replace(obs=obs_out, info=info_out)

    def step(self, state, action):
        state = self.teacher_env.step(state, action)
        return self._attach_student_fields(
            state,
            action,
            compute_student_obs=self.get_student_obs,
        )