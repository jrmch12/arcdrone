"""Placeholder reward module for vision-mode student env.

This environment is currently used to produce student observations
(``pixels/*`` + action buffer) and is not trained with RL yet.
"""

from jax import numpy as jp


def _check_episode_events_impl(self, state):
	"""No-op termination placeholder for vision-only usage."""
	info = state.info.copy()
	info.update(
		{
			'goal_achieved': jp.array(0.0, dtype=jp.float32),
			'steps_within_success': jp.array(0, dtype=jp.int32),
			'ground_violation': jp.array(0.0, dtype=jp.float32),
		}
	)
	return state.replace(done=jp.array(0.0, dtype=jp.float32), info=info)


def _get_reward_impl(self, state, action):
	"""No-op reward placeholder for vision-only usage."""
	del action
	zero = jp.array(0.0, dtype=jp.float32)

	metrics = state.metrics.copy()
	metrics.update(
		{
			'reward_distance': zero,
			'reward_time_penalty': zero,
			'reward_overshoot': zero,
			'reward_oscillation': zero,
			'reward_action_chattering': zero,
			'reward_action_penalty': zero,
			'reward_attitude_level': zero,
			'reward_low_angvel': zero,
			'reward_low_linvel': zero,
			'reward_soft_landing': zero,
			'reward_success_bonus': zero,
			'reward_ground_penalty': zero,
			'reward_total': zero,
		}
	)

	return state.replace(reward=zero, metrics=metrics)
