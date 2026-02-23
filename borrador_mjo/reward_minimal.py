from jax import numpy as jp



MAX_EPISODE_STEPS = 1000
TIME_PENALTY = 0.01

REWARD_XY_WEIGHT = 0.1
REWARD_Z_WEIGHT = 0.1

XY_SUCCESS_THRESHOLD = 0.15
Z_SUCCESS_THRESHOLD = 0.10

SUCCESS_STEPS_REQUIRED = 5
SUCCESS_BONUS = 10.0


def _compute_task_events(state, position, xy_error, z_error):
    success_condition = jp.logical_and(
        xy_error < XY_SUCCESS_THRESHOLD,
        z_error < Z_SUCCESS_THRESHOLD,
    )
    steps_within_success = jp.where(
        success_condition,
        state.state_vars['steps_within_success'] + 1,
        0,
    )
    is_success = steps_within_success >= SUCCESS_STEPS_REQUIRED
    is_timeout = state.state_vars['step'] >= MAX_EPISODE_STEPS
    return {
        'success_condition': success_condition,
        'steps_within_success': steps_within_success,
        'is_success': is_success,
        'is_crash': False,  # No crash logic
        'is_timeout': is_timeout,
    }



def _get_reward_impl(self, state, action):
    position = state.pipeline_state.qpos[0:3]
    target_pos = state.state_vars.get('target_pos', jp.zeros(3))

    xy_error = jp.linalg.norm(position[0:2] - target_pos[0:2])
    z_error = jp.abs(position[2] - target_pos[2])

    # Use squared error for reward, weighted
    reward_xy = -REWARD_XY_WEIGHT * (xy_error ** 2)
    reward_z = -REWARD_Z_WEIGHT * (z_error ** 2)
    reward_time_penalty = -TIME_PENALTY

    # Compute event flags using helper (minimal)
    events = _compute_task_events(state, position, xy_error, z_error)
    reward_success_bonus = jp.where(events['success_condition'], SUCCESS_BONUS, 0.0)

    reward_total = reward_xy + reward_z + reward_time_penalty + reward_success_bonus

    state_vars = state.state_vars.copy()
    state_vars.update({
        'goal_achieved': events['success_condition'].astype(jp.float32),
        'steps_within_success': events['steps_within_success'],
        'prev_xy_error': xy_error,
        'prev_z_error': z_error,
        'is_success': events['is_success'],
        'is_crash': events['is_crash'],
        'is_timeout': events['is_timeout'],
    })
    metrics = state.metrics.copy()
    metrics.update({
        'reward_xy': reward_xy,
        'reward_z': reward_z,
        'reward_time_penalty': reward_time_penalty,
        'reward_success_bonus': reward_success_bonus,
        'reward_total': reward_total,
    })
    return state.replace(
        reward=reward_total,
        metrics=metrics,
        state_vars=state_vars,
    )
