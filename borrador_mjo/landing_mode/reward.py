def _compute_task_events(state, position, xy_error, z_error, cfg=None):
    # Use config thresholds if provided, else fallback to defaults
    xy_thresh = cfg.hover_xy_success_threshold if cfg and hasattr(cfg, 'hover_xy_success_threshold') else 0.15
    z_thresh = cfg.hover_z_success_threshold if cfg and hasattr(cfg, 'hover_z_success_threshold') else 0.10
    steps_required = cfg.success_steps_required if cfg and hasattr(cfg, 'success_steps_required') else 5
    max_steps = cfg.max_episode_steps if cfg and hasattr(cfg, 'max_episode_steps') else 1000

    success_condition = jp.logical_and(
        xy_error < xy_thresh,
        z_error < z_thresh,
    )
    steps_within_success = jp.where(
        success_condition,
        state.state_vars.get('steps_within_success', 0) + 1,
        0,
    )
    is_success = steps_within_success >= steps_required
    is_timeout = state.state_vars.get('step', 0) >= max_steps
    return {
        'success_condition': success_condition,
        'steps_within_success': steps_within_success,
        'is_success': is_success,
        'is_crash': False,  # No crash logic
        'is_timeout': is_timeout,
    }
from jax import numpy as jp


def _get_reward_impl(self, state, action):
    """Calculate reward from current state."""
    
    # ==== Extract data from MuJoCo and state ====
    # Target hover position (x=0, y=0, z=1.5)
    target_pos = jp.array([0.0, 0.0, 1.5])
    current_pos = state.pipeline_state.qpos[:3]  # [x, y, z]

    # ==== Distance reward (position tracking) ====
    pos_error = jp.linalg.norm(target_pos - current_pos)
    r_distance = jp.exp(-self.cfg.hover_z_exp_scale * pos_error) * self.cfg.hover_z_dense_weight

    # ==== Oscillation penalty (position component sign flip detection) ====
    pos_buffer = state.state_vars.get('pos_buffer', jp.zeros((3, 3)))  # shape: (history, 3)
    pos_t = pos_buffer[0]
    pos_t_minus_1 = pos_buffer[1]
    pos_t_minus_2 = pos_buffer[2]
    # Check if position error changed sign (oscillation indicator)
    pos_error_t = pos_t - target_pos
    pos_error_t_minus_1 = pos_t_minus_1 - target_pos
    pos_error_t_minus_2 = pos_t_minus_2 - target_pos
    x_osc = (pos_error_t_minus_1[0] * pos_error_t_minus_2[0]) < 0
    y_osc = (pos_error_t_minus_1[1] * pos_error_t_minus_2[1]) < 0
    z_osc = (pos_error_t_minus_1[2] * pos_error_t_minus_2[2]) < 0
    osc_flags = jp.array([x_osc, y_osc, z_osc])
    r_osc = -self.cfg.hover_z_weight * jp.mean(osc_flags.astype(jp.float32))

    # ==== Overshoot penalty (crossing target position) ====
    x_overshoot = (pos_error_t[0] * pos_error_t_minus_1[0]) < 0
    y_overshoot = (pos_error_t[1] * pos_error_t_minus_1[1]) < 0
    z_overshoot = (pos_error_t[2] * pos_error_t_minus_1[2]) < 0
    any_overshoot = x_overshoot | y_overshoot | z_overshoot
    ro = jp.where(any_overshoot, -self.cfg.hover_z_progress_weight, 0.0)

    # ==== Action chattering penalty ====
    action_buffer = state.state_vars.get('action_buffer', jp.zeros((5, action.shape[0])))
    current_action = action_buffer[0]
    previous_action = action_buffer[1]
    action_change = jp.linalg.norm(current_action - previous_action)
    r_action_chattering = -self.cfg.action_chattering_weight * action_change

    # ==== Action penalty ====
    actuator_forces = state.pipeline_state.actuator_force
    r_action_penalty = -self.cfg.action_penalty_weight * jp.mean(jp.square(actuator_forces))
    
    # ==== Time penalty ====
    rt = -self.cfg.time_penalty

    # ==== Ground penalty ====
    z_position = state.pipeline_state.qpos[2]
    ground_violation = jp.maximum(0.0, self.cfg.crash_height - z_position)
    r_ground = -self.cfg.crash_penalty * jp.square(ground_violation / self.cfg.crash_height)

    # ==== Compute event flags using helper ====
    xy_error = jp.linalg.norm(current_pos[0:2] - target_pos[0:2])
    z_error = jp.abs(current_pos[2] - target_pos[2])
    events = _compute_task_events(state, current_pos, xy_error, z_error, self.cfg)
    success_bonus = jp.where(events['success_condition'], self.cfg.success_bonus, 0.0)

    # ==== Total reward ====
    total_reward = (r_distance + rt + ro + r_osc + r_action_chattering + r_action_penalty +
                    r_ground + success_bonus)

    # ==== Update variables ====
    state_vars = state.state_vars.copy()
    state_vars.update({
        'goal_achieved': events['success_condition'].astype(jp.float32),
        'steps_within_success': events['steps_within_success'],
        'is_success': events['is_success'],
        'is_crash': events['is_crash'],
        'is_timeout': events['is_timeout'],
        'pos_buffer': jp.concatenate([
            current_pos[jp.newaxis, :],
            pos_buffer[:-1, :]
        ], axis=0),
    })

    # Reward metrics for visualization
    metrics = state.metrics.copy()
    metrics.update({
        'reward_distance': r_distance,
        'reward_time_penalty': rt,
        'reward_overshoot': ro,
        'reward_oscillation': r_osc,
        'reward_action_chattering': r_action_chattering,
        'reward_action_penalty': r_action_penalty,
        'reward_ground_penalty': r_ground,
        'reward_success_bonus': success_bonus,
        'reward_total': total_reward,
    })

    return state.replace(
        reward=total_reward,
        metrics=metrics,
        state_vars=state_vars
    )
