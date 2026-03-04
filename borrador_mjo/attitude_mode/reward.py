from jax import numpy as jp
from ....utils.math_utils import quaternion_to_euler, euler_to_quaternion, quaternion_error


def _get_reward_impl(self, state, action):
    """Calculate reward from current state."""
    
    # ==== Extract data from MuJoCo and state ====
    
    # Add more buffers if needed

    # ==== Distance reward (attitude tracking) ====
    target_attitude_euler = state.state_vars['target_attitude_buffer'][0]  # [roll, pitch, yaw_rate]
    current_quat = state.state_vars['quat_buffer'][0]
    current_angvel = state.state_vars['angvel_buffer'][0]
    target_quat = euler_to_quaternion(target_attitude_euler[0], target_attitude_euler[1], 0.0)
    quat_error = quaternion_error(current_quat, target_quat)
    yaw_rate_error = jp.abs(current_angvel[2] - target_attitude_euler[2])
    r_distance = (jp.exp(-self.cfg.distance_scale * quat_error) + 
                  jp.exp(-self.cfg.distance_scale * yaw_rate_error)) * 0.5 * self.cfg.distance_weight

    # ==== Oscillation penalty (velocity sign flip detection) ====

    angvel_buffer = state.state_vars['angvel_buffer']
    angvel_t = angvel_buffer[0]  
    angvel_t_minus_1 = angvel_buffer[1]  
    angvel_t_minus_2 = angvel_buffer[2]  # Two steps ago: better to use t-2 and t-1 for more stable oscillation detection. TODO: Think this deeper
    roll_osc = (angvel_t_minus_1[0] * angvel_t_minus_2[0]) < 0
    pitch_osc = (angvel_t_minus_1[1] * angvel_t_minus_2[1]) < 0
    yaw_accel_t_minus_1 = angvel_t_minus_1[2] - angvel_t_minus_2[2]
    yaw_accel_t_minus_2 = angvel_t_minus_2[2] - angvel_buffer[3, 2]
    yaw_osc = (yaw_accel_t_minus_1 * yaw_accel_t_minus_2) < 0
    osc_flags = jp.array([roll_osc, pitch_osc, yaw_osc])
    r_osc = -self.cfg.oscillation_penalty * jp.mean(osc_flags.astype(jp.float32))

    # ==== Overshoot penalty ====

    quat_buffer = state.state_vars['quat_buffer']
    target_attitude_prev_euler = state.state_vars['target_attitude_buffer'][1]
    target_quat_prev = euler_to_quaternion(target_attitude_prev_euler[0], target_attitude_prev_euler[1], 0.0)
    quat_prev = quat_buffer[1]
    angvel_prev = state.state_vars['angvel_buffer'][1]
    quat_error_current = quaternion_error(current_quat, target_quat)
    quat_error_prev = quaternion_error(quat_prev, target_quat_prev)
    quat_overshoot = (quat_error_current * quat_error_prev) < 0
    yaw_rate_error_current = current_angvel[2] - target_attitude_euler[2]
    yaw_rate_error_prev = angvel_prev[2] - target_attitude_prev_euler[2]
    yaw_rate_overshoot = (yaw_rate_error_current * yaw_rate_error_prev) < 0
    any_overshoot = quat_overshoot | yaw_rate_overshoot
    ro = jp.where(any_overshoot, -self.cfg.overshoot_penalty, 0.0)
         
    # ==== Action chattering penalty ====

    action_buffer = state.state_vars.get('action_buffer', jp.zeros((5, action.shape[0])))
    current_action = action_buffer[0]  # Most recent action (from previous step)
    previous_action = action_buffer[1]  # Action before that
    action_change = jp.linalg.norm(current_action - previous_action)
    r_action_chattering = -self.cfg.action_chattering_weight * action_change
    
    # ==== Action penalty ====

    actuator_forces = state.pipeline_state.actuator_force
    r_action_penalty = -self.cfg.action_penalty_weight * jp.mean(jp.square(actuator_forces))
    
    # ==== Time penalty ====

    rt = -self.cfg.time_penalty
    
    # ==== Goal success bonus ====

    goal_achieved = jp.all(dist_t < self.cfg.success_threshold)
    steps_within_success = jp.where(
        goal_achieved,
        state.state_vars['steps_within_success'] + 1,
        0
    )
    success_bonus = jp.where(
        steps_within_success >= self.cfg.success_steps_required,
        self.cfg.success_bonus,
        0.0
    )
    
    # ==== Total reward ====

    total_reward = (r_distance + rt + ro + r_osc + 
                    r_action_chattering + r_action_penalty + success_bonus)
    

    # ==== Update variables ====

    # Update dynamic vars with reward-related state
    state_vars = state.state_vars.copy()
    state_vars.update({
        'goal_achieved': goal_achieved.astype(jp.float32),
        'steps_within_success': steps_within_success,
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
        'reward_total': total_reward,
        'roll_error': roll_error,
        'pitch_error': pitch_error,
        'yaw_rate_error': yaw_rate_error,
        'num_oscillating_axes': jp.sum(osc_flags.astype(jp.float32)),
        'roll_oscillating': roll_osc.astype(jp.float32),
        'pitch_oscillating': pitch_osc.astype(jp.float32),
        'yaw_oscillating': yaw_osc.astype(jp.float32),
    })


    return state.replace(
        reward=total_reward,
        metrics=metrics,    
        state_vars=state_vars
    )
