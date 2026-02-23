from jax import numpy as jp


def _get_reward_impl(self, state, action):
    """Calculate reward from current state."""
    
    # ==== Extract data from MuJoCo and state ====
    
    # Current and target velocities
    target_vel = state.state_vars['target_vel_buffer'][0]  # [vx, vy, vz] target velocity
    current_vel = state.state_vars['pos_buffer'][0]     # [vx, vy, vz] current velocity
    
    # ==== Distance reward (velocity tracking) ====
    # Compute velocity error (Euclidean distance)
    vel_error = jp.linalg.norm(target_vel - current_vel)
    r_distance = jp.exp(-self.cfg.distance_scale * vel_error) * self.cfg.distance_weight

    # ==== Oscillation penalty (velocity component sign flip detection) ====
    # Detect oscillations in each velocity component
    linvel_buffer = state.state_vars['pos_buffer']
    linvel_t = linvel_buffer[0]  
    linvel_t_minus_1 = linvel_buffer[1]  
    linvel_t_minus_2 = linvel_buffer[2]  
    
    # Check if velocity changed sign between t-2 and t-1 (oscillation indicator)
    vx_osc = (linvel_t_minus_1[0] * linvel_t_minus_2[0]) < 0
    vy_osc = (linvel_t_minus_1[1] * linvel_t_minus_2[1]) < 0
    vz_osc = (linvel_t_minus_1[2] * linvel_t_minus_2[2]) < 0
    
    osc_flags = jp.array([vx_osc, vy_osc, vz_osc])
    r_osc = -self.cfg.oscillation_penalty * jp.mean(osc_flags.astype(jp.float32))

    # ==== Overshoot penalty ====
    # Detect if velocity error changed sign (crossed target)
    target_vel_prev = state.state_vars['target_vel_buffer'][1]
    linvel_prev = linvel_buffer[1]
    
    # Velocity errors
    vel_error_current = current_vel - target_vel  # [vx_err, vy_err, vz_err]
    vel_error_prev = linvel_prev - target_vel_prev
    
    # Check if error changed sign in any component (overshoot)
    vx_overshoot = (vel_error_current[0] * vel_error_prev[0]) < 0
    vy_overshoot = (vel_error_current[1] * vel_error_prev[1]) < 0
    vz_overshoot = (vel_error_current[2] * vel_error_prev[2]) < 0
    
    any_overshoot = vx_overshoot | vy_overshoot | vz_overshoot
    ro = jp.where(any_overshoot, -self.cfg.overshoot_penalty, 0.0)
         
    # # ==== Action chattering penalty ====

    action_buffer = state.state_vars.get('action_buffer', jp.zeros((5, action.shape[0])))
    current_action = action_buffer[0]  # Most recent action (from previous step)
    previous_action = action_buffer[1]  # Action before that
    action_change = jp.linalg.norm(current_action - previous_action)
    r_action_chattering = -self.cfg.action_chattering_weight * action_change
    
    # # ==== Action penalty ====

    actuator_forces = state.pipeline_state.actuator_force
    r_action_penalty = -self.cfg.action_penalty_weight * jp.mean(jp.square(actuator_forces))
    
    # ==== Time penalty ====

    rt = -self.cfg.time_penalty
    
    # ==== Ground penalty ====

    # Heavily penalize being near or at ground level to prevent falling
    z_position = state.pipeline_state.qpos[2]  # z-coordinate of drone position
    ground_violation = jp.maximum(0.0, self.cfg.ground_threshold - z_position)
    r_ground = -self.cfg.ground_penalty_weight * jp.square(ground_violation / self.cfg.ground_threshold)
    
    # ==== Goal success bonus ====

    goal_achieved = vel_error < self.cfg.success_threshold
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

    total_reward = (r_distance + rt + ro + r_osc + r_action_chattering + r_action_penalty +
                    r_ground + success_bonus)
    

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
        'reward_ground_penalty': r_ground,
        'reward_success_bonus': success_bonus,
        'reward_total': total_reward,
    })


    return state.replace(
        reward=total_reward,
        metrics=metrics,    
        state_vars=state_vars
    )
