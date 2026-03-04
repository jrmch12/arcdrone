from jax import numpy as jp


def _check_episode_events_impl(self, state):
    """Check if episode should terminate and some event flags needed for the reward fn."""

    target = state.info['target_buffer'][0]
    current_pos = state.info['pos_buffer'][0]
    # Compute position error and reduce to a scalar (Euclidean norm)
    pos_error_current = current_pos - target
    pos_error_norm = jp.linalg.norm(pos_error_current)
    # goal_achieved is a scalar boolean: True if within threshold
    goal_achieved = pos_error_norm < self.cfg.success_threshold
    # steps_within_success should be a scalar int (not an array per axis)
    steps_within_success = jp.where(
        goal_achieved,
        state.info['steps_within_success'] + 1,
        jp.array(0, dtype=state.info['steps_within_success'].dtype),
    )


    z_position = state.data.qpos[2]  # z-coordinate of drone position
    ground_violation = jp.maximum(0.0, self.cfg.ground_threshold_penalty - z_position)

  
    # Check termination conditions
    done = jp.logical_or(
        steps_within_success >= self.cfg.success_steps_required,
        state.info.get('step', 0) >= self.cfg.max_episode_steps,
    )
    
    # Add ground collision termination (z <= threshold means drone is too close to ground)
    ground_collision = z_position <= self.cfg.ground_threshold_event
    done = jp.logical_or(done, ground_collision)


    # Update dynamic vars with reward-related state
    state_info = state.info.copy()
    state_info.update({
        'goal_achieved': goal_achieved.astype(jp.float32),
        'steps_within_success': steps_within_success,
        'ground_violation': ground_violation,
    })


    return state.replace(done=done.astype(jp.float32),
        info=state_info,
    )

def _get_reward_impl(self, state, action):
    """Calculate reward from current state."""
    
    # ==== Extract data from MuJoCo and state ====
    
    # Current and target velocities
    target = state.info['target_buffer'][0]  
    current_pos = state.info['pos_buffer'][0]     
    
    # ==== Distance reward (velocity tracking) ====
    # Compute velocity error (Euclidean distance)
    pos_error = jp.linalg.norm(target - current_pos)
    r_distance = jp.exp(-self.cfg.distance_scale * pos_error) * self.cfg.distance_weight

    # ==== Oscillation penalty (velocity component sign flip detection) ====
    # Detect oscillations in each velocity component
    pos_buffer = state.info['pos_buffer']
    pos_t = pos_buffer[0]
    pos_t_minus_1 = pos_buffer[1]
    pos_t_minus_2 = pos_buffer[2]
    
    # Check if position changed sign between t-2 and t-1 (oscillation indicator)
    posx_osc = (pos_t_minus_1[0] * pos_t_minus_2[0]) < 0
    posy_osc = (pos_t_minus_1[1] * pos_t_minus_2[1]) < 0
    posz_osc = (pos_t_minus_1[2] * pos_t_minus_2[2]) < 0
    
    osc_flags = jp.array([posx_osc, posy_osc, posz_osc])
    r_osc = -self.cfg.oscillation_penalty * jp.mean(osc_flags.astype(jp.float32))

    # ==== Overshoot penalty ====
    # Detect if position error changed sign (crossed target)
    target_prev = state.info['target_buffer'][1]
    pos_prev = pos_buffer[1]
    
    # Position errors
    pos_error_current = current_pos - target  # [x_err, y_err, z_err]
    pos_error_prev = pos_prev - target_prev
    
    # Check if error changed sign in any component (overshoot)
    posx_overshoot = (pos_error_current[0] * pos_error_prev[0]) < 0
    posy_overshoot = (pos_error_current[1] * pos_error_prev[1]) < 0
    posz_overshoot = (pos_error_current[2] * pos_error_prev[2]) < 0
    
    any_overshoot = posx_overshoot | posy_overshoot | posz_overshoot
    ro = jp.where(any_overshoot, -self.cfg.overshoot_penalty, 0.0)
         
    # # ==== Action chattering penalty ====

    action_buffer = state.info.get('action_buffer', jp.zeros((5, action.shape[0])))
    current_action = action_buffer[0]  # Most recent action (from previous step)
    previous_action = action_buffer[1]  # Action before that
    action_change = jp.linalg.norm(current_action - previous_action)
    r_action_chattering = -self.cfg.action_chattering_weight * action_change
    
    # # ==== Action penalty ====

    actuator_forces = state.data.actuator_force
    r_action_penalty = -self.cfg.action_penalty_weight * jp.mean(jp.square(actuator_forces))
    
    # ==== Time penalty ====

    rt = -self.cfg.time_penalty
    
    # ==== Ground penalty ====

    # Heavily penalize being near or at ground level to prevent falling
    ground_violation = state.info['ground_violation']
    r_ground = -self.cfg.ground_penalty_weight * jp.square(ground_violation / self.cfg.ground_threshold_penalty)
    
    # ==== Goal success bonus ====


    steps_within_success = state.info['steps_within_success']
    success_bonus = jp.where(
        steps_within_success >= self.cfg.success_steps_required,
        self.cfg.success_bonus,
        0.0
    )
    
    # ==== Total reward ====

    total_reward = (r_distance + rt + ro + r_osc + r_action_chattering + r_action_penalty +
                    r_ground + success_bonus)
    

    # ==== Update variables ====


    
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
    )
