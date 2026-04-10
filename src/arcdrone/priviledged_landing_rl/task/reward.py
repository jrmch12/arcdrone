from jax import numpy as jp


def _quat_rotate(quat, vec):
    """Rotate vec by quaternion (w, x, y, z)."""
    w = quat[0]
    xyz = quat[1:4]
    t = 2.0 * jp.cross(xyz, vec)
    return vec + w * t + jp.cross(xyz, t)


# def _check_episode_events_impl(self, state):
#     """Check if episode should terminate and some event flags needed for the reward fn."""

#     target = state.info['target_buffer'][0]
#     current_pos = state.info['pos_buffer'][0]
#     # Compute position error and reduce to a scalar (Euclidean norm)
#     pos_error_current = current_pos - target
#     pos_error_norm = jp.linalg.norm(pos_error_current)
#     # goal_achieved is a scalar boolean: True if within threshold
#     goal_achieved = pos_error_norm < self.cfg.success_threshold
#     # steps_within_success should be a scalar int (not an array per axis)
#     steps_within_success = jp.where(
#         goal_achieved,
#         state.info['steps_within_success'] + 1,
#         jp.array(0, dtype=state.info['steps_within_success'].dtype),
#     )


#     z_position = state.data.qpos[2]  # z-coordinate of drone position
#     xy_radius = jp.linalg.norm(current_pos[0:2])
#     in_landing_cylinder = xy_radius <= 0.25  # diameter 0.5 m centered at (0, 0)

#     ground_violation_raw = jp.maximum(0.0, self.cfg.ground_threshold_penalty - z_position)
#     ground_violation = jp.where(in_landing_cylinder, 0.0, ground_violation_raw)

  
#     # Check termination conditions
#     done = jp.logical_or(
#         steps_within_success >= self.cfg.success_steps_required,
#         state.info.get('step', 0) >= self.cfg.max_episode_steps,
#     )

#     # Add ground collision termination (disabled inside the landing cylinder)
#     ground_collision = jp.logical_and(
#         z_position <= self.cfg.ground_threshold_event,
#         jp.logical_not(in_landing_cylinder),
#     )
#     done = jp.logical_or(done, ground_collision)


#     # Update dynamic vars with reward-related state
#     state_info = state.info.copy()
#     state_info.update({
#         'goal_achieved': goal_achieved.astype(jp.float32),
#         'steps_within_success': steps_within_success,
#         'ground_violation': ground_violation,
#     })


#     return state.replace(done=done.astype(jp.float32),
#         info=state_info,
#     )


def _check_episode_events_impl(self, state):
    """Check if episode should terminate and some event flags needed for the reward fn."""

    target = state.info['target_buffer'][0]
    current_pos = state.info['pos_buffer'][0]

    z_position = state.data.qpos[2]
    xy_radius = jp.linalg.norm(current_pos[0:2])
    in_landing_cylinder = xy_radius <= 0.4 # TODO: is defined twice on this script

    ground_violation_raw = jp.maximum(0.0, self.cfg.ground_threshold_penalty - z_position)
    ground_violation = jp.where(in_landing_cylinder, 0.0, ground_violation_raw)

    # ── Contact with fiducial plate ───────────────────────────────
    # MJX Data may not expose `contact` (unlike MuJoCo CPU). Fall back to
    # a geometric heuristic when contact info is unavailable.
    
    # Fiducial mesh in scene_mocap.xml spans [-0.3, 0.3] in x/y and
    # is offset to z=0.002 with thickness 0.004 -> top at z=0.006.
    plate_half_size = getattr(self.cfg, "fiducial_half_size", 0.3)
    plate_top_z = getattr(self.cfg, "fiducial_top_z", 0.006)
    contact_tol = getattr(self.cfg, "fiducial_contact_tol", 0.05)
    touching_fiducial = jp.logical_and(
        jp.logical_and(
            jp.abs(current_pos[0]) <= plate_half_size,
            jp.abs(current_pos[1]) <= plate_half_size,
        ),
        z_position <= plate_top_z + contact_tol,
    )


    # ── Consecutive steps touching fiducial ───────────────────────
    # steps_within_success = jp.where(
    #     touching_fiducial,
    #     state.info['steps_within_success'] + 1,
    #     jp.array(0, dtype=state.info['steps_within_success'].dtype),
    # )
    goal_achieved = touching_fiducial  # for logging/reward use

    # ── Termination conditions ────────────────────────────────────
    # done = jp.logical_or(
    #     steps_within_success >= self.cfg.success_steps_required,
    #     state.info.get('step', 0) >= self.cfg.max_episode_steps,
    # )
    done = state.info.get('step', 0) >= self.cfg.max_episode_steps

    # Ground collision outside landing cylinder
    ground_collision = jp.logical_and(
        z_position <= self.cfg.ground_threshold_event,
        jp.logical_not(in_landing_cylinder),
    )
    done = jp.logical_or(done, ground_collision)

    # ── Update state info ─────────────────────────────────────────
    state_info = state.info.copy()
    state_info.update({
        'goal_achieved': goal_achieved.astype(jp.float32),
        'steps_within_success': jp.zeros_like(state.info['steps_within_success']),
        'ground_violation': ground_violation,
    })

    return state.replace(
        done=done.astype(jp.float32),
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
    r_distance = -pos_error * 0.01 + r_distance

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
    

    # Lets add three conservative rewards. A reward or penalty that encourage the drone to have a horizontal attitude. And a reward or penalty that the drone to have low angular and linear velocities.


    attitude_weight = 0.15
    angvel_weight = 0.005
    linvel_weight = 0.003



    quat = state.info['quat_buffer'][0]
    quat = quat / jp.maximum(jp.linalg.norm(quat), 1e-8)
    cos_tilt = jp.clip(1.0 - 2.0 * (quat[1] * quat[1] + quat[2] * quat[2]), -1.0, 1.0)
    tilt_error = 1.0 - cos_tilt
    r_attitude_level = -attitude_weight * tilt_error

    angvel = state.info['angvel_buffer'][0]
    linvel = state.info['linvel_buffer'][0]
    r_low_angvel = -angvel_weight * jp.mean(jp.square(angvel))
    r_low_linvel = -linvel_weight * jp.mean(jp.square(linvel))

    # reward for low velocity at low altitudes, to smooth landing
    # only active inside the landing cylinder (r <= 0.25 m from origin)
    in_landing_cylinder = jp.linalg.norm(current_pos[0:2]) <= 0.4

    z = state.data.qpos[2]                  # height
    vz = state.info['linvel_buffer'][0][2]  # vertical velocity

    # # --- constants ---
    # k = 5.0    # height sharpness
    # c = 25.0   # velocity sharpness
    # weight = 10.0  # max reward at z=0, vz=0

    # # height shaping: 0 at z=1m, 1 at z=0m
    # exp_k = jp.exp(-k)
    # height_term = (jp.exp(-k * z) - exp_k) / (1.0 - exp_k)
    # height_term = jp.clip(height_term, 0.0, 1.0)

    # # velocity shaping: 1 at vz=0, smooth decay
    # vel_term = jp.exp(-c * vz**2)


    # Soft landing reward — only inside landing cylinder
    k       = 20.0    # height sharpness (0 at z=1m, 1 at z=0m)
    soft_weight  = 0.25    # max reward magnitude

    exp_k       = jp.exp(-k)
    height_term = (jp.exp(-k * z) - exp_k) / (1.0 - exp_k)
    height_term = jp.clip(height_term, 0.0, 1.0)

    vz_desired  = -0.05  # m/s gentle descent target
    c_vz        = 40.0
    c_vxy       = 30.0

    vxy = jp.linalg.norm(state.info['linvel_buffer'][0][0:2])
    vel_z_term  = jp.exp(-c_vz * (vz - vz_desired)**2)
    vel_xy_term = jp.exp(-c_vxy * vxy**2)
    vel_term    = vel_z_term * vel_xy_term

    r_soft_landing = jp.where(in_landing_cylinder, soft_weight * height_term * vel_term, 0.0)


    # ==== Ground penalty ====

    # Heavily penalize being near or at ground level to prevent falling
    ground_violation = state.info['ground_violation']
    r_ground = -self.cfg.ground_penalty_weight * jp.square(ground_violation / self.cfg.ground_threshold_penalty)
    
    # ==== Goal success bonus ====


    # steps_within_success = state.info['steps_within_success']
    goal_achieved = state.info.get('goal_achieved', jp.array(0.0, dtype=jp.float32))
    success_bonus = jp.where(
        goal_achieved,
        self.cfg.success_bonus,
        0.0
    )
    
    # # Testing !

    # # Hard crash penalty — penalize fast descent near ground inside landing cylinder
    # _IMPACT_TOL    = 0.3   # m/s downward speed allowed before penalty kicks in
    # _CRASH_Z       = 0.15  # m altitude below which penalty is active
    # _CRASH_WEIGHT  = 20  # penalty scale

    # impact_speed = jp.maximum(0.0, -vz - _IMPACT_TOL)
    # r_crash_penalty = jp.where(
    #     jp.logical_and(in_landing_cylinder, z < _CRASH_Z),
    #     -_CRASH_WEIGHT * impact_speed,
    #     0.0
    # )



    # ==== Camera alignment reward ====
    # Reward the camera gimbal for pointing toward the landing target.
    # Camera forward in mount-local frame = (-cos(tilt), 0, sin(tilt))
    # where tilt = qpos[7] (hinge around y-axis, range [-pi/2, 0]).
    cam_alignment_weight = float(getattr(self.cfg, 'camera_alignment_weight', 0.3))

    pos = state.data.qpos[0:3]
    body_quat = state.data.qpos[3:7]
    tilt = state.data.qpos[7]

    cam_fwd_body = jp.array([-jp.cos(tilt), 0.0, jp.sin(tilt)])
    cam_fwd_world = _quat_rotate(body_quat, cam_fwd_body)

    to_target = target - pos
    to_target_norm = to_target / (jp.linalg.norm(to_target) + 1e-8)

    alignment = jp.dot(cam_fwd_world, to_target_norm)  # [-1, 1]
    r_camera = cam_alignment_weight * alignment

    # ==== Total reward ====

    total_reward = (
        r_distance
        + rt
        + ro
        + r_osc
        + r_action_chattering
        + r_action_penalty
        + r_attitude_level
        + r_low_angvel
        + r_low_linvel
        + r_soft_landing
        + r_ground
        + success_bonus
        + r_camera
        # + r_crash_penalty
    )
    

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
        'reward_attitude_level': r_attitude_level,
        'reward_low_angvel': r_low_angvel,
        'reward_low_linvel': r_low_linvel,
        'reward_soft_landing': r_soft_landing,
        'reward_ground_penalty': r_ground,
        'reward_success_bonus': success_bonus,
        'reward_crash_penalty': 0.0,
        'reward_camera_alignment': r_camera,
        'reward_total': total_reward,
        
    })


    return state.replace(
        reward=total_reward,
        metrics=metrics,
    )
