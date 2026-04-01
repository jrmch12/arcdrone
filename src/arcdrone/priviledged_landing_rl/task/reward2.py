from jax import numpy as jp

# ── Hardcoded constants (edit here, no yaml needed) ───────────────────────────

# Episode
MAX_STEPS           = 500
GROUND_THRESHOLD    = 0.05      # z below this = ground collision (outside safe zone)

# Safe landing zone — no ground collision termination inside this radius
SAFE_RADIUS         = 0.6       # m, wide enough to not punish near-landing drift

# Fiducial plate (success detection)
PLATE_HALF_SIZE     = 0.3       # m, x/y half-size of landing pad
PLATE_TOP_Z         = 0.006     # m, top of fiducial plate
CONTACT_TOL         = 0.05      # m, tolerance above plate top to count as touching
SUCCESS_STEPS       = 10        # consecutive steps on plate to count as landed

# Reward weights
W_DISTANCE          = 2.0       # reward for being close to target (landing pad)
W_DISTANCE_SCALE    = 1.5       # sharpness of exponential distance reward
W_ATTITUDE          = 1.0       # penalty for tilting
W_ANGVEL            = 0.02      # penalty for angular velocity (keep low to avoid explosion)
W_LINVEL            = 0.02      # penalty for linear velocity
W_ACTION            = 0.01      # penalty for large actuator forces
W_SOFT_LANDING      = 1.0       # reward for slow descent inside safe zone
W_SUCCESS_BONUS     = 50.0      # one-time bonus per step on the pad
W_TIME              = 0.05      # small per-step time penalty


def _check_episode_events_impl(self, state):
    """Check if episode should terminate and compute event flags for reward."""

    current_pos = state.info['pos_buffer'][0]
    z_position  = state.data.qpos[2]
    xy_radius   = jp.linalg.norm(current_pos[0:2])

    # ── Success: touching the fiducial plate ──────────────────────
    touching_fiducial = jp.logical_and(
        jp.logical_and(
            jp.abs(current_pos[0]) <= PLATE_HALF_SIZE,
            jp.abs(current_pos[1]) <= PLATE_HALF_SIZE,
        ),
        z_position <= PLATE_TOP_Z + CONTACT_TOL,
    )

    steps_within_success = jp.where(
        touching_fiducial,
        state.info['steps_within_success'] + 1,
        jp.array(0, dtype=state.info['steps_within_success'].dtype),
    )

    # ── Termination ───────────────────────────────────────────────
    # 1. Max steps
    done = state.info.get('step', 0) >= MAX_STEPS

    # 2. Stayed on pad long enough
    done = jp.logical_or(done, steps_within_success >= SUCCESS_STEPS)

    # 3. Ground collision — only outside the safe radius
    in_safe_zone      = xy_radius <= SAFE_RADIUS
    ground_collision  = jp.logical_and(
        z_position <= GROUND_THRESHOLD,
        jp.logical_not(in_safe_zone),
    )
    done = jp.logical_or(done, ground_collision)

    # ── Update state ──────────────────────────────────────────────
    state_info = state.info.copy()
    state_info.update({
        'goal_achieved':       touching_fiducial.astype(jp.float32),
        'steps_within_success': steps_within_success,
        'ground_violation':    jp.array(0.0),   # unused but kept for compat
    })

    return state.replace(
        done=done.astype(jp.float32),
        info=state_info,
    )


def _get_reward_impl(self, state, action):
    """Calculate reward. Goal: land on the fiducial plate."""

    current_pos = state.info['pos_buffer'][0]
    z           = state.data.qpos[2]

    # ── 1. Distance to landing pad (target = origin at ground) ────
    # Target is the pad center: [0, 0, 0]
    landing_target = jp.zeros(3)
    pos_error      = jp.linalg.norm(current_pos - landing_target)
    r_distance     = W_DISTANCE * jp.exp(-W_DISTANCE_SCALE * pos_error)

    # ── 2. Attitude (keep level) ───────────────────────────────────
    quat     = state.info['quat_buffer'][0]
    quat     = quat / jp.maximum(jp.linalg.norm(quat), 1e-8)
    cos_tilt = jp.clip(1.0 - 2.0 * (quat[1]**2 + quat[2]**2), -1.0, 1.0)
    r_attitude = -W_ATTITUDE * (1.0 - cos_tilt)

    # ── 3. Angular velocity penalty ───────────────────────────────
    angvel     = state.info['angvel_buffer'][0]
    r_angvel   = -W_ANGVEL * jp.sum(jp.square(angvel))

    # ── 4. Linear velocity penalty ────────────────────────────────
    linvel     = state.info['linvel_buffer'][0]
    r_linvel   = -W_LINVEL * jp.sum(jp.square(linvel))

    # ── 5. Action smoothness penalty ──────────────────────────────
    forces     = state.data.actuator_force
    r_action   = -W_ACTION * jp.mean(jp.square(forces))

    # ── 6. Soft landing: slow descent inside safe zone ────────────
    in_safe_zone = jp.linalg.norm(current_pos[0:2]) <= SAFE_RADIUS
    vz           = state.info['linvel_buffer'][0][2]
    vxy          = jp.linalg.norm(state.info['linvel_buffer'][0][0:2])

    # Reward: close to ground + slow vz (target -0.1 m/s) + low vxy
    height_term = jp.exp(-5.0 * z)                        # peaks near ground
    vz_term     = jp.exp(-20.0 * (vz + 0.1)**2)           # peaks at vz = -0.1 m/s
    vxy_term    = jp.exp(-10.0 * vxy**2)                  # peaks at vxy = 0
    r_soft      = jp.where(
        in_safe_zone,
        W_SOFT_LANDING * height_term * vz_term * vxy_term,
        0.0,
    )

    # ── 7. Success bonus ──────────────────────────────────────────
    goal_achieved = state.info.get('goal_achieved', jp.array(0.0, dtype=jp.float32))
    r_success     = jp.where(goal_achieved > 0.0, W_SUCCESS_BONUS, 0.0)

    # ── 8. Time penalty ───────────────────────────────────────────
    r_time = -W_TIME

    # ── Total ─────────────────────────────────────────────────────
    total_reward = (
        r_distance
        + r_attitude
        + r_angvel
        + r_linvel
        + r_action
        + r_soft
        + r_success
        + r_time
    )

    # ── Metrics ───────────────────────────────────────────────────
    metrics = state.metrics.copy()
    metrics.update({
        'reward_distance':      r_distance,
        'reward_time_penalty':  r_time,
        'reward_overshoot':     0.0,
        'reward_oscillation':   0.0,
        'reward_action_chattering': 0.0,
        'reward_action_penalty': r_action,
        'reward_attitude_level': r_attitude,
        'reward_low_angvel':    r_angvel,
        'reward_low_linvel':    r_linvel,
        'reward_soft_landing':  r_soft,
        'reward_ground_penalty': 0.0,
        'reward_success_bonus': r_success,
        'reward_crash_penalty': 0.0,
        'reward_total':         total_reward,
    })

    return state.replace(
        reward=total_reward,
        metrics=metrics,
    )