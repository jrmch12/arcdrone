from jax import numpy as jp


def _extract_state(self, state):
    position = state.pipeline_state.qpos[0:3]
    quat = state.pipeline_state.sensordata[0:4]
    angvel = state.pipeline_state.sensordata[4:7]
    linvel = state.pipeline_state.sensordata[10:13]
    target_pos = state.state_vars['target_pos']

    xy_error = jp.linalg.norm(position[0:2] - target_pos[0:2])
    z_error = jp.abs(position[2] - target_pos[2])
    linvel_norm = jp.linalg.norm(linvel)
    angvel_norm = jp.linalg.norm(angvel)

    cos_tilt = 1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2)
    cos_tilt = jp.clip(cos_tilt, -1.0, 1.0)

    prev_xy_error = state.state_vars.get('prev_xy_error', xy_error)
    prev_z_error = state.state_vars.get('prev_z_error', z_error)

    return {
        'position': position,
        'xy_error': xy_error,
        'z_error': z_error,
        'linvel_norm': linvel_norm,
        'angvel_norm': angvel_norm,
        'cos_tilt': cos_tilt,
        'prev_xy_error': prev_xy_error,
        'prev_z_error': prev_z_error,
    }


def _build_common_terms(self, state, s):
    action_effort = jp.mean(jp.square(state.pipeline_state.actuator_force))
    tilt_error = 1.0 - s['cos_tilt']

    r_xy = self.cfg.xy_dense_weight * jp.exp(-self.cfg.xy_exp_scale * s['xy_error'])
    r_z = self.cfg.z_dense_weight * jp.exp(-self.cfg.z_exp_scale * s['z_error'])
    r_xy_progress = jp.array(0.0)
    r_z_progress = jp.array(0.0)
    r_upright = self.cfg.upright_dense_weight * jp.exp(-self.cfg.upright_exp_scale * tilt_error)

    r_linvel_penalty = self.cfg.linvel_dense_weight * jp.exp(-self.cfg.linvel_exp_scale * s['linvel_norm'])
    r_angvel_penalty = self.cfg.angvel_dense_weight * jp.exp(-self.cfg.angvel_exp_scale * s['angvel_norm'])
    r_action_penalty = self.cfg.action_dense_weight * jp.exp(-self.cfg.action_exp_scale * action_effort)
    r_time_penalty = jp.array(0.0)

    return {
        'reward_xy_alignment': r_xy,
        'reward_z_alignment': r_z,
        'reward_xy_progress': r_xy_progress,
        'reward_z_progress': r_z_progress,
        'reward_upright': r_upright,
        'reward_linvel_penalty': r_linvel_penalty,
        'reward_angvel_penalty': r_angvel_penalty,
        'reward_action_penalty': r_action_penalty,
        'reward_time_penalty': r_time_penalty,
    }


def _finalize_reward(self, state, s, components, success_condition):
    steps_within_success = jp.where(
        success_condition,
        state.state_vars['steps_within_success'] + 1,
        0,
    )

    stage_bonus = jp.where(
        steps_within_success >= self.cfg.success_steps_required,
        self.cfg.success_bonus,
        0.0,
    )

    stage_progress = jp.array(0.0)

    crashed = jp.logical_and(s['position'][2] <= self.cfg.crash_height, jp.logical_not(success_condition))
    crash_penalty = jp.where(crashed, -self.cfg.crash_penalty, 0.0)

    total_reward = (
        components['reward_xy_alignment']
        + components['reward_z_alignment']
        + components['reward_xy_progress']
        + components['reward_z_progress']
        + components['reward_upright']
        + components['reward_linvel_penalty']
        + components['reward_angvel_penalty']
        + components['reward_action_penalty']
        + components['reward_time_penalty']
        + stage_progress
        + stage_bonus
        + crash_penalty
    )

    state_vars = state.state_vars.copy()
    state_vars.update({
        'goal_achieved': success_condition.astype(jp.float32),
        'steps_within_success': steps_within_success,
        'prev_xy_error': s['xy_error'],
        'prev_z_error': s['z_error'],
    })

    metrics = state.metrics.copy()
    metrics.update({
        **components,
        'reward_touchdown_progress': stage_progress,
        'reward_touchdown_bonus': stage_bonus,
        'reward_crash_penalty': crash_penalty,
        'reward_total': total_reward,
    })

    return state.replace(
        reward=total_reward,
        metrics=metrics,
        state_vars=state_vars,
    )


def _get_reward_hover_impl(self, state):
    s = _extract_state(self, state)
    hover_z_error = jp.abs(s['position'][2] - self.cfg.hover_target_z)
    hover_prev_z_error = state.state_vars.get('prev_hover_z_error', hover_z_error)

    components = _build_common_terms(self, state, s)
    components['reward_z_alignment'] = self.cfg.hover_z_dense_weight * jp.exp(-self.cfg.hover_z_exp_scale * hover_z_error)
    components['reward_z_progress'] = self.cfg.hover_z_progress_weight * (hover_prev_z_error - hover_z_error)

    hover_success = jp.logical_and(
        jp.logical_and(s['xy_error'] < self.cfg.hover_xy_success_threshold, hover_z_error < self.cfg.hover_z_success_threshold),
        jp.logical_and(s['linvel_norm'] < self.cfg.hover_linvel_success_threshold, s['angvel_norm'] < self.cfg.hover_angvel_success_threshold),
    )

    updated_state = _finalize_reward(self, state, s, components, hover_success)
    updated_state_vars = updated_state.state_vars.copy()
    updated_state_vars.update({'prev_hover_z_error': hover_z_error})
    return updated_state.replace(state_vars=updated_state_vars)


def _get_reward_approach_impl(self, state):
    s = _extract_state(self, state)
    components = _build_common_terms(self, state, s)

    approach_success = jp.logical_and(
        s['xy_error'] < self.cfg.approach_xy_success_threshold,
        s['z_error'] < self.cfg.approach_z_success_threshold,
    )
    return _finalize_reward(self, state, s, components, approach_success)


def _get_reward_landing_impl(self, state):
    s = _extract_state(self, state)
    components = _build_common_terms(self, state, s)

    touchdown = jp.logical_and(
        jp.logical_and(s['xy_error'] < self.cfg.xy_success_threshold, s['position'][2] < self.cfg.z_success_threshold),
        jp.logical_and(
            s['linvel_norm'] < self.cfg.linvel_success_threshold,
            jp.logical_and(
                s['angvel_norm'] < self.cfg.angvel_success_threshold,
                s['cos_tilt'] > self.cfg.upright_success_threshold,
            ),
        ),
    )
    return _finalize_reward(self, state, s, components, touchdown)


def _get_reward_impl(self, state, action):
    stage = self.cfg.get('curriculum_stage', 'landing')

    if stage == 'hover':
        return _get_reward_hover_impl(self, state)
    if stage == 'approach':
        return _get_reward_approach_impl(self, state)
    return _get_reward_landing_impl(self, state)
