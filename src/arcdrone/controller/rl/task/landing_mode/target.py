from jax import numpy as jp


def _get_target_impl(self, state):
    stage = self.cfg.get('curriculum_stage', 'landing')

    target_pos = jp.array([
        self.cfg.landing_target_x,
        self.cfg.landing_target_y,
        self.cfg.hover_target_z if stage == 'hover' else self.cfg.landing_target_z,
    ])

    state_vars = state.state_vars.copy()
    state_vars.update({'target_pos': target_pos})
    return state.replace(state_vars=state_vars)
