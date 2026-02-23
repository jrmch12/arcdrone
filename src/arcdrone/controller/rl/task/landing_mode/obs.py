from jax import numpy as jp


def _get_obs_impl(self, state, action):
    """Compute observations from current physics state."""

    position = state.pipeline_state.qpos[0:3]
    quat = state.pipeline_state.sensordata[0:4]
    angvel = state.pipeline_state.sensordata[4:7]
    linvel = state.pipeline_state.sensordata[10:13]
    target_pos = state.state_vars.get('target_pos', jp.zeros(3))

    full_state = jp.concatenate([
        position,
        linvel,
        quat,
        angvel,
        target_pos,
    ])

    obs = {
        "state": full_state,
        "privileged_state": full_state,
    }

    return state.replace(obs=obs)
