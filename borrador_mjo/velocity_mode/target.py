from jax import numpy as jp

def _get_target_impl(self, state):

    """ Update target attitude (pitch, roll, yaw rate) for attitude control """
    
    # Set target attitude to zero (hover in place)
    # [roll, pitch, yaw_rate]
    target_vel = jp.zeros(3)
    
    # Update state_vars with new target
    state_vars = state.state_vars.copy()
    state_vars.update({
        'target_vel': target_vel
    })
    
    return state.replace(state_vars=state_vars)