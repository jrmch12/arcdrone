from jax import numpy as jp

def _get_target_impl(self, state):

    """ Update target attitude (pitch, roll, yaw rate) for attitude control """
    
    # Set target attitude to zero (hover in place)
    # [roll, pitch, yaw_rate]
    target_vel = jp.array([0.0, 0.0, 1.5])
    
    # Update state_vars with new target
    info = state.info.copy()
    info.update({
        'target_vel': target_vel,
        # also prepend into target_vel_buffer for history if present
        'target_vel_buffer': jp.concatenate([
            target_vel[jp.newaxis, :],
            info.get('target_vel_buffer', jp.tile(target_vel, (self.cfg.buffer_size, 1)))[:-1, :]
        ], axis=0),
    })

    return state.replace(info=info)