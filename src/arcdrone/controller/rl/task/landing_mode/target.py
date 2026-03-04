from jax import numpy as jp

def _get_target_impl(self, state):

    """ Update target  """
    

    target = jp.array([0.0, 0.0, 0.0])
    
    # Update state_vars with new target
    info = state.info.copy()
    info.update({
        'target': target,
        # also prepend into target_buffer for history if present
        'target_buffer': jp.concatenate([
            target[jp.newaxis, :],
            info.get('target_buffer', jp.tile(target, (self.cfg.buffer_size, 1)))[:-1, :]
        ], axis=0),
    })

    return state.replace(info=info)