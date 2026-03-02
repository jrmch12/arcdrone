from jax import numpy as jp
import jax

def _get_obs_impl(self, state, action):
    """Compute observations from current physics state."""
    
    # ========== Extract data from MuJoCo and state ==========

    # Will collect propio data 
    # will run self.renderer.render! as in cartpole example!
    # Collect observation history!


    return state.replace(
        obs=obs,
        info=state_info
    )

    


