from jax import numpy as jp
import jax

def _get_obs_impl(self, state, action):
    """Compute observations from current physics state."""
    
    # ========== Extract data from MuJoCo and state ==========
    quat = state.pipeline_state.sensordata[0:4]    # body_quat (4D)
    angvel = state.pipeline_state.sensordata[4:7]  # body_angvel (3D) - angular velocity
    linacc = state.pipeline_state.sensordata[7:10] # body_linacc (3D) - already without gravity
    linvel = state.pipeline_state.sensordata[10:13] # body_linvel (3D) - linear velocity
    
    # Get target attitude from state_vars (roll, pitch, yaw_rate)
    target_attitude = state.state_vars.get('target_attitude', jp.zeros(3))


    
    # ========== Add sensor noise ==========
    rng = state.info.get('rng', jax.random.PRNGKey(0))
    rng_linacc, rng_quat, rng_angvel, rng_linvel = jax.random.split(rng, 4)
    
    linacc_noisy = linacc + jax.random.normal(rng_linacc, shape=(3,)) * self.cfg.linacc_noise
    
    quat_noisy = quat + jax.random.normal(rng_quat, shape=(4,)) * self.cfg.quat_noise
    quat_norm = jp.linalg.norm(quat_noisy)
    quat_noisy = quat_noisy / jp.maximum(quat_norm, 1e-8)
    
    angvel_noisy = angvel + jax.random.normal(rng_angvel, shape=(3,)) * self.cfg.angvel_noise
    
    linvel_noisy = linvel + jax.random.normal(rng_linvel, shape=(3,)) * self.cfg.linvel_noise
    

    # ========== Update buffers (FIFO: newest at index 0) ==========
    # Update action buffer
    action_buffer = jp.concatenate([
        action[jp.newaxis, :],
        state.state_vars["action_buffer"][:-1, :]
    ], axis=0)
    
    # Update target attitude buffer (roll, pitch, yaw_rate)
    target_vel_buffer = jp.concatenate([
        target_attitude[jp.newaxis, :],
        state.state_vars["target_vel_buffer"][:-1, :]
    ], axis=0)
    
    # Update noisy sensor buffers (for actor)
    linacc_buffer_noisy = jp.concatenate([
        linacc_noisy[jp.newaxis, :],
        state.state_vars["linacc_buffer_noisy"][:-1, :]
    ], axis=0)
    
    quat_buffer_noisy = jp.concatenate([
        quat_noisy[jp.newaxis, :],
        state.state_vars["quat_buffer_noisy"][:-1, :]
    ], axis=0)
    
    angvel_buffer_noisy = jp.concatenate([
        angvel_noisy[jp.newaxis, :],
        state.state_vars["angvel_buffer_noisy"][:-1, :]
    ], axis=0)
    
    linvel_buffer_noisy = jp.concatenate([
        linvel_noisy[jp.newaxis, :],
        state.state_vars["linvel_buffer_noisy"][:-1, :]
    ], axis=0)
    
    # Update clean sensor buffers (for critic)
    linacc_buffer = jp.concatenate([
        linacc[jp.newaxis, :],
        state.state_vars["linacc_buffer"][:-1, :]
    ], axis=0)
    
    quat_buffer = jp.concatenate([
        quat[jp.newaxis, :],
        state.state_vars["quat_buffer"][:-1, :]
    ], axis=0)
    
    angvel_buffer = jp.concatenate([
        angvel[jp.newaxis, :],
        state.state_vars["angvel_buffer"][:-1, :]
    ], axis=0)
    
    linvel_buffer = jp.concatenate([
        linvel[jp.newaxis, :],
        state.state_vars["linvel_buffer"][:-1, :]
    ], axis=0)
    


    # ========== Build observations ==========
    # Actor observation (noisy sensors - what the real robot would see)
    obs_actor = jp.concatenate([
        linacc_buffer_noisy.flatten(),
        linvel_buffer_noisy.flatten(),
        quat_buffer_noisy.flatten(),
        angvel_buffer_noisy.flatten(),
        target_vel_buffer.flatten(),
        action_buffer.flatten()
    ])
    
    # Critic observation (privileged - clean sensors)
    obs_critic = jp.concatenate([
        linacc_buffer.flatten(),
        linvel_buffer.flatten(),
        quat_buffer.flatten(),
        angvel_buffer.flatten(),
        target_vel_buffer.flatten(),
        action_buffer.flatten()
    ])
    
    obs = {
        "state": obs_actor,
        "privileged_state": obs_critic
    }
    
    # ========== Update state variables ==========
    state_vars = state.state_vars.copy()
    state_vars.update({
        'action_buffer': action_buffer,
        'target_vel_buffer': target_vel_buffer,
        'linacc_buffer_noisy': linacc_buffer_noisy,
        'quat_buffer_noisy': quat_buffer_noisy,
        'angvel_buffer_noisy': angvel_buffer_noisy,
        'linvel_buffer_noisy': linvel_buffer_noisy,
        'linacc_buffer': linacc_buffer,
        'quat_buffer': quat_buffer,
        'angvel_buffer': angvel_buffer,
        'linvel_buffer': linvel_buffer,
    })
    
    return state.replace(
        obs=obs,
        state_vars=state_vars
    )

    


