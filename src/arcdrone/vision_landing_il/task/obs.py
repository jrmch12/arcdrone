import jax.numpy as jp
from mujoco import mjx


def _get_obs_impl(self, state, action):
    """Render two frames (front + side camera), shift frame stacks + sensor buffers, return pixel obs."""
    data = state.data

    # ── Pixels: three cameras ──
    render_data = mjx.refit_bvh(self.mjx_model, data, self._rc_pytree)
    out = mjx.render(self.mjx_model, render_data, self._rc_pytree)

    # Camera 0: outer_camera
    rgb0 = mjx.get_rgb(self._rc_pytree, 0, out[0])
    prev_stack_0 = state.info["frame_stack_0"]  # (H, W, history)
    # gray0 = jp.mean(rgb0, axis=-1, keepdims=True) - 0.5  # (H, W, 1)
    # frame_stack_0 = jp.concatenate([prev_stack_0[..., 1:], gray0], axis=-1)
    rgb0_norm = rgb0 - 0.5  # (H, W, 3)  # kept for easy revert
    frame_stack_0 = jp.concatenate([prev_stack_0[..., 3:], rgb0_norm], axis=-1)  # RGB revert

    # Camera 1: outer_camera_side
    rgb1 = mjx.get_rgb(self._rc_pytree, 1, out[0])
    prev_stack_1 = state.info["frame_stack_1"]  # (H, W, history)
    # gray1 = jp.mean(rgb1, axis=-1, keepdims=True) - 0.5  # (H, W, 1)
    # frame_stack_1 = jp.concatenate([prev_stack_1[..., 1:], gray1], axis=-1)
    rgb1_norm = rgb1 - 0.5  # kept for easy revert
    frame_stack_1 = jp.concatenate([prev_stack_1[..., 3:], rgb1_norm], axis=-1)  # RGB revert

    # Camera 2: outer_camera_up
    rgb2 = mjx.get_rgb(self._rc_pytree, 2, out[0])
    prev_stack_2 = state.info["frame_stack_2"]  # (H, W, history)
    # gray2 = jp.mean(rgb2, axis=-1, keepdims=True) - 0.5  # (H, W, 1)
    # frame_stack_2 = jp.concatenate([prev_stack_2[..., 1:], gray2], axis=-1)
    rgb2_norm = rgb2 - 0.5  # kept for easy revert
    frame_stack_2 = jp.concatenate([prev_stack_2[..., 3:], rgb2_norm], axis=-1)  # RGB revert

    # ── Priviledged ──────────────────────────
    action_buffer = state.info["action_buffer"]   # (history, nu)
    action_buffer = jp.roll(action_buffer, 1, axis=0)
    action_buffer = action_buffer.at[0].set(action)


    quat    = data.sensordata[0:4]
    angvel  = data.sensordata[4:7]
    linacc  = data.sensordata[7:10]
    linvel  = data.sensordata[10:13]
    pos     = data.qpos[0:3]
    target  = state.info.get("target", jp.zeros(3))

    linacc_buffer = jp.concatenate([linacc[jp.newaxis, :], state.info["linacc_buffer"][:-1, :]], axis=0)
    linvel_buffer = jp.concatenate([linvel[jp.newaxis, :], state.info["linvel_buffer"][:-1, :]], axis=0)
    quat_buffer   = jp.concatenate([quat[jp.newaxis, :],   state.info["quat_buffer"][:-1, :]],   axis=0)
    angvel_buffer = jp.concatenate([angvel[jp.newaxis, :], state.info["angvel_buffer"][:-1, :]], axis=0)
    pos_buffer    = jp.concatenate([pos[jp.newaxis, :],    state.info["pos_buffer"][:-1, :]],    axis=0)
    target_buffer = jp.concatenate([target[jp.newaxis, :], state.info["target_buffer"][:-1, :]], axis=0)


    priviledged_state = jp.concatenate([
        linacc_buffer.flatten(),
        linvel_buffer.flatten(),
        quat_buffer.flatten(),
        angvel_buffer.flatten(),
        target_buffer.flatten(),
        action_buffer.flatten(),
        pos_buffer.flatten(),
    ])

    # ── Build obs dict ───────────────────────────────────────────────────────
    # Flat structure: pixels/view_* keys are stripped by Brax's _remove_pixels
    # Proprio: IMU-like (linacc, angvel, quat) buffers, all flattened
    proprio = jp.concatenate([
        action_buffer.flatten(),
        linacc_buffer.flatten(),
        angvel_buffer.flatten(),
        quat_buffer.flatten(),
    ])
    obs = {
        "pixels/view_0": frame_stack_0,     # (H, W, history) — front camera
        "pixels/view_1": frame_stack_1,     # (H, W, history) — side camera
        "pixels/view_2": frame_stack_2,     # (H, W, history) — up camera
        "propio": priviledged_state,  # (history * (3+3+4),)
        "value_obs": priviledged_state,           # critic obs
        "teacher_obs": priviledged_state,  #  
    }

    info = {
        **state.info,
        "frame_stack_0":  frame_stack_0,
        "frame_stack_1":  frame_stack_1,
        "frame_stack_2":  frame_stack_2,
        "action_buffer": action_buffer,
        "linacc_buffer": linacc_buffer,
        "linvel_buffer": linvel_buffer,
        "quat_buffer":   quat_buffer,
        "angvel_buffer": angvel_buffer,
        "pos_buffer":    pos_buffer,
        "target_buffer": target_buffer,
    }
    return state.replace(obs=obs, info=info)
