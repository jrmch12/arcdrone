import jax.numpy as jp
from mujoco import mjx


def _get_obs_impl(self, state, action):
    """Render two frames (front + side camera), shift frame stacks + sensor buffers, return pixel obs."""
    data = state.data

    prev_stack_0 = state.info["frame_stack_0"]
    prev_stack_1 = state.info["frame_stack_1"]
    prev_stack_2 = state.info["frame_stack_2"]
    frame_stack_0 = prev_stack_0
    frame_stack_1 = prev_stack_1
    frame_stack_2 = prev_stack_2

    # Frame skip: only shift the stack every `frame_skip` steps.
    # This gives the CNN frames spaced frame_skip steps apart for better
    # velocity inference from pixel differences.
    frame_skip = int(getattr(self.cfg, 'frame_skip', 1))
    frame_skip_counter = state.info["frame_skip_counter"]
    do_shift = (frame_skip_counter % frame_skip) == 0
    cpf = self._pixel_channels_per_frame

    if self._vision_enabled:
        # ── Pixels: three cameras ──
        render_data = mjx.refit_bvh(self.mjx_model, data, self._rc_pytree)
        out = mjx.render(self.mjx_model, render_data, self._rc_pytree)

        # Camera 0: outer_camera
        frame0 = self._format_frame(mjx.get_rgb(self._rc_pytree, 0, out[0]))
        shifted_0 = jp.concatenate([prev_stack_0[..., cpf:], frame0], axis=-1)
        frame_stack_0 = jp.where(do_shift, shifted_0, prev_stack_0.at[..., -cpf:].set(frame0))

        # Camera 1: outer_camera_side
        frame1 = self._format_frame(mjx.get_rgb(self._rc_pytree, 1, out[0]))
        shifted_1 = jp.concatenate([prev_stack_1[..., cpf:], frame1], axis=-1)
        frame_stack_1 = jp.where(do_shift, shifted_1, prev_stack_1.at[..., -cpf:].set(frame1))

        # Camera 2: outer_camera_up
        frame2 = self._format_frame(mjx.get_rgb(self._rc_pytree, 2, out[0]))
        shifted_2 = jp.concatenate([prev_stack_2[..., cpf:], frame2], axis=-1)
        frame_stack_2 = jp.where(do_shift, shifted_2, prev_stack_2.at[..., -cpf:].set(frame2))

    frame_skip_counter = frame_skip_counter + 1

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
        # linvel_buffer.flatten(),  # TODO: do not forget to delete! this is just for debugging
        angvel_buffer.flatten(),
        quat_buffer.flatten(),
    ])
    obs = {
        "pixels/view_0": frame_stack_0,     # (H, W, history) — front camera
        "pixels/view_1": frame_stack_1,     # (H, W, history) — side camera
        "pixels/view_2": frame_stack_2,     # (H, W, history) — up camera
        "proprio_obs": proprio,  # (history * (3+3+4),)
        "value_obs": priviledged_state,           # critic obs
        "teacher_obs": priviledged_state,
    }

    info = {
        **state.info,
        "frame_stack_0":  frame_stack_0,
        "frame_stack_1":  frame_stack_1,
        "frame_stack_2":  frame_stack_2,
        "frame_skip_counter": frame_skip_counter,
        "action_buffer": action_buffer,
        "linacc_buffer": linacc_buffer,
        "linvel_buffer": linvel_buffer,
        "quat_buffer":   quat_buffer,
        "angvel_buffer": angvel_buffer,
        "pos_buffer":    pos_buffer,
        "target_buffer": target_buffer,
    }
    return state.replace(obs=obs, info=info)
