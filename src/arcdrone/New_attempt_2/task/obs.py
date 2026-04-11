import jax.numpy as jp
from mujoco import mjx


def _get_obs_impl(self, state, action):
    """Render one frame from the mounted camera, shift frame stack + sensor buffers, return pixel obs."""
    data = state.data

    prev_stack_0 = state.info["frame_stack_0"]
    frame_stack_0 = prev_stack_0

    # Frame skip: only shift the stack every `frame_skip` steps.
    frame_skip = int(getattr(self.cfg, 'frame_skip', 1))
    frame_skip_counter = state.info["frame_skip_counter"]
    do_shift = (frame_skip_counter % frame_skip) == 0
    cpf = self._pixel_channels_per_frame

    if self._vision_enabled:
        # ── Pixels: single mounted camera ──
        render_data = mjx.refit_bvh(self.mjx_model, data, self._rc_pytree)
        out = mjx.render(self.mjx_model, render_data, self._rc_pytree)

        # Camera 0: x2_camera (mounted on drone)
        frame0 = self._format_frame(mjx.get_rgb(self._rc_pytree, 0, out[0]))
        shifted_0 = jp.concatenate([prev_stack_0[..., cpf:], frame0], axis=-1)
        frame_stack_0 = jp.where(do_shift, shifted_0, prev_stack_0.at[..., -cpf:].set(frame0))

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
    # Proprio: IMU-like sensors + velocity + altitude + actions, all flattened
    # altitude = jp.array([data.qpos[2]])  # barometer-like
    proprio = jp.concatenate([
        action_buffer.flatten(),
        linacc_buffer.flatten(),
        angvel_buffer.flatten(),
        quat_buffer.flatten(),
        # altitude,
        # linvel_buffer.flatten(),
    ])

    # Build pixel observation: raw frames + optional diff channels
    if self._diff_channels > 0:
        cpf = self._pixel_channels_per_frame
        diffs = []
        for i in range(int(self.cfg.buffer_size_pixels) - 1):
            curr = frame_stack_0[..., i * cpf:(i + 1) * cpf]
            prev = frame_stack_0[..., (i + 1) * cpf:(i + 2) * cpf]
            diffs.append(curr - prev)
        diff_stack = jp.concatenate(diffs, axis=-1)
        pixel_obs_0 = jp.concatenate([frame_stack_0, diff_stack], axis=-1)
    else:
        pixel_obs_0 = frame_stack_0

    obs = {
        "pixels/view_0": pixel_obs_0,     # (H, W, history + diffs) — mounted camera
        "proprio_obs": proprio,
        "value_obs": priviledged_state,           # critic obs
        "teacher_obs": priviledged_state,
    }

    info = {
        **state.info,
        "frame_stack_0":  frame_stack_0,
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
