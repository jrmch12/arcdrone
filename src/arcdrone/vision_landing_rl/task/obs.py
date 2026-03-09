import jax.numpy as jp
from mujoco import mjx


def _get_obs_impl(self, state, action):
    """Render one frame, shift frame_stack + all sensor buffers, return pixel obs."""
    data = state.data

    # ── Pixels ─────────────────────────────────
    render_data = mjx.refit_bvh(self.mjx_model, data, self._rc_pytree)
    out = mjx.render(self.mjx_model, render_data, self._rc_pytree)
    rgb = mjx.get_rgb(self._rc_pytree, 0, out[0])
    gray = jp.mean(rgb, axis=-1, keepdims=True) - 0.5  # (H, W, 1)
    prev_stack = state.info["frame_stack"]  # (H, W, history)
    frame_stack = jp.concatenate([prev_stack[..., 1:], gray], axis=-1)

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


    value_state = jp.concatenate([
        linacc_buffer.flatten(),
        linvel_buffer.flatten(),
        quat_buffer.flatten(),
        angvel_buffer.flatten(),
        target_buffer.flatten(),
        action_buffer.flatten(),
        pos_buffer.flatten(),
    ])

    # ── Build obs dict ───────────────────────────────────────────────────────
    # If using Brax ppo/train.py we need this dict to be a
    # Flat structure: pixels/view_0 key is stripped by Brax's _remove_pixels
    obs = {
        "pixels/view_0": frame_stack,       # (H, W, history) — excluded from normalizer
        "propio": action_buffer.flatten(),  # (history * nu,)
        "value_obs": value_state,           # critic obs
    }

    info = {
        **state.info,
        "frame_stack":   frame_stack,
        "action_buffer": action_buffer,
        "linacc_buffer": linacc_buffer,
        "linvel_buffer": linvel_buffer,
        "quat_buffer":   quat_buffer,
        "angvel_buffer": angvel_buffer,
        "pos_buffer":    pos_buffer,
        "target_buffer": target_buffer,
    }
    return state.replace(obs=obs, info=info)
