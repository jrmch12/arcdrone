import jax.numpy as jp


def _rgba_to_grayscale(rgba):
    """Intensity-weighted RGBA → grayscale. Channels must be in the last dim."""
    r, g, b = rgba[..., 0], rgba[..., 1], rgba[..., 2]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b


def _get_obs_impl(self, state, action):
    """Render one frame, roll both image and action history buffers, return pixel obs."""
    _, rgb, _ = self.renderer.render(state.info["render_token"], state.data)

    # ── Image buffer: index 0 = newest frame ────────────────────────────────
    obs_history = state.info["obs_history"]
    obs_history = jp.roll(obs_history, 1, axis=0)
    obs_history = obs_history.at[0].set(
        _rgba_to_grayscale(rgb[0].astype(jp.float32)) / 255.0
    )

    # ── Action buffer: index 0 = most recent action ──────────────────────────
    action_buffer = state.info["action_buffer"]   # (history, nu)
    action_buffer = jp.roll(action_buffer, 1, axis=0)
    action_buffer = action_buffer.at[0].set(action)

    # ── Build obs dict ───────────────────────────────────────────────────────
    obs = {
        "pixels/view_0": obs_history.transpose(1, 2, 0),   # (H, W, history)
        "state": action_buffer.flatten(),                   # (history * nu,)
    }

    info = {**state.info, "obs_history": obs_history, "action_buffer": action_buffer}
    return state.replace(obs=obs, info=info)
