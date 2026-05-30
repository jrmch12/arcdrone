import jax
import jax.numpy as jnp

from jax_gsplat._types import Type


class RenderFwdTypes:
    def __init__(
        self,
        num_points: int,
        batch_size: int,
        img_shape: tuple[int, int],
    ):
        H, W = img_shape
        self.in_means3d = Type((num_points, 3), jnp.float32)
        self.in_scales = Type((num_points, 3), jnp.float32)
        self.in_quats = Type((num_points, 4), jnp.float32)
        self.in_colors = Type((num_points, 3), jnp.float32)
        self.in_opacities = Type((num_points,), jnp.float32)
        self.in_viewmats = Type((batch_size, 4, 4), jnp.float32)
        self.in_background = Type((3,), jnp.float32)

        self.out_img = Type((batch_size, H, W, 3), jnp.float32)


def _render_fwd_abs(
    means3d: jax.Array,
    scales: jax.Array,
    quats: jax.Array,
    colors: jax.Array,
    opacities: jax.Array,
    viewmats: jax.Array,
    background: jax.Array,
    *,
    num_points: int,
    batch_size: int,
    img_shape: tuple[int, int],
    f: tuple[float, float],
    c: tuple[float, float],
    glob_scale: float,
    clip_thresh: float,
    block_width: int,
):
    t = RenderFwdTypes(num_points, batch_size, img_shape)

    t.in_means3d.assert_(means3d)
    t.in_scales.assert_(scales)
    t.in_quats.assert_(quats)
    t.in_colors.assert_(colors)
    t.in_opacities.assert_(opacities)
    t.in_viewmats.assert_(viewmats)
    t.in_background.assert_(background)

    return (t.out_img.shaped_array(),)
