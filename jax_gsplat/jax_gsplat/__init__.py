"""jax_gsplat: Batched 3D Gaussian Splatting renderer for JAX."""

import jax
import jax.numpy as jnp

from jax_gsplat.scene import GSScene, load_ply
from jax_gsplat._render.impl import _render_fwd_p


def render(
    scene: GSScene,
    viewmats: jax.Array,
    *,
    background: jax.Array | None = None,
    img_shape: tuple[int, int] = (480, 640),
    f: tuple[float, float] | None = None,
    c: tuple[float, float] | None = None,
    glob_scale: float = 1.0,
    clip_thresh: float = 0.01,
    block_size: int = 16,
) -> jax.Array:
    """
    Render a 3DGS scene from one or more camera viewpoints.

    Args:
        scene: GSScene containing Gaussian parameters (shared, not batched).
        viewmats: (B, 4, 4) or (4, 4) array of view matrices (row-major, OpenCV convention).
        background: (3,) background color, defaults to black.
        img_shape: (H, W) output image dimensions.
        f: (fx, fy) focal lengths. If None, computed assuming 90-deg vertical FOV.
        c: (cx, cy) principal point. If None, uses image center.
        glob_scale: Global Gaussian scale multiplier.
        clip_thresh: Near-plane clipping distance.
        block_size: Tile size for rasterization (default 16).

    Returns:
        (B, H, W, 3) or (H, W, 3) rendered images as float32 in [0, 1].
    """
    H, W = img_shape
    N = scene.means3d.shape[0]

    if background is None:
        background = jnp.zeros(3, dtype=jnp.float32)

    if f is None:
        fy = (H / 2.0) / jnp.tan(jnp.pi / 4.0)
        f = (float(fy), float(fy))

    if c is None:
        c = (W / 2.0, H / 2.0)

    single = viewmats.ndim == 2
    if single:
        viewmats = viewmats[None, :, :]

    B = viewmats.shape[0]

    (out_img,) = _render_fwd_p.bind(
        scene.means3d,
        scene.scales,
        scene.quats,
        scene.colors,
        scene.opacities,
        viewmats,
        background,
        num_points=N,
        batch_size=B,
        img_shape=img_shape,
        f=f,
        c=c,
        glob_scale=glob_scale,
        clip_thresh=clip_thresh,
        block_width=block_size,
    )

    if single:
        out_img = out_img[0]

    return out_img


def compute_viewmat(cam_xpos: jax.Array, cam_xmat: jax.Array) -> jax.Array:
    """Convert MuJoCo camera pose to OpenCV view matrix (4x4, row-major).

    MuJoCo cameras look along -Z; OpenCV/jaxsplat looks along +Z.
    """
    flip = jnp.diag(jnp.array([1.0, -1.0, -1.0]))
    R = flip @ cam_xmat.reshape(3, 3).T
    t = -R @ cam_xpos

    viewmat = jnp.eye(4, dtype=jnp.float32)
    viewmat = viewmat.at[:3, :3].set(R)
    viewmat = viewmat.at[:3, 3].set(t)
    return viewmat


__all__ = ["render", "compute_viewmat", "load_ply", "GSScene"]
