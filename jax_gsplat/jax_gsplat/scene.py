"""PLY loader for 3D Gaussian Splatting scenes."""

from typing import NamedTuple
import numpy as np
import jax
import jax.numpy as jnp


class GSScene(NamedTuple):
    means3d: jax.Array    # (N, 3) float32
    scales: jax.Array     # (N, 3) float32 — exp() applied
    quats: jax.Array      # (N, 4) float32 — normalized, [w, x, y, z]
    colors: jax.Array     # (N, 3) float32 — SH DC → RGB, clipped [0, 1]
    opacities: jax.Array  # (N,) float32 — sigmoid() applied


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def load_ply(path: str) -> GSScene:
    """Load a 3DGS PLY file and return a GSScene with activations applied."""
    from plyfile import PlyData

    plydata = PlyData.read(path)
    vertex = plydata["vertex"]

    # Positions
    x = np.array(vertex["x"], dtype=np.float32)
    y = np.array(vertex["y"], dtype=np.float32)
    z = np.array(vertex["z"], dtype=np.float32)
    means3d = np.stack([x, y, z], axis=-1)

    # Scales (stored as log-scale in PLY)
    sx = np.array(vertex["scale_0"], dtype=np.float32)
    sy = np.array(vertex["scale_1"], dtype=np.float32)
    sz = np.array(vertex["scale_2"], dtype=np.float32)
    scales = np.exp(np.stack([sx, sy, sz], axis=-1))

    # Quaternions [w, x, y, z]
    qw = np.array(vertex["rot_0"], dtype=np.float32)
    qx = np.array(vertex["rot_1"], dtype=np.float32)
    qy = np.array(vertex["rot_2"], dtype=np.float32)
    qz = np.array(vertex["rot_3"], dtype=np.float32)
    quats = np.stack([qw, qx, qy, qz], axis=-1)
    norms = np.linalg.norm(quats, axis=-1, keepdims=True)
    quats = quats / np.maximum(norms, 1e-8)

    # Colors from SH DC coefficients → RGB
    SH_C0 = 0.28209479177387814
    f_dc_0 = np.array(vertex["f_dc_0"], dtype=np.float32)
    f_dc_1 = np.array(vertex["f_dc_1"], dtype=np.float32)
    f_dc_2 = np.array(vertex["f_dc_2"], dtype=np.float32)
    colors = np.stack([f_dc_0, f_dc_1, f_dc_2], axis=-1) * SH_C0 + 0.5
    colors = np.clip(colors, 0.0, 1.0)

    # Opacities (stored as logit in PLY)
    opacity_logit = np.array(vertex["opacity"], dtype=np.float32)
    opacities = _sigmoid(opacity_logit)

    return GSScene(
        means3d=jnp.array(means3d),
        scales=jnp.array(scales),
        quats=jnp.array(quats),
        colors=jnp.array(colors),
        opacities=jnp.array(opacities),
    )
