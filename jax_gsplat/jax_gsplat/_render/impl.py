from jax.interpreters import mlir, xla, batching
from jax._src.core import Primitive

import functools
import jax

import _jax_gsplat
from jax_gsplat._render import lowering, abstract


# Register GPU XLA custom calls (legacy untyped ABI → api_version=0)
for name, value in _jax_gsplat.registrations().items():
    jax.ffi.register_ffi_target(name, value, platform="gpu", api_version=0)


_render_fwd_p = Primitive("jax_gsplat_render_fwd")
_render_fwd_p.multiple_results = True
_render_fwd_p.def_impl(functools.partial(xla.apply_primitive, _render_fwd_p))
_render_fwd_p.def_abstract_eval(abstract._render_fwd_abs)

mlir.register_lowering(
    prim=_render_fwd_p,
    rule=lowering._render_fwd_rule,
    platform="gpu",
)


def _render_fwd_batching(args, dims, **kwargs):
    """Batching rule for vmap: merge batch dim into the B dimension of viewmats."""
    (means3d, scales, quats, colors, opacities, viewmats, background) = args
    (d_means, d_scales, d_quats, d_colors, d_opac, d_view, d_bg) = dims

    # Scene arrays should NOT be batched (shared across envs)
    assert d_means is batching.not_mapped, "means3d must not be vmapped"
    assert d_scales is batching.not_mapped, "scales must not be vmapped"
    assert d_quats is batching.not_mapped, "quats must not be vmapped"
    assert d_colors is batching.not_mapped, "colors must not be vmapped"
    assert d_opac is batching.not_mapped, "opacities must not be vmapped"
    assert d_bg is batching.not_mapped, "background must not be vmapped"

    # viewmats IS batched: (vmap_B, 4, 4)
    assert d_view == 0, "viewmats must be batched on dim 0"

    import jax.numpy as jnp

    old_batch_size = kwargs["batch_size"]
    vmap_size = viewmats.shape[0]

    if viewmats.ndim == 3:
        new_batch_size = vmap_size
        new_viewmats = viewmats
    else:
        new_batch_size = vmap_size * old_batch_size
        new_viewmats = viewmats.reshape(new_batch_size, 4, 4)

    new_kwargs = dict(kwargs)
    new_kwargs["batch_size"] = new_batch_size

    (out_img,) = _render_fwd_p.bind(
        means3d, scales, quats, colors, opacities, new_viewmats, background,
        **new_kwargs
    )

    H, W = kwargs["img_shape"]
    out_img = out_img.reshape(vmap_size, old_batch_size if viewmats.ndim > 3 else 1, H, W, 3)
    if viewmats.ndim == 3:
        out_img = out_img.squeeze(1)

    return (out_img,), (0,)


batching.primitive_batchers[_render_fwd_p] = _render_fwd_batching
