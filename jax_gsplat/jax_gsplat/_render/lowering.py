from jax.interpreters import mlir
from jax.interpreters.mlir import ir
from jax._src.interpreters.mlir import custom_call

import _jax_gsplat
from jax_gsplat._render.abstract import RenderFwdTypes


def _render_fwd_rule(
    ctx: mlir.LoweringRuleContext,
    means3d: ir.Value,
    scales: ir.Value,
    quats: ir.Value,
    colors: ir.Value,
    opacities: ir.Value,
    viewmats: ir.Value,
    background: ir.Value,
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
    opaque = _jax_gsplat.make_descriptor(
        num_points=num_points,
        batch_size=batch_size,
        img_shape=img_shape,
        f=f,
        c=c,
        glob_scale=glob_scale,
        clip_thresh=clip_thresh,
        block_width=block_width,
    )

    t = RenderFwdTypes(num_points, batch_size, img_shape)

    op = custom_call(
        "render_fwd",
        operands=[
            means3d,
            scales,
            quats,
            colors,
            opacities,
            viewmats,
            background,
        ],
        operand_layouts=[
            t.in_means3d.layout(),
            t.in_scales.layout(),
            t.in_quats.layout(),
            t.in_colors.layout(),
            t.in_opacities.layout(),
            t.in_viewmats.layout(),
            t.in_background.layout(),
        ],
        result_types=[
            t.out_img.ir_tensor_type(),
        ],
        result_layouts=[
            t.out_img.layout(),
        ],
        backend_config=opaque,
        api_version=1,
    )

    return op.results
