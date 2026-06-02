"""
ply_info.py — quick stats for a Gaussian Splat .ply
Usage: python ply_info.py splat.ply
"""

import sys
from pathlib import Path
import numpy as np
from plyfile import PlyData

def main():
    path = Path(sys.argv[1])
    v = PlyData.read(str(path))["vertex"]
    props = [p.name for p in v.properties]
    N = len(v)

    x = np.asarray(v["x"], np.float32)
    y = np.asarray(v["y"], np.float32)
    z = np.asarray(v["z"], np.float32)

    # Opacity
    if "opacity" in props:
        raw_op = np.asarray(v["opacity"], np.float32)
        op = 1.0 / (1.0 + np.exp(-raw_op))
        op_info = f"  mean={op.mean():.3f}  min={op.min():.3f}  max={op.max():.3f}"
        visible = (op > 0.1).sum()
    else:
        op_info = "  (no opacity field)"
        visible = N

    # Scales
    if "scale_0" in props:
        scales = np.exp(np.stack([
            np.asarray(v["scale_0"], np.float32),
            np.asarray(v["scale_1"], np.float32),
            np.asarray(v["scale_2"], np.float32),
        ], -1))
        sc_mean = scales.mean(axis=0)
        sc_info = f"  mean per axis: x={sc_mean[0]:.4f}  y={sc_mean[1]:.4f}  z={sc_mean[2]:.4f}"
        sc_info += f"\n  global min={scales.min():.4f}  max={scales.max():.4f}"
    else:
        sc_info = "  (no scale fields)"

    # Colour
    SH_C0 = 0.28209479177387814
    if "f_dc_0" in props:
        rgb = np.clip(0.5 + SH_C0 * np.stack([
            np.asarray(v["f_dc_0"], np.float32),
            np.asarray(v["f_dc_1"], np.float32),
            np.asarray(v["f_dc_2"], np.float32),
        ], -1), 0, 1)
        col_info = (f"  mean RGB: ({rgb[:,0].mean():.3f}, "
                    f"{rgb[:,1].mean():.3f}, {rgb[:,2].mean():.3f})")
        sh_bands = sum(1 for p in props if p.startswith("f_rest_")) // 3
        col_info += f"\n  SH bands (beyond DC): {sh_bands}"
    else:
        col_info = "  (no SH colour fields)"

    print(f"""
╔══ {path.name} {'═'*(max(0,50-len(path.name)))}
║  File size      : {path.stat().st_size / 1e6:.2f} MB
║  Gaussians      : {N:,}
║  Visible (op>0.1): {visible:,}  ({100*visible/N:.1f}%)
║
║  Bounding box
║    X : {x.min():.4f} → {x.max():.4f}  (span {x.max()-x.min():.4f} m)
║    Y : {y.min():.4f} → {y.max():.4f}  (span {y.max()-y.min():.4f} m)
║    Z : {z.min():.4f} → {z.max():.4f}  (span {z.max()-z.min():.4f} m)
║  Centre         : ({x.mean():.4f}, {y.mean():.4f}, {z.mean():.4f})
║
║  Opacity
║{op_info}
║
║  Scale
║{sc_info}
║
║  Colour (DC SH → RGB)
║{col_info}
║
║  All properties ({len(props)}):
║    {', '.join(props)}
╚{'═'*54}
""")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ply_info.py splat.ply")
        sys.exit(1)
    main()