"""
Step 1b: Filter a raw 3DGS point cloud before normal estimation.

Why this is needed
------------------
Gaussian Splatting places Gaussians on *everything the camera sees*: sky, tree
canopy, distant hills, floaty background blobs.  When Poisson reconstruction
runs on this cloud it wraps a surface through all of these elevated points,
creating bumpy terrain over what should be a flat grassy area.

This script removes those offending points so Poisson only sees the ground
surface and low obstacles (stones, low vegetation, etc.).

What it does
------------
1.  Statistical outlier removal — removes isolated floating Gaussians.
2.  Height clip — discards any point above (ground_level + max_height).
    Ground level is estimated from the lowest percentile of the up-axis
    coordinate, which is robust against a few deep outliers.
3.  Optional: keeps only points within a horizontal radius of the scene
    centre — useful if the 3DGS scene captured a lot of distant background.

Usage
-----
    python mjo/step1b_filter_cloud.py \\
        --input-ply  mjo/output/pointcloud_raw.ply \\
        --output-ply mjo/output/pointcloud_filtered.ply \\
        --max-height 2.5

    # If your cloud is already fixed/reoriented with Y-up:
    python mjo/step1b_filter_cloud.py \\
        --input-ply  mjo/output/pointcloud_raw_fixed.ply \\
        --output-ply mjo/output/pointcloud_filtered.ply \\
        --up-axis y --max-height 2.0 --radius 30.0

    Then run step2 on the filtered cloud:
    python mjo/step2_estimate_normals.py \\
        --input-ply mjo/output/pointcloud_filtered.ply \\
        --output-ply mjo/output/pointcloud_normals.ply

"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


# ─────────────────────────────────────────────────────────────────────────────

_AXIS_IDX = {"x": 0, "y": 1, "z": 2}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Filter a 3DGS point cloud: outlier removal + height clip.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input-ply",  type=Path, required=True)
    p.add_argument("--output-ply", type=Path, required=True)

    p.add_argument(
        "--up-axis",
        choices=["x", "y", "z"],
        default="y",
        help="Which axis is 'up' in the cloud. Use 'y' if you ran fix_and_export first.",
    )
    p.add_argument(
        "--max-height",
        type=float,
        default=2.5,
        help=(
            "Maximum height (metres) above the estimated ground level to keep. "
            "Tune this to the tallest obstacle in your scene. "
            "For a stone circle with no trees inside: 1.5–2.0 m. "
            "If bushes at the perimeter are causing walls, reduce to 1.0–1.5 m."
        ),
    )
    p.add_argument(
        "--ground-percentile",
        type=float,
        default=2.0,
        help=(
            "Percentile of the up-axis used to estimate the ground level (0–100). "
            "Keep low (1–5) so a few below-ground outliers don't pull the estimate down."
        ),
    )
    p.add_argument(
        "--remove-outliers",
        action="store_true",
        default=True,
        help="Run statistical outlier removal before height clipping (recommended).",
    )
    p.add_argument(
        "--no-remove-outliers",
        dest="remove_outliers",
        action="store_false",
        help="Skip statistical outlier removal.",
    )
    p.add_argument(
        "--outlier-nb-neighbors",
        type=int,
        default=20,
        help="Neighbourhood size for statistical outlier removal.",
    )
    p.add_argument(
        "--outlier-std-ratio",
        type=float,
        default=2.0,
        help=(
            "Points whose mean-neighbour-distance is more than this many std-devs "
            "above the global mean are removed. Lower = more aggressive."
        ),
    )
    p.add_argument(
        "--radius",
        type=float,
        default=0.0,
        help=(
            "If > 0, discard points farther than this many metres from the "
            "horizontal centre of the cloud. Useful for removing distant background."
        ),
    )
    p.add_argument(
        "--voxel-size",
        type=float,
        default=0.0,
        help="If > 0, apply a final voxel downsample at this resolution (metres).",
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    print(f"Loading {args.input_ply} ...")
    pcd = o3d.io.read_point_cloud(str(args.input_ply))
    n0 = len(pcd.points)
    print(f"  {n0:,} points loaded")

    up_idx = _AXIS_IDX[args.up_axis]

    # ── 1. statistical outlier removal ───────────────────────────────────────
    if args.remove_outliers:
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=args.outlier_nb_neighbors,
            std_ratio=args.outlier_std_ratio,
        )
        n1 = len(pcd.points)
        print(f"  After outlier removal: {n1:,} points  (removed {n0 - n1:,})")
    else:
        n1 = n0

    pts = np.asarray(pcd.points)

    # ── 2. horizontal radius crop ─────────────────────────────────────────────
    if args.radius > 0:
        horiz_axes = [i for i in range(3) if i != up_idx]
        centre = pts[:, horiz_axes].mean(axis=0)
        dist   = np.linalg.norm(pts[:, horiz_axes] - centre, axis=1)
        mask   = dist <= args.radius
        pcd    = pcd.select_by_index(np.where(mask)[0])
        pts    = np.asarray(pcd.points)
        n2     = len(pts)
        print(
            f"  After radius crop (r={args.radius} m): {n2:,} points  "
            f"(removed {n1 - n2:,})"
        )
        n1 = n2

    # ── 3. height clip ────────────────────────────────────────────────────────
    heights      = pts[:, up_idx]
    ground_level = float(np.percentile(heights, args.ground_percentile))
    ceiling      = ground_level + args.max_height

    print(
        f"  Height range in cloud: [{heights.min():.3f}, {heights.max():.3f}]"
    )
    print(
        f"  Ground estimate (p{args.ground_percentile:.0f}): {ground_level:.3f}  "
        f"  Ceiling: {ceiling:.3f}  (max_height={args.max_height} m)"
    )

    keep = (heights >= ground_level - 0.3) & (heights <= ceiling)
    pcd  = pcd.select_by_index(np.where(keep)[0])
    n3   = len(pcd.points)
    print(
        f"  After height clip: {n3:,} points  (removed {n1 - n3:,} "
        f"= {100 * (n1 - n3) / max(n1, 1):.1f}%)"
    )

    # ── 4. optional final voxel downsample ────────────────────────────────────
    if args.voxel_size > 0:
        pcd = pcd.voxel_down_sample(args.voxel_size)
        n4  = len(pcd.points)
        print(f"  After voxel downsample ({args.voxel_size} m): {n4:,} points")

    # ── save ──────────────────────────────────────────────────────────────────
    args.output_ply.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(args.output_ply), pcd)
    print(f"\nSaved → {args.output_ply}  ({len(pcd.points):,} points)")
    print(
        f"\nTip: if the mesh still looks bumpy, reduce --max-height (currently {args.max_height}).\n"
        f"     If important geometry is missing, increase it.\n"
        f"     Use:  python mjo/tools/viewer_pointcloud.py --ply {args.output_ply}\n"
        f"     to inspect the result before running step2."
    )


if __name__ == "__main__":
    main()
