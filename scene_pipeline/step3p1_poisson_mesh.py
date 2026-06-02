"""Step 3.1: Reconstruct mesh from oriented points using Open3D Poisson surface reconstruction.

Alternative to NKSR (step 3) — no GPU or neural network required.

python mjo/step3p1_poisson_mesh.py \
  --input-ply mjo/output/pointcloud_normals.ply \
  --output-mesh mjo/output/mesh_raw_poisson.ply \
  --depth 10 \
  --density-threshold 0.05

"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Open3D Poisson surface reconstruction on an oriented point cloud.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-ply",
        type=Path,
        required=True,
        help="Point cloud with normals (from step 2)",
    )
    parser.add_argument(
        "--output-mesh",
        type=Path,
        default=Path("mesh_raw_poisson.ply"),
        help="Output mesh path (.ply or .obj)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=10,
        help=(
            "Octree depth for Poisson reconstruction. "
            "Higher = more detail but slower and more memory. "
            "Typical range: 8–12."
        ),
    )
    parser.add_argument(
        "--width",
        type=float,
        default=0,
        help=(
            "Target width of the finest level octree cells. "
            "If non-zero, overrides --depth. 0 means use --depth."
        ),
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.1,
        help="Ratio of the diameter of the cube used for reconstruction to the diameter of the samples.",
    )
    parser.add_argument(
        "--linear-fit",
        action="store_true",
        default=False,
        help="Use linear interpolation for iso-vertex positions instead of quadratic.",
    )
    parser.add_argument(
        "--density-threshold",
        type=float,
        default=0.05,
        help=(
            "Remove low-density vertices below this quantile (0–1). "
            "Higher values trim more of the 'floaty' exterior surface. "
            "Set to 0 to keep all vertices."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path: Path = args.input_ply.expanduser().resolve()
    output_path: Path = args.output_mesh.expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ load
    print(f"Loading point cloud: {input_path}")
    pcd = o3d.io.read_point_cloud(str(input_path))

    n_points = np.asarray(pcd.points).shape[0]
    print(f"  {n_points:,} points loaded")

    if not pcd.has_normals():
        raise ValueError(
            "Point cloud has no normals. "
            "Run step2_estimate_normals.py first."
        )

    # ----------------------------------------------------------- reconstruct
    print(
        f"Running Poisson reconstruction (depth={args.depth}, "
        f"width={args.width}, scale={args.scale}) ..."
    )
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=args.depth,
        width=args.width,
        scale=args.scale,
        linear_fit=args.linear_fit,
    )

    n_verts_raw = np.asarray(mesh.vertices).shape[0]
    n_tris_raw = np.asarray(mesh.triangles).shape[0]
    print(f"  Raw mesh: {n_verts_raw:,} vertices, {n_tris_raw:,} triangles")

    # ------------------------------------------------- density-based pruning
    if args.density_threshold > 0.0:
        densities_np = np.asarray(densities)
        cutoff = np.quantile(densities_np, args.density_threshold)
        print(
            f"Removing low-density vertices "
            f"(quantile threshold={args.density_threshold:.3f}, "
            f"density cutoff={cutoff:.4f}) ..."
        )
        mask = densities_np < cutoff
        mesh.remove_vertices_by_mask(mask)

        n_verts_pruned = np.asarray(mesh.vertices).shape[0]
        n_tris_pruned = np.asarray(mesh.triangles).shape[0]
        print(f"  After pruning: {n_verts_pruned:,} vertices, {n_tris_pruned:,} triangles")

    # -------------------------------------------------------------- save
    mesh.compute_vertex_normals()
    print(f"Writing mesh: {output_path}")
    o3d.io.write_triangle_mesh(str(output_path), mesh)
    print("Done.")


if __name__ == "__main__":
    main()
