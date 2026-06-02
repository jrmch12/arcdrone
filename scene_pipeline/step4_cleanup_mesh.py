"""Step 4: Final mesh cleanup and optional hole-filling/decimation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Clean and simplify mesh for collision use.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument("--input-mesh", type=Path, required=True, help="Input mesh")
  parser.add_argument(
    "--output-mesh",
    type=Path,
    default=Path("mesh_collision_final.ply"),
    help="Output cleaned mesh",
  )
  parser.add_argument(
    "--crop-to-cloud",
    type=Path,
    default=None,
    metavar="PLY",
    help=(
      "Point cloud PLY used to define the crop bounding box. "
      "All mesh vertices outside the cloud's bbox (+margin) are removed. "
      "This is the most reliable fix for Poisson thin-shell boundary artifacts."
    ),
  )
  parser.add_argument(
    "--crop-margin",
    type=float,
    default=0.5,
    metavar="M",
    help="Extra margin (meters) added to each side of the bounding box when cropping.",
  )
  parser.add_argument(
    "--fill-holes",
    action="store_true",
    help="Apply Open3D tensor fill_holes() before decimation",
  )
  parser.add_argument(
    "--target-triangles",
    type=int,
    default=0,
    help="If >0, decimate to this triangle count",
  )
  parser.add_argument(
    "--keep-largest",
    action="store_true",
    help="Keep only largest connected component",
  )
  return parser.parse_args()


def crop_to_cloud_bbox(
  mesh: o3d.geometry.TriangleMesh,
  cloud_path: Path,
  margin: float,
) -> o3d.geometry.TriangleMesh:
  """Remove mesh vertices that fall outside the point cloud's axis-aligned bounding box.

  Poisson reconstruction creates thin phantom walls at the scan boundary because
  the solver must close its implicit surface somewhere beyond the data.  Any vertex
  outside the cloud's spatial extent + margin is guaranteed to be such an artifact.
  """
  pcd = o3d.io.read_point_cloud(str(cloud_path))
  pts = np.asarray(pcd.points)
  bbox_min = pts.min(axis=0) - margin
  bbox_max = pts.max(axis=0) + margin
  print(
    f"Crop bbox: [{bbox_min[0]:.2f}, {bbox_min[1]:.2f}, {bbox_min[2]:.2f}]"
    f"  →  [{bbox_max[0]:.2f}, {bbox_max[1]:.2f}, {bbox_max[2]:.2f}]"
    f"  (margin={margin} m)"
  )

  verts = np.asarray(mesh.vertices)
  outside = np.any((verts < bbox_min) | (verts > bbox_max), axis=1)
  n_removed = int(outside.sum())
  mesh.remove_vertices_by_mask(outside.tolist())
  mesh.remove_unreferenced_vertices()
  print(
    f"Removed {n_removed:,} out-of-bbox vertices → "
    f"verts={len(mesh.vertices):,}, tris={len(mesh.triangles):,}"
  )
  return mesh


def keep_largest_component(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
  triangle_clusters, _, cluster_areas = mesh.cluster_connected_triangles()
  triangle_clusters = np.asarray(triangle_clusters)
  cluster_areas = np.asarray(cluster_areas)
  if len(cluster_areas) == 0:
    return mesh
  largest_id = int(np.argmax(cluster_areas))
  mask = triangle_clusters != largest_id
  mesh.remove_triangles_by_mask(mask.tolist())
  mesh.remove_unreferenced_vertices()
  return mesh


def basic_cleanup(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
  mesh.remove_degenerate_triangles()
  mesh.remove_duplicated_triangles()
  mesh.remove_duplicated_vertices()
  mesh.remove_non_manifold_edges()
  mesh.remove_unreferenced_vertices()
  return mesh


def main() -> None:
  args = parse_args()
  mesh = o3d.io.read_triangle_mesh(str(args.input_mesh))
  if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
    raise ValueError(f"Invalid/empty mesh: {args.input_mesh}")

  n_tris_in = len(mesh.triangles)
  print(
    f"Loaded mesh: verts={len(mesh.vertices)}, tris={n_tris_in} "
    f"from {args.input_mesh}"
  )
  mesh = basic_cleanup(mesh)

  if args.crop_to_cloud is not None:
    mesh = crop_to_cloud_bbox(mesh, args.crop_to_cloud, args.crop_margin)
    mesh = basic_cleanup(mesh)

  # keep_largest MUST come before fill_holes: fill_holes() (tensor API) restructures
  # mesh topology in a way that breaks cluster_connected_triangles(), so filtering
  # after fill_holes can silently discard the entire mesh.
  if args.keep_largest:
    mesh = keep_largest_component(mesh)
    print(
      f"After keeping largest component: verts={len(mesh.vertices)}, "
      f"tris={len(mesh.triangles)}"
    )

  if args.fill_holes:
    tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    tmesh = tmesh.fill_holes()
    mesh = tmesh.to_legacy()
    mesh = basic_cleanup(mesh)
    print(f"After fill_holes(): verts={len(mesh.vertices)}, tris={len(mesh.triangles)}")

  # Sanity check: warn if we lost more than 80% of the mesh
  n_tris_now = len(mesh.triangles)
  if n_tris_now < 0.2 * n_tris_in:
    print(
      f"\n  ⚠  WARNING: only {n_tris_now} / {n_tris_in} triangles remain "
      f"({100 * n_tris_now / n_tris_in:.1f}%). "
      f"Try re-running without --fill-holes.\n"
    )

  if args.target_triangles > 0 and len(mesh.triangles) > args.target_triangles:
    mesh = mesh.simplify_quadric_decimation(args.target_triangles)
    mesh = basic_cleanup(mesh)
    print(f"After decimation: verts={len(mesh.vertices)}, tris={len(mesh.triangles)}")

  mesh.compute_vertex_normals()
  args.output_mesh.parent.mkdir(parents=True, exist_ok=True)
  o3d.io.write_triangle_mesh(str(args.output_mesh), mesh)
  print(
    f"Saved mesh: {args.output_mesh} "
    f"(verts={len(mesh.vertices)}, tris={len(mesh.triangles)})"
  )


if __name__ == "__main__":
  main()

