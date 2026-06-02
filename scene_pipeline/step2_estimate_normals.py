"""

Step 2: Estimate and orient normals for NKSR.

python mjo/step2_estimate_normals.py \
  --input-ply mjo/output/pointcloud_raw.ply \
  --output-ply mjo/output/pointcloud_normals.ply \
  --normal-radius 0.15 \
  --normal-max-nn 60 \
  --orient-k 30

"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Estimate and orient normals on a point cloud.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument(
    "--input-ply",
    type=Path,
    required=True,
    help="Input point cloud from step 1",
  )
  parser.add_argument(
    "--output-ply",
    type=Path,
    default=Path("pointcloud_normals.ply"),
    help="Output oriented point cloud with normals",
  )
  parser.add_argument(
    "--normal-radius",
    type=float,
    default=0.15,
    help="Radius (meters) for normal estimation neighborhood",
  )
  parser.add_argument(
    "--normal-max-nn",
    type=int,
    default=60,
    help="Max neighbors for normal estimation",
  )
  parser.add_argument(
    "--orient-k",
    type=int,
    default=30,
    help="Neighbors for tangent-plane normal orientation",
  )
  parser.add_argument(
    "--recenter",
    action="store_true",
    help="Translate cloud to zero-mean before writing",
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  pcd = o3d.io.read_point_cloud(str(args.input_ply))
  if len(pcd.points) == 0:
    raise ValueError(f"Empty point cloud: {args.input_ply}")

  print(f"Loaded: {args.input_ply}")
  print(f"Point count: {len(pcd.points)}")

  pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(
      radius=args.normal_radius,
      max_nn=args.normal_max_nn,
    )
  )
  pcd.orient_normals_consistent_tangent_plane(args.orient_k)
  pcd.normalize_normals()

  if args.recenter:
    xyz = np.asarray(pcd.points)
    mean_xyz = xyz.mean(axis=0, keepdims=True)
    xyz -= mean_xyz
    pcd.points = o3d.utility.Vector3dVector(xyz)
    print(f"Applied recentering offset: {mean_xyz.squeeze().tolist()}")

  args.output_ply.parent.mkdir(parents=True, exist_ok=True)
  o3d.io.write_point_cloud(str(args.output_ply), pcd)
  print(f"Saved: {args.output_ply}")


if __name__ == "__main__":
  main()

