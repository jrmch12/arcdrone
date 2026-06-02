"""
Step 1: Extract a point cloud from a 3DGS PLY.

python mjo/step1_extractpointcloud.py \
  --input-ply "/home/jrmch12/Documents/code/260430_learning_gs/scenes/SuperSplat/Boscawen-Ûn/scene.ply" \
  --output-ply mjo/output/pointcloud_raw.ply \
  --opacity-logit-threshold 0.0 \
  --voxel-size 0.05


"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
from plyfile import PlyData


def sigmoid(x: np.ndarray) -> np.ndarray:
  return 1.0 / (1.0 + np.exp(-x))


def get_vertex_property_names(vertices) -> set[str]:
  return {p.name for p in vertices.properties}


def extract_colors(vertices, mask: np.ndarray, prop_names: set[str]) -> np.ndarray | None:
  # 3DGS SH DC layout
  if {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(prop_names):
    f_dc = np.stack(
      [
        np.array(vertices["f_dc_0"]),
        np.array(vertices["f_dc_1"]),
        np.array(vertices["f_dc_2"]),
      ],
      axis=1,
    )[mask]
    return np.clip(0.5 + 0.282095 * f_dc, 0.0, 1.0)

  # Generic RGB layout
  if {"red", "green", "blue"}.issubset(prop_names):
    rgb = np.stack(
      [
        np.array(vertices["red"]),
        np.array(vertices["green"]),
        np.array(vertices["blue"]),
      ],
      axis=1,
    )[mask].astype(np.float32)
    if rgb.max() > 1.0:
      rgb /= 255.0
    return np.clip(rgb, 0.0, 1.0)

  return None


def load_3dgs_ply(path: Path, opacity_logit_threshold: float) -> tuple[np.ndarray, np.ndarray | None]:
  print(f"Loading 3DGS .ply: {path}")
  plydata = PlyData.read(str(path))
  vertices = plydata["vertex"]
  prop_names = get_vertex_property_names(vertices)
  print(f"Detected vertex properties: {sorted(prop_names)}")

  xyz = np.stack(
    [
      np.array(vertices["x"]),
      np.array(vertices["y"]),
      np.array(vertices["z"]),
    ],
    axis=1,
  )

  # Some exports may not include opacity. In that case keep all points.
  if "opacity" in prop_names:
    opacity_logit = np.array(vertices["opacity"])
    opacity = sigmoid(opacity_logit)
    opacity_prob_threshold = float(sigmoid(np.array([opacity_logit_threshold]))[0])
    mask = opacity > opacity_prob_threshold
    print(f"Total Gaussians: {len(xyz)}")
    print(f"After opacity filter (> {opacity_prob_threshold:.3f}): {int(mask.sum())}")
  else:
    mask = np.ones(len(xyz), dtype=bool)
    print("No opacity field found. Keeping all points.")

  xyz = xyz[mask]
  colors = extract_colors(vertices, mask, prop_names)
  if colors is None:
    print("No color fields found (or unsupported format).")
  else:
    print("Color extraction: enabled")
  return xyz, colors


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Extract and lightly denoise point cloud from 3DGS PLY.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument("--input-ply", type=Path, required=True, help="Input 3DGS .ply file")
  parser.add_argument(
    "--output-ply",
    type=Path,
    default=Path("pointcloud_raw.ply"),
    help="Output point cloud path",
  )
  parser.add_argument(
    "--opacity-logit-threshold",
    type=float,
    default=0.0,
    help="Opacity threshold in logit space (3DGS stores opacity as logits)",
  )
  parser.add_argument(
    "--outlier-nb-neighbors",
    type=int,
    default=20,
    help="Neighbors for statistical outlier removal (set <=0 to disable)",
  )
  parser.add_argument(
    "--outlier-std-ratio",
    type=float,
    default=2.0,
    help="Std ratio for statistical outlier removal",
  )
  parser.add_argument(
    "--voxel-size",
    type=float,
    default=0.05,
    help="Voxel size in meters (set <=0 to disable)",
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  xyz, colors = load_3dgs_ply(args.input_ply, args.opacity_logit_threshold)

  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
  if colors is not None:
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

  print(f"Point count before filtering: {len(pcd.points)}")

  if args.outlier_nb_neighbors > 0:
    pcd, _ = pcd.remove_statistical_outlier(
      nb_neighbors=args.outlier_nb_neighbors,
      std_ratio=args.outlier_std_ratio,
    )
    print(f"After outlier removal: {len(pcd.points)}")

  if args.voxel_size > 0:
    pcd = pcd.voxel_down_sample(voxel_size=args.voxel_size)
    print(f"After voxel downsample ({args.voxel_size} m): {len(pcd.points)}")

  args.output_ply.parent.mkdir(parents=True, exist_ok=True)
  o3d.io.write_point_cloud(str(args.output_ply), pcd)
  print(f"Saved: {args.output_ply}")


if __name__ == "__main__":
  main()
