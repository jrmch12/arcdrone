"""Step 3: Reconstruct mesh from oriented points using NKSR."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import nksr
from nksr.configs import get_hparams
from huggingface_hub import hf_hub_download


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Run NKSR on point cloud + normals and write mesh.",
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
    default=Path("mesh_raw_nksr.ply"),
    help="Output mesh path",
  )
  parser.add_argument(
    "--device",
    type=str,
    default="cuda:0",
    help="Torch device. NKSR is typically used on CUDA.",
  )
  parser.add_argument(
    "--detail-level",
    type=float,
    default=1.0,
    help="NKSR detail level",
  )
  parser.add_argument(
    "--mise-iter",
    type=int,
    default=2,
    help="Dual-mesh extraction iterations",
  )
  parser.add_argument(
    "--solver-tol",
    type=float,
    default=1e-5,
    help="Linear solver tolerance",
  )
  parser.add_argument(
    "--checkpoint",
    type=Path,
    default=None,
    help="Optional local NKSR checkpoint (.pth).",
  )
  parser.add_argument(
    "--hf-repo-id",
    type=str,
    default="escontra/ks",
    help="HF repo for checkpoint download when --checkpoint is not provided.",
  )
  parser.add_argument(
    "--hf-filename",
    type=str,
    default="ks.pth",
    help="HF checkpoint filename",
  )
  return parser.parse_args()


def build_reconstructor(device: torch.device, args: argparse.Namespace):
  nksr_config = get_hparams("ks")
  if args.checkpoint is not None:
    ckpt = str(args.checkpoint.expanduser().resolve())
  else:
    ckpt = hf_hub_download(repo_id=args.hf_repo_id, filename=args.hf_filename)
  nksr_config["url"] = ckpt
  print(f"Using NKSR checkpoint: {ckpt}")
  reconstructor = nksr.Reconstructor(device, nksr_config)
  reconstructor.chunk_tmp_device = torch.device("cpu")
  return reconstructor


def open3d_mesh_from_nksr(mesh) -> o3d.geometry.TriangleMesh:
  faces_np = mesh.f.cpu().numpy()
  vertices_np = mesh.v.cpu().numpy()

  o3d_mesh = o3d.geometry.TriangleMesh()
  o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices_np)
  o3d_mesh.triangles = o3d.utility.Vector3iVector(faces_np)

  if hasattr(mesh, "c"):
    colors_np = mesh.c.cpu().numpy()
    if colors_np.ndim == 2 and colors_np.shape[1] == 3:
      o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(colors_np, 0.0, 1.0))

  return o3d_mesh


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


def main() -> None:
  args = parse_args()
  device = torch.device(args.device)
  if device.type == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("CUDA requested but not available. Use --device cpu if needed.")

  pcd = o3d.io.read_point_cloud(str(args.input_ply))
  if len(pcd.points) == 0:
    raise ValueError(f"Empty point cloud: {args.input_ply}")
  if not pcd.has_normals():
    raise ValueError("Input point cloud has no normals. Run step2 first.")

  xyz = np.asarray(pcd.points)
  normals = np.asarray(pcd.normals)
  print(f"Loaded points: {xyz.shape[0]}")

  input_xyz = torch.from_numpy(xyz).float().to(device)
  input_normals = torch.from_numpy(normals).float().to(device)
  reconstructor = build_reconstructor(device, args)

  field = reconstructor.reconstruct(
    input_xyz,
    input_normals,
    detail_level=args.detail_level,
    approx_kernel_grad=False,
    solver_tol=args.solver_tol,
    fused_mode=True,
  )

  if pcd.has_colors():
    colors = np.asarray(pcd.colors)
    input_color = torch.from_numpy(colors).float().to(device)
    field.set_texture_field(nksr.fields.PCNNField(input_xyz, input_color))

  mesh = field.extract_dual_mesh(mise_iter=args.mise_iter)
  o3d_mesh = open3d_mesh_from_nksr(mesh)

  o3d_mesh.remove_duplicated_vertices()
  o3d_mesh.remove_duplicated_triangles()
  o3d_mesh.remove_degenerate_triangles()
  o3d_mesh.remove_unreferenced_vertices()
  o3d_mesh = keep_largest_component(o3d_mesh)
  o3d_mesh.compute_vertex_normals()

  args.output_mesh.parent.mkdir(parents=True, exist_ok=True)
  o3d.io.write_triangle_mesh(str(args.output_mesh), o3d_mesh)
  print(
    f"Saved mesh: {args.output_mesh} "
    f"(verts={len(o3d_mesh.vertices)}, tris={len(o3d_mesh.triangles)})"
  )


if __name__ == "__main__":
  main()

