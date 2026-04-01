#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

try:
    import pymeshlab
except ImportError as exc:
    raise SystemExit(
        "pymeshlab is not installed. Install it with:\n"
        "  python -m pip install pymeshlab\n"
        "Then rerun this script."
    ) from exc


def _write_mesh(vertices: list[tuple[float, float, float]],
                faces: list[tuple[int, int, int]],
                out_path: Path) -> None:
    mesh = pymeshlab.Mesh(
        vertex_matrix=np.asarray(vertices, dtype=float),
        face_matrix=np.asarray(faces, dtype=int),
    )
    ms = pymeshlab.MeshSet()
    ms.add_mesh(mesh, out_path.stem)
    ms.save_current_mesh(str(out_path))


def _make_box(half_sizes: tuple[float, float, float]) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    hx, hy, hz = half_sizes
    v = [
        (-hx, -hy, -hz),  # 0
        ( hx, -hy, -hz),  # 1
        ( hx,  hy, -hz),  # 2
        (-hx,  hy, -hz),  # 3
        (-hx, -hy,  hz),  # 4
        ( hx, -hy,  hz),  # 5
        ( hx,  hy,  hz),  # 6
        (-hx,  hy,  hz),  # 7
    ]
    # Winding is set so normals point outward (right-hand rule).
    f = [
        (0, 2, 1), (0, 3, 2),  # bottom (-z)
        (4, 5, 6), (4, 6, 7),  # top (+z)
        (0, 1, 5), (0, 5, 4),  # -y
        (1, 2, 6), (1, 6, 5),  # +x
        (3, 6, 2), (3, 7, 6),  # +y
        (0, 4, 7), (0, 7, 3),  # -x
    ]
    return v, f


def _make_cylinder(radius: float, half_height: float, segments: int = 32) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    v: list[tuple[float, float, float]] = []
    f: list[tuple[int, int, int]] = []

    for i in range(segments):
        a = 2.0 * math.pi * i / segments
        x = radius * math.cos(a)
        y = radius * math.sin(a)
        v.append((x, y, -half_height))  # bottom ring
    for i in range(segments):
        a = 2.0 * math.pi * i / segments
        x = radius * math.cos(a)
        y = radius * math.sin(a)
        v.append((x, y, half_height))  # top ring

    bottom_center = len(v)
    v.append((0.0, 0.0, -half_height))
    top_center = len(v)
    v.append((0.0, 0.0, half_height))

    for i in range(segments):
        j = (i + 1) % segments
        b0 = i
        b1 = j
        t0 = i + segments
        t1 = j + segments
        f.append((b0, b1, t1))
        f.append((b0, t1, t0))

        f.append((top_center, t0, t1))
        f.append((bottom_center, b1, b0))

    return v, f


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate X2 collision meshes with pymeshlab.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("assets/skydio_x2/assets/collision"),
        help="Output directory for generated OBJ files.",
    )
    parser.add_argument("--segments", type=int, default=12, help="Cylinder radial segments.")
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    boxes = {
        "x2_collision_box_060_027_020": (0.06, 0.027, 0.02),
        "x2_collision_box_050_027_020": (0.05, 0.027, 0.02),
        "x2_collision_box_023_017_010": (0.023, 0.017, 0.01),
    }
    cylinders = {
        "x2_rotor_cyl_013_010": (0.13, 0.01),
    }

    for name, half_sizes in boxes.items():
        verts, faces = _make_box(half_sizes)
        _write_mesh(verts, faces, out_dir / f"{name}.obj")
        print(f"Wrote {out_dir / f'{name}.obj'}")

    for name, (radius, half_height) in cylinders.items():
        verts, faces = _make_cylinder(radius, half_height, segments=args.segments)
        _write_mesh(verts, faces, out_dir / f"{name}.obj")
        print(f"Wrote {out_dir / f'{name}.obj'}")


if __name__ == "__main__":
    main()
