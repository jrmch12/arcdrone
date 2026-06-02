"""
fix_and_export.py
-----------------
Interactive tool to visually correct PLY geometry and export the fixed file.

Works for both point clouds (with or without normals) and triangle meshes.

Problems solved
---------------
• Scene is upside-down / wrongly oriented (3DGS pipeline often flips Y or Z).
• Normals point inward instead of outward (pre-Poisson point cloud).
• Mesh faces wound the wrong way (shading / colour appears on the inside).

How it works
------------
The orientation buttons rotate a *visual* scene frame (no data changes yet).
The "Flip normals" / "Flip face winding" buttons modify the data in memory.
Clicking **Export** bakes the current visual rotation into the actual vertex /
point coordinates (and normals), then writes a corrected PLY file.

Usage
-----
    # Point cloud with normals (output of step2)
    python mjo/tools/fix_and_export.py \\
        --input  mjo/output/pointcloud_normals.ply \\
        --output mjo/output/pointcloud_normals_fixed.ply

    # Poisson mesh (output of step3p1)
    python mjo/tools/fix_and_export.py \\
        --input  mjo/output/mesh_raw_poisson.ply \\
        --output mjo/output/mesh_fixed.ply

    Open http://localhost:8080 in any browser.
    Adjust orientation + normals / winding, then click Export.

Dependencies
------------
    pip install viser trimesh numpy scipy open3d plyfile
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import scipy.spatial.transform as st
import trimesh
import open3d as o3d
import viser
from plyfile import PlyData


# ─────────────────────────── file type detection ──────────────────────────────

def _detect_type(path: Path) -> str:
    """Return 'mesh', 'pcd_normals', or 'pcd'."""
    m = trimesh.load(str(path), process=False)
    if isinstance(m, trimesh.Trimesh) and len(m.faces) > 0:
        return "mesh"
    props = {p.name for p in PlyData.read(str(path))["vertex"].properties}
    if {"nx", "ny", "nz"} <= props:
        return "pcd_normals"
    return "pcd"


# ─────────────────────────── loaders ──────────────────────────────────────────

def _load_pcd(path: Path) -> dict:
    v     = PlyData.read(str(path))["vertex"]
    props = {p.name for p in v.properties}

    points = np.stack([np.asarray(v["x"], np.float32),
                       np.asarray(v["y"], np.float32),
                       np.asarray(v["z"], np.float32)], axis=-1)

    normals = None
    if {"nx", "ny", "nz"} <= props:
        normals = np.stack([np.asarray(v["nx"], np.float32),
                            np.asarray(v["ny"], np.float32),
                            np.asarray(v["nz"], np.float32)], axis=-1)

    if {"red", "green", "blue"} <= props:
        raw = np.stack([np.asarray(v["red"]), np.asarray(v["green"]), np.asarray(v["blue"])], -1)
    elif {"r", "g", "b"} <= props:
        raw = np.stack([np.asarray(v["r"]), np.asarray(v["g"]), np.asarray(v["b"])], -1)
    else:
        raw = np.full((len(points), 3), 180)

    colors = (np.clip(raw, 0, 255) if raw.max() > 1.0 else (np.clip(raw, 0.0, 1.0) * 255)).astype(np.uint8)
    return {"points": points, "normals": normals, "colors": colors}


def _load_mesh(path: Path) -> trimesh.Trimesh:
    m = trimesh.load(str(path), force="mesh", process=False)
    if not isinstance(m, trimesh.Trimesh):
        raise ValueError(f"Could not load as triangle mesh: {path}")
    return m


# ─────────────────────────── rotation helpers ─────────────────────────────────

def _current_R(scene_frame) -> st.Rotation:
    """Read the scene frame quaternion as a scipy Rotation."""
    wxyz = np.array(scene_frame.wxyz, np.float32)
    return st.Rotation.from_quat(wxyz[[1, 2, 3, 0]])   # scipy uses xyzw


def _set_R(scene_frame, R: st.Rotation) -> None:
    q = R.as_quat()                                      # xyzw
    scene_frame.wxyz = np.array([q[3], q[0], q[1], q[2]], np.float32)


def _rotate(scene_frame, axis: str, deg: float) -> None:
    _set_R(scene_frame, st.Rotation.from_euler(axis, np.deg2rad(deg)) * _current_R(scene_frame))


# ─────────────────────────── normal quill helpers ─────────────────────────────

def _quill_segs(points: np.ndarray, normals: np.ndarray, length: float, every: int) -> np.ndarray:
    pts = points[::every]
    nrm = normals[::every]
    return np.stack([pts, pts + nrm * length], axis=1).astype(np.float32)


def _quill_colors(normals: np.ndarray, every: int, M: int) -> np.ndarray:
    """Color normals by direction (xyz → rgb)."""
    return ((normals[::every][:M] * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)


# ─────────────────────────── shared orientation panel ─────────────────────────

def _add_orientation_panel(server: viser.ViserServer, scene_frame) -> None:
    """Add the standard rotation / flip buttons to the GUI sidebar."""
    _up_rots = {
        "+Y ↑": st.Rotation.identity(),
        "-Y ↓": st.Rotation.from_euler("z", np.pi),
        "+Z ↑": st.Rotation.from_euler("x", -np.pi / 2),
        "-Z ↓": st.Rotation.from_euler("x",  np.pi / 2),
    }

    with server.gui.add_folder("🌐 Scene orientation"):
        server.gui.add_markdown(
            "*Rotations are visual only — the geometry is unchanged until you click Export.*"
        )
        server.gui.add_markdown("**Quick flips** (cumulative)")
        btn_row1 = [server.gui.add_button("⬆ Flip X"),
                    server.gui.add_button("⬆ Flip Y"),
                    server.gui.add_button("⬆ Flip Z")]
        server.gui.add_markdown("**90° steps**")
        btn_row2 = [server.gui.add_button("↻ +90° X"),
                    server.gui.add_button("↻ +90° Y"),
                    server.gui.add_button("↻ +90° Z")]
        btn_row3 = [server.gui.add_button("↺ -90° X"),
                    server.gui.add_button("↺ -90° Y"),
                    server.gui.add_button("↺ -90° Z")]
        server.gui.add_markdown("**Snap to up-direction**")
        up_btns = {k: server.gui.add_button(k) for k in _up_rots}
        server.gui.add_markdown("**Fine-tune** (roll / pitch / yaw °)")
        roll_sl  = server.gui.add_slider("Roll",  min=-180, max=180, step=1, initial_value=0)
        pitch_sl = server.gui.add_slider("Pitch", min=-180, max=180, step=1, initial_value=0)
        yaw_sl   = server.gui.add_slider("Yaw",   min=-180, max=180, step=1, initial_value=0)
        apply_rpy_btn   = server.gui.add_button("✔ Apply RPY")
        reset_scene_btn = server.gui.add_button("↺ Reset to identity")

    @btn_row1[0].on_click
    def _fx(_): _rotate(scene_frame, "x", 180)
    @btn_row1[1].on_click
    def _fy(_): _rotate(scene_frame, "y", 180)
    @btn_row1[2].on_click
    def _fz(_): _rotate(scene_frame, "z", 180)

    @btn_row2[0].on_click
    def _p90x(_): _rotate(scene_frame, "x",  90)
    @btn_row2[1].on_click
    def _p90y(_): _rotate(scene_frame, "y",  90)
    @btn_row2[2].on_click
    def _p90z(_): _rotate(scene_frame, "z",  90)

    @btn_row3[0].on_click
    def _m90x(_): _rotate(scene_frame, "x", -90)
    @btn_row3[1].on_click
    def _m90y(_): _rotate(scene_frame, "y", -90)
    @btn_row3[2].on_click
    def _m90z(_): _rotate(scene_frame, "z", -90)

    def _make_up_cb(label):
        def _cb(_): _set_R(scene_frame, _up_rots[label])
        return _cb
    for label, btn in up_btns.items():
        btn.on_click(_make_up_cb(label))

    @apply_rpy_btn.on_click
    def _apply_rpy(_):
        _set_R(scene_frame, st.Rotation.from_euler(
            "xyz", np.deg2rad([roll_sl.value, pitch_sl.value, yaw_sl.value])
        ))

    @reset_scene_btn.on_click
    def _reset(_):
        scene_frame.wxyz = np.array([1, 0, 0, 0], np.float32)
        roll_sl.value = pitch_sl.value = yaw_sl.value = 0


# ─────────────────────────── point cloud viewer ───────────────────────────────

def run_pcd_viewer(cloud: dict, input_path: Path, output_path: Path, port: int) -> None:
    has_normals = cloud["normals"] is not None

    server = viser.ViserServer(port=port)
    server.scene.set_up_direction("+y")
    print(f"\n  ✓ Viser running → http://localhost:{port}")
    print(f"  {len(cloud['points']):,} points {'(with normals)' if has_normals else '(no normals)'}\n")

    scene_frame = server.scene.add_frame(
        "/scene",
        wxyz=np.array([1, 0, 0, 0], np.float32),
        position=np.zeros(3, np.float32),
        show_axes=False,
    )

    _add_orientation_panel(server, scene_frame)

    # ── points ────────────────────────────────────────────────────────────────
    with server.gui.add_folder("☁ Points"):
        size_sl = server.gui.add_slider("Point size", min=0.001, max=0.05,
                                         step=0.001, initial_value=0.005)

    pc_handle = server.scene.add_point_cloud(
        "/scene/points",
        points=cloud["points"],
        colors=cloud["colors"],
        point_size=0.005,
    )

    @size_sl.on_update
    def _sz(e): pc_handle.point_size = e.target.value

    # ── normals ───────────────────────────────────────────────────────────────
    if has_normals:
        init_every = 10
        init_len   = 0.04

        segs0  = _quill_segs(cloud["points"], cloud["normals"], init_len, init_every)
        M0     = len(segs0)
        cols0  = np.repeat(_quill_colors(cloud["normals"], init_every, M0), 2, 0).reshape(M0, 2, 3)
        normal_handle = server.scene.add_line_segments(
            "/scene/normals", points=segs0, colors=cols0, line_width=1.0
        )

        def _rebuild_quills():
            ev   = int(every_sl.value)
            ln   = float(length_sl.value)
            segs = _quill_segs(cloud["points"], cloud["normals"], ln, ev)
            M    = len(segs)
            cols = np.repeat(_quill_colors(cloud["normals"], ev, M), 2, 0).reshape(M, 2, 3)
            normal_handle.points = segs
            normal_handle.colors = cols

        with server.gui.add_folder("→ Normals"):
            show_nrm_cb  = server.gui.add_checkbox("Show normals", initial_value=True)
            length_sl    = server.gui.add_slider("Length",    min=0.005, max=0.3,  step=0.005, initial_value=init_len)
            every_sl     = server.gui.add_slider("Every Nth", min=1,     max=200,  step=1,     initial_value=init_every)
            flip_nrm_btn = server.gui.add_button(
                "↕ Flip all normals",
                hint="Negate all normal vectors in memory — preview updates immediately.",
            )

        @show_nrm_cb.on_update
        def _tn(e): normal_handle.visible = e.target.value

        @length_sl.on_update
        def _len(_): _rebuild_quills()

        @every_sl.on_update
        def _ev(_): _rebuild_quills()

        @flip_nrm_btn.on_click
        def _flip_nrm(_):
            cloud["normals"] *= -1
            _rebuild_quills()
            print("  Normals flipped (in-memory). Click Export to save.")

    # ── export ────────────────────────────────────────────────────────────────
    with server.gui.add_folder("💾 Export"):
        server.gui.add_markdown(
            f"Bakes the current **rotation** and any **normal flip** into the "
            f"geometry coordinates, then writes:  \n`{output_path}`"
        )
        export_btn = server.gui.add_button("✔ Export fixed PLY", color="green")

    @export_btn.on_click
    def _export(_):
        R_mat = _current_R(scene_frame).as_matrix()   # (3,3)
        pts   = (cloud["points"] @ R_mat.T).astype(np.float32)

        pcd_out = o3d.geometry.PointCloud()
        pcd_out.points = o3d.utility.Vector3dVector(pts)

        if has_normals:
            nrm_out = (cloud["normals"] @ R_mat.T).astype(np.float32)
            pcd_out.normals = o3d.utility.Vector3dVector(nrm_out)

        pcd_out.colors = o3d.utility.Vector3dVector(
            cloud["colors"].astype(np.float64) / 255.0
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(str(output_path), pcd_out)
        print(f"  ✓ Exported → {output_path}")

    print("  Viewer ready — navigate in browser, Ctrl-C to quit.\n")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nShutting down.")


# ─────────────────────────── mesh viewer ──────────────────────────────────────

def run_mesh_viewer(mesh: trimesh.Trimesh, input_path: Path, output_path: Path, port: int) -> None:
    server = viser.ViserServer(port=port)
    server.scene.set_up_direction("+y")
    print(f"\n  ✓ Viser running → http://localhost:{port}")
    print(f"  {len(mesh.vertices):,} vertices · {len(mesh.faces):,} faces\n")

    scene_frame = server.scene.add_frame(
        "/scene",
        wxyz=np.array([1, 0, 0, 0], np.float32),
        position=np.zeros(3, np.float32),
        show_axes=False,
    )

    _add_orientation_panel(server, scene_frame)

    # ── mesh + wireframe ──────────────────────────────────────────────────────
    mesh_handle = server.scene.add_mesh_trimesh("/scene/mesh", mesh=mesh)

    def _wire_segs(m: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
        edges     = m.edges_unique
        verts     = np.array(m.vertices, np.float32)
        segs      = np.stack([verts[edges[:, 0]], verts[edges[:, 1]]], axis=1)
        cols      = np.full((*segs.shape[:2], 3), 30, np.uint8)
        return segs, cols

    w_segs, w_cols = _wire_segs(mesh)
    wire_handle = server.scene.add_line_segments(
        "/scene/wireframe", points=w_segs, colors=w_cols, line_width=0.5
    )
    wire_handle.visible = False

    def _refresh_mesh() -> None:
        """Push updated mesh geometry to viser (same name path = in-place update)."""
        server.scene.add_mesh_trimesh("/scene/mesh", mesh=mesh)
        w_segs2, w_cols2 = _wire_segs(mesh)
        wire_handle.points = w_segs2
        wire_handle.colors = w_cols2

    with server.gui.add_folder("△ Mesh"):
        server.gui.add_markdown(
            f"*{len(mesh.vertices):,} vertices · {len(mesh.faces):,} faces · {input_path.name}*"
        )
        show_mesh_cb   = server.gui.add_checkbox("Show mesh",      initial_value=True)
        show_wire_cb   = server.gui.add_checkbox("Show wireframe",  initial_value=False)
        flip_faces_btn = server.gui.add_button(
            "↕ Flip face winding",
            hint=(
                "Reverses the winding order of all triangles, which flips the "
                "direction surface normals point. Fixes inside-out shading. "
                "Preview updates immediately."
            ),
        )

    @show_mesh_cb.on_update
    def _tm(e): mesh_handle.visible = e.target.value

    @show_wire_cb.on_update
    def _tw(e): wire_handle.visible = e.target.value

    @flip_faces_btn.on_click
    def _flip_faces(_):
        mesh.invert()
        _refresh_mesh()
        print("  Face winding flipped (in-memory). Click Export to save.")

    # ── export ────────────────────────────────────────────────────────────────
    with server.gui.add_folder("💾 Export"):
        server.gui.add_markdown(
            f"Bakes the current **rotation** and any **face-winding flip** into "
            f"the mesh geometry, then writes:  \n`{output_path}`"
        )
        export_btn = server.gui.add_button("✔ Export fixed PLY", color="green")

    @export_btn.on_click
    def _export(_):
        R_mat = _current_R(scene_frame).as_matrix()
        T     = np.eye(4)
        T[:3, :3] = R_mat

        out_mesh = mesh.copy()
        out_mesh.apply_transform(T)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        out_mesh.export(str(output_path))
        print(f"  ✓ Exported → {output_path}")

    print("  Viewer ready — navigate in browser, Ctrl-C to quit.\n")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nShutting down.")


# ─────────────────────────── CLI ──────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Interactively fix PLY orientation / normals and export the corrected file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--input",  type=Path, required=True,
                    help="Input PLY file (point cloud or triangle mesh).")
    ap.add_argument("--output", type=Path, default=None,
                    help="Output path. Defaults to <input_stem>_fixed.ply in the same folder.")
    ap.add_argument("--port",   type=int,  default=8080)
    args = ap.parse_args()

    if args.output is None:
        args.output = args.input.parent / (args.input.stem + "_fixed.ply")

    print(f"Loading {args.input} ...")
    kind = _detect_type(args.input)
    print(f"  Detected type: {kind}")

    if kind == "mesh":
        run_mesh_viewer(_load_mesh(args.input), args.input, args.output, args.port)
    else:
        run_pcd_viewer(_load_pcd(args.input), args.input, args.output, args.port)


if __name__ == "__main__":
    main()
