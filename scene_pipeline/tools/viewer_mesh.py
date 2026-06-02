"""
viewer_mesh.py
--------------
Interactive triangle mesh viewer using Viser.

Loads a .ply triangle mesh (e.g. from open3d Poisson reconstruction) and
renders it with scene orientation controls, a wireframe toggle, and a
density-based vertex-trim slider (useful for cleaning Poisson meshes).

Usage
-----
    python scene_pipeline/tools/viewer_mesh.py --ply mesh_raw_poisson.ply
    python scene_pipeline/tools/viewer_mesh.py --ply mesh_raw_poisson.ply --port 8081

    Then open  http://localhost:8080  in any browser.

Install
-------
    pip install viser trimesh numpy scipy
"""

import argparse
import time
from pathlib import Path

import numpy as np
import scipy.spatial.transform as st
import trimesh
import viser


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(str(path), force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"{path.name} did not load as a triangle mesh (got {type(mesh).__name__})")
    print(f"  {len(mesh.vertices):,} vertices  ·  {len(mesh.faces):,} faces")
    return mesh


def run_viewer(mesh: trimesh.Trimesh, path: Path, port: int = 8080):
    server = viser.ViserServer(port=port)
    server.scene.set_up_direction("+y")
    print(f"\n  ✓ Viser running → http://localhost:{port}\n")

    scene_frame = server.scene.add_frame(
        "/scene",
        wxyz=np.array([1, 0, 0, 0], np.float32),
        position=np.zeros(3, np.float32),
        show_axes=False,
    )

    # ── orientation controls ──────────────────────────────────────────────────
    with server.gui.add_folder("🌐 Scene orientation"):
        server.gui.add_markdown("**Quick fixes** (cumulative)")
        btn_row1 = [
            server.gui.add_button("⬆ Flip X", hint="Rotate 180° around X"),
            server.gui.add_button("⬆ Flip Y", hint="Rotate 180° around Y"),
            server.gui.add_button("⬆ Flip Z", hint="Rotate 180° around Z"),
        ]
        btn_row2 = [server.gui.add_button("↻ +90° X"), server.gui.add_button("↻ +90° Y"), server.gui.add_button("↻ +90° Z")]
        btn_row3 = [server.gui.add_button("↺ -90° X"), server.gui.add_button("↺ -90° Y"), server.gui.add_button("↺ -90° Z")]
        server.gui.add_markdown("**Up direction**")
        up_buttons = {
            "+Y ↑": server.gui.add_button("+Y ↑"),
            "-Y ↓": server.gui.add_button("-Y ↓"),
            "+Z ↑": server.gui.add_button("+Z ↑"),
            "-Z ↓": server.gui.add_button("-Z ↓"),
        }
        server.gui.add_markdown("**Free rotation** (roll / pitch / yaw °)")
        roll_sl  = server.gui.add_slider("Roll",  min=-180, max=180, step=1, initial_value=0)
        pitch_sl = server.gui.add_slider("Pitch", min=-180, max=180, step=1, initial_value=0)
        yaw_sl   = server.gui.add_slider("Yaw",   min=-180, max=180, step=1, initial_value=0)
        apply_rpy_btn   = server.gui.add_button("✔ Apply RPY (deg)")
        reset_scene_btn = server.gui.add_button("↺ Reset orientation")

    def _rotate_scene(axis: str, deg: float):
        cur_wxyz = np.array(scene_frame.wxyz, np.float32)
        cur_R    = st.Rotation.from_quat(cur_wxyz[[1, 2, 3, 0]])
        new_R    = st.Rotation.from_euler(axis, np.deg2rad(deg)) * cur_R
        q        = new_R.as_quat()
        scene_frame.wxyz = np.array([q[3], q[0], q[1], q[2]], np.float32)

    @btn_row1[0].on_click
    def _fx(_): _rotate_scene("x", 180)
    @btn_row1[1].on_click
    def _fy(_): _rotate_scene("y", 180)
    @btn_row1[2].on_click
    def _fz(_): _rotate_scene("z", 180)

    @btn_row2[0].on_click
    def _p90x(_): _rotate_scene("x",  90)
    @btn_row2[1].on_click
    def _p90y(_): _rotate_scene("y",  90)
    @btn_row2[2].on_click
    def _p90z(_): _rotate_scene("z",  90)

    @btn_row3[0].on_click
    def _m90x(_): _rotate_scene("x", -90)
    @btn_row3[1].on_click
    def _m90y(_): _rotate_scene("y", -90)
    @btn_row3[2].on_click
    def _m90z(_): _rotate_scene("z", -90)

    _up_rots = {
        "+Y ↑": st.Rotation.identity(),
        "-Y ↓": st.Rotation.from_euler("z", np.pi),
        "+Z ↑": st.Rotation.from_euler("x", -np.pi / 2),
        "-Z ↓": st.Rotation.from_euler("x",  np.pi / 2),
    }
    def _make_up_cb(label):
        def _cb(_):
            q = _up_rots[label].as_quat()
            scene_frame.wxyz = np.array([q[3], q[0], q[1], q[2]], np.float32)
        return _cb
    for label, btn in up_buttons.items():
        btn.on_click(_make_up_cb(label))

    @apply_rpy_btn.on_click
    def _apply_rpy(_):
        q = st.Rotation.from_euler("xyz", [
            np.deg2rad(roll_sl.value),
            np.deg2rad(pitch_sl.value),
            np.deg2rad(yaw_sl.value),
        ]).as_quat()
        scene_frame.wxyz = np.array([q[3], q[0], q[1], q[2]], np.float32)
        print(f"  Scene RPY set to ({roll_sl.value}°, {pitch_sl.value}°, {yaw_sl.value}°)")

    @reset_scene_btn.on_click
    def _reset_scene(_):
        scene_frame.wxyz     = np.array([1, 0, 0, 0], np.float32)
        scene_frame.position = np.zeros(3, np.float32)
        roll_sl.value = pitch_sl.value = yaw_sl.value = 0

    # ── mesh display ──────────────────────────────────────────────────────────
    mesh_handle = server.scene.add_mesh_trimesh(
        name="/scene/mesh",
        mesh=mesh,
    )

    # wireframe overlay: edges of the mesh as line segments
    edges      = mesh.edges_unique                          # (E,2)
    verts      = np.array(mesh.vertices, np.float32)
    wire_segs  = np.stack([verts[edges[:, 0]], verts[edges[:, 1]]], axis=1)  # (E,2,3)
    wire_color = np.full((*wire_segs.shape[:2], 3), 30, dtype=np.uint8)      # dark grey

    wire_handle = server.scene.add_line_segments(
        name="/scene/wireframe",
        points=wire_segs,
        colors=wire_color,
        line_width=0.5,
    )
    wire_handle.visible = False   # off by default

    with server.gui.add_folder("△ Mesh"):
        server.gui.add_markdown(
            f"*{len(mesh.vertices):,} vertices · {len(mesh.faces):,} faces · {path.name}*"
        )
        show_mesh_cb = server.gui.add_checkbox("Show mesh", initial_value=True)
        show_wire_cb = server.gui.add_checkbox("Show wireframe", initial_value=False)

    @show_mesh_cb.on_update
    def _toggle_mesh(event): mesh_handle.visible = event.target.value

    @show_wire_cb.on_update
    def _toggle_wire(event): wire_handle.visible = event.target.value

    print("  Viewer ready — navigate in browser, Ctrl-C to quit.\n")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nShutting down.")


def main():
    ap = argparse.ArgumentParser(description="Triangle mesh viewer (Viser).")
    ap.add_argument("--ply",  type=Path, required=True,
                    help=".ply triangle mesh file (e.g. from open3d Poisson reconstruction).")
    ap.add_argument("--port", type=int,  default=8080)
    args = ap.parse_args()

    print(f"Loading {args.ply} ...")
    mesh = load_mesh(args.ply)

    run_viewer(mesh, path=args.ply, port=args.port)


if __name__ == "__main__":
    main()
