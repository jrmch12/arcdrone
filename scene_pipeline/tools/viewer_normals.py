"""
viewer_normals.py
-----------------
Interactive viewer for point clouds with normals using Viser.

Loads a .ply that contains x,y,z,nx,ny,nz (e.g. output of nerfstudio / open3d
normal estimation) and renders the points plus toggleable normal quills.

Usage
-----
    python viewer_normals.py --ply pointcloud_normals.ply
    python viewer_normals.py --ply pointcloud_normals.ply --normal_length 0.05 --normal_every 20

    Then open  http://localhost:8080  in any browser.

Install
-------
    pip install viser plyfile numpy scipy
"""

import argparse
import time
from pathlib import Path

import numpy as np
import scipy.spatial.transform as st
from plyfile import PlyData
import viser


def load_cloud_with_normals(path: Path) -> dict:
    """Load a .ply with x,y,z,nx,ny,nz and optional RGB."""
    v = PlyData.read(str(path))["vertex"]
    prop_names = {p.name for p in v.properties}

    points = np.stack([
        np.asarray(v["x"], np.float32),
        np.asarray(v["y"], np.float32),
        np.asarray(v["z"], np.float32),
    ], axis=-1)

    missing = {"nx", "ny", "nz"} - prop_names
    if missing:
        raise ValueError(f"{path.name} is missing normal fields: {missing}")

    normals = np.stack([
        np.asarray(v["nx"], np.float32),
        np.asarray(v["ny"], np.float32),
        np.asarray(v["nz"], np.float32),
    ], axis=-1)

    if {"red", "green", "blue"} <= prop_names:
        raw = np.stack([np.asarray(v["red"]), np.asarray(v["green"]), np.asarray(v["blue"])], -1)
    elif {"r", "g", "b"} <= prop_names:
        raw = np.stack([np.asarray(v["r"]), np.asarray(v["g"]), np.asarray(v["b"])], -1)
    else:
        raw = np.full((len(points), 3), 180)

    if raw.dtype == np.uint8 or raw.max() > 1.0:
        colors = np.clip(raw, 0, 255).astype(np.uint8)
    else:
        colors = (np.clip(raw, 0, 1) * 255).astype(np.uint8)

    return {"points": points, "normals": normals, "colors": colors, "path": path}


def _normal_quills(points: np.ndarray, normals: np.ndarray, length: float, every: int):
    """Build (M,2,3) line-segment pairs for a subsampled set of normals."""
    pts = points[::every]
    nrm = normals[::every]
    starts = pts
    ends   = pts + nrm * length
    return np.stack([starts, ends], axis=1).astype(np.float32)   # (M,2,3)


def run_viewer(cloud: dict, port: int = 8080, point_size: float = 0.005,
               normal_length: float = 0.03, normal_every: int = 10):
    server = viser.ViserServer(port=port)
    server.scene.set_up_direction("+y")
    print(f"\n  ✓ Viser running → http://localhost:{port}\n")
    print(f"  {len(cloud['points']):,} points  —  {cloud['path'].name}")

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

    # ── point cloud ───────────────────────────────────────────────────────────
    with server.gui.add_folder("☁ Points"):
        size_sl = server.gui.add_slider("Point size", min=0.001, max=0.05, step=0.001, initial_value=point_size)

    pc = server.scene.add_point_cloud(
        name="/scene/points",
        points=cloud["points"],
        colors=cloud["colors"],
        point_size=point_size,
    )

    @size_sl.on_update
    def _size_update(event): pc.point_size = event.target.value

    # ── normals ───────────────────────────────────────────────────────────────
    # Color normals by direction: map xyz ∈ [-1,1] → rgb ∈ [0,255]
    nrm_unit  = cloud["normals"]
    nrm_color = ((nrm_unit * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)

    quill_segs = _normal_quills(cloud["points"], nrm_unit, normal_length, normal_every)
    M = len(quill_segs)
    # Repeat color per segment (one color per start point)
    quill_colors = np.repeat(nrm_color[::normal_every][:M], 2, axis=0).reshape(M, 2, 3)

    with server.gui.add_folder("→ Normals"):
        show_cb    = server.gui.add_checkbox("Show normals", initial_value=True)
        length_sl  = server.gui.add_slider("Length",   min=0.001, max=0.2,  step=0.001, initial_value=normal_length)
        every_sl   = server.gui.add_slider("Every Nth", min=1,    max=200,  step=1,     initial_value=normal_every)
        server.gui.add_markdown(f"*{M:,} quills shown (1 every {normal_every} points)*")

    normal_handle = server.scene.add_line_segments(
        name="/scene/normals",
        points=quill_segs,
        colors=quill_colors,
        line_width=1.0,
    )

    def _rebuild_quills():
        ev   = int(every_sl.value)
        ln   = float(length_sl.value)
        segs = _normal_quills(cloud["points"], nrm_unit, ln, ev)
        M2   = len(segs)
        cols = np.repeat(nrm_color[::ev][:M2], 2, axis=0).reshape(M2, 2, 3)
        normal_handle.points = segs
        normal_handle.colors = cols

    @show_cb.on_update
    def _toggle(event):
        normal_handle.visible = event.target.value

    @length_sl.on_update
    def _length_update(_): _rebuild_quills()

    @every_sl.on_update
    def _every_update(_): _rebuild_quills()

    print("  Viewer ready — navigate in browser, Ctrl-C to quit.\n")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nShutting down.")


def main():
    ap = argparse.ArgumentParser(description="Point cloud + normals viewer (Viser).")
    ap.add_argument("--ply",           type=Path,  required=True,
                    help=".ply file with x,y,z,nx,ny,nz fields.")
    ap.add_argument("--point_size",    type=float, default=0.005,
                    help="Initial point size (default: 0.005)")
    ap.add_argument("--normal_length", type=float, default=0.03,
                    help="Initial normal quill length (default: 0.03)")
    ap.add_argument("--normal_every",  type=int,   default=10,
                    help="Draw a quill every Nth point (default: 10)")
    ap.add_argument("--port",          type=int,   default=8080)
    args = ap.parse_args()

    print(f"Loading {args.ply} ...")
    cloud = load_cloud_with_normals(args.ply)
    print(f"  {len(cloud['points']):,} points with normals")

    run_viewer(
        cloud,
        port=args.port,
        point_size=args.point_size,
        normal_length=args.normal_length,
        normal_every=args.normal_every,
    )


if __name__ == "__main__":
    main()
