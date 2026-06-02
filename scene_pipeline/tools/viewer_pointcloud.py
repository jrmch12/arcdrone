"""
viewer_pointcloud.py
--------------------
Interactive point cloud viewer using Viser.

Loads one or more plain .ply point cloud files (x,y,z + optional RGB)
and renders them with scene orientation controls and a live point-size slider.

Usage
-----
    python viewer_pointcloud.py --ply cloud.ply
    python viewer_pointcloud.py --ply a.ply b.ply --point_size 0.01 --port 8080

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


def load_pointcloud(path: Path) -> dict:
    """Load a plain point cloud .ply (x,y,z + optional RGB)."""
    v = PlyData.read(str(path))["vertex"]
    prop_names = {p.name for p in v.properties}

    points = np.stack([
        np.asarray(v["x"], np.float32),
        np.asarray(v["y"], np.float32),
        np.asarray(v["z"], np.float32),
    ], axis=-1)

    if {"red", "green", "blue"} <= prop_names:
        raw = np.stack([np.asarray(v["red"]), np.asarray(v["green"]), np.asarray(v["blue"])], -1)
    elif {"r", "g", "b"} <= prop_names:
        raw = np.stack([np.asarray(v["r"]), np.asarray(v["g"]), np.asarray(v["b"])], -1)
    else:
        raw = np.full((len(points), 3), 200)

    if raw.dtype == np.uint8 or raw.max() > 1.0:
        colors = np.clip(raw, 0, 255).astype(np.uint8)
    else:
        colors = (np.clip(raw, 0, 1) * 255).astype(np.uint8)

    return {"points": points, "colors": colors, "path": path}


def run_viewer(clouds: list[dict], port: int = 8080, point_size: float = 0.005):
    server = viser.ViserServer(port=port)
    server.scene.set_up_direction("+y")
    print(f"\n  ✓ Viser running → http://localhost:{port}\n")

    scene_frame = server.scene.add_frame(
        "/scene",
        wxyz=np.array([1, 0, 0, 0], np.float32),
        position=np.zeros(3, np.float32),
        show_axes=False,
    )

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
        delta    = st.Rotation.from_euler(axis, np.deg2rad(deg))
        new_R    = delta * cur_R
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
        r = np.deg2rad(roll_sl.value)
        p = np.deg2rad(pitch_sl.value)
        y = np.deg2rad(yaw_sl.value)
        q = st.Rotation.from_euler("xyz", [r, p, y]).as_quat()
        scene_frame.wxyz = np.array([q[3], q[0], q[1], q[2]], np.float32)
        print(f"  Scene RPY set to ({roll_sl.value}°, {pitch_sl.value}°, {yaw_sl.value}°)")

    @reset_scene_btn.on_click
    def _reset_scene(_):
        scene_frame.wxyz     = np.array([1, 0, 0, 0], np.float32)
        scene_frame.position = np.zeros(3, np.float32)
        roll_sl.value = pitch_sl.value = yaw_sl.value = 0

    pc_handles = []
    with server.gui.add_folder("☁ Point clouds"):
        size_sl = server.gui.add_slider("Point size", min=0.001, max=0.05, step=0.001, initial_value=point_size)

    for i, cloud in enumerate(clouds):
        pc = server.scene.add_point_cloud(
            name=f"/scene/pc_{i}",
            points=cloud["points"],
            colors=cloud["colors"],
            point_size=point_size,
        )
        pc_handles.append(pc)
        print(f"  [{i}] {cloud['path'].name}  — {len(cloud['points']):,} points")

    @size_sl.on_update
    def _size_update(event):
        for pc in pc_handles:
            pc.point_size = event.target.value

    print("  Viewer ready — navigate in browser, Ctrl-C to quit.\n")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nShutting down.")


def main():
    ap = argparse.ArgumentParser(description="Point cloud viewer (Viser).")
    ap.add_argument("--ply",        type=Path, nargs="+", required=True,
                    help="One or more plain point cloud .ply files.")
    ap.add_argument("--point_size", type=float, default=0.005,
                    help="Initial point size (default: 0.005)")
    ap.add_argument("--port",       type=int,   default=8080)
    args = ap.parse_args()

    clouds = []
    for p in args.ply:
        print(f"Loading {p} ...")
        c = load_pointcloud(p)
        print(f"  {len(c['points']):,} points  —  {p.name}")
        clouds.append(c)

    run_viewer(clouds, port=args.port, point_size=args.point_size)


if __name__ == "__main__":
    main()
