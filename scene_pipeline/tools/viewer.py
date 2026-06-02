"""
03_viewer.py
------------
Interactive Gaussian Splat viewer using Viser with PROPER 3DGS rendering.
Uses viser's native add_gaussian_splats() — full alpha-blended WebGL rasterizer.

Features
--------
- True Gaussian splatting rendering (not point clouds)
- Load one or more .ply splat files (scene + robots)
- Per-splat transform gizmo: drag to re-pose each splat live
- GUI sliders for per-splat opacity and scale multiplier
- "Export composed.ply" button — saves merged .ply at current poses
- Subsampling for large splats (controllable via --max_gaussians)

Usage
-----
    # Single splat
    python scene_pipeline/tools/viewer.py --ply robot_splat.ply

    # Scene + robot (robot gets a drag gizmo)
    python scene_pipeline/tools/viewer.py --ply scene.ply robot_splat.ply

    # Robot at a specific initial pose
    python scene_pipeline/tools/viewer.py --ply scene.ply robot_splat.ply \
        --init_xyz "0.5,0,0.1" --init_rpy "0,0,1.5708"

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
from plyfile import PlyData, PlyElement
import viser


# ── PLY helpers ───────────────────────────────────────────────────────────────

SH_C0 = 0.28209479177387814


def load_splat(path: Path) -> dict:
    """Load a nerfstudio-compatible Gaussian Splat .ply into numpy arrays."""
    v = PlyData.read(str(path))["vertex"]

    def g(name):
        return np.asarray(v[name], np.float32)

    means = np.stack([g("x"), g("y"), g("z")], axis=-1)          # (N,3)

    # DC SH → RGB [0,1]
    sh   = np.stack([g("f_dc_0"), g("f_dc_1"), g("f_dc_2")], -1)
    rgbs = np.clip(0.5 + SH_C0 * sh, 0, 1).astype(np.float32)   # (N,3)

    # Opacity: sigmoid of stored logit → (N,1)
    opacities = (1.0 / (1.0 + np.exp(-g("opacity"))))[:, None].astype(np.float32)

    # Scales: exp of stored log-scales → (N,3)
    scales = np.exp(
        np.stack([g("scale_0"), g("scale_1"), g("scale_2")], -1)
    ).astype(np.float32)

    # Quaternions: stored as w,x,y,z → keep as (N,4)
    quats_wxyz = np.stack([g("rot_0"), g("rot_1"), g("rot_2"), g("rot_3")], -1).astype(np.float32)
    norms = np.linalg.norm(quats_wxyz, axis=-1, keepdims=True)
    quats_wxyz = quats_wxyz / (norms + 1e-9)

    # Raw properties for export
    raw = {p.name: np.asarray(v[p.name], np.float32) for p in v.properties}

    return {
        "means":      means,
        "rgbs":       rgbs,
        "opacities":  opacities,
        "scales":     scales,
        "quats_wxyz": quats_wxyz,
        "raw":        raw,
        "path":       path,
    }


def quats_scales_to_covs(quats_wxyz: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """
    Compute 3×3 covariance matrices from quaternions and scales.
    Sigma = R @ diag(s)^2 @ R^T

    quats_wxyz : (N,4)  w,x,y,z
    scales     : (N,3)  actual scale values (not log)
    returns    : (N,3,3) float32
    """
    q_xyzw = quats_wxyz[:, [1, 2, 3, 0]]
    R      = st.Rotation.from_quat(q_xyzw).as_matrix().astype(np.float32)   # (N,3,3)
    S      = np.zeros((len(scales), 3, 3), np.float32)
    S[:, 0, 0] = scales[:, 0]
    S[:, 1, 1] = scales[:, 1]
    S[:, 2, 2] = scales[:, 2]
    RS  = R @ S
    cov = RS @ RS.transpose(0, 2, 1)
    return cov.astype(np.float32)


def subsample(splat: dict, max_n: int, seed: int = 0) -> dict:
    N = len(splat["means"])
    if N <= max_n:
        return splat
    rng = np.random.default_rng(seed)
    idx = rng.choice(N, max_n, replace=False)
    return {
        "means":      splat["means"][idx],
        "rgbs":       splat["rgbs"][idx],
        "opacities":  splat["opacities"][idx],
        "scales":     splat["scales"][idx],
        "quats_wxyz": splat["quats_wxyz"][idx],
        "raw":        {k: v[idx] for k, v in splat["raw"].items()},
        "path":       splat["path"],
    }


# ── export helper ─────────────────────────────────────────────────────────────

def _rotate_raw_quats(raw: dict, R: np.ndarray) -> dict:
    r_rot  = st.Rotation.from_matrix(R)
    q_wxyz = np.stack([raw["rot_0"], raw["rot_1"], raw["rot_2"], raw["rot_3"]], -1)
    q_xyzw = q_wxyz[:, [1, 2, 3, 0]]
    q_new  = (r_rot * st.Rotation.from_quat(q_xyzw)).as_quat()
    q_new_wxyz = q_new[:, [3, 0, 1, 2]].astype(np.float32)
    raw = dict(raw)
    raw["rot_0"], raw["rot_1"], raw["rot_2"], raw["rot_3"] = (
        q_new_wxyz[:, 0], q_new_wxyz[:, 1],
        q_new_wxyz[:, 2], q_new_wxyz[:, 3],
    )
    return raw


def export_composed(splats: list, frame_handles: list, out_path: Path,
                    scene_frame=None):
    # Bake scene-level rotation (from the orientation buttons) into the export
    R_scene   = np.eye(3, dtype=np.float32)
    pos_scene = np.zeros(3, dtype=np.float32)
    if scene_frame is not None:
        sf_wxyz   = np.array(scene_frame.wxyz, np.float32)
        pos_scene = np.array(scene_frame.position, np.float32)
        R_scene   = st.Rotation.from_quat(sf_wxyz[[1,2,3,0]]).as_matrix().astype(np.float32)

    all_raws = []
    for splat, fh in zip(splats, frame_handles):
        wxyz    = np.array(fh.wxyz, np.float32)             # w,x,y,z
        pos     = np.array(fh.position, np.float32)
        R_local = st.Rotation.from_quat(wxyz[[1,2,3,0]]).as_matrix().astype(np.float32)

        # world = R_scene @ (R_local @ p + pos_local) + pos_scene
        R         = R_scene @ R_local
        pos_world = R_scene @ pos + pos_scene

        raw   = dict(splat["raw"])   # scale already baked in by Apply scale button
        means = np.stack([raw["x"], raw["y"], raw["z"]], -1)
        means = (means @ R.T) + pos_world[None, :]
        raw["x"], raw["y"], raw["z"] = means[:,0], means[:,1], means[:,2]
        raw   = _rotate_raw_quats(raw, R)
        all_raws.append(raw)

    keys   = all_raws[0].keys()
    merged = {k: np.concatenate([r[k] for r in all_raws]) for k in keys}
    N      = len(merged["x"])
    dtype  = [(k, "f4") for k in merged]
    arr    = np.zeros(N, dtype=dtype)
    for k, v in merged.items():
        arr[k] = v
    el = PlyElement.describe(arr, "vertex")
    PlyData([el]).write(str(out_path))
    print(f"  Exported {N:,} Gaussians → {out_path}")


# ── viewer ────────────────────────────────────────────────────────────────────

def run_viewer(
    splats: list[dict],
    init_xyz=None,
    init_rpy=None,
    port: int = 8080,
    max_gaussians: int = 500_000,
    export_path: Path = Path("composed_export.ply"),
):
    server = viser.ViserServer(port=port)
    server.scene.set_up_direction("+y")
    print(f"\n  ✓ Viser running → http://localhost:{port}\n")

    # Subsample for display
    display = [subsample(s, max_gaussians, seed=i) for i, s in enumerate(splats)]

    gs_handles    = []   # viser GaussianSplatHandle — for live opacity/scale updates
    frame_handles = []   # viser FrameHandle with gizmo — for pose

    # ── scene root frame (reorientation) ────────────────────────────────────
    # All splats are children of this frame — rotating it rotates everything.
    scene_frame = server.scene.add_frame(
        "/scene",
        wxyz=np.array([1, 0, 0, 0], np.float32),
        position=np.zeros(3, np.float32),
        show_axes=False,
    )

    # ── global controls ──────────────────────────────────────────────────────
    with server.gui.add_folder("🌐 Scene orientation"):
        server.gui.add_markdown("**Quick fixes** (cumulative)")
        btn_row1 = [
            server.gui.add_button("⬆ Flip X",  hint="Rotate 180° around X"),
            server.gui.add_button("⬆ Flip Y",  hint="Rotate 180° around Y"),
            server.gui.add_button("⬆ Flip Z",  hint="Rotate 180° around Z"),
        ]
        btn_row2 = [
            server.gui.add_button("↻ +90° X"),
            server.gui.add_button("↻ +90° Y"),
            server.gui.add_button("↻ +90° Z"),
        ]
        btn_row3 = [
            server.gui.add_button("↺ -90° X"),
            server.gui.add_button("↺ -90° Y"),
            server.gui.add_button("↺ -90° Z"),
        ]

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
        apply_rpy_btn = server.gui.add_button("✔ Apply RPY (deg)")
        reset_scene_btn = server.gui.add_button("↺ Reset orientation")

    # Helper: compose a small rotation on top of the current scene frame
    def _rotate_scene(axis: str, deg: float):
        cur_wxyz = np.array(scene_frame.wxyz, np.float32)
        cur_R    = st.Rotation.from_quat(cur_wxyz[[1,2,3,0]])
        delta    = st.Rotation.from_euler(axis, np.deg2rad(deg))
        new_R    = delta * cur_R                            # left-multiply = world-space
        q_xyzw   = new_R.as_quat()
        scene_frame.wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], np.float32)

    # Flip buttons (180°)
    @btn_row1[0].on_click
    def _fx(_): _rotate_scene("x", 180)
    @btn_row1[1].on_click
    def _fy(_): _rotate_scene("y", 180)
    @btn_row1[2].on_click
    def _fz(_): _rotate_scene("z", 180)

    # +90° buttons
    @btn_row2[0].on_click
    def _p90x(_): _rotate_scene("x",  90)
    @btn_row2[1].on_click
    def _p90y(_): _rotate_scene("y",  90)
    @btn_row2[2].on_click
    def _p90z(_): _rotate_scene("z",  90)

    # -90° buttons
    @btn_row3[0].on_click
    def _m90x(_): _rotate_scene("x", -90)
    @btn_row3[1].on_click
    def _m90y(_): _rotate_scene("y", -90)
    @btn_row3[2].on_click
    def _m90z(_): _rotate_scene("z", -90)

    # Up-direction presets (canonical rotations to make the given axis point up)
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
            print(f"  Up direction set to {label}")
        return _cb
    for label, btn in up_buttons.items():
        btn.on_click(_make_up_cb(label))

    # Free RPY apply
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
        print("  Scene orientation reset")

    with server.gui.add_folder("💾 Export"):
        export_btn = server.gui.add_button("Export composed.ply")

    @export_btn.on_click
    def _export(_):
        print(f"  Exporting → {export_path}")
        export_composed(splats, frame_handles, export_path, scene_frame=scene_frame)

    # ── per-splat controls ───────────────────────────────────────────────────
    for i, (splat, ds) in enumerate(zip(splats, display)):
        name = splat["path"].stem
        N    = len(ds["means"])
        label = f"[{i}] {name} ({N:,} G)"

        # Initial pose
        if i == 1 and init_xyz is not None:
            pos  = np.array(init_xyz, np.float32)
        else:
            pos  = np.zeros(3, np.float32)

        if i == 1 and init_rpy is not None:
            q_xyzw = st.Rotation.from_euler("xyz", init_rpy).as_quat()
            wxyz   = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], np.float32)
        else:
            wxyz   = np.array([1, 0, 0, 0], np.float32)

        # Covariance matrices for proper GS rendering
        covs = quats_scales_to_covs(ds["quats_wxyz"], ds["scales"])  # (N,3,3)

        # Parent frame (gizmo)
        frame = server.scene.add_frame(
            f"/scene/splat_{i}",
            wxyz=wxyz,
            position=pos,
            show_axes=(len(splats) > 1),   # only show axes when compositing
            axes_length=0.12,
            axes_radius=0.004,
        )
        frame_handles.append(frame)

        # ── PROPER Gaussian splat rendering ──────────────────────────────────
        gs = server.scene.add_gaussian_splats(
            name=f"/scene/splat_{i}/gs",
            centers=ds["means"],
            covariances=covs,
            rgbs=ds["rgbs"],
            opacities=ds["opacities"],
        )
        gs_handles.append(gs)

        # Per-splat GUI controls
        with server.gui.add_folder(label):
            op_slider  = server.gui.add_slider(
                "Opacity scale", min=0.0, max=1.0, step=0.01, initial_value=1.0
            )
            sc_slider  = server.gui.add_slider(
                "Scale multiplier", min=0.1, max=3.0, step=0.05, initial_value=1.0
            )
            reset_btn  = server.gui.add_button("↺ Reset pose")
            info_label = server.gui.add_markdown(
                f"*{N:,} Gaussians — {splat['path'].name}*"
            )

            # ── live opacity update ──────────────────────────────────────────
            _base_op = ds["opacities"].copy()
            _ds      = ds
            _gs      = gs

            @op_slider.on_update
            def _op_update(event, gs=_gs, base_op=_base_op, ds=_ds):
                new_op = np.clip(base_op * event.target.value, 0, 1).astype(np.float32)
                gs.opacities = new_op

            # ── live scale update (Gaussian blob size) ───────────────────────
            _base_sc = ds["scales"].copy()
            _base_q  = ds["quats_wxyz"].copy()

            @sc_slider.on_update
            def _sc_update(event, gs=_gs, base_sc=_base_sc, base_q=_base_q):
                new_scales = (base_sc * event.target.value).astype(np.float32)
                gs.covariances = quats_scales_to_covs(base_q, new_scales)

            # ── bake scale to PLY ────────────────────────────────────────────
            scale_inp = server.gui.add_number(
                "Scale factor", initial_value=1.0, min=0.0001, step=0.001
            )
            apply_scale_btn = server.gui.add_button("Apply scale to PLY")

            @apply_scale_btn.on_click
            def _apply_scale(_, sp=splat, ds_=ds, gs=_gs,
                             base_sc=_base_sc, base_q=_base_q, inp=scale_inp):
                s = float(inp.value)
                if s <= 0:
                    print("  Scale must be > 0, skipped.")
                    return
                log_s = float(np.log(s))
                # Bake into full raw data (used by export)
                sp["raw"]["x"] = sp["raw"]["x"] * s
                sp["raw"]["y"] = sp["raw"]["y"] * s
                sp["raw"]["z"] = sp["raw"]["z"] * s
                sp["raw"]["scale_0"] = sp["raw"]["scale_0"] + log_s
                sp["raw"]["scale_1"] = sp["raw"]["scale_1"] + log_s
                sp["raw"]["scale_2"] = sp["raw"]["scale_2"] + log_s
                # Bake into display subset so viewport reflects new scale
                ds_["means"]  *= s
                ds_["scales"] *= s
                base_sc       *= s   # in-place: sc_slider stays calibrated
                # Reload viser handle
                gs.centers     = ds_["means"].astype(np.float32)
                gs.covariances = quats_scales_to_covs(base_q, base_sc)
                print(f"  Baked scale {s}× into {sp['path'].name}")

            # ── reset pose ───────────────────────────────────────────────────
            _frame = frame
            _pos   = pos.copy()
            _wxyz  = wxyz.copy()

            @reset_btn.on_click
            def _reset(_, fr=_frame, p=_pos, q=_wxyz):
                fr.position = p
                fr.wxyz     = q

    print("  Viewer ready — navigate in browser, Ctrl-C to quit.\n")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nShutting down.")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Proper 3DGS viewer (Viser native WebGL rasterizer)."
    )
    ap.add_argument("--ply",          type=Path, nargs="+", required=True,
                    help="One or more .ply splat files. First = scene, rest = robots.")
    ap.add_argument("--init_xyz",     type=str,  default=None,
                    help="Initial XYZ for 2nd splat: '0.5,0,0.1'")
    ap.add_argument("--init_rpy",     type=str,  default=None,
                    help="Initial RPY (rad) for 2nd splat: '0,0,1.5708'")
    ap.add_argument("--port",         type=int,  default=8080)
    ap.add_argument("--max_gaussians",type=int,  default=5_000_000,
                    help="Max Gaussians per splat for display (subsampled if larger)")
    ap.add_argument("--out",          type=Path, default=Path("composed_export.ply"),
                    help="Output path for the export button")
    args = ap.parse_args()

    init_xyz = [float(x) for x in args.init_xyz.split(",")] if args.init_xyz else None
    init_rpy = [float(x) for x in args.init_rpy.split(",")] if args.init_rpy else None

    splats = []
    for p in args.ply:
        print(f"Loading {p} ...")
        s = load_splat(p)
        print(f"  {len(s['means']):,} Gaussians  —  {p.name}")
        splats.append(s)

    run_viewer(
        splats,
        init_xyz=init_xyz,
        init_rpy=init_rpy,
        port=args.port,
        max_gaussians=args.max_gaussians,
        export_path=args.out,
    )


if __name__ == "__main__":
    main()