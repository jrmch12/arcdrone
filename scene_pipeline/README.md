# 3DGS -> Collision Mesh (NKSR)

This folder implements the mesh-only pipeline you asked for:

1. Extract points from your 3DGS `.ply`
2. Estimate/orient normals
3. Reconstruct mesh with NKSR
4. Clean/decimate final collision mesh

## Install

```bash
pip install -r mjo/requirements.txt
```

If `pip install nksr` fails in your environment, install NKSR from source or use the wheel command from NKSR docs.

## Run on your scene

Input scene:
`/home/jrmch12/Documents/code/260430_learning_gs/scenes/SuperSplat/Boscawen-Ûn/scene.ply`

```bash
mkdir -p mjo/output

python mjo/step1_extractpointcloud.py \
  --input-ply "/home/jrmch12/Documents/code/260430_learning_gs/scenes/SuperSplat/Boscawen-Ûn/scene.ply" \
  --output-ply mjo/output/pointcloud_raw.ply \
  --opacity-logit-threshold 0.0 \
  --voxel-size 0.05

python mjo/step1b_filter_cloud.py \
  --input-ply mjo/output/pointcloud_raw.ply \
  --output-ply mjo/output/pointcloud_filtered.ply \
  --up-axis y \
  --max-height 2.0

python mjo/step2_estimate_normals.py \
  --input-ply mjo/output/pointcloud_filtered.ply \
  --output-ply mjo/output/pointcloud_normals.ply \
  --normal-radius 0.15 \
  --normal-max-nn 60 \
  --orient-k 30

python mjo/step3_nksr_mesh.py \
  --input-ply mjo/output/pointcloud_normals.ply \
  --output-mesh mjo/output/mesh_raw_nksr.ply \
  --device cuda:0 \
  --detail-level 1.0 \
  --mise-iter 2

# Alternative: Poisson reconstruction (Open3D only, no GPU/neural network needed)
python mjo/step3p1_poisson_mesh.py \
  --input-ply mjo/output/pointcloud_normals.ply \
  --output-mesh mjo/output/mesh_raw_poisson.ply \
  --depth 10 \
  --density-threshold 0.05

python mjo/step4_cleanup_mesh.py \
  --input-mesh mjo/output/mesh_raw_poisson.ply \
  --output-mesh mjo/output/mesh_collision_final.ply \
  --fill-holes \
  --keep-largest \
  --target-triangles 200000
```

## Fixing flipped geometry (very common with 3DGS scenes)

3DGS scenes often come out with Y or Z flipped.  Use the interactive
fix-and-export tool to correct orientation and/or normals **before** running
Poisson reconstruction, or to fix the mesh afterwards.

```bash
# Fix an upside-down point cloud with normals (pre-Poisson)
python mjo/tools/fix_and_export.py \
  --input  mjo/output/pointcloud_normals.ply \
  --output mjo/output/pointcloud_normals_fixed.ply

# Fix a Poisson mesh whose faces are inside-out
python mjo/tools/fix_and_export.py \
  --input  mjo/output/mesh_raw_poisson.ply \
  --output mjo/output/mesh_fixed.ply
```

Open `http://localhost:8080` in your browser.

**Typical workflow for a flipped scene:**

1. Open the tool on your point cloud / mesh.
2. In **Scene orientation** → click **⬆ Flip X** (or Flip Y / Flip Z, or use
   the 90° buttons) until the geometry looks right-side up in the browser.
3. For a point cloud: open **→ Normals** and toggle *Show normals*.  The
   coloured quills should point *away from the surface* (outward into the air).
   If they point into the ground, click **↕ Flip all normals** — the preview
   updates live.
4. For a mesh: if the textured/coloured side faces inward, click
   **↕ Flip face winding** in the **△ Mesh** panel — the preview updates live.
5. Click **✔ Export fixed PLY** in the **💾 Export** panel.  The tool bakes
   the visual rotation *and* any normals/winding flip into the file's actual
   coordinates and saves it.

Then rerun the subsequent step using the fixed file as input.

> **Why do normals point inward?**  `orient_normals_consistent_tangent_plane`
> propagates orientation consistently but has no notion of "inside" vs
> "outside".  The initial seed direction is arbitrary, so the whole cloud can
> end up flipped.  Clicking **↕ Flip all normals** is the fast fix.

## Tuning tips

- If mesh is noisy: increase `--opacity-logit-threshold`, increase `--voxel-size`.
- If geometry is too smooth/lost detail: decrease `--voxel-size` and increase `--detail-level`.
- If normals are unstable: increase `--normal-radius` and `--orient-k`.

