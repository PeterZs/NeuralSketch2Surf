# Data

This directory contains sample data for the NeuralSketch2Surf training. 

The paper dataset was generated from closed manifold meshes selected from Greyc3D, SHREC07, and SHREC15. For each source mesh, the synthetic pipeline creates sparse geodesic sketches and aligned dense occupancy labels.

## Directory Layout

```text
data/
|-- original_meshes/             # Source meshes used by the synthetic data pipeline
`-- sketch_dataset_112/
    |-- geo/                     # Raw geodesic sketch curves (.obj)
    |-- voxelize_geo/            # Voxelized sketch inputs (.npy)
    |-- voxelize_label/          # Filled ground-truth occupancy volumes (.npy)
    `-- meta/                    # Normalization and alignment metadata (.npz)
```

The training script expects `voxelize_geo/` and `voxelize_label/` to be paired by filename. The `meta/` files are used to keep the sketch voxels and label voxels in the same normalized coordinate frame.

## Folder Details

### `geo/`

- **Format:** Wavefront OBJ (`.obj`)
- **Content:** Synthetic geodesic sketch curves sampled on the source mesh surface.
- **OBJ structure:** vertices (`v`), normals (`vn`), and line elements (`l`); these files do not contain mesh faces.
- **Role:** Human-sketch proxy used for visualization and inference-style curve inputs.

Example:

```text
1_geo_N25_L80.obj
```

### `voxelize_geo/`

- **Format:** NumPy array (`.npy`)
- **Shape:** `(112, 112, 112)`
- **Values:** binary occupancy of sparse sketch curves.
- **Role:** Input tensor for S2V-Net.

`1` means the voxel intersects a sketch curve; `0` means empty space.

### `voxelize_label/`

- **Format:** NumPy array (`.npy`)
- **Shape:** `(112, 112, 112)`
- **Values:** binary solid occupancy of the target mesh.
- **Role:** Supervision target for training.

`1` means inside or on the target object; `0` means outside.

### `meta/`

- **Format:** NumPy archive (`.npz`)
- **Role:** Stores the normalization parameters shared by the sketch and label volumes.

Common keys:

- `resolution`: voxel grid size, normally `112`;
- `voxel_size`: physical size of one voxel in normalized space;
- `o3d_origin`: voxel grid origin used by Open3D voxelization;
- `center`: source mesh bounding-box center;
- `max_extent`: source mesh scale reference;
- `margin`: normalized voxelization margin.

## Naming Convention

Files encode the source model and generation parameters:

```text
{ModelName}_{Type}_N{CurveCount}_L{LengthPercent}.{ext}
```

Examples:

```text
1_geo_N25_L80.obj
1_voxelize_geodesic_N25_L80.npy
1_voxelize_label_N25_L80.npy
1_meta_N25_L80.npz
```

Parameter meanings:

- `N{CurveCount}`: number of geodesic curves generated on the mesh;
- `L{LengthPercent}`: curve length as a percentage of the source mesh bounding-box diagonal.

## Generate Data

Put source OBJ meshes in:

```text
data/original_meshes/
```

Then run from the repository root:

```bash
python synthetic_data/pipeline.py \
  --n_curves 25 \
  --len_percent 80 \
  --farthest
```

The pipeline performs three stages:

1. generate geodesic curves and save them to `geo/`;
2. voxelize the source mesh into `voxelize_label/` and save alignment metadata to `meta/`;
3. voxelize the geodesic curves into `voxelize_geo/` using the same coordinate frame.

## Quick Sanity Check

You can inspect a generated pair with:

```python
import numpy as np

sketch = np.load("data/sketch_dataset_112/voxelize_geo/1_voxelize_geodesic_N25_L80.npy")
label = np.load("data/sketch_dataset_112/voxelize_label/1_voxelize_label_N25_L80.npy")

print(sketch.shape, sketch.dtype, sketch.sum())
print(label.shape, label.dtype, label.sum())
```

Both arrays should have shape `(112, 112, 112)`. The sketch occupancy is sparse, while the label occupancy is a filled volume.

## Notes

- The train/validation split in `train112TVloss.py` is object-level: all variants from the same source object stay in the same split.
- The current release targets closed-surface reconstruction. Open surfaces and isolated decorative strokes are not represented by these labels.
- Keep the four generated folders synchronized. Mixing `meta/`, `voxelize_geo/`, and `voxelize_label/` files from different generation parameters will misalign the input and target volumes.
