****  Currently, each folder displays only the top 20 entries for each data category. For the complete dataset, please refer to the link below:
https://nextcloud.r2.enst.fr/nextcloud/index.php/s/dsjq7k8zGkNNCzc   ****




## 1. Directory Overview
The output directory contains four subfolders corresponding to different stages of the data generation pipeline:

```text
Data/Sketch_Dataset_112/
├── 📁 geo/                 # Stage 1: Raw 3D geodesic curves (.obj)
├── 📁 voxelize_label/      # Stage 2: Solid filled voxel grid of the mesh (.npy)
├── 📁 meta/                # Stage 2: Spatial alignment & normalization parameters (.npz)
└── 📁 voxelize_geo/        # Stage 3: Voxelized representation of the curves (.npy)
```

---

## 2. Detailed File Descriptions

### 🔹 1. `geo/` (Geodesic Curves)
*   **Format:** Wavefront OBJ (`.obj`)
*   **Content:** Contains the computed random geodesic curves on the mesh surface.
*   **Structure:** Unlike standard meshes, these OBJ files contain vertices (`v`), normals (`vn`), and line segments (`l`). They **do not** contain faces (`f`).
*   **Usage:** Synthetic Data for 3D Sketch Simulation.

### 🔹 2. `voxelize_label/` (Ground Truth Volume)
*   **Format:** NumPy Binary (`.npy`)
*   **Shape:** $(`112, 112, 112`)
*   **Content:** A binary 3D grid representing the solid shape of the original object.
    *   `1`: Inside the object or on the surface.
    *   `0`: Empty space / Background.
*   **Usage:** Used as the target label for 3D shape reconstruction tasks.

### 🔹 3. `meta/` (Normalization Metadata)
*   **Format:** NumPy Zipped Archive (`.npz`)
*   **Content:** Stores the spatial transformation parameters used to normalize the mesh into the voxel grid.
*   **Key Keys:**
    *   `resolution`: The grid size (e.g., 112).
    *   `voxel_size`: Physical size of one voxel unit.
    *   `o3d_origin`: The world coordinate of the voxel grid's origin $(0,0,0)$.
    *   `center` & `max_extent`: Used for centering and scaling the original mesh.
*   **Importance:** These parameters ensure that the **Label** (Folder 2) and the **Geodesic Voxels** (Folder 4) are perfectly aligned in the same 3D coordinate system.

### 🔹 4. `voxelize_geo/` (Voxelized Curves)
*   **Format:** NumPy Binary (`.npy`)
*   **Shape:** $( `112, 112, 112`)
*   **Content:** A sparse binary 3D grid representing *only* the geodesic curves.
    *   `1`: The voxel contains a part of a geodesic curve.
    *   `0`: Empty space.
*   **Usage:** This is  input "sketch"  for 3D completion networks.

---

## 3. Naming Convention

Files follow a consistent naming schema to indicate generation parameters:

```text
{ModelName}_[Type]_{Suffix}.ext
```

*   **ModelName**: Original filename of the mesh (e.g., `bunny`, `2_liter`).
*   **Type**:
    *   `geo`: Geodesic OBJ file.
    *   `voxelize_label`: Solid voxel grid.
    *   `voxelize_geodesic`: Curve voxel grid.
    *   `meta`: Metadata file.
*   **Suffix** (Example: `_N25_L80`):
    *   `N{int}`: Number of geodesic curves generated per mesh (e.g., 25 curves).
    *   `L{float}`: Length of curves as a percentage of the mesh diagonal (e.g., 80%).

**Example:**
*   `bunny_geo_N25_L80.obj`
*   `bunny_meta_N25_L80.npz`