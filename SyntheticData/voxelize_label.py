import numpy as np
import trimesh
import open3d as o3d
from scipy.ndimage import binary_fill_holes
from pathlib import Path

def voxelize_label_from_stl(
    stl_path: str,
    output_vox_path: str,
    output_meta_path: str,
    resolution: int = 64,
    margin: float = 1.05,
) -> None:

    mesh = trimesh.load(stl_path, force='mesh')
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    
    verts_np = mesh.vertices.view(np.ndarray).astype(np.float64)
    vmin = verts_np.min(axis=0)
    vmax = verts_np.max(axis=0)
    center = (vmin + vmax) / 2.0
    extent = vmax - vmin
    max_extent_scalar = float(max(extent.max(), 1e-12))
    max_extent = np.array([max_extent_scalar] * 3, dtype=np.float64)
    R = int(resolution)
    voxel_size = (2.0 * margin) / (R - 1)

    # Normalize
    target_radius = 1.0  
    scale = target_radius / (max_extent_scalar / 2.0)
    verts_norm = (verts_np - center) * scale

    # Open3D voxelization
    o3d_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(verts_norm),
        triangles=o3d.utility.Vector3iVector(mesh.faces.astype(np.int32)),
    )
    o3d_mesh.compute_vertex_normals()
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(
        o3d_mesh,
        voxel_size=voxel_size,
    )
    
    o3d_origin = np.asarray(voxel_grid.origin, dtype=np.float64)

    target_origin = np.array([-margin, -margin, -margin])
    
    offset_float = (o3d_origin - target_origin) / voxel_size
    
    offset_int = np.round(offset_float).astype(np.int32)
    
    real_origin = o3d_origin - (offset_int * voxel_size)
    
    vox = np.zeros((R, R, R), dtype=np.uint8)
    
    for v in voxel_grid.get_voxels():
        gi_local = np.array(v.grid_index, dtype=np.int32)
        gi_global = gi_local + offset_int
        
        if np.all(gi_global >= 0) and np.all(gi_global < R):
            vox[gi_global[0], gi_global[1], gi_global[2]] = 1

    vox = binary_fill_holes(vox).astype(np.uint8)

    # Construct meta 
    meta = {
        "center": center.astype(np.float64),
        "max_extent": max_extent.astype(np.float64),
        "margin": float(margin),
        "resolution": R,
        "voxel_size": float(voxel_size),
        "o3d_origin": real_origin, 
    }

    Path(output_vox_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_meta_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(output_vox_path, vox)
    np.savez(output_meta_path, **meta)
    
    print(f"finish {Path(stl_path).name} -> shape={vox.shape}, sum={vox.sum()}")