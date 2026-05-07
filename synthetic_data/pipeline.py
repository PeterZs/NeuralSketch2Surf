"""Generate paired geodesic sketches and voxel occupancy labels.

The synthetic data pipeline creates sparse curve inputs and dense occupancy
targets in a shared normalized grid, matching the supervision used by S2V-Net.
"""
from __future__ import annotations
import os
import argparse
import sys
from pathlib import Path
import numpy as np
import openmesh as om
import random

from geodesic import initGeodesic, findNextPoint
from voxelize_label import voxelize_label_from_stl
from voxelize_geodesic import voxelize_obj_geodesic

INPUT_MESH_DIR = Path("data/original_meshes")
OUTPUT_ROOT = Path("data/sketch_dataset_112")
OUTPUT_GEO_DIR = OUTPUT_ROOT / "geo"               
OUTPUT_LABEL_DIR = OUTPUT_ROOT / "voxelize_label"  
OUTPUT_META_DIR = OUTPUT_ROOT / "meta"             
OUTPUT_VOX_GEO_DIR = OUTPUT_ROOT / "voxelize_geo"  

RESOLUTION = 112
MARGIN = 1.1

random.seed(42)
np.random.seed(42)

def get_param_suffix(n_curves, len_percent, use_farthest):
    """Encode generation settings in output filenames."""
    len_str = f"{int(len_percent)}" if len_percent.is_integer() else f"{len_percent}"
    suffix = f"_N{n_curves}_L{len_str}"
    if use_farthest:
        suffix += "_farthest"
    return suffix

def save_curves_as_obj(filename, curves, normals):
    """Write generated geodesics as OBJ polylines."""
    with open(filename, "w") as f:
        f.write("# Exported geodesic curves\n")
        v_offset = 1
        for curve, nrm in zip(curves, normals):
            for v, vn in zip(curve, nrm):
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                f.write(f"vn {vn[0]:.6f} {vn[1]:.6f} {vn[2]:.6f}\n")
            for i in range(len(curve) - 1):
                f.write(f"l {v_offset + i} {v_offset + i + 1}\n")
            v_offset += len(curve)

def compute_geodesics_to_obj(mesh_path: Path, out_obj_path: Path, 
                             n_curves: int, len_percent: float) -> None:
    """Trace random geodesic curves on one mesh."""
    mesh = om.read_trimesh(str(mesh_path))
    mesh.update_normals()
    mesh_vertices = np.array(mesh.points())
    mesh_face_indices = np.array([[v.idx() for v in mesh.fv(fh)] for fh in mesh.faces()], dtype=int)
    
    if len(mesh_vertices) > 0:
        bbox_min = np.min(mesh_vertices, axis=0)
        bbox_max = np.max(mesh_vertices, axis=0)
        diagonal = np.linalg.norm(bbox_max - bbox_min)
    else:
        diagonal = 1.0
    
    target_length = (len_percent / 100.0) * diagonal
    faces_with_points = set()
    all_curves = []
    all_normals = []

    for _ in range(n_curves):
        try:
            fh, heh, lmbda, theta, P = initGeodesic(mesh, mesh_vertices, mesh_face_indices, faces_with_points)
        except ValueError:
            continue
        except Exception as e:
            print(f"Error in initGeodesic: {e}")
            continue

        faces_with_points.add(fh.idx())
        polyLine = [np.array(P)]
        polyNormals = []
        
        # Normals are stored for visualization; S2V-Net does not require them.
        nA = np.array(mesh.normal(mesh.from_vertex_handle(heh)))
        nB = np.array(mesh.normal(mesh.to_vertex_handle(heh)))
        start_n = (1.0 - lmbda) * nA + lmbda * nB
        start_n /= (np.linalg.norm(start_n) + 1e-12)
        polyNormals.append(start_n)

        current_len = 0.0
        step_count = 0
        safety_max_steps = 10000 

        while current_len < target_length and step_count < safety_max_steps:
            try:
                heh, lmbda, theta, nP = findNextPoint(mesh, heh, lmbda, theta, polyLine[-1])
                seg_len = np.linalg.norm(nP - polyLine[-1])
                current_len += seg_len
                polyLine.append(np.array(nP))
                faces_with_points.add(mesh.face_handle(heh).idx())
                
                nA = np.array(mesh.normal(mesh.from_vertex_handle(heh)))
                nB = np.array(mesh.normal(mesh.to_vertex_handle(heh)))
                interpN = (1.0 - lmbda) * nA + lmbda * nB
                interpN /= (np.linalg.norm(interpN) + 1e-12)
                polyNormals.append(interpN)
                step_count += 1
            except Exception:
                break
        
        all_curves.append(np.vstack(polyLine))
        all_normals.append(np.vstack(polyNormals))

    out_obj_path.parent.mkdir(parents=True, exist_ok=True)
    save_curves_as_obj(str(out_obj_path), all_curves, all_normals)


def stage1_geodesic_export(input_dir: Path, out_geo_dir: Path, 
                           param_suffix: str, n_curves: int, len_percent: float):
    """Generate raw curve sketches for all source meshes."""
    out_geo_dir.mkdir(parents=True, exist_ok=True)
    mesh_files = sorted(input_dir.glob("*.obj"))
    
    print(f"\n--- Stage 1: Geodesic Calculation (Suffix: {param_suffix}) ---")
    if not mesh_files:
        print("[WARN] No .obj files found in input directory.")

    for i, fpath in enumerate(mesh_files, 1):
        name = fpath.stem

        out_obj = out_geo_dir / f"{name}_geo{param_suffix}.obj"
        
        if out_obj.exists():
            print(f"[1/3] ({i}/{len(mesh_files)}) Skip (Exists): {out_obj.name}")
            continue

        try:
            print(f"[1/3] ({i}/{len(mesh_files)}) Processing: {out_obj.name}")
            compute_geodesics_to_obj(fpath, out_obj, n_curves, len_percent)
        except Exception as e:
            print(f"[ERR] Failed: {fpath.name} -> {e}")


def stage2_voxelize_label(input_dir: Path, out_label_dir: Path, out_meta_dir: Path, 
                          param_suffix: str):
    """Voxelize source meshes and save alignment metadata."""
    out_label_dir.mkdir(parents=True, exist_ok=True)
    out_meta_dir.mkdir(parents=True, exist_ok=True)
    mesh_files = sorted(input_dir.glob("*.obj"))
    
    print(f"\n--- Stage 2: Label Voxelization (Suffix: {param_suffix}) ---")
    for i, fpath in enumerate(mesh_files, 1):
        name = fpath.stem

        out_vox = out_label_dir / f"{name}_voxelize_label{param_suffix}.npy"
        out_meta = out_meta_dir / f"{name}_meta{param_suffix}.npz"
        
        if out_vox.exists() and out_meta.exists():
            print(f"[2/3] ({i}/{len(mesh_files)}) Skip (Exists): {out_vox.name}")
            continue

        try:
            print(f"[2/3] ({i}/{len(mesh_files)}) Processing: {out_vox.name}")
            voxelize_label_from_stl(
                stl_path=str(fpath),
                output_vox_path=str(out_vox),
                output_meta_path=str(out_meta),
                resolution=RESOLUTION,
                margin=MARGIN,
            )
        except Exception as e:
            print(f"[ERR] Failed: {fpath.name} -> {e}")

def stage3_voxelize_geodesic(geo_dir: Path, meta_dir: Path, out_vox_geo_dir: Path, 
                             param_suffix: str):
    """Voxelize geodesic curves using the mesh label coordinate frame."""
    out_vox_geo_dir.mkdir(parents=True, exist_ok=True)
    
    pattern = f"*_geo{param_suffix}.obj"
    geo_files = sorted(geo_dir.glob(pattern))
    
    print(f"\n--- Stage 3: Geodesic Voxelization (Suffix: {param_suffix}) ---")
    if not geo_files:
        print(f"[WARN] No geodesic files found matching pattern: {pattern}")
        return

    for i, geo_path in enumerate(geo_files, 1):
        suffix_len = len(f"_geo{param_suffix}.obj")
        name_part = geo_path.name[:-suffix_len]

        meta_path = meta_dir / f"{name_part}_meta{param_suffix}.npz"
        out_vox_path = out_vox_geo_dir / f"{name_part}_voxelize_geodesic{param_suffix}.npy"

        if out_vox_path.exists():
            print(f"[3/3] ({i}/{len(geo_files)}) Skip (Exists): {out_vox_path.name}")
            continue
        
        if not meta_path.exists():
            print(f"[ERR] Meta missing for {geo_path.name}. Expected: {meta_path.name}")
            continue

        print(f"[3/3] ({i}/{len(geo_files)}) Processing: {out_vox_path.name}")
        
        try:
            voxelize_obj_geodesic(str(geo_path), str(meta_path), str(out_vox_path))
        except Exception as e:
            print(f"[ERR] Failed: {geo_path.name} -> {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Geodesic Voxelization Pipeline")
    parser.add_argument("--n_curves", type=int, default=40, 
                        help="Number of geodesic curves per mesh (default: 130)")
    parser.add_argument("--len_percent", type=float, default=35.0, 
                        help="Curve length as percentage of diagonal (default: 20.0)")
    parser.add_argument("--farthest", action="store_true", 
                        help="Use farthest point sampling for geodesic initialization")
    
    args = parser.parse_args()
    
    if args.farthest:
        if "-farthest" not in sys.argv:
            sys.argv.append("-farthest")

    n_curves_val = args.n_curves
    len_percent_val = args.len_percent
    use_farthest_val = args.farthest
    
    current_suffix = get_param_suffix(n_curves_val, len_percent_val, use_farthest_val)

    if not INPUT_MESH_DIR.exists():
        raise SystemExit(f"Input directory does not exist: {INPUT_MESH_DIR}")

    print("=== Pipeline Started ===")
    print(f"Config: N_CURVES={n_curves_val}, LEN_PERCENT={len_percent_val}, FARTHEST={use_farthest_val}")
    print(f"Current File Suffix: {current_suffix}")

    stage1_geodesic_export(INPUT_MESH_DIR, OUTPUT_GEO_DIR, 
                           current_suffix, n_curves_val, len_percent_val)
    
    stage2_voxelize_label(INPUT_MESH_DIR, OUTPUT_LABEL_DIR, OUTPUT_META_DIR, 
                          current_suffix)
    
    stage3_voxelize_geodesic(OUTPUT_GEO_DIR, OUTPUT_META_DIR, OUTPUT_VOX_GEO_DIR, 
                             current_suffix)

    print("\n=== Pipeline Completed ===")
    print(f"Output Geo Dir     : {OUTPUT_GEO_DIR}")
    print(f"Output Label Dir   : {OUTPUT_LABEL_DIR}")
    print(f"Output Meta Dir    : {OUTPUT_META_DIR}")
    print(f"Output Vox Geo Dir : {OUTPUT_VOX_GEO_DIR}")
