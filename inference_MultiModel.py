"""Inference helper for sketches with multiple disconnected components.

This variant preserves normalized input curves and keeps multiple sufficiently
large mesh components after Marching Cubes.
"""
import os
import glob
import argparse
import time
import numpy as np
import torch
import trimesh
from skimage import measure
from tqdm import tqdm

from train112TVloss import SwinReconstructionModule



def save_lines_to_obj(path, verts, edges):
    """Export normalized sketch curves for visual debugging."""

    with open(path, 'w') as f:
        f.write("# Transformed Input Sketch (Normalized)\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for u, v in edges:
            f.write(f"l {u+1} {v+1}\n")


def bresenham3d(p0, p1):
    """Rasterize a 3D line segment into voxel coordinates."""
    x1, y1, z1 = map(int, p0)
    x2, y2, z2 = map(int, p1)
    dx, dy, dz = abs(x2 - x1), abs(y2 - y1), abs(z2 - z1)
    xs = 1 if x2 > x1 else -1
    ys = 1 if y2 > y1 else -1
    zs = 1 if z2 > z1 else -1
    pts = []
    
    if dx >= dy and dx >= dz:
        p1_err, p2_err = 2 * dy - dx, 2 * dz - dx
        while True:
            pts.append((x1, y1, z1))
            if x1 == x2: break
            x1 += xs
            if p1_err > 0: y1 += ys; p1_err -= 2 * dx
            if p2_err > 0: z1 += zs; p2_err -= 2 * dx
            p1_err += 2 * dy; p2_err += 2 * dz
    elif dy >= dx and dy >= dz:
        p1_err, p2_err = 2 * dx - dy, 2 * dz - dy
        while True:
            pts.append((x1, y1, z1))
            if y1 == y2: break
            y1 += ys
            if p1_err > 0: x1 += xs; p1_err -= 2 * dy
            if p2_err > 0: z1 += zs; p2_err -= 2 * dy
            p1_err += 2 * dx; p2_err += 2 * dz
    else:
        p1_err, p2_err = 2 * dy - dz, 2 * dx - dz
        while True:
            pts.append((x1, y1, z1))
            if z1 == z2: break
            z1 += zs
            if p1_err > 0: y1 += ys; p1_err -= 2 * dz
            if p2_err > 0: x1 += xs; p2_err -= 2 * dz
            p1_err += 2 * dy; p2_err += 2 * dx
    return pts

def parse_obj_robust(obj_path):
    """Read OBJ vertices and recover sketch edges from line or face elements."""
    verts, edges = [], []
    with open(obj_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts: continue
            if parts[0] == 'v':
                verts.append(list(map(float, parts[1:4])))
            elif parts[0] == 'l':
                indices = [int(x.split('/')[0]) - 1 for x in parts[1:]]
                for i in range(len(indices)-1):
                    edges.append((indices[i], indices[i+1]))
            elif parts[0] == 'f':
                indices = [int(x.split('/')[0]) - 1 for x in parts[1:]]
                for i in range(len(indices)):
                    edges.append((indices[i], indices[(i+1)%len(indices)]))
    return np.array(verts), edges

def compute_alignment_params(verts, resolution=112, margin=1.1):
    """Compute the canonical sketch-to-voxel transform."""
    if len(verts) == 0: return None

    vmin = verts.min(axis=0)
    vmax = verts.max(axis=0)
    center = (vmin + vmax) / 2.0
    extent = vmax - vmin
    
    max_extent_scalar = float(max(extent.max(), 1e-12))
    
    target_radius = 1.0
    scale = target_radius / (max_extent_scalar / 2.0)
    
    R = int(resolution)
    voxel_size = (2.0 * margin) / (R - 1)
    
    origin = np.array([-margin, -margin, -margin], dtype=np.float64)

    return {
        "center": center,
        "max_extent": np.array([max_extent_scalar]*3),
        "scale": scale,
        "resolution": R,
        "voxel_size": voxel_size,
        "origin": origin,
        "margin": margin
    }

def voxelize_strict_aligned(obj_path, resolution=112, margin=1.1):
    """Convert a curve sketch OBJ into a sparse occupancy tensor."""
    verts, edges = parse_obj_robust(obj_path)
    if len(verts) == 0: return None, None
    
    params = compute_alignment_params(verts, resolution, margin)
    
    center = params['center']
    scale = params['scale']
    origin = params['origin']
    voxel_size = params['voxel_size']
    R = params['resolution']
    
    verts_norm = (verts - center) * scale
    
    verts_vox_float = (verts_norm - origin) / voxel_size + 1e-6
    verts_idx = np.floor(verts_vox_float).astype(np.int32)
    verts_idx = np.clip(verts_idx, 0, R - 1)

    volume = np.zeros((R, R, R), dtype=np.float32)
    for u, v in edges:
        p0, p1 = verts_idx[u], verts_idx[v]
        points = bresenham3d(p0, p1)
        for x, y, z in points:
            if 0 <= x < R and 0 <= y < R and 0 <= z < R:
                volume[x, y, z] = 1.0
                
    return volume[None, None, ...], params

class InferenceEngine:
    """Run S2V-Net inference while retaining multiple output components."""
    def __init__(self, model_path, device='cuda', img_size=112, feature_size=24):
        self.device = torch.device(device)
        print(f"Loading model from: {model_path}")
        
        if model_path.endswith('.pt'):
            self.model = torch.jit.load(model_path, map_location=self.device)
        elif model_path.endswith('.ckpt'):
            class DummyArgs:
                def __init__(self):
                    self.img_size = img_size
                    self.feature_size = feature_size
                    self.dropout = 0.0
                    self.wce_weight = 1.0
                    self.tv_weight = 0.0
                    self.lr = 1e-4
            args = DummyArgs()
            self.model = SwinReconstructionModule.load_from_checkpoint(
                model_path, args=args, map_location=self.device, strict=False 
            )
        else:
            raise ValueError("Model path must end with .pt or .ckpt")

        self.model.eval()
        self.model.to(self.device)

    def process_and_save(self, obj_path, save_obj_path, save_npz_path, resolution=112, threshold=0.6, margin=1.1):
        """Process one sketch and save reconstruction outputs."""
        t_start = time.time()
        
        input_np, params = voxelize_strict_aligned(obj_path, resolution, margin)
        if input_np is None:
            return False
        
        t_voxelization = time.time() - t_start

        raw_verts, raw_edges = parse_obj_robust(obj_path)
        input_norm_verts = (raw_verts - params['center']) * params['scale']
        
        save_input_norm_path = save_obj_path.replace("_recon.obj", "_input_norm.obj")
        save_lines_to_obj(save_input_norm_path, input_norm_verts, raw_edges)

        t_infer_start = time.time()
        input_tensor = torch.from_numpy(input_np).to(self.device)
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = torch.sigmoid(logits)
        
        probs_np = probs.cpu().numpy()[0, 0, :, :, :]
        
        t_inference = time.time() - t_infer_start
        
        if probs_np.max() < threshold:
            print(f"Empty prediction for {os.path.basename(obj_path)}")
            return False

        t_recon_start = time.time()
        try:
            verts_mc, faces_mc, normals_mc, values_mc = measure.marching_cubes(probs_np, level=threshold)
            
            voxel_size = params['voxel_size']
            origin = params['origin']
            scale = params['scale']
            center = params['center']
            
            verts_norm = verts_mc * voxel_size + origin
            verts_orig = verts_norm / scale + center
            
            mesh = trimesh.Trimesh(vertices=verts_orig, faces=faces_mc)
            
            components = mesh.split(only_watertight=False)
            if len(components) > 0:
                valid_components = [c for c in components if len(c.vertices) > 100]
                if len(valid_components) > 0:
                    mesh = trimesh.util.concatenate(valid_components)
                else:
                    mesh = max(components, key=lambda x: len(x.vertices))
            
            mesh.fix_normals()
            mesh.export(save_obj_path)
            
            t_reconstruction = time.time() - t_recon_start
            t_total = time.time() - t_start

            np.savez(
                save_npz_path,
                raw_probability_grid=probs_np,
                center=center,
                max_extent=params['max_extent'],
                margin=float(params['margin']),
                resolution=int(params['resolution']),
                voxel_size=float(params['voxel_size']),
                o3d_origin=params['origin'],
                scale=scale,
                
                time_voxelization_sec=t_voxelization,
                time_inference_sec=t_inference,
                time_reconstruction_sec=t_reconstruction,
                time_total_sec=t_total,
                
                mc_vertices=verts_orig,
                mc_faces=faces_mc
            )
            
            return True
            
        except Exception as e:
            print(f"Reconstruction failed for {obj_path}: {e}")
            return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model (.pt or .ckpt)")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input sketch .obj files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save reconstructed outputs")
    parser.add_argument("--threshold", type=float, default=0.5, help="Surface threshold")
    parser.add_argument("--img_size", type=int, default=112, help="Voxel resolution")
    parser.add_argument("--feature_size", type=int, default=24, help="Feature size used in training")
    
    parser.add_argument("--margin", type=float, default=1.2, help="Margin for voxelization")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    engine = InferenceEngine(
        args.model_path, 
        img_size=args.img_size, 
        feature_size=args.feature_size
    )
    
    files = sorted(glob.glob(os.path.join(args.input_dir, "**/*.obj"), recursive=True))
    print(f"Found {len(files)} sketches. Using Strict Margin {args.margin} Logic...")
    
    count = 0
    for f_path in tqdm(files):
        if "_recon.obj" in f_path: continue
        
        f_name = os.path.basename(f_path)
        save_name_obj = f_name.replace(".obj", "_recon.obj")
        save_name_npz = f_name.replace(".obj", "_data.npz")
        
        save_path_obj = os.path.join(args.output_dir, save_name_obj)
        save_path_npz = os.path.join(args.output_dir, save_name_npz)
        
        if engine.process_and_save(f_path, save_path_obj, save_path_npz, args.img_size, args.threshold, args.margin):
            count += 1
            
    print(f"\nCompleted! Results saved to {args.output_dir}")
