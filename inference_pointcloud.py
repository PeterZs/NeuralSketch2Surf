import os
import glob
import argparse
import time
import numpy as np
import torch
import trimesh
from skimage import measure
from tqdm import tqdm
import sys

# Supports checkpoint inference
try:
    from train112TVloss import SwinReconstructionModule
except ImportError:
    print("Warning: train112TVloss.py not found")


def save_points_to_obj(path, verts):

    with open(path, 'w') as f:
        f.write("# Transformed Input PointCloud (Normalized)\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

def parse_obj_points(obj_path):

    # Read only the vertices (v) from the OBJ file
    verts = []
    try:
        with open(obj_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts: continue
                if parts[0] == 'v':
                    verts.append(list(map(float, parts[1:4])))
    except Exception as e:
        print(f"Error reading {obj_path}: {e}")
        return np.array([])
        
    return np.array(verts)

def compute_alignment_params(verts, resolution=112, margin=1.1):
    if len(verts) == 0: return None

    vmin = verts.min(axis=0)
    vmax = verts.max(axis=0)
    center = (vmin + vmax) / 2.0
    extent = vmax - vmin
    
    max_extent_scalar = float(max(extent.max(), 1e-12))
    
    target_radius = 1.0
    scale = target_radius / (max_extent_scalar / 2.0)
    
    # 3. Voxel Size
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

def voxelize_points(obj_path, resolution=112, margin=1.1):

    verts = parse_obj_points(obj_path)
    if len(verts) == 0: return None, None, None
    
    params = compute_alignment_params(verts, resolution, margin)
    
    center = params['center']
    scale = params['scale']
    origin = params['origin']
    voxel_size = params['voxel_size']
    R = params['resolution']
    
    verts_norm = (verts - center) * scale
    verts_vox_float = (verts_norm - origin) / voxel_size
    verts_idx = np.floor(verts_vox_float).astype(np.int32)
    
    volume = np.zeros((R, R, R), dtype=np.float32)
    
    mask = (verts_idx[:, 0] >= 0) & (verts_idx[:, 0] < R) & \
           (verts_idx[:, 1] >= 0) & (verts_idx[:, 1] < R) & \
           (verts_idx[:, 2] >= 0) & (verts_idx[:, 2] < R)
    
    valid_idx = verts_idx[mask]
    
    volume[valid_idx[:, 0], valid_idx[:, 1], valid_idx[:, 2]] = 1.0
                
    return volume[None, None, ...], params, verts_norm


class InferenceEngine:
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
        t_start = time.time()
        
        input_np, params, input_norm_verts = voxelize_points(obj_path, resolution, margin)
        if input_np is None:
            return False
        
        t_voxelization = time.time() - t_start

        save_input_norm_path = save_obj_path.replace("_recon.obj", "_input_norm.obj")
        save_points_to_obj(save_input_norm_path, input_norm_verts)

        # inference
        t_infer_start = time.time()
        input_tensor = torch.from_numpy(input_np).to(self.device)
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = torch.sigmoid(logits)
        
        # save raw tensor
        probs_np = probs.cpu().numpy()[0, 0, :, :, :]
        
        t_inference = time.time() - t_infer_start
        
        if probs_np.max() < threshold:
            print(f"⚠️ Empty prediction for {os.path.basename(obj_path)} (Max prob: {probs_np.max():.4f})")
            return False

        # Marching Cubes
        t_recon_start = time.time()
        try:
            verts_mc, faces_mc, normals_mc, values_mc = measure.marching_cubes(probs_np, level=threshold)
            
            voxel_size = params['voxel_size']
            origin = params['origin']
            scale = params['scale']
            center = params['center']
            
            # Index -> Norm
            verts_norm = verts_mc * voxel_size + origin
            # Norm -> World
            verts_orig = verts_norm / scale + center
            
            mesh = trimesh.Trimesh(vertices=verts_orig, faces=faces_mc)
            components = mesh.split(only_watertight=False)
            if len(components) > 0:
                mesh = max(components, key=lambda x: len(x.vertices))
            mesh.fix_normals()
            mesh.export(save_obj_path)
            
            t_reconstruction = time.time() - t_recon_start
            t_total = time.time() - t_start

            np.savez(
                save_npz_path,
                raw_probability_grid=probs_np,
                
                # Meta Data
                center=center,
                max_extent=params['max_extent'],
                margin=float(params['margin']),
                resolution=int(params['resolution']),
                voxel_size=float(params['voxel_size']),
                o3d_origin=params['origin'],
                scale=scale,
                
                # Timing 
                time_voxelization_sec=t_voxelization,
                time_inference_sec=t_inference,
                time_reconstruction_sec=t_reconstruction,
                time_total_sec=t_total,
                
                # Result Mesh
                mc_vertices=verts_orig,
                mc_faces=faces_mc
            )
            
            return True
            
        except Exception as e:
            print(f"Reconstruction failed for {obj_path}: {e}")
            return False

# main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/home/ids/asureshk-22/Hongsheng/checkpoint/best_model_jit_gpu.pt")
    parser.add_argument("--input_dir", type=str, default="/home/ids/asureshk-22/Hongsheng/SwinUNETRV2/pointcloud")
    parser.add_argument("--output_dir", type=str, default="./test_pointcloud")
    parser.add_argument("--threshold", type=float, default=0.5, help="Surface threshold")
    parser.add_argument("--img_size", type=int, default=112, help="Voxel resolution")
    parser.add_argument("--feature_size", type=int, default=24, help="Feature size used in training")
    
    parser.add_argument("--margin", type=float, default=1.4, help="Margin for voxelization")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    engine = InferenceEngine(
        args.model_path, 
        img_size=args.img_size, 
        feature_size=args.feature_size
    )
    
    files = sorted(glob.glob(os.path.join(args.input_dir, "**/*.obj"), recursive=True))
    print(f"Found {len(files)} point clouds. Using Strict Margin {args.margin} Logic...")
    
    count = 0
    for f_path in tqdm(files):
        if "_recon.obj" in f_path or "_input_norm.obj" in f_path: continue
        f_name = os.path.basename(f_path)
        save_name_obj = f_name.replace(".obj", "_recon.obj")
        save_name_npz = f_name.replace(".obj", "_data.npz")
        
        save_path_obj = os.path.join(args.output_dir, save_name_obj)
        save_path_npz = os.path.join(args.output_dir, save_name_npz)
        
        if engine.process_and_save(f_path, save_path_obj, save_path_npz, args.img_size, args.threshold, args.margin):
            count += 1
            
    print(f"\nCompleted! Results saved to {args.output_dir}")