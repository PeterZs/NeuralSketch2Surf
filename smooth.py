"""Interactive fidelity-vs-smoothness mesh post-processing.

The network predicts a voxel-derived proxy mesh; this tool lets users blend
between a smoother state and a sketch-adherent state while preserving topology.
"""
import numpy as np
import igl
import polyscope as ps
import trimesh
import os
import sys
from scipy import sparse

DATA = {
    'V_smooth': None,
    'V_fidelity': None,
    'F': None,
    'ratio': 0.4,
    'mesh_handle': None,
    'base_name': "output",
    'screenshot_count': 0
}

def read_obj_lines(filename):
    """Load OBJ line elements as a curve network."""
    if not os.path.exists(filename):
        print(f"Error: Skeleton file not found at {filename}")
        return np.array([]), np.array([])

    vertices, edges = [], []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('v '):
                vertices.append([float(x) for x in line.split()[1:4]])
            elif line.startswith('l '):
                tokens = line.split()[1:]
                indices = [int(t.split('/')[0]) - 1 for t in tokens]
                for i in range(len(indices) - 1):
                    edges.append([indices[i], indices[i+1]])
    
    V_arr = np.array(vertices, dtype=np.float64)
    if len(V_arr) == 0:
        return V_arr, np.array([])

    if len(edges) == 0:
        E_arr = np.column_stack([np.arange(len(V_arr)-1), np.arange(1, len(V_arr))])
    else:
        E_arr = np.array(edges, dtype=np.int64)
    return V_arr, E_arr

def solve_state(V_orig, S, target_pts, kp_val, lambd_val, iterations=100):
    """Iteratively balance Laplacian smoothing with attraction to sketch samples."""
    V = V_orig.copy()
    for _ in range(iterations):
        V = V + lambd_val * (S.dot(V) - V) + kp_val * (target_pts - V)
    return V

def update_mesh():
    """Update the displayed mesh for the current fidelity-smoothness ratio."""
    if DATA['V_smooth'] is None or DATA['V_fidelity'] is None:
        return
    t = DATA['ratio']
    V_blended = (1.0 - t) * DATA['V_smooth'] + t * DATA['V_fidelity']
    DATA['mesh_handle'].update_vertex_positions(V_blended)

def callback():
    """Polyscope UI callback for blending, screenshots, and export."""
    ps.imgui.PushItemWidth(200)
    
    changed, new_ratio = ps.imgui.SliderFloat("Fidelity vs Smooth", DATA['ratio'], 0.0, 1.0)
    if changed:
        DATA['ratio'] = new_ratio
        update_mesh()

    if ps.imgui.Button("Take Custom Screenshot"):
        current_dir = os.getcwd()
        shot_name = f"{DATA['base_name']}_{DATA['screenshot_count']}.png"
        full_path = os.path.join(current_dir, shot_name)
        ps.screenshot(full_path, False)
        print(f"Saved custom screenshot to: {full_path}")
        DATA['screenshot_count'] += 1

    ps.imgui.Separator()

    if ps.imgui.Button("Reset Rotation/Position"):
        DATA['mesh_handle'].set_transform(np.eye(4))

    if ps.imgui.Button("Export & Repair Mesh"):
        t = DATA['ratio']
        V_blended = (1.0 - t) * DATA['V_smooth'] + t * DATA['V_fidelity']
        
        T = DATA['mesh_handle'].get_transform()
        V_homo = np.hstack([V_blended, np.ones((V_blended.shape[0], 1))])
        V_final = (V_homo @ T.T)[:, :3]
        
        mesh = trimesh.Trimesh(vertices=V_final, faces=DATA['F'], process=True)
        print("Healing mesh geometry...")
        mesh.fill_holes()
        mesh.remove_degenerate_faces()
        mesh.remove_duplicate_faces()
        mesh.merge_vertices()
        mesh.fix_normals()
        mesh = trimesh.smoothing.filter_taubin(mesh, iterations=5)
        
        out_name = f"{DATA['base_name']}_smooth.obj"
        
        mesh.export(out_name)
        print(f"--- Exported: {out_name} ---")
        
    ps.imgui.PopItemWidth()

def main(skeleton_path, mesh_path):
    """Initialize smoothing states and launch the interactive viewer."""
    print(f"Loading Skeleton: {skeleton_path}")
    print(f"Loading Mesh:     {mesh_path}")

    V_curve, E_curve = read_obj_lines(skeleton_path)
    if len(V_curve) == 0:
        print("Error: Skeleton data is empty or file error.")
        return

    if not os.path.exists(mesh_path):
        print(f"Error: Mesh file not found at {mesh_path}")
        return

    V_p, F_p = igl.read_triangle_mesh(mesh_path)
    V_orig = np.array(V_p)
    DATA['F'] = np.array(F_p)
    DATA['base_name'] = os.path.splitext(os.path.basename(mesh_path))[0]

    mesh_tri = trimesh.Trimesh(vertices=V_orig, faces=DATA['F'], process=False)
    n = len(V_orig)
    edges = mesh_tri.edges_unique
    rows = np.concatenate([edges[:,0], edges[:,1]])
    cols = np.concatenate([edges[:,1], edges[:,0]])
    adj = sparse.coo_matrix((np.ones(len(edges)*2), (rows, cols)), shape=(n, n)).tocsr()
    degree = np.array(adj.sum(axis=1)).flatten()
    degree[degree == 0] = 1.0
    S = sparse.diags(1.0 / degree) @ adj 

    dummy_f = np.arange(len(V_curve), dtype=np.int64).reshape(-1, 1)
    # Closest sketch samples provide the fidelity anchors for surface editing.
    _, _, target_pts = igl.point_mesh_squared_distance(V_orig, V_curve, dummy_f)

    print("Caching smooth states...")
    DATA['V_smooth'] = solve_state(V_orig, S, target_pts, kp_val=0.01, lambd_val=0.6)
    DATA['V_fidelity'] = solve_state(V_orig, S, target_pts, kp_val=0.45, lambd_val=0.1)

    ps.init()
    ps.set_ground_plane_mode("none")
    ps.set_user_callback(callback)
    
    DATA['mesh_handle'] = ps.register_surface_mesh("Surface", V_orig, DATA['F'])
    
    try:
        DATA['mesh_handle'].set_transform_gizmo_enabled(True)
    except AttributeError:
        DATA['mesh_handle'].set_enabled_gizmo("full")

    DATA['mesh_handle'].set_smooth_shade(True)
    ps.register_curve_network("Skeleton", V_curve, E_curve)
    
    update_mesh()
    ps.show()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python smooth.py <sketch.obj> <reconstruction_mesh.obj>")

    skeleton_path = sys.argv[1]
    mesh_path = sys.argv[2]
    
    if os.path.exists(skeleton_path) and os.path.exists(mesh_path):
        main(skeleton_path, mesh_path)
    else:
        print("Error: files could not be found.")
        print(f"Skeleton path checked: {skeleton_path}")
        print(f"Mesh path checked:     {mesh_path}")
