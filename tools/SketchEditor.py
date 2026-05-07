"""Interactive editor for cleaning 3D sketch curve networks."""
import polyscope as ps
import polyscope.imgui as psim
import numpy as np
import sys
import os
import math
from pathlib import Path

class SketchEditor:
    """Polyscope-based eraser tool for ASCII PLY edge sketches."""

    def __init__(self, filename):
        self.filename = filename
        self.raw_vertices = None
        self.raw_edges = None
        
        self.eraser_mode = False
        
        self.debug_mode = False       

        self.load_ply(filename)
        
        ps.init()
        ps.set_up_dir("z_up")

        self.update_visualization()
        
        ps.set_user_callback(self.callback)
        
        print(f"Loaded file: {filename}")
        print("1. Check 'Eraser Mode'")
        print("2. Move mouse to turn red, [Left Click] to delete")
        
        ps.show()

    def load_ply(self, path):
        """Load vertices and edges from a simple ASCII PLY file."""
        with open(path, 'r') as f:
            lines = f.readlines()

        header_end = 0
        num_verts = 0
        num_edges = 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith("element vertex"):
                num_verts = int(line.split()[-1])
            elif line.startswith("element edge"):
                num_edges = int(line.split()[-1])
            elif line == "end_header":
                header_end = i + 1
                break
        
        vert_data = []
        for i in range(header_end, header_end + num_verts):
            vert_data.append(list(map(float, lines[i].strip().split()[:3])))
            
        edge_data = []
        for i in range(header_end + num_verts, header_end + num_verts + num_edges):
            edge_data.append(list(map(int, lines[i].strip().split()[:2])))

        self.raw_vertices = np.array(vert_data)
        self.raw_edges = np.array(edge_data)

    def get_clean_data(self):
        """Compact vertices after edge deletion."""

        if self.raw_edges is None or len(self.raw_edges) == 0:
            return np.zeros((0, 3)), np.zeros((0, 2), dtype=int)

        used_indices, new_edge_indices = np.unique(self.raw_edges, return_inverse=True)
        
        clean_vertices = self.raw_vertices[used_indices]
        
        clean_edges = new_edge_indices.reshape(self.raw_edges.shape)
        
        return clean_vertices, clean_edges

    def save_ply(self, path):
        """Save the currently visible curve network as ASCII PLY."""
        clean_verts, clean_edges = self.get_clean_data()
        
        if len(clean_edges) == 0:
            print("Warning: Model is empty, not saved.")
            return
        
        with open(path, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(clean_verts)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write(f"element edge {len(clean_edges)}\n")
            f.write("property int vertex1\n")
            f.write("property int vertex2\n")
            f.write("end_header\n")
            
            for v in clean_verts:
                f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
            for e in clean_edges:
                f.write(f"{e[0]} {e[1]}\n")
        
        print(f"model saved: {path}")

    def update_visualization(self):
        """Refresh the Polyscope curve network after edits."""

        viz_verts, viz_edges = self.get_clean_data()
        
        if len(viz_edges) > 0:
            ps_net = ps.register_curve_network("Sketch Lines", viz_verts, viz_edges)
            ps_net.set_color((0.2, 0.2, 0.2)) 
            ps_net.set_radius(0.005, relative=False)
        else:
            if ps.has_curve_network("Sketch Lines"):
                ps.remove_curve_network("Sketch Lines")

    def build_view_matrix(self, eye, look_dir, up):
        f = look_dir / np.linalg.norm(look_dir)
        u = up / np.linalg.norm(up)
        s = np.cross(f, u)
        s = s / np.linalg.norm(s)
        u = np.cross(s, f)
        
        view_mat = np.eye(4)
        view_mat[0, :3] = s
        view_mat[1, :3] = u
        view_mat[2, :3] = -f
        view_mat[0, 3] = -np.dot(s, eye)
        view_mat[1, 3] = -np.dot(u, eye)
        view_mat[2, 3] = np.dot(f, eye)
        return view_mat

    def build_projection_matrix(self, fov_deg, aspect, near=0.01, far=1000.0):
        fov_rad = math.radians(fov_deg)
        tan_half_fov = math.tan(fov_rad / 2.0)
        proj_mat = np.zeros((4, 4))
        proj_mat[0, 0] = 1.0 / (aspect * tan_half_fov)
        proj_mat[1, 1] = 1.0 / tan_half_fov
        proj_mat[2, 2] = -(far + near) / (far - near)
        proj_mat[2, 3] = -(2.0 * far * near) / (far - near)
        proj_mat[3, 2] = -1.0
        return proj_mat

    def get_mvp_matrix_manual(self):
        params = ps.get_view_camera_parameters()
        pos = np.array(params.get_position())
        look_dir = np.array(params.get_look_dir())
        up_dir = np.array(params.get_up_dir())
        fov = params.get_fov_vertical_deg()
        win_w, win_h = ps.get_window_size()
        aspect = win_w / win_h
        
        view_mat = self.build_view_matrix(pos, look_dir, up_dir)
        proj_mat = self.build_projection_matrix(fov, aspect)
        vp_mat = proj_mat @ view_mat
        
        return vp_mat, win_w, win_h

    def project_to_screen(self, points_3d):
        vp_mat, width, height = self.get_mvp_matrix_manual()
        ones = np.ones((len(points_3d), 1))
        verts_homo = np.hstack([points_3d, ones])
        clip_coords = (vp_mat @ verts_homo.T).T
        
        w = clip_coords[:, 3:4]
        w[w < 0.001] = 0.001 
        ndc_coords = clip_coords[:, :3] / w
        
        screen_x = (ndc_coords[:, 0] + 1) * 0.5 * width
        screen_y = (1 - ndc_coords[:, 1]) * 0.5 * height 
        
        return np.stack([screen_x, screen_y], axis=1)

    def find_closest_edge(self, mouse_x, mouse_y):
        """Pick the closest projected edge to the mouse cursor."""
        if self.raw_edges is None or len(self.raw_edges) == 0:
            return -1
            
        edge_starts = self.raw_vertices[self.raw_edges[:, 0]]
        edge_ends = self.raw_vertices[self.raw_edges[:, 1]]
        
        all_points = np.vstack([edge_starts, edge_ends])
        screen_points = self.project_to_screen(all_points)
        
        num = len(self.raw_edges)
        p1 = screen_points[:num]
        p2 = screen_points[num:]
        
        p_mouse = np.array([mouse_x, mouse_y])
        
        ab = p2 - p1
        ap = p_mouse - p1
        ab_sq = np.sum(ab**2, axis=1)
        ab_sq[ab_sq == 0] = 1e-6 
        t = np.sum(ap * ab, axis=1) / ab_sq
        t = np.clip(t, 0.0, 1.0)
        
        closest = p1 + ab * t[:, np.newaxis]
        dists = np.linalg.norm(p_mouse - closest, axis=1)
        
        min_idx = np.argmin(dists)
        
        if dists[min_idx] < 60.0:
            return min_idx
        return -1

    def callback(self):
        """Draw the editor UI and handle eraser interactions."""
        psim.SetNextWindowPos((10, 10), psim.ImGuiCond_FirstUseEver)
        psim.SetNextWindowSize((320, 0)) 
        psim.Begin("Toolbox", True)
        
        curr_v, curr_e = self.get_clean_data()
        psim.Text(f"Vertices: {len(curr_v)} | Edges: {len(curr_e)}")
        
        psim.Separator()
        _, self.eraser_mode = psim.Checkbox("Eraser Mode", self.eraser_mode)
        if self.eraser_mode:
            psim.TextColored((1, 0, 0, 1), ">> Left Click to DELETE <<")
        else:
            psim.TextDisabled(">> View Mode <<")
            
        psim.Separator()
        _, self.debug_mode = psim.Checkbox("Debug Mode", self.debug_mode)

        psim.Separator()
        if psim.Button("Save Cleaned PLY"):
            base, ext = os.path.splitext(self.filename)
            save_name = f"{base}_cleaned{ext}"
            self.save_ply(save_name)
            
        psim.End()

        if not self.eraser_mode:
            if ps.has_curve_network("Highlight"): ps.remove_curve_network("Highlight")
            return

        io = psim.GetIO()
        if io.WantCaptureMouse: return
        
        try: mouse_pos = (io.MousePos[0], io.MousePos[1])
        except: mouse_pos = (io.MousePos.x, io.MousePos.y)

        closest_idx = self.find_closest_edge(mouse_pos[0], mouse_pos[1])
        
        if closest_idx != -1:
            edge = self.raw_edges[closest_idx]
            h_verts = self.raw_vertices[edge]
            h_net = ps.register_curve_network("Highlight", h_verts, np.array([[0, 1]]))
            h_net.set_color((1.0, 0.0, 0.0))
            h_net.set_radius(0.015, relative=False)
            
            if psim.IsMouseClicked(0):
                self.raw_edges = np.delete(self.raw_edges, closest_idx, axis=0)
                ps.remove_curve_network("Highlight")
                self.update_visualization()
        else:
            if ps.has_curve_network("Highlight"): ps.remove_curve_network("Highlight")
            
        if self.debug_mode:
            draw_list = psim.GetWindowDrawList()
            draw_list.AddCircle(mouse_pos, 10, psim.GetColorU32((0, 1, 0, 1)))

if __name__ == "__main__":
    target_file = Path(__file__).resolve().parent / "sample" / "hand.ply"
    if len(sys.argv) > 1:
        target_file = Path(sys.argv[1])
    
    if not os.path.exists(target_file):
        print(f"Error: {target_file} not found.")
    else:
        app = SketchEditor(target_file)
