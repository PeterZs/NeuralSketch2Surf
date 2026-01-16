import trimesh
import numpy as np
import os

def convert_ribbon_to_edge_midpoint_path(input_path, output_path, precision=3):

    try:
        scene = trimesh.load(input_path, force='mesh')
    except Exception as e:
        print(f"  [Error] Loading failed: {e}")
        return

    if isinstance(scene, trimesh.Scene):
        if len(scene.geometry) == 0:
            print("  [Skip] Empty scene.")
            return
        mesh = trimesh.util.concatenate(tuple(scene.geometry.values()))
    else:
        mesh = scene

    mesh.vertices = np.round(mesh.vertices, precision)
    mesh.merge_vertices()
    mesh.remove_duplicate_faces()

    try:
        strokes = mesh.split(only_watertight=False)
    except Exception:
        strokes = [mesh]
    
    all_verts = []
    all_edges = []
    vert_count = 0

    found_any = False

    for stroke in strokes:
        if len(stroke.faces) < 2:
            continue

        adj_edges = stroke.face_adjacency_edges

        if len(adj_edges) == 0:
            continue

        v_coords = stroke.vertices
        edge_midpoints = v_coords[adj_edges].mean(axis=1)

        if len(edge_midpoints) > 1:
            for i in range(len(edge_midpoints) - 1):
                p1 = edge_midpoints[i]
                p2 = edge_midpoints[i+1]
                
                if np.linalg.norm(p1 - p2) < 0.1: 
                    all_verts.append(p1)
                    all_verts.append(p2)
                    all_edges.append([vert_count, vert_count + 1])
                    vert_count += 2
                    found_any = True

    if not found_any:
        print(f"  [Warning] No valid edge midpoints found for {os.path.basename(input_path)}.")
        return

    # 4. Export as PLY
    save_as_ply_edges(np.array(all_verts), np.array(all_edges), output_path)

def save_as_ply_edges(verts, edges, filename):
    with open(filename, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element edge {len(edges)}\n")
        f.write("property int vertex1\nproperty int vertex2\n")
        f.write("end_header\n")
        for v in verts:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")
        for e in edges:
            f.write(f"{e[0]} {e[1]}\n")
    print(f"  [Success] Saved to {os.path.basename(filename)}")

def batch_convert_folder(input_folder, output_folder=None, precision=3):

    # If no output directory is specified, create a 'Convert_result_ply' folder inside the input directory by default
    if output_folder is None:
        output_folder = os.path.join(input_folder, "Convert_result_ply")
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output directory: {output_folder}")

    # Supported file extensions
    valid_extensions = ('.glb', '.gltf', '.obj', '.stl', '.ply')

    files = os.listdir(input_folder)
    count = 0
    
    print(f"Scanning folder: {input_folder}...")

    for filename in files:
        if filename.lower().endswith(valid_extensions):
            input_path = os.path.join(input_folder, filename)
            
            # Construct output filename (e.g., model.glb -> model_convert.ply)
            name_without_ext = os.path.splitext(filename)[0]
            output_filename = f"{name_without_ext}_convert.ply"
            output_path = os.path.join(output_folder, output_filename)

            print(f"Processing ({count+1}): {filename} ...")
            
            try:
                convert_ribbon_to_edge_midpoint_path(input_path, output_path, precision)
            except Exception as e:
                print(f"  [Fatal Error] Failed to process {filename}: {e}")
            
            count += 1

    print(f"\nBatch processing complete. Processed {count} files.")

if __name__ == "__main__":
    
    # Path to your input folder
    INPUT_DIR = "sample/RibbonSculpt"  
    
    # you can set OUTPUT_DIR to None to use default behavior
    OUTPUT_DIR = None 

    if os.path.exists(INPUT_DIR):
        batch_convert_folder(INPUT_DIR, OUTPUT_DIR, precision=3)
    else:
        print(f"Path '{INPUT_DIR}' not found. Using current directory.")
        batch_convert_folder(".", precision=3)