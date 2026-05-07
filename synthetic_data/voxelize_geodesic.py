"""Voxelize synthetic geodesic sketches into sparse occupancy grids."""
import os
from pathlib import Path
from typing import List, Tuple
import numpy as np


def bresenham3d(p0, p1):
    """Rasterize a 3D segment into voxel coordinates."""
    x1, y1, z1 = map(int, p0)
    x2, y2, z2 = map(int, p1)
    dx, dy, dz = abs(x2 - x1), abs(y2 - y1), abs(z2 - z1)
    xs = 1 if x2 > x1 else -1
    ys = 1 if y2 > y1 else -1
    zs = 1 if z2 > z1 else -1
    pts = []
    if dx >= dy and dx >= dz:
        p1_err = 2 * dy - dx
        p2_err = 2 * dz - dx
        while True:
            pts.append((x1, y1, z1))
            if x1 == x2:
                break
            x1 += xs
            if p1_err > 0:
                y1 += ys
                p1_err -= 2 * dx
            if p2_err > 0:
                z1 += zs
                p2_err -= 2 * dx
            p1_err += 2 * dy
            p2_err += 2 * dz
    elif dy >= dx and dy >= dz:
        p1_err = 2 * dx - dy
        p2_err = 2 * dz - dy
        while True:
            pts.append((x1, y1, z1))
            if y1 == y2:
                break
            y1 += ys
            if p1_err > 0:
                x1 += xs
                p1_err -= 2 * dy
            if p2_err > 0:
                z1 += zs
                p2_err -= 2 * dy
            p1_err += 2 * dx
            p2_err += 2 * dz
    else:
        p1_err = 2 * dy - dz
        p2_err = 2 * dx - dz
        while True:
            pts.append((x1, y1, z1))
            if z1 == z2:
                break
            z1 += zs
            if p1_err > 0:
                y1 += ys
                p1_err -= 2 * dz
            if p2_err > 0:
                x1 += xs
                p2_err -= 2 * dz
            p1_err += 2 * dy
            p2_err += 2 * dx
    return pts


def _parse_index(tok: str) -> int:
    """Parse an absolute OBJ vertex index."""
    if tok.startswith('-'):
        raise ValueError("OBJ uses relative (negative) indices, which are not supported in the current implementation.")
    return int(tok.split('/')[0]) - 1


def parse_geodesic_obj(obj_path: str) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Read curve vertices and unique edges from an OBJ sketch file."""
    verts, edges = [], []
    with open(obj_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            tag = parts[0]
            if tag == 'v':
                x, y, z = map(float, parts[1:4])
                verts.append([x, y, z])
            elif tag == 'l':
                idxs = [_parse_index(t) for t in parts[1:]]
                edges += list(zip(idxs[:-1], idxs[1:]))
            elif tag == 'f':
                idxs = [_parse_index(t) for t in parts[1:]]
                edges += list(zip(idxs, idxs[1:] + [idxs[0]]))
    if not verts:
        raise AssertionError("OBJ did not contain any vertices (v).")
    edge_set = set()
    for a, b in edges:
        if a == b:
            continue
        edge_set.add(tuple(sorted((a, b))))
    return np.asarray(verts, dtype=np.float64), list(edge_set)


def voxelize_obj_geodesic(obj_path: str, meta_path: str, output_path: str) -> None:
    """Voxelize sketch curves using the label volume metadata."""
    m = np.load(meta_path)
    center = m["center"].astype(np.float64)
    max_extent_scalar = float(np.max(m["max_extent"]))
    R = int(m["resolution"])
    voxel_size = float(m["voxel_size"])

    verts_world, edges = parse_geodesic_obj(obj_path)
    if not len(edges):
        raise AssertionError("OBJ did not contain any edges to voxelize.")

    target_radius = 1.0
    scale = target_radius / (max_extent_scalar / 2.0)
    verts_norm = (verts_world - center) * scale

    real_origin = m["o3d_origin"] 

    def to_index(p_norm: np.ndarray) -> np.ndarray:
        return np.floor((p_norm - real_origin) / voxel_size + 1e-6).astype(np.int32)

    gi = np.clip(to_index(verts_norm), 0, R - 1)

    vox = np.zeros((R, R, R), dtype=np.uint8)
    for i, j in edges:
        for x, y, z in bresenham3d(gi[i], gi[j]):
            if 0 <= x < R and 0 <= y < R and 0 <= z < R:
                vox[x, y, z] = 1

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, vox)
    print(f"[Done] {Path(obj_path).name} -> {Path(output_path).name}, Voxels={vox.sum()}")
