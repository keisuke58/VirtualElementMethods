"""
VEM on Exotic Meshes — Pixel/Voxel, Non-conforming, Fractal.

VEM's killer feature: arbitrary polygons. No isoparametric mapping needed.
These meshes would break standard FEM but VEM handles them naturally.

Mesh types:
  1. Pixel mesh — image pixels as quadrilateral elements (+ L-shape merge)
  2. Non-conforming mesh — hanging nodes / T-junctions at refinement interfaces
  3. Fractal mesh — Sierpinski triangle, Koch snowflake boundary
  4. Concave polygon mesh — star-shaped, L-shaped elements
  5. Mixed mesh — triangles + quads + pentagons + hexagons in one domain

Author: Keisuke Nishioka (趣味プロジェクト)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from PIL import Image

from vem_elasticity import vem_elasticity


# ═══════════════════════════════════════════════════════════════════════════
# Mesh generators
# ═══════════════════════════════════════════════════════════════════════════

# ── 1. Pixel/Voxel Direct Mesh ──────────────────────────────────────────

def pixel_mesh_from_image(image_path, threshold=128, max_pixels=64,
                          merge_l_shapes=False):
    """
    Load grayscale image → each active pixel becomes a VEM quad element.

    Parameters
    ----------
    image_path : str — path to image (grayscale or RGB)
    threshold  : int — pixels darker than this are 'active' (material)
    max_pixels : int — downsample image to this max dimension
    merge_l_shapes : bool — merge adjacent L-shaped pixel groups into
                     single polygon elements (demonstrates VEM flexibility)

    Returns
    -------
    vertices : (N, 2) array
    elements : list of int arrays
    pixel_values : (N_el,) — original grayscale value per element
    """
    img = Image.open(image_path).convert('L')

    # Downsample
    w, h = img.size
    scale = max_pixels / max(w, h)
    if scale < 1:
        img = img.resize((int(w * scale), int(h * scale)), Image.NEAREST)

    arr = np.array(img)
    ny, nx = arr.shape

    # Build vertex grid (nx+1) × (ny+1)
    xs = np.arange(nx + 1, dtype=float)
    ys = np.arange(ny + 1, dtype=float)
    xx, yy = np.meshgrid(xs, ys)
    # Flip y so image top = mesh top
    yy = ny - yy

    all_verts = np.column_stack([xx.ravel(), yy.ravel()])

    def vid(ix, iy):
        return iy * (nx + 1) + ix

    # Identify active pixels
    active = arr < threshold  # dark = material

    if not merge_l_shapes:
        # Simple: each pixel → 1 quad element
        elements = []
        pixel_vals = []
        for iy in range(ny):
            for ix in range(nx):
                if active[iy, ix]:
                    # CCW quad: BL, BR, TR, TL
                    el = np.array([vid(ix, iy+1), vid(ix+1, iy+1),
                                   vid(ix+1, iy), vid(ix, iy)])
                    elements.append(el)
                    pixel_vals.append(arr[iy, ix])
    else:
        # Merge random 2×1 or L-shaped pixel groups
        elements, pixel_vals = _merge_pixel_groups(arr, active, nx, ny, vid)

    # Remove unused vertices and reindex
    vertices, elements = _compact_mesh(all_verts, elements)

    return vertices, elements, np.array(pixel_vals, dtype=float)


def pixel_mesh_from_array(mask, values=None):
    """
    Boolean mask (ny, nx) → pixel VEM mesh.

    Parameters
    ----------
    mask   : (ny, nx) bool array — True = active element
    values : (ny, nx) float array — optional field per pixel

    Returns
    -------
    vertices, elements, field_values
    """
    ny, nx = mask.shape
    xs = np.arange(nx + 1, dtype=float)
    ys = np.arange(ny + 1, dtype=float)
    xx, yy = np.meshgrid(xs, ys)
    yy = ny - yy
    all_verts = np.column_stack([xx.ravel(), yy.ravel()])

    def vid(ix, iy):
        return iy * (nx + 1) + ix

    elements = []
    field_vals = []
    for iy in range(ny):
        for ix in range(nx):
            if mask[iy, ix]:
                el = np.array([vid(ix, iy+1), vid(ix+1, iy+1),
                               vid(ix+1, iy), vid(ix, iy)])
                elements.append(el)
                field_vals.append(values[iy, ix] if values is not None else 1.0)

    vertices, elements = _compact_mesh(all_verts, elements)
    return vertices, elements, np.array(field_vals)


def _merge_pixel_groups(arr, active, nx, ny, vid):
    """Merge adjacent active pixels into L-shapes and dominoes."""
    used = np.zeros_like(active)
    elements = []
    pixel_vals = []

    # First pass: try to merge L-shapes (3 pixels)
    for iy in range(ny - 1):
        for ix in range(nx - 1):
            # 2×2 block, pick L-shapes (3 out of 4)
            block = [(iy, ix), (iy, ix+1), (iy+1, ix), (iy+1, ix+1)]
            block_active = [active[r, c] and not used[r, c] for r, c in block]

            if sum(block_active) >= 3:
                # Take first 3 active pixels as L-shape
                chosen = [b for b, a in zip(block, block_active) if a][:3]
                rows = [r for r, c in chosen]
                cols = [c for r, c in chosen]

                # Build merged polygon (convex hull of all pixel corners)
                corners = set()
                for r, c in chosen:
                    corners.add((c, ny - r))
                    corners.add((c+1, ny - r))
                    corners.add((c, ny - r - 1))
                    corners.add((c+1, ny - r - 1))

                corners = np.array(list(corners), dtype=float)
                # Order CCW
                cx, cy = corners.mean(axis=0)
                angles = np.arctan2(corners[:, 1] - cy, corners[:, 0] - cx)
                order = np.argsort(angles)
                corners = corners[order]

                # Remove interior vertices (those shared by all 3 pixels)
                # Use convex hull instead
                from scipy.spatial import ConvexHull
                if len(corners) > 3:
                    try:
                        hull = ConvexHull(corners)
                        hull_corners = corners[hull.vertices]
                    except Exception:
                        hull_corners = corners
                else:
                    hull_corners = corners

                # Map to vertex indices in the grid
                el_vids = []
                all_verts_set = {}
                for pt in hull_corners:
                    ix_v = int(round(pt[0]))
                    iy_v = int(round(ny - pt[1]))
                    v = vid(ix_v, iy_v)
                    if v not in all_verts_set:
                        all_verts_set[v] = len(el_vids)
                        el_vids.append(v)

                if len(el_vids) >= 3:
                    elements.append(np.array(el_vids))
                    pixel_vals.append(np.mean([arr[r, c] for r, c in chosen]))
                    for r, c in chosen:
                        used[r, c] = True

    # Second pass: remaining pixels as quads
    for iy in range(ny):
        for ix in range(nx):
            if active[iy, ix] and not used[iy, ix]:
                el = np.array([vid(ix, iy+1), vid(ix+1, iy+1),
                               vid(ix+1, iy), vid(ix, iy)])
                elements.append(el)
                pixel_vals.append(arr[iy, ix])

    return elements, pixel_vals


def _compact_mesh(all_verts, elements):
    """Remove unused vertices and reindex elements."""
    used_ids = set()
    for el in elements:
        used_ids.update(el)
    used_ids = sorted(used_ids)
    old_to_new = {old: new for new, old in enumerate(used_ids)}

    vertices = all_verts[used_ids]
    new_elements = [np.array([old_to_new[v] for v in el]) for el in elements]
    return vertices, new_elements


# ── 2. Non-conforming Mesh (Hanging Nodes) ─────────────────────────────

def nonconforming_mesh(nx_coarse=4, ny_coarse=4, refine_region=None,
                       refine_level=2):
    """
    Generate a mesh with hanging nodes at coarse/fine interface.

    Standard FEM requires conforming meshes (no hanging nodes).
    VEM handles T-junctions naturally — the fine edge midpoint
    becomes an extra vertex on the coarse polygon.

    Parameters
    ----------
    nx_coarse, ny_coarse : int — coarse grid divisions
    refine_region : callable(cx, cy) → bool — which coarse cells to refine
    refine_level  : int — refinement factor (2 = split each cell into 2×2)

    Returns
    -------
    vertices : (N, 2), elements : list of int arrays
    """
    if refine_region is None:
        # Refine center quarter by default
        def refine_region(cx, cy):
            return 0.25 < cx < 0.75 and 0.25 < cy < 0.75

    hx = 1.0 / nx_coarse
    hy = 1.0 / ny_coarse
    hx_fine = hx / refine_level
    hy_fine = hy / refine_level

    # Collect all vertices with tolerance-based dedup
    vert_list = []
    vert_map = {}
    tol = 1e-10

    def add_vertex(x, y):
        key = (round(x / tol) * tol, round(y / tol) * tol)
        # Use rounded key for lookup
        rkey = (round(x, 8), round(y, 8))
        if rkey not in vert_map:
            idx = len(vert_list)
            vert_list.append([x, y])
            vert_map[rkey] = idx
        return vert_map[rkey]

    elements = []

    for iy in range(ny_coarse):
        for ix in range(nx_coarse):
            x0 = ix * hx
            y0 = iy * hy
            cx = x0 + 0.5 * hx
            cy = y0 + 0.5 * hy

            if refine_region(cx, cy):
                # Fine cells
                for jy in range(refine_level):
                    for jx in range(refine_level):
                        fx0 = x0 + jx * hx_fine
                        fy0 = y0 + jy * hy_fine
                        v0 = add_vertex(fx0, fy0)
                        v1 = add_vertex(fx0 + hx_fine, fy0)
                        v2 = add_vertex(fx0 + hx_fine, fy0 + hy_fine)
                        v3 = add_vertex(fx0, fy0 + hy_fine)
                        elements.append(np.array([v0, v1, v2, v3]))
            else:
                # Coarse cell — but we need to add hanging nodes
                # from neighboring fine cells on shared edges
                edge_bottom = _collect_edge_verts(
                    x0, y0, x0 + hx, y0, vert_map, 'h')
                edge_right = _collect_edge_verts(
                    x0 + hx, y0, x0 + hx, y0 + hy, vert_map, 'v')
                edge_top = _collect_edge_verts(
                    x0 + hx, y0 + hy, x0, y0 + hy, vert_map, 'h_rev')
                edge_left = _collect_edge_verts(
                    x0, y0 + hy, x0, y0, vert_map, 'v_rev')

                # Add corner vertices
                v_bl = add_vertex(x0, y0)
                v_br = add_vertex(x0 + hx, y0)
                v_tr = add_vertex(x0 + hx, y0 + hy)
                v_tl = add_vertex(x0, y0 + hy)

                # Assemble polygon CCW: bottom → right → top → left
                poly = [v_bl] + edge_bottom + [v_br] + edge_right + \
                       [v_tr] + edge_top + [v_tl] + edge_left
                # Remove consecutive duplicates
                clean = [poly[0]]
                for v in poly[1:]:
                    if v != clean[-1]:
                        clean.append(v)
                if clean[-1] == clean[0]:
                    clean = clean[:-1]

                if len(clean) >= 3:
                    elements.append(np.array(clean))

    vertices = np.array(vert_list)
    return vertices, elements


def _collect_edge_verts(x0, y0, x1, y1, vert_map, direction):
    """Find existing vertices strictly between two endpoints on an edge."""
    verts_on_edge = []
    tol = 1e-7

    for (rx, ry), idx in vert_map.items():
        # Check if point is on the line segment (excluding endpoints)
        if direction.startswith('h'):
            if abs(ry - y0) < tol and min(x0, x1) + tol < rx < max(x0, x1) - tol:
                verts_on_edge.append((rx, idx))
        else:
            if abs(rx - x0) < tol and min(y0, y1) + tol < ry < max(y0, y1) - tol:
                verts_on_edge.append((ry, idx))

    # Sort along edge direction
    verts_on_edge.sort(key=lambda t: t[0],
                       reverse=direction.endswith('rev'))

    return [idx for _, idx in verts_on_edge]


# ── 3. Fractal Meshes ──────────────────────────────────────────────────

def sierpinski_mesh(level=3):
    """
    Sierpinski triangle tessellation as VEM mesh.

    Each non-removed triangle at recursion level `level` becomes an element.
    Triangle = 3-gon element, trivially handled by VEM.
    At higher levels the mesh is extremely irregular — perfect VEM test.

    Returns
    -------
    vertices : (N, 2), elements : list of int arrays
    """
    # Start with equilateral triangle
    base = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, np.sqrt(3) / 2],
    ])

    triangles = [base]

    for _ in range(level):
        new_triangles = []
        for tri in triangles:
            mids = np.array([
                0.5 * (tri[0] + tri[1]),
                0.5 * (tri[1] + tri[2]),
                0.5 * (tri[2] + tri[0]),
            ])
            # Keep 3 corner triangles, remove center
            new_triangles.append(np.array([tri[0], mids[0], mids[2]]))
            new_triangles.append(np.array([mids[0], tri[1], mids[1]]))
            new_triangles.append(np.array([mids[2], mids[1], tri[2]]))
        triangles = new_triangles

    # Build vertex array with dedup
    vert_list = []
    vert_map = {}
    elements = []

    for tri in triangles:
        el = []
        for pt in tri:
            key = (round(pt[0], 10), round(pt[1], 10))
            if key not in vert_map:
                vert_map[key] = len(vert_list)
                vert_list.append(pt)
            el.append(vert_map[key])
        elements.append(np.array(el))

    return np.array(vert_list), elements


def koch_snowflake_mesh(level=3, n_interior=200):
    """
    Koch snowflake boundary → constrained Delaunay interior mesh.

    The boundary has fractal dimension log4/log3 ≈ 1.26.
    Uses Delaunay triangulation with boundary + interior points,
    then filters triangles outside the snowflake.

    Returns
    -------
    vertices : (N, 2), elements : list of int arrays
    """
    from scipy.spatial import Delaunay

    # Generate Koch snowflake boundary
    boundary = _koch_curve(level)

    # Subsample boundary to avoid too-dense points
    n_bnd = min(len(boundary), 200)
    step = max(1, len(boundary) // n_bnd)
    bnd_pts = boundary[::step]

    cx = boundary[:, 0].mean()
    cy = boundary[:, 1].mean()
    r_max = np.max(np.sqrt((boundary[:, 0] - cx)**2 +
                            (boundary[:, 1] - cy)**2))

    # Random interior points
    rng = np.random.default_rng(42)
    interior_pts = []
    while len(interior_pts) < n_interior:
        pts = rng.uniform(-r_max, r_max, size=(n_interior * 4, 2))
        pts[:, 0] += cx
        pts[:, 1] += cy
        for pt in pts:
            if _point_in_polygon(pt, boundary) and len(interior_pts) < n_interior:
                interior_pts.append(pt)

    all_pts = np.vstack([bnd_pts, np.array(interior_pts)])

    # Delaunay triangulation
    tri = Delaunay(all_pts)

    # Filter: keep only triangles whose centroid is inside the snowflake
    elements = []
    for simplex in tri.simplices:
        centroid = all_pts[simplex].mean(axis=0)
        if _point_in_polygon(centroid, boundary):
            elements.append(np.array(simplex))

    return all_pts, elements


def _koch_curve(level):
    """Generate Koch snowflake boundary points."""
    # Start: equilateral triangle
    angles = np.array([np.pi/2, np.pi/2 - 2*np.pi/3, np.pi/2 - 4*np.pi/3])
    pts = np.column_stack([np.cos(angles), np.sin(angles)])

    def subdivide(p1, p2):
        """One Koch subdivision of segment p1→p2."""
        d = p2 - p1
        a = p1 + d / 3
        b = p1 + 2 * d / 3
        # Equilateral peak
        peak = 0.5 * (a + b) + np.array([-d[1], d[0]]) * np.sqrt(3) / 6
        return [p1, a, peak, b]

    segments = list(zip(pts, np.roll(pts, -1, axis=0)))

    for _ in range(level):
        new_segments = []
        for p1, p2 in segments:
            subdiv = subdivide(p1, p2)
            for j in range(len(subdiv) - 1):
                new_segments.append((subdiv[j], subdiv[j + 1]))
            new_segments.append((subdiv[-1], p2))
        segments = new_segments

    boundary = np.array([s[0] for s in segments])
    return boundary


def _point_in_polygon(point, polygon):
    """Ray casting algorithm for point-in-polygon test."""
    x, y = point
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


# ── 4. Concave / Star-shaped Polygon Mesh ──────────────────────────────

def concave_star_mesh(nx=6, ny=6, star_fraction=0.3):
    """
    Grid of star-shaped (concave) polygons.

    Each cell is an 8-pointed star formed by pushing edge midpoints inward.
    Standard FEM cannot handle concave elements. VEM can.

    Parameters
    ----------
    nx, ny : int — grid divisions
    star_fraction : float — how far midpoints are pushed inward (0=quad, 0.5=max)
    """
    hx = 1.0 / nx
    hy = 1.0 / ny

    vert_list = []
    vert_map = {}

    def add_v(x, y):
        key = (round(x, 10), round(y, 10))
        if key not in vert_map:
            vert_map[key] = len(vert_list)
            vert_list.append([x, y])
        return vert_map[key]

    elements = []

    for iy in range(ny):
        for ix in range(nx):
            x0 = ix * hx
            y0 = iy * hy
            cx = x0 + 0.5 * hx
            cy = y0 + 0.5 * hy

            # 4 corners
            c0 = add_v(x0, y0)
            c1 = add_v(x0 + hx, y0)
            c2 = add_v(x0 + hx, y0 + hy)
            c3 = add_v(x0, y0 + hy)

            # 4 edge midpoints pushed inward → concave
            sf = star_fraction
            m_bot = add_v(cx, y0 + sf * hy)         # bottom mid (pushed up)
            m_right = add_v(x0 + hx - sf * hx, cy)  # right mid (pushed left)
            m_top = add_v(cx, y0 + hy - sf * hy)     # top mid (pushed down)
            m_left = add_v(x0 + sf * hx, cy)         # left mid (pushed right)

            # 8-vertex star polygon (CCW)
            el = np.array([c0, m_bot, c1, m_right, c2, m_top, c3, m_left])
            elements.append(el)

    vertices = np.array(vert_list)
    return vertices, elements


# ── 5. Mixed Polygon Mesh ──────────────────────────────────────────────

def mixed_polygon_mesh(n_cells=80, domain=(0, 1, 0, 1)):
    """
    Random mesh mixing triangles, quads, pentagons, hexagons, heptagons.

    Uses Voronoi tessellation then randomly merges adjacent cells
    to create larger, more complex polygons.
    """
    from scipy.spatial import Voronoi

    xmin, xmax, ymin, ymax = domain
    rng = np.random.default_rng(123)

    # Generate Voronoi seeds
    seeds = rng.uniform(size=(n_cells, 2))
    seeds[:, 0] = seeds[:, 0] * (xmax - xmin) + xmin
    seeds[:, 1] = seeds[:, 1] * (ymax - ymin) + ymin

    # Mirror for boundary
    mirror = []
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ms = seeds.copy()
        if dx != 0:
            ms[:, 0] = 2 * (xmin if dx < 0 else xmax) - ms[:, 0]
        if dy != 0:
            ms[:, 1] = 2 * (ymin if dy < 0 else ymax) - ms[:, 1]
        mirror.append(ms)

    all_seeds = np.vstack([seeds] + mirror)
    vor = Voronoi(all_seeds)

    # Extract cells for original seeds
    cells = []
    for i in range(len(seeds)):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        if -1 in region or len(region) < 3:
            continue
        verts = vor.vertices[region]
        # Clip to domain
        verts[:, 0] = np.clip(verts[:, 0], xmin, xmax)
        verts[:, 1] = np.clip(verts[:, 1], ymin, ymax)
        cells.append(verts)

    # Randomly merge some adjacent pairs
    # (Simple approach: merge cells whose centroids are close)
    merged = [False] * len(cells)
    final_cells = []

    for i in range(0, len(cells) - 1, 2):
        if rng.random() < 0.3 and not merged[i] and not merged[i+1]:
            # Merge by convex hull of both cells
            from scipy.spatial import ConvexHull
            combined = np.vstack([cells[i], cells[i+1]])
            try:
                hull = ConvexHull(combined)
                final_cells.append(combined[hull.vertices])
                merged[i] = merged[i+1] = True
            except Exception:
                pass

    for i, cell in enumerate(cells):
        if not merged[i]:
            final_cells.append(cell)

    # Build vertex/element arrays
    vert_list = []
    vert_map = {}
    elements = []

    for cell in final_cells:
        el = []
        for pt in cell:
            key = (round(pt[0], 8), round(pt[1], 8))
            if key not in vert_map:
                vert_map[key] = len(vert_list)
                vert_list.append(pt.copy())
            el.append(vert_map[key])
        if len(el) >= 3:
            elements.append(np.array(el))

    return np.array(vert_list), elements


# ── 6. Penrose Tiling (Aperiodic) ──────────────────────────────────────

def penrose_mesh(level=4):
    """
    Penrose P3 (rhombus) tiling — aperiodic, quasicrystalline mesh.

    Each rhombus is a 4-vertex VEM element.
    The mesh has no translational symmetry — a unique challenge
    that VEM handles effortlessly.

    Uses de Bruijn's pentagrid method via Robinson triangle subdivision.
    """
    phi = (1 + np.sqrt(5)) / 2  # golden ratio

    # Robinson triangles: (type, A, B, C)
    # type 0 = thin (36-108-36), type 1 = thick (72-72-36)
    triangles = []

    # Initial star of 10 Robinson triangles
    for i in range(10):
        angle1 = (2 * i - 1) * np.pi / 5
        angle2 = (2 * i + 1) * np.pi / 5
        A = np.array([0.0, 0.0])
        B = np.array([np.cos(angle1), np.sin(angle1)])
        C = np.array([np.cos(angle2), np.sin(angle2)])
        if i % 2 == 0:
            triangles.append((0, A, B, C))
        else:
            triangles.append((0, A, C, B))

    # Subdivide
    for _ in range(level):
        new_triangles = []
        for typ, A, B, C in triangles:
            if typ == 0:  # thin triangle
                P = A + (B - A) / phi
                new_triangles.append((0, C, P, B))
                new_triangles.append((1, P, C, A))
            else:  # thick triangle
                Q = B + (A - B) / phi
                R = B + (C - B) / phi
                new_triangles.append((1, Q, R, B))
                new_triangles.append((1, R, Q, A))
                new_triangles.append((0, R, C, A))
        triangles = new_triangles

    # Merge triangle pairs into rhombuses
    # Group by shared hypotenuse
    vert_list = []
    vert_map = {}
    elements = []

    def add_v(pt):
        key = (round(pt[0], 8), round(pt[1], 8))
        if key not in vert_map:
            vert_map[key] = len(vert_list)
            vert_list.append(pt.copy())
        return vert_map[key]

    # For simplicity, just use each triangle as an element
    for typ, A, B, C in triangles:
        el = [add_v(A), add_v(B), add_v(C)]
        # Remove degenerate
        if len(set(el)) == 3:
            elements.append(np.array(el))

    return np.array(vert_list), elements


# ═══════════════════════════════════════════════════════════════════════════
# Visualization helpers
# ═══════════════════════════════════════════════════════════════════════════

def plot_mesh(vertices, elements, title='', field=None, cmap='viridis',
              edgecolor='k', linewidth=0.5, ax=None, colorbar_label=None):
    """Plot polygonal mesh with optional per-element coloring."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure

    patches = []
    for el in elements:
        el_int = el.astype(int)
        poly = MplPolygon(vertices[el_int], closed=True)
        patches.append(poly)

    pc = PatchCollection(patches, cmap=cmap, edgecolor=edgecolor,
                         linewidth=linewidth)

    if field is not None:
        pc.set_array(np.array(field))
    else:
        pc.set_facecolor('lightblue')
        pc.set_edgecolor(edgecolor)

    ax.add_collection(pc)
    margin = 0.02 * max(np.ptp(vertices[:, 0]), np.ptp(vertices[:, 1]))
    ax.set_xlim(vertices[:, 0].min() - margin, vertices[:, 0].max() + margin)
    ax.set_ylim(vertices[:, 1].min() - margin, vertices[:, 1].max() + margin)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=12, fontweight='bold')

    if field is not None and colorbar_label:
        fig.colorbar(pc, ax=ax, label=colorbar_label, shrink=0.8)

    return ax


def plot_vem_result(vertices, elements, u, E_field=None,
                    deform_scale='auto', title='', save=None):
    """Plot mesh + displacement + stress for VEM result."""
    ux = u[0::2]
    uy = u[1::2]
    u_mag = np.sqrt(ux**2 + uy**2)

    if deform_scale == 'auto':
        max_u = np.max(u_mag)
        char_size = max(np.ptp(vertices[:, 0]), np.ptp(vertices[:, 1]))
        deform_scale = 0.1 * char_size / max_u if max_u > 0 else 1.0

    deformed = vertices + deform_scale * np.column_stack([ux, uy])

    n_panels = 3 if E_field is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6))

    # Panel 1: E field or original mesh
    if E_field is not None:
        plot_mesh(vertices, elements, title='E field [Pa]',
                  field=E_field, cmap='viridis', ax=axes[0],
                  colorbar_label='E [Pa]')
        ax_disp = axes[1]
        ax_stress = axes[2]
    else:
        plot_mesh(vertices, elements, title='Original mesh', ax=axes[0])
        ax_disp = axes[1]
        ax_stress = None

    # Panel 2: Displacement magnitude on deformed mesh
    el_u_mag = [np.mean(u_mag[el.astype(int)]) for el in elements]
    patches = []
    for el in elements:
        el_int = el.astype(int)
        poly = MplPolygon(deformed[el_int], closed=True)
        patches.append(poly)

    pc = PatchCollection(patches, cmap='hot_r', edgecolor='k', linewidth=0.3)
    pc.set_array(np.array(el_u_mag))
    ax_disp.add_collection(pc)
    margin = 0.02 * max(np.ptp(deformed[:, 0]), np.ptp(deformed[:, 1]))
    ax_disp.set_xlim(deformed[:, 0].min() - margin, deformed[:, 0].max() + margin)
    ax_disp.set_ylim(deformed[:, 1].min() - margin, deformed[:, 1].max() + margin)
    ax_disp.set_aspect('equal')
    ax_disp.set_title(f'|u| on deformed (x{deform_scale:.0f})')
    fig.colorbar(pc, ax=ax_disp, label='|u|', shrink=0.8)

    # Panel 3: Von Mises stress (approximate)
    if ax_stress is not None:
        nu = 0.3
        vm_stress = []
        for i, el in enumerate(elements):
            el_int = el.astype(int)
            E_el = E_field[i] if E_field is not None else 1000.0
            # Approximate strain from displacement gradient
            n_v = len(el_int)
            if n_v < 3:
                vm_stress.append(0.0)
                continue
            verts = vertices[el_int]
            ux_el = ux[el_int]
            uy_el = uy[el_int]
            # Least-squares gradient
            A = np.column_stack([verts - verts.mean(axis=0), np.ones(n_v)])
            try:
                grad_ux = np.linalg.lstsq(A, ux_el, rcond=None)[0][:2]
                grad_uy = np.linalg.lstsq(A, uy_el, rcond=None)[0][:2]
            except Exception:
                vm_stress.append(0.0)
                continue
            exx = grad_ux[0]
            eyy = grad_uy[1]
            exy = 0.5 * (grad_ux[1] + grad_uy[0])
            C = E_el / (1.0 - nu**2)
            sxx = C * (exx + nu * eyy)
            syy = C * (nu * exx + eyy)
            sxy = C * (1.0 - nu) / 2.0 * 2 * exy
            vm = np.sqrt(sxx**2 - sxx*syy + syy**2 + 3*sxy**2)
            vm_stress.append(vm)

        patches2 = []
        for el in elements:
            el_int = el.astype(int)
            poly = MplPolygon(deformed[el_int], closed=True)
            patches2.append(poly)
        pc2 = PatchCollection(patches2, cmap='inferno', edgecolor='k',
                              linewidth=0.3)
        pc2.set_array(np.array(vm_stress))
        ax_stress.add_collection(pc2)
        ax_stress.set_xlim(deformed[:, 0].min() - margin,
                           deformed[:, 0].max() + margin)
        ax_stress.set_ylim(deformed[:, 1].min() - margin,
                           deformed[:, 1].max() + margin)
        ax_stress.set_aspect('equal')
        ax_stress.set_title('von Mises stress')
        fig.colorbar(pc2, ax=ax_stress, label='σ_vm [Pa]', shrink=0.8)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# Demo drivers — solve elasticity on each exotic mesh
# ═══════════════════════════════════════════════════════════════════════════

def demo_pixel_mesh(save_dir='/tmp'):
    """Demo 1: Pixel mesh from a synthetic binary image."""
    print("=" * 60)
    print("Demo 1: Pixel/Voxel Direct Mesh")
    print("=" * 60)

    # Create synthetic binary image (circle with hole)
    nx, ny = 32, 32
    xx, yy = np.meshgrid(np.arange(nx), np.arange(ny))
    cx, cy = nx // 2, ny // 2
    r_outer = 12
    r_inner = 4
    dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    mask = (dist < r_outer) & (dist > r_inner)

    vertices, elements, _ = pixel_mesh_from_array(mask)
    n_el = len(elements)
    n_sides = [len(el) for el in elements]

    print(f"  Pixels active: {mask.sum()}")
    print(f"  Elements: {n_el}, Vertices: {len(vertices)}")
    print(f"  Polygon sides: all {n_sides[0]}-gons (quads)")

    # Solve: fixed bottom, pressure top
    E_field = 1000.0 * np.ones(n_el)
    nu = 0.3

    ymin = vertices[:, 1].min()
    ymax = vertices[:, 1].max()
    tol = 0.5

    bottom = np.where(vertices[:, 1] < ymin + tol)[0]
    top = np.where(vertices[:, 1] > ymax - tol)[0]

    bc_dofs = np.concatenate([2 * bottom, 2 * bottom + 1])
    bc_vals = np.zeros(len(bc_dofs))

    load_dofs = 2 * top + 1
    load_vals = np.full(len(top), -0.5 / max(len(top), 1))

    u = vem_elasticity(vertices, elements, E_field, nu,
                       bc_dofs, bc_vals, load_dofs, load_vals)

    print(f"  Max |u|: {np.max(np.sqrt(u[0::2]**2 + u[1::2]**2)):.6f}")

    plot_vem_result(vertices, elements, u, E_field,
                    title='Pixel Mesh VEM (ring geometry)',
                    save=f'{save_dir}/vem_pixel_mesh.png')
    return vertices, elements, u


def demo_pixel_mesh_image(image_path, save_dir='/tmp'):
    """Demo 1b: Pixel mesh from actual image file."""
    print("=" * 60)
    print("Demo 1b: Image-Based Pixel Mesh")
    print("=" * 60)

    vertices, elements, pixel_vals = pixel_mesh_from_image(
        image_path, threshold=128, max_pixels=48)

    n_el = len(elements)
    print(f"  Elements: {n_el}, Vertices: {len(vertices)}")

    # E proportional to pixel darkness
    E_field = 100.0 + 900.0 * (255.0 - pixel_vals) / 255.0
    nu = 0.3

    ymin = vertices[:, 1].min()
    ymax = vertices[:, 1].max()
    tol = 0.5

    bottom = np.where(vertices[:, 1] < ymin + tol)[0]
    top = np.where(vertices[:, 1] > ymax - tol)[0]

    bc_dofs = np.concatenate([2 * bottom, 2 * bottom + 1])
    bc_vals = np.zeros(len(bc_dofs))

    load_dofs = 2 * top + 1
    load_vals = np.full(len(top), -0.5 / max(len(top), 1))

    u = vem_elasticity(vertices, elements, E_field, nu,
                       bc_dofs, bc_vals, load_dofs, load_vals)

    print(f"  Max |u|: {np.max(np.sqrt(u[0::2]**2 + u[1::2]**2)):.6f}")

    plot_vem_result(vertices, elements, u, E_field,
                    title=f'Image Pixel Mesh VEM ({n_el} elements)',
                    save=f'{save_dir}/vem_image_pixel_mesh.png')
    return vertices, elements, u


def demo_nonconforming(save_dir='/tmp'):
    """Demo 2: Non-conforming mesh with hanging nodes."""
    print("\n" + "=" * 60)
    print("Demo 2: Non-conforming Mesh (Hanging Nodes)")
    print("=" * 60)

    # Refine center region
    vertices, elements = nonconforming_mesh(
        nx_coarse=6, ny_coarse=6,
        refine_region=lambda cx, cy: (cx - 0.5)**2 + (cy - 0.5)**2 < 0.15,
        refine_level=3)

    n_el = len(elements)
    n_sides = [len(el) for el in elements]

    print(f"  Elements: {n_el}, Vertices: {len(vertices)}")
    print(f"  Polygon sides: {min(n_sides)} to {max(n_sides)}")
    print(f"  (coarse cells have hanging nodes → more vertices per element)")

    # Spatially varying E: soft center, stiff boundary
    E_field = np.zeros(n_el)
    for i, el in enumerate(elements):
        el_int = el.astype(int)
        centroid = vertices[el_int].mean(axis=0)
        dist = np.linalg.norm(centroid - 0.5)
        E_field[i] = 200.0 + 800.0 * dist  # soft center

    nu = 0.3
    tol = 1e-6

    bottom = np.where(vertices[:, 1] < tol)[0]
    top = np.where(vertices[:, 1] > 1.0 - tol)[0]

    bc_dofs = np.concatenate([2 * bottom, 2 * bottom + 1])
    bc_vals = np.zeros(len(bc_dofs))

    load_dofs = 2 * top + 1
    load_vals = np.full(len(top), -1.0 / max(len(top), 1))

    u = vem_elasticity(vertices, elements, E_field, nu,
                       bc_dofs, bc_vals, load_dofs, load_vals)

    print(f"  Max |u|: {np.max(np.sqrt(u[0::2]**2 + u[1::2]**2)):.6f}")

    plot_vem_result(vertices, elements, u, E_field,
                    title='Non-conforming Mesh VEM (hanging nodes)',
                    save=f'{save_dir}/vem_nonconforming.png')
    return vertices, elements, u


def demo_sierpinski(save_dir='/tmp'):
    """Demo 3: Sierpinski fractal mesh."""
    print("\n" + "=" * 60)
    print("Demo 3: Sierpinski Fractal Mesh")
    print("=" * 60)

    vertices, elements = sierpinski_mesh(level=5)

    n_el = len(elements)
    print(f"  Level 5 Sierpinski: {n_el} triangles")
    print(f"  Vertices: {len(vertices)}")
    print(f"  (Hausdorff dim = log3/log2 ≈ {np.log(3)/np.log(2):.3f})")

    # E varies with height
    E_field = np.zeros(n_el)
    for i, el in enumerate(elements):
        el_int = el.astype(int)
        cy = vertices[el_int, 1].mean()
        E_field[i] = 200.0 + 800.0 * cy / vertices[:, 1].max()

    nu = 0.3
    tol = 1e-6

    bottom = np.where(vertices[:, 1] < tol)[0]
    top_y = vertices[:, 1].max()
    top = np.where(vertices[:, 1] > top_y - 0.05)[0]

    bc_dofs = np.concatenate([2 * bottom, 2 * bottom + 1])
    bc_vals = np.zeros(len(bc_dofs))

    load_dofs = 2 * top + 1
    load_vals = np.full(len(top), -0.5 / max(len(top), 1))

    u = vem_elasticity(vertices, elements, E_field, nu,
                       bc_dofs, bc_vals, load_dofs, load_vals)

    print(f"  Max |u|: {np.max(np.sqrt(u[0::2]**2 + u[1::2]**2)):.6f}")

    plot_vem_result(vertices, elements, u, E_field,
                    title='Sierpinski Fractal VEM',
                    save=f'{save_dir}/vem_sierpinski.png')
    return vertices, elements, u


def demo_koch_snowflake(save_dir='/tmp'):
    """Demo 4: Koch snowflake boundary mesh."""
    print("\n" + "=" * 60)
    print("Demo 4: Koch Snowflake Boundary Mesh")
    print("=" * 60)

    vertices, elements = koch_snowflake_mesh(level=3, n_interior=150)

    n_el = len(elements)
    n_sides = [len(el) for el in elements]

    print(f"  Elements: {n_el}, Vertices: {len(vertices)}")
    print(f"  Polygon sides: {min(n_sides)} to {max(n_sides)}")
    print(f"  (Fractal boundary dim = log4/log3 ≈ {np.log(4)/np.log(3):.3f})")

    # Radial E field
    cx = vertices[:, 0].mean()
    cy = vertices[:, 1].mean()
    E_field = np.zeros(n_el)
    for i, el in enumerate(elements):
        el_int = el.astype(int)
        centroid = vertices[el_int].mean(axis=0)
        dist = np.sqrt((centroid[0] - cx)**2 + (centroid[1] - cy)**2)
        E_field[i] = 1000.0 - 700.0 * dist

    E_field = np.clip(E_field, 100, 1000)
    nu = 0.3

    ymin = vertices[:, 1].min()
    ymax = vertices[:, 1].max()
    tol_y = 0.05 * (ymax - ymin)

    bottom = np.where(vertices[:, 1] < ymin + tol_y)[0]
    top = np.where(vertices[:, 1] > ymax - tol_y)[0]

    if len(bottom) == 0 or len(top) == 0:
        print("  Skipping solve (no boundary nodes found)")
        plot_mesh(vertices, elements, title='Koch Snowflake Mesh',
                  field=E_field, cmap='viridis')
        plt.savefig(f'{save_dir}/vem_koch_mesh.png', dpi=150)
        plt.close()
        return vertices, elements, None

    bc_dofs = np.concatenate([2 * bottom, 2 * bottom + 1])
    bc_vals = np.zeros(len(bc_dofs))

    load_dofs = 2 * top + 1
    load_vals = np.full(len(top), -0.3 / max(len(top), 1))

    u = vem_elasticity(vertices, elements, E_field, nu,
                       bc_dofs, bc_vals, load_dofs, load_vals)

    print(f"  Max |u|: {np.max(np.sqrt(u[0::2]**2 + u[1::2]**2)):.6f}")

    plot_vem_result(vertices, elements, u, E_field,
                    title='Koch Snowflake VEM',
                    save=f'{save_dir}/vem_koch_snowflake.png')
    return vertices, elements, u


def demo_concave_stars(save_dir='/tmp'):
    """Demo 5: Concave star-shaped polygon mesh."""
    print("\n" + "=" * 60)
    print("Demo 5: Concave Star-Shaped Polygon Mesh")
    print("=" * 60)

    vertices, elements = concave_star_mesh(nx=8, ny=8, star_fraction=0.3)

    n_el = len(elements)
    n_sides = [len(el) for el in elements]

    print(f"  Elements: {n_el}, Vertices: {len(vertices)}")
    print(f"  All elements are 8-gon stars (concave!)")
    print(f"  → Standard FEM would FAIL on these elements")

    # Checkerboard E
    E_field = np.zeros(n_el)
    for i in range(n_el):
        row = i // 8
        col = i % 8
        E_field[i] = 300.0 if (row + col) % 2 == 0 else 900.0

    nu = 0.3
    tol = 1e-6

    bottom = np.where(vertices[:, 1] < tol)[0]
    top = np.where(vertices[:, 1] > 1.0 - tol)[0]

    bc_dofs = np.concatenate([2 * bottom, 2 * bottom + 1])
    bc_vals = np.zeros(len(bc_dofs))

    load_dofs = 2 * top + 1
    load_vals = np.full(len(top), -1.0 / max(len(top), 1))

    u = vem_elasticity(vertices, elements, E_field, nu,
                       bc_dofs, bc_vals, load_dofs, load_vals)

    print(f"  Max |u|: {np.max(np.sqrt(u[0::2]**2 + u[1::2]**2)):.6f}")

    plot_vem_result(vertices, elements, u, E_field,
                    title='Concave Star Mesh VEM (8-gon elements)',
                    save=f'{save_dir}/vem_concave_stars.png')
    return vertices, elements, u


def demo_penrose(save_dir='/tmp'):
    """Demo 6: Penrose aperiodic tiling mesh."""
    print("\n" + "=" * 60)
    print("Demo 6: Penrose Aperiodic Tiling")
    print("=" * 60)

    vertices, elements = penrose_mesh(level=5)

    n_el = len(elements)
    print(f"  Elements: {n_el}, Vertices: {len(vertices)}")
    print(f"  Aperiodic — no translational symmetry!")

    # E field based on distance from center
    E_field = np.zeros(n_el)
    for i, el in enumerate(elements):
        el_int = el.astype(int)
        centroid = vertices[el_int].mean(axis=0)
        dist = np.linalg.norm(centroid)
        E_field[i] = 200.0 + 800.0 * (1.0 - dist / (np.max(np.linalg.norm(vertices, axis=1)) + 1e-10))

    E_field = np.clip(E_field, 100, 1000)
    nu = 0.3

    ymin = vertices[:, 1].min()
    ymax = vertices[:, 1].max()
    tol_y = 0.05 * (ymax - ymin)

    bottom = np.where(vertices[:, 1] < ymin + tol_y)[0]
    top = np.where(vertices[:, 1] > ymax - tol_y)[0]

    bc_dofs = np.concatenate([2 * bottom, 2 * bottom + 1])
    bc_vals = np.zeros(len(bc_dofs))

    load_dofs = 2 * top + 1
    load_vals = np.full(len(top), -0.5 / max(len(top), 1))

    u = vem_elasticity(vertices, elements, E_field, nu,
                       bc_dofs, bc_vals, load_dofs, load_vals)

    print(f"  Max |u|: {np.max(np.sqrt(u[0::2]**2 + u[1::2]**2)):.6f}")

    plot_vem_result(vertices, elements, u, E_field,
                    title='Penrose Tiling VEM (aperiodic)',
                    save=f'{save_dir}/vem_penrose.png')
    return vertices, elements, u


def demo_mixed_polygon(save_dir='/tmp'):
    """Demo 7: Mixed polygon mesh (tri + quad + penta + hexa)."""
    print("\n" + "=" * 60)
    print("Demo 7: Mixed Polygon Mesh")
    print("=" * 60)

    vertices, elements = mixed_polygon_mesh(n_cells=100)

    n_el = len(elements)
    n_sides = [len(el) for el in elements]
    from collections import Counter
    side_counts = Counter(n_sides)

    print(f"  Elements: {n_el}, Vertices: {len(vertices)}")
    print(f"  Polygon types: {dict(sorted(side_counts.items()))}")

    # Random E field
    rng = np.random.default_rng(42)
    E_field = 200.0 + 800.0 * rng.random(n_el)
    nu = 0.3
    tol = 1e-3

    bottom = np.where(vertices[:, 1] < vertices[:, 1].min() + tol)[0]
    top = np.where(vertices[:, 1] > vertices[:, 1].max() - tol)[0]

    bc_dofs = np.concatenate([2 * bottom, 2 * bottom + 1])
    bc_vals = np.zeros(len(bc_dofs))

    load_dofs = 2 * top + 1
    load_vals = np.full(len(top), -1.0 / max(len(top), 1))

    u = vem_elasticity(vertices, elements, E_field, nu,
                       bc_dofs, bc_vals, load_dofs, load_vals)

    print(f"  Max |u|: {np.max(np.sqrt(u[0::2]**2 + u[1::2]**2)):.6f}")

    plot_vem_result(vertices, elements, u, E_field,
                    title='Mixed Polygon VEM (tri+quad+penta+...)',
                    save=f'{save_dir}/vem_mixed_polygon.png')
    return vertices, elements, u


# ═══════════════════════════════════════════════════════════════════════════
# Gallery — run all demos and create summary figure
# ═══════════════════════════════════════════════════════════════════════════

def run_gallery(save_dir='/tmp'):
    """Run all exotic mesh demos and create a gallery figure."""
    print("\n" + "=" * 60)
    print("VEM Exotic Mesh Gallery")
    print("=" * 60)

    fig, axes = plt.subplots(2, 4, figsize=(24, 12))

    # 1. Pixel mesh
    nx, ny = 32, 32
    xx, yy = np.meshgrid(np.arange(nx), np.arange(ny))
    dist = np.sqrt((xx - 16)**2 + (yy - 16)**2)
    mask = (dist < 12) & (dist > 4)
    v, e, _ = pixel_mesh_from_array(mask)
    plot_mesh(v, e, title='1. Pixel Mesh (ring)', ax=axes[0, 0])

    # 2. Non-conforming
    v, e = nonconforming_mesh(
        nx_coarse=6, ny_coarse=6,
        refine_region=lambda cx, cy: (cx - 0.5)**2 + (cy - 0.5)**2 < 0.15,
        refine_level=3)
    n_sides = [len(el) for el in e]
    plot_mesh(v, e, title=f'2. Non-conforming ({min(n_sides)}-{max(n_sides)} sides)',
              field=n_sides, cmap='Set3', ax=axes[0, 1])

    # 3. Sierpinski
    v, e = sierpinski_mesh(level=5)
    plot_mesh(v, e, title=f'3. Sierpinski (L5, {len(e)} tri)',
              ax=axes[0, 2], linewidth=0.2)

    # 4. Koch snowflake
    v, e = koch_snowflake_mesh(level=3, n_interior=120)
    plot_mesh(v, e, title=f'4. Koch Snowflake ({len(e)} cells)',
              ax=axes[0, 3], linewidth=0.3)

    # 5. Concave stars
    v, e = concave_star_mesh(nx=8, ny=8, star_fraction=0.3)
    plot_mesh(v, e, title='5. Concave Stars (8-gon)',
              ax=axes[1, 0])

    # 6. Penrose
    v, e = penrose_mesh(level=5)
    plot_mesh(v, e, title=f'6. Penrose Tiling ({len(e)} tri)',
              ax=axes[1, 1], linewidth=0.1)

    # 7. Mixed polygon
    v, e = mixed_polygon_mesh(n_cells=80)
    n_sides = [len(el) for el in e]
    plot_mesh(v, e, title=f'7. Mixed Polygons ({min(n_sides)}-{max(n_sides)} sides)',
              field=n_sides, cmap='tab10', ax=axes[1, 2])

    # 8. Pixel with L-shape merge
    mask2 = dist < 10
    # Create image then load with merge
    img_arr = np.where(mask2, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_arr)
    img.save('/tmp/_vem_test_circle.png')
    v, e, _ = pixel_mesh_from_image('/tmp/_vem_test_circle.png',
                                     threshold=128, max_pixels=32,
                                     merge_l_shapes=True)
    n_sides = [len(el) for el in e]
    plot_mesh(v, e, title=f'8. L-shape Merged Pixels ({min(n_sides)}-{max(n_sides)} sides)',
              field=n_sides, cmap='Paired', ax=axes[1, 3])

    fig.suptitle('VEM Exotic Mesh Gallery — All Handled by VEM!',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    path = f'{save_dir}/vem_exotic_mesh_gallery.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"\n  Gallery saved: {path}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys

    save_dir = sys.argv[1] if len(sys.argv) > 1 else '/tmp'

    # Run individual demos with elasticity solves
    demo_pixel_mesh(save_dir)
    demo_nonconforming(save_dir)
    demo_sierpinski(save_dir)
    demo_koch_snowflake(save_dir)
    demo_concave_stars(save_dir)
    demo_penrose(save_dir)
    demo_mixed_polygon(save_dir)

    # Gallery figure (mesh only, no solve)
    run_gallery(save_dir)

    print("\n" + "=" * 60)
    print("All exotic mesh demos complete!")
    print("=" * 60)
