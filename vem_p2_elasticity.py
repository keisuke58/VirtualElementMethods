"""
Second-order (P2) VEM for 2D Linear Elasticity on Polygonal Meshes.

Extension of vem_elasticity.py (P1, lowest-order) to P2 (second-order).
Key change: vertex-only DOFs -> vertex + edge midpoint DOFs.

P1 (vem_elasticity.py):
  - DOFs: 2 per vertex node (u_x, u_y)
  - Polynomial basis: 6 functions (3 rigid body + 3 strain modes)
  - Accuracy: O(h) for H1, O(h^2) for L2

P2 (this file):
  - DOFs: 2 per vertex + 2 per edge midpoint = 4*n_v per element
  - Polynomial basis: 12 functions (dim P2^2 in 2D = 2 x 6)
  - Accuracy: O(h^2) for H1, O(h^3) for L2

P2 scalar monomials (in scaled coordinates xhat, yhat):
  {1, xhat, yhat, xhat^2, xhat*yhat, yhat^2}  (6 total)

Vector P2 basis (12 total):
  (m_i, 0) for i=1..6  and  (0, m_i) for i=1..6
  Split: 3 rigid body + 9 strain modes.

Boundary integrals use Simpson's rule (vertex + midpoint) instead of trapezoidal.

References:
  - Artioli, Beirao da Veiga, Lovadina, Sacco (2017) "Arbitrary order 2D virtual
    elements for polygonal meshes: Part I and II" (arXiv:1701.04306)
  - Beirao da Veiga et al. (2013) "Basic principles of VEM"
  - Gain, Talischi, Paulino (2014) "VEM for 3D elasticity"
  - Sutton (2017) "The VEM in 50 lines of MATLAB"
"""

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from scipy.spatial import Voronoi


# ── Mesh Utilities ────────────────────────────────────────────────────────

def generate_voronoi_mesh(n_cells, domain=(0, 1, 0, 1), seed=42):
    """
    Generate a clipped Voronoi mesh on a rectangular domain.

    Parameters
    ----------
    n_cells : int — approximate number of cells
    domain  : (xmin, xmax, ymin, ymax)
    seed    : int — random seed

    Returns
    -------
    vertices : (N, 2) array
    elements : list of int arrays (0-based connectivity, vertex-only)
    boundary : array of boundary node indices
    """
    rng = np.random.RandomState(seed)
    xmin, xmax, ymin, ymax = domain
    Lx, Ly = xmax - xmin, ymax - ymin

    # Generate seed points inside domain
    pts = rng.rand(n_cells, 2) * [Lx, Ly] + [xmin, ymin]

    # Mirror for bounded Voronoi
    pts_mirror = np.vstack([
        pts,
        np.column_stack([2 * xmin - pts[:, 0], pts[:, 1]]),
        np.column_stack([2 * xmax - pts[:, 0], pts[:, 1]]),
        np.column_stack([pts[:, 0], 2 * ymin - pts[:, 1]]),
        np.column_stack([pts[:, 0], 2 * ymax - pts[:, 1]]),
    ])

    vor = Voronoi(pts_mirror)

    # Clip to domain
    tol = 1e-12
    vertices_list = []
    vert_map = {}
    elements = []

    def add_vertex(v):
        key = (round(v[0], 10), round(v[1], 10))
        if key not in vert_map:
            vert_map[key] = len(vertices_list)
            vertices_list.append(v.copy())
        return vert_map[key]

    for i in range(n_cells):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        if -1 in region or len(region) < 3:
            continue

        poly = vor.vertices[region]

        # Clip to domain
        clipped = _clip_polygon_to_box(poly, xmin, xmax, ymin, ymax)
        if clipped is None or len(clipped) < 3:
            continue

        # Order CCW
        cx = clipped[:, 0].mean()
        cy = clipped[:, 1].mean()
        angles = np.arctan2(clipped[:, 1] - cy, clipped[:, 0] - cx)
        order = np.argsort(angles)
        clipped = clipped[order]

        el_ids = []
        for v in clipped:
            el_ids.append(add_vertex(v))
        elements.append(np.array(el_ids, dtype=int))

    vertices = np.array(vertices_list)

    # Boundary nodes
    boundary = []
    for i, v in enumerate(vertices):
        if (abs(v[0] - xmin) < tol or abs(v[0] - xmax) < tol or
                abs(v[1] - ymin) < tol or abs(v[1] - ymax) < tol):
            boundary.append(i)
    boundary = np.array(boundary, dtype=int)

    return vertices, elements, boundary


def _clip_polygon_to_box(poly, xmin, xmax, ymin, ymax):
    """Sutherland-Hodgman polygon clipping to axis-aligned box."""
    output = list(poly)
    edges = [
        (lambda p: p[0] - xmin, lambda p, q: _intersect_edge(p, q, 0, xmin)),
        (lambda p: xmax - p[0], lambda p, q: _intersect_edge(p, q, 0, xmax)),
        (lambda p: p[1] - ymin, lambda p, q: _intersect_edge(p, q, 1, ymin)),
        (lambda p: ymax - p[1], lambda p, q: _intersect_edge(p, q, 1, ymax)),
    ]
    for inside_fn, intersect_fn in edges:
        if len(output) == 0:
            return None
        inp = output
        output = []
        for i in range(len(inp)):
            curr = inp[i]
            prev = inp[i - 1]
            curr_in = inside_fn(curr) >= -1e-14
            prev_in = inside_fn(prev) >= -1e-14
            if curr_in:
                if not prev_in:
                    output.append(intersect_fn(prev, curr))
                output.append(curr)
            elif prev_in:
                output.append(intersect_fn(prev, curr))
    if len(output) < 3:
        return None
    return np.array(output)


def _intersect_edge(p, q, axis, val):
    """Intersection of line segment p->q with axis=val."""
    t = (val - p[axis]) / (q[axis] - p[axis] + 1e-30)
    result = p + t * (q - p)
    result[axis] = val
    return result


# ── Edge Midpoint Augmentation ────────────────────────────────────────────

def add_edge_midpoints(vertices, elements):
    """
    Augment mesh with edge midpoint nodes for P2 VEM.

    For each polygon element, compute midpoint of each edge.
    Shared edges between adjacent elements get a single midpoint node.

    Parameters
    ----------
    vertices : (N, 2) array — original vertex coordinates
    elements : list of int arrays — P1 connectivity (vertex-only)

    Returns
    -------
    new_vertices : (N + N_mid, 2) array — vertices + midpoints
    new_elements : list of int arrays — each element: [v0..v_{n-1}, m0..m_{n-1}]
        where m_i is the midpoint of edge (v_i, v_{i+1 mod n})
    edge_midpoint_map : dict — (min_v, max_v) -> midpoint node index
    """
    new_vertices = list(vertices)
    edge_midpoint_map = {}
    new_elements = []

    for el in elements:
        el_int = el.astype(int)
        n_v = len(el_int)
        midpoint_ids = []

        for i in range(n_v):
            v1 = el_int[i]
            v2 = el_int[(i + 1) % n_v]
            edge_key = (min(v1, v2), max(v1, v2))

            if edge_key not in edge_midpoint_map:
                mid = 0.5 * (vertices[v1] + vertices[v2])
                mid_idx = len(new_vertices)
                new_vertices.append(mid)
                edge_midpoint_map[edge_key] = mid_idx
            midpoint_ids.append(edge_midpoint_map[edge_key])

        # Element DOF ordering: [v0, v1, ..., v_{n-1}, m0, m1, ..., m_{n-1}]
        new_el = np.concatenate([el_int, np.array(midpoint_ids, dtype=int)])
        new_elements.append(new_el)

    new_vertices = np.array(new_vertices)
    return new_vertices, new_elements, edge_midpoint_map


# ── P2 Polynomial Basis (P1-compatible ordering) ────────────────────────
#
# P1 basis (first 6, identical to vem_elasticity.py):
#   0: (1, 0)            — translation x
#   1: (0, 1)            — translation y
#   2: (-yhat, xhat)     — rigid rotation  (zero strain)
#   3: (xhat, 0)         — eps_xx = 1/h
#   4: (0, yhat)         — eps_yy = 1/h
#   5: (yhat, xhat)      — 2*eps_xy = 2/h  (symmetric shear)
#
# P2 extension (6 quadratic modes with non-constant strain):
#   6:  (xhat^2, 0)      — eps_xx = 2*xhat/h
#   7:  (0, yhat^2)      — eps_yy = 2*yhat/h
#   8:  (xhat*yhat, 0)   — eps_xx = yhat/h, 2*eps_xy = xhat/h
#   9:  (0, xhat*yhat)   — eps_yy = xhat/h, 2*eps_xy = yhat/h
#   10: (yhat^2, 0)      — 2*eps_xy = 2*yhat/h
#   11: (0, xhat^2)      — 2*eps_xy = 2*xhat/h
#
# Key insight: rotation mode MUST be at index 2 so that the vorticity
# B-row (row 2) matches the D-column (col 2). The old (m_i,0)/(0,m_i)
# interleaved ordering placed (xhat,0) at index 2, which has ZERO
# vorticity, causing G[2,2]=0 → singular G matrix.
#
# For quadratic modes (6-11), div(sigma(p_alpha)) != 0, so the B matrix
# needs a volume correction term in addition to the boundary integral.

def p2_polynomial_basis_2d():
    """
    Information about the 12 P2 basis functions for 2D vector elasticity.
    P1-compatible ordering: first 6 = P1 basis, next 6 = quadratic extension.
    """
    return {
        'n_polys': 12,
        'n_scalar_monomials': 6,
        'n_rigid': 3,
        'n_strain': 9,
        'basis': [
            '(1, 0)', '(0, 1)', '(-yhat, xhat)',
            '(xhat, 0)', '(0, yhat)', '(yhat, xhat)',
            '(xhat^2, 0)', '(0, yhat^2)', '(xhat*yhat, 0)',
            '(0, xhat*yhat)', '(yhat^2, 0)', '(0, xhat^2)',
        ],
    }


def _eval_p2_vector_basis(x, y, xc, yc, h):
    """
    Evaluate all 12 P2 vector basis functions at a point.
    P1-compatible ordering (see module-level comment).

    Returns: (12, 2) array where row i = (p_i^x, p_i^y).
    """
    xh = (x - xc) / h
    yh = (y - yc) / h
    return np.array([
        [1.0,  0.0],        # 0: (1, 0)
        [0.0,  1.0],        # 1: (0, 1)
        [-yh,  xh],         # 2: (-yhat, xhat)  rotation
        [xh,   0.0],        # 3: (xhat, 0)
        [0.0,  yh],         # 4: (0, yhat)
        [yh,   xh],         # 5: (yhat, xhat)   shear
        [xh**2,  0.0],      # 6: (xhat^2, 0)
        [0.0,  yh**2],      # 7: (0, yhat^2)
        [xh*yh,  0.0],      # 8: (xhat*yhat, 0)
        [0.0,  xh*yh],      # 9: (0, xhat*yhat)
        [yh**2,  0.0],      # 10: (yhat^2, 0)
        [0.0,  xh**2],      # 11: (0, xhat^2)
    ])


def _eval_p2_strain(x, y, xc, yc, h):
    """
    Evaluate strain (Voigt: eps_xx, eps_yy, 2*eps_xy) of each of the 12 P2
    vector basis functions at a point.  P1-compatible ordering.

    Returns: (12, 3) array where row i = strain of p_i in Voigt notation.
    """
    xh = (x - xc) / h
    yh = (y - yc) / h
    ih = 1.0 / h

    return np.array([
        [0.0,      0.0,      0.0],       # 0: (1, 0)
        [0.0,      0.0,      0.0],       # 1: (0, 1)
        [0.0,      0.0,      0.0],       # 2: (-yhat, xhat) — rotation, zero strain
        [ih,       0.0,      0.0],       # 3: (xhat, 0) — eps_xx=1/h
        [0.0,      ih,       0.0],       # 4: (0, yhat) — eps_yy=1/h
        [0.0,      0.0,      2*ih],      # 5: (yhat, xhat) — 2eps_xy=1/h+1/h=2/h
        [2*xh*ih,  0.0,      0.0],       # 6: (xhat^2, 0)
        [0.0,      2*yh*ih,  0.0],       # 7: (0, yhat^2)
        [yh*ih,    0.0,      xh*ih],     # 8: (xhat*yhat, 0)
        [0.0,      xh*ih,    yh*ih],     # 9: (0, xhat*yhat)
        [0.0,      0.0,      2*yh*ih],   # 10: (yhat^2, 0)
        [0.0,      0.0,      2*xh*ih],   # 11: (0, xhat^2)
    ])


def _compute_div_sigma(C, h):
    """
    Compute div(sigma(p_alpha)) for each of the 12 P2 basis functions.
    For linear modes (0-5), div=0. For quadratic modes (6-11), div is constant.

    Uses: div(sigma)_x = d(sigma_xx)/dx + d(sigma_xy)/dy
          div(sigma)_y = d(sigma_xy)/dx + d(sigma_yy)/dy

    Returns: (12, 2) array where row i = [div_x, div_y] of sigma(p_i).
    """
    h2 = h * h
    C00, C01, C11, C22 = C[0, 0], C[0, 1], C[1, 1], C[2, 2]

    div_sigma = np.zeros((12, 2))
    # Modes 0-5: constant or zero strain → div(sigma) = 0

    # Mode 6: (xhat^2, 0) — sigma_xx = C00*2xhat/h, sigma_yy = C01*2xhat/h, sigma_xy = 0
    #   div_x = d(C00*2xhat/h)/dx = 2*C00/h^2
    div_sigma[6] = [2 * C00 / h2, 0.0]

    # Mode 7: (0, yhat^2) — sigma_yy = C11*2yhat/h, sigma_xx = C01*2yhat/h, sigma_xy = 0
    #   div_y = d(C11*2yhat/h)/dy = 2*C11/h^2
    div_sigma[7] = [0.0, 2 * C11 / h2]

    # Mode 8: (xhat*yhat, 0) — eps=(yhat/h, 0, xhat/h)
    #   sigma_xx = C00*yhat/h, sigma_yy = C01*yhat/h, sigma_xy = C22*xhat/h
    #   div_x = d(sigma_xx)/dx + d(sigma_xy)/dy = 0 + 0 = 0
    #   div_y = d(sigma_xy)/dx + d(sigma_yy)/dy = C22/h^2 + C01/h^2
    div_sigma[8] = [0.0, (C22 + C01) / h2]

    # Mode 9: (0, xhat*yhat) — eps=(0, xhat/h, yhat/h)
    #   sigma_xx = C01*xhat/h, sigma_yy = C11*xhat/h, sigma_xy = C22*yhat/h
    #   div_x = d(sigma_xx)/dx + d(sigma_xy)/dy = C01/h^2 + C22/h^2 (wait...)
    #   Actually: sigma_xx = C01*(xhat/h), d(sigma_xx)/dx = C01/h^2
    #   sigma_xy = C22*(yhat/h), d(sigma_xy)/dy = C22/h^2
    #   div_x = C01/h^2 + C22/h^2 ... but wait, sigma_xy dep on y, dsigma_xy/dy = C22/h^2? No!
    #   sigma_xy = C22*yhat/h. d/dy of yhat/h = 1/h^2. So d(sigma_xy)/dy = C22/h^2.
    #   Actually: d(sigma_xy)/dy is part of div_x? No!
    #   div_x = d(sigma_xx)/dx + d(sigma_xy)/dy = C01/h^2 + 0 (sigma_xy = C22*yhat/h, d/dy = C22/h^2)
    #   Wait: sigma_xy depends on yhat. d(sigma_xy)/dy = C22*(1/h)*(1/h) = C22/h^2.
    #   So div_x = C01/h^2 + C22/h^2? Hmm let me recheck.
    #   sigma_xx = C[0,1]*eps_yy + ... = C01*(xhat/h). d(sigma_xx)/dx = C01/h^2. ✓
    #   sigma_xy = C22*(yhat/h). d(sigma_xy)/dy = C22/h^2. ✓
    #   div_x = C01/h^2 + C22/h^2. Hmm but that seems wrong...
    #   Let me recompute: eps = (0, xhat/h, yhat/h).
    #   sigma = C @ eps: sigma_xx = C[0,0]*0 + C[0,1]*xhat/h + C[0,2]*yhat/h = C01*xhat/h
    #   sigma_yy = C[1,1]*xhat/h, sigma_xy = C[2,2]*yhat/h.
    #   d(sigma_xx)/dx = C01*(1/h)*(d xhat/dx) = C01/h^2. ✓
    #   d(sigma_xy)/dy = C22*(1/h)*(d yhat/dy) = C22/h^2. ✓
    #   div_x = C01/h^2 + C22/h^2 ... but actually this is wrong because
    #   d(sigma_xy)/dy contributes to div_x only: div_x = d(sigma_xx)/dx + d(sigma_xy)/dy. ✓
    #   Wait no: div_x = d(sigma_xx)/dx + d(sigma_xy)/dy. For 2D stress:
    #   equilibrium: d(sigma_xx)/dx + d(sigma_xy)/dy = -f_x  (first eq)
    #                d(sigma_xy)/dx + d(sigma_yy)/dy = -f_y  (second eq)
    #   For mode 9: sigma_xy = C22*yhat/h. d(sigma_xy)/dy = C22/h^2
    #   So div_x = d(sigma_xx)/dx + d(sigma_xy)/dy = C01/h^2 + C22/h^2. Hmm.
    #   But sigma_xy depends on y, NOT x. So d(sigma_xy)/dx = 0.
    #   div_y = d(sigma_xy)/dx + d(sigma_yy)/dy = 0 + 0 = 0
    #   (sigma_yy = C11*xhat/h depends on x only, d/dy = 0.)
    div_sigma[9] = [(C01 + C22) / h2, 0.0]

    # Mode 10: (yhat^2, 0) — eps=(0, 0, 2*yhat/h)
    #   sigma_xy = C22*2*yhat/h, all others 0
    #   div_x = d(sigma_xx)/dx + d(sigma_xy)/dy = 0 + 2*C22/h^2
    div_sigma[10] = [2 * C22 / h2, 0.0]

    # Mode 11: (0, xhat^2) — eps=(0, 0, 2*xhat/h)
    #   sigma_xy = C22*2*xhat/h, all others 0
    #   div_y = d(sigma_xy)/dx + d(sigma_yy)/dy = 2*C22/h^2 + 0
    div_sigma[11] = [0.0, 2 * C22 / h2]

    return div_sigma


def _compute_strain_energy_matrix(verts, xc, yc, h, C):
    """
    Compute the 12x12 strain energy matrix a_K[alpha, beta] analytically.

    a_K[α, β] = ∫_K σ(p_α) : ε(p_β) dK = ∫_K ε(p_α)^T C ε(p_β) dK

    Uses sub-triangulation from centroid with 3-point Gauss quadrature
    on each triangle (exact for degree 2 polynomials → sufficient for
    the product of two linear strains).

    Returns: (12, 12) symmetric PSD matrix.
    """
    n_polys = 12
    n_v = len(verts)
    a_K = np.zeros((n_polys, n_polys))

    # 3-point Gauss quadrature on reference triangle (exact for degree 2)
    # Points: (1/6, 1/6), (2/3, 1/6), (1/6, 2/3), weights: 1/6 each
    gauss_pts = np.array([[1.0/6, 1.0/6], [2.0/3, 1.0/6], [1.0/6, 2.0/3]])
    gauss_wts = np.array([1.0/6, 1.0/6, 1.0/6])

    for i in range(n_v):
        j = (i + 1) % n_v
        # Triangle: centroid, vertex i, vertex j
        t0 = np.array([xc, yc])
        t1 = verts[i]
        t2 = verts[j]
        # Triangle area
        tri_area = 0.5 * abs((t1[0]-t0[0])*(t2[1]-t0[1]) - (t2[0]-t0[0])*(t1[1]-t0[1]))
        if tri_area < 1e-30:
            continue

        for gp, gw in zip(gauss_pts, gauss_wts):
            # Map from reference triangle to physical
            x_g = t0[0]*(1-gp[0]-gp[1]) + t1[0]*gp[0] + t2[0]*gp[1]
            y_g = t0[1]*(1-gp[0]-gp[1]) + t1[1]*gp[0] + t2[1]*gp[1]

            strain_all = _eval_p2_strain(x_g, y_g, xc, yc, h)  # (12, 3)
            # a_K[alpha, beta] += w * tri_area * eps_alpha^T @ C @ eps_beta
            for alpha in range(n_polys):
                eps_a = strain_all[alpha]
                if np.all(np.abs(eps_a) < 1e-30):
                    continue
                sig_a = C @ eps_a
                for beta in range(alpha, n_polys):
                    eps_b = strain_all[beta]
                    val = 2.0 * tri_area * gw * np.dot(sig_a, eps_b)
                    a_K[alpha, beta] += val
                    if beta != alpha:
                        a_K[beta, alpha] += val

    return a_K


# ── P2 VEM Core ───────────────────────────────────────────────────────────

def vem_p2_elasticity(vertices, elements, E_field, nu, bc_fixed_dofs, bc_vals,
                      load_dofs=None, load_vals=None, stabilization_alpha=1.0):
    """
    Second-order (P2) VEM for 2D plane-stress linear elasticity.

    Parameters
    ----------
    vertices : (N, 2) array — node coordinates (vertices + midpoints)
    elements : list of int arrays — P2 connectivity:
        [v0, v1, ..., v_{n-1}, m0, m1, ..., m_{n-1}]
        First n_v entries are vertices, next n_v are edge midpoints.
    E_field  : float or (N_el,) array — Young's modulus per element
    nu       : float — Poisson's ratio
    bc_fixed_dofs : array of int — constrained DOF indices (global)
    bc_vals  : array — prescribed values for fixed DOFs
    load_dofs : array of int — DOFs with applied point loads
    load_vals : array — load values
    stabilization_alpha : float — stabilization parameter (default 1.0)

    Returns
    -------
    u : (2*N,) displacement vector
    """
    n_nodes = vertices.shape[0]
    n_dofs = 2 * n_nodes
    n_polys = 12  # dim P_2^2 in 2D

    # Sparse assembly via COO triplets
    row_idx = []
    col_idx = []
    val_data = []
    F_global = np.zeros(n_dofs)

    for el_id in range(len(elements)):
        el_nodes = elements[el_id].astype(int)
        n_total = len(el_nodes)
        n_v = n_total // 2  # half vertex, half midpoint
        n_el_dofs = 2 * n_total  # = 4 * n_v

        # Vertex and midpoint coordinates
        vert_ids = el_nodes[:n_v]
        mid_ids = el_nodes[n_v:]
        verts = vertices[vert_ids]
        mids = vertices[mid_ids]
        all_coords = vertices[el_nodes]  # (2*n_v, 2)

        # ── Element E ──
        E_el = E_field[el_id] if hasattr(E_field, '__len__') else E_field

        # Plane stress constitutive matrix (Voigt: [sigma_xx, sigma_yy, sigma_xy])
        C = (E_el / (1.0 - nu**2)) * np.array([
            [1.0, nu,  0.0],
            [nu,  1.0, 0.0],
            [0.0, 0.0, (1.0 - nu) / 2.0]
        ])

        # ── Geometry (based on vertices only) ──
        area_comp = (verts[:, 0] * np.roll(verts[:, 1], -1)
                     - np.roll(verts[:, 0], -1) * verts[:, 1])
        area = 0.5 * abs(np.sum(area_comp))
        if area < 1e-30:
            continue

        centroid = np.sum(
            (np.roll(verts, -1, axis=0) + verts) * area_comp[:, None],
            axis=0) / (6.0 * area)

        # Diameter
        h = max(np.linalg.norm(verts[i] - verts[j])
                for i in range(n_v) for j in range(i + 1, n_v))
        if h < 1e-30:
            continue

        xc, yc = centroid

        # ── D matrix (n_el_dofs x 12) ──
        D = np.zeros((n_el_dofs, n_polys))
        for i in range(n_total):
            x_i, y_i = all_coords[i]
            basis_vals = _eval_p2_vector_basis(x_i, y_i, xc, yc, h)
            D[2 * i,     :] = basis_vals[:, 0]
            D[2 * i + 1, :] = basis_vals[:, 1]

        # ── B matrix (12 x n_el_dofs) ──
        B = np.zeros((n_polys, n_el_dofs))

        # --- Rows 0-1: translations (average displacement) ---
        for i in range(n_total):
            B[0, 2 * i]     = 1.0 / n_total
            B[1, 2 * i + 1] = 1.0 / n_total

        # --- Row 2: rigid rotation via vorticity boundary integral ---
        # 2*area*omega_avg = oint (u_y*n_x - u_x*n_y) ds
        for i in range(n_v):
            j = (i + 1) % n_v
            nx = verts[j][1] - verts[i][1]
            ny = verts[i][0] - verts[j][0]
            edge_len = np.sqrt(nx**2 + ny**2)
            if edge_len < 1e-30:
                continue
            nx_u, ny_u = nx / edge_len, ny / edge_len
            w_v = edge_len / 6.0
            w_m = 4.0 * edge_len / 6.0

            B[2, 2 * i]             += -ny_u * w_v
            B[2, 2 * i + 1]         +=  nx_u * w_v
            B[2, 2 * (n_v + i)]     += -ny_u * w_m
            B[2, 2 * (n_v + i) + 1] +=  nx_u * w_m
            B[2, 2 * j]             += -ny_u * w_v
            B[2, 2 * j + 1]         +=  nx_u * w_v
        B[2, :] /= (2.0 * area)

        # --- Rows 3-11: strain modes via boundary integrals ---
        # B[alpha, :] . u = oint sigma(p_alpha) . n . u_h ds
        # For quadratic modes (6-11), we also need volume correction:
        #   - oint ... = int_K eps(u_h) : C : eps(p_alpha) dK + int_K div(sigma(p_alpha)) . u_h dK
        # So: int_K eps:C:eps dK = oint ... - int_K div(sigma) . u_h dK
        # The B row should give the strain energy, so:
        # B[alpha, :] . u = boundary_integral - volume_correction

        # Boundary integral part (Simpson's rule on each edge)
        for i in range(n_v):
            j = (i + 1) % n_v
            nx = verts[j][1] - verts[i][1]
            ny = verts[i][0] - verts[j][0]
            edge_len = np.sqrt(nx**2 + ny**2)
            if edge_len < 1e-30:
                continue
            nx_u, ny_u = nx / edge_len, ny / edge_len

            pts = [verts[i], mids[i], verts[j]]
            dof_indices = [i, n_v + i, j]
            sw = [edge_len / 6.0, 4.0 * edge_len / 6.0, edge_len / 6.0]

            for pt_idx in range(3):
                px, py = pts[pt_idx]
                node_idx = dof_indices[pt_idx]
                w = sw[pt_idx]
                strain_all = _eval_p2_strain(px, py, xc, yc, h)

                for alpha in range(9):
                    poly_idx = 3 + alpha
                    eps_voigt = strain_all[poly_idx]
                    sigma = C @ eps_voigt
                    tx = sigma[0] * nx_u + sigma[2] * ny_u
                    ty = sigma[2] * nx_u + sigma[1] * ny_u
                    B[poly_idx, 2 * node_idx]     += w * tx
                    B[poly_idx, 2 * node_idx + 1] += w * ty

        # Volume correction for quadratic modes (6-11):
        # B[alpha, :] . u should = boundary - int_K div(sigma) . u_h dK
        # Approximate int_K div(sigma) . u_h dK ≈ div_sigma . (area/n_total) * sum(u_i)
        div_sigma = _compute_div_sigma(C, h)
        for alpha_idx in range(6, 12):
            dx, dy = div_sigma[alpha_idx]
            if abs(dx) + abs(dy) > 1e-30:
                for i in range(n_total):
                    B[alpha_idx, 2 * i]     -= dx * area / n_total
                    B[alpha_idx, 2 * i + 1] -= dy * area / n_total

        # ── Projector ──
        G = B @ D  # 12 x 12
        cond = np.linalg.cond(G)
        if cond > 1e12:
            G += 1e-10 * np.eye(n_polys)

        projector = np.linalg.solve(G, B)

        # Consistency: use analytically computed strain energy (guaranteed PSD)
        a_K_matrix = _compute_strain_energy_matrix(verts, xc, yc, h, C)
        K_cons = projector.T @ a_K_matrix @ projector

        # Stabilization (Wriggers projection)
        I_minus_PiD = np.eye(n_el_dofs) - D @ projector
        trace_k = np.trace(K_cons)
        stab_param = stabilization_alpha * trace_k / n_el_dofs if trace_k > 0 else E_el
        K_stab = stab_param * (I_minus_PiD.T @ I_minus_PiD)
        K_local = K_cons + K_stab

        # ── Assemble (sparse triplet) ──
        gdofs = np.zeros(n_el_dofs, dtype=int)
        for i in range(n_total):
            gdofs[2 * i]     = 2 * el_nodes[i]
            gdofs[2 * i + 1] = 2 * el_nodes[i] + 1

        ii, jj = np.meshgrid(gdofs, gdofs, indexing='ij')
        row_idx.append(ii.ravel())
        col_idx.append(jj.ravel())
        val_data.append(K_local.ravel())

    # Build sparse global stiffness matrix
    row_idx = np.concatenate(row_idx)
    col_idx = np.concatenate(col_idx)
    val_data = np.concatenate(val_data)
    K_global = sp.csr_matrix((val_data, (row_idx, col_idx)), shape=(n_dofs, n_dofs))

    # ── Point loads ──
    if load_dofs is not None and load_vals is not None:
        F_global[load_dofs] += load_vals

    # ── Solve with BCs ──
    u = np.zeros(n_dofs)
    bc_set = set(bc_fixed_dofs)
    internal = np.array([i for i in range(n_dofs) if i not in bc_set])

    u[bc_fixed_dofs] = bc_vals
    F_global -= K_global[:, bc_fixed_dofs].toarray() @ bc_vals

    K_ii = K_global[np.ix_(internal, internal)]
    u[internal] = sp.linalg.spsolve(K_ii, F_global[internal])

    return u


# ── Visualization ─────────────────────────────────────────────────────────

def plot_p2_elasticity(vertices, elements, u, field='magnitude',
                       deform_scale=1.0, title=None, ax=None, save=None):
    """Plot deformed P2 mesh colored by displacement field (vertex cells only)."""
    ux = u[0::2]
    uy = u[1::2]
    deformed = vertices + deform_scale * np.column_stack([ux, uy])

    if field == 'magnitude':
        vals = np.sqrt(ux**2 + uy**2)
        cbar_label = '|u|'
    elif field == 'ux':
        vals = ux
        cbar_label = '$u_x$'
    else:
        vals = uy
        cbar_label = '$u_y$'

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    else:
        fig = ax.figure

    patches = []
    patch_colors = []
    for el in elements:
        el_int = el.astype(int)
        n_v = len(el_int) // 2
        vert_ids = el_int[:n_v]  # Only use vertices for polygon shape
        poly = MplPolygon(deformed[vert_ids], closed=True)
        patches.append(poly)
        # Color by average displacement of all nodes
        patch_colors.append(np.mean(vals[el_int]))

    pc = PatchCollection(patches, cmap='viridis', edgecolor='k', linewidth=0.3)
    pc.set_array(np.array(patch_colors))
    ax.add_collection(pc)
    ax.set_xlim(deformed[:, 0].min() - 0.05, deformed[:, 0].max() + 0.05)
    ax.set_ylim(deformed[:, 1].min() - 0.05, deformed[:, 1].max() + 0.05)
    ax.set_aspect('equal')
    fig.colorbar(pc, ax=ax, label=cbar_label, shrink=0.8)

    if title:
        ax.set_title(title)

    if save and own_fig:
        plt.tight_layout()
        plt.savefig(save, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save}")
        plt.close()

    return ax


# ── Demo: P2 vs P1 Comparison ────────────────────────────────────────────

def demo_p2_vs_p1():
    """
    Compare P1 and P2 VEM on a Voronoi mesh with DI-dependent E field.

    - 20-cell Voronoi mesh, domain (0,1)x(0,1)
    - E(DI) with spatial gradient
    - Bottom fixed, top loaded
    - 2x2 comparison figure
    """
    print("=" * 60)
    print("Demo: P2 vs P1 VEM Comparison")
    print("=" * 60)

    import vem_elasticity as p1_module

    # ── Generate Voronoi mesh ──
    vertices, elements, boundary = generate_voronoi_mesh(20, seed=42)
    n_nodes_p1 = vertices.shape[0]
    n_els = len(elements)
    print(f"  Mesh: {n_els} elements, {n_nodes_p1} P1 nodes")

    # ── E(DI) field ──
    E_max = 1000.0
    E_min = 30.0
    n_hill = 2
    nu = 0.3

    center = np.array([0.5, 0.5])
    max_dist = 0.5 * np.sqrt(2)

    E_per_element = np.zeros(n_els)
    for i, el in enumerate(elements):
        el_int = el.astype(int)
        el_centroid = vertices[el_int].mean(axis=0)
        dist = np.linalg.norm(el_centroid - center)
        DI = 0.9 - 0.8 * (dist / max_dist)
        DI = np.clip(DI, 0.05, 0.95)
        E_per_element[i] = E_min + (E_max - E_min) * (1.0 - DI) ** n_hill

    print(f"  E range: [{E_per_element.min():.0f}, {E_per_element.max():.0f}] Pa")

    # ── Boundary conditions ──
    tol = 1e-6

    # --- P1 solve ---
    print("  Solving P1...")
    bottom_p1 = np.where(vertices[:, 1] < tol)[0]
    bc_dofs_p1 = np.concatenate([2 * bottom_p1, 2 * bottom_p1 + 1])
    bc_vals_p1 = np.zeros(len(bc_dofs_p1))

    top_p1 = np.where(vertices[:, 1] > 1.0 - tol)[0]
    load_per_node_p1 = -0.5 / max(len(top_p1), 1)
    load_dofs_p1 = 2 * top_p1 + 1
    load_vals_p1 = np.full(len(top_p1), load_per_node_p1)

    u_p1 = p1_module.vem_elasticity(
        vertices, elements, E_per_element, nu,
        bc_dofs_p1, bc_vals_p1, load_dofs_p1, load_vals_p1)

    ux_p1 = u_p1[0::2]
    uy_p1 = u_p1[1::2]
    mag_p1 = np.sqrt(ux_p1**2 + uy_p1**2)
    print(f"  P1 max |u|: {mag_p1.max():.6f}")

    # --- P2 solve ---
    print("  Solving P2...")
    p2_vertices, p2_elements, edge_map = add_edge_midpoints(vertices, elements)
    n_nodes_p2 = p2_vertices.shape[0]
    print(f"  P2 nodes: {n_nodes_p2} ({n_nodes_p1} vertices + {n_nodes_p2 - n_nodes_p1} midpoints)")

    # BCs for P2: fix all nodes on bottom edge (vertices + midpoints)
    bottom_p2 = np.where(p2_vertices[:, 1] < tol)[0]
    bc_dofs_p2 = np.concatenate([2 * bottom_p2, 2 * bottom_p2 + 1])
    bc_vals_p2 = np.zeros(len(bc_dofs_p2))

    top_p2 = np.where(p2_vertices[:, 1] > 1.0 - tol)[0]
    load_per_node_p2 = -0.5 / max(len(top_p2), 1)
    load_dofs_p2 = 2 * top_p2 + 1
    load_vals_p2 = np.full(len(top_p2), load_per_node_p2)

    u_p2 = vem_p2_elasticity(
        p2_vertices, p2_elements, E_per_element, nu,
        bc_dofs_p2, bc_vals_p2, load_dofs_p2, load_vals_p2)

    ux_p2 = u_p2[0::2]
    uy_p2 = u_p2[1::2]
    mag_p2 = np.sqrt(ux_p2**2 + uy_p2**2)
    print(f"  P2 max |u|: {mag_p2.max():.6f}")

    # ── 2x2 comparison plot ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # (a) P1 deformed mesh + |u| colormap
    ax = axes[0, 0]
    deformed_p1 = vertices + 200 * np.column_stack([ux_p1, uy_p1])
    patches_p1 = []
    colors_p1 = []
    for el in elements:
        el_int = el.astype(int)
        poly = MplPolygon(deformed_p1[el_int], closed=True)
        patches_p1.append(poly)
        colors_p1.append(np.mean(mag_p1[el_int]))
    pc1 = PatchCollection(patches_p1, cmap='viridis', edgecolor='k', linewidth=0.3)
    pc1.set_array(np.array(colors_p1))
    ax.add_collection(pc1)
    ax.set_xlim(deformed_p1[:, 0].min() - 0.05, deformed_p1[:, 0].max() + 0.05)
    ax.set_ylim(deformed_p1[:, 1].min() - 0.05, deformed_p1[:, 1].max() + 0.05)
    ax.set_aspect('equal')
    ax.set_title('(a) P1 VEM: deformed mesh + |u|')
    fig.colorbar(pc1, ax=ax, label='|u|', shrink=0.8)

    # (b) P2 deformed mesh + |u| colormap (show vertex polygons only)
    ax = axes[0, 1]
    deformed_p2 = p2_vertices + 200 * np.column_stack([ux_p2, uy_p2])
    patches_p2 = []
    colors_p2 = []
    for el in p2_elements:
        el_int = el.astype(int)
        n_v = len(el_int) // 2
        vert_ids = el_int[:n_v]
        poly = MplPolygon(deformed_p2[vert_ids], closed=True)
        patches_p2.append(poly)
        colors_p2.append(np.mean(mag_p2[el_int]))
    pc2 = PatchCollection(patches_p2, cmap='viridis', edgecolor='k', linewidth=0.3)
    pc2.set_array(np.array(colors_p2))
    ax.add_collection(pc2)
    ax.set_xlim(deformed_p2[:, 0].min() - 0.05, deformed_p2[:, 0].max() + 0.05)
    ax.set_ylim(deformed_p2[:, 1].min() - 0.05, deformed_p2[:, 1].max() + 0.05)
    ax.set_aspect('equal')
    ax.set_title('(b) P2 VEM: deformed mesh + |u|')
    fig.colorbar(pc2, ax=ax, label='|u|', shrink=0.8)

    # (c) Difference field |u_P2 - u_P1_interp|
    # Interpolate P1 to P2 nodes: P1 vertices map directly, midpoints = avg of endpoints
    ax = axes[1, 0]
    u_p1_interp = np.zeros(2 * n_nodes_p2)
    # Vertex nodes: direct copy
    u_p1_interp[:2 * n_nodes_p1] = u_p1.copy()
    # Midpoint nodes: average of endpoint values
    for (v1, v2), mid_idx in edge_map.items():
        u_p1_interp[2 * mid_idx]     = 0.5 * (u_p1[2 * v1]     + u_p1[2 * v2])
        u_p1_interp[2 * mid_idx + 1] = 0.5 * (u_p1[2 * v1 + 1] + u_p1[2 * v2 + 1])

    diff_ux = ux_p2 - u_p1_interp[0::2]
    diff_uy = uy_p2 - u_p1_interp[1::2]
    diff_mag = np.sqrt(diff_ux**2 + diff_uy**2)

    patches_diff = []
    colors_diff = []
    for el in p2_elements:
        el_int = el.astype(int)
        n_v = len(el_int) // 2
        vert_ids = el_int[:n_v]
        poly = MplPolygon(vertices[vert_ids], closed=True)
        patches_diff.append(poly)
        colors_diff.append(np.mean(diff_mag[el_int]))
    pc3 = PatchCollection(patches_diff, cmap='hot_r', edgecolor='k', linewidth=0.3)
    pc3.set_array(np.array(colors_diff))
    ax.add_collection(pc3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    ax.set_title('(c) Difference |u_P2 - u_P1_interp|')
    fig.colorbar(pc3, ax=ax, label='|u_P2 - u_P1|', shrink=0.8)

    # (d) Stress comparison: sigma_xx along y=0.5
    ax = axes[1, 1]
    # Sample stress at element centroids near y=0.5
    y_target = 0.5
    y_tol = 0.15

    x_vals_p1, sxx_p1 = _sample_stress_along_line(
        vertices, elements, u_p1, E_per_element, nu, y_target, y_tol, order=1)
    x_vals_p2, sxx_p2 = _sample_stress_along_line(
        p2_vertices, p2_elements, u_p2, E_per_element, nu, y_target, y_tol, order=2)

    if len(x_vals_p1) > 0:
        sort_p1 = np.argsort(x_vals_p1)
        ax.plot(x_vals_p1[sort_p1], sxx_p1[sort_p1], 'o-', label='P1', markersize=4)
    if len(x_vals_p2) > 0:
        sort_p2 = np.argsort(x_vals_p2)
        ax.plot(x_vals_p2[sort_p2], sxx_p2[sort_p2], 's--', label='P2', markersize=4)
    ax.set_xlabel('x')
    ax.set_ylabel(r'$\sigma_{xx}$')
    ax.set_title(r'(d) $\sigma_{xx}$ along y=0.5')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle('P2 vs P1 VEM: Biofilm E(DI) Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()

    import os
    save_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, 'vem_p2_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close()


def _sample_stress_along_line(vertices, elements, u, E_field, nu, y_target, y_tol,
                              order=1):
    """
    Sample sigma_xx at element centroids near y=y_target.

    Parameters
    ----------
    order : 1 for P1 elements, 2 for P2 elements
    """
    x_vals = []
    sxx_vals = []

    for el_id, el in enumerate(elements):
        el_int = el.astype(int)
        if order == 2:
            n_v = len(el_int) // 2
            vert_ids = el_int[:n_v]
        else:
            n_v = len(el_int)
            vert_ids = el_int

        verts = vertices[vert_ids]
        centroid = verts.mean(axis=0)

        if abs(centroid[1] - y_target) > y_tol:
            continue

        E_el = E_field[el_id] if hasattr(E_field, '__len__') else E_field
        C = (E_el / (1.0 - nu**2)) * np.array([
            [1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, (1.0 - nu) / 2.0]])

        # Compute average strain from boundary integral (vertex normals)
        area_comp = (verts[:, 0] * np.roll(verts[:, 1], -1)
                     - np.roll(verts[:, 0], -1) * verts[:, 1])
        area = 0.5 * abs(np.sum(area_comp))
        if area < 1e-30:
            continue

        # Average strain via boundary integral
        eps_avg = np.zeros(3)  # [eps_xx, eps_yy, 2*eps_xy]

        if order == 1:
            # P1: trapezoidal
            for i in range(n_v):
                j = (i + 1) % n_v
                vi_id = vert_ids[i]
                vj_id = vert_ids[j]
                nx = verts[j, 1] - verts[i, 1]
                ny = verts[i, 0] - verts[j, 0]
                ux_avg = 0.5 * (u[2*vi_id] + u[2*vj_id])
                uy_avg = 0.5 * (u[2*vi_id+1] + u[2*vj_id+1])
                eps_avg[0] += ux_avg * nx
                eps_avg[1] += uy_avg * ny
                eps_avg[2] += ux_avg * ny + uy_avg * nx
        else:
            # P2: Simpson with midpoints
            mid_ids = el_int[n_v:]
            for i in range(n_v):
                j = (i + 1) % n_v
                vi_id = vert_ids[i]
                vj_id = vert_ids[j]
                mi_id = mid_ids[i]
                nx = verts[j, 1] - verts[i, 1]
                ny = verts[i, 0] - verts[j, 0]
                edge_len = np.sqrt(nx**2 + ny**2)
                if edge_len < 1e-30:
                    continue
                nx_u = nx / edge_len
                ny_u = ny / edge_len
                # Simpson weights
                w1 = edge_len / 6.0
                w4 = 4.0 * edge_len / 6.0
                w2 = edge_len / 6.0
                # eps_xx contribution: u_x * n_x
                eps_avg[0] += nx_u * (w1 * u[2*vi_id] + w4 * u[2*mi_id] + w2 * u[2*vj_id])
                # eps_yy contribution: u_y * n_y
                eps_avg[1] += ny_u * (w1 * u[2*vi_id+1] + w4 * u[2*mi_id+1] + w2 * u[2*vj_id+1])
                # 2*eps_xy: u_x*n_y + u_y*n_x
                eps_avg[2] += ny_u * (w1 * u[2*vi_id] + w4 * u[2*mi_id] + w2 * u[2*vj_id])
                eps_avg[2] += nx_u * (w1 * u[2*vi_id+1] + w4 * u[2*mi_id+1] + w2 * u[2*vj_id+1])

        eps_avg /= area
        sigma = C @ eps_avg

        x_vals.append(centroid[0])
        sxx_vals.append(sigma[0])

    return np.array(x_vals), np.array(sxx_vals)


# ── Convergence Study ─────────────────────────────────────────────────────

def convergence_p2_vs_p1():
    """
    h-convergence study comparing P1 and P2 VEM with manufactured solution.

    Exact solution:
      u_x = sin(pi*x) * sin(pi*y)
      u_y = cos(pi*x) * cos(pi*y)

    Mesh sizes: 10, 20, 40, 80 Voronoi cells.
    Compute L2 and H1 errors for both P1 and P2.
    """
    print("\n" + "=" * 60)
    print("Convergence Study: P2 vs P1 VEM")
    print("=" * 60)

    import vem_elasticity as p1_module

    E_val = 1000.0
    nu = 0.3

    def exact_ux(x, y):
        return np.sin(np.pi * x) * np.sin(np.pi * y)

    def exact_uy(x, y):
        return np.cos(np.pi * x) * np.cos(np.pi * y)

    def exact_eps_xx(x, y):
        return np.pi * np.cos(np.pi * x) * np.sin(np.pi * y)

    def exact_eps_yy(x, y):
        return -np.pi * np.cos(np.pi * x) * np.sin(np.pi * y)

    def exact_2eps_xy(x, y):
        return (np.pi * np.sin(np.pi * x) * np.cos(np.pi * y)
                - np.pi * np.sin(np.pi * x) * np.cos(np.pi * y))

    n_cells_list = [10, 20, 40, 80]
    h_list = []
    err_L2_p1, err_L2_p2 = [], []
    err_H1_p1, err_H1_p2 = [], []

    for n_cells in n_cells_list:
        print(f"\n  n_cells = {n_cells}")

        # Generate mesh
        vertices, elements, boundary = generate_voronoi_mesh(n_cells, seed=123)
        n_els = len(elements)

        # Compute average mesh size
        h_avg = 1.0 / np.sqrt(n_els)
        h_list.append(h_avg)

        # E field (uniform for manufactured solution)
        E_per_el = np.full(n_els, E_val)

        # Compute body force from exact solution (not trivial for manufactured soln)
        # Instead: prescribe exact BC on all boundary nodes, solve, compare interior

        # --- P1 ---
        bc_dofs_p1 = np.concatenate([2 * boundary, 2 * boundary + 1])
        bc_vals_p1 = np.concatenate([
            exact_ux(vertices[boundary, 0], vertices[boundary, 1]),
            exact_uy(vertices[boundary, 0], vertices[boundary, 1]),
        ])

        # Body force from equilibrium: div(sigma) + f = 0
        # For manufactured solution, compute f at each element and apply as load
        # Simplified: just use Dirichlet BC on boundary + body force
        f_global_p1 = _compute_body_force_p1(
            vertices, elements, E_per_el, nu, exact_ux, exact_uy)

        u_p1 = _solve_with_body_force_p1(
            vertices, elements, E_per_el, nu, bc_dofs_p1, bc_vals_p1, f_global_p1)

        err_l2_1, err_h1_1 = _compute_errors_p1(
            vertices, elements, u_p1, exact_ux, exact_uy,
            exact_eps_xx, exact_eps_yy, exact_2eps_xy)
        err_L2_p1.append(err_l2_1)
        err_H1_p1.append(err_h1_1)
        print(f"    P1: L2={err_l2_1:.4e}, H1={err_h1_1:.4e}")

        # --- P2 ---
        p2_verts, p2_elems, edge_map = add_edge_midpoints(vertices, elements)
        n_nodes_p2 = p2_verts.shape[0]

        # P2 boundary: all nodes on domain boundary
        tol = 1e-6
        bnd_p2 = np.where(
            (p2_verts[:, 0] < tol) | (p2_verts[:, 0] > 1.0 - tol) |
            (p2_verts[:, 1] < tol) | (p2_verts[:, 1] > 1.0 - tol)
        )[0]

        bc_dofs_p2 = np.concatenate([2 * bnd_p2, 2 * bnd_p2 + 1])
        bc_vals_p2 = np.concatenate([
            exact_ux(p2_verts[bnd_p2, 0], p2_verts[bnd_p2, 1]),
            exact_uy(p2_verts[bnd_p2, 0], p2_verts[bnd_p2, 1]),
        ])

        f_global_p2 = _compute_body_force_p2(
            p2_verts, p2_elems, E_per_el, nu, exact_ux, exact_uy)

        u_p2 = _solve_with_body_force_p2(
            p2_verts, p2_elems, E_per_el, nu, bc_dofs_p2, bc_vals_p2, f_global_p2)

        err_l2_2, err_h1_2 = _compute_errors_p2(
            p2_verts, p2_elems, u_p2, exact_ux, exact_uy,
            exact_eps_xx, exact_eps_yy, exact_2eps_xy)
        err_L2_p2.append(err_l2_2)
        err_H1_p2.append(err_h1_2)
        print(f"    P2: L2={err_l2_2:.4e}, H1={err_h1_2:.4e}")

    # Compute convergence rates
    h_arr = np.array(h_list)

    print("\n  Convergence rates (log-log slope):")
    if len(h_list) >= 2:
        rate_L2_p1 = np.polyfit(np.log(h_arr), np.log(np.array(err_L2_p1) + 1e-16), 1)[0]
        rate_H1_p1 = np.polyfit(np.log(h_arr), np.log(np.array(err_H1_p1) + 1e-16), 1)[0]
        rate_L2_p2 = np.polyfit(np.log(h_arr), np.log(np.array(err_L2_p2) + 1e-16), 1)[0]
        rate_H1_p2 = np.polyfit(np.log(h_arr), np.log(np.array(err_H1_p2) + 1e-16), 1)[0]
        print(f"    P1: L2 rate = {rate_L2_p1:.2f} (expect ~2), H1 rate = {rate_H1_p1:.2f} (expect ~1)")
        print(f"    P2: L2 rate = {rate_L2_p2:.2f} (expect ~3), H1 rate = {rate_H1_p2:.2f} (expect ~2)")

    # ── Plot convergence ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.loglog(h_arr, err_L2_p1, 'bo-', label=f'P1 (rate={rate_L2_p1:.2f})', markersize=6)
    ax.loglog(h_arr, err_L2_p2, 'rs-', label=f'P2 (rate={rate_L2_p2:.2f})', markersize=6)
    # Reference slopes
    ax.loglog(h_arr, h_arr**2 * err_L2_p1[0] / h_arr[0]**2, 'b--', alpha=0.3, label='O(h^2)')
    ax.loglog(h_arr, h_arr**3 * err_L2_p2[0] / h_arr[0]**3, 'r--', alpha=0.3, label='O(h^3)')
    ax.set_xlabel('h (mesh size)')
    ax.set_ylabel('L2 error')
    ax.set_title('L2 Error Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    ax = axes[1]
    ax.loglog(h_arr, err_H1_p1, 'bo-', label=f'P1 (rate={rate_H1_p1:.2f})', markersize=6)
    ax.loglog(h_arr, err_H1_p2, 'rs-', label=f'P2 (rate={rate_H1_p2:.2f})', markersize=6)
    ax.loglog(h_arr, h_arr**1 * err_H1_p1[0] / h_arr[0]**1, 'b--', alpha=0.3, label='O(h)')
    ax.loglog(h_arr, h_arr**2 * err_H1_p2[0] / h_arr[0]**2, 'r--', alpha=0.3, label='O(h^2)')
    ax.set_xlabel('h (mesh size)')
    ax.set_ylabel('H1 error')
    ax.set_title('H1 Error Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    fig.suptitle('VEM Convergence: P2 vs P1 (Manufactured Solution)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    import os
    save_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, 'vem_p2_convergence.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"\n  Saved: {path}")
    plt.close()


# ── Helper functions for convergence study ────────────────────────────────

def _compute_body_force_p1(vertices, elements, E_field, nu, exact_ux, exact_uy):
    """
    Compute consistent body force vector for P1 VEM from manufactured solution.
    Uses sub-triangulation quadrature to integrate -div(sigma(u_exact)).
    """
    n_dofs = 2 * vertices.shape[0]
    f = np.zeros(n_dofs)

    C_factor = 1.0 / (1.0 - nu**2)

    for el_id, el in enumerate(elements):
        el_int = el.astype(int)
        n_v = len(el_int)
        verts = vertices[el_int]

        E_el = E_field[el_id] if hasattr(E_field, '__len__') else E_field

        area_comp = (verts[:, 0] * np.roll(verts[:, 1], -1)
                     - np.roll(verts[:, 0], -1) * verts[:, 1])
        area = 0.5 * abs(np.sum(area_comp))
        centroid = verts.mean(axis=0)

        # Body force at centroid from manufactured solution:
        # f = -div(sigma(u_exact))
        # For u_x = sin(pi*x)*sin(pi*y), u_y = cos(pi*x)*cos(pi*y)
        # sigma_xx = E/(1-nu^2) * (eps_xx + nu*eps_yy)
        # Compute -div(sigma) analytically
        x, y = centroid
        pi2 = np.pi**2

        # eps_xx = pi*cos(pi*x)*sin(pi*y)
        # eps_yy = -pi*cos(pi*x)*sin(pi*y)
        # 2*eps_xy = 0 (for this solution)
        # d(eps_xx)/dx = -pi^2*sin(pi*x)*sin(pi*y)
        # d(eps_yy)/dy = -pi^2*cos(pi*x)*cos(pi*y)
        # d(eps_xy)/dy = 0, d(eps_xy)/dx = 0

        # sigma_xx = E*C_f*(eps_xx + nu*eps_yy)
        # dsigma_xx/dx = E*C_f*(d(eps_xx)/dx + nu*d(eps_yy)/dx)
        # dsigma_xy/dy = E*C_f*((1-nu)/2)*(d(2eps_xy)/dy)

        # f_x = -(dsigma_xx/dx + dsigma_xy/dy)
        # f_y = -(dsigma_xy/dx + dsigma_yy/dy)

        sin_px = np.sin(np.pi * x)
        cos_px = np.cos(np.pi * x)
        sin_py = np.sin(np.pi * y)
        cos_py = np.cos(np.pi * y)

        # Full stress divergence (all terms):
        # eps_xx = pi*cos(px)*sin(py)
        # eps_yy = -pi*cos(px)*sin(py)   [d(cos(px)*cos(py))/dy = -pi*cos(px)*sin(py)]
        # 2*eps_xy = pi*sin(px)*cos(py) + (-pi*sin(px)*cos(py)) = 0
        # Actually let me recompute:
        # du_x/dy = pi*sin(px)*cos(py)
        # du_y/dx = -pi*sin(px)*cos(py)
        # 2*eps_xy = du_x/dy + du_y/dx = 0

        # So sigma_xy = 0 everywhere.
        # sigma_xx = E*C_f*(1+nu)*pi*cos(px)*sin(py) ... wait:
        # eps_xx + nu*eps_yy = pi*cos(px)*sin(py)(1 - nu)
        # sigma_xx = E*C_f * pi*cos(px)*sin(py)*(1-nu)
        # dsigma_xx/dx = E*C_f * (-pi^2)*sin(px)*sin(py)*(1-nu)

        # eps_yy + nu*eps_xx = -pi*cos(px)*sin(py)(1-nu)
        # sigma_yy = -E*C_f * pi*cos(px)*sin(py)*(1-nu)
        # dsigma_yy/dy = -E*C_f * pi^2*cos(px)*cos(py)*(1-nu)

        # f_x = -dsigma_xx/dx = E*C_f*pi^2*sin(px)*sin(py)*(1-nu)
        # f_y = -dsigma_yy/dy = E*C_f*pi^2*cos(px)*cos(py)*(1-nu)

        fx = E_el * C_factor * pi2 * sin_px * sin_py * (1.0 - nu)
        fy = E_el * C_factor * pi2 * cos_px * cos_py * (1.0 - nu)

        # Distribute to vertices (equal share for P1)
        for i in range(n_v):
            f[2 * el_int[i]]     += fx * area / n_v
            f[2 * el_int[i] + 1] += fy * area / n_v

    return f


def _solve_with_body_force_p1(vertices, elements, E_field, nu,
                               bc_fixed_dofs, bc_vals, f_global):
    """Solve P1 VEM with given body force vector."""
    import vem_elasticity as p1_module

    n_nodes = vertices.shape[0]
    n_dofs = 2 * n_nodes

    # Assemble stiffness (reuse internal function if available)
    K = p1_module._assemble_stiffness_sparse(vertices, elements, E_field, nu)

    u = np.zeros(n_dofs)
    bc_set = set(bc_fixed_dofs)
    internal = np.array([i for i in range(n_dofs) if i not in bc_set])

    u[bc_fixed_dofs] = bc_vals
    rhs = f_global.copy()
    rhs -= K[:, bc_fixed_dofs].toarray() @ bc_vals

    K_ii = K[np.ix_(internal, internal)]
    u[internal] = sp.linalg.spsolve(K_ii, rhs[internal])
    return u


def _compute_body_force_p2(vertices, elements, E_field, nu, exact_ux, exact_uy):
    """
    Compute consistent body force vector for P2 VEM from manufactured solution.

    Uses projector-based distribution: F = proj^T @ integral(f · p_alpha dK).
    Sub-triangulation Gauss quadrature for the volume integrals.
    """
    n_dofs = 2 * vertices.shape[0]
    f = np.zeros(n_dofs)
    n_polys = 12

    C_factor = 1.0 / (1.0 - nu**2)

    # 3-point Gauss on reference triangle
    gauss_pts = np.array([[1.0/6, 1.0/6], [2.0/3, 1.0/6], [1.0/6, 2.0/3]])
    gauss_wts = np.array([1.0/6, 1.0/6, 1.0/6])

    for el_id, el in enumerate(elements):
        el_int = el.astype(int)
        n_total = len(el_int)
        n_v = n_total // 2
        n_el_dofs = 2 * n_total
        vert_ids = el_int[:n_v]
        mid_ids = el_int[n_v:]
        verts = vertices[vert_ids]
        mids = vertices[mid_ids]
        all_coords = vertices[el_int]

        E_el = E_field[el_id] if hasattr(E_field, '__len__') else E_field
        C = (E_el / (1.0 - nu**2)) * np.array([
            [1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, (1.0 - nu) / 2.0]])

        area_comp = (verts[:, 0] * np.roll(verts[:, 1], -1)
                     - np.roll(verts[:, 0], -1) * verts[:, 1])
        area = 0.5 * abs(np.sum(area_comp))
        if area < 1e-30:
            continue
        centroid = np.sum(
            (np.roll(verts, -1, axis=0) + verts) * area_comp[:, None],
            axis=0) / (6.0 * area)
        h = max(np.linalg.norm(verts[i] - verts[j])
                for i in range(n_v) for j in range(i + 1, n_v))
        if h < 1e-30:
            continue
        xc, yc = centroid

        # Build projector for this element (same as in assembly)
        D = np.zeros((n_el_dofs, n_polys))
        for i in range(n_total):
            bv = _eval_p2_vector_basis(all_coords[i][0], all_coords[i][1], xc, yc, h)
            D[2*i, :] = bv[:, 0]
            D[2*i+1, :] = bv[:, 1]

        B = np.zeros((n_polys, n_el_dofs))
        for i in range(n_total):
            B[0, 2*i] = 1.0 / n_total
            B[1, 2*i+1] = 1.0 / n_total
        for i in range(n_v):
            j = (i+1) % n_v
            nx = verts[j][1]-verts[i][1]; ny = verts[i][0]-verts[j][0]
            el2 = np.sqrt(nx**2 + ny**2)
            if el2 < 1e-30:
                continue
            nxu, nyu = nx/el2, ny/el2
            wv = el2/6.0; wm = 4.0*el2/6.0
            B[2, 2*i] += -nyu*wv; B[2, 2*i+1] += nxu*wv
            B[2, 2*(n_v+i)] += -nyu*wm; B[2, 2*(n_v+i)+1] += nxu*wm
            B[2, 2*j] += -nyu*wv; B[2, 2*j+1] += nxu*wv
        B[2, :] /= (2.0 * area)
        for i in range(n_v):
            j = (i+1) % n_v
            nx = verts[j][1]-verts[i][1]; ny = verts[i][0]-verts[j][0]
            el2 = np.sqrt(nx**2 + ny**2)
            if el2 < 1e-30:
                continue
            nxu, nyu = nx/el2, ny/el2
            pts = [verts[i], mids[i], verts[j]]
            didx = [i, n_v+i, j]
            sw = [el2/6.0, 4.0*el2/6.0, el2/6.0]
            for pi in range(3):
                px, py = pts[pi]; ni = didx[pi]; w = sw[pi]
                sa = _eval_p2_strain(px, py, xc, yc, h)
                for a in range(9):
                    pidx = 3+a; eps = sa[pidx]; sig = C @ eps
                    tx = sig[0]*nxu + sig[2]*nyu
                    ty = sig[2]*nxu + sig[1]*nyu
                    B[pidx, 2*ni] += w*tx; B[pidx, 2*ni+1] += w*ty
        ds = _compute_div_sigma(C, h)
        for ai in range(6, 12):
            dx, dy = ds[ai]
            if abs(dx)+abs(dy) > 1e-30:
                for i in range(n_total):
                    B[ai, 2*i] -= dx*area/n_total
                    B[ai, 2*i+1] -= dy*area/n_total

        G = B @ D
        cond = np.linalg.cond(G)
        if cond > 1e12:
            G += 1e-10 * np.eye(n_polys)
        projector = np.linalg.solve(G, B)

        # Compute integral(f . p_alpha dK) via sub-triangulation
        f_poly = np.zeros(n_polys)  # integral of f . p_alpha over K
        for i in range(n_v):
            j = (i + 1) % n_v
            t0 = np.array([xc, yc])
            t1 = verts[i]; t2 = verts[j]
            tri_area = 0.5 * abs((t1[0]-t0[0])*(t2[1]-t0[1])
                                 - (t2[0]-t0[0])*(t1[1]-t0[1]))
            if tri_area < 1e-30:
                continue
            for gp, gw in zip(gauss_pts, gauss_wts):
                x_g = t0[0]*(1-gp[0]-gp[1]) + t1[0]*gp[0] + t2[0]*gp[1]
                y_g = t0[1]*(1-gp[0]-gp[1]) + t1[1]*gp[0] + t2[1]*gp[1]

                # Body force at Gauss point
                pi2 = np.pi**2
                spx = np.sin(np.pi * x_g); cpx = np.cos(np.pi * x_g)
                spy = np.sin(np.pi * y_g); cpy = np.cos(np.pi * y_g)
                fx_g = E_el * C_factor * pi2 * spx * spy * (1.0 - nu)
                fy_g = E_el * C_factor * pi2 * cpx * cpy * (1.0 - nu)

                # Evaluate basis functions at Gauss point
                bv = _eval_p2_vector_basis(x_g, y_g, xc, yc, h)  # (12, 2)
                for alpha in range(n_polys):
                    f_poly[alpha] += 2.0 * tri_area * gw * (
                        fx_g * bv[alpha, 0] + fy_g * bv[alpha, 1])

        # Map to DOF space: F_element = proj^T @ f_poly
        f_el = projector.T @ f_poly

        # Assemble into global force vector
        for i in range(n_total):
            f[2 * el_int[i]]     += f_el[2 * i]
            f[2 * el_int[i] + 1] += f_el[2 * i + 1]

    return f


def _solve_with_body_force_p2(vertices, elements, E_field, nu,
                               bc_fixed_dofs, bc_vals, f_global):
    """Solve P2 VEM with given body force vector."""
    n_nodes = vertices.shape[0]
    n_dofs = 2 * n_nodes

    # Assemble P2 stiffness
    K = _assemble_p2_stiffness_sparse(vertices, elements, E_field, nu)

    u = np.zeros(n_dofs)
    bc_set = set(bc_fixed_dofs)
    internal = np.array([i for i in range(n_dofs) if i not in bc_set])

    u[bc_fixed_dofs] = bc_vals
    rhs = f_global.copy()
    rhs -= K[:, bc_fixed_dofs].toarray() @ bc_vals

    K_ii = K[np.ix_(internal, internal)]
    u[internal] = sp.linalg.spsolve(K_ii, rhs[internal])
    return u


def _assemble_p2_stiffness_sparse(vertices, elements, E_field, nu,
                                   stabilization_alpha=1.0):
    """Assemble P2 stiffness matrix only (no solve), returns sparse CSR."""
    n_nodes = vertices.shape[0]
    n_dofs = 2 * n_nodes
    n_polys = 12

    row_idx, col_idx, val_data = [], [], []

    for el_id in range(len(elements)):
        el_nodes = elements[el_id].astype(int)
        n_total = len(el_nodes)
        n_v = n_total // 2
        n_el_dofs = 2 * n_total

        vert_ids = el_nodes[:n_v]
        mid_ids = el_nodes[n_v:]
        verts = vertices[vert_ids]
        mids = vertices[mid_ids]
        all_coords = vertices[el_nodes]

        E_el = E_field[el_id] if hasattr(E_field, '__len__') else E_field
        C = (E_el / (1.0 - nu**2)) * np.array([
            [1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, (1.0 - nu) / 2.0]])

        area_comp = (verts[:, 0] * np.roll(verts[:, 1], -1)
                     - np.roll(verts[:, 0], -1) * verts[:, 1])
        area = 0.5 * abs(np.sum(area_comp))
        if area < 1e-30:
            continue
        centroid = np.sum(
            (np.roll(verts, -1, axis=0) + verts) * area_comp[:, None],
            axis=0) / (6.0 * area)
        h = max(np.linalg.norm(verts[i] - verts[j])
                for i in range(n_v) for j in range(i + 1, n_v))
        if h < 1e-30:
            continue
        xc, yc = centroid

        # D matrix
        D = np.zeros((n_el_dofs, n_polys))
        for i in range(n_total):
            x_i, y_i = all_coords[i]
            basis_vals = _eval_p2_vector_basis(x_i, y_i, xc, yc, h)
            D[2 * i,     :] = basis_vals[:, 0]
            D[2 * i + 1, :] = basis_vals[:, 1]

        # B matrix
        B = np.zeros((n_polys, n_el_dofs))
        for i in range(n_total):
            B[0, 2 * i]     = 1.0 / n_total
            B[1, 2 * i + 1] = 1.0 / n_total

        # Rotation row (Simpson)
        for i in range(n_v):
            j = (i + 1) % n_v
            nx = verts[j][1] - verts[i][1]
            ny = verts[i][0] - verts[j][0]
            edge_len = np.sqrt(nx**2 + ny**2)
            if edge_len < 1e-30:
                continue
            nx_u, ny_u = nx / edge_len, ny / edge_len
            w_v = edge_len / 6.0
            w_m = 4.0 * edge_len / 6.0
            B[2, 2 * i]             += -ny_u * w_v
            B[2, 2 * i + 1]         +=  nx_u * w_v
            B[2, 2 * (n_v + i)]     += -ny_u * w_m
            B[2, 2 * (n_v + i) + 1] +=  nx_u * w_m
            B[2, 2 * j]             += -ny_u * w_v
            B[2, 2 * j + 1]         +=  nx_u * w_v
        B[2, :] /= (2.0 * area)

        # Strain rows (boundary integral + volume correction)
        for i in range(n_v):
            j = (i + 1) % n_v
            nx = verts[j][1] - verts[i][1]
            ny = verts[i][0] - verts[j][0]
            edge_len = np.sqrt(nx**2 + ny**2)
            if edge_len < 1e-30:
                continue
            nx_u, ny_u = nx / edge_len, ny / edge_len
            pts = [verts[i], mids[i], verts[j]]
            dof_indices = [i, n_v + i, j]
            sw = [edge_len / 6.0, 4.0 * edge_len / 6.0, edge_len / 6.0]
            for pt_idx in range(3):
                px, py = pts[pt_idx]
                node_idx = dof_indices[pt_idx]
                w = sw[pt_idx]
                strain_all = _eval_p2_strain(px, py, xc, yc, h)
                for alpha in range(9):
                    poly_idx = 3 + alpha
                    eps_voigt = strain_all[poly_idx]
                    sigma = C @ eps_voigt
                    tx = sigma[0] * nx_u + sigma[2] * ny_u
                    ty = sigma[2] * nx_u + sigma[1] * ny_u
                    B[poly_idx, 2 * node_idx]     += w * tx
                    B[poly_idx, 2 * node_idx + 1] += w * ty

        # Volume correction for quadratic modes
        div_sigma = _compute_div_sigma(C, h)
        for alpha_idx in range(6, 12):
            dx, dy = div_sigma[alpha_idx]
            if abs(dx) + abs(dy) > 1e-30:
                for i in range(n_total):
                    B[alpha_idx, 2 * i]     -= dx * area / n_total
                    B[alpha_idx, 2 * i + 1] -= dy * area / n_total

        G = B @ D
        cond = np.linalg.cond(G)
        if cond > 1e12:
            G += 1e-10 * np.eye(n_polys)
        projector = np.linalg.solve(G, B)
        a_K_matrix = _compute_strain_energy_matrix(verts, xc, yc, h, C)
        K_cons = projector.T @ a_K_matrix @ projector
        I_minus_PiD = np.eye(n_el_dofs) - D @ projector
        trace_k = np.trace(K_cons)
        stab_param = stabilization_alpha * trace_k / n_el_dofs if trace_k > 0 else E_el
        K_local = K_cons + stab_param * (I_minus_PiD.T @ I_minus_PiD)

        gdofs = np.zeros(n_el_dofs, dtype=int)
        for i in range(n_total):
            gdofs[2 * i]     = 2 * el_nodes[i]
            gdofs[2 * i + 1] = 2 * el_nodes[i] + 1
        ii, jj = np.meshgrid(gdofs, gdofs, indexing='ij')
        row_idx.append(ii.ravel())
        col_idx.append(jj.ravel())
        val_data.append(K_local.ravel())

    row_idx = np.concatenate(row_idx)
    col_idx = np.concatenate(col_idx)
    val_data = np.concatenate(val_data)
    return sp.csr_matrix((val_data, (row_idx, col_idx)), shape=(n_dofs, n_dofs))


def _compute_errors_p1(vertices, elements, u, exact_ux, exact_uy,
                        exact_eps_xx, exact_eps_yy, exact_2eps_xy):
    """Compute L2 and H1 errors for P1 VEM solution."""
    ux = u[0::2]
    uy = u[1::2]

    err_L2_sq = 0.0
    err_H1_sq = 0.0
    norm_L2_sq = 0.0
    norm_H1_sq = 0.0

    for el_id, el in enumerate(elements):
        el_int = el.astype(int)
        n_v = len(el_int)
        verts = vertices[el_int]

        area_comp = (verts[:, 0] * np.roll(verts[:, 1], -1)
                     - np.roll(verts[:, 0], -1) * verts[:, 1])
        area = 0.5 * abs(np.sum(area_comp))
        centroid = verts.mean(axis=0)

        # L2 error at centroid (1-point quadrature)
        cx, cy = centroid
        ux_h = np.mean(ux[el_int])
        uy_h = np.mean(uy[el_int])
        ux_ex = exact_ux(cx, cy)
        uy_ex = exact_uy(cx, cy)

        err_L2_sq += area * ((ux_h - ux_ex)**2 + (uy_h - uy_ex)**2)
        norm_L2_sq += area * (ux_ex**2 + uy_ex**2)

        # H1 error: strain error
        # Compute numerical strain via boundary integral
        eps_h = np.zeros(3)
        vertex_normals = np.zeros((n_v, 2))
        for i in range(n_v):
            j = (i + 1) % n_v
            nx = verts[j, 1] - verts[i, 1]
            ny = verts[i, 0] - verts[j, 0]
            ux_avg = 0.5 * (ux[el_int[i]] + ux[el_int[j]])
            uy_avg = 0.5 * (uy[el_int[i]] + uy[el_int[j]])
            eps_h[0] += ux_avg * nx
            eps_h[1] += uy_avg * ny
            eps_h[2] += ux_avg * ny + uy_avg * nx
        eps_h /= area

        eps_ex = np.array([
            exact_eps_xx(cx, cy),
            exact_eps_yy(cx, cy),
            exact_2eps_xy(cx, cy),
        ])

        err_H1_sq += area * np.sum((eps_h - eps_ex)**2)
        norm_H1_sq += area * np.sum(eps_ex**2)

    err_L2 = np.sqrt(err_L2_sq / (norm_L2_sq + 1e-30))
    err_H1 = np.sqrt(err_H1_sq / (norm_H1_sq + 1e-30))
    return err_L2, err_H1


def _compute_errors_p2(vertices, elements, u, exact_ux, exact_uy,
                        exact_eps_xx, exact_eps_yy, exact_2eps_xy):
    """Compute L2 and H1 errors for P2 VEM solution."""
    ux = u[0::2]
    uy = u[1::2]

    err_L2_sq = 0.0
    err_H1_sq = 0.0
    norm_L2_sq = 0.0
    norm_H1_sq = 0.0

    for el_id, el in enumerate(elements):
        el_int = el.astype(int)
        n_total = len(el_int)
        n_v = n_total // 2
        vert_ids = el_int[:n_v]
        mid_ids = el_int[n_v:]
        verts = vertices[vert_ids]
        mids = vertices[mid_ids]

        area_comp = (verts[:, 0] * np.roll(verts[:, 1], -1)
                     - np.roll(verts[:, 0], -1) * verts[:, 1])
        area = 0.5 * abs(np.sum(area_comp))
        centroid = verts.mean(axis=0)

        # L2: average over all nodes (weighted by area)
        cx, cy = centroid
        ux_h = np.mean(ux[el_int])
        uy_h = np.mean(uy[el_int])
        ux_ex = exact_ux(cx, cy)
        uy_ex = exact_uy(cx, cy)

        err_L2_sq += area * ((ux_h - ux_ex)**2 + (uy_h - uy_ex)**2)
        norm_L2_sq += area * (ux_ex**2 + uy_ex**2)

        # H1: strain error via Simpson boundary integral
        eps_h = np.zeros(3)
        for i in range(n_v):
            j = (i + 1) % n_v
            vi = verts[i]
            vj = verts[j]
            nx = vj[1] - vi[1]
            ny = vi[0] - vj[0]
            edge_len = np.sqrt(nx**2 + ny**2)
            if edge_len < 1e-30:
                continue
            nx_u = nx / edge_len
            ny_u = ny / edge_len
            w1 = edge_len / 6.0
            w4 = 4.0 * edge_len / 6.0
            w2 = edge_len / 6.0
            vi_id = vert_ids[i]
            vj_id = vert_ids[j]
            mi_id = mid_ids[i]
            eps_h[0] += nx_u * (w1*ux[vi_id] + w4*ux[mi_id] + w2*ux[vj_id])
            eps_h[1] += ny_u * (w1*uy[vi_id] + w4*uy[mi_id] + w2*uy[vj_id])
            eps_h[2] += ny_u * (w1*ux[vi_id] + w4*ux[mi_id] + w2*ux[vj_id])
            eps_h[2] += nx_u * (w1*uy[vi_id] + w4*uy[mi_id] + w2*uy[vj_id])
        eps_h /= area

        eps_ex = np.array([
            exact_eps_xx(cx, cy),
            exact_eps_yy(cx, cy),
            exact_2eps_xy(cx, cy),
        ])

        err_H1_sq += area * np.sum((eps_h - eps_ex)**2)
        norm_H1_sq += area * np.sum(eps_ex**2)

    err_L2 = np.sqrt(err_L2_sq / (norm_L2_sq + 1e-30))
    err_H1 = np.sqrt(err_H1_sq / (norm_H1_sq + 1e-30))
    return err_L2, err_H1


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import os

    print("P2 VEM for 2D Linear Elasticity")
    print("=" * 60)

    # Print basis info
    info = p2_polynomial_basis_2d()
    print(f"\nP2 basis: {info['n_polys']} polynomials "
          f"({info['n_rigid']} rigid body + {info['n_strain']} strain modes)")
    print(f"Scalar monomials: {info['scalar_monomials']}")

    # Demo 1: P2 vs P1 comparison
    demo_p2_vs_p1()

    # Demo 2: Convergence study
    convergence_p2_vs_p1()

    print("\n" + "=" * 60)
    print("All P2 VEM demos complete.")
    print("=" * 60)
