#!/usr/bin/env python3
"""
vem_viscoelastic.py -- Production viscoelastic VEM (A3: VE-VEM)
================================================================

Standard sequential time-stepping VEM with Simo 1987 exponential integrator
for SLS (Standard Linear Solid / Zener model) internal variables.

SLS Model (Zener):
    sigma(t) = C_inf . epsilon + h(t)
    h_{n+1}  = exp(-dt/tau) . h_n  +  gamma . C_1 . (epsilon_{n+1} - epsilon_n)
    gamma    = (tau/dt) . (1 - exp(-dt/tau))

Algorithmic tangent:
    C_alg = C_inf + gamma . C_1

DI-dependent parameters:
    E_inf(DI), E_1(DI), tau(DI), eta(DI) from material_models.py

Spatial discretization: P1 VEM from vem_elasticity.py (constant strain per element).
Time integration: Simo 1987 unconditionally stable exponential integrator.

References:
    Simo & Hughes (1998) "Computational Inelasticity", Ch. 10
    Simo (1987) "On a fully three-dimensional finite-strain viscoelastic damage model"
    Beirao da Veiga et al. (2013) "Basic principles of VEM"
"""

import numpy as np
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from scipy.spatial import Voronoi
from pathlib import Path


# ---------------------------------------------------------------------------
# Mesh utilities (self-contained, from vem_p2_elasticity)
# ---------------------------------------------------------------------------

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


def generate_voronoi_mesh(n_cells, domain=(0, 1, 0, 1), seed=42):
    """
    Generate a clipped Voronoi mesh on a rectangular domain.

    Returns
    -------
    vertices : (N, 2)
    elements : list of int arrays (0-based, CCW ordered)
    boundary : array of boundary node indices
    """
    rng = np.random.RandomState(seed)
    xmin, xmax, ymin, ymax = domain
    Lx, Ly = xmax - xmin, ymax - ymin

    pts = rng.rand(n_cells, 2) * [Lx, Ly] + [xmin, ymin]

    pts_mirror = np.vstack([
        pts,
        np.column_stack([2 * xmin - pts[:, 0], pts[:, 1]]),
        np.column_stack([2 * xmax - pts[:, 0], pts[:, 1]]),
        np.column_stack([pts[:, 0], 2 * ymin - pts[:, 1]]),
        np.column_stack([pts[:, 0], 2 * ymax - pts[:, 1]]),
    ])

    vor = Voronoi(pts_mirror)

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
        clipped = _clip_polygon_to_box(poly, xmin, xmax, ymin, ymax)
        if clipped is None or len(clipped) < 3:
            continue

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

    boundary = []
    for i, v in enumerate(vertices):
        if (abs(v[0] - xmin) < tol or abs(v[0] - xmax) < tol or
                abs(v[1] - ymin) < tol or abs(v[1] - ymax) < tol):
            boundary.append(i)
    boundary = np.array(boundary, dtype=int)

    return vertices, elements, boundary


# ---------------------------------------------------------------------------
# SLS parameter computation from DI field
# ---------------------------------------------------------------------------

def sls_params_from_di(DI_field, E_max=1000.0, E_min=10.0,
                       tau_max=60.0, tau_min=2.0, n_hill=2,
                       ratio_min=2.0, ratio_max=5.0, tau_exp=1.5):
    """
    Compute per-element SLS parameters from DI field.

    Parameters
    ----------
    DI_field : (N_el,) array -- dysbiosis index per element, in [0, 1]
    E_max : float -- commensal equilibrium modulus [Pa]
    E_min : float -- dysbiotic equilibrium modulus [Pa]
    tau_max : float -- commensal relaxation time [s]
    tau_min : float -- dysbiotic relaxation time [s]
    n_hill : float -- power-law exponent for E_inf(DI)
    ratio_min : float -- E_0/E_inf ratio for commensal (DI=0)
    ratio_max : float -- E_0/E_inf ratio for dysbiotic (DI=1)
    tau_exp : float -- power-law exponent for tau(DI)

    Returns
    -------
    dict with keys: E_inf, E_0, E_1, tau, eta (each (N_el,) arrays)
    """
    di = np.asarray(DI_field, dtype=np.float64)
    r = np.clip(di, 0.0, 1.0)

    # Equilibrium modulus: power-law decay with DI
    E_inf = E_max * (1.0 - r) ** n_hill + E_min * r

    # E_0/E_inf ratio increases with DI (more Maxwell arm contribution)
    ratio = ratio_min + (ratio_max - ratio_min) * r
    E_0 = E_inf * ratio
    E_1 = E_0 - E_inf  # Maxwell arm spring

    # Relaxation time: commensal slow, dysbiotic fast
    tau = tau_max * (1.0 - r) ** tau_exp + tau_min * r

    # Viscosity
    eta = E_1 * tau

    return {"E_inf": E_inf, "E_0": E_0, "E_1": E_1, "tau": tau, "eta": eta}


# ---------------------------------------------------------------------------
# Plane-stress constitutive matrix (Voigt: [sigma_xx, sigma_yy, sigma_xy])
# ---------------------------------------------------------------------------

def _plane_stress_C(E, nu):
    """Return 3x3 plane-stress constitutive matrix for given E, nu."""
    fac = E / (1.0 - nu ** 2)
    return fac * np.array([
        [1.0, nu,  0.0],
        [nu,  1.0, 0.0],
        [0.0, 0.0, (1.0 - nu) / 2.0],
    ])


# ---------------------------------------------------------------------------
# VEM element computation: projector + strain operator
# ---------------------------------------------------------------------------

def _compute_element_vem(verts, nu):
    """
    Compute VEM element geometry and projection operators.

    For P1 VEM the strain is constant per element.

    Parameters
    ----------
    verts : (n_v, 2) -- element vertex coordinates
    nu : float -- Poisson's ratio

    Returns
    -------
    dict with: area, centroid, h, D, B, G, projector, strain_proj, etc.
    """
    n_v = len(verts)
    n_el_dofs = 2 * n_v
    n_polys = 6

    # Geometry
    area_comp = (verts[:, 0] * np.roll(verts[:, 1], -1)
                 - np.roll(verts[:, 0], -1) * verts[:, 1])
    area = 0.5 * abs(np.sum(area_comp))
    centroid = np.sum(
        (np.roll(verts, -1, axis=0) + verts) * area_comp[:, None],
        axis=0) / (6.0 * area)

    h = max(np.linalg.norm(verts[i] - verts[j])
            for i in range(n_v) for j in range(i + 1, n_v))

    xc, yc = centroid

    # D matrix (n_el_dofs x 6)
    D = np.zeros((n_el_dofs, n_polys))
    for i in range(n_v):
        dx = (verts[i, 0] - xc) / h
        dy = (verts[i, 1] - yc) / h
        D[2 * i,     :] = [1.0, 0.0, -dy, dx, 0.0, dy]
        D[2 * i + 1, :] = [0.0, 1.0, dx,  0.0, dy, dx]

    # B matrix (6 x n_el_dofs) -- uses a reference C with E=1
    C_ref = _plane_stress_C(1.0, nu)

    B = np.zeros((n_polys, n_el_dofs))

    # Vertex normals
    vertex_normals = np.zeros((n_v, 2))
    for i in range(n_v):
        prev_v = verts[(i - 1) % n_v]
        next_v = verts[(i + 1) % n_v]
        vertex_normals[i] = [next_v[1] - prev_v[1],
                             prev_v[0] - next_v[0]]

    # Rows 0-1: translations
    for i in range(n_v):
        B[0, 2 * i]     = 1.0 / n_v
        B[1, 2 * i + 1] = 1.0 / n_v

    # Row 2: rigid rotation
    for i in range(n_v):
        B[2, 2 * i]     = -vertex_normals[i, 1] / (4.0 * area)
        B[2, 2 * i + 1] =  vertex_normals[i, 0] / (4.0 * area)

    # Rows 3-5: strain modes
    strain_basis = np.array([
        [1.0 / h, 0.0,     0.0],
        [0.0,     1.0 / h, 0.0],
        [0.0,     0.0,     2.0 / h],
    ])

    for i in range(n_v):
        vn = vertex_normals[i]
        for alpha in range(3):
            sigma = C_ref @ strain_basis[alpha]
            tx = sigma[0] * vn[0] + sigma[2] * vn[1]
            ty = sigma[2] * vn[0] + sigma[1] * vn[1]
            B[3 + alpha, 2 * i]     += 0.5 * tx
            B[3 + alpha, 2 * i + 1] += 0.5 * ty

    # Projector: Pi = G^{-1} B, where G = B D
    G = B @ D
    projector = np.linalg.solve(G, B)  # (6, n_el_dofs)

    # Strain projector: maps DOFs -> constant Voigt strain (3,)
    # epsilon_xx = (1/h) * (projector[3,:] @ u_el)
    # epsilon_yy = (1/h) * (projector[4,:] @ u_el)
    # gamma_xy   = (2/h) * (projector[5,:] @ u_el)
    strain_proj = np.zeros((3, n_el_dofs))
    strain_proj[0, :] = (1.0 / h) * projector[3, :]
    strain_proj[1, :] = (1.0 / h) * projector[4, :]
    strain_proj[2, :] = (2.0 / h) * projector[5, :]

    return {
        "area": area,
        "centroid": centroid,
        "h": h,
        "n_v": n_v,
        "n_el_dofs": n_el_dofs,
        "D": D,
        "B": B,
        "G": G,
        "projector": projector,
        "strain_proj": strain_proj,
        "vertex_normals": vertex_normals,
    }


# ---------------------------------------------------------------------------
# Core assembly for one time step (algorithmic tangent + pseudo-load)
# ---------------------------------------------------------------------------

def _assemble_viscoelastic_step(
    vertices, elements, elem_data, C_alg_all, h_star_all,
    bc_fixed_dofs, bc_vals, F_ext,
    stabilization_alpha=1.0,
):
    """
    Assemble and solve one viscoelastic time step.

    The system is:
        K(C_alg) * u = F_ext - F_h
    where F_h is the pseudo-load from the frozen internal variable h_star.

    Parameters
    ----------
    C_alg_all : (N_el, 3, 3) -- algorithmic tangent per element
    h_star_all : (N_el, 3) -- effective internal variable (Voigt) per element

    Returns
    -------
    u : (n_dofs,) -- displacement solution
    """
    n_nodes = vertices.shape[0]
    n_dofs = 2 * n_nodes

    row_idx = []
    col_idx = []
    val_data = []
    F_h = np.zeros(n_dofs)

    for el_id in range(len(elements)):
        vert_ids = elements[el_id].astype(int)
        ed = elem_data[el_id]
        n_v = ed["n_v"]
        n_el_dofs = ed["n_el_dofs"]
        area = ed["area"]
        C_alg = C_alg_all[el_id]
        h_star = h_star_all[el_id]

        projector = ed["projector"]
        strain_proj = ed["strain_proj"]  # (3, n_el_dofs)
        D = ed["D"]

        # Consistency stiffness: K_cons = area * strain_proj^T @ C_alg @ strain_proj
        K_cons = area * strain_proj.T @ C_alg @ strain_proj

        # Stabilization (Wriggers recipe)
        I_minus_PiD = np.eye(n_el_dofs) - D @ projector
        trace_k = np.trace(K_cons)
        stab_param = stabilization_alpha * trace_k / n_el_dofs if trace_k > 0 else 1.0
        K_stab = stab_param * (I_minus_PiD.T @ I_minus_PiD)

        K_local = K_cons + K_stab

        # Internal variable pseudo-load: f_h = area * strain_proj^T @ h_star
        f_h_local = area * strain_proj.T @ h_star

        # Map to global DOFs
        gdofs = np.zeros(n_el_dofs, dtype=int)
        for i in range(n_v):
            gdofs[2 * i]     = 2 * vert_ids[i]
            gdofs[2 * i + 1] = 2 * vert_ids[i] + 1

        # Sparse assembly
        ii, jj = np.meshgrid(gdofs, gdofs, indexing='ij')
        row_idx.append(ii.ravel())
        col_idx.append(jj.ravel())
        val_data.append(K_local.ravel())

        # Pseudo-load assembly
        F_h[gdofs] += f_h_local

    row_idx = np.concatenate(row_idx)
    col_idx = np.concatenate(col_idx)
    val_data = np.concatenate(val_data)
    K_global = sp.csr_matrix((val_data, (row_idx, col_idx)), shape=(n_dofs, n_dofs))

    # RHS
    F_rhs = F_ext - F_h

    # Solve with BCs (penalty-free direct elimination)
    u = np.zeros(n_dofs)
    bc_set = set(bc_fixed_dofs)
    internal = np.array([i for i in range(n_dofs) if i not in bc_set])

    u[bc_fixed_dofs] = bc_vals
    F_rhs -= K_global[:, bc_fixed_dofs].toarray() @ bc_vals

    K_ii = K_global[np.ix_(internal, internal)]
    u[internal] = sp.linalg.spsolve(K_ii, F_rhs[internal])

    return u


# ---------------------------------------------------------------------------
# Main viscoelastic VEM solver
# ---------------------------------------------------------------------------

def vem_viscoelastic_sls(vertices, elements, DI_field, nu, bc_fixed_dofs, bc_vals,
                          t_array, load_dofs=None, load_vals_func=None,
                          E_max=1000.0, E_min=10.0, tau_max=60.0, tau_min=2.0,
                          n_hill=2, ratio_min=2.0, ratio_max=5.0, tau_exp=1.5,
                          stabilization_alpha=1.0):
    """
    Time-stepping VEM for SLS viscoelasticity with DI-dependent parameters.

    Uses Simo 1987 exponential integrator for internal variable update.

    Parameters
    ----------
    vertices : (N_nodes, 2) -- node coordinates
    elements : list of int arrays -- polygon connectivity (0-based, CCW)
    DI_field : (N_el,) array -- dysbiosis index per element, in [0, 1]
    nu : float -- Poisson's ratio
    bc_fixed_dofs : array of int -- constrained DOF indices
    bc_vals : array -- prescribed displacement values (constant over time)
    t_array : (N_t,) array -- time points (t_array[0] should be 0)
    load_dofs : array of int -- DOFs with applied loads (None for displacement-only)
    load_vals_func : callable(t) -> array of load values, or None
    E_max, E_min : float -- E_inf bounds [Pa]
    tau_max, tau_min : float -- relaxation time bounds [s]
    n_hill : float -- power-law exponent for E_inf
    ratio_min, ratio_max : float -- E_0/E_inf ratio range
    tau_exp : float -- power-law exponent for tau
    stabilization_alpha : float -- VEM stabilization parameter

    Returns
    -------
    u_history : (N_t, 2*N_nodes) -- displacement at each time step
    sigma_history : (N_t, N_el, 3) -- Voigt stress per element
    h_history : (N_t, N_el, 3) -- internal variable history
    """
    n_nodes = vertices.shape[0]
    n_dofs = 2 * n_nodes
    n_el = len(elements)
    n_t = len(t_array)

    # Compute SLS parameters per element
    params = sls_params_from_di(
        DI_field, E_max=E_max, E_min=E_min,
        tau_max=tau_max, tau_min=tau_min, n_hill=n_hill,
        ratio_min=ratio_min, ratio_max=ratio_max, tau_exp=tau_exp,
    )
    E_inf = params["E_inf"]  # (N_el,)
    E_1 = params["E_1"]      # (N_el,)
    tau = params["tau"]       # (N_el,)

    # Precompute per-element constitutive matrices
    C_inf_all = np.zeros((n_el, 3, 3))
    C_1_all = np.zeros((n_el, 3, 3))
    for k in range(n_el):
        C_inf_all[k] = _plane_stress_C(E_inf[k], nu)
        C_1_all[k] = _plane_stress_C(E_1[k], nu)

    # Precompute VEM element data (geometry-dependent, time-independent)
    elem_data = []
    for el_id in range(n_el):
        vert_ids = elements[el_id].astype(int)
        verts = vertices[vert_ids]
        elem_data.append(_compute_element_vem(verts, nu))

    # Storage
    u_history = np.zeros((n_t, n_dofs))
    sigma_history = np.zeros((n_t, n_el, 3))
    h_history = np.zeros((n_t, n_el, 3))

    # State variables
    h_all = np.zeros((n_el, 3))       # current internal variable per element
    eps_prev = np.zeros((n_el, 3))    # previous strain per element

    for ti in range(n_t):
        t = t_array[ti]
        dt = t_array[ti] - t_array[ti - 1] if ti > 0 else 0.0

        # Simo exponential integrator coefficients
        if ti == 0:
            gamma = np.ones(n_el)
            exp_dt = np.zeros(n_el)
        elif dt > 1e-15:
            exp_dt = np.exp(-dt / tau)
            gamma = (tau / dt) * (1.0 - exp_dt)
        else:
            exp_dt = np.ones(n_el)
            gamma = np.ones(n_el)

        # Algorithmic tangent: C_alg = C_inf + gamma * C_1
        C_alg_all = C_inf_all + gamma[:, None, None] * C_1_all

        # Effective history stress h_star = exp(-dt/tau)*h_old - gamma*C_1*eps_old
        if ti == 0:
            h_star = np.zeros((n_el, 3))
        else:
            h_star = np.zeros((n_el, 3))
            for k in range(n_el):
                h_star[k] = exp_dt[k] * h_all[k] - gamma[k] * C_1_all[k] @ eps_prev[k]

        # External load
        F_ext = np.zeros(n_dofs)
        if load_dofs is not None and load_vals_func is not None:
            F_ext[load_dofs] = load_vals_func(t)

        # Assemble and solve
        u = _assemble_viscoelastic_step(
            vertices, elements, elem_data,
            C_alg_all, h_star,
            bc_fixed_dofs, bc_vals, F_ext,
            stabilization_alpha=stabilization_alpha,
        )

        # Post-process: compute strain and update internal variables
        for k in range(n_el):
            vert_ids = elements[k].astype(int)
            ed = elem_data[k]
            n_v = ed["n_v"]

            gdofs_local = np.zeros(ed["n_el_dofs"], dtype=int)
            for i in range(n_v):
                gdofs_local[2 * i]     = 2 * vert_ids[i]
                gdofs_local[2 * i + 1] = 2 * vert_ids[i] + 1

            u_el = u[gdofs_local]
            eps_new = ed["strain_proj"] @ u_el  # (3,) Voigt strain

            # Update internal variable via Simo exponential rule
            if ti == 0:
                h_all[k] = C_1_all[k] @ eps_new
            else:
                h_all[k] = exp_dt[k] * h_all[k] + gamma[k] * C_1_all[k] @ (eps_new - eps_prev[k])

            # Total stress: sigma = C_inf * eps + h
            sigma_history[ti, k] = C_inf_all[k] @ eps_new + h_all[k]
            eps_prev[k] = eps_new.copy()

        u_history[ti] = u
        h_history[ti] = h_all.copy()

    return u_history, sigma_history, h_history


# ---------------------------------------------------------------------------
# Validation: uniform SLS against analytical solution
# ---------------------------------------------------------------------------

def validate_sls_relaxation():
    """
    Validate against analytical SLS solution for uniform material.

    For uniform DI with step strain epsilon_0 applied via prescribed displacement:
        sigma(t) = [E_inf + E_1 * exp(-t/tau)] * epsilon_0

    Shows max relative error < 1% for dt < tau/10.
    """
    print("=" * 60)
    print("Validation: VEM-SLS vs analytical stress relaxation")
    print("=" * 60)

    # Setup: refined mesh, uniform material
    n_cells = 64
    vertices, elements, boundary = generate_voronoi_mesh(n_cells, seed=123)
    n_el = len(elements)
    n_nodes = vertices.shape[0]

    # Uniform DI -> uniform SLS parameters
    DI_val = 0.4
    DI_field = np.full(n_el, DI_val)
    nu = 0.3

    params = sls_params_from_di(DI_field)
    E_inf_val = params["E_inf"][0]
    E_1_val = params["E_1"][0]
    tau_val = params["tau"][0]

    print(f"  DI = {DI_val}")
    print(f"  E_inf = {E_inf_val:.2f} Pa")
    print(f"  E_1   = {E_1_val:.2f} Pa")
    print(f"  E_0   = {E_inf_val + E_1_val:.2f} Pa")
    print(f"  tau   = {tau_val:.2f} s")

    # Laterally confined step displacement: u_x=0 everywhere, fix bottom, prescribe u_y on top
    # This gives perfectly uniform strain eps_yy = eps_0, eps_xx = 0
    # sigma_yy(t) = [E_inf + E_1*exp(-t/tau)] / (1-nu^2) * eps_0 for plane stress confined
    eps_0 = 0.01
    tol = 1e-6

    bottom = np.where(vertices[:, 1] < tol)[0]
    top = np.where(vertices[:, 1] > 1.0 - tol)[0]
    all_nodes = np.arange(n_nodes)

    bc_dofs = np.concatenate([
        2 * all_nodes,       # u_x = 0 for ALL nodes (laterally confined)
        2 * bottom + 1,      # bottom u_y = 0
        2 * top + 1,         # top u_y = eps_0
    ])
    bc_vals = np.concatenate([
        np.zeros(len(all_nodes)),
        np.zeros(len(bottom)),
        np.full(len(top), eps_0),
    ])

    # Remove duplicates
    bc_dofs, unique_idx = np.unique(bc_dofs, return_index=True)
    bc_vals = bc_vals[unique_idx]

    # Time array: fine enough for good accuracy
    t_array = np.concatenate([
        [0.0],
        np.linspace(tau_val / 20, 4 * tau_val, 40),
    ])

    u_hist, sigma_hist, h_hist = vem_viscoelastic_sls(
        vertices, elements, DI_field, nu,
        bc_dofs, bc_vals, t_array,
    )

    # Analytical: for laterally confined plane stress (eps_xx=0, eps_yy=eps_0):
    #   sigma_yy = C_22(t) * eps_0 where C_22 = E(t)/(1-nu^2)
    # So sigma_yy(t) = [E_inf + E_1*exp(-t/tau)] / (1-nu^2) * eps_0
    confined_fac = 1.0 / (1.0 - nu ** 2)
    sigma_analytical = (E_inf_val + E_1_val * np.exp(-t_array / tau_val)) * confined_fac * eps_0

    # Average sigma_yy across elements
    sigma_yy_avg = sigma_hist[:, :, 1].mean(axis=1)

    print(f"\n  {'t [s]':>8} {'sigma_yy VEM':>14} {'sigma_yy ana':>14} {'rel_err':>10}")
    print("  " + "-" * 50)

    max_rel_err = 0.0
    for ti in range(len(t_array)):
        err = abs(sigma_yy_avg[ti] - sigma_analytical[ti])
        rel = err / (abs(sigma_analytical[ti]) + 1e-12)
        max_rel_err = max(max_rel_err, rel)
        if ti % 5 == 0 or ti == len(t_array) - 1:
            print(f"  {t_array[ti]:8.2f} {sigma_yy_avg[ti]:14.6f} "
                  f"{sigma_analytical[ti]:14.6f} {rel:10.2e}")

    print(f"\n  Max relative error: {max_rel_err:.4e}")

    # Long-time limit
    sigma_inf_vem = sigma_yy_avg[-1]
    sigma_inf_ana = E_inf_val * confined_fac * eps_0
    print(f"  Long-time sigma: VEM={sigma_inf_vem:.6f}, analytical={sigma_inf_ana:.6f}")

    if max_rel_err < 0.01:
        print("  PASSED (< 1%)")
    else:
        print(f"  WARNING: error {max_rel_err:.2e} > 1% -- check mesh/dt refinement")

    return max_rel_err


# ---------------------------------------------------------------------------
# Demo: stress relaxation on Voronoi mesh with DI gradient
# ---------------------------------------------------------------------------

def demo_ve_vem():
    """
    Demo: Stress relaxation on Voronoi mesh with DI gradient.

    Setup:
    - 30-cell Voronoi mesh, domain [0,1]x[0,1]
    - DI gradient: 0.1 (left, commensal) -> 0.8 (right, dysbiotic)
    - Bottom fixed, top: step displacement eps_0=0.01 at t=0
    - Time: 0 to 120s (covers ~2*tau for both commensal and dysbiotic)

    Output: 2x3 figure
    - (a) DI field
    - (b) E_inf(DI) field
    - (c) tau(DI) field
    - (d) sigma_xx at t=0 (initial)
    - (e) sigma_yy at t=60s (mid-relaxation)
    - (f) Stress relaxation curves at 3 points (left/center/right)

    Saved to results/vem_viscoelastic_demo.png
    """
    print("=" * 60)
    print("Demo: VE-VEM Stress Relaxation with DI Gradient")
    print("=" * 60)

    # Mesh
    n_cells = 30
    vertices, elements, boundary = generate_voronoi_mesh(n_cells, seed=42)
    n_el = len(elements)
    n_nodes = vertices.shape[0]
    print(f"  Mesh: {n_el} elements, {n_nodes} nodes")

    # DI gradient: left (commensal) -> right (dysbiotic)
    DI_field = np.zeros(n_el)
    centroids = np.zeros((n_el, 2))
    for k in range(n_el):
        vert_ids = elements[k].astype(int)
        centroids[k] = vertices[vert_ids].mean(axis=0)
        DI_field[k] = 0.1 + 0.7 * centroids[k, 0]

    # SLS parameters
    params = sls_params_from_di(DI_field)
    E_inf = params["E_inf"]
    E_1 = params["E_1"]
    tau_arr = params["tau"]

    print(f"  DI range: [{DI_field.min():.2f}, {DI_field.max():.2f}]")
    print(f"  E_inf range: [{E_inf.min():.1f}, {E_inf.max():.1f}] Pa")
    print(f"  tau range: [{tau_arr.min():.1f}, {tau_arr.max():.1f}] s")

    # BCs
    nu = 0.3
    eps_0 = 0.01
    tol = 1e-6

    bottom = np.where(vertices[:, 1] < tol)[0]
    top = np.where(vertices[:, 1] > 1.0 - tol)[0]

    bc_dofs = np.concatenate([
        2 * bottom,
        2 * bottom + 1,
        2 * top + 1,
    ])
    bc_vals = np.concatenate([
        np.zeros(len(bottom)),
        np.zeros(len(bottom)),
        np.full(len(top), eps_0),
    ])
    bc_dofs, unique_idx = np.unique(bc_dofs, return_index=True)
    bc_vals = bc_vals[unique_idx]

    # Time array
    t_array = np.concatenate([
        [0.0],
        np.geomspace(0.5, 120.0, 50),
    ])
    t_array = np.sort(np.unique(t_array))

    print(f"  Time steps: {len(t_array)}, t_max = {t_array[-1]:.1f} s")

    # Solve
    u_hist, sigma_hist, h_hist = vem_viscoelastic_sls(
        vertices, elements, DI_field, nu,
        bc_dofs, bc_vals, t_array,
    )
    print("  Solve complete.")

    # Find time index closest to t=60s
    ti_mid = np.argmin(np.abs(t_array - 60.0))

    # Find 3 probe elements: left, center, right
    x_targets = [0.15, 0.50, 0.85]
    probe_els = []
    for xt in x_targets:
        dists = np.abs(centroids[:, 0] - xt) + np.abs(centroids[:, 1] - 0.5)
        probe_els.append(np.argmin(dists))

    # ---- Plot 2x3 figure ----
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    def _plot_field(ax, field, cmap, label, title):
        patches = []
        for k in range(n_el):
            vert_ids = elements[k].astype(int)
            poly = MplPolygon(vertices[vert_ids], closed=True)
            patches.append(poly)
        pc = PatchCollection(patches, cmap=cmap, edgecolor='k', linewidth=0.3)
        pc.set_array(np.array(field))
        ax.add_collection(pc)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        fig.colorbar(pc, ax=ax, label=label, shrink=0.8)
        ax.set_title(title, fontsize=11, fontweight='bold')

    # (a) DI field
    _plot_field(axes[0, 0], DI_field, 'RdYlGn_r', 'DI', '(a) Dysbiosis Index')

    # (b) E_inf(DI) field
    _plot_field(axes[0, 1], E_inf, 'viridis', '$E_\\infty$ [Pa]',
                '(b) Equilibrium Modulus $E_\\infty$(DI)')

    # (c) tau(DI) field
    _plot_field(axes[0, 2], tau_arr, 'plasma', r'$\tau$ [s]',
                r'(c) Relaxation Time $\tau$(DI)')

    # (d) sigma_xx at t=0
    _plot_field(axes[1, 0], sigma_hist[0, :, 0], 'coolwarm',
                r'$\sigma_{xx}$ [Pa]',
                f'(d) $\\sigma_{{xx}}$ at t={t_array[0]:.1f}s')

    # (e) sigma_yy at t~60s
    _plot_field(axes[1, 1], sigma_hist[ti_mid, :, 1], 'coolwarm',
                r'$\sigma_{yy}$ [Pa]',
                f'(e) $\\sigma_{{yy}}$ at t={t_array[ti_mid]:.1f}s')

    # (f) Stress relaxation curves at 3 probes
    ax = axes[1, 2]
    colors = ['#2166ac', '#4dac26', '#d73027']
    labels_probe = ['Left (commensal)', 'Center', 'Right (dysbiotic)']
    for idx, (el_k, c, lab) in enumerate(zip(probe_els, colors, labels_probe)):
        sig_yy = sigma_hist[:, el_k, 1]
        ax.plot(t_array, sig_yy, color=c, lw=2, label=lab)

        # Analytical reference
        E_inf_k = E_inf[el_k]
        E_1_k = E_1[el_k]
        tau_k = tau_arr[el_k]
        sig_ana = (E_inf_k + E_1_k * np.exp(-t_array / tau_k)) * eps_0
        ax.plot(t_array, sig_ana, color=c, ls='--', lw=1, alpha=0.7)

        # Mark tau
        ax.axvline(tau_k, color=c, ls=':', lw=0.8, alpha=0.5)

    ax.set_xlabel('Time [s]', fontsize=10)
    ax.set_ylabel(r'$\sigma_{yy}$ [Pa]', fontsize=10)
    ax.set_title('(f) Stress Relaxation (solid=VEM, dashed=analytical)', fontsize=11,
                 fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(alpha=0.3)
    ax.set_xlim(0, t_array[-1])

    fig.suptitle('VE-VEM: Viscoelastic Virtual Element Method with DI Gradient',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "vem_viscoelastic_demo.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")

    # Summary
    print("\n  Summary:")
    for idx, (el_k, lab) in enumerate(zip(probe_els, labels_probe)):
        sig_0 = sigma_hist[0, el_k, 1]
        sig_end = sigma_hist[-1, el_k, 1]
        relaxation_pct = (1.0 - sig_end / sig_0) * 100 if abs(sig_0) > 1e-12 else 0.0
        print(f"    {lab}: sigma_yy(0)={sig_0:.4f} Pa -> "
              f"sigma_yy({t_array[-1]:.0f}s)={sig_end:.4f} Pa "
              f"({relaxation_pct:.1f}% relaxation)")

    return u_hist, sigma_hist, h_hist


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print()
    err = validate_sls_relaxation()
    print()
    demo_ve_vem()
    print("\nDone.")
