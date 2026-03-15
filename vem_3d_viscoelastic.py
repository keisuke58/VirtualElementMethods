#!/usr/bin/env python3
"""
vem_3d_viscoelastic.py -- 3D Viscoelastic VEM (SLS + Simo 1987)
================================================================

Extends the 2D VE-VEM approach to 3D polyhedral elements.

SLS Model (3D Voigt: [σxx, σyy, σzz, σyz, σxz, σxy]):
    σ(t) = C_inf · ε + h(t)
    h_{n+1} = exp(-dt/τ) · h_n + γ · C_1 · (ε_{n+1} - ε_n)
    C_alg = C_inf + γ · C_1

Spatial: P₁ VEM on polyhedra (12 polynomial basis, vertex DOFs).
Time: Simo 1987 exponential integrator.
"""

import numpy as np
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import sys, os

sys.path.insert(0, os.path.dirname(__file__))

from vem_3d import (
    make_hex_mesh, face_normal_area, polyhedron_volume,
    isotropic_3d, traction_from_voigt,
)
from vem_viscoelastic import sls_params_from_di


# ---------------------------------------------------------------------------
# 3D VEM element projector and strain operator
# ---------------------------------------------------------------------------

def _compute_element_3d(vertices, vert_ids, faces, nu):
    """
    Compute 3D VEM element projector and strain operator.

    Returns dict with: vol, centroid, h, projector, strain_proj, D, B, G
    """
    coords = vertices[vert_ids]
    n_v = len(vert_ids)
    n_el_dofs = 3 * n_v
    n_polys = 12

    # Geometry
    centroid = coords.mean(axis=0)
    h = max(np.linalg.norm(coords[i] - coords[j])
            for i in range(n_v) for j in range(i + 1, n_v))
    vol = polyhedron_volume(vertices, faces)

    xc, yc, zc = centroid
    vmap = {int(g): loc for loc, g in enumerate(vert_ids)}

    # Reference C (E=1) for B matrix construction
    C_ref = isotropic_3d(1.0, nu)

    strain_ids = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 2, 0, 0],
        [0, 0, 0, 0, 2, 0],
        [0, 0, 0, 0, 0, 2],
    ], dtype=float)

    # D matrix (3·n_v × 12)
    D = np.zeros((n_el_dofs, n_polys))
    for i in range(n_v):
        dx = (coords[i, 0] - xc) / h
        dy = (coords[i, 1] - yc) / h
        dz = (coords[i, 2] - zc) / h
        r = 3 * i
        D[r,   :] = [1, 0, 0, 0,   dz, -dy, dx, 0,  0,  0,  dz, dy]
        D[r+1, :] = [0, 1, 0, -dz, 0,  dx,  0,  dy, 0,  dz, 0,  dx]
        D[r+2, :] = [0, 0, 1, dy,  -dx, 0,  0,  0,  dz, dy, dx, 0]

    # B matrix (12 × 3·n_v)
    B = np.zeros((n_polys, n_el_dofs))

    for i in range(n_v):
        B[0, 3 * i] = 1.0 / n_v
        B[1, 3 * i + 1] = 1.0 / n_v
        B[2, 3 * i + 2] = 1.0 / n_v

    for face in faces:
        face_int = face.astype(int)
        pts = vertices[face_int]
        n_f, A_f = face_normal_area(pts)

        fc = pts.mean(axis=0)
        if np.dot(n_f, fc - centroid) < 0:
            n_f = -n_f

        k_f = len(face_int)
        for gv in face_int:
            if gv not in vmap:
                continue
            li = vmap[gv]
            w = A_f / k_f

            wrot = w / (2.0 * vol)
            B[3, 3*li + 1] += -wrot * n_f[2]
            B[3, 3*li + 2] += wrot * n_f[1]
            B[4, 3*li + 0] += wrot * n_f[2]
            B[4, 3*li + 2] += -wrot * n_f[0]
            B[5, 3*li + 0] += -wrot * n_f[1]
            B[5, 3*li + 1] += wrot * n_f[0]

            for alpha in range(6):
                eps_a = strain_ids[alpha] / h
                sigma_a = C_ref @ eps_a
                t_f = traction_from_voigt(sigma_a, n_f)
                B[6 + alpha, 3*li + 0] += w * t_f[0]
                B[6 + alpha, 3*li + 1] += w * t_f[1]
                B[6 + alpha, 3*li + 2] += w * t_f[2]

    # Projector
    G = B @ D
    projector = np.linalg.solve(G, B)

    # Strain projector: maps DOFs -> 6-component Voigt strain
    strain_proj = np.zeros((6, n_el_dofs))
    for alpha in range(6):
        strain_proj[alpha, :] = (strain_ids[alpha, alpha] / h) * projector[6 + alpha, :]

    return {
        "vol": vol,
        "centroid": centroid,
        "h": h,
        "n_v": n_v,
        "n_el_dofs": n_el_dofs,
        "D": D,
        "B": B,
        "G": G,
        "projector": projector,
        "strain_proj": strain_proj,
    }


# ---------------------------------------------------------------------------
# 3D VE-VEM solver
# ---------------------------------------------------------------------------

def vem_3d_viscoelastic_sls(vertices, cells, cell_faces, DI_field, nu,
                             bc_fixed_dofs, bc_vals, t_array,
                             load_dofs=None, load_vals_func=None,
                             **sls_kwargs):
    """
    3D time-stepping VEM for SLS viscoelasticity with DI-dependent parameters.

    Parameters
    ----------
    vertices : (N, 3) node coordinates
    cells : list of int arrays — vertex indices per cell
    cell_faces : list of lists of int arrays — face vertices per cell
    DI_field : (N_el,) DI per element
    nu : float
    bc_fixed_dofs : int array
    bc_vals : float array
    t_array : (N_t,) time points
    load_dofs, load_vals_func : optional external loads

    Returns
    -------
    u_history : (N_t, 3*N_nodes)
    sigma_history : (N_t, N_el, 6)
    h_history : (N_t, N_el, 6)
    """
    n_nodes = len(vertices)
    n_dofs = 3 * n_nodes
    n_el = len(cells)
    n_t = len(t_array)

    params = sls_params_from_di(DI_field, **sls_kwargs)
    E_inf = params["E_inf"]
    E_1 = params["E_1"]
    tau = params["tau"]

    C_inf_all = np.zeros((n_el, 6, 6))
    C_1_all = np.zeros((n_el, 6, 6))
    for k in range(n_el):
        C_inf_all[k] = isotropic_3d(E_inf[k], nu)
        C_1_all[k] = isotropic_3d(E_1[k], nu)

    # Precompute element data
    elem_data = []
    for el_id in range(n_el):
        vert_ids = cells[el_id].astype(int)
        faces = cell_faces[el_id]
        elem_data.append(_compute_element_3d(vertices, vert_ids, faces, nu))

    # Storage
    u_history = np.zeros((n_t, n_dofs))
    sigma_history = np.zeros((n_t, n_el, 6))
    h_history = np.zeros((n_t, n_el, 6))

    h_all = np.zeros((n_el, 6))
    eps_prev = np.zeros((n_el, 6))

    for ti in range(n_t):
        t = t_array[ti]
        dt = t_array[ti] - t_array[ti - 1] if ti > 0 else 0.0

        if ti == 0:
            gamma_coeff = np.ones(n_el)
            exp_dt = np.zeros(n_el)
        elif dt > 1e-15:
            exp_dt = np.exp(-dt / tau)
            gamma_coeff = (tau / dt) * (1.0 - exp_dt)
        else:
            exp_dt = np.ones(n_el)
            gamma_coeff = np.ones(n_el)

        C_alg_all = C_inf_all + gamma_coeff[:, None, None] * C_1_all

        if ti == 0:
            h_star = np.zeros((n_el, 6))
        else:
            h_star = np.zeros((n_el, 6))
            for k in range(n_el):
                h_star[k] = exp_dt[k] * h_all[k] - gamma_coeff[k] * C_1_all[k] @ eps_prev[k]

        F_ext = np.zeros(n_dofs)
        if load_dofs is not None and load_vals_func is not None:
            F_ext[load_dofs] = load_vals_func(t)

        # Assemble and solve
        K_global = np.zeros((n_dofs, n_dofs))
        F_h = np.zeros(n_dofs)

        for el_id in range(n_el):
            vert_ids = cells[el_id].astype(int)
            ed = elem_data[el_id]
            n_v = ed["n_v"]
            n_el_dofs = ed["n_el_dofs"]
            vol = ed["vol"]
            C_alg = C_alg_all[el_id]

            strain_proj = ed["strain_proj"]
            projector = ed["projector"]
            D = ed["D"]

            K_cons = vol * strain_proj.T @ C_alg @ strain_proj

            I_minus_PiD = np.eye(n_el_dofs) - D @ projector
            trace_k = np.trace(K_cons)
            stab_param = trace_k / n_el_dofs if trace_k > 0 else 1.0
            K_stab = stab_param * (I_minus_PiD.T @ I_minus_PiD)

            K_local = K_cons + K_stab
            f_h_local = vol * strain_proj.T @ h_star[el_id]

            gdofs = np.zeros(n_el_dofs, dtype=int)
            for i in range(n_v):
                gdofs[3 * i] = 3 * vert_ids[i]
                gdofs[3 * i + 1] = 3 * vert_ids[i] + 1
                gdofs[3 * i + 2] = 3 * vert_ids[i] + 2

            for i in range(n_el_dofs):
                for j in range(n_el_dofs):
                    K_global[gdofs[i], gdofs[j]] += K_local[i, j]
            F_h[gdofs] += f_h_local

        F_rhs = F_ext - F_h

        u = np.zeros(n_dofs)
        bc_set = set(bc_fixed_dofs.tolist())
        internal = np.array([i for i in range(n_dofs) if i not in bc_set])

        u[bc_fixed_dofs] = bc_vals
        F_rhs -= K_global[:, bc_fixed_dofs] @ bc_vals

        K_ii = K_global[np.ix_(internal, internal)]
        u[internal] = np.linalg.solve(K_ii, F_rhs[internal])

        # Post-process
        for k in range(n_el):
            vert_ids = cells[k].astype(int)
            ed = elem_data[k]
            n_v = ed["n_v"]

            gdofs = np.zeros(ed["n_el_dofs"], dtype=int)
            for i in range(n_v):
                gdofs[3 * i] = 3 * vert_ids[i]
                gdofs[3 * i + 1] = 3 * vert_ids[i] + 1
                gdofs[3 * i + 2] = 3 * vert_ids[i] + 2

            u_el = u[gdofs]
            eps_new = ed["strain_proj"] @ u_el

            if ti == 0:
                h_all[k] = C_1_all[k] @ eps_new
            else:
                h_all[k] = exp_dt[k] * h_all[k] + \
                    gamma_coeff[k] * C_1_all[k] @ (eps_new - eps_prev[k])

            sigma_history[ti, k] = C_inf_all[k] @ eps_new + h_all[k]
            eps_prev[k] = eps_new.copy()

        u_history[ti] = u
        h_history[ti] = h_all.copy()

    return u_history, sigma_history, h_history


# ---------------------------------------------------------------------------
# Validation: confined compression on hex mesh
# ---------------------------------------------------------------------------

def validate_3d_sls():
    """Validate 3D VE-VEM against analytical SLS for confined compression."""
    print("=" * 60)
    print("Validation: 3D VE-VEM SLS vs analytical")
    print("=" * 60)

    vertices, cells, cell_faces = make_hex_mesh(nx=3, ny=3, nz=3, perturb=0.0, seed=42)
    n_el = len(cells)
    n_nodes = len(vertices)
    print(f"  Mesh: {n_el} cells, {n_nodes} nodes")

    DI_val = 0.4
    DI_field = np.full(n_el, DI_val)
    nu = 0.3
    eps_0 = 0.01

    params = sls_params_from_di(DI_field)
    E_inf_val = params["E_inf"][0]
    E_1_val = params["E_1"][0]
    tau_val = params["tau"][0]

    print(f"  DI={DI_val}, E_inf={E_inf_val:.1f}, E_1={E_1_val:.1f}, tau={tau_val:.1f}")

    # Confined: fix u_x, u_y everywhere; fix u_z on bottom; prescribe u_z on top
    tol = 1e-6
    bottom = np.where(vertices[:, 2] < tol)[0]
    top = np.where(vertices[:, 2] > 1.0 - tol)[0]
    all_nodes = np.arange(n_nodes)

    bc_dofs = np.concatenate([
        3 * all_nodes,       # u_x = 0
        3 * all_nodes + 1,   # u_y = 0
        3 * bottom + 2,      # u_z = 0 (bottom)
        3 * top + 2,         # u_z = eps_0 (top)
    ])
    bc_vals = np.concatenate([
        np.zeros(n_nodes),
        np.zeros(n_nodes),
        np.zeros(len(bottom)),
        np.full(len(top), eps_0),
    ])
    bc_dofs, unique_idx = np.unique(bc_dofs, return_index=True)
    bc_vals = bc_vals[unique_idx]

    t_array = np.concatenate([[0.0], np.linspace(tau_val / 10, 3 * tau_val, 20)])

    u_hist, sigma_hist, h_hist = vem_3d_viscoelastic_sls(
        vertices, cells, cell_faces, DI_field, nu,
        bc_dofs, bc_vals, t_array,
    )

    # Analytical: for 3D confined (eps_xx=eps_yy=0, eps_zz=eps_0):
    # sigma_zz = C_33(t) * eps_0 where C_33 = lambda + 2*mu
    lam_inf = E_inf_val * nu / ((1 + nu) * (1 - 2 * nu))
    mu_inf = E_inf_val / (2 * (1 + nu))
    lam_1 = E_1_val * nu / ((1 + nu) * (1 - 2 * nu))
    mu_1 = E_1_val / (2 * (1 + nu))
    C33_inf = lam_inf + 2 * mu_inf
    C33_1 = lam_1 + 2 * mu_1

    sigma_ana = (C33_inf + C33_1 * np.exp(-t_array / tau_val)) * eps_0
    sigma_zz_vem = sigma_hist[:, :, 2].mean(axis=1)

    max_rel_err = 0.0
    print(f"\n  {'t':>6s} {'σ_zz VEM':>12s} {'σ_zz ana':>12s} {'err':>10s}")
    print("  " + "-" * 44)
    for ti in range(len(t_array)):
        err = abs(sigma_zz_vem[ti] - sigma_ana[ti])
        rel = err / (abs(sigma_ana[ti]) + 1e-12)
        max_rel_err = max(max_rel_err, rel)
        if ti % 3 == 0 or ti == len(t_array) - 1:
            print(f"  {t_array[ti]:6.1f} {sigma_zz_vem[ti]:12.6f} "
                  f"{sigma_ana[ti]:12.6f} {rel:10.2e}")

    print(f"\n  Max relative error: {max_rel_err:.4e}")
    if max_rel_err < 0.01:
        print("  PASSED (< 1%)")
    else:
        print(f"  WARNING: {max_rel_err:.2e} > 1%")

    return max_rel_err


if __name__ == "__main__":
    validate_3d_sls()
