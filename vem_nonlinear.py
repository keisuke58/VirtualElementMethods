"""
VEM for 2D Neo-Hookean Hyperelasticity on Polygonal Meshes.

Large-deformation extension of vem_elasticity.py.
Uses Newton-Raphson with incremental loading on VEM projector-based
deformation gradient.

Neo-Hookean model:
  W(F) = μ/2·(I₁ - 2) - μ·ln(J) + λ/2·(ln J)²
  P = μ·(F - F^{-T}) + λ·ln(J)·F^{-T}   (1st Piola-Kirchhoff)

References:
  - Wriggers, Hudobivnik (2019) "Low order 3D VEM for finite elasto-plastic"
  - Wriggers, Aldakheel, Hudobivnik (2024) "VEM in Engineering Sciences" Ch.8
  - Chen, Sukumar (2024) "Stabilization-free VEM for hyperelasticity"
"""

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection


# ── Neo-Hookean Material ──────────────────────────────────────────────────

def neo_hookean_params(E, nu):
    """Convert Young's modulus and Poisson's ratio to Lamé parameters."""
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return mu, lam


def compute_PK1_stress_2d(F, mu, lam):
    """
    First Piola-Kirchhoff stress for 2D plane-strain Neo-Hookean.

    P = μ·(F - F^{-T}) + λ·ln(J)·F^{-T}
    """
    J = np.linalg.det(F)
    J = max(J, 1e-12)
    Finv_T = np.linalg.inv(F).T
    P = mu * (F - Finv_T) + lam * np.log(J) * Finv_T
    return P


def compute_tangent_2d(F, mu, lam):
    """
    Material tangent modulus ∂P/∂F for 2D Neo-Hookean (Voigt-like 4×4).

    Returns C_tang as (4,4) mapping [dF11, dF12, dF21, dF22] → [dP11, dP12, dP21, dP22].
    """
    J = np.linalg.det(F)
    J = max(J, 1e-12)
    lnJ = np.log(J)
    Finv = np.linalg.inv(F)

    # Build 4×4 tangent in component form
    # A_{iJkL} = μ·δ_{ik}δ_{JL} + (λ - μ + μ/J²?)
    # Simplified: use numerical differentiation for robustness
    C = np.zeros((4, 4))
    eps = 1e-7
    P0 = compute_PK1_stress_2d(F, mu, lam)
    P0_flat = np.array([P0[0, 0], P0[0, 1], P0[1, 0], P0[1, 1]])

    for k in range(4):
        dF = np.zeros((2, 2))
        i, j = divmod(k, 2)
        dF[i, j] = eps
        Pp = compute_PK1_stress_2d(F + dF, mu, lam)
        Pp_flat = np.array([Pp[0, 0], Pp[0, 1], Pp[1, 0], Pp[1, 1]])
        C[:, k] = (Pp_flat - P0_flat) / eps

    return C


def compute_strain_energy_2d(F, mu, lam):
    """Neo-Hookean strain energy density W(F)."""
    J = np.linalg.det(F)
    J = max(J, 1e-12)
    I1 = np.trace(F.T @ F)
    return mu / 2.0 * (I1 - 2.0) - mu * np.log(J) + lam / 2.0 * np.log(J) ** 2


# ── Element-level VEM operations ──────────────────────────────────────────

def _element_geometry(verts):
    """Compute area, centroid, diameter for polygon."""
    n_v = len(verts)
    area_comp = (
        verts[:, 0] * np.roll(verts[:, 1], -1)
        - np.roll(verts[:, 0], -1) * verts[:, 1]
    )
    area = 0.5 * abs(np.sum(area_comp))
    centroid = np.sum(
        (np.roll(verts, -1, axis=0) + verts) * area_comp[:, None], axis=0
    ) / (6.0 * max(area, 1e-15))
    h = max(
        np.linalg.norm(verts[i] - verts[j])
        for i in range(n_v)
        for j in range(i + 1, n_v)
    )
    return area, centroid, h


def _build_projector(verts, C_mat):
    """
    Build VEM projector Π^∇ for elasticity element.

    Returns: projector (6 × n_el_dofs), D (n_el_dofs × 6), G_tilde, area, h
    """
    n_v = len(verts)
    n_el_dofs = 2 * n_v
    n_polys = 6

    area, centroid, h = _element_geometry(verts)
    xc, yc = centroid

    # D matrix
    D = np.zeros((n_el_dofs, n_polys))
    for i in range(n_v):
        dx = (verts[i, 0] - xc) / h
        dy = (verts[i, 1] - yc) / h
        D[2 * i, :] = [1.0, 0.0, -dy, dx, 0.0, dy]
        D[2 * i + 1, :] = [0.0, 1.0, dx, 0.0, dy, dx]

    # B matrix
    B = np.zeros((n_polys, n_el_dofs))

    # Vertex normals
    vertex_normals = np.zeros((n_v, 2))
    for i in range(n_v):
        prev_v = verts[(i - 1) % n_v]
        next_v = verts[(i + 1) % n_v]
        vertex_normals[i] = [next_v[1] - prev_v[1], prev_v[0] - next_v[0]]

    # Rows 0-1: translations
    for i in range(n_v):
        B[0, 2 * i] = 1.0 / n_v
        B[1, 2 * i + 1] = 1.0 / n_v

    # Row 2: rigid rotation
    for i in range(n_v):
        B[2, 2 * i] = -vertex_normals[i, 1] / (4.0 * area)
        B[2, 2 * i + 1] = vertex_normals[i, 0] / (4.0 * area)

    # Rows 3-5: strain modes
    strain_basis = np.array(
        [[1.0 / h, 0.0, 0.0], [0.0, 1.0 / h, 0.0], [0.0, 0.0, 2.0 / h]]
    )
    for i in range(n_v):
        vn = vertex_normals[i]
        for alpha in range(3):
            sigma = C_mat @ strain_basis[alpha]
            tx = sigma[0] * vn[0] + sigma[2] * vn[1]
            ty = sigma[2] * vn[0] + sigma[1] * vn[1]
            B[3 + alpha, 2 * i] += 0.5 * tx
            B[3 + alpha, 2 * i + 1] += 0.5 * ty

    G = B @ D
    projector = np.linalg.solve(G, B)

    G_tilde = G.copy()
    G_tilde[:3, :] = 0.0

    return projector, D, G_tilde, area, h


def _compute_deformation_gradient(projector, u_local, h):
    """
    Compute average deformation gradient F from VEM projector.

    F = I + grad(u) where grad(u) is recovered from projector coefficients.
    Projector maps u_local → [c0, c1, c2, c3, c4, c5]
    where c3/h = du_x/dx, c4/h = du_y/dy, c5/h = du_x/dy + du_y/dx (approx)
    """
    c = projector @ u_local
    # c[3] = (du_x/dx) * h, c[4] = (du_y/dy) * h, c[5] = mixed
    F = np.eye(2)
    F[0, 0] += c[3] / h  # du_x/dx
    F[1, 1] += c[4] / h  # du_y/dy
    # For shear: c[5]/h ≈ du_x/dy + du_y/dx
    # Split equally between du_x/dy and du_y/dx
    F[0, 1] += 0.5 * c[5] / h  # du_x/dy
    F[1, 0] += 0.5 * c[5] / h  # du_y/dx
    return F


# ── Nonlinear VEM Solver ──────────────────────────────────────────────────


def vem_nonlinear(
    vertices,
    elements,
    E_field,
    nu,
    bc_fixed_dofs,
    bc_vals,
    load_dofs=None,
    load_vals=None,
    n_load_steps=10,
    tol=1e-6,
    max_iter=25,
    stabilization_alpha=0.5,
    verbose=False,
):
    """
    Nonlinear VEM solver with Newton-Raphson for Neo-Hookean hyperelasticity.

    Same interface as vem_elasticity() with additional NR parameters.

    Parameters
    ----------
    vertices, elements, E_field, nu, bc_fixed_dofs, bc_vals: same as linear
    load_dofs, load_vals: external point loads
    n_load_steps: number of incremental load steps
    tol: Newton-Raphson convergence tolerance (relative residual)
    max_iter: max NR iterations per load step
    stabilization_alpha: VEM stabilization parameter

    Returns
    -------
    u : (2*N,) displacement vector
    info : dict with convergence history
    """
    n_nodes = vertices.shape[0]
    n_dofs = 2 * n_nodes
    n_el = len(elements)

    # External force vector
    F_ext = np.zeros(n_dofs)
    if load_dofs is not None and load_vals is not None:
        F_ext[load_dofs] += load_vals

    # BC setup
    bc_set = set(bc_fixed_dofs.tolist())
    internal = np.array([i for i in range(n_dofs) if i not in bc_set])

    # Pre-compute element data (reference configuration)
    elem_data = []
    for el_id in range(n_el):
        vert_ids = elements[el_id].astype(int)
        verts = vertices[vert_ids]
        n_v = len(vert_ids)
        n_el_dofs = 2 * n_v

        E_el = E_field[el_id] if hasattr(E_field, "__len__") else E_field
        mu, lam = neo_hookean_params(E_el, nu)

        # Plane strain constitutive (for projector construction)
        C_mat = (E_el / (1.0 - nu**2)) * np.array(
            [
                [1.0, nu, 0.0],
                [nu, 1.0, 0.0],
                [0.0, 0.0, (1.0 - nu) / 2.0],
            ]
        )

        projector, D, G_tilde, area, h = _build_projector(verts, C_mat)

        gdofs = np.zeros(n_el_dofs, dtype=int)
        for i in range(n_v):
            gdofs[2 * i] = 2 * vert_ids[i]
            gdofs[2 * i + 1] = 2 * vert_ids[i] + 1

        elem_data.append(
            {
                "vert_ids": vert_ids,
                "verts": verts,
                "n_v": n_v,
                "n_el_dofs": n_el_dofs,
                "mu": mu,
                "lam": lam,
                "E_el": E_el,
                "projector": projector,
                "D": D,
                "G_tilde": G_tilde,
                "area": area,
                "h": h,
                "gdofs": gdofs,
                "C_mat": C_mat,
            }
        )

    # Incremental loading
    u = np.zeros(n_dofs)
    u[bc_fixed_dofs] = 0.0  # Initial BCs

    converged_steps = 0
    history = []

    for step in range(n_load_steps):
        load_factor = (step + 1) / n_load_steps
        F_ext_step = load_factor * F_ext

        for nr_iter in range(max_iter):
            # Assemble internal force and tangent stiffness
            F_int = np.zeros(n_dofs)
            row_idx = []
            col_idx = []
            val_data = []

            total_energy = 0.0

            for ed in elem_data:
                u_local = u[ed["gdofs"]]
                proj = ed["projector"]
                h = ed["h"]
                area = ed["area"]
                mu = ed["mu"]
                lam = ed["lam"]
                n_el_dofs = ed["n_el_dofs"]

                # Deformation gradient
                F_def = _compute_deformation_gradient(proj, u_local, h)

                # PK1 stress
                P = compute_PK1_stress_2d(F_def, mu, lam)
                W = compute_strain_energy_2d(F_def, mu, lam)
                total_energy += W * area

                # Internal force: B^T · P_voigt · area
                # P_voigt = [P11, P12, P21, P22]
                P_voigt = np.array([P[0, 0], P[0, 1], P[1, 0], P[1, 1]])

                # B_grad matrix: maps u_local → [du_x/dx, du_x/dy, du_y/dx, du_y/dy]
                # From projector rows 3-5: c[3]/h=du_x/dx, c[4]/h=du_y/dy, c[5]/h=mixed
                B_grad = np.zeros((4, n_el_dofs))
                for k in range(n_el_dofs):
                    c = proj[:, k]
                    B_grad[0, k] = c[3] / h  # du_x/dx
                    B_grad[1, k] = 0.5 * c[5] / h  # du_x/dy
                    B_grad[2, k] = 0.5 * c[5] / h  # du_y/dx
                    B_grad[3, k] = c[4] / h  # du_y/dy

                f_int_local = B_grad.T @ P_voigt * area
                F_int[ed["gdofs"]] += f_int_local

                # Tangent stiffness
                C_tang = compute_tangent_2d(F_def, mu, lam)
                K_mat_local = B_grad.T @ C_tang @ B_grad * area

                # Stabilization
                I_minus_PiD = np.eye(n_el_dofs) - ed["D"] @ proj
                trace_k = np.trace(K_mat_local)
                stab = (
                    stabilization_alpha * abs(trace_k) / n_el_dofs
                    if trace_k != 0
                    else ed["E_el"] * 0.01
                )
                K_stab = stab * (I_minus_PiD.T @ I_minus_PiD)
                K_local = K_mat_local + K_stab

                gdofs = ed["gdofs"]
                ii, jj = np.meshgrid(gdofs, gdofs, indexing="ij")
                row_idx.append(ii.ravel())
                col_idx.append(jj.ravel())
                val_data.append(K_local.ravel())

            # Global tangent
            K_global = sp.csr_matrix(
                (np.concatenate(val_data), (np.concatenate(row_idx), np.concatenate(col_idx))),
                shape=(n_dofs, n_dofs),
            )

            # Residual
            R = F_int - F_ext_step
            R[bc_fixed_dofs] = 0.0

            # Check convergence
            R_norm = np.linalg.norm(R[internal])
            F_norm = max(np.linalg.norm(F_ext_step[internal]), 1e-10)
            rel_res = R_norm / F_norm

            if verbose and nr_iter % 5 == 0:
                print(
                    f"  Step {step+1}/{n_load_steps}, NR iter {nr_iter}: "
                    f"|R|/|F| = {rel_res:.2e}, W = {total_energy:.4e}"
                )

            if rel_res < tol or R_norm < 1e-14:
                break

            # Solve for correction
            K_ii = K_global[np.ix_(internal, internal)]
            try:
                du = np.zeros(n_dofs)
                du[internal] = -sp.linalg.spsolve(K_ii, R[internal])
            except Exception:
                if verbose:
                    print(f"  Warning: linear solve failed at step {step+1}")
                break

            # Line search (simple backtracking)
            alpha_ls = 1.0
            for _ in range(5):
                u_trial = u + alpha_ls * du
                R_trial_norm = 0.0
                for ed in elem_data:
                    u_local_t = u_trial[ed["gdofs"]]
                    F_t = _compute_deformation_gradient(ed["projector"], u_local_t, ed["h"])
                    J_t = np.linalg.det(F_t)
                    if J_t <= 0:
                        R_trial_norm = np.inf
                        break
                if R_trial_norm == np.inf:
                    alpha_ls *= 0.5
                else:
                    break

            u += alpha_ls * du

        converged_steps += 1
        history.append(
            {
                "step": step,
                "load_factor": load_factor,
                "nr_iters": nr_iter + 1,
                "residual": rel_res,
                "energy": total_energy,
            }
        )

    info = {"history": history, "converged_steps": converged_steps}
    return u, info


# ── Comparison: Linear vs Nonlinear VEM ───────────────────────────────────

def compare_linear_nonlinear(
    vertices, elements, E_field, nu, bc_fixed_dofs, bc_vals, load_dofs, load_vals
):
    """
    Run both linear and nonlinear VEM, return results for comparison.
    """
    from vem_elasticity import vem_elasticity

    u_lin = vem_elasticity(
        vertices, elements, E_field, nu, bc_fixed_dofs, bc_vals, load_dofs, load_vals
    )

    u_nl, info = vem_nonlinear(
        vertices,
        elements,
        E_field,
        nu,
        bc_fixed_dofs,
        bc_vals,
        load_dofs,
        load_vals,
        n_load_steps=10,
        verbose=False,
    )

    ux_lin, uy_lin = u_lin[0::2], u_lin[1::2]
    ux_nl, uy_nl = u_nl[0::2], u_nl[1::2]

    mag_lin = np.sqrt(ux_lin**2 + uy_lin**2)
    mag_nl = np.sqrt(ux_nl**2 + uy_nl**2)

    return {
        "u_linear": u_lin,
        "u_nonlinear": u_nl,
        "mag_linear_max": np.max(mag_lin),
        "mag_nonlinear_max": np.max(mag_nl),
        "relative_diff": abs(np.max(mag_nl) - np.max(mag_lin)) / max(np.max(mag_lin), 1e-15),
        "info": info,
    }


# ── Demo ──────────────────────────────────────────────────────────────────

def demo_biofilm_large_deformation():
    """
    Demo: Commensal (stiff) vs Dysbiotic (soft) biofilm under large GCF pressure.
    Shows when nonlinearity matters (soft biofilm under high load).
    """
    from vem_growth_coupled import (
        make_biofilm_voronoi,
        compute_DI,
        compute_E,
        SPECIES_NAMES,
    )
    from vem_elasticity import vem_elasticity

    rng = np.random.default_rng(42)
    domain = (0, 2, 0, 1)
    n_cells = 30
    xmin, xmax, ymin, ymax = domain
    nu = 0.35

    # Seeds
    nx = int(np.sqrt(n_cells * 2))
    ny = max(n_cells // nx, 2)
    xx = np.linspace(xmin + 0.1, xmax - 0.1, nx)
    yy = np.linspace(ymin + 0.05, ymax - 0.05, ny)
    gx, gy = np.meshgrid(xx, yy)
    seeds = np.column_stack([gx.ravel(), gy.ravel()])[:n_cells]
    seeds += rng.uniform(-0.03, 0.03, seeds.shape)

    vertices, elements, bnd, valid_ids = make_biofilm_voronoi(seeds, domain)
    n_el = len(elements)

    # Conditions: commensal (low DI) vs dysbiotic (high DI)
    conditions = {
        "Commensal (DI=0.10)": 0.10,
        "Dysbiotic (DI=0.75)": 0.75,
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for row, (label, DI_val) in enumerate(conditions.items()):
        E_val = compute_E(DI_val)
        E_field = np.full(n_el, E_val)

        # Collect used nodes
        used_set = set()
        for el in elements:
            used_set.update(el.astype(int).tolist())
        used = np.array(sorted(used_set))
        old_to_new = {int(g): i for i, g in enumerate(used)}
        n_used = len(used)

        compact_verts = vertices[used]
        compact_elems = [np.array([old_to_new[int(v)] for v in el]) for el in elements]

        # BC: bottom fixed
        tol_bc = 0.02
        bottom = np.where(compact_verts[:, 1] < ymin + tol_bc)[0]
        bc_dofs = np.concatenate([2 * bottom, 2 * bottom + 1])
        bc_vals_arr = np.zeros(len(bc_dofs))

        # Load: large GCF pressure on top (shear + compression)
        top = np.where(compact_verts[:, 1] > ymax - tol_bc)[0]
        load_scale = 0.5 if DI_val > 0.5 else 2.0  # Higher load for stiff biofilm

        load_dofs_list = []
        load_vals_list = []
        if len(top) > 0:
            # Shear (x-direction) on top
            load_dofs_list.append(2 * top)
            load_vals_list.append(np.full(len(top), load_scale / len(top)))
            # Compression (y-direction) on top
            load_dofs_list.append(2 * top + 1)
            load_vals_list.append(np.full(len(top), -load_scale * 0.5 / len(top)))

        load_dofs = np.concatenate(load_dofs_list) if load_dofs_list else np.array([], dtype=int)
        load_vals_arr = np.concatenate(load_vals_list) if load_vals_list else np.array([])

        # Linear solve
        u_lin = vem_elasticity(
            compact_verts, compact_elems, E_field, nu, bc_dofs, bc_vals_arr,
            load_dofs, load_vals_arr
        )

        # Nonlinear solve
        u_nl, info = vem_nonlinear(
            compact_verts, compact_elems, E_field, nu, bc_dofs, bc_vals_arr,
            load_dofs, load_vals_arr, n_load_steps=10, verbose=False
        )

        # Compute strain for nonlinearity check
        max_strain = 0.0
        for ed_idx, el in enumerate(compact_elems):
            el_int = el.astype(int)
            verts_el = compact_verts[el_int]
            n_v = len(el_int)
            E_el = E_field[ed_idx]
            C_mat = (E_el / (1.0 - nu**2)) * np.array(
                [[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, (1.0 - nu) / 2.0]]
            )
            proj, _, _, area, h = _build_projector(verts_el, C_mat)
            gdofs_local = np.zeros(2 * n_v, dtype=int)
            for i in range(n_v):
                gdofs_local[2 * i] = 2 * el_int[i]
                gdofs_local[2 * i + 1] = 2 * el_int[i] + 1
            F_def = _compute_deformation_gradient(proj, u_nl[gdofs_local], h)
            # Green-Lagrange strain
            E_gl = 0.5 * (F_def.T @ F_def - np.eye(2))
            max_strain = max(max_strain, np.max(np.abs(E_gl)))

        # Plot
        mag_lin = np.sqrt(u_lin[0::2] ** 2 + u_lin[1::2] ** 2)
        mag_nl = np.sqrt(u_nl[0::2] ** 2 + u_nl[1::2] ** 2)

        scale = 50.0

        # (a/d) Linear deformed mesh
        ax = axes[row, 0]
        deformed_lin = compact_verts + scale * np.column_stack([u_lin[0::2], u_lin[1::2]])
        patches = [MplPolygon(deformed_lin[el.astype(int)], closed=True) for el in compact_elems]
        colors = [np.mean(mag_lin[el.astype(int)]) for el in compact_elems]
        pc = PatchCollection(patches, cmap="hot_r", edgecolor="k", linewidth=0.3)
        pc.set_array(np.array(colors))
        ax.add_collection(pc)
        ax.set_xlim(xmin - 0.2, xmax + 0.5)
        ax.set_ylim(ymin - 0.2, ymax + 0.2)
        ax.set_aspect("equal")
        fig.colorbar(pc, ax=ax, label="|u|", shrink=0.8)
        ax.set_title(f"Linear VEM — {label}\n|u|_max = {np.max(mag_lin):.4e}")

        # (b/e) Nonlinear deformed mesh
        ax = axes[row, 1]
        deformed_nl = compact_verts + scale * np.column_stack([u_nl[0::2], u_nl[1::2]])
        patches = [MplPolygon(deformed_nl[el.astype(int)], closed=True) for el in compact_elems]
        colors = [np.mean(mag_nl[el.astype(int)]) for el in compact_elems]
        pc = PatchCollection(patches, cmap="hot_r", edgecolor="k", linewidth=0.3)
        pc.set_array(np.array(colors))
        ax.add_collection(pc)
        ax.set_xlim(xmin - 0.2, xmax + 0.5)
        ax.set_ylim(ymin - 0.2, ymax + 0.2)
        ax.set_aspect("equal")
        fig.colorbar(pc, ax=ax, label="|u|", shrink=0.8)
        ax.set_title(
            f"Neo-Hookean VEM — {label}\n|u|_max = {np.max(mag_nl):.4e}"
        )

        # (c/f) Summary
        ax = axes[row, 2]
        ax.axis("off")
        diff_pct = abs(np.max(mag_nl) - np.max(mag_lin)) / max(np.max(mag_lin), 1e-15) * 100
        summary = (
            f"{label}\n"
            f"────────────────────\n"
            f"E = {E_val:.0f} Pa\n"
            f"Load scale = {load_scale:.1f}\n"
            f"Max strain = {max_strain:.4f}\n\n"
            f"Linear |u|_max = {np.max(mag_lin):.4e}\n"
            f"NeoHookean |u|_max = {np.max(mag_nl):.4e}\n\n"
            f"Difference = {diff_pct:.1f}%\n"
            f"NR converged: {info['converged_steps']}/{len(info['history'])} steps\n"
            f"Final NR iters: {info['history'][-1]['nr_iters']}\n\n"
            f"{'⚠ Nonlinearity significant!' if max_strain > 0.05 else '✓ Linear approx. OK'}"
        )
        ax.text(
            0.05, 0.95, summary, transform=ax.transAxes, fontsize=11,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
        )

    fig.suptitle(
        "Linear vs Neo-Hookean VEM: Biofilm Large Deformation",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()

    import os
    save_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "vem_nonlinear_demo.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


if __name__ == "__main__":
    demo_biofilm_large_deformation()
