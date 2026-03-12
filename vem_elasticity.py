"""
VEM for 2D Linear Elasticity on Polygonal Meshes.

Extension of Sutton 2017 "VEM in 50 lines" from scalar Poisson to vector elasticity.
Key change: scalar DOFs → 2 DOFs/node, 3 polynomials → 6 polynomials (P_1^2).

Polynomial basis (3 rigid body + 3 strain modes):
  p_1 = (1, 0)                     — translation x
  p_2 = (0, 1)                     — translation y
  p_3 = (-(y-yc)/h, (x-xc)/h)     — rigid rotation
  p_4 = ((x-xc)/h, 0)             — ε_xx mode
  p_5 = (0, (y-yc)/h)             — ε_yy mode
  p_6 = ((y-yc)/h, (x-xc)/h)     — ε_xy mode (symmetric shear)

References:
  - Beirao da Veiga et al. (2013) "Basic principles of VEM"
  - Gain, Talischi, Paulino (2014) "VEM for 3D elasticity on polyhedral meshes"
  - Sutton (2017) "The VEM in 50 lines of MATLAB"
"""

import numpy as np
import scipy.io
import scipy.sparse as sp
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection


# ── VEM Core ──────────────────────────────────────────────────────────────

def vem_elasticity(vertices, elements, E_field, nu, bc_fixed_dofs, bc_vals,
                   load_dofs=None, load_vals=None, stabilization_alpha=1.0):
    """
    Lowest-order VEM for 2D plane-stress linear elasticity.

    Parameters
    ----------
    vertices : (N, 2) array — node coordinates
    elements : list of int arrays — connectivity (0-based)
    E_field  : float or (N_el,) array — Young's modulus per element
    nu       : float — Poisson's ratio
    bc_fixed_dofs : array of int — constrained DOF indices (global)
    bc_vals  : array — prescribed values for fixed DOFs
    load_dofs : array of int — DOFs with applied point loads
    load_vals : array — load values
    stabilization_alpha : float — stabilization parameter (default 1.0)
        IKM/Wriggers recipe: α·trace(K_cons)/n_dof. α=1.0 (standard),
        α=0.15 (Xu/Wriggers 2025), α=0.5 (recommended for heterogeneous E).

    Returns
    -------
    u : (2*N,) displacement vector
    """
    n_nodes = vertices.shape[0]
    n_dofs = 2 * n_nodes
    n_polys = 6  # dim P_1^2 in 2D

    # C2: Sparse assembly via COO triplets (mVEM pattern)
    # Pre-estimate nnz: each element contributes (2*n_v)^2 entries
    row_idx = []
    col_idx = []
    val_data = []
    F_global = np.zeros(n_dofs)

    for el_id in range(len(elements)):
        vert_ids = elements[el_id].astype(int)
        verts = vertices[vert_ids]
        n_v = len(vert_ids)
        n_el_dofs = 2 * n_v

        # ── Element E ──
        E_el = E_field[el_id] if hasattr(E_field, '__len__') else E_field

        # Plane stress constitutive matrix (Voigt: [σ_xx, σ_yy, σ_xy])
        C = (E_el / (1.0 - nu**2)) * np.array([
            [1.0, nu,  0.0],
            [nu,  1.0, 0.0],
            [0.0, 0.0, (1.0 - nu) / 2.0]
        ])

        # ── Geometry ──
        area_comp = (verts[:, 0] * np.roll(verts[:, 1], -1)
                     - np.roll(verts[:, 0], -1) * verts[:, 1])
        area = 0.5 * abs(np.sum(area_comp))
        centroid = np.sum(
            (np.roll(verts, -1, axis=0) + verts) * area_comp[:, None],
            axis=0) / (6.0 * area)

        # Diameter
        h = max(np.linalg.norm(verts[i] - verts[j])
                for i in range(n_v) for j in range(i + 1, n_v))

        xc, yc = centroid

        # ── D matrix (n_el_dofs × 6) ──
        # Basis: p1=(1,0), p2=(0,1), p3=(-(y-yc)/h,(x-xc)/h),
        #        p4=((x-xc)/h,0), p5=(0,(y-yc)/h), p6=((y-yc)/h,(x-xc)/h)
        D = np.zeros((n_el_dofs, n_polys))
        for i in range(n_v):
            dx = (verts[i, 0] - xc) / h
            dy = (verts[i, 1] - yc) / h
            # x-component of each basis:     p1  p2  p3   p4  p5  p6
            D[2 * i,     :] = [1.0, 0.0, -dy, dx, 0.0, dy]
            # y-component of each basis:     p1  p2  p3   p4  p5  p6
            D[2 * i + 1, :] = [0.0, 1.0, dx,  0.0, dy, dx]

        # ── B matrix (6 × n_el_dofs) ──
        B = np.zeros((n_polys, n_el_dofs))

        # Vertex normals
        vertex_normals = np.zeros((n_v, 2))
        for i in range(n_v):
            prev_v = verts[(i - 1) % n_v]
            next_v = verts[(i + 1) % n_v]
            vertex_normals[i] = [next_v[1] - prev_v[1],
                                 prev_v[0] - next_v[0]]

        # Rows 0-1: translations (average displacement)
        for i in range(n_v):
            B[0, 2 * i]     = 1.0 / n_v
            B[1, 2 * i + 1] = 1.0 / n_v

        # Row 2: rigid rotation (from boundary integral of vorticity)
        # 2·area·ω_avg = ∮ (v_y n_x - v_x n_y) ds
        #              = Σ_i (1/2)(vn_x_i · v_y^i - vn_y_i · v_x^i)
        for i in range(n_v):
            B[2, 2 * i]     = -vertex_normals[i, 1] / (4.0 * area)
            B[2, 2 * i + 1] =  vertex_normals[i, 0] / (4.0 * area)

        # Rows 3-5: strain modes via boundary integrals
        # Strain of each mode (Voigt [ε_xx, ε_yy, 2ε_xy]):
        #   p4: [1/h, 0, 0]
        #   p5: [0, 1/h, 0]
        #   p6: [0, 0, 2/h]
        strain_basis = np.array([
            [1.0 / h, 0.0,     0.0],
            [0.0,     1.0 / h, 0.0],
            [0.0,     0.0,     2.0 / h],
        ])

        for i in range(n_v):
            vn = vertex_normals[i]
            for alpha in range(3):
                sigma = C @ strain_basis[alpha]  # [σ_xx, σ_yy, σ_xy]
                tx = sigma[0] * vn[0] + sigma[2] * vn[1]
                ty = sigma[2] * vn[0] + sigma[1] * vn[1]
                B[3 + alpha, 2 * i]     += 0.5 * tx
                B[3 + alpha, 2 * i + 1] += 0.5 * ty

        # ── Projector ──
        G = B @ D                           # 6 × 6
        projector = np.linalg.solve(G, B)   # 6 × n_el_dofs

        # Consistency: zero out rigid body rows (no strain energy)
        G_tilde = G.copy()
        G_tilde[:3, :] = 0.0

        K_cons = projector.T @ G_tilde @ projector

        # Stabilization (C4: Wriggers projection stabilization)
        I_minus_PiD = np.eye(n_el_dofs) - D @ projector
        # IKM/Wriggers recipe: α·trace(K_cons)/(n_el_dofs) with α∈[0.1,1]
        # Gain et al. (2014): α=1; Wriggers et al.: α tunable
        alpha_stab = stabilization_alpha
        trace_k = np.trace(K_cons)
        stab_param = alpha_stab * trace_k / n_el_dofs if trace_k > 0 else E_el
        K_stab = stab_param * (I_minus_PiD.T @ I_minus_PiD)

        K_local = K_cons + K_stab

        # ── Assemble (C2: sparse triplet) ──
        gdofs = np.zeros(n_el_dofs, dtype=int)
        for i in range(n_v):
            gdofs[2 * i]     = 2 * vert_ids[i]
            gdofs[2 * i + 1] = 2 * vert_ids[i] + 1

        ii, jj = np.meshgrid(gdofs, gdofs, indexing='ij')
        row_idx.append(ii.ravel())
        col_idx.append(jj.ravel())
        val_data.append(K_local.ravel())

    # C2: Build sparse global stiffness matrix
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


# ── Mesh Utilities ────────────────────────────────────────────────────────

def load_mesh(mesh_file):
    """Load .mat mesh, return vertices, elements, boundary."""
    mesh = scipy.io.loadmat(mesh_file)
    vertices = mesh['vertices']
    elements = np.array(
        [i[0].reshape(-1) - 1 for i in mesh['elements']], dtype=object)
    boundary = mesh['boundary'].T[0] - 1
    return vertices, elements, boundary


# ── Visualization ─────────────────────────────────────────────────────────

def plot_elasticity(vertices, elements, u, field='magnitude',
                    deform_scale=1.0, title=None, save=None):
    """Plot deformed mesh colored by displacement field."""
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

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, (coords, title_str) in enumerate([
        (vertices, 'Undeformed'),
        (deformed, f'Deformed (×{deform_scale})')
    ]):
        ax = axes[ax_idx]
        patches = []
        patch_colors = []
        for el in elements:
            el_int = el.astype(int)
            poly = MplPolygon(coords[el_int], closed=True)
            patches.append(poly)
            patch_colors.append(np.mean(vals[el_int]))

        pc = PatchCollection(patches, cmap='viridis', edgecolor='k',
                             linewidth=0.3)
        pc.set_array(np.array(patch_colors))
        ax.add_collection(pc)
        ax.set_xlim(coords[:, 0].min() - 0.05, coords[:, 0].max() + 0.05)
        ax.set_ylim(coords[:, 1].min() - 0.05, coords[:, 1].max() + 0.05)
        ax.set_aspect('equal')
        ax.set_title(title_str)
        fig.colorbar(pc, ax=ax, label=cbar_label)

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save}")
    plt.close()


# ── Demo 1: Patch Test ────────────────────────────────────────────────────

def demo_patch_test(mesh_file):
    """
    Uniform tension σ_xx = 1. Exact: u_x = x/E, u_y = -ν·y/E.
    VEM should reproduce this exactly (linear displacement).
    """
    print("=" * 60)
    print("Demo 1: Patch Test (uniform tension σ_xx = 1)")
    print("=" * 60)

    vertices, elements, boundary = load_mesh(mesh_file)

    E_mod = 1000.0
    nu = 0.3

    # Exact solution for uniform σ_xx = 1 (plane stress)
    exact_ux = vertices[:, 0] / E_mod
    exact_uy = -nu * vertices[:, 1] / E_mod

    # Prescribe exact displacement on boundary
    bc_dofs = np.concatenate([2 * boundary, 2 * boundary + 1])
    bc_vals = np.concatenate([exact_ux[boundary], exact_uy[boundary]])

    u = vem_elasticity(vertices, elements, E_mod, nu, bc_dofs, bc_vals)

    ux = u[0::2]
    uy = u[1::2]

    err_x = np.max(np.abs(ux - exact_ux))
    err_y = np.max(np.abs(uy - exact_uy))
    print(f"  Max error u_x: {err_x:.2e}")
    print(f"  Max error u_y: {err_y:.2e}")
    passed = max(err_x, err_y) < 1e-10
    print(f"  PASS: {passed}")
    return passed


# ── Demo 2: Cantilever Beam ───────────────────────────────────────────────

def demo_cantilever(mesh_file, save_dir='/tmp'):
    """
    Cantilever: fix left edge (x≈0), point load downward on right edge (x≈1).
    """
    print("\n" + "=" * 60)
    print("Demo 2: Cantilever Beam on Polygonal Mesh")
    print("=" * 60)

    vertices, elements, boundary = load_mesh(mesh_file)

    E_mod = 1000.0
    nu = 0.3
    tol = 1e-6

    # Fix left edge
    left = np.where(vertices[:, 0] < tol)[0]
    bc_dofs = np.concatenate([2 * left, 2 * left + 1])
    bc_vals = np.zeros(len(bc_dofs))

    # Downward load on right edge
    right = np.where(vertices[:, 0] > 1.0 - tol)[0]
    load_per_node = -1.0 / len(right)
    load_dofs = 2 * right + 1  # y-DOF
    load_vals = np.full(len(right), load_per_node)

    print(f"  Fixed nodes (left):   {len(left)}")
    print(f"  Loaded nodes (right): {len(right)}")

    u = vem_elasticity(vertices, elements, E_mod, nu, bc_dofs, bc_vals,
                       load_dofs, load_vals)

    ux = u[0::2]
    uy = u[1::2]
    print(f"  Max |u_x|: {np.max(np.abs(ux)):.6f}")
    print(f"  Max |u_y|: {np.max(np.abs(uy)):.6f}")
    print(f"  Tip deflection (avg right u_y): {np.mean(uy[right]):.6f}")

    plot_elasticity(vertices, elements, u, field='magnitude',
                    deform_scale=100,
                    title='VEM Cantilever (Voronoi mesh)',
                    save=f'{save_dir}/vem_cantilever.png')
    return u


# ── Demo 3: Biofilm-Inspired E(DI) ───────────────────────────────────────

def demo_biofilm_edi(mesh_file, save_dir='/tmp'):
    """
    Spatially varying E = E(DI) on polygonal mesh.
    E(DI) = E_min + (E_max - E_min) * (1 - DI)^n
    DI: high at center (dysbiotic core), low at edges (commensal periphery).
    """
    print("\n" + "=" * 60)
    print("Demo 3: Biofilm-Inspired E(DI) on Polygonal Mesh")
    print("=" * 60)

    vertices, elements, boundary = load_mesh(mesh_file)

    # Material parameters (from our biofilm model)
    E_max = 1000.0  # Pa (commensal)
    E_min = 30.0    # Pa (dysbiotic)
    n_hill = 2
    nu = 0.3

    # DI field: high at center, low at edges
    center = np.array([0.5, 0.5])
    max_dist = 0.5 * np.sqrt(2)

    E_per_element = np.zeros(len(elements))
    DI_per_element = np.zeros(len(elements))
    for i, el in enumerate(elements):
        el_int = el.astype(int)
        el_centroid = vertices[el_int].mean(axis=0)
        dist = np.linalg.norm(el_centroid - center)
        DI = 0.9 - 0.8 * (dist / max_dist)
        DI = np.clip(DI, 0.05, 0.95)
        DI_per_element[i] = DI
        E_per_element[i] = E_min + (E_max - E_min) * (1.0 - DI) ** n_hill

    print(f"  DI range: [{DI_per_element.min():.2f}, {DI_per_element.max():.2f}]")
    print(f"  E  range: [{E_per_element.min():.0f}, {E_per_element.max():.0f}] Pa")
    print(f"  E ratio:  {E_per_element.max() / E_per_element.min():.1f}x")

    tol = 1e-6

    # Fix bottom edge
    bottom = np.where(vertices[:, 1] < tol)[0]
    bc_dofs = np.concatenate([2 * bottom, 2 * bottom + 1])
    bc_vals = np.zeros(len(bc_dofs))

    # Uniform pressure on top edge
    top = np.where(vertices[:, 1] > 1.0 - tol)[0]
    load_per_node = -0.5 / len(top)
    load_dofs = 2 * top + 1
    load_vals = np.full(len(top), load_per_node)

    u = vem_elasticity(vertices, elements, E_per_element, nu,
                       bc_dofs, bc_vals, load_dofs, load_vals)

    ux = u[0::2]
    uy = u[1::2]
    print(f"  Max |u|: {np.max(np.sqrt(ux**2 + uy**2)):.6f}")

    # ── Plot: 3-panel (DI, E, displacement) ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax_idx, (data, cmap, label) in enumerate([
        (DI_per_element, 'RdYlGn_r', 'Dysbiosis Index'),
        (E_per_element, 'viridis', "Young's Modulus E [Pa]"),
        (None, 'hot_r', 'Displacement |u|'),
    ]):
        ax = axes[ax_idx]
        patches = []
        colors = []
        for i, el in enumerate(elements):
            el_int = el.astype(int)
            if ax_idx < 2:
                coords = vertices[el_int]
                colors.append(data[i])
            else:
                deformed = vertices + 200 * np.column_stack([ux, uy])
                coords = deformed[el_int]
                u_mag = np.sqrt(ux[el_int]**2 + uy[el_int]**2)
                colors.append(np.mean(u_mag))
            patches.append(MplPolygon(coords, closed=True))

        pc = PatchCollection(patches, cmap=cmap, edgecolor='k',
                             linewidth=0.2)
        pc.set_array(np.array(colors))
        ax.add_collection(pc)

        if ax_idx < 2:
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
        else:
            ax.set_xlim(deformed[:, 0].min() - 0.05,
                        deformed[:, 0].max() + 0.05)
            ax.set_ylim(deformed[:, 1].min() - 0.05,
                        deformed[:, 1].max() + 0.05)
        ax.set_aspect('equal')
        fig.colorbar(pc, ax=ax, label=label, shrink=0.8)

    axes[0].set_title('DI Field')
    axes[1].set_title('E(DI) Distribution')
    axes[2].set_title('Deformed (x200)')
    fig.suptitle('VEM + E(DI): Biofilm Mechanical Response',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = f'{save_dir}/vem_biofilm_edi.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close()

    return u


# ── C3: Mixed (u,p) VEM Formulation ──────────────────────────────────────

def vem_elasticity_mixed(vertices, elements, E_field, nu, bc_fixed_dofs, bc_vals,
                         load_dofs=None, load_vals=None):
    """
    Mixed (u,p) VEM for near-incompressible 2D plane-strain elasticity.
    Avoids volumetric locking by treating pressure as independent variable.

    Saddle-point system:
        [K   G] [u]   [f]
        [G^T 0] [p] = [0]

    where K = 2μ deviatoric stiffness, G = divergence coupling, p = -λ div(u).
    """
    n_nodes = vertices.shape[0]
    n_dofs_u = 2 * n_nodes
    n_els = len(elements)
    n_dofs_total = n_dofs_u + n_els  # 1 pressure DOF per element (P0)

    # Sparse assembly
    row_K, col_K, val_K = [], [], []
    row_G, col_G, val_G = [], [], []
    row_M, col_M, val_M = [], [], []  # pressure mass matrix (1/λ)
    F_global = np.zeros(n_dofs_total)

    for el_id in range(n_els):
        vert_ids = elements[el_id].astype(int)
        verts = vertices[vert_ids]
        n_v = len(vert_ids)
        n_el_dofs = 2 * n_v

        E_el = E_field[el_id] if hasattr(E_field, '__len__') else E_field

        # Lamé parameters (plane strain)
        mu = E_el / (2.0 * (1.0 + nu))
        lam = E_el * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

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

        # D and B matrices (same as displacement VEM)
        n_polys = 6
        D = np.zeros((n_el_dofs, n_polys))
        for i in range(n_v):
            dx = (verts[i, 0] - xc) / h
            dy = (verts[i, 1] - yc) / h
            D[2*i,     :] = [1.0, 0.0, -dy, dx, 0.0, dy]
            D[2*i + 1, :] = [0.0, 1.0,  dx, 0.0, dy, dx]

        B = np.zeros((n_polys, n_el_dofs))
        vertex_normals = np.zeros((n_v, 2))
        for i in range(n_v):
            prev_v = verts[(i - 1) % n_v]
            next_v = verts[(i + 1) % n_v]
            vertex_normals[i] = [next_v[1] - prev_v[1], prev_v[0] - next_v[0]]

        for i in range(n_v):
            B[0, 2*i] = 1.0 / n_v
            B[1, 2*i+1] = 1.0 / n_v
            B[2, 2*i]     = -vertex_normals[i, 1] / (4.0 * area)
            B[2, 2*i + 1] =  vertex_normals[i, 0] / (4.0 * area)

        # Deviatoric constitutive (2μ only, no λ)
        C_dev = 2.0 * mu * np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.5],
        ])
        strain_basis = np.array([
            [1.0/h, 0.0,   0.0],
            [0.0,   1.0/h, 0.0],
            [0.0,   0.0,   2.0/h],
        ])
        for i in range(n_v):
            vn = vertex_normals[i]
            for alpha in range(3):
                sigma = C_dev @ strain_basis[alpha]
                tx = sigma[0] * vn[0] + sigma[2] * vn[1]
                ty = sigma[2] * vn[0] + sigma[1] * vn[1]
                B[3 + alpha, 2*i]     += 0.5 * tx
                B[3 + alpha, 2*i + 1] += 0.5 * ty

        G_mat = B @ D
        projector = np.linalg.solve(G_mat, B)

        G_tilde = G_mat.copy()
        G_tilde[:3, :] = 0.0
        K_cons = projector.T @ G_tilde @ projector
        I_minus_PiD = np.eye(n_el_dofs) - D @ projector
        trace_k = np.trace(K_cons)
        stab_param = trace_k / n_el_dofs if trace_k > 0 else 2.0 * mu
        K_stab = stab_param * (I_minus_PiD.T @ I_minus_PiD)
        K_local = K_cons + K_stab

        # G_local: divergence coupling (u → p)
        # div(u) ≈ (1/area) Σ_i u_i · n_i / 2
        G_local = np.zeros(n_el_dofs)
        for i in range(n_v):
            G_local[2*i]     = vertex_normals[i, 0] / 2.0
            G_local[2*i + 1] = vertex_normals[i, 1] / 2.0

        # Assemble K
        gdofs = np.zeros(n_el_dofs, dtype=int)
        for i in range(n_v):
            gdofs[2*i]     = 2 * vert_ids[i]
            gdofs[2*i + 1] = 2 * vert_ids[i] + 1

        ii, jj = np.meshgrid(gdofs, gdofs, indexing='ij')
        row_K.append(ii.ravel())
        col_K.append(jj.ravel())
        val_K.append(K_local.ravel())

        # Assemble G (coupling block): rows = u DOFs, col = pressure DOF (element-local index)
        p_idx = el_id  # pressure DOF index within pressure block
        for i in range(n_el_dofs):
            row_G.append(gdofs[i])
            col_G.append(p_idx)
            val_G.append(-G_local[i])

        # Pressure mass: (1/λ) * area
        row_M.append(p_idx)
        col_M.append(p_idx)
        val_M.append(area / lam)

    # Build sparse system
    row_K = np.concatenate(row_K)
    col_K = np.concatenate(col_K)
    val_K = np.concatenate(val_K)
    K_sp = sp.csr_matrix((val_K, (row_K, col_K)), shape=(n_dofs_u, n_dofs_u))

    row_G = np.array(row_G)
    col_G = np.array(col_G)
    val_G = np.array(val_G)
    G_sp = sp.csr_matrix((val_G, (row_G, col_G)), shape=(n_dofs_u, n_els))

    row_M = np.array(row_M)
    col_M = np.array(col_M)
    val_M = np.array(val_M)
    M_p = sp.csr_matrix((val_M, (row_M, col_M)), shape=(n_els, n_els))

    # Saddle-point: [K, G; G^T, -M_p] [u; p] = [f; 0]
    A = sp.bmat([
        [K_sp,    G_sp],
        [G_sp.T, -M_p],
    ], format='csr')

    F = np.zeros(n_dofs_total)
    if load_dofs is not None and load_vals is not None:
        F[load_dofs] += load_vals

    # BCs (displacement only, pressure is free)
    sol = np.zeros(n_dofs_total)
    bc_set = set(bc_fixed_dofs)
    internal = np.array([i for i in range(n_dofs_total) if i not in bc_set])

    sol[bc_fixed_dofs] = bc_vals
    F -= A[:, bc_fixed_dofs].toarray() @ bc_vals

    A_ii = A[np.ix_(internal, internal)]
    sol[internal] = sp.linalg.spsolve(A_ii, F[internal])

    u = sol[:n_dofs_u]
    p = sol[n_dofs_u:]
    return u, p


# ── C1: Two-Way Picard Coupling ─────────────────────────────────────────

def stress_dependent_diffusivity(sigma_vol, m0=0.1, m1=1e-4):
    """
    Stress-dependent diffusivity: M = m0 * exp(-m1 * σ_vol).
    Higher compressive stress → lower diffusivity → reduced growth.

    Args:
        sigma_vol: volumetric stress per element (trace(σ)/2 for 2D)
        m0: baseline diffusivity
        m1: stress sensitivity (positive = compression reduces diffusion)
    """
    return m0 * np.exp(-m1 * sigma_vol)


def compute_element_stress(vertices, elements, u, E_field, nu):
    """
    Compute element-wise volumetric stress from VEM displacement field.
    Returns σ_vol = (σ_xx + σ_yy) / 2 per element.
    """
    n_els = len(elements)
    sigma_vol = np.zeros(n_els)

    for el_id in range(n_els):
        vert_ids = elements[el_id].astype(int)
        verts = vertices[vert_ids]
        n_v = len(vert_ids)

        E_el = E_field[el_id] if hasattr(E_field, '__len__') else E_field
        mu = E_el / (2.0 * (1.0 + nu))
        lam = E_el * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

        # Compute div(u) via boundary integral: div(u) = (1/area) * Σ u_i · n_i / 2
        area_comp = (verts[:, 0] * np.roll(verts[:, 1], -1)
                     - np.roll(verts[:, 0], -1) * verts[:, 1])
        area = 0.5 * abs(np.sum(area_comp))

        vertex_normals = np.zeros((n_v, 2))
        for i in range(n_v):
            prev_v = verts[(i - 1) % n_v]
            next_v = verts[(i + 1) % n_v]
            vertex_normals[i] = [next_v[1] - prev_v[1], prev_v[0] - next_v[0]]

        div_u = 0.0
        for i in range(n_v):
            nid = vert_ids[i]
            div_u += u[2*nid] * vertex_normals[i, 0] + u[2*nid+1] * vertex_normals[i, 1]
        div_u /= (2.0 * area)

        # σ_vol = λ * div(u) + μ * div(u) = (λ + μ) * div(u) for isotropic
        # Actually: tr(σ)/2 = (λ + μ) * div(u)  (plane strain)
        sigma_vol[el_id] = (lam + mu) * div_u

    return sigma_vol


def picard_coupled_solve(vertices, elements, E_base, nu,
                         bc_fixed_dofs, bc_vals, load_dofs=None, load_vals=None,
                         growth_rate_fn=None, m0=0.1, m1=1e-4,
                         tol=1e-8, max_iter=20, verbose=True):
    """
    Two-way Picard coupling: Elasticity ↔ Growth/Diffusion.

    Loop:
        1. Solve elasticity → u, σ
        2. Compute stress-dependent diffusivity M(σ)
        3. Update growth rates / E(DI) based on M
        4. Check convergence: ||u_k - u_{k-1}|| < tol
        5. Repeat until converged

    Args:
        vertices, elements: VEM mesh
        E_base: float or (n_els,) base Young's modulus
        nu: Poisson's ratio
        bc_fixed_dofs, bc_vals: displacement BCs
        load_dofs, load_vals: applied loads
        growth_rate_fn: callable(DI, M_field) → E_field
            Takes DI per element and diffusivity, returns updated E per element.
            If None, uses default E(DI) = E_min + (E_max - E_min) * (1 - DI)^n
        m0, m1: stress-dependent diffusivity parameters
        tol: convergence tolerance (relative L2 norm of u change)
        max_iter: maximum Picard iterations
        verbose: print convergence info

    Returns:
        u: converged displacement
        E_field: converged E per element
        info: dict with convergence history
    """
    n_nodes = vertices.shape[0]
    n_els = len(elements)

    # Default growth rate function: stress modulates DI → E
    E_MAX, E_MIN, N_HILL = 1000.0, 30.0, 2
    if growth_rate_fn is None:
        # Compute initial DI from E_base
        if hasattr(E_base, '__len__'):
            DI_field = 1.0 - ((np.array(E_base) - E_MIN) / (E_MAX - E_MIN)) ** (1.0 / N_HILL)
            DI_field = np.clip(DI_field, 0.01, 0.99)
        else:
            DI_field = np.full(n_els, 0.5)

        def growth_rate_fn(DI, M_field):
            # Stress reduces effective DI: higher stress → lower diffusivity → higher DI shift
            M_ratio = M_field / m0  # normalized: 1.0 = no stress effect
            # Feedback: compression (M_ratio < 1) → DI increases slightly
            DI_eff = np.clip(DI + 0.1 * (1.0 - M_ratio), 0.01, 0.99)
            return E_MIN + (E_MAX - E_MIN) * (1.0 - DI_eff) ** N_HILL
    else:
        if hasattr(E_base, '__len__'):
            DI_field = 1.0 - ((np.array(E_base) - E_MIN) / (E_MAX - E_MIN)) ** (1.0 / N_HILL)
            DI_field = np.clip(DI_field, 0.01, 0.99)
        else:
            DI_field = np.full(n_els, 0.5)

    E_field = np.array(E_base) if hasattr(E_base, '__len__') else np.full(n_els, E_base)
    u_prev = np.zeros(2 * n_nodes)
    history = []

    for it in range(max_iter):
        # Step 1: Solve elasticity
        u = vem_elasticity(vertices, elements, E_field, nu,
                          bc_fixed_dofs, bc_vals, load_dofs, load_vals)

        # Step 2: Compute element stresses
        sigma_vol = compute_element_stress(vertices, elements, u, E_field, nu)

        # Step 3: Stress → diffusivity → growth → E update
        M_field = stress_dependent_diffusivity(sigma_vol, m0, m1)
        E_field = growth_rate_fn(DI_field, M_field)
        E_field = np.clip(E_field, E_MIN, E_MAX)

        # Step 4: Convergence check
        du = np.linalg.norm(u - u_prev)
        u_norm = np.linalg.norm(u) + 1e-15
        rel_change = du / u_norm
        history.append({'iter': it + 1, 'rel_change': rel_change,
                        'E_range': (E_field.min(), E_field.max()),
                        'M_range': (M_field.min(), M_field.max())})

        if verbose:
            print(f"  Picard iter {it+1}: ||Δu||/||u|| = {rel_change:.2e}, "
                  f"E=[{E_field.min():.0f}, {E_field.max():.0f}], "
                  f"M=[{M_field.min():.4f}, {M_field.max():.4f}]")

        if rel_change < tol and it > 0:
            if verbose:
                print(f"  Converged in {it+1} iterations (tol={tol:.0e})")
            break

        u_prev = u.copy()
    else:
        if verbose:
            print(f"  Warning: max iterations ({max_iter}) reached, "
                  f"rel_change={rel_change:.2e}")

    info = {
        'n_iter': it + 1,
        'converged': rel_change < tol,
        'history': history,
        'sigma_vol': sigma_vol,
        'M_field': M_field,
        'DI_field': DI_field,
    }
    return u, E_field, info


# ── C6: VEM Mass Matrix ──────────────────────────────────────────────────

def assemble_mass_matrix(vertices, elements, rho_field, lumped=False):
    """
    Consistent or lumped VEM mass matrix for 2D elasticity.

    M_cons = ρ · (Πˢ)ᵀ · H · Πˢ + α · trace(M_cons)/n · (I - Π D)ᵀ(I - Π D)

    where H = ∫_K m_α(x) m_β(x) dK is the monomial Gram matrix,
    evaluated via sub-triangulation quadrature.

    Parameters
    ----------
    vertices : (N, 2) — node coordinates
    elements : list of int arrays — connectivity
    rho_field : float or (N_el,) — density per element
    lumped : bool — if True, row-sum lumping (diagonal M)

    Returns
    -------
    M : sparse CSR matrix (2N × 2N) — mass matrix
    """
    n_nodes = vertices.shape[0]
    n_dofs = 2 * n_nodes
    n_polys = 6

    row_idx, col_idx, val_data = [], [], []

    for el_id in range(len(elements)):
        vert_ids = elements[el_id].astype(int)
        verts = vertices[vert_ids]
        n_v = len(vert_ids)
        n_el_dofs = 2 * n_v

        rho = rho_field[el_id] if hasattr(rho_field, '__len__') else rho_field

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

        # D matrix
        D = np.zeros((n_el_dofs, n_polys))
        for i in range(n_v):
            dx = (verts[i, 0] - xc) / h
            dy = (verts[i, 1] - yc) / h
            D[2*i,     :] = [1.0, 0.0, -dy, dx, 0.0, dy]
            D[2*i + 1, :] = [0.0, 1.0,  dx, 0.0, dy, dx]

        # B matrix (same as stiffness — need projector)
        B = np.zeros((n_polys, n_el_dofs))
        vertex_normals = np.zeros((n_v, 2))
        for i in range(n_v):
            prev_v = verts[(i - 1) % n_v]
            next_v = verts[(i + 1) % n_v]
            vertex_normals[i] = [next_v[1] - prev_v[1], prev_v[0] - next_v[0]]

        for i in range(n_v):
            B[0, 2*i] = 1.0 / n_v
            B[1, 2*i+1] = 1.0 / n_v
            B[2, 2*i]     = -vertex_normals[i, 1] / (4.0 * area)
            B[2, 2*i + 1] =  vertex_normals[i, 0] / (4.0 * area)

        # For mass matrix, strain rows not needed — just use D, B[:3] for projector
        # But we need the full projector for (I - Π D) stabilization
        # Use identity constitutive for B rows 3-5
        for i in range(n_v):
            vn = vertex_normals[i]
            strain_basis = np.array([
                [1.0/h, 0.0, 0.0], [0.0, 1.0/h, 0.0], [0.0, 0.0, 2.0/h]])
            for alpha in range(3):
                # Use identity (no material weighting for mass)
                B[3+alpha, 2*i]     += 0.5 * (strain_basis[alpha, 0] * vn[0]
                                              + strain_basis[alpha, 2] * vn[1])
                B[3+alpha, 2*i + 1] += 0.5 * (strain_basis[alpha, 2] * vn[0]
                                              + strain_basis[alpha, 1] * vn[1])

        G = B @ D
        projector = np.linalg.solve(G, B)

        # H matrix: ∫_K m_α m_β dK via sub-triangulation
        # For VEM k=1: monomials are {(1,0), (0,1), (-dy/h, dx/h),
        #                              (dx/h,0), (0,dy/h), (dy/h,dx/h)}
        # Simplified: H_αβ = ∫_K pα · pβ dK
        # Use centroid-based quadrature (1st order accurate, sufficient for k=1)
        H = np.zeros((n_polys, n_polys))
        # Sub-triangulate from centroid
        for i in range(n_v):
            j = (i + 1) % n_v
            # Triangle: centroid, v_i, v_j
            tri = np.array([centroid, verts[i], verts[j]])
            tri_area = 0.5 * abs(
                (tri[1, 0] - tri[0, 0]) * (tri[2, 1] - tri[0, 1])
                - (tri[2, 0] - tri[0, 0]) * (tri[1, 1] - tri[0, 1]))

            # 3-point triangle quadrature (midpoints, order 2)
            qpts = np.array([
                0.5 * (tri[0] + tri[1]),
                0.5 * (tri[1] + tri[2]),
                0.5 * (tri[0] + tri[2]),
            ])
            qw = tri_area / 3.0

            for qp in qpts:
                dx_q = (qp[0] - xc) / h
                dy_q = (qp[1] - yc) / h
                # Evaluate each vector monomial at quadrature point
                # p_α(qp) is a 2-vector; dot product gives scalar
                m_vals = np.array([
                    [1.0, 0.0],     # p1
                    [0.0, 1.0],     # p2
                    [-dy_q, dx_q],  # p3
                    [dx_q, 0.0],    # p4
                    [0.0, dy_q],    # p5
                    [dy_q, dx_q],   # p6
                ])
                for a in range(n_polys):
                    for b in range(n_polys):
                        H[a, b] += qw * np.dot(m_vals[a], m_vals[b])

        # Consistent mass: M = ρ · Πˢᵀ · H · Πˢ + stabilization
        M_cons = rho * projector.T @ H @ projector

        # Stabilization for mass (same structure as stiffness)
        I_minus_PiD = np.eye(n_el_dofs) - D @ projector
        trace_m = np.trace(M_cons)
        stab_m = trace_m / n_el_dofs if trace_m > 0 else rho * area / n_el_dofs
        M_local = M_cons + stab_m * (I_minus_PiD.T @ I_minus_PiD)

        # Assemble
        gdofs = np.zeros(n_el_dofs, dtype=int)
        for i in range(n_v):
            gdofs[2*i]     = 2 * vert_ids[i]
            gdofs[2*i + 1] = 2 * vert_ids[i] + 1

        ii, jj = np.meshgrid(gdofs, gdofs, indexing='ij')
        row_idx.append(ii.ravel())
        col_idx.append(jj.ravel())
        val_data.append(M_local.ravel())

    row_idx = np.concatenate(row_idx)
    col_idx = np.concatenate(col_idx)
    val_data = np.concatenate(val_data)
    M = sp.csr_matrix((val_data, (row_idx, col_idx)), shape=(n_dofs, n_dofs))

    if lumped:
        diag = np.array(M.sum(axis=1)).ravel()
        M = sp.diags(diag, format='csr')

    return M


# ── Elastodynamics (Newmark-β) ───────────────────────────────────────────

def vem_elastodynamics(vertices, elements, E_field, nu, rho_field,
                       bc_fixed_dofs, bc_vals, load_func,
                       dt=0.01, n_steps=100, beta_nm=0.25, gamma_nm=0.5,
                       stabilization_alpha=1.0, lumped_mass=False,
                       damping_alpha=0.0, damping_beta=0.0):
    """
    VEM elastodynamics via Newmark-β time integration.

    Solves: M·ü + C·u̇ + K·u = F(t)

    Parameters
    ----------
    vertices, elements : VEM mesh
    E_field : float or (N_el,) — Young's modulus
    nu : float — Poisson's ratio
    rho_field : float or (N_el,) — density
    bc_fixed_dofs : constrained DOFs
    bc_vals : prescribed values (constant in time)
    load_func : callable(t, n_dofs) → F vector, or None for zero load
    dt : time step
    n_steps : number of time steps
    beta_nm, gamma_nm : Newmark parameters (0.25, 0.5 = implicit, unconditionally stable)
    stabilization_alpha : VEM stabilization parameter
    lumped_mass : use lumped mass matrix
    damping_alpha, damping_beta : Rayleigh damping C = α·M + β·K

    Returns
    -------
    t_hist : (n_steps+1,) time array
    u_hist : (n_steps+1, 2N) displacement history
    v_hist : (n_steps+1, 2N) velocity history
    """
    n_nodes = vertices.shape[0]
    n_dofs = 2 * n_nodes

    # Assemble K and M
    K = _assemble_stiffness_sparse(vertices, elements, E_field, nu,
                                    stabilization_alpha)
    M = assemble_mass_matrix(vertices, elements, rho_field, lumped=lumped_mass)

    # Rayleigh damping
    C_damp = damping_alpha * M + damping_beta * K

    # Effective stiffness
    K_eff = M + gamma_nm * dt * C_damp + beta_nm * dt**2 * K

    # Apply BCs to K_eff
    bc_set = set(bc_fixed_dofs)
    internal = np.array([i for i in range(n_dofs) if i not in bc_set])

    K_eff_ii = K_eff[np.ix_(internal, internal)]
    # Regularize for near-singular cases (disconnected nodes in random meshes)
    K_eff_ii = K_eff_ii + 1e-12 * sp.eye(K_eff_ii.shape[0], format='csr')
    K_eff_factored = sp.linalg.splu(K_eff_ii.tocsc())

    # Initialize
    u = np.zeros(n_dofs)
    u[bc_fixed_dofs] = bc_vals
    v = np.zeros(n_dofs)
    a = np.zeros(n_dofs)

    # Initial acceleration: M·a = F(0) - K·u - C·v
    F0 = load_func(0.0, n_dofs) if load_func is not None else np.zeros(n_dofs)
    rhs_init = F0 - K @ u - C_damp @ v
    if lumped_mass:
        M_diag = M.diagonal()
        a[internal] = rhs_init[internal] / M_diag[internal]
    else:
        M_ii = M[np.ix_(internal, internal)]
        M_ii = M_ii + 1e-12 * sp.eye(M_ii.shape[0], format='csr')
        a[internal] = sp.linalg.spsolve(M_ii, rhs_init[internal])

    t_hist = np.zeros(n_steps + 1)
    u_hist = np.zeros((n_steps + 1, n_dofs))
    v_hist = np.zeros((n_steps + 1, n_dofs))
    u_hist[0] = u.copy()
    v_hist[0] = v.copy()

    for step in range(n_steps):
        t_new = (step + 1) * dt
        t_hist[step + 1] = t_new

        # Predictors
        u_pred = u + dt * v + 0.5 * dt**2 * (1 - 2*beta_nm) * a
        v_pred = v + dt * (1 - gamma_nm) * a

        # External force
        F_ext = load_func(t_new, n_dofs) if load_func is not None else np.zeros(n_dofs)

        # RHS
        rhs = (M + gamma_nm * dt * C_damp) @ a * 0  # zero
        rhs = F_ext - K @ u_pred - C_damp @ v_pred

        # Solve for acceleration increment
        a_new = np.zeros(n_dofs)
        rhs_bc = rhs.copy()
        rhs_bc -= K_eff[:, bc_fixed_dofs].toarray() @ np.zeros(len(bc_fixed_dofs))
        a_new[internal] = K_eff_factored.solve(rhs_bc[internal])

        # Correctors
        u_new = u_pred + beta_nm * dt**2 * a_new
        v_new = v_pred + gamma_nm * dt * a_new

        # Enforce BCs
        u_new[bc_fixed_dofs] = bc_vals
        v_new[bc_fixed_dofs] = 0.0
        a_new[bc_fixed_dofs] = 0.0

        u, v, a = u_new, v_new, a_new
        u_hist[step + 1] = u.copy()
        v_hist[step + 1] = v.copy()

    return t_hist, u_hist, v_hist


def _assemble_stiffness_sparse(vertices, elements, E_field, nu,
                                stabilization_alpha=1.0):
    """Assemble stiffness matrix only (no solve), returns sparse CSR."""
    n_nodes = vertices.shape[0]
    n_dofs = 2 * n_nodes
    n_polys = 6

    row_idx, col_idx, val_data = [], [], []

    for el_id in range(len(elements)):
        vert_ids = elements[el_id].astype(int)
        verts = vertices[vert_ids]
        n_v = len(vert_ids)
        n_el_dofs = 2 * n_v

        E_el = E_field[el_id] if hasattr(E_field, '__len__') else E_field
        C = (E_el / (1.0 - nu**2)) * np.array([
            [1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, (1.0 - nu) / 2.0]])

        area_comp = (verts[:, 0] * np.roll(verts[:, 1], -1)
                     - np.roll(verts[:, 0], -1) * verts[:, 1])
        area = 0.5 * abs(np.sum(area_comp))
        centroid = np.sum(
            (np.roll(verts, -1, axis=0) + verts) * area_comp[:, None],
            axis=0) / (6.0 * area)
        h = max(np.linalg.norm(verts[i] - verts[j])
                for i in range(n_v) for j in range(i + 1, n_v))
        xc, yc = centroid

        D = np.zeros((n_el_dofs, n_polys))
        for i in range(n_v):
            dx = (verts[i, 0] - xc) / h
            dy = (verts[i, 1] - yc) / h
            D[2*i,     :] = [1.0, 0.0, -dy, dx, 0.0, dy]
            D[2*i + 1, :] = [0.0, 1.0,  dx, 0.0, dy, dx]

        B = np.zeros((n_polys, n_el_dofs))
        vertex_normals = np.zeros((n_v, 2))
        for i in range(n_v):
            prev_v = verts[(i - 1) % n_v]
            next_v = verts[(i + 1) % n_v]
            vertex_normals[i] = [next_v[1] - prev_v[1], prev_v[0] - next_v[0]]

        for i in range(n_v):
            B[0, 2*i] = 1.0 / n_v
            B[1, 2*i+1] = 1.0 / n_v
            B[2, 2*i]     = -vertex_normals[i, 1] / (4.0 * area)
            B[2, 2*i + 1] =  vertex_normals[i, 0] / (4.0 * area)

        strain_basis = np.array([
            [1.0/h, 0.0, 0.0], [0.0, 1.0/h, 0.0], [0.0, 0.0, 2.0/h]])
        for i in range(n_v):
            vn = vertex_normals[i]
            for alpha in range(3):
                sigma = C @ strain_basis[alpha]
                B[3+alpha, 2*i]     += 0.5 * (sigma[0]*vn[0] + sigma[2]*vn[1])
                B[3+alpha, 2*i + 1] += 0.5 * (sigma[2]*vn[0] + sigma[1]*vn[1])

        G = B @ D
        projector = np.linalg.solve(G, B)
        G_tilde = G.copy()
        G_tilde[:3, :] = 0.0
        K_cons = projector.T @ G_tilde @ projector

        I_minus_PiD = np.eye(n_el_dofs) - D @ projector
        trace_k = np.trace(K_cons)
        stab_param = stabilization_alpha * trace_k / n_el_dofs if trace_k > 0 else E_el
        K_local = K_cons + stab_param * (I_minus_PiD.T @ I_minus_PiD)

        gdofs = np.zeros(n_el_dofs, dtype=int)
        for i in range(n_v):
            gdofs[2*i]     = 2 * vert_ids[i]
            gdofs[2*i + 1] = 2 * vert_ids[i] + 1

        ii, jj = np.meshgrid(gdofs, gdofs, indexing='ij')
        row_idx.append(ii.ravel())
        col_idx.append(jj.ravel())
        val_data.append(K_local.ravel())

    row_idx = np.concatenate(row_idx)
    col_idx = np.concatenate(col_idx)
    val_data = np.concatenate(val_data)
    return sp.csr_matrix((val_data, (row_idx, col_idx)), shape=(n_dofs, n_dofs))


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import os
    mesh_dir = os.path.join(os.path.dirname(__file__), 'meshes')

    # Demo 1: Patch test on Voronoi mesh
    passed = demo_patch_test(os.path.join(mesh_dir, 'voronoi.mat'))

    # Demo 2: Cantilever on Voronoi mesh
    demo_cantilever(os.path.join(mesh_dir, 'voronoi.mat'))

    # Demo 3: Biofilm E(DI) on smoothed Voronoi mesh
    demo_biofilm_edi(os.path.join(mesh_dir, 'smoothed-voronoi.mat'))

    print("\n" + "=" * 60)
    print("All demos complete.")
    print("=" * 60)
