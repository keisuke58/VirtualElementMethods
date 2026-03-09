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
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection


# ── VEM Core ──────────────────────────────────────────────────────────────

def vem_elasticity(vertices, elements, E_field, nu, bc_fixed_dofs, bc_vals,
                   load_dofs=None, load_vals=None):
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

    Returns
    -------
    u : (2*N,) displacement vector
    """
    n_nodes = vertices.shape[0]
    n_dofs = 2 * n_nodes
    n_polys = 6  # dim P_1^2 in 2D

    K_global = np.zeros((n_dofs, n_dofs))
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

        # Stabilization
        I_minus_PiD = np.eye(n_el_dofs) - D @ projector
        trace_k = np.trace(K_cons)
        stab_param = trace_k / n_el_dofs if trace_k > 0 else E_el
        K_stab = stab_param * (I_minus_PiD.T @ I_minus_PiD)

        K_local = K_cons + K_stab

        # ── Assemble ──
        gdofs = np.zeros(n_el_dofs, dtype=int)
        for i in range(n_v):
            gdofs[2 * i]     = 2 * vert_ids[i]
            gdofs[2 * i + 1] = 2 * vert_ids[i] + 1

        for i in range(n_el_dofs):
            for j in range(n_el_dofs):
                K_global[gdofs[i], gdofs[j]] += K_local[i, j]

    # ── Point loads ──
    if load_dofs is not None and load_vals is not None:
        F_global[load_dofs] += load_vals

    # ── Solve with BCs ──
    u = np.zeros(n_dofs)
    bc_set = set(bc_fixed_dofs)
    internal = np.array([i for i in range(n_dofs) if i not in bc_set])

    u[bc_fixed_dofs] = bc_vals
    F_global -= K_global[:, bc_fixed_dofs] @ bc_vals

    K_ii = K_global[np.ix_(internal, internal)]
    u[internal] = np.linalg.solve(K_ii, F_global[internal])

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
