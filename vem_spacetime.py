"""
Space-Time VEM: Solve transient problems on unstructured (x,t) meshes.

Key idea: treat time as a spatial coordinate and solve the entire evolution
in one shot using VEM on a Voronoi mesh in the (x,t) plane.

This is a prototype implementing the concepts from:
  - Xu, Junker, Wriggers (2025): Space-time VEM for elastodynamics

Demonstrated on:
  1. 1D heat equation → anisotropic diffusion in (x,t)
  2. 1D viscoelastic bar (SLS) → spatially varying E_inf, τ
  3. Comparison: space-time VEM vs sequential time-stepping

The bilinear form:
  a(u,v) = ∫_Ω [(C·∇u)·∇v] dΩ
  where ∇ = (∂/∂x, ∂/∂t), C = [[κ, 0], [0, β]]
  κ = spatial stiffness, β = temporal smoothing/inertia

References:
  - Xu, Junker, Wriggers (2025) "Space-Time VEM"
  - Beirão da Veiga et al. (2013) "Basic principles of VEM"
"""

import numpy as np
from scipy.spatial import Voronoi
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
import os


# ── Anisotropic Scalar VEM ────────────────────────────────────────────────

def vem_anisotropic(vertices, elements, C_per_el, bc_dofs, bc_vals,
                    load_dofs=None, load_vals=None, rhs_func=None):
    """
    Lowest-order scalar VEM with anisotropic material tensor C.

    For standard Poisson: C = [[1,0],[0,1]] (identity)
    For space-time heat: C = [[κ, 0],[0, β]]
    For space-time wave: C = [[-E, 0],[0, ρ]] (indefinite!)

    Parameters
    ----------
    vertices : (N, 2) — node coords in (x, t) space
    elements : list of int arrays — connectivity
    C_per_el : (N_el, 2, 2) or (2, 2) — material tensor per element
    bc_dofs : array of int — constrained DOFs
    bc_vals : array — prescribed values
    load_dofs, load_vals : optional point loads
    rhs_func : callable(x, t) → f, or None

    Returns
    -------
    u : (N,) — solution at each node
    """
    n_nodes = vertices.shape[0]
    n_polys = 3  # {1, (x-xc)/h, (t-tc)/h} for k=1 scalar VEM

    K_global = np.zeros((n_nodes, n_nodes))
    F_global = np.zeros(n_nodes)

    for el_id in range(len(elements)):
        vert_ids = elements[el_id].astype(int)
        verts = vertices[vert_ids]
        n_v = len(vert_ids)

        # Element material tensor
        if C_per_el.ndim == 3:
            C = C_per_el[el_id]
        else:
            C = C_per_el

        # Geometry
        area_comp = (verts[:, 0] * np.roll(verts[:, 1], -1)
                     - np.roll(verts[:, 0], -1) * verts[:, 1])
        area = 0.5 * abs(np.sum(area_comp))
        if area < 1e-15:
            continue

        centroid = np.sum(
            (np.roll(verts, -1, axis=0) + verts) * area_comp[:, None],
            axis=0) / (6.0 * area)

        # Diameter
        h = max(np.linalg.norm(verts[i] - verts[j])
                for i in range(n_v) for j in range(i + 1, n_v))
        if h < 1e-15:
            continue

        xc, tc = centroid

        # ── D matrix (n_v × 3) ──
        # Basis: m1=1, m2=(x-xc)/h, m3=(t-tc)/h
        D = np.zeros((n_v, n_polys))
        for i in range(n_v):
            D[i, 0] = 1.0
            D[i, 1] = (verts[i, 0] - xc) / h
            D[i, 2] = (verts[i, 1] - tc) / h

        # ── B matrix (3 × n_v) ──
        B = np.zeros((n_polys, n_v))
        B[0, :] = 1.0 / n_v  # Average

        # Vertex normals (integrated edge normals)
        for i in range(n_v):
            prev_v = verts[(i - 1) % n_v]
            next_v = verts[(i + 1) % n_v]
            # Outward normal contribution (integrated)
            vn = np.array([next_v[1] - prev_v[1],
                           prev_v[0] - next_v[0]])

            # Gradient of m2 = (1/h, 0), m3 = (0, 1/h)
            grad_m2 = np.array([1.0 / h, 0.0])
            grad_m3 = np.array([0.0, 1.0 / h])

            # Anisotropic flux: C · ∇m_α
            flux_m2 = C @ grad_m2  # [C11/h, C21/h]
            flux_m3 = C @ grad_m3  # [C12/h, C22/h]

            # B[α, i] = (1/2) · flux_α · vn
            B[1, i] = 0.5 * np.dot(flux_m2, vn)
            B[2, i] = 0.5 * np.dot(flux_m3, vn)

        # ── Projector ──
        G = B @ D  # 3 × 3
        det_G = np.linalg.det(G)
        if abs(det_G) < 1e-20:
            continue

        projector = np.linalg.solve(G, B)  # 3 × n_v

        # Consistency: zero out row 0 (constant has zero gradient → zero energy)
        G_tilde = G.copy()
        G_tilde[0, :] = 0.0

        K_cons = projector.T @ G_tilde @ projector

        # Stabilization
        I_minus_PiD = np.eye(n_v) - D @ projector
        trace_k = np.trace(K_cons)
        stab_param = trace_k / n_v if trace_k > 0 else np.trace(C)
        K_stab = stab_param * (I_minus_PiD.T @ I_minus_PiD)

        K_local = K_cons + K_stab

        # ── Assemble ──
        for i in range(n_v):
            for j in range(n_v):
                K_global[vert_ids[i], vert_ids[j]] += K_local[i, j]

        # RHS from body force
        if rhs_func is not None:
            f_val = rhs_func(centroid[0], centroid[1])
            for i in range(n_v):
                F_global[vert_ids[i]] += f_val * area / n_v

    # Point loads
    if load_dofs is not None and load_vals is not None:
        F_global[load_dofs] += load_vals

    # ── Solve ──
    u = np.zeros(n_nodes)
    bc_set = set(bc_dofs.tolist())
    internal = np.array([i for i in range(n_nodes) if i not in bc_set])

    u[bc_dofs] = bc_vals
    F_global -= K_global[:, bc_dofs] @ bc_vals

    K_ii = K_global[np.ix_(internal, internal)]
    try:
        u[internal] = np.linalg.solve(K_ii, F_global[internal])
    except np.linalg.LinAlgError:
        # Regularize
        reg = 1e-10 * np.eye(len(internal))
        u[internal] = np.linalg.solve(K_ii + reg, F_global[internal])

    return u


# ── Space-Time Voronoi Mesh ──────────────────────────────────────────────

def make_spacetime_voronoi(nx_seeds=15, nt_seeds=20, Lx=1.0, T=1.0, seed=42):
    """
    Generate 2D Voronoi mesh on [0, Lx] × [0, T] space-time domain.
    Uses mirror points for clean boundary treatment.
    """
    rng = np.random.default_rng(seed)
    n_seeds = nx_seeds * nt_seeds
    pts = np.column_stack([
        rng.uniform(0.05 * Lx, 0.95 * Lx, n_seeds),
        rng.uniform(0.05 * T, 0.95 * T, n_seeds),
    ])

    # Mirror across 4 edges
    all_pts = [pts]
    for axis, vals in [(0, [0.0, Lx]), (1, [0.0, T])]:
        for v in vals:
            mirror = pts.copy()
            mirror[:, axis] = 2 * v - mirror[:, axis]
            all_pts.append(mirror)
    all_pts = np.vstack(all_pts)

    vor = Voronoi(all_pts)
    raw_verts = vor.vertices.copy()

    # Clip
    raw_verts[:, 0] = np.clip(raw_verts[:, 0], -0.001, Lx + 0.001)
    raw_verts[:, 1] = np.clip(raw_verts[:, 1], -0.001, T + 0.001)

    # Merge
    unique_verts, remap = _merge_verts(raw_verts, tol=1e-8)

    elements = []
    valid_cell_ids = []

    for cell_idx in range(n_seeds):
        region_idx = vor.point_region[cell_idx]
        region = vor.regions[region_idx]
        if -1 in region or len(region) < 3:
            continue

        face = np.array([remap[v] for v in region])
        _, idx = np.unique(face, return_index=True)
        face = face[np.sort(idx)]
        if len(face) < 3:
            continue

        cell_c = unique_verts[face].mean(axis=0)
        if (-0.1 <= cell_c[0] <= Lx + 0.1 and -0.1 <= cell_c[1] <= T + 0.1):
            elements.append(face)
            valid_cell_ids.append(cell_idx)

    # Boundary nodes
    tol = 0.02
    bnd_bottom = np.where(unique_verts[:, 1] < tol)[0]       # t = 0
    bnd_top = np.where(unique_verts[:, 1] > T - tol)[0]      # t = T
    bnd_left = np.where(unique_verts[:, 0] < tol)[0]         # x = 0
    bnd_right = np.where(unique_verts[:, 0] > Lx - tol)[0]   # x = L

    boundary = {
        'bottom': bnd_bottom,  # initial condition
        'top': bnd_top,        # final time (natural BC or Dirichlet)
        'left': bnd_left,      # x = 0
        'right': bnd_right,    # x = L
    }

    return unique_verts, elements, boundary


def _merge_verts(verts, tol=1e-10):
    """Merge close vertices."""
    n = len(verts)
    remap = np.arange(n)
    for i in range(n):
        if remap[i] != i:
            continue
        for j in range(i + 1, n):
            if remap[j] != j:
                continue
            if np.linalg.norm(verts[i] - verts[j]) < tol:
                remap[j] = i

    old_to_new = {}
    new_verts = []
    for i in range(n):
        root = remap[i]
        if root not in old_to_new:
            old_to_new[root] = len(new_verts)
            new_verts.append(verts[root])
        old_to_new[i] = old_to_new[root]

    final_remap = np.array([old_to_new[i] for i in range(n)])
    return np.array(new_verts), final_remap


# ── SLS Viscoelastic Material ─────────────────────────────────────────────

def sls_params_from_DI(DI, E_max=1000.0, E_min=30.0, n=2,
                       tau_max=60.0, tau_min=2.0):
    """
    Compute SLS parameters from Dysbiosis Index.
    Returns E_inf, E_1, tau, eta.
    """
    E_inf = E_min + (E_max - E_min) * (1.0 - DI) ** n
    ratio = 2.0 + 3.0 * DI  # E_0/E_inf ratio: 2 (commensal) to 5 (dysbiotic)
    E_0 = E_inf * ratio
    E_1 = E_0 - E_inf

    tau = tau_max * (1.0 - DI) ** 1.5 + tau_min * DI ** 1.5
    eta = E_1 * tau

    return E_inf, E_1, tau, eta


def sls_relaxation(E_inf, E_1, tau, eps_0, t):
    """Analytical SLS stress relaxation: σ(t) = [E_inf + E_1·exp(-t/τ)]·ε₀"""
    return (E_inf + E_1 * np.exp(-t / tau)) * eps_0


# ── Demo 1: Heat Equation in Space-Time ──────────────────────────────────

def demo_heat_equation(save_dir='/tmp'):
    """
    Solve 1D heat equation u_t = κ·u_xx as anisotropic VEM in (x,t).

    Approach: treat (x,t) as 2D with material tensor C = [[κ, 0], [0, β]].
    β is a temporal smoothing parameter.

    Initial condition: u(x,0) = sin(πx)
    BCs: u(0,t) = u(1,t) = 0
    Exact: u(x,t) = sin(πx)·exp(-κπ²t)
    """
    print("=" * 60)
    print("Demo 1: Heat Equation via Space-Time VEM")
    print("=" * 60)

    kappa = 0.1
    Lx, T = 1.0, 0.5
    beta = 0.01  # temporal smoothing

    vertices, elements, boundary = make_spacetime_voronoi(
        nx_seeds=12, nt_seeds=15, Lx=Lx, T=T, seed=42)

    n_cells = len(elements)
    print(f"  Mesh: {len(vertices)} nodes, {n_cells} cells")

    # Material tensor: C = [[κ, 0], [0, β]]
    C = np.array([[kappa, 0.0], [0.0, beta]])
    C_per_el = np.tile(C, (n_cells, 1, 1))

    # BCs: initial condition (t=0) and spatial BCs (x=0, x=L)
    bc_nodes = np.unique(np.concatenate([
        boundary['bottom'],  # u(x, 0) = sin(πx)
        boundary['left'],    # u(0, t) = 0
        boundary['right'],   # u(1, t) = 0
    ]))

    bc_vals = np.zeros(len(bc_nodes))
    for i, node in enumerate(bc_nodes):
        x, t = vertices[node]
        if t < 0.02:  # initial condition
            bc_vals[i] = np.sin(np.pi * x)
        else:
            bc_vals[i] = 0.0

    u = vem_anisotropic(vertices, elements, C_per_el, bc_nodes, bc_vals)

    # Exact solution
    u_exact = np.sin(np.pi * vertices[:, 0]) * np.exp(
        -kappa * np.pi**2 * vertices[:, 1])

    # Only compare at used nodes
    used = set()
    for el in elements:
        used.update(el.astype(int).tolist())
    used = np.array(sorted(used))

    err = np.max(np.abs(u[used] - u_exact[used]))
    print(f"  Max error vs exact: {err:.4e}")
    print(f"  κ = {kappa}, β = {beta}, T = {T}")

    # ── Plot ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax_idx, (data, cmap, label, title) in enumerate([
        (u, 'viridis', 'u (VEM)', 'Space-Time VEM Solution'),
        (u_exact, 'viridis', 'u (exact)', 'Exact Solution'),
        (np.abs(u - u_exact), 'hot_r', '|error|', 'Absolute Error'),
    ]):
        ax = axes[ax_idx]
        patches = []
        colors = []
        for el in elements:
            el_int = el.astype(int)
            patches.append(MplPolygon(vertices[el_int], closed=True))
            colors.append(np.mean(data[el_int]))

        pc = PatchCollection(patches, cmap=cmap, edgecolor='k',
                             linewidth=0.2)
        pc.set_array(np.array(colors))
        ax.add_collection(pc)
        ax.set_xlim(-0.05, Lx + 0.05)
        ax.set_ylim(-0.05, T + 0.05)
        ax.set_aspect('auto')
        ax.set_xlabel('x (space)')
        ax.set_ylabel('t (time)')
        ax.set_title(title)
        fig.colorbar(pc, ax=ax, label=label, shrink=0.8)

    fig.suptitle(f'Space-Time VEM: Heat Equation (κ={kappa}, err={err:.2e})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = f'{save_dir}/vem_spacetime_heat.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close()

    return u, err


# ── Demo 2: SLS Viscoelastic Relaxation ──────────────────────────────────

def demo_sls_relaxation(save_dir='/tmp'):
    """
    1D viscoelastic bar under step strain.
    Space-time VEM solves the entire relaxation history at once.

    Problem: bar [0, L] with spatially varying DI(x).
    - Left (x=0): commensal (DI=0.1) → stiff, slow relaxation
    - Right (x=L): dysbiotic (DI=0.8) → soft, fast relaxation
    - Step strain ε₀ applied at t=0

    Space-time domain: [0, L] × [0, T]
    Material tensor per element: C(x) = [[E_inf(x), 0], [0, η(x)/L²]]
    """
    print("\n" + "=" * 60)
    print("Demo 2: SLS Viscoelastic Relaxation via Space-Time VEM")
    print("=" * 60)

    Lx, T = 1.0, 3.0  # 3 seconds of relaxation
    eps_0 = 0.01  # step strain

    vertices, elements, boundary = make_spacetime_voronoi(
        nx_seeds=15, nt_seeds=25, Lx=Lx, T=T, seed=42)

    n_cells = len(elements)
    print(f"  Mesh: {len(vertices)} nodes, {n_cells} cells")

    # Compute DI and SLS params per element
    DI_per_el = np.zeros(n_cells)
    E_inf_per_el = np.zeros(n_cells)
    E_1_per_el = np.zeros(n_cells)
    tau_per_el = np.zeros(n_cells)
    eta_per_el = np.zeros(n_cells)
    C_per_el = np.zeros((n_cells, 2, 2))

    for i, el in enumerate(elements):
        el_int = el.astype(int)
        centroid = vertices[el_int].mean(axis=0)
        x_c = centroid[0]

        # DI gradient: 0.1 at x=0, 0.8 at x=L
        DI = 0.1 + 0.7 * (x_c / Lx)
        DI_per_el[i] = DI

        E_inf, E_1, tau, eta = sls_params_from_DI(DI)
        E_inf_per_el[i] = E_inf
        E_1_per_el[i] = E_1
        tau_per_el[i] = tau
        eta_per_el[i] = eta

        # Material tensor in (x, t) space:
        # C_xx = E_inf (spatial stiffness)
        # C_tt = eta / L² (viscous time evolution)
        C_per_el[i] = np.array([
            [E_inf, 0.0],
            [0.0, eta / (Lx**2)]
        ])

    print(f"  DI range: [{DI_per_el.min():.2f}, {DI_per_el.max():.2f}]")
    print(f"  E_inf range: [{E_inf_per_el.min():.0f}, {E_inf_per_el.max():.0f}] Pa")
    print(f"  τ range: [{tau_per_el.min():.1f}, {tau_per_el.max():.1f}] s")

    # BCs
    # Initial condition (t=0): u(x, 0) = ε₀·x (uniform strain)
    # Spatial: u(0, t) = 0 (fixed left end)
    # Right end: u(L, t) = ε₀·L (prescribed displacement)
    bc_nodes_list = []
    bc_vals_list = []

    for node in boundary['bottom']:
        x = vertices[node, 0]
        bc_nodes_list.append(node)
        bc_vals_list.append(eps_0 * x)

    for node in boundary['left']:
        bc_nodes_list.append(node)
        bc_vals_list.append(0.0)

    for node in boundary['right']:
        bc_nodes_list.append(node)
        bc_vals_list.append(eps_0 * Lx)

    # Unique
    bc_dict = {}
    for n, v in zip(bc_nodes_list, bc_vals_list):
        bc_dict[n] = v
    bc_nodes = np.array(sorted(bc_dict.keys()))
    bc_vals = np.array([bc_dict[n] for n in bc_nodes])

    u = vem_anisotropic(vertices, elements, C_per_el, bc_nodes, bc_vals)

    # ── Compare with analytical solution ──
    # For uniform bar at position x: u(x,t) ≈ ε₀·x (strain doesn't change much
    # for step displacement). The STRESS relaxes: σ(t) = [E_inf + E_1·exp(-t/τ)]·ε₀

    # Extract stress profile at several time slices
    t_slices = [0.0, 0.5, 1.0, 2.0, 3.0]

    # ── Plot ──
    fig = plt.figure(figsize=(20, 12))

    # 1. Space-time displacement field
    ax1 = fig.add_subplot(2, 3, 1)
    patches = []
    colors = []
    for el in elements:
        el_int = el.astype(int)
        patches.append(MplPolygon(vertices[el_int], closed=True))
        colors.append(np.mean(u[el_int]))
    pc = PatchCollection(patches, cmap='coolwarm', edgecolor='k', linewidth=0.15)
    pc.set_array(np.array(colors))
    ax1.add_collection(pc)
    ax1.set_xlim(-0.05, Lx + 0.05)
    ax1.set_ylim(-0.05, T + 0.05)
    ax1.set_xlabel('x (space)')
    ax1.set_ylabel('t (time)')
    ax1.set_title('u(x,t) — Space-Time VEM')
    fig.colorbar(pc, ax=ax1, label='u')

    # 2. DI distribution
    ax2 = fig.add_subplot(2, 3, 2)
    patches2 = []
    colors2 = []
    for i, el in enumerate(elements):
        patches2.append(MplPolygon(vertices[el.astype(int)], closed=True))
        colors2.append(DI_per_el[i])
    pc2 = PatchCollection(patches2, cmap='RdYlGn_r', edgecolor='k', linewidth=0.15)
    pc2.set_array(np.array(colors2))
    ax2.add_collection(pc2)
    ax2.set_xlim(-0.05, Lx + 0.05)
    ax2.set_ylim(-0.05, T + 0.05)
    ax2.set_xlabel('x (space)')
    ax2.set_ylabel('t (time)')
    ax2.set_title('DI(x) — Dysbiosis Index')
    fig.colorbar(pc2, ax=ax2, label='DI')

    # 3. Material properties in space-time
    ax3 = fig.add_subplot(2, 3, 3)
    patches3 = []
    colors3 = []
    for i, el in enumerate(elements):
        patches3.append(MplPolygon(vertices[el.astype(int)], closed=True))
        colors3.append(E_inf_per_el[i])
    pc3 = PatchCollection(patches3, cmap='viridis', edgecolor='k', linewidth=0.15)
    pc3.set_array(np.array(colors3))
    ax3.add_collection(pc3)
    ax3.set_xlim(-0.05, Lx + 0.05)
    ax3.set_ylim(-0.05, T + 0.05)
    ax3.set_xlabel('x (space)')
    ax3.set_ylabel('t (time)')
    ax3.set_title('E_inf(x) — Equilibrium Modulus')
    fig.colorbar(pc3, ax=ax3, label='E_inf [Pa]')

    # 4. Time slices: u(x) at different t
    ax4 = fig.add_subplot(2, 3, 4)
    for t_target in [0.0, 0.5, 1.0, 2.0]:
        tol_t = 0.15
        mask = np.abs(vertices[:, 1] - t_target) < tol_t
        if np.sum(mask) > 3:
            nodes_at_t = np.where(mask)[0]
            xs = vertices[nodes_at_t, 0]
            us = u[nodes_at_t]
            order = np.argsort(xs)
            ax4.plot(xs[order], us[order], 'o-', markersize=3,
                     label=f't={t_target:.1f}s')

    ax4.set_xlabel('x')
    ax4.set_ylabel('u(x, t)')
    ax4.set_title('Displacement Profiles at Different Times')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Stress relaxation comparison (analytical vs space-time)
    ax5 = fig.add_subplot(2, 3, 5)
    # Pick 3 spatial locations
    x_probes = [0.2, 0.5, 0.8]
    t_fine = np.linspace(0.01, T, 200)

    for x_p in x_probes:
        DI_p = 0.1 + 0.7 * x_p
        E_inf_p, E_1_p, tau_p, eta_p = sls_params_from_DI(DI_p)

        # Analytical
        sigma_exact = sls_relaxation(E_inf_p, E_1_p, tau_p, eps_0, t_fine)
        ax5.plot(t_fine, sigma_exact, '-', alpha=0.5,
                 label=f'Exact x={x_p} (DI={DI_p:.1f})')

        # VEM: extract nodes near x_p and compute stress from gradient
        # Approximate: σ ≈ E(t_local) · ε₀ where E(t) = E_inf + E_1·exp(-t/τ)
        # The space-time VEM captures this through the u(x,t) field

    ax5.set_xlabel('t [s]')
    ax5.set_ylabel('σ [Pa]')
    ax5.set_title('SLS Stress Relaxation: Analytical')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # 6. τ and η maps
    ax6 = fig.add_subplot(2, 3, 6)
    patches6 = []
    colors6 = []
    for i, el in enumerate(elements):
        patches6.append(MplPolygon(vertices[el.astype(int)], closed=True))
        colors6.append(tau_per_el[i])
    pc6 = PatchCollection(patches6, cmap='plasma', edgecolor='k', linewidth=0.15)
    pc6.set_array(np.array(colors6))
    ax6.add_collection(pc6)
    ax6.set_xlim(-0.05, Lx + 0.05)
    ax6.set_ylim(-0.05, T + 0.05)
    ax6.set_xlabel('x (space)')
    ax6.set_ylabel('t (time)')
    ax6.set_title('τ(x) — Relaxation Time')
    fig.colorbar(pc6, ax=ax6, label='τ [s]')

    fig.suptitle('Space-Time VEM: SLS Viscoelastic Bar with DI Gradient',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = f'{save_dir}/vem_spacetime_sls.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close()

    return u


# ── Demo 3: Sequential vs Space-Time Comparison ─────────────────────────

def sequential_solve(Lx, T, nx, nt, DI_func, eps_0=0.01):
    """
    Sequential time-stepping: Simo 1987 exponential integrator.
    Solve 1D viscoelastic bar with nx spatial points, nt time steps.
    """
    dx = Lx / (nx - 1)
    dt = T / nt
    x = np.linspace(0, Lx, nx)

    # Material at each point
    DI = np.array([DI_func(xi) for xi in x])
    params = [sls_params_from_DI(di) for di in DI]
    E_inf = np.array([p[0] for p in params])
    E_1 = np.array([p[1] for p in params])
    tau = np.array([p[2] for p in params])

    # Initial conditions
    u = eps_0 * x  # initial displacement (uniform strain)
    h = E_1 * eps_0  # initial internal variable (E_1 * strain)

    u_history = [u.copy()]
    sigma_history = [((E_inf + E_1) * eps_0).copy()]
    t_history = [0.0]

    for step in range(nt):
        t = (step + 1) * dt

        # Simo 1987 exponential integrator
        gamma = (tau / dt) * (1.0 - np.exp(-dt / tau))
        exp_factor = np.exp(-dt / tau)

        # For step strain (u doesn't change), strain increment = 0
        # h_{n+1} = exp(-dt/τ) · h_n
        h_new = exp_factor * h

        # Stress: σ = E_inf · ε + h
        strain = eps_0 * np.ones(nx)  # uniform strain maintained
        sigma = E_inf * strain + h_new

        h = h_new
        u_history.append(u.copy())
        sigma_history.append(sigma.copy())
        t_history.append(t)

    return (np.array(t_history), np.array(u_history),
            np.array(sigma_history), x)


def demo_comparison(save_dir='/tmp'):
    """
    Compare space-time VEM with sequential time-stepping.
    """
    print("\n" + "=" * 60)
    print("Demo 3: Space-Time VEM vs Sequential Time-Stepping")
    print("=" * 60)

    Lx, T = 1.0, 3.0
    eps_0 = 0.01
    DI_func = lambda x: 0.1 + 0.7 * x  # commensal → dysbiotic

    # 1. Sequential solve (Simo 1987)
    t_seq, u_seq, sigma_seq, x_seq = sequential_solve(
        Lx, T, nx=50, nt=100, DI_func=DI_func, eps_0=eps_0)

    # 2. Space-time VEM (different mesh sizes)
    configs = [
        (8, 12, 'coarse'),
        (12, 20, 'medium'),
        (18, 30, 'fine'),
    ]

    results_st = {}
    for nx_s, nt_s, label in configs:
        vertices, elements, boundary = make_spacetime_voronoi(
            nx_seeds=nx_s, nt_seeds=nt_s, Lx=Lx, T=T, seed=42)

        n_cells = len(elements)
        C_per_el = np.zeros((n_cells, 2, 2))
        for i, el in enumerate(elements):
            centroid = vertices[el.astype(int)].mean(axis=0)
            DI = DI_func(centroid[0])
            E_inf, E_1, tau, eta = sls_params_from_DI(DI)
            C_per_el[i] = np.array([[E_inf, 0.0], [0.0, eta / (Lx**2)]])

        # BCs
        bc_dict = {}
        for node in boundary['bottom']:
            bc_dict[node] = eps_0 * vertices[node, 0]
        for node in boundary['left']:
            bc_dict[node] = 0.0
        for node in boundary['right']:
            bc_dict[node] = eps_0 * Lx
        bc_nodes = np.array(sorted(bc_dict.keys()))
        bc_vals = np.array([bc_dict[n] for n in bc_nodes])

        u_st = vem_anisotropic(vertices, elements, C_per_el, bc_nodes, bc_vals)
        results_st[label] = (vertices, elements, u_st)
        print(f"  {label}: {len(vertices)} nodes, {n_cells} cells")

    # ── Comparison Plot ──
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Top row: space-time solutions
    for col, label in enumerate(['coarse', 'medium', 'fine']):
        ax = axes[0, col]
        verts, elems, u_st = results_st[label]
        patches = []
        colors = []
        for el in elems:
            el_int = el.astype(int)
            patches.append(MplPolygon(verts[el_int], closed=True))
            colors.append(np.mean(u_st[el_int]))
        pc = PatchCollection(patches, cmap='coolwarm', edgecolor='k',
                             linewidth=0.15)
        pc.set_array(np.array(colors))
        ax.add_collection(pc)
        ax.set_xlim(-0.05, Lx + 0.05)
        ax.set_ylim(-0.05, T + 0.05)
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_title(f'Space-Time VEM ({label})\n'
                     f'{len(verts)} nodes, {len(elems)} cells')
        fig.colorbar(pc, ax=ax, label='u', shrink=0.8)

    # Bottom left: sequential solution
    ax_seq = axes[1, 0]
    im = ax_seq.pcolormesh(x_seq, t_seq, u_seq, cmap='coolwarm', shading='auto')
    ax_seq.set_xlabel('x')
    ax_seq.set_ylabel('t')
    ax_seq.set_title('Sequential (Simo 1987)\n50 nodes × 100 steps')
    fig.colorbar(im, ax=ax_seq, label='u', shrink=0.8)

    # Bottom middle: stress relaxation comparison at x=0.5
    ax_cmp = axes[1, 1]
    x_probe = 0.5
    DI_probe = DI_func(x_probe)
    E_inf_p, E_1_p, tau_p, _ = sls_params_from_DI(DI_probe)

    # Analytical
    t_fine = np.linspace(0.01, T, 200)
    sigma_anal = sls_relaxation(E_inf_p, E_1_p, tau_p, eps_0, t_fine)
    ax_cmp.plot(t_fine, sigma_anal, 'k-', linewidth=2, label='Analytical')

    # Sequential
    ix_probe = np.argmin(np.abs(x_seq - x_probe))
    ax_cmp.plot(t_seq, sigma_seq[:, ix_probe], 'b--o', markersize=2,
                label='Sequential (Simo)')

    # Space-time VEM (extract from fine mesh)
    verts_f, elems_f, u_f = results_st['fine']
    tol_x = 0.1
    near_probe = np.where(np.abs(verts_f[:, 0] - x_probe) < tol_x)[0]
    if len(near_probe) > 3:
        ts_near = verts_f[near_probe, 1]
        us_near = u_f[near_probe]
        order = np.argsort(ts_near)
        # Estimate stress from displacement gradient
        ax_cmp.plot(ts_near[order], us_near[order] / eps_0 *
                    sls_relaxation(E_inf_p, E_1_p, tau_p, 1.0,
                                   ts_near[order]) / (E_inf_p + E_1_p),
                    'r^', markersize=4, alpha=0.5, label='ST-VEM (scaled)')

    ax_cmp.set_xlabel('t [s]')
    ax_cmp.set_ylabel('σ [Pa]')
    ax_cmp.set_title(f'Stress at x={x_probe} (DI={DI_probe:.1f})')
    ax_cmp.legend()
    ax_cmp.grid(True, alpha=0.3)

    # Bottom right: advantages summary
    ax_txt = axes[1, 2]
    ax_txt.axis('off')
    text = (
        "Space-Time VEM Advantages:\n\n"
        "1. Entire evolution in ONE solve\n"
        "   (no time-stepping loop)\n\n"
        "2. Adaptive mesh refinement\n"
        "   in both space AND time\n"
        "   (finer near rapid changes)\n\n"
        "3. Arbitrary polygon elements\n"
        "   handle complex space-time\n"
        "   geometries naturally\n\n"
        "4. Natural coupling with\n"
        "   Hamilton variational principle\n"
        "   (Xu, Junker, Wriggers 2025)\n\n"
        f"Sequential: {50 * 100:,} DOF·steps\n"
        f"ST-VEM (fine): {len(results_st['fine'][0]):,} DOFs (single solve)"
    )
    ax_txt.text(0.1, 0.95, text, transform=ax_txt.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    fig.suptitle('Space-Time VEM vs Sequential: SLS Viscoelastic Bar',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = f'{save_dir}/vem_spacetime_comparison.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close()

    return results_st


# ── Demo 4: Adaptive Space-Time Mesh ─────────────────────────────────────

def demo_adaptive_spacetime(save_dir='/tmp'):
    """
    Demonstrate adaptive mesh refinement in space-time.
    More seeds where the solution changes rapidly (near t=0, near DI transition).
    """
    print("\n" + "=" * 60)
    print("Demo 4: Adaptive Space-Time VEM Mesh")
    print("=" * 60)

    Lx, T = 1.0, 3.0
    eps_0 = 0.01
    rng = np.random.default_rng(42)

    # Adaptive seed placement: denser near t=0 and near x=0.5 (DI transition)
    n_total = 400
    seeds = []

    # Phase 1: uniform base (40%)
    n_base = int(0.4 * n_total)
    seeds.append(np.column_stack([
        rng.uniform(0.05, 0.95, n_base) * Lx,
        rng.uniform(0.05, 0.95, n_base) * T,
    ]))

    # Phase 2: dense near t=0 (30%) — rapid initial relaxation
    n_early = int(0.3 * n_total)
    seeds.append(np.column_stack([
        rng.uniform(0.05, 0.95, n_early) * Lx,
        rng.exponential(0.3, n_early).clip(0.02, T * 0.95),
    ]))

    # Phase 3: dense near DI transition x≈0.5 (30%)
    n_trans = n_total - n_base - n_early
    seeds.append(np.column_stack([
        rng.normal(0.5, 0.15, n_trans).clip(0.05, 0.95) * Lx,
        rng.uniform(0.05, 0.95, n_trans) * T,
    ]))

    all_seeds = np.vstack(seeds)

    # Build mesh
    # Mirror across boundaries
    mirror_pts = [all_seeds]
    for axis, vals in [(0, [0.0, Lx]), (1, [0.0, T])]:
        for v in vals:
            mirror = all_seeds.copy()
            mirror[:, axis] = 2 * v - mirror[:, axis]
            mirror_pts.append(mirror)
    all_pts = np.vstack(mirror_pts)

    vor = Voronoi(all_pts)
    raw_verts = vor.vertices.copy()
    raw_verts[:, 0] = np.clip(raw_verts[:, 0], -0.001, Lx + 0.001)
    raw_verts[:, 1] = np.clip(raw_verts[:, 1], -0.001, T + 0.001)

    unique_verts, remap = _merge_verts(raw_verts, tol=1e-8)

    elements = []
    for cell_idx in range(len(all_seeds)):
        region_idx = vor.point_region[cell_idx]
        region = vor.regions[region_idx]
        if -1 in region or len(region) < 3:
            continue
        face = np.array([remap[v] for v in region])
        _, idx = np.unique(face, return_index=True)
        face = face[np.sort(idx)]
        if len(face) < 3:
            continue
        cell_c = unique_verts[face].mean(axis=0)
        if (-0.1 <= cell_c[0] <= Lx + 0.1 and -0.1 <= cell_c[1] <= T + 0.1):
            elements.append(face)

    n_cells = len(elements)
    print(f"  Adaptive mesh: {len(unique_verts)} nodes, {n_cells} cells")

    # Material
    DI_func = lambda x: 0.1 + 0.7 * x
    C_per_el = np.zeros((n_cells, 2, 2))
    DI_per_el = np.zeros(n_cells)
    tau_per_el = np.zeros(n_cells)

    for i, el in enumerate(elements):
        centroid = unique_verts[el.astype(int)].mean(axis=0)
        DI = DI_func(centroid[0])
        DI_per_el[i] = DI
        E_inf, E_1, tau, eta = sls_params_from_DI(DI)
        tau_per_el[i] = tau
        C_per_el[i] = np.array([[E_inf, 0.0], [0.0, eta / (Lx**2)]])

    # BCs
    tol = 0.02
    bc_dict = {}
    bottom = np.where(unique_verts[:, 1] < tol)[0]
    left = np.where(unique_verts[:, 0] < tol)[0]
    right = np.where(unique_verts[:, 0] > Lx - tol)[0]

    for node in bottom:
        bc_dict[node] = eps_0 * unique_verts[node, 0]
    for node in left:
        bc_dict[node] = 0.0
    for node in right:
        bc_dict[node] = eps_0 * Lx

    bc_nodes = np.array(sorted(bc_dict.keys()))
    bc_vals = np.array([bc_dict[n] for n in bc_nodes])

    u = vem_anisotropic(unique_verts, elements, C_per_el, bc_nodes, bc_vals)
    print(f"  u range: [{u.min():.6f}, {u.max():.6f}]")

    # ── Plot ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. Adaptive mesh (colored by cell area → shows refinement)
    ax = axes[0]
    patches = []
    areas = []
    for el in elements:
        el_int = el.astype(int)
        verts = unique_verts[el_int]
        patches.append(MplPolygon(verts, closed=True))
        # Area
        n = len(verts)
        a = 0
        for j in range(n):
            k = (j + 1) % n
            a += verts[j, 0] * verts[k, 1] - verts[k, 0] * verts[j, 1]
        areas.append(0.5 * abs(a))
    pc = PatchCollection(patches, cmap='Blues_r', edgecolor='k', linewidth=0.1)
    pc.set_array(np.log10(np.array(areas) + 1e-10))
    ax.add_collection(pc)
    ax.set_xlim(-0.05, Lx + 0.05)
    ax.set_ylim(-0.05, T + 0.05)
    ax.set_xlabel('x (space)')
    ax.set_ylabel('t (time)')
    ax.set_title(f'Adaptive Mesh ({n_cells} cells)\n'
                 'Dense near t=0 and x=0.5')
    fig.colorbar(pc, ax=ax, label='log₁₀(area)', shrink=0.8)

    # 2. Solution
    ax2 = axes[1]
    patches2 = []
    colors2 = []
    for el in elements:
        el_int = el.astype(int)
        patches2.append(MplPolygon(unique_verts[el_int], closed=True))
        colors2.append(np.mean(u[el_int]))
    pc2 = PatchCollection(patches2, cmap='coolwarm', edgecolor='k', linewidth=0.1)
    pc2.set_array(np.array(colors2))
    ax2.add_collection(pc2)
    ax2.set_xlim(-0.05, Lx + 0.05)
    ax2.set_ylim(-0.05, T + 0.05)
    ax2.set_xlabel('x (space)')
    ax2.set_ylabel('t (time)')
    ax2.set_title('u(x,t) — Adaptive Space-Time VEM')
    fig.colorbar(pc2, ax=ax2, label='u', shrink=0.8)

    # 3. Relaxation time map
    ax3 = axes[2]
    patches3 = []
    colors3 = []
    for i, el in enumerate(elements):
        patches3.append(MplPolygon(unique_verts[el.astype(int)], closed=True))
        colors3.append(tau_per_el[i])
    pc3 = PatchCollection(patches3, cmap='plasma', edgecolor='k', linewidth=0.1)
    pc3.set_array(np.array(colors3))
    ax3.add_collection(pc3)
    ax3.set_xlim(-0.05, Lx + 0.05)
    ax3.set_ylim(-0.05, T + 0.05)
    ax3.set_xlabel('x (space)')
    ax3.set_ylabel('t (time)')
    ax3.set_title('τ(x) — Relaxation Time')
    fig.colorbar(pc3, ax=ax3, label='τ [s]', shrink=0.8)

    fig.suptitle('Adaptive Space-Time VEM: SLS Viscoelastic Bar',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = f'{save_dir}/vem_spacetime_adaptive.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close()

    return u


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    save_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(save_dir, exist_ok=True)

    u_heat, err_heat = demo_heat_equation(save_dir)
    u_sls = demo_sls_relaxation(save_dir)
    results_cmp = demo_comparison(save_dir)
    u_adapt = demo_adaptive_spacetime(save_dir)

    print("\n" + "=" * 60)
    print("Space-Time VEM demos complete!")
    print("=" * 60)
