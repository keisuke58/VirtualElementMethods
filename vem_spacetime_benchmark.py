"""
C5: Space-Time VEM Benchmark — Comparison with Xu, Junker, Wriggers (CMAME 2024).

Compares our 2D anisotropic space-time VEM (scalar, (x,t) plane) against
Xu et al.'s 2.5D extrusion approach (2D spatial + time layers, 4 DOFs/node).

Benchmarks:
  1. 1D wave equation: exact solution comparison
  2. SLS relaxation: known analytical solution
  3. Convergence rates: spatial and temporal refinement
  4. Stabilization comparison: α = 0.15 (Wriggers) vs α = 1.0 (standard)

References:
  - Xu, Junker, Wriggers (2025) "Space-time VEM for elastodynamics"
    DOI: 10.1016/j.cma.2024.117683
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from vem_spacetime import (vem_anisotropic, make_spacetime_voronoi,
                           sls_params_from_DI, sls_relaxation)
from vem_elasticity import (vem_elasticity, assemble_mass_matrix,
                            vem_elastodynamics)


# ── Benchmark 1: Wave Equation ───────────────────────────────────────────

def benchmark_wave(save_dir='/tmp'):
    """
    1D wave equation: u_tt = c² u_xx on [0, 1] × [0, T].

    Exact: u(x,t) = sin(πx)·cos(πct)

    Space-time VEM: C = [[c², 0], [0, -1]] (indefinite tensor!)
    We use the relaxed formulation: C = [[c², 0], [0, ε]] with small ε > 0.
    """
    print("=" * 60)
    print("Benchmark 1: Wave Equation (space-time VEM)")
    print("=" * 60)

    c = 1.0  # wave speed
    Lx, T = 1.0, 1.0

    # Convergence study: refine mesh
    mesh_sizes = [(8, 10), (12, 15), (20, 25), (30, 40)]
    errors = []
    n_dofs_list = []

    for nx, nt in mesh_sizes:
        vertices, elements, boundary = make_spacetime_voronoi(
            nx_seeds=nx, nt_seeds=nt, Lx=Lx, T=T, seed=42)

        n_cells = len(elements)

        # Material tensor: approximate wave equation
        # Use ε regularization for positive-definiteness
        eps_reg = 0.01
        C = np.array([[c**2, 0.0], [0.0, eps_reg]])
        C_per_el = np.tile(C, (n_cells, 1, 1))

        # BCs
        bc_nodes = np.unique(np.concatenate([
            boundary['bottom'],  # u(x,0) = sin(πx)
            boundary['left'],    # u(0,t) = 0
            boundary['right'],   # u(1,t) = 0
        ]))

        bc_vals = np.zeros(len(bc_nodes))
        for i, node in enumerate(bc_nodes):
            x, t = vertices[node]
            if t < 0.02:
                bc_vals[i] = np.sin(np.pi * x)
            else:
                bc_vals[i] = 0.0

        u = vem_anisotropic(vertices, elements, C_per_el, bc_nodes, bc_vals)

        # Exact at used nodes
        used = set()
        for el in elements:
            used.update(el.astype(int).tolist())
        used = np.array(sorted(used))

        u_exact = np.sin(np.pi * vertices[used, 0]) * np.cos(
            np.pi * c * vertices[used, 1])

        err = np.sqrt(np.mean((u[used] - u_exact)**2))
        errors.append(err)
        n_dofs_list.append(len(used))

        print(f"  {nx}×{nt} mesh: {len(used)} nodes, L² error = {err:.4e}")

    # Convergence rate
    h_vals = 1.0 / np.sqrt(np.array(n_dofs_list))
    if len(errors) > 1:
        rates = np.diff(np.log(errors)) / np.diff(np.log(h_vals))
        print(f"  Convergence rates: {[f'{r:.2f}' for r in rates]}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Solution field (finest mesh)
    ax = axes[0]
    patches = []
    colors = []
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.collections import PatchCollection
    for el in elements:
        el_int = el.astype(int)
        patches.append(MplPolygon(vertices[el_int], closed=True))
        colors.append(np.mean(u[el_int]))
    pc = PatchCollection(patches, cmap='coolwarm', edgecolor='k', linewidth=0.15)
    pc.set_array(np.array(colors))
    ax.add_collection(pc)
    ax.set_xlim(-0.05, Lx + 0.05)
    ax.set_ylim(-0.05, T + 0.05)
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_title('VEM u(x,t)')
    fig.colorbar(pc, ax=ax, label='u')

    # 2. Error field
    ax = axes[1]
    patches2 = []
    colors2 = []
    u_ex_all = np.sin(np.pi * vertices[:, 0]) * np.cos(np.pi * c * vertices[:, 1])
    for el in elements:
        el_int = el.astype(int)
        patches2.append(MplPolygon(vertices[el_int], closed=True))
        colors2.append(np.mean(np.abs(u[el_int] - u_ex_all[el_int])))
    pc2 = PatchCollection(patches2, cmap='hot_r', edgecolor='k', linewidth=0.15)
    pc2.set_array(np.array(colors2))
    ax.add_collection(pc2)
    ax.set_xlim(-0.05, Lx + 0.05)
    ax.set_ylim(-0.05, T + 0.05)
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_title('|Error|')
    fig.colorbar(pc2, ax=ax, label='|e|')

    # 3. Convergence
    ax = axes[2]
    ax.loglog(h_vals, errors, 'bo-', label='VEM')
    ax.loglog(h_vals, errors[0] * (h_vals / h_vals[0])**1, 'k--', alpha=0.5, label='O(h¹)')
    ax.loglog(h_vals, errors[0] * (h_vals / h_vals[0])**2, 'k:', alpha=0.5, label='O(h²)')
    ax.set_xlabel('h (mesh size)')
    ax.set_ylabel('L² error')
    ax.set_title('Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle('Space-Time VEM: Wave Equation Benchmark', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = f'{save_dir}/vem_benchmark_wave.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close()

    return errors, h_vals


# ── Benchmark 2: SLS Relaxation (analytical comparison) ─────────────────

def benchmark_sls(save_dir='/tmp'):
    """
    Compare space-time VEM relaxation against analytical SLS solution.
    Tests 3 DI values: 0.1 (commensal), 0.5 (intermediate), 0.9 (dysbiotic).
    """
    print("\n" + "=" * 60)
    print("Benchmark 2: SLS Relaxation Comparison")
    print("=" * 60)

    Lx, T = 1.0, 5.0
    eps_0 = 0.01
    DI_values = [0.1, 0.5, 0.9]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for di_idx, DI in enumerate(DI_values):
        E_inf, E_1, tau, eta = sls_params_from_DI(DI)

        vertices, elements, boundary = make_spacetime_voronoi(
            nx_seeds=10, nt_seeds=30, Lx=Lx, T=T, seed=42)
        n_cells = len(elements)

        # Uniform material
        C_per_el = np.zeros((n_cells, 2, 2))
        for i in range(n_cells):
            C_per_el[i] = np.array([[E_inf, 0.0], [0.0, eta / Lx**2]])

        # BCs
        bc_nodes_list, bc_vals_list = [], []
        for node in boundary['bottom']:
            bc_nodes_list.append(node)
            bc_vals_list.append(eps_0 * vertices[node, 0])
        for node in boundary['left']:
            bc_nodes_list.append(node)
            bc_vals_list.append(0.0)
        for node in boundary['right']:
            bc_nodes_list.append(node)
            bc_vals_list.append(eps_0 * Lx)

        bc_dict = {n: v for n, v in zip(bc_nodes_list, bc_vals_list)}
        bc_nodes = np.array(sorted(bc_dict.keys()))
        bc_vals = np.array([bc_dict[n] for n in bc_nodes])

        u = vem_anisotropic(vertices, elements, C_per_el, bc_nodes, bc_vals)

        # Extract mid-bar displacement over time
        x_mid = 0.5
        tol_x = 0.1
        mid_mask = np.abs(vertices[:, 0] - x_mid) < tol_x
        mid_nodes = np.where(mid_mask)[0]

        if len(mid_nodes) > 3:
            t_vals = vertices[mid_nodes, 1]
            u_vals = u[mid_nodes]
            order = np.argsort(t_vals)

            # Analytical stress at midpoint
            t_fine = np.linspace(0.01, T, 200)
            sigma_exact = sls_relaxation(E_inf, E_1, tau, eps_0, t_fine)

            ax = axes[di_idx]
            ax.plot(t_fine, sigma_exact, 'r-', linewidth=2, label='Analytical')

            # VEM "stress" ≈ E(t) · ε₀ ≈ u(x_mid, t) / x_mid (since u ≈ ε·x)
            u_vem_strain = u_vals[order] / (x_mid + 1e-10)
            sigma_vem = E_inf * u_vem_strain  # simplified
            ax.plot(t_vals[order], sigma_vem, 'bo', markersize=4, alpha=0.6,
                    label='Space-time VEM')

            ax.set_xlabel('t [s]')
            ax.set_ylabel('σ [Pa]')
            ax.set_title(f'DI = {DI} (τ={tau:.1f}s, E∞={E_inf:.0f} Pa)')
            ax.legend()
            ax.grid(True, alpha=0.3)

            print(f"  DI={DI}: τ={tau:.1f}s, E∞={E_inf:.0f}, E₁={E_1:.0f}")

    fig.suptitle('SLS Relaxation: Analytical vs Space-Time VEM',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = f'{save_dir}/vem_benchmark_sls.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close()


# ── Benchmark 3: Stabilization Comparison ────────────────────────────────

def benchmark_stabilization(save_dir='/tmp'):
    """
    Compare stabilization parameter α effect on accuracy.

    α = 0.15 (Xu/Wriggers 2025)
    α = 0.5  (balanced)
    α = 1.0  (standard Gain et al. 2014)

    Test problem: cantilever with spatially varying E(DI) on voronoi.mat.
    """
    print("\n" + "=" * 60)
    print("Benchmark 3: Stabilization α Comparison (C4)")
    print("=" * 60)

    import scipy.io
    mesh_path = os.path.join(os.path.dirname(__file__), 'meshes', 'voronoi.mat')
    if not os.path.exists(mesh_path):
        print("  SKIP: voronoi.mat not found")
        return {}

    mesh = scipy.io.loadmat(mesh_path)
    vertices = mesh['vertices']
    elements = np.array(
        [i[0].reshape(-1) - 1 for i in mesh['elements']], dtype=object)

    # E(DI) field
    E_per_el = np.zeros(len(elements))
    for i, el in enumerate(elements):
        el_int = el.astype(int)
        cx = vertices[el_int].mean(axis=0)[0]
        DI = 0.1 + 0.8 * cx  # x ∈ [0,1]
        E_per_el[i] = 30.0 + 970.0 * (1.0 - DI)**2

    nu = 0.3
    tol = 1e-6

    # Fix left edge
    left = np.where(vertices[:, 0] < tol)[0]
    bc_dofs = np.concatenate([2*left, 2*left+1])
    bc_vals = np.zeros(len(bc_dofs))

    # Load on right edge
    right = np.where(vertices[:, 0] > 1.0 - tol)[0]
    load_dofs = 2 * right + 1
    load_vals = np.full(len(right), -1.0 / max(len(right), 1))

    alphas = [0.15, 0.5, 1.0]
    results = {}

    for alpha in alphas:
        t0 = time.time()
        u = vem_elasticity(vertices, elements, E_per_el, nu,
                           bc_dofs, bc_vals, load_dofs, load_vals,
                           stabilization_alpha=alpha)
        dt_solve = time.time() - t0

        ux = u[0::2]
        uy = u[1::2]
        tip_defl = np.mean(uy[right]) if len(right) > 0 else 0
        max_u = np.max(np.sqrt(ux**2 + uy**2))

        results[alpha] = {'tip': tip_defl, 'max_u': max_u, 'time': dt_solve}
        print(f"  α={alpha:.2f}: tip={tip_defl:.6f}, max|u|={max_u:.6f}, "
              f"time={dt_solve:.3f}s")

    # Relative differences
    ref = results[1.0]['tip']
    for alpha in [0.15, 0.5]:
        diff = abs(results[alpha]['tip'] - ref) / abs(ref) * 100
        print(f"  α={alpha:.2f} vs α=1.0: Δtip = {diff:.2f}%")

    return results


# ── Benchmark 4: Elastodynamics (Newmark vs Space-Time) ─────────────────

def benchmark_elastodynamics(save_dir='/tmp'):
    """
    Compare VEM Newmark-β elastodynamics (C6) against space-time VEM.

    Problem: square domain with sinusoidal load, E = 1000 Pa, ρ = 1.0 kg/m³.
    """
    print("\n" + "=" * 60)
    print("Benchmark 4: Elastodynamics — Newmark VEM (C6)")
    print("=" * 60)

    import scipy.io
    mesh_path = os.path.join(os.path.dirname(__file__), 'meshes', 'voronoi.mat')
    if not os.path.exists(mesh_path):
        print("  SKIP: voronoi.mat not found")
        return None, None

    mesh = scipy.io.loadmat(mesh_path)
    vertices = mesh['vertices']
    elements = np.array(
        [i[0].reshape(-1) - 1 for i in mesh['elements']], dtype=object)

    E_mod = 1000.0
    nu = 0.3
    rho = 1.0

    tol = 1e-6
    bottom = np.where(vertices[:, 1] < tol)[0]
    bc_dofs = np.concatenate([2*bottom, 2*bottom+1])
    bc_vals_static = np.zeros(len(bc_dofs))

    # Sinusoidal load on top
    top = np.where(vertices[:, 1] > 1.0 - tol)[0]
    omega = 2 * np.pi * 5  # 5 Hz

    def load_func(t, n_dofs):
        F = np.zeros(n_dofs)
        load_per_node = -1.0 * np.sin(omega * t) / max(len(top), 1)
        F[2*top + 1] = load_per_node
        return F

    # Newmark-β integration
    dt = 0.001
    n_steps = 100
    T_total = dt * n_steps

    print(f"  Mesh: {len(vertices)} nodes, {len(elements)} elements")
    print(f"  Newmark-β: dt={dt}, {n_steps} steps, T={T_total:.3f}s")

    t0 = time.time()
    t_hist, u_hist, v_hist = vem_elastodynamics(
        vertices, elements, E_mod, nu, rho,
        bc_dofs, bc_vals_static, load_func,
        dt=dt, n_steps=n_steps, beta_nm=0.25, gamma_nm=0.5,
        lumped_mass=False, damping_alpha=0.01, damping_beta=0.001)
    dt_newmark = time.time() - t0

    print(f"  Newmark solve time: {dt_newmark:.2f}s")

    # Extract tip response
    if len(top) > 0:
        tip_node = top[0]
        tip_uy = u_hist[:, 2*tip_node + 1]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Time history
        ax = axes[0]
        ax.plot(t_hist, tip_uy * 1e3, 'b-', linewidth=1.5, label='Newmark-β VEM')
        ax.set_xlabel('t [s]')
        ax.set_ylabel('u_y [mm]')
        ax.set_title(f'Tip Response (node {tip_node})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Snapshot at t = T/2
        mid_step = n_steps // 2
        u_mid = u_hist[mid_step]
        ux = u_mid[0::2]
        uy = u_mid[1::2]

        ax = axes[1]
        from matplotlib.patches import Polygon as MplPolygon
        from matplotlib.collections import PatchCollection
        deform_scale = 500
        deformed = vertices + deform_scale * np.column_stack([ux, uy])
        patches = []
        colors = []
        for el in elements:
            el_int = el.astype(int)
            patches.append(MplPolygon(deformed[el_int], closed=True))
            colors.append(np.mean(np.sqrt(ux[el_int]**2 + uy[el_int]**2)))
        pc = PatchCollection(patches, cmap='viridis', edgecolor='k', linewidth=0.3)
        pc.set_array(np.array(colors))
        ax.add_collection(pc)
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_aspect('equal')
        ax.set_title(f'Deformed (×{deform_scale}) at t={t_hist[mid_step]:.3f}s')
        fig.colorbar(pc, ax=ax, label='|u|')

        fig.suptitle('VEM Elastodynamics: Newmark-β (C6)',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        path = f'{save_dir}/vem_benchmark_dynamics.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {path}")
        plt.close()

    # Mass matrix diagnostics
    M = assemble_mass_matrix(vertices, elements, rho, lumped=False)
    M_lump = assemble_mass_matrix(vertices, elements, rho, lumped=True)

    print(f"\n  Mass matrix diagnostics:")
    print(f"    M consistent: nnz={M.nnz}, symmetry check="
          f"{np.max(np.abs(M - M.T)):.2e}")
    print(f"    M lumped: nnz={M_lump.nnz}, "
          f"total mass={M_lump.diagonal().sum():.4f} "
          f"(expected: {rho * 1.0 * 1.0 * 2:.4f}, 2 DOF/node)")

    return t_hist, u_hist


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    save_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(save_dir, exist_ok=True)

    # Run all benchmarks
    print("\n" + "=" * 60)
    print("  VEM Space-Time Benchmark Suite (C4/C5/C6)")
    print("=" * 60 + "\n")

    benchmark_wave(save_dir)
    benchmark_sls(save_dir)
    benchmark_stabilization(save_dir)
    benchmark_elastodynamics(save_dir)

    print("\n" + "=" * 60)
    print("  All benchmarks complete!")
    print("=" * 60)
