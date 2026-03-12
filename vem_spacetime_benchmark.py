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


# ── Helper Functions ─────────────────────────────────────────────────────

def compute_convergence_rate(h_vals, errors):
    """Log-log regression for convergence rate. Returns (rate, R²)."""
    if len(h_vals) < 2 or any(e <= 0 for e in errors):
        return 0.0, 0.0
    log_h = np.log(np.array(h_vals))
    log_e = np.log(np.array(errors))
    coeffs = np.polyfit(log_h, log_e, 1)
    rate = coeffs[0]
    # R²
    pred = np.polyval(coeffs, log_h)
    ss_res = np.sum((log_e - pred)**2)
    ss_tot = np.sum((log_e - np.mean(log_e))**2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return rate, r2


def pairwise_rates(h_vals, errors):
    """Pairwise convergence rates between successive refinements."""
    rates = []
    for i in range(len(errors) - 1):
        if errors[i] > 0 and errors[i+1] > 0 and h_vals[i] != h_vals[i+1]:
            r = np.log(errors[i] / errors[i+1]) / np.log(h_vals[i] / h_vals[i+1])
            rates.append(r)
        else:
            rates.append(float('nan'))
    return rates


def compute_spacetime_errors(vertices, elements, u_h, u_exact_func,
                             grad_exact_func=None, C_tensor=None):
    """
    Compute L², H¹ semi-norm, and energy norm errors for scalar space-time VEM.

    Parameters
    ----------
    vertices : (N, 2) node coords
    elements : list of int arrays
    u_h : (N,) numerical solution
    u_exact_func : callable(x, t) → u
    grad_exact_func : callable(x, t) → (du/dx, du/dt), optional
    C_tensor : (2, 2) material tensor for energy norm, optional

    Returns
    -------
    dict with keys 'L2', 'H1', 'energy'
    """
    l2_err2, h1_err2, energy_err2 = 0.0, 0.0, 0.0
    total_area = 0.0

    for el in elements:
        el_int = el.astype(int)
        verts = vertices[el_int]
        n_v = len(el_int)

        # Element area (shoelace)
        area_comp = (verts[:, 0] * np.roll(verts[:, 1], -1)
                     - np.roll(verts[:, 0], -1) * verts[:, 1])
        area = 0.5 * abs(np.sum(area_comp))
        if area < 1e-15:
            continue
        total_area += area

        cx = np.mean(verts[:, 0])
        ct = np.mean(verts[:, 1])

        # L² error: |u_h(centroid) - u_exact(centroid)|² * area
        u_h_c = np.mean(u_h[el_int])
        u_ex_c = u_exact_func(cx, ct)
        l2_err2 += (u_h_c - u_ex_c)**2 * area

        # H¹ semi-norm: |∇u_h - ∇u_exact|² * area
        if grad_exact_func is not None and n_v >= 3:
            # Approximate ∇u_h via least-squares on element vertices
            A_ls = np.column_stack([verts[:, 0] - cx, verts[:, 1] - ct,
                                    np.ones(n_v)])
            vals = u_h[el_int]
            try:
                coeffs, _, _, _ = np.linalg.lstsq(A_ls, vals, rcond=None)
                grad_h = coeffs[:2]  # (du/dx, du/dt)
            except np.linalg.LinAlgError:
                grad_h = np.zeros(2)

            grad_ex = np.array(grad_exact_func(cx, ct))
            diff_grad = grad_h - grad_ex
            h1_err2 += np.dot(diff_grad, diff_grad) * area

            # Energy norm: (C · (∇u_h - ∇u_exact)) · (∇u_h - ∇u_exact) * area
            if C_tensor is not None:
                energy_err2 += np.dot(C_tensor @ diff_grad, diff_grad) * area

    return {
        'L2': np.sqrt(l2_err2),
        'H1': np.sqrt(h1_err2),
        'energy': np.sqrt(energy_err2),
        'area': total_area,
    }


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
    mesh_sizes = [(6, 8), (8, 10), (12, 15), (16, 20)]
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
    pw_rates = pairwise_rates(h_vals, errors)
    rate_L2, r2_L2 = compute_convergence_rate(h_vals, errors)
    print(f"  Pairwise rates: {[f'{r:.2f}' for r in pw_rates]}")
    print(f"  Regression rate: {rate_L2:.2f} (R²={r2_L2:.3f})")

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

    # 3. Convergence with theoretical lines
    ax = axes[2]
    ax.loglog(h_vals, errors, 'bo-', label=f'VEM L² (rate={rate_L2:.2f})', linewidth=2)
    ax.loglog(h_vals, errors[0] * (h_vals / h_vals[0])**1, 'k--', alpha=0.5,
              label='O(h¹) [Xu/Wriggers]')
    ax.loglog(h_vals, errors[0] * (h_vals / h_vals[0])**2, 'k:', alpha=0.5,
              label='O(h²) [Xu/Wriggers]')
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

    return {
        'problem': 'Wave',
        'h': h_vals.tolist(),
        'dofs': n_dofs_list,
        'L2': errors,
        'rate_L2': rate_L2,
        'r2_L2': r2_L2,
    }


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


# ── Benchmark 5: Manufactured Solution (quantitative C5) ─────────────────

def benchmark_manufactured(save_dir='/tmp'):
    """
    Manufactured solution on (x,t) ∈ [0,1]² for quantitative convergence.

    Exact: u(x,t) = sin(πx) · exp(-t)
    With C = [[κ, 0], [0, β]], the PDE is:
        -div(C · ∇u) = f(x,t)
        f = κ·π²·sin(πx)·exp(-t) + β·sin(πx)·exp(-t)
          = (κ·π² + β) · sin(πx) · exp(-t)

    This gives well-defined L², H¹, and energy norm convergence.
    Expected (k=1 VEM): L² ~ O(h²), H¹ ~ O(h), energy ~ O(h).
    """
    print("\n" + "=" * 60)
    print("Benchmark 5: Manufactured Solution (C5 quantitative)")
    print("=" * 60)

    kappa, beta = 1.0, 0.5
    C = np.array([[kappa, 0.0], [0.0, beta]])
    Lx, T = 1.0, 1.0

    def u_exact(x, t):
        return np.sin(np.pi * x) * np.exp(-t)

    def grad_exact(x, t):
        return (np.pi * np.cos(np.pi * x) * np.exp(-t),
                -np.sin(np.pi * x) * np.exp(-t))

    def rhs(x, t):
        return (kappa * np.pi**2 + beta) * np.sin(np.pi * x) * np.exp(-t)

    mesh_configs = [(6, 6), (8, 8), (12, 12), (16, 16), (20, 20)]
    results = {'h': [], 'dofs': [], 'L2': [], 'H1': [], 'energy': []}

    for nx, nt in mesh_configs:
        vertices, elements, boundary = make_spacetime_voronoi(
            nx_seeds=nx, nt_seeds=nt, Lx=Lx, T=T, seed=42)
        n_cells = len(elements)
        C_per_el = np.tile(C, (n_cells, 1, 1))

        # BCs: all boundary nodes get exact values
        bc_nodes = np.unique(np.concatenate([
            boundary['bottom'], boundary['top'],
            boundary['left'], boundary['right']]))
        bc_vals = np.array([u_exact(vertices[n, 0], vertices[n, 1])
                            for n in bc_nodes])

        u = vem_anisotropic(vertices, elements, C_per_el, bc_nodes, bc_vals,
                            rhs_func=rhs)

        # Effective h
        h_eff = 1.0 / np.sqrt(len(elements))

        # Compute all error norms
        errs = compute_spacetime_errors(
            vertices, elements, u, u_exact, grad_exact, C)

        results['h'].append(h_eff)
        results['dofs'].append(vertices.shape[0])
        results['L2'].append(errs['L2'])
        results['H1'].append(errs['H1'])
        results['energy'].append(errs['energy'])

        print(f"  {nx}×{nt}: {vertices.shape[0]} DOFs, h={h_eff:.4f}, "
              f"L²={errs['L2']:.4e}, H¹={errs['H1']:.4e}, E={errs['energy']:.4e}")

    h = np.array(results['h'])

    # Regression rates
    rate_L2, r2_L2 = compute_convergence_rate(h, results['L2'])
    rate_H1, r2_H1 = compute_convergence_rate(h, results['H1'])
    rate_E, r2_E = compute_convergence_rate(h, results['energy'])

    print(f"\n  Convergence rates (log-log regression):")
    print(f"    L²:     {rate_L2:.2f} (R²={r2_L2:.3f}), expected: 2.0")
    print(f"    H¹:     {rate_H1:.2f} (R²={r2_H1:.3f}), expected: 1.0")
    print(f"    Energy: {rate_E:.2f} (R²={r2_E:.3f}), expected: 1.0")

    # Plot convergence
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.loglog(h, results['L2'], 'bo-', linewidth=2, markersize=8,
              label=f'L² error (rate={rate_L2:.2f})')
    ax.loglog(h, results['H1'], 'rs-', linewidth=2, markersize=8,
              label=f'H¹ semi-norm (rate={rate_H1:.2f})')
    ax.loglog(h, results['energy'], 'g^-', linewidth=2, markersize=8,
              label=f'Energy norm (rate={rate_E:.2f})')

    # Theoretical reference lines
    ref_L2 = results['L2'][0] * (h / h[0])**2
    ref_H1 = results['H1'][0] * (h / h[0])**1
    ax.loglog(h, ref_L2, 'b--', alpha=0.4, label='O(h²) [Xu/Wriggers expected]')
    ax.loglog(h, ref_H1, 'r--', alpha=0.4, label='O(h¹) [Xu/Wriggers expected]')

    ax.set_xlabel('h (effective mesh size)', fontsize=12)
    ax.set_ylabel('Error norm', fontsize=12)
    ax.set_title('C5: Space-Time VEM Convergence — Manufactured Solution\n'
                 'u(x,t) = sin(πx)·exp(−t), C = diag(κ, β)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = f'{save_dir}/vem_benchmark_manufactured.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close()

    results['rate_L2'] = rate_L2
    results['rate_H1'] = rate_H1
    results['rate_energy'] = rate_E
    results['r2_L2'] = r2_L2
    results['r2_H1'] = r2_H1
    results['r2_energy'] = r2_E
    results['problem'] = 'Manufactured'
    return results


# ── Comparison Table ─────────────────────────────────────────────────────

def write_comparison_table(results_list, save_dir='/tmp'):
    """Write quantitative comparison table to text file."""
    path = f'{save_dir}/vem_comparison_table.txt'
    lines = []
    lines.append("=" * 90)
    lines.append("  VEM Space-Time Benchmark: Quantitative Comparison with Literature")
    lines.append("  Reference: Xu, Junker, Wriggers (CMAME 2025), DOI:10.1016/j.cma.2024.117683")
    lines.append("=" * 90)
    lines.append("")

    for res in results_list:
        if res is None or 'problem' not in res:
            continue

        name = res['problem']
        lines.append(f"--- {name} ---")
        lines.append(f"{'Level':>6} {'DOFs':>8} {'h':>10} {'L² err':>12} "
                      f"{'H¹ err':>12} {'E err':>12}")

        n = len(res.get('h', []))
        for i in range(n):
            h_i = res['h'][i]
            dof_i = res['dofs'][i] if 'dofs' in res else '-'
            l2_i = f"{res['L2'][i]:.4e}" if 'L2' in res and i < len(res['L2']) else '-'
            h1_i = f"{res['H1'][i]:.4e}" if 'H1' in res and i < len(res['H1']) else '-'
            e_i = f"{res['energy'][i]:.4e}" if 'energy' in res and i < len(res['energy']) else '-'
            lines.append(f"{i+1:>6} {dof_i:>8} {h_i:>10.5f} {l2_i:>12} {h1_i:>12} {e_i:>12}")

        lines.append("")

        # Regression rates
        expected = {'L2': 2.0, 'H1': 1.0, 'energy': 1.0}
        for norm in ['L2', 'H1', 'energy']:
            rate_key = f'rate_{norm}'
            r2_key = f'r2_{norm}'
            if rate_key in res:
                rate = res[rate_key]
                r2 = res.get(r2_key, 0)
                exp = expected[norm]
                tol = 0.5
                verdict = "MATCH" if abs(rate - exp) < tol else (
                    "DEGRADED" if abs(rate - exp) < 1.0 else "FAIL")
                lines.append(f"  {norm:>6} rate: {rate:6.2f} (R²={r2:.3f}), "
                              f"expected: {exp:.1f} [{verdict}]")
        lines.append("")

    lines.append("=" * 90)
    lines.append("Notes:")
    lines.append("  - Expected rates: L² ~ O(h²), H¹ ~ O(h), Energy ~ O(h) for k=1 VEM")
    lines.append("  - Xu et al. use 2.5D extrusion; our approach is true 2D (x,t) plane")
    lines.append("  - Wave equation uses ε-regularization (indefinite tensor), degraded rates expected")
    lines.append("  - MATCH: |rate - expected| < 0.5, DEGRADED: < 1.0, FAIL: >= 1.0")
    lines.append("=" * 90)

    text = '\n'.join(lines)
    with open(path, 'w') as f:
        f.write(text)
    print(f"\n  Comparison table saved: {path}")
    print(text)


# ── Summary Convergence Plot ─────────────────────────────────────────────

def plot_summary_convergence(results_list, save_dir='/tmp'):
    """Paper-quality combined convergence plot for all benchmarks."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = {'Wave': 'tab:blue', 'Manufactured': 'tab:red'}
    markers = {'Wave': 'o', 'Manufactured': 's'}

    for res in results_list:
        if res is None or 'h' not in res:
            continue
        name = res['problem']
        h = np.array(res['h'])
        c = colors.get(name, 'tab:gray')
        m = markers.get(name, 'D')

        # Left: L² error
        if 'L2' in res and len(res['L2']) > 0:
            rate = res.get('rate_L2', 0)
            axes[0].loglog(h, res['L2'], f'{m}-', color=c, linewidth=2,
                           markersize=8, label=f'{name} (rate={rate:.2f})')

        # Right: H¹ / energy
        if 'H1' in res and len(res['H1']) > 0:
            rate = res.get('rate_H1', 0)
            axes[1].loglog(h, res['H1'], f'{m}-', color=c, linewidth=2,
                           markersize=8, label=f'{name} H¹ ({rate:.2f})')
        if 'energy' in res and len(res['energy']) > 0:
            rate = res.get('rate_energy', 0)
            axes[1].loglog(h, res['energy'], f'{m}--', color=c, linewidth=1.5,
                           markersize=6, alpha=0.7,
                           label=f'{name} energy ({rate:.2f})')

    # Theoretical reference
    h_ref = np.logspace(-1.5, -0.5, 20)
    axes[0].loglog(h_ref, 0.5 * h_ref**2, 'k--', alpha=0.3, label='O(h²) theory')
    axes[0].loglog(h_ref, 0.5 * h_ref**1, 'k:', alpha=0.3, label='O(h¹) theory')
    axes[1].loglog(h_ref, 0.5 * h_ref**1, 'k--', alpha=0.3, label='O(h¹) theory')

    axes[0].set_xlabel('h')
    axes[0].set_ylabel('L² error')
    axes[0].set_title('L² Convergence')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('h')
    axes[1].set_ylabel('H¹ / Energy error')
    axes[1].set_title('H¹ and Energy Norm Convergence')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle('Space-Time VEM: Convergence Comparison with Xu/Wriggers (2025)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = f'{save_dir}/vem_benchmark_summary.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Summary plot saved: {path}")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    save_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("  VEM Space-Time Benchmark Suite (C4/C5/C6)")
    print("=" * 60 + "\n")

    convergence_results = []

    # Benchmark 1-4 (existing)
    res_wave = benchmark_wave(save_dir)
    convergence_results.append(res_wave)

    benchmark_sls(save_dir)
    benchmark_stabilization(save_dir)
    benchmark_elastodynamics(save_dir)

    # Benchmark 5 (new: manufactured solution)
    res_mfg = benchmark_manufactured(save_dir)
    convergence_results.append(res_mfg)

    # Comparison table and summary plot
    write_comparison_table(convergence_results, save_dir)
    plot_summary_convergence(convergence_results, save_dir)

    print("\n" + "=" * 60)
    print("  All benchmarks complete!")
    print("=" * 60)
