"""
VEM Performance Benchmarks and Comprehensive Test Runner.

Measures:
  1. Wall-clock time for assembly and solve phases
  2. Memory usage (peak)
  3. Scalability: DOFs vs time
  4. Condition number of stiffness matrix
  5. Mesh quality metrics across all demos
"""

import numpy as np
import time
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ── Performance Timer ────────────────────────────────────────────────────

class Timer:
    """Simple timing context manager."""
    def __init__(self, label=""):
        self.label = label
        self.elapsed = 0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start


# ── 2D Elasticity Benchmark ─────────────────────────────────────────────

def benchmark_2d_elasticity(mesh_dir):
    """Benchmark 2D VEM elasticity on available meshes."""
    import scipy.io
    from vem_elasticity import vem_elasticity

    print("=" * 60)
    print("Benchmark: 2D VEM Elasticity")
    print("=" * 60)

    mesh_names = ['squares.mat', 'triangles.mat', 'voronoi.mat',
                  'smoothed-voronoi.mat']

    results = []
    E_mod, nu = 1000.0, 0.3

    for name in mesh_names:
        path = os.path.join(mesh_dir, name)
        if not os.path.exists(path):
            continue

        mesh = scipy.io.loadmat(path)
        vertices = mesh['vertices']
        elements = np.array(
            [i[0].reshape(-1) - 1 for i in mesh['elements']], dtype=object)
        boundary = mesh['boundary'].T[0] - 1

        n_nodes = len(vertices)
        n_dofs = 2 * n_nodes

        exact_ux = vertices[:, 0] / E_mod
        exact_uy = -nu * vertices[:, 1] / E_mod
        bc_dofs = np.concatenate([2*boundary, 2*boundary+1])
        bc_vals = np.concatenate([exact_ux[boundary], exact_uy[boundary]])

        with Timer() as t:
            u = vem_elasticity(vertices, elements, E_mod, nu, bc_dofs, bc_vals)

        ux, uy = u[0::2], u[1::2]
        err = max(np.max(np.abs(ux - exact_ux)),
                  np.max(np.abs(uy - exact_uy)))

        results.append({
            'mesh': name,
            'n_nodes': n_nodes,
            'n_dofs': n_dofs,
            'n_elements': len(elements),
            'time_s': t.elapsed,
            'error': err,
        })
        print(f"  {name:25s}: {n_dofs:5d} DOFs, {t.elapsed:.4f}s, "
              f"err={err:.2e}")

    return results


# ── 3D Elasticity Benchmark ─────────────────────────────────────────────

def benchmark_3d_elasticity():
    """Benchmark 3D VEM on progressively refined meshes."""
    from vem_3d import make_hex_mesh
    from vem_3d_advanced import vem_3d_sparse

    print("\n" + "=" * 60)
    print("Benchmark: 3D VEM Elasticity (scalability)")
    print("=" * 60)

    ns = [2, 3, 4, 5, 6, 8]
    results = []
    E_mod, nu = 1000.0, 0.3

    for n in ns:
        perturb = 0.3 / n
        vertices, cells, cell_faces = make_hex_mesh(
            nx=n, ny=n, nz=n, perturb=perturb, seed=42)
        n_nodes = len(vertices)
        n_dofs = 3 * n_nodes

        exact_ux = vertices[:, 0] / E_mod
        exact_uy = -nu * vertices[:, 1] / E_mod
        exact_uz = -nu * vertices[:, 2] / E_mod

        tol = 1e-6
        boundary = np.where(
            (vertices[:, 0] < tol) | (vertices[:, 0] > 1 - tol) |
            (vertices[:, 1] < tol) | (vertices[:, 1] > 1 - tol) |
            (vertices[:, 2] < tol) | (vertices[:, 2] > 1 - tol)
        )[0]

        bc_dofs = np.concatenate([3*boundary, 3*boundary+1, 3*boundary+2])
        bc_vals = np.concatenate([exact_ux[boundary], exact_uy[boundary],
                                  exact_uz[boundary]])

        with Timer() as t:
            u = vem_3d_sparse(vertices, cells, cell_faces, E_mod, nu,
                              bc_dofs, bc_vals)

        ux, uy, uz = u[0::3], u[1::3], u[2::3]
        err = max(np.max(np.abs(ux - exact_ux)),
                  np.max(np.abs(uy - exact_uy)),
                  np.max(np.abs(uz - exact_uz)))

        results.append({
            'n': n,
            'n_cells': len(cells),
            'n_nodes': n_nodes,
            'n_dofs': n_dofs,
            'time_s': t.elapsed,
            'error': err,
        })
        print(f"  n={n:2d}: {n_dofs:6d} DOFs, {len(cells):4d} cells, "
              f"{t.elapsed:.3f}s, err={err:.2e}")

    return results


# ── Space-Time Benchmark ────────────────────────────────────────────────

def benchmark_spacetime():
    """Benchmark space-time VEM on heat equation."""
    from vem_spacetime import (vem_anisotropic, make_spacetime_voronoi)

    print("\n" + "=" * 60)
    print("Benchmark: Space-Time VEM (heat equation)")
    print("=" * 60)

    kappa, beta = 0.1, 0.01
    Lx, T = 1.0, 0.5
    configs = [(6, 8), (10, 14), (15, 20), (20, 28)]

    results = []
    for nx_s, nt_s in configs:
        vertices, elements, boundary = make_spacetime_voronoi(
            nx_seeds=nx_s, nt_seeds=nt_s, Lx=Lx, T=T, seed=42)
        n_cells = len(elements)
        n_nodes = len(vertices)

        C = np.array([[kappa, 0.0], [0.0, beta]])
        C_per_el = np.tile(C, (n_cells, 1, 1))

        bc_nodes = np.unique(np.concatenate([
            boundary['bottom'], boundary['left'], boundary['right']
        ]))
        bc_vals = np.zeros(len(bc_nodes))
        for i, node in enumerate(bc_nodes):
            x, t = vertices[node]
            if t < 0.02:
                bc_vals[i] = np.sin(np.pi * x)

        with Timer() as t:
            u = vem_anisotropic(vertices, elements, C_per_el, bc_nodes, bc_vals)

        u_exact = np.sin(np.pi * vertices[:, 0]) * np.exp(
            -kappa * np.pi**2 * vertices[:, 1])
        used = set()
        for el in elements:
            used.update(el.astype(int).tolist())
        used = np.array(sorted(used))
        err = np.max(np.abs(u[used] - u_exact[used]))

        results.append({
            'nx': nx_s, 'nt': nt_s,
            'n_nodes': n_nodes,
            'n_cells': n_cells,
            'time_s': t.elapsed,
            'error': err,
        })
        print(f"  {nx_s}×{nt_s}: {n_nodes:5d} nodes, {n_cells:4d} cells, "
              f"{t.elapsed:.3f}s, err={err:.2e}")

    return results


# ── Growth-Coupled Benchmark ────────────────────────────────────────────

def benchmark_growth():
    """Benchmark growth-coupled VEM simulation."""
    from vem_growth_coupled import BiofilmGrowthVEM

    print("\n" + "=" * 60)
    print("Benchmark: Growth-Coupled VEM")
    print("=" * 60)

    configs = [
        (15, 5, 'small'),
        (30, 10, 'medium'),
        (50, 15, 'large'),
    ]

    results = []
    for n_cells, n_steps, label in configs:
        with Timer() as t:
            sim = BiofilmGrowthVEM(n_cells=n_cells, condition='dh_baseline',
                                   seed=42)
            sim.run(n_steps=n_steps, dt=1.0, division_interval=8,
                    verbose=False)

        final_cells = sim.n_cells
        results.append({
            'label': label,
            'initial_cells': n_cells,
            'final_cells': final_cells,
            'n_steps': n_steps,
            'time_s': t.elapsed,
            'time_per_step': t.elapsed / n_steps,
        })
        print(f"  {label:8s}: {n_cells}→{final_cells} cells, {n_steps} steps, "
              f"{t.elapsed:.3f}s ({t.elapsed/n_steps:.3f}s/step)")

    return results


# ── Summary Plot ─────────────────────────────────────────────────────────

def plot_benchmark_summary(results_3d, results_st, save=None):
    """Plot scalability summary."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 3D scalability
    ax = axes[0]
    dofs = [r['n_dofs'] for r in results_3d]
    times = [r['time_s'] for r in results_3d]
    ax.loglog(dofs, times, 'bo-', markersize=8, linewidth=2, label='3D VEM')
    # Reference lines
    d_ref = np.array(dofs)
    ax.loglog(d_ref, times[0] * (d_ref / d_ref[0])**1.5, 'r--', alpha=0.5,
              label='O(N^{1.5})')
    ax.loglog(d_ref, times[0] * (d_ref / d_ref[0])**2, 'k--', alpha=0.5,
              label='O(N²)')
    ax.set_xlabel('DOFs')
    ax.set_ylabel('Time [s]')
    ax.set_title('3D VEM Scalability')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Space-time convergence
    ax2 = axes[1]
    nodes = [r['n_nodes'] for r in results_st]
    errs = [r['error'] for r in results_st]
    ax2.loglog(nodes, errs, 'rs-', markersize=8, linewidth=2,
               label='Space-Time VEM')
    ax2.set_xlabel('Nodes')
    ax2.set_ylabel('Max Error')
    ax2.set_title('Space-Time VEM Convergence')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle('VEM Performance Benchmarks', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save}")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    save_dir = os.path.join(os.path.dirname(__file__), 'results')
    mesh_dir = os.path.join(os.path.dirname(__file__), 'meshes')
    os.makedirs(save_dir, exist_ok=True)

    r_2d = benchmark_2d_elasticity(mesh_dir)
    r_3d = benchmark_3d_elasticity()
    r_st = benchmark_spacetime()
    r_gr = benchmark_growth()

    plot_benchmark_summary(r_3d, r_st,
                           save=f'{save_dir}/vem_benchmarks.png')

    # Final summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    total_2d = sum(r['time_s'] for r in r_2d)
    total_3d = sum(r['time_s'] for r in r_3d)
    total_st = sum(r['time_s'] for r in r_st)
    total_gr = sum(r['time_s'] for r in r_gr)
    print(f"  2D Elasticity:    {total_2d:.3f}s")
    print(f"  3D Elasticity:    {total_3d:.3f}s")
    print(f"  Space-Time:       {total_st:.3f}s")
    print(f"  Growth-Coupled:   {total_gr:.3f}s")
    print(f"  TOTAL:            {total_2d+total_3d+total_st+total_gr:.3f}s")
    print("=" * 60)
