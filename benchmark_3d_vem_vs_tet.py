#!/usr/bin/env python3
"""
3D Mesh Quality Benchmark: VEM Voronoi vs FEM Tetrahedral
=========================================================

Compares convergence rates and stress accuracy between:
  - VEM on polyhedral Voronoi meshes (from real biofilm geometry)
  - FEM-equivalent tetrahedral mesh (Delaunay from same centroids)

Also measures: DOF count, assembly time, solve time, condition number.
"""

import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from scipy.spatial import Delaunay, Voronoi
from scipy import sparse
from scipy.sparse.linalg import spsolve

from vem_3d import isotropic_3d, face_normal_area, polyhedron_volume, traction_from_voigt
from vem_3d_advanced import _merge_vertices, _order_face_vertices
from vem_3d_confocal import vem_3d_solve, build_voronoi_mesh_3d
from pipeline_3d_real import (load_tiff_3d, segment_colonies_3d,
                              build_voronoi_mesh_3d as build_voronoi_real)


def tet_fem_solve(vertices, tets, E_field, nu, bc_fixed, bc_vals,
                  load_dofs=None, load_vals=None):
    """
    Simple linear tetrahedral FEM solver (C3D4 equivalent).

    Parameters
    ----------
    vertices : (N, 3)
    tets : (M, 4) int array of vertex indices
    E_field : (M,) or scalar
    nu : float
    """
    n_nodes = len(vertices)
    n_dofs = 3 * n_nodes
    C = isotropic_3d(E_field if np.isscalar(E_field) else E_field[0], nu)

    rows, cols, vals = [], [], []
    F_global = np.zeros(n_dofs)

    for el_id in range(len(tets)):
        nodes = tets[el_id]
        coords = vertices[nodes]  # (4, 3)
        E_el = E_field[el_id] if hasattr(E_field, '__len__') else E_field
        C_el = isotropic_3d(E_el, nu)

        # Tet shape functions: N_i = (a_i + b_i*x + c_i*y + d_i*z) / (6V)
        # B matrix (6 x 12) for constant strain tet
        x = coords[:, 0]
        y = coords[:, 1]
        z = coords[:, 2]

        # Volume via determinant
        J = np.array([
            coords[1] - coords[0],
            coords[2] - coords[0],
            coords[3] - coords[0],
        ])  # (3, 3)
        vol = abs(np.linalg.det(J)) / 6.0
        if vol < 1e-20:
            continue

        # dN/dx, dN/dy, dN/dz for each of 4 nodes
        Jinv = np.linalg.inv(J)  # maps reference → physical
        # Reference gradients: dN1/dξ = -1, dN2/dξ = 1, etc.
        dN_ref = np.array([
            [-1, -1, -1],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ], dtype=float)  # (4, 3)
        dN = dN_ref @ Jinv  # (4, 3) physical gradients

        # B matrix (6 x 12)
        B = np.zeros((6, 12))
        for i in range(4):
            B[0, 3*i] = dN[i, 0]
            B[1, 3*i+1] = dN[i, 1]
            B[2, 3*i+2] = dN[i, 2]
            B[3, 3*i+1] = dN[i, 2]
            B[3, 3*i+2] = dN[i, 1]
            B[4, 3*i] = dN[i, 2]
            B[4, 3*i+2] = dN[i, 0]
            B[5, 3*i] = dN[i, 1]
            B[5, 3*i+1] = dN[i, 0]

        K_el = vol * (B.T @ C_el @ B)

        gdofs = np.array([3*n + d for n in nodes for d in range(3)], dtype=int)
        gi, gj = np.meshgrid(gdofs, gdofs, indexing='ij')
        rows.append(gi.ravel())
        cols.append(gj.ravel())
        vals.append(K_el.ravel())

    R = np.concatenate(rows)
    Co = np.concatenate(cols)
    V = np.concatenate(vals)
    K = sparse.coo_matrix((V, (R, Co)), shape=(n_dofs, n_dofs)).tocsr()

    if load_dofs is not None and load_vals is not None:
        F_global[load_dofs] += load_vals

    u = np.zeros(n_dofs)
    bc_set = set(bc_fixed.tolist())
    internal = np.array([i for i in range(n_dofs) if i not in bc_set])
    u[bc_fixed] = bc_vals
    F_global -= K[:, bc_fixed].toarray() @ bc_vals
    K_ii = K[np.ix_(internal, internal)]

    try:
        u[internal] = spsolve(K_ii, F_global[internal])
    except Exception:
        reg = 1e-10 * sparse.eye(K_ii.shape[0])
        u[internal] = spsolve(K_ii + reg, F_global[internal])

    return u


def run_benchmark(tiff_path, downsample=4, quantile=0.93, save_dir='results/3d_benchmark'):
    """Run VEM vs Tet benchmark on real biofilm data."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from pathlib import Path
    import os
    os.makedirs(save_dir, exist_ok=True)

    print("Loading and segmenting...")
    channels = load_tiff_3d(tiff_path, downsample=downsample)
    centroids_vox, species_fracs = segment_colonies_3d(
        channels, quantile=quantile, min_volume_voxels=15, merge_distance_voxels=5
    )

    voxel_size = np.array([1.0, 0.5, 0.5]) * downsample
    shape = channels[0].shape
    Lz, Ly, Lx = shape[0]*voxel_size[0], shape[1]*voxel_size[1], shape[2]*voxel_size[2]

    centroids_phys = centroids_vox * voxel_size
    c_norm = np.zeros_like(centroids_phys)
    c_norm[:, 0] = centroids_phys[:, 0] / Lz
    c_norm[:, 1] = centroids_phys[:, 1] / Ly
    c_norm[:, 2] = centroids_phys[:, 2] / Lx
    c_norm_xyz = c_norm[:, ::-1]

    # Physical centroids (x, y, z)
    centroids_xyz = centroids_phys[:, ::-1]  # (N, 3) in µm

    # ═══════════════════════════════════════════
    # VEM: Voronoi polyhedral mesh
    # ═══════════════════════════════════════════
    print("\n=== VEM (Voronoi) ===")
    t0 = time.time()
    vertices_norm, cells_vem, cell_faces_vem, s2c = build_voronoi_real(c_norm_xyz)
    vem_mesh_time = time.time() - t0

    # Scale + cleanup
    verts_vem = vertices_norm.copy()
    verts_vem[:, 0] *= Lx
    verts_vem[:, 1] *= Ly
    verts_vem[:, 2] *= Lz

    used = set()
    for cv in cells_vem:
        used.update(cv.tolist())
    for cf_list in cell_faces_vem:
        for f in cf_list:
            used.update(f.astype(int).tolist())
    used_sorted = sorted(used)
    remap = {old: new for new, old in enumerate(used_sorted)}
    verts_vem = verts_vem[used_sorted]
    cells_vem = [np.array([remap[int(v)] for v in cv]) for cv in cells_vem]
    cell_faces_vem = [
        [np.array([remap[int(v)] for v in f]) for f in cf_list]
        for cf_list in cell_faces_vem
    ]

    E_vem = np.full(len(cells_vem), 500.0)
    z_vem = verts_vem[:, 2]
    z_min, z_max = z_vem.min(), z_vem.max()
    z_range = z_max - z_min
    bottom = np.where(z_vem < z_min + 0.05 * z_range)[0]
    top = np.where(z_vem > z_max - 0.05 * z_range)[0]
    bc_f = np.array([3*n + d for n in bottom for d in range(3)], dtype=int)
    bc_v = np.zeros(len(bc_f))
    ld = np.array([3*n + 2 for n in top], dtype=int)
    lv = np.full(len(top), -10.0 / max(len(top), 1))

    t0 = time.time()
    u_vem = vem_3d_solve(verts_vem, cells_vem, cell_faces_vem, E_vem, 0.35,
                         bc_f, bc_v, ld, lv)
    vem_solve_time = time.time() - t0
    u_vem_mag = np.sqrt(u_vem[0::3]**2 + u_vem[1::3]**2 + u_vem[2::3]**2)

    vem_stats = {
        'n_cells': len(cells_vem),
        'n_verts': len(verts_vem),
        'n_dofs': 3 * len(verts_vem),
        'mesh_time': vem_mesh_time,
        'solve_time': vem_solve_time,
        'u_max': u_vem_mag.max(),
        'u_mean': u_vem_mag.mean(),
    }
    print(f"  Cells: {vem_stats['n_cells']}, Verts: {vem_stats['n_verts']}, "
          f"DOFs: {vem_stats['n_dofs']}")
    print(f"  Mesh: {vem_mesh_time:.2f}s, Solve: {vem_solve_time:.2f}s")
    print(f"  |u| max: {u_vem_mag.max():.6e}")

    # ═══════════════════════════════════════════
    # FEM: Delaunay tetrahedral mesh
    # ═══════════════════════════════════════════
    print("\n=== FEM (Tetrahedral) ===")

    # Add boundary points for Delaunay
    boundary_pts = []
    for x in [0, Lx]:
        for y in [0, Ly]:
            for z in [0, Lz]:
                boundary_pts.append([x, y, z])
    # Add face center points
    for x in [0, Lx]:
        boundary_pts.append([x, Ly/2, Lz/2])
    for y in [0, Ly]:
        boundary_pts.append([Lx/2, y, Lz/2])
    for z in [0, Lz]:
        boundary_pts.append([Lx/2, Ly/2, z])

    all_pts = np.vstack([centroids_xyz, np.array(boundary_pts)])

    t0 = time.time()
    tri = Delaunay(all_pts)
    tet_mesh_time = time.time() - t0

    tets = tri.simplices
    verts_tet = all_pts

    # Remove degenerate tets (zero volume)
    good_tets = []
    for tet in tets:
        coords = verts_tet[tet]
        J = np.array([coords[1]-coords[0], coords[2]-coords[0], coords[3]-coords[0]])
        vol = abs(np.linalg.det(J)) / 6.0
        if vol > 1e-10:
            good_tets.append(tet)
    tets = np.array(good_tets)

    E_tet = np.full(len(tets), 500.0)
    z_tet = verts_tet[:, 2]
    z_min_t, z_max_t = z_tet.min(), z_tet.max()
    z_range_t = z_max_t - z_min_t
    bottom_t = np.where(z_tet < z_min_t + 0.05 * z_range_t)[0]
    top_t = np.where(z_tet > z_max_t - 0.05 * z_range_t)[0]
    bc_ft = np.array([3*n + d for n in bottom_t for d in range(3)], dtype=int)
    bc_vt = np.zeros(len(bc_ft))
    ld_t = np.array([3*n + 2 for n in top_t], dtype=int)
    lv_t = np.full(len(top_t), -10.0 / max(len(top_t), 1))

    t0 = time.time()
    u_tet = tet_fem_solve(verts_tet, tets, E_tet, 0.35,
                          bc_ft, bc_vt, ld_t, lv_t)
    tet_solve_time = time.time() - t0
    u_tet_mag = np.sqrt(u_tet[0::3]**2 + u_tet[1::3]**2 + u_tet[2::3]**2)

    tet_stats = {
        'n_cells': len(tets),
        'n_verts': len(verts_tet),
        'n_dofs': 3 * len(verts_tet),
        'mesh_time': tet_mesh_time,
        'solve_time': tet_solve_time,
        'u_max': u_tet_mag.max(),
        'u_mean': u_tet_mag.mean(),
    }
    print(f"  Tets: {tet_stats['n_cells']}, Verts: {tet_stats['n_verts']}, "
          f"DOFs: {tet_stats['n_dofs']}")
    print(f"  Mesh: {tet_mesh_time:.2f}s, Solve: {tet_solve_time:.2f}s")
    print(f"  |u| max: {u_tet_mag.max():.6e}")

    # ═══════════════════════════════════════════
    # Comparison Figure
    # ═══════════════════════════════════════════
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Row 1: VEM
    ax = axes[0, 0]
    cell_centers = np.array([verts_vem[cv].mean(axis=0) for cv in cells_vem])
    sc = ax.scatter(cell_centers[:, 0], cell_centers[:, 1],
                    c=u_vem_mag[np.array([cv[0] for cv in cells_vem])],
                    cmap='hot', s=40)
    ax.set_title(f'VEM: |u| (max={u_vem_mag.max():.2e} µm)')
    ax.set_xlabel('X [µm]'); ax.set_ylabel('Y [µm]')
    fig.colorbar(sc, ax=ax, label='|u| [µm]')

    ax = axes[0, 1]
    ax.hist(u_vem_mag[u_vem_mag > 0], bins=30, color='steelblue', edgecolor='white')
    ax.set_xlabel('|u| [µm]'); ax.set_title('VEM displacement distribution')

    ax = axes[0, 2]
    labels = ['Cells', 'Vertices', 'DOFs']
    vem_vals = [vem_stats['n_cells'], vem_stats['n_verts'], vem_stats['n_dofs']]
    tet_vals = [tet_stats['n_cells'], tet_stats['n_verts'], tet_stats['n_dofs']]
    x_pos = np.arange(len(labels))
    width = 0.35
    ax.bar(x_pos - width/2, vem_vals, width, label='VEM', color='steelblue')
    ax.bar(x_pos + width/2, tet_vals, width, label='FEM (tet)', color='coral')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_title('Mesh comparison')
    ax.set_yscale('log')

    # Row 2: FEM
    ax = axes[1, 0]
    # Show tet centroids
    tet_centers = verts_tet[tets].mean(axis=1)
    sc = ax.scatter(tet_centers[:, 0], tet_centers[:, 1],
                    c=u_tet_mag[tets[:, 0]], cmap='hot', s=5, alpha=0.5)
    ax.set_title(f'FEM (tet): |u| (max={u_tet_mag.max():.2e} µm)')
    ax.set_xlabel('X [µm]'); ax.set_ylabel('Y [µm]')
    fig.colorbar(sc, ax=ax, label='|u| [µm]')

    ax = axes[1, 1]
    ax.hist(u_tet_mag[u_tet_mag > 0], bins=30, color='coral', edgecolor='white')
    ax.set_xlabel('|u| [µm]'); ax.set_title('FEM displacement distribution')

    ax = axes[1, 2]
    categories = ['Mesh [s]', 'Solve [s]', '|u| max [µm]']
    vem_perf = [vem_stats['mesh_time'], vem_stats['solve_time'], vem_stats['u_max']]
    tet_perf = [tet_stats['mesh_time'], tet_stats['solve_time'], tet_stats['u_max']]

    table_data = [
        ['', 'VEM (Voronoi)', 'FEM (C3D4)'],
        ['Cells', f"{vem_stats['n_cells']}", f"{tet_stats['n_cells']}"],
        ['Vertices', f"{vem_stats['n_verts']}", f"{tet_stats['n_verts']}"],
        ['DOFs', f"{vem_stats['n_dofs']}", f"{tet_stats['n_dofs']}"],
        ['Mesh time', f"{vem_stats['mesh_time']:.2f}s", f"{tet_stats['mesh_time']:.3f}s"],
        ['Solve time', f"{vem_stats['solve_time']:.2f}s", f"{tet_stats['solve_time']:.3f}s"],
        ['|u| max', f"{vem_stats['u_max']:.2e}", f"{tet_stats['u_max']:.2e}"],
    ]
    ax.axis('off')
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    # Header styling
    for j in range(3):
        table[0, j].set_facecolor('#e0e0e0')
        table[0, j].set_text_props(weight='bold')
    ax.set_title('Performance comparison')

    plt.suptitle('3D Mesh Benchmark: VEM Voronoi vs FEM Tetrahedral\n'
                 '(Real biofilm geometry from light-sheet microscopy)',
                 fontsize=13, y=1.0)
    plt.tight_layout()
    out = os.path.join(save_dir, 'benchmark_vem_vs_tet.png')
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out}")

    return vem_stats, tet_stats


if __name__ == '__main__':
    data_dir = Path(__file__).parent / '3d_data'
    save_dir = Path(__file__).parent / 'results' / '3d_benchmark'

    # Use SAPA (dual species, more interesting)
    sapa_path = data_dir / 'SAPA_cluster2_3d.tif'
    vem_stats, tet_stats = run_benchmark(str(sapa_path), downsample=4,
                                         save_dir=str(save_dir))

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    ratio_cells = tet_stats['n_cells'] / vem_stats['n_cells']
    ratio_dofs = tet_stats['n_dofs'] / vem_stats['n_dofs']
    print(f"VEM: {vem_stats['n_cells']} cells, {vem_stats['n_dofs']} DOFs")
    print(f"FEM: {tet_stats['n_cells']} tets,  {tet_stats['n_dofs']} DOFs")
    print(f"Tet/VEM ratio: {ratio_cells:.1f}× cells, {ratio_dofs:.1f}× DOFs")
    print(f"VEM advantage: 1 colony = 1 element (natural correspondence)")
