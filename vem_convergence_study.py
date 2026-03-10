"""
h-Convergence study for VEM (2D elasticity) with manufactured solution.
Also: quantitative comparison VEM vs standard FEM (triangular).

Manufactured solution (plane stress, E=1000, nu=0.3):
  u_x = sin(pi*x) * sin(pi*y)
  u_y = cos(pi*x) * cos(pi*y)
  => compute body force f = -div(sigma) analytically

References:
  - Beirao da Veiga et al. (2013) patch test + convergence theory
  - Expected rates: O(h) in H1, O(h^2) in L2 for k=1 VEM
"""

import numpy as np
from scipy.spatial import Voronoi
import matplotlib.pyplot as plt
import os, sys, time

sys.path.insert(0, os.path.dirname(__file__))
from vem_elasticity import vem_elasticity


# ── Manufactured Solution ────────────────────────────────────────────────

def manufactured_solution(x, y):
    """Exact displacement field."""
    ux = np.sin(np.pi * x) * np.sin(np.pi * y)
    uy = np.cos(np.pi * x) * np.cos(np.pi * y)
    return ux, uy


def manufactured_body_force(x, y, E=1000.0, nu=0.3):
    """
    Body force f = -div(sigma) for the manufactured solution.
    Plane stress: sigma = C * epsilon(u)
    """
    pi = np.pi
    # Strain components
    # eps_xx = pi*cos(pi*x)*sin(pi*y)
    # eps_yy = -pi*cos(pi*x)*cos(pi*y)  (wait, let me recompute)

    # u_x = sin(pi*x)*sin(pi*y)
    # u_y = cos(pi*x)*cos(pi*y)
    # du_x/dx = pi*cos(pi*x)*sin(pi*y)
    # du_x/dy = pi*sin(pi*x)*cos(pi*y)
    # du_y/dx = -pi*sin(pi*x)*cos(pi*y)
    # du_y/dy = -pi*cos(pi*x)*sin(pi*y)

    # eps_xx = du_x/dx = pi*cos(pi*x)*sin(pi*y)
    # eps_yy = du_y/dy = -pi*cos(pi*x)*sin(pi*y)
    # eps_xy = 0.5*(du_x/dy + du_y/dx) = 0.5*pi*(sin(pi*x)*cos(pi*y) - sin(pi*x)*cos(pi*y)) = 0

    # Plane stress C matrix
    c = E / (1 - nu**2)

    # sigma_xx = c*(eps_xx + nu*eps_yy) = c*pi*cos(pi*x)*sin(pi*y)*(1 - nu)
    # sigma_yy = c*(nu*eps_xx + eps_yy) = c*pi*cos(pi*x)*sin(pi*y)*(nu - 1)
    # sigma_xy = c*(1-nu)/2 * 2*eps_xy = 0

    # dsigma_xx/dx = c*pi*(1-nu)*(-pi*sin(pi*x)*sin(pi*y))
    # dsigma_xy/dy = 0
    # dsigma_yy/dy = c*pi*(nu-1)*(pi*cos(pi*x)*cos(pi*y))
    # dsigma_xy/dx = 0

    # f_x = -(dsigma_xx/dx + dsigma_xy/dy) = c*pi^2*(1-nu)*sin(pi*x)*sin(pi*y)
    # f_y = -(dsigma_xy/dx + dsigma_yy/dy) = -c*pi^2*(nu-1)*cos(pi*x)*cos(pi*y)
    #      = c*pi^2*(1-nu)*cos(pi*x)*cos(pi*y)

    fx = c * pi**2 * (1 - nu) * np.sin(pi * x) * np.sin(pi * y)
    fy = c * pi**2 * (1 - nu) * np.cos(pi * x) * np.cos(pi * y)
    return fx, fy


# ── Voronoi Mesh Generation ─────────────────────────────────────────────

def generate_voronoi_mesh(n_per_side, Lx=1.0, Ly=1.0, seed=42, perturbation=0.3):
    """
    Generate a perturbed-grid Voronoi mesh on [0,Lx]×[0,Ly].

    Uses a regular grid with random perturbation as seeds, then constructs
    a bounded Voronoi diagram using mirror points. This ensures no
    degenerate boundary clipping issues.
    """
    rng = np.random.default_rng(seed)
    h = Lx / n_per_side

    # Regular grid seeds + perturbation
    seeds = []
    for j in range(n_per_side):
        for i in range(n_per_side):
            cx = (i + 0.5) * h + perturbation * h * (rng.random() - 0.5)
            cy = (j + 0.5) * h + perturbation * h * (rng.random() - 0.5)
            cx = np.clip(cx, 0.05 * h, Lx - 0.05 * h)
            cy = np.clip(cy, 0.05 * h, Ly - 0.05 * h)
            seeds.append([cx, cy])
    seeds = np.array(seeds)
    n_seeds = len(seeds)

    # Mirror for bounded Voronoi
    all_pts = [seeds]
    for axis, bounds in [(0, [0.0, Lx]), (1, [0.0, Ly])]:
        for val in bounds:
            mirror = seeds.copy()
            mirror[:, axis] = 2 * val - mirror[:, axis]
            all_pts.append(mirror)
    all_pts = np.vstack(all_pts)
    vor = Voronoi(all_pts)

    # Extract bounded cells
    elements = []
    used_verts = set()

    for i in range(n_seeds):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]

        if -1 in region or len(region) < 3:
            continue

        verts = vor.vertices[region]
        # Clip vertices to domain
        verts[:, 0] = np.clip(verts[:, 0], 0, Lx)
        verts[:, 1] = np.clip(verts[:, 1], 0, Ly)

        elements.append(np.array(region))
        for vi in region:
            used_verts.add(vi)

    # Compact indexing
    used_sorted = sorted(used_verts)
    old_to_new = {old: new for new, old in enumerate(used_sorted)}
    vertices = vor.vertices[used_sorted].copy()
    vertices[:, 0] = np.clip(vertices[:, 0], 0, Lx)
    vertices[:, 1] = np.clip(vertices[:, 1], 0, Ly)
    elements_compact = [np.array([old_to_new[vi] for vi in el]) for el in elements]

    # Merge duplicate vertices (from clipping)
    # Round to avoid floating point duplicates
    tol_merge = 1e-10
    unique_map = {}
    new_vertices = []
    old_to_merged = {}
    for i, v in enumerate(vertices):
        key = (round(v[0] / tol_merge), round(v[1] / tol_merge))
        if key not in unique_map:
            unique_map[key] = len(new_vertices)
            new_vertices.append(v)
        old_to_merged[i] = unique_map[key]

    vertices = np.array(new_vertices)
    elements_compact = [np.array([old_to_merged[vi] for vi in el]) for el in elements_compact]

    # Remove degenerate elements (< 3 unique vertices or zero area)
    good_elements = []
    for el in elements_compact:
        unique_verts = np.unique(el)
        if len(unique_verts) < 3:
            continue
        verts_el = vertices[unique_verts]
        ac = verts_el[:, 0] * np.roll(verts_el[:, 1], -1) - np.roll(verts_el[:, 0], -1) * verts_el[:, 1]
        if abs(ac.sum()) > 1e-14:
            good_elements.append(unique_verts)
    elements_compact = good_elements

    # Boundary nodes
    tol = 1e-8
    boundary = np.where(
        (vertices[:, 0] < tol) | (vertices[:, 0] > Lx - tol) |
        (vertices[:, 1] < tol) | (vertices[:, 1] > Ly - tol)
    )[0]

    return vertices, elements_compact, boundary


def generate_triangle_mesh(n_per_side):
    """Generate a structured triangular mesh on [0,1]^2 for FEM comparison."""
    x = np.linspace(0, 1, n_per_side + 1)
    y = np.linspace(0, 1, n_per_side + 1)
    xx, yy = np.meshgrid(x, y)
    vertices = np.column_stack([xx.ravel(), yy.ravel()])

    elements = []
    for j in range(n_per_side):
        for i in range(n_per_side):
            n0 = j * (n_per_side + 1) + i
            n1 = n0 + 1
            n2 = n0 + (n_per_side + 1)
            n3 = n2 + 1
            elements.append(np.array([n0, n1, n3]))
            elements.append(np.array([n0, n3, n2]))

    tol = 1e-8
    boundary = np.where(
        (vertices[:, 0] < tol) | (vertices[:, 0] > 1 - tol) |
        (vertices[:, 1] < tol) | (vertices[:, 1] > 1 - tol)
    )[0]

    return vertices, elements, boundary


# ── Error Computation ────────────────────────────────────────────────────

def compute_errors(vertices, elements, u):
    """Compute L2 and H1 errors against manufactured solution."""
    ux_h = u[0::2]
    uy_h = u[1::2]

    ux_exact, uy_exact = manufactured_solution(vertices[:, 0], vertices[:, 1])

    # L2 error (node-based, area-weighted)
    l2_err_sq = 0.0
    h1_err_sq = 0.0
    total_area = 0.0

    for el in elements:
        el_int = np.array(el, dtype=int)
        verts = vertices[el_int]
        n_v = len(el_int)

        # Polygon area
        area_comp = verts[:, 0] * np.roll(verts[:, 1], -1) - np.roll(verts[:, 0], -1) * verts[:, 1]
        area = 0.5 * abs(area_comp.sum())
        if area < 1e-15:
            continue
        total_area += area

        # L2: average nodal error^2 * area
        err_x = ux_h[el_int] - ux_exact[el_int]
        err_y = uy_h[el_int] - uy_exact[el_int]
        l2_err_sq += area * np.mean(err_x**2 + err_y**2)

        # H1 seminorm: gradient error (approximate via polygon)
        # Use centroidal gradient approximation
        h = max(np.linalg.norm(verts[i] - verts[j])
                for i in range(n_v) for j in range(i + 1, n_v))

        # Gradient of u_h (least-squares on element nodes)
        if n_v >= 3:
            centroid = verts.mean(axis=0)
            dx = verts[:, 0] - centroid[0]
            dy = verts[:, 1] - centroid[1]
            A = np.column_stack([np.ones(n_v), dx, dy])
            if np.linalg.matrix_rank(A) >= 3:
                # Fit linear: ux = a + b*dx + c*dy
                coeff_x, _, _, _ = np.linalg.lstsq(A, ux_h[el_int], rcond=None)
                coeff_y, _, _, _ = np.linalg.lstsq(A, uy_h[el_int], rcond=None)
                grad_ux_h = np.array([coeff_x[1], coeff_x[2]])
                grad_uy_h = np.array([coeff_y[1], coeff_y[2]])

                # Exact gradient at centroid
                xc, yc = centroid
                pi = np.pi
                grad_ux_exact = np.array([
                    pi * np.cos(pi * xc) * np.sin(pi * yc),
                    pi * np.sin(pi * xc) * np.cos(pi * yc)
                ])
                grad_uy_exact = np.array([
                    -pi * np.sin(pi * xc) * np.cos(pi * yc),
                    -pi * np.cos(pi * xc) * np.sin(pi * yc)
                ])

                h1_err_sq += area * (
                    np.sum((grad_ux_h - grad_ux_exact)**2) +
                    np.sum((grad_uy_h - grad_uy_exact)**2)
                )

    l2_err = np.sqrt(l2_err_sq)
    h1_err = np.sqrt(h1_err_sq)
    return l2_err, h1_err


# ── Convergence Study ────────────────────────────────────────────────────

def convergence_study_vem(seed_counts=None, E=1000.0, nu=0.3):
    """Run h-convergence study on Voronoi meshes."""
    if seed_counts is None:
        seed_counts = [4, 6, 8, 12, 16, 24]

    results = []
    for n_per_side in seed_counts:
        vertices, elements, boundary = generate_voronoi_mesh(n_per_side, seed=42)
        n_cells = len(elements)
        h = 1.0 / n_per_side  # characteristic mesh size

        # Exact BCs
        ux_exact, uy_exact = manufactured_solution(vertices[:, 0], vertices[:, 1])
        bc_dofs = np.concatenate([2 * boundary, 2 * boundary + 1])
        bc_vals = np.concatenate([ux_exact[boundary], uy_exact[boundary]])

        # Body force: per-element centroidal integration, distributed to vertices
        n_verts = len(vertices)
        F_body = np.zeros(2 * n_verts)
        for el in elements:
            el_int = np.array(el, dtype=int)
            verts_el = vertices[el_int]
            n_v = len(el_int)
            ac = verts_el[:, 0] * np.roll(verts_el[:, 1], -1) - np.roll(verts_el[:, 0], -1) * verts_el[:, 1]
            area = 0.5 * abs(ac.sum())
            centroid = verts_el.mean(axis=0)
            # Evaluate body force at centroid
            fx_c, fy_c = manufactured_body_force(
                centroid[0], centroid[1], E, nu)
            # Distribute equally to element nodes
            for vi in el_int:
                F_body[2 * vi] += fx_c * area / n_v
                F_body[2 * vi + 1] += fy_c * area / n_v

        load_dofs = np.arange(2 * n_verts)
        load_vals = F_body

        t0 = time.time()
        u = vem_elasticity(vertices, elements, E, nu, bc_dofs, bc_vals,
                           load_dofs, load_vals)
        dt = time.time() - t0

        l2_err, h1_err = compute_errors(vertices, elements, u)
        results.append({
            'n_per_side': n_per_side, 'n_cells': n_cells,
            'n_nodes': len(vertices), 'h': h,
            'l2_err': l2_err, 'h1_err': h1_err,
            'time': dt, 'method': 'VEM-Voronoi'
        })
        print(f"  VEM n={n_per_side:4d}: {n_cells:4d} cells, h={h:.4f}, "
              f"L2={l2_err:.2e}, H1={h1_err:.2e}, t={dt:.2f}s")

    return results


def convergence_study_fem(n_per_sides=None, E=1000.0, nu=0.3):
    """Run h-convergence study on triangular FEM meshes (for comparison)."""
    if n_per_sides is None:
        n_per_sides = [4, 6, 8, 12, 16, 24]

    results = []
    for nps in n_per_sides:
        vertices, elements, boundary = generate_triangle_mesh(nps)
        n_cells = len(elements)
        h = 1.0 / nps

        # Exact BCs
        ux_exact, uy_exact = manufactured_solution(vertices[:, 0], vertices[:, 1])
        bc_dofs = np.concatenate([2 * boundary, 2 * boundary + 1])
        bc_vals = np.concatenate([ux_exact[boundary], uy_exact[boundary]])

        # Body force
        fx, fy = manufactured_body_force(vertices[:, 0], vertices[:, 1], E, nu)
        load_dofs_x = np.arange(0, 2 * len(vertices), 2)
        load_dofs_y = np.arange(1, 2 * len(vertices), 2)
        load_dofs = np.concatenate([load_dofs_x, load_dofs_y])

        nodal_area = np.zeros(len(vertices))
        for el in elements:
            el_int = np.array(el, dtype=int)
            verts = vertices[el_int]
            area = 0.5 * abs(
                (verts[1, 0] - verts[0, 0]) * (verts[2, 1] - verts[0, 1]) -
                (verts[2, 0] - verts[0, 0]) * (verts[1, 1] - verts[0, 1])
            )
            for vi in el_int:
                nodal_area[vi] += area / 3

        load_vals = np.concatenate([fx * nodal_area, fy * nodal_area])

        t0 = time.time()
        # Use VEM solver on triangles (VEM reduces to linear FEM on triangles)
        u = vem_elasticity(vertices, elements, E, nu, bc_dofs, bc_vals,
                           load_dofs, load_vals)
        dt = time.time() - t0

        l2_err, h1_err = compute_errors(vertices, elements, u)
        results.append({
            'n_per_side': nps, 'n_cells': n_cells,
            'n_nodes': len(vertices), 'h': h,
            'l2_err': l2_err, 'h1_err': h1_err,
            'time': dt, 'method': 'FEM-Triangle'
        })
        print(f"  FEM n={nps:4d}: {n_cells:4d} cells, h={h:.4f}, "
              f"L2={l2_err:.2e}, H1={h1_err:.2e}, t={dt:.2f}s")

    return results


def convergence_study_mat_meshes(E=1000.0, nu=0.3):
    """Test on .mat Voronoi meshes (reference quality meshes)."""
    from vem_elasticity import load_mesh
    mesh_dir = os.path.join(os.path.dirname(__file__), 'meshes')

    results = []
    for mname in ['voronoi.mat', 'squares.mat', 'smoothed-voronoi.mat']:
        path = os.path.join(mesh_dir, mname)
        if not os.path.exists(path):
            continue
        vertices, elements, boundary = load_mesh(path)
        n_cells = len(elements)
        h = 1.0 / np.sqrt(n_cells)

        ux_exact, uy_exact = manufactured_solution(vertices[:, 0], vertices[:, 1])
        bc_dofs = np.concatenate([2 * boundary, 2 * boundary + 1])
        bc_vals = np.concatenate([ux_exact[boundary], uy_exact[boundary]])

        # Body force
        n_verts = len(vertices)
        F_body = np.zeros(2 * n_verts)
        for el in elements:
            el_int = el.astype(int)
            verts_el = vertices[el_int]
            n_v = len(el_int)
            ac = verts_el[:, 0] * np.roll(verts_el[:, 1], -1) - np.roll(verts_el[:, 0], -1) * verts_el[:, 1]
            area = 0.5 * abs(ac.sum())
            centroid = verts_el.mean(axis=0)
            fx_c, fy_c = manufactured_body_force(centroid[0], centroid[1], E, nu)
            for vi in el_int:
                F_body[2 * vi] += fx_c * area / n_v
                F_body[2 * vi + 1] += fy_c * area / n_v

        t0 = time.time()
        u = vem_elasticity(vertices, elements, E, nu, bc_dofs, bc_vals,
                           np.arange(2 * n_verts), F_body)
        dt = time.time() - t0

        l2_err, h1_err = compute_errors(vertices, elements, u)
        label = mname.replace('.mat', '')
        results.append({
            'n_cells': n_cells, 'n_nodes': n_verts, 'h': h,
            'l2_err': l2_err, 'h1_err': h1_err,
            'time': dt, 'method': f'VEM-{label}'
        })
        print(f"  {label:20s}: {n_cells:4d} cells, h={h:.4f}, "
              f"L2={l2_err:.2e}, H1={h1_err:.2e}")

    return results


def convergence_study_fem_quad(n_per_sides=None, E=1000.0, nu=0.3):
    """Run h-convergence on quad meshes (each quad = one VEM element)."""
    if n_per_sides is None:
        n_per_sides = [4, 6, 8, 12, 16, 24]

    results = []
    for nps in n_per_sides:
        x = np.linspace(0, 1, nps + 1)
        y = np.linspace(0, 1, nps + 1)
        xx, yy = np.meshgrid(x, y)
        vertices = np.column_stack([xx.ravel(), yy.ravel()])

        elements = []
        for j in range(nps):
            for i in range(nps):
                n0 = j * (nps + 1) + i
                n1 = n0 + 1
                n2 = n0 + (nps + 1) + 1
                n3 = n0 + (nps + 1)
                elements.append(np.array([n0, n1, n2, n3]))

        tol = 1e-8
        boundary = np.where(
            (vertices[:, 0] < tol) | (vertices[:, 0] > 1 - tol) |
            (vertices[:, 1] < tol) | (vertices[:, 1] > 1 - tol)
        )[0]

        n_cells = len(elements)
        h = 1.0 / nps

        ux_exact, uy_exact = manufactured_solution(vertices[:, 0], vertices[:, 1])
        bc_dofs = np.concatenate([2 * boundary, 2 * boundary + 1])
        bc_vals = np.concatenate([ux_exact[boundary], uy_exact[boundary]])

        fx, fy = manufactured_body_force(vertices[:, 0], vertices[:, 1], E, nu)
        load_dofs = np.concatenate([
            np.arange(0, 2 * len(vertices), 2),
            np.arange(1, 2 * len(vertices), 2)
        ])

        nodal_area = np.zeros(len(vertices))
        for el in elements:
            el_int = np.array(el, dtype=int)
            area = h * h
            for vi in el_int:
                nodal_area[vi] += area / 4

        load_vals = np.concatenate([fx * nodal_area, fy * nodal_area])

        t0 = time.time()
        u = vem_elasticity(vertices, elements, E, nu, bc_dofs, bc_vals,
                           load_dofs, load_vals)
        dt = time.time() - t0

        l2_err, h1_err = compute_errors(vertices, elements, u)
        results.append({
            'n_per_side': nps, 'n_cells': n_cells,
            'n_nodes': len(vertices), 'h': h,
            'l2_err': l2_err, 'h1_err': h1_err,
            'time': dt, 'method': 'VEM-Quad'
        })
        print(f"  Quad n={nps:4d}: {n_cells:4d} cells, h={h:.4f}, "
              f"L2={l2_err:.2e}, H1={h1_err:.2e}, t={dt:.2f}s")

    return results


# ── Plotting ─────────────────────────────────────────────────────────────

def plot_convergence(all_results, save_path=None):
    """Plot convergence comparison: VEM (Voronoi/Quad) vs FEM (Triangle)."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    markers = {'VEM-Voronoi': 'o', 'FEM-Triangle': 's', 'VEM-Quad': 'D'}
    colors = {'VEM-Voronoi': '#2196F3', 'FEM-Triangle': '#F44336', 'VEM-Quad': '#4CAF50'}

    for method in ['VEM-Voronoi', 'FEM-Triangle', 'VEM-Quad']:
        res = [r for r in all_results if r['method'] == method]
        if not res:
            continue
        h = np.array([r['h'] for r in res])
        l2 = np.array([r['l2_err'] for r in res])
        h1 = np.array([r['h1_err'] for r in res])
        times = np.array([r['time'] for r in res])
        n_dofs = np.array([2 * r['n_nodes'] for r in res])

        # Convergence rates
        if len(h) >= 2:
            l2_rate = np.polyfit(np.log(h), np.log(l2), 1)[0]
            h1_rate = np.polyfit(np.log(h), np.log(h1), 1)[0]
        else:
            l2_rate = h1_rate = 0

        # L2 error
        axes[0].loglog(h, l2, f'-{markers[method]}', color=colors[method],
                       label=f'{method} (rate={l2_rate:.2f})', linewidth=2, markersize=8)

        # H1 error
        axes[1].loglog(h, h1, f'-{markers[method]}', color=colors[method],
                       label=f'{method} (rate={h1_rate:.2f})', linewidth=2, markersize=8)

        # Time vs DOFs
        axes[2].loglog(n_dofs, times, f'-{markers[method]}', color=colors[method],
                       label=method, linewidth=2, markersize=8)

    # Reference slopes
    h_ref = np.array([0.03, 0.3])
    axes[0].loglog(h_ref, 0.5 * h_ref**2, 'k--', alpha=0.3, label='O(h²)')
    axes[1].loglog(h_ref, 2.0 * h_ref**1, 'k--', alpha=0.3, label='O(h)')

    axes[0].set_xlabel('h (mesh size)')
    axes[0].set_ylabel('L² error')
    axes[0].set_title('L² Convergence')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('h (mesh size)')
    axes[1].set_ylabel('H¹ seminorm error')
    axes[1].set_title('H¹ Convergence')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].set_xlabel('DOFs')
    axes[2].set_ylabel('Time [s]')
    axes[2].set_title('Computational Cost')
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle('VEM vs FEM: h-Convergence with Manufactured Solution',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved: {save_path}")
    plt.close()


def plot_mesh_comparison(save_path=None):
    """Visualize the three mesh types used in the convergence study."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Voronoi
    verts_v, elems_v, _ = generate_voronoi_mesh(8, seed=42)
    ax = axes[0]
    for el in elems_v:
        poly = verts_v[el]
        poly_closed = np.vstack([poly, poly[0]])
        ax.plot(poly_closed[:, 0], poly_closed[:, 1], 'b-', linewidth=0.5)
    ax.set_title(f'VEM-Voronoi ({len(elems_v)} cells)')
    ax.set_aspect('equal')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    # Triangles
    verts_t, elems_t, _ = generate_triangle_mesh(8)
    ax = axes[1]
    for el in elems_t:
        poly = verts_t[el]
        poly_closed = np.vstack([poly, poly[0]])
        ax.plot(poly_closed[:, 0], poly_closed[:, 1], 'r-', linewidth=0.5)
    ax.set_title(f'FEM-Triangle ({len(elems_t)} cells)')
    ax.set_aspect('equal')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    # Quads
    n = 8
    ax = axes[2]
    for j in range(n):
        for i in range(n):
            x0, y0 = i / n, j / n
            rect = plt.Rectangle((x0, y0), 1/n, 1/n,
                                fill=False, edgecolor='green', linewidth=0.5)
            ax.add_patch(rect)
    ax.set_title(f'VEM-Quad ({n*n} cells)')
    ax.set_aspect('equal')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    fig.suptitle('Mesh Types for Convergence Study', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    out_dir = os.path.join(os.path.dirname(__file__), 'results', 'convergence')
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("h-Convergence Study: VEM vs FEM")
    print("Manufactured solution: u = (sin(pi*x)sin(pi*y), cos(pi*x)cos(pi*y))")
    print("=" * 60)

    print("\n--- VEM on quadrilateral meshes ---")
    quad_results = convergence_study_fem_quad([4, 6, 8, 12, 16, 24])

    print("\n--- FEM on triangular meshes ---")
    fem_results = convergence_study_fem([4, 6, 8, 12, 16, 24])

    print("\n--- VEM on .mat Voronoi mesh (single point) ---")
    mat_results = convergence_study_mat_meshes(1000.0, 0.3)

    all_results = quad_results + fem_results + mat_results

    # Convergence rates
    print("\n" + "=" * 60)
    print("CONVERGENCE RATES")
    print("=" * 60)
    for method in ['VEM-Voronoi', 'FEM-Triangle', 'VEM-Quad']:
        res = [r for r in all_results if r['method'] == method]
        if len(res) >= 2:
            h = np.array([r['h'] for r in res])
            l2 = np.array([r['l2_err'] for r in res])
            h1 = np.array([r['h1_err'] for r in res])
            l2_rate = np.polyfit(np.log(h), np.log(l2), 1)[0]
            h1_rate = np.polyfit(np.log(h), np.log(h1), 1)[0]
            print(f"  {method:15s}: L2 rate = {l2_rate:.2f} (expected 2.0), "
                  f"H1 rate = {h1_rate:.2f} (expected 1.0)")

    plot_convergence(all_results, os.path.join(out_dir, 'vem_vs_fem_convergence.png'))
    plot_mesh_comparison(os.path.join(out_dir, 'mesh_comparison.png'))

    print("\nDone.")
