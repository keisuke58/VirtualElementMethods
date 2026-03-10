"""
A posteriori error estimator and h-adaptive VEM.

Implements:
  1. Residual-based a posteriori error estimator (ZZ-type)
  2. L²/H¹ error norm computation
  3. h-adaptive mesh refinement for 2D VEM
  4. Mesh quality metrics

References:
  - Beirão da Veiga et al. (2015) "A posteriori error estimation for VEM"
  - Cangiani et al. (2017) "A posteriori error estimates for the VEM"
  - Zienkiewicz-Zhu (ZZ) error estimator adapted for VEM
"""

import numpy as np
from scipy.spatial import Voronoi
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection

from vem_elasticity import vem_elasticity


# ── Error Norms ──────────────────────────────────────────────────────────

def l2_error(u_h, u_exact, vertices, elements):
    """
    Compute L² error norm: ||u_h - u_exact||_{L²} ≈ sqrt(Σ_E |E|·|u_h - u_ex|²_avg).

    For vector fields (2 DOFs/node): sum over both components.
    """
    err_sq = 0.0
    n_nodes = len(vertices)
    is_vector = len(u_h) == 2 * n_nodes

    for el in elements:
        el_int = el.astype(int)
        verts = vertices[el_int]
        n_v = len(el_int)

        # Element area (shoelace formula)
        area_comp = (verts[:, 0] * np.roll(verts[:, 1], -1)
                     - np.roll(verts[:, 0], -1) * verts[:, 1])
        area = 0.5 * abs(np.sum(area_comp))

        if is_vector:
            for d in range(2):
                dof_ids = 2 * el_int + d
                diff = u_h[dof_ids] - u_exact[dof_ids]
                err_sq += area * np.mean(diff ** 2)
        else:
            diff = u_h[el_int] - u_exact[el_int]
            err_sq += area * np.mean(diff ** 2)

    return np.sqrt(err_sq)


def h1_seminorm_error(u_h, u_exact, vertices, elements):
    """
    Approximate H¹ semi-norm: |u_h - u_exact|_{H¹} via finite differences.

    Uses element-wise gradient approximation:
      ∇u ≈ (1/|E|) Σ_{edges} u_mid · n_edge · |edge|
    """
    err_sq = 0.0
    n_nodes = len(vertices)
    is_vector = len(u_h) == 2 * n_nodes

    for el in elements:
        el_int = el.astype(int)
        verts = vertices[el_int]
        n_v = len(el_int)

        area_comp = (verts[:, 0] * np.roll(verts[:, 1], -1)
                     - np.roll(verts[:, 0], -1) * verts[:, 1])
        area = 0.5 * abs(np.sum(area_comp))
        if area < 1e-20:
            continue

        if is_vector:
            for d in range(2):
                dof_ids = 2 * el_int + d
                diff = u_h[dof_ids] - u_exact[dof_ids]
                # Approximate gradient via edge contributions
                grad = np.zeros(2)
                for i in range(n_v):
                    j = (i + 1) % n_v
                    edge_mid_val = 0.5 * (diff[i] + diff[j])
                    # Outward normal of edge
                    dx = verts[j, 0] - verts[i, 0]
                    dy = verts[j, 1] - verts[i, 1]
                    normal = np.array([dy, -dx])
                    grad += edge_mid_val * normal
                grad /= area
                err_sq += area * np.dot(grad, grad)
        else:
            diff = u_h[el_int] - u_exact[el_int]
            grad = np.zeros(2)
            for i in range(n_v):
                j = (i + 1) % n_v
                edge_mid_val = 0.5 * (diff[i] + diff[j])
                dx = verts[j, 0] - verts[i, 0]
                dy = verts[j, 1] - verts[i, 1]
                normal = np.array([dy, -dx])
                grad += edge_mid_val * normal
            grad /= area
            err_sq += area * np.dot(grad, grad)

    return np.sqrt(err_sq)


# ── A Posteriori Error Estimator ──────────────────────────────────────────

def estimate_element_error(u, vertices, elements, E_field, nu):
    """
    Residual-based a posteriori error indicator per element.

    For each element E:
      η_E² = h_E² · ||r||² + h_E · ||j||²
    where:
      r = interior residual (≈ 0 for equilibrium)
      j = jump in traction across edges (inter-element stress discontinuity)

    Simplified estimator: use stress recovery (ZZ-type).
    Compute element-wise stress, then estimate jumps at edges.
    """
    n_nodes = len(vertices)
    n_el = len(elements)

    # Compute element-wise stress from displacement gradient
    stress_per_el = np.zeros((n_el, 3))  # [σ_xx, σ_yy, σ_xy] Voigt

    for i, el in enumerate(elements):
        el_int = el.astype(int)
        verts = vertices[el_int]
        n_v = len(el_int)

        area_comp = (verts[:, 0] * np.roll(verts[:, 1], -1)
                     - np.roll(verts[:, 0], -1) * verts[:, 1])
        area = 0.5 * abs(np.sum(area_comp))
        if area < 1e-20:
            continue

        E_el = E_field[i] if hasattr(E_field, '__len__') else E_field
        C = (E_el / (1 - nu**2)) * np.array([
            [1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]
        ])

        # Approximate strain from boundary integral
        grad_ux = np.zeros(2)
        grad_uy = np.zeros(2)
        for k in range(n_v):
            j = (k + 1) % n_v
            dx = verts[j, 0] - verts[k, 0]
            dy = verts[j, 1] - verts[k, 1]
            normal = np.array([dy, -dx])
            ux_mid = 0.5 * (u[2*el_int[k]] + u[2*el_int[j]])
            uy_mid = 0.5 * (u[2*el_int[k]+1] + u[2*el_int[j]+1])
            grad_ux += ux_mid * normal
            grad_uy += uy_mid * normal
        grad_ux /= area
        grad_uy /= area

        eps = np.array([grad_ux[0], grad_uy[1],
                        grad_ux[1] + grad_uy[0]])
        stress_per_el[i] = C @ eps

    # Build edge → elements adjacency
    edge_to_el = {}
    for i, el in enumerate(elements):
        el_int = el.astype(int)
        n_v = len(el_int)
        for k in range(n_v):
            v1, v2 = el_int[k], el_int[(k+1) % n_v]
            edge_key = (min(v1, v2), max(v1, v2))
            if edge_key not in edge_to_el:
                edge_to_el[edge_key] = []
            edge_to_el[edge_key].append(i)

    # Compute error indicator per element
    eta = np.zeros(n_el)
    for i, el in enumerate(elements):
        el_int = el.astype(int)
        verts = vertices[el_int]
        n_v = len(el_int)

        # Element diameter
        h_E = max(np.linalg.norm(verts[a] - verts[b])
                  for a in range(n_v) for b in range(a+1, n_v))

        # Sum of traction jumps over edges
        jump_sq = 0.0
        for k in range(n_v):
            v1, v2 = el_int[k], el_int[(k+1) % n_v]
            edge_key = (min(v1, v2), max(v1, v2))
            neighbors = edge_to_el.get(edge_key, [])

            if len(neighbors) == 2:
                el_a, el_b = neighbors
                stress_jump = stress_per_el[el_a] - stress_per_el[el_b]
                edge_len = np.linalg.norm(vertices[v2] - vertices[v1])
                jump_sq += edge_len * np.dot(stress_jump, stress_jump)

        eta[i] = np.sqrt(h_E * jump_sq)

    return eta


# ── Mesh Quality Metrics ─────────────────────────────────────────────────

def compute_mesh_quality(vertices, elements):
    """
    Compute mesh quality metrics for 2D polygonal mesh.

    Returns dict with:
      - aspect_ratios: per-element aspect ratio (max_edge / min_edge)
      - areas: per-element area
      - regularity: h_E / ρ_E (diameter / inradius approximation)
      - min_angle: smallest internal angle per element
      - summary: aggregate statistics
    """
    n_el = len(elements)
    aspect_ratios = np.zeros(n_el)
    areas = np.zeros(n_el)
    regularity = np.zeros(n_el)
    min_angles = np.zeros(n_el)

    for i, el in enumerate(elements):
        el_int = el.astype(int)
        verts = vertices[el_int]
        n_v = len(el_int)

        # Area
        area_comp = (verts[:, 0] * np.roll(verts[:, 1], -1)
                     - np.roll(verts[:, 0], -1) * verts[:, 1])
        areas[i] = 0.5 * abs(np.sum(area_comp))

        # Edge lengths
        edges = np.array([np.linalg.norm(verts[(k+1) % n_v] - verts[k])
                          for k in range(n_v)])
        if edges.min() > 1e-15:
            aspect_ratios[i] = edges.max() / edges.min()
        else:
            aspect_ratios[i] = np.inf

        # Diameter and inradius approximation
        h_E = max(np.linalg.norm(verts[a] - verts[b])
                  for a in range(n_v) for b in range(a+1, n_v))
        perimeter = edges.sum()
        rho_E = 2 * areas[i] / perimeter if perimeter > 0 else 1e-15
        regularity[i] = h_E / rho_E if rho_E > 1e-15 else np.inf

        # Minimum interior angle
        angles = []
        for k in range(n_v):
            p = verts[(k-1) % n_v]
            c = verts[k]
            n = verts[(k+1) % n_v]
            v1 = p - c
            v2 = n - c
            cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-15)
            angles.append(np.arccos(np.clip(cos_a, -1, 1)))
        min_angles[i] = np.min(angles) if angles else 0

    summary = {
        'n_elements': n_el,
        'n_vertices': len(vertices),
        'area_min': areas.min(),
        'area_max': areas.max(),
        'area_mean': areas.mean(),
        'aspect_ratio_min': aspect_ratios[np.isfinite(aspect_ratios)].min()
            if np.any(np.isfinite(aspect_ratios)) else np.inf,
        'aspect_ratio_max': aspect_ratios[np.isfinite(aspect_ratios)].max()
            if np.any(np.isfinite(aspect_ratios)) else np.inf,
        'aspect_ratio_mean': aspect_ratios[np.isfinite(aspect_ratios)].mean()
            if np.any(np.isfinite(aspect_ratios)) else np.inf,
        'regularity_max': regularity[np.isfinite(regularity)].max()
            if np.any(np.isfinite(regularity)) else np.inf,
        'min_angle_deg': np.degrees(min_angles.min()),
    }

    return {
        'aspect_ratios': aspect_ratios,
        'areas': areas,
        'regularity': regularity,
        'min_angles': min_angles,
        'summary': summary,
    }


# ── h-Adaptive VEM ───────────────────────────────────────────────────────

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
    return np.array(new_verts), np.array([old_to_new[i] for i in range(n)])


def refine_mesh_adaptive(vertices, elements, eta, theta=0.3,
                          domain=(0, 1, 0, 1)):
    """
    h-adaptive refinement: mark elements where η > θ·max(η),
    add new Voronoi seeds at marked element centroids,
    rebuild mesh.

    Parameters
    ----------
    vertices, elements : current mesh
    eta : error indicator per element
    theta : marking fraction (Dörfler marking)
    domain : (xmin, xmax, ymin, ymax)

    Returns
    -------
    new_vertices, new_elements, new_boundary : refined mesh
    marked : indices of marked elements
    """
    xmin, xmax, ymin, ymax = domain

    # Current seeds ≈ element centroids
    seeds = []
    for el in elements:
        el_int = el.astype(int)
        seeds.append(vertices[el_int].mean(axis=0))
    seeds = np.array(seeds)

    # Mark elements with largest error
    threshold = theta * eta.max()
    marked = np.where(eta > threshold)[0]

    # Add new seeds at marked element centroids (bisection)
    new_seeds = list(seeds)
    for idx in marked:
        el_int = elements[idx].astype(int)
        verts = vertices[el_int]
        centroid = verts.mean(axis=0)
        n_v = len(el_int)
        # Add seeds at edge midpoints of marked element
        for k in range(n_v):
            mid = 0.5 * (verts[k] + verts[(k+1) % n_v])
            new_seeds.append(mid)

    new_seeds = np.array(new_seeds)

    # Clip to domain
    new_seeds[:, 0] = np.clip(new_seeds[:, 0], xmin + 0.01, xmax - 0.01)
    new_seeds[:, 1] = np.clip(new_seeds[:, 1], ymin + 0.01, ymax - 0.01)

    # Remove duplicates
    unique = [new_seeds[0]]
    for s in new_seeds[1:]:
        if all(np.linalg.norm(s - u) > 0.005 for u in unique):
            unique.append(s)
    new_seeds = np.array(unique)

    # Build new Voronoi mesh
    Lx, Ly = xmax - xmin, ymax - ymin
    all_pts = [new_seeds]
    for axis, vals in [(0, [xmin, xmax]), (1, [ymin, ymax])]:
        for v in vals:
            mirror = new_seeds.copy()
            mirror[:, axis] = 2 * v - mirror[:, axis]
            all_pts.append(mirror)
    all_pts = np.vstack(all_pts)

    n_orig = len(new_seeds)
    vor = Voronoi(all_pts)
    raw_verts = vor.vertices.copy()
    raw_verts[:, 0] = np.clip(raw_verts[:, 0], xmin - 0.001, xmax + 0.001)
    raw_verts[:, 1] = np.clip(raw_verts[:, 1], ymin - 0.001, ymax + 0.001)

    unique_verts, remap = _merge_verts(raw_verts, tol=1e-8)

    new_elements = []
    for cell_idx in range(n_orig):
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
        if (xmin - 0.1 <= cell_c[0] <= xmax + 0.1 and
                ymin - 0.1 <= cell_c[1] <= ymax + 0.1):
            new_elements.append(face)

    tol_bnd = 0.02
    bnd = np.where(
        (unique_verts[:, 0] < xmin + tol_bnd) |
        (unique_verts[:, 0] > xmax - tol_bnd) |
        (unique_verts[:, 1] < ymin + tol_bnd) |
        (unique_verts[:, 1] > ymax - tol_bnd)
    )[0]

    return unique_verts, new_elements, bnd, marked


def adaptive_vem_solve(E_field_func, nu, n_refine=3, theta=0.3,
                       n_initial_seeds=20, domain=(0, 1, 0, 1), seed=42):
    """
    Adaptive VEM loop:
      1. Solve on current mesh
      2. Estimate error
      3. Mark & refine
      4. Repeat

    Parameters
    ----------
    E_field_func : callable(centroid_x, centroid_y) → E
    nu : Poisson's ratio
    n_refine : number of adaptive refinement steps
    theta : marking fraction
    n_initial_seeds : initial mesh density
    domain : (xmin, xmax, ymin, ymax)

    Returns
    -------
    results : list of dicts with mesh, solution, error info per level
    """
    xmin, xmax, ymin, ymax = domain
    rng = np.random.default_rng(seed)

    # Initial mesh
    seeds = np.column_stack([
        rng.uniform(xmin + 0.05, xmax - 0.05, n_initial_seeds),
        rng.uniform(ymin + 0.05, ymax - 0.05, n_initial_seeds),
    ])

    all_pts = [seeds]
    for axis, vals in [(0, [xmin, xmax]), (1, [ymin, ymax])]:
        for v in vals:
            mirror = seeds.copy()
            mirror[:, axis] = 2 * v - mirror[:, axis]
            all_pts.append(mirror)
    all_pts = np.vstack(all_pts)

    n_orig = len(seeds)
    vor = Voronoi(all_pts)
    raw_verts = vor.vertices.copy()
    raw_verts[:, 0] = np.clip(raw_verts[:, 0], xmin - 0.001, xmax + 0.001)
    raw_verts[:, 1] = np.clip(raw_verts[:, 1], ymin - 0.001, ymax + 0.001)
    unique_verts, remap = _merge_verts(raw_verts, tol=1e-8)

    elements = []
    for cell_idx in range(n_orig):
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
        if (xmin - 0.1 <= cell_c[0] <= xmax + 0.1 and
                ymin - 0.1 <= cell_c[1] <= ymax + 0.1):
            elements.append(face)

    tol_bnd = 0.02
    boundary = np.where(
        (unique_verts[:, 0] < xmin + tol_bnd) |
        (unique_verts[:, 0] > xmax - tol_bnd) |
        (unique_verts[:, 1] < ymin + tol_bnd) |
        (unique_verts[:, 1] > ymax - tol_bnd)
    )[0]

    vertices = unique_verts
    results = []

    for level in range(n_refine + 1):
        # Re-index to used nodes
        used_set = set()
        for el in elements:
            used_set.update(el.astype(int).tolist())
        used = np.array(sorted(used_set))
        n_used = len(used)
        old_to_new = {int(g): i for i, g in enumerate(used)}

        compact_verts = vertices[used]
        compact_elems = []
        for el in elements:
            compact_elems.append(np.array([old_to_new[int(v)] for v in el]))

        # E field
        E_per_el = np.array([
            E_field_func(*compact_verts[el.astype(int)].mean(axis=0))
            for el in compact_elems
        ])

        # BCs: fix bottom, load top
        tol = 0.02
        bottom = np.where(compact_verts[:, 1] < ymin + tol)[0]
        bc_dofs = np.concatenate([2*bottom, 2*bottom+1])
        bc_vals = np.zeros(len(bc_dofs))

        top = np.where(compact_verts[:, 1] > ymax - tol)[0]
        load_dofs = 2 * top + 1
        load_vals = np.full(len(top), -0.5 / max(len(top), 1))

        # Solve
        try:
            u = vem_elasticity(compact_verts, compact_elems, E_per_el, nu,
                               bc_dofs, bc_vals, load_dofs, load_vals)
        except np.linalg.LinAlgError:
            u = np.zeros(2 * n_used)

        # Error estimate
        eta = estimate_element_error(u, compact_verts, compact_elems,
                                     E_per_el, nu)

        # Mesh quality
        quality = compute_mesh_quality(compact_verts, compact_elems)

        results.append({
            'level': level,
            'vertices': compact_verts.copy(),
            'elements': [el.copy() for el in compact_elems],
            'u': u.copy(),
            'eta': eta.copy(),
            'eta_total': np.sqrt(np.sum(eta**2)),
            'n_cells': len(compact_elems),
            'n_nodes': n_used,
            'quality': quality,
        })

        if level < n_refine:
            # Map back to original indices for refinement
            new_to_old = {i: g for g, i in old_to_new.items()}
            full_elements = []
            for el in compact_elems:
                full_elements.append(
                    np.array([new_to_old[int(v)] for v in el]))

            vertices, elements, boundary, marked = refine_mesh_adaptive(
                vertices, full_elements, eta, theta=theta, domain=domain)

    return results


# ── Convergence Study with L²/H¹ Norms ───────────────────────────────────

def convergence_study_2d(mesh_dir, save_dir='/tmp'):
    """
    Systematic convergence study for 2D VEM elasticity
    with L² and H¹ error norms.
    """
    import scipy.io
    import os

    print("=" * 60)
    print("2D VEM Convergence Study (L² and H¹ norms)")
    print("=" * 60)

    mesh_names = ['squares.mat', 'triangles.mat', 'voronoi.mat',
                  'smoothed-voronoi.mat']

    E_mod, nu = 1000.0, 0.3

    all_results = {}
    for name in mesh_names:
        path = os.path.join(mesh_dir, name)
        if not os.path.exists(path):
            continue

        mesh = scipy.io.loadmat(path)
        vertices = mesh['vertices']
        elements = np.array(
            [i[0].reshape(-1) - 1 for i in mesh['elements']], dtype=object)
        boundary = mesh['boundary'].T[0] - 1

        # Exact: uniform tension σ_xx = 1
        exact_ux = vertices[:, 0] / E_mod
        exact_uy = -nu * vertices[:, 1] / E_mod
        exact = np.zeros(2 * len(vertices))
        exact[0::2] = exact_ux
        exact[1::2] = exact_uy

        bc_dofs = np.concatenate([2*boundary, 2*boundary+1])
        bc_vals = np.concatenate([exact_ux[boundary], exact_uy[boundary]])

        u = vem_elasticity(vertices, elements, E_mod, nu, bc_dofs, bc_vals)

        l2_err = l2_error(u, exact, vertices, elements)
        h1_err = h1_seminorm_error(u, exact, vertices, elements)
        linf_err = np.max(np.abs(u - exact))

        quality = compute_mesh_quality(vertices, elements)

        result = {
            'n_nodes': len(vertices),
            'n_elements': len(elements),
            'l2_error': l2_err,
            'h1_error': h1_err,
            'linf_error': linf_err,
            'quality': quality['summary'],
        }
        all_results[name] = result
        print(f"  {name:25s}: L²={l2_err:.2e}, H¹={h1_err:.2e}, "
              f"L∞={linf_err:.2e}, AR={quality['summary']['aspect_ratio_mean']:.2f}")

    return all_results


# ── Visualization ─────────────────────────────────────────────────────────

def plot_adaptive_results(results, save=None):
    """Plot adaptive refinement history."""
    n_levels = len(results)
    fig, axes = plt.subplots(2, n_levels, figsize=(5 * n_levels, 9))
    if n_levels == 1:
        axes = axes.reshape(-1, 1)

    for col, res in enumerate(results):
        verts = res['vertices']
        elems = res['elements']
        eta = res['eta']

        # Top: mesh colored by error indicator
        ax = axes[0, col]
        patches = []
        colors = []
        for i, el in enumerate(elems):
            el_int = el.astype(int)
            patches.append(MplPolygon(verts[el_int], closed=True))
            colors.append(eta[i])
        pc = PatchCollection(patches, cmap='hot_r', edgecolor='k',
                             linewidth=0.3)
        pc.set_array(np.array(colors))
        ax.add_collection(pc)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        ax.set_title(f'Level {res["level"]}\n'
                     f'{res["n_cells"]} cells, η={res["eta_total"]:.2e}')
        fig.colorbar(pc, ax=ax, label='η_E', shrink=0.7)

        # Bottom: displacement
        u = res['u']
        ux, uy = u[0::2], u[1::2]
        ax2 = axes[1, col]
        deform_scale = 200
        deformed = verts + deform_scale * np.column_stack([ux, uy])
        patches2 = []
        colors2 = []
        for i, el in enumerate(elems):
            el_int = el.astype(int)
            patches2.append(MplPolygon(deformed[el_int], closed=True))
            u_mag = np.mean(np.sqrt(ux[el_int]**2 + uy[el_int]**2))
            colors2.append(u_mag)
        pc2 = PatchCollection(patches2, cmap='viridis', edgecolor='k',
                              linewidth=0.3)
        pc2.set_array(np.array(colors2))
        ax2.add_collection(pc2)
        ax2.set_xlim(-0.1, 1.1)
        ax2.set_ylim(-0.1, 1.1)
        ax2.set_aspect('equal')
        ax2.set_title(f'Deformed (×{deform_scale})')
        fig.colorbar(pc2, ax=ax2, label='|u|', shrink=0.7)

    fig.suptitle('h-Adaptive VEM Refinement', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save}")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import os

    save_dir = os.path.join(os.path.dirname(__file__), 'results')
    mesh_dir = os.path.join(os.path.dirname(__file__), 'meshes')
    os.makedirs(save_dir, exist_ok=True)

    # 1. Convergence study
    conv_results = convergence_study_2d(mesh_dir, save_dir)

    # 2. Adaptive refinement demo
    print("\n" + "=" * 60)
    print("Adaptive VEM Demo")
    print("=" * 60)

    def E_field(x, y):
        """Spatially varying E: soft at center, stiff at edges."""
        DI = 0.9 - 0.8 * np.sqrt((x - 0.5)**2 + (y - 0.5)**2) / (0.5 * np.sqrt(2))
        DI = np.clip(DI, 0.05, 0.95)
        return 30 + 970 * (1 - DI)**2

    results = adaptive_vem_solve(
        E_field, nu=0.3, n_refine=3, theta=0.3,
        n_initial_seeds=15, domain=(0, 1, 0, 1))

    for res in results:
        print(f"  Level {res['level']}: {res['n_cells']} cells, "
              f"η_total={res['eta_total']:.4e}")

    plot_adaptive_results(results, save=f'{save_dir}/vem_adaptive.png')

    print("\nError estimator + adaptive VEM complete!")
