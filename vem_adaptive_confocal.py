"""
Adaptive Non-conforming VEM on Confocal FISH Images.

Combines:
  1. Pixel-based E(DI) field from FISH image (vem_pixel_fish.py)
  2. Non-conforming mesh with hanging nodes (vem_exotic_meshes.py)
  3. A posteriori error estimator (vem_error_estimator.py)
  4. Adaptive refinement loop

Strategy:
  - Start with coarse uniform grid covering the biofilm
  - Solve VEM, estimate error per element
  - Refine elements with high error (stress gradients / DI gradients)
  - Non-conforming: refined cells create hanging nodes on coarse neighbors
  - Repeat until convergence or max refinement level

This gives fine resolution where it matters (pathogen-commensal interface,
stress concentrations) without the cost of full pixel resolution.
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from vem_elasticity import vem_elasticity
from process_heine_fish import decompose_fish_channels
from vem_pixel_fish import compute_pixel_di, compute_pixel_E
from vem_error_estimator import estimate_element_error


# ── Adaptive Non-conforming Mesh for Images ──────────────────────────────

def build_initial_grid(active_mask, nx, ny):
    """
    Build coarse quad grid aligned to image pixels.
    Only create elements that overlap with active (biofilm) region.

    Parameters
    ----------
    active_mask : (H, W) bool
    nx, ny : coarse grid divisions

    Returns
    -------
    vertices : (N, 2)
    elements : list of int arrays
    cell_info : list of dicts with pixel coverage info
    """
    H, W = active_mask.shape
    hx = W / nx
    hy = H / ny

    vert_list = []
    vert_map = {}

    def add_v(x, y):
        key = (round(x, 8), round(y, 8))
        if key not in vert_map:
            vert_map[key] = len(vert_list)
            vert_list.append([x, y])
        return vert_map[key]

    elements = []
    cell_info = []

    for iy in range(ny):
        for ix in range(nx):
            # Pixel range for this cell
            px0 = int(round(ix * hx))
            px1 = int(round((ix + 1) * hx))
            py0 = int(round(iy * hy))
            py1 = int(round((iy + 1) * hy))

            px1 = min(px1, W)
            py1 = min(py1, H)

            # Check if cell overlaps with biofilm
            cell_mask = active_mask[py0:py1, px0:px1]
            coverage = cell_mask.sum() / max(cell_mask.size, 1)

            if coverage < 0.1:  # skip mostly empty cells
                continue

            # Physical coords (y flipped: image top = high y)
            x0 = ix * hx
            x1 = (ix + 1) * hx
            y0 = (ny - iy - 1) * hy
            y1 = (ny - iy) * hy

            v0 = add_v(x0, y0)
            v1 = add_v(x1, y0)
            v2 = add_v(x1, y1)
            v3 = add_v(x0, y1)

            elements.append(np.array([v0, v1, v2, v3]))
            cell_info.append({
                'ix': ix, 'iy': iy,
                'px_range': (px0, px1, py0, py1),
                'coverage': coverage,
                'level': 0,
            })

    vertices = np.array(vert_list)
    return vertices, elements, cell_info


def compute_cell_E(channels, DI_map, E_map, cell_info, active_mask):
    """Compute mean E per coarse cell from pixel-level data."""
    E_per_cell = np.zeros(len(cell_info))
    DI_per_cell = np.zeros(len(cell_info))

    for i, info in enumerate(cell_info):
        px0, px1, py0, py1 = info['px_range']
        mask = active_mask[py0:py1, px0:px1]
        if mask.sum() > 0:
            E_per_cell[i] = E_map[py0:py1, px0:px1][mask].mean()
            DI_per_cell[i] = DI_map[py0:py1, px0:px1][mask].mean()
        else:
            E_per_cell[i] = 500.0  # default
            DI_per_cell[i] = 0.5

    return E_per_cell, DI_per_cell


def refine_nonconforming(vertices, elements, cell_info,
                          marked_indices, active_mask,
                          DI_map, E_map):
    """
    Refine marked elements into 2x2 sub-cells.
    Neighboring coarse elements get hanging nodes on shared edges.

    Returns new mesh with non-conforming interfaces.
    """
    H, W = active_mask.shape
    vert_list = list(vertices)
    vert_map = {}

    # Build vert_map from existing vertices
    for i, v in enumerate(vertices):
        key = (round(v[0], 8), round(v[1], 8))
        vert_map[key] = i

    def add_v(x, y):
        key = (round(x, 8), round(y, 8))
        if key not in vert_map:
            vert_map[key] = len(vert_list)
            vert_list.append([x, y])
        return vert_map[key]

    new_elements = []
    new_cell_info = []
    marked_set = set(marked_indices)

    # Track new midpoints on edges for hanging node injection
    edge_midpoints = {}  # (min_vid, max_vid) → mid_vid

    for i, (el, info) in enumerate(zip(elements, cell_info)):
        el_int = el.astype(int)

        if i in marked_set and len(el_int) >= 4:
            # Split into 2x2 using bounding box corners
            verts = np.array(vert_list)[el_int]

            # Use bounding box for robust splitting
            x0 = verts[:, 0].min()
            x1 = verts[:, 0].max()
            y0 = verts[:, 1].min()
            y1 = verts[:, 1].max()

            if x1 - x0 < 1e-10 or y1 - y0 < 1e-10:
                new_elements.append(el.copy())
                new_cell_info.append(info.copy())
                continue

            cx = 0.5 * (x0 + x1)
            cy = 0.5 * (y0 + y1)

            # Find or create corner vertices
            v_bl = add_v(x0, y0)
            v_br = add_v(x1, y0)
            v_tr = add_v(x1, y1)
            v_tl = add_v(x0, y1)

            # Edge midpoints
            v_mb = add_v(cx, y0)  # mid bottom
            v_mr = add_v(x1, cy)  # mid right
            v_mt = add_v(cx, y1)  # mid top
            v_ml = add_v(x0, cy)  # mid left
            v_cc = add_v(cx, cy)  # center

            # Register edge midpoints for hanging node injection
            for va, vb, vm in [
                (v_bl, v_br, v_mb), (v_br, v_tr, v_mr),
                (v_tr, v_tl, v_mt), (v_tl, v_bl, v_ml)
            ]:
                edge_key = (min(va, vb), max(va, vb))
                edge_midpoints[edge_key] = vm

            # 4 sub-cells
            sub_els = [
                np.array([v_bl, v_mb, v_cc, v_ml]),  # BL
                np.array([v_mb, v_br, v_mr, v_cc]),  # BR
                np.array([v_cc, v_mr, v_tr, v_mt]),  # TR
                np.array([v_ml, v_cc, v_mt, v_tl]),  # TL
            ]

            px0, px1_, py0, py1_ = info['px_range']
            pmx = (px0 + px1_) // 2
            pmy = (py0 + py1_) // 2

            sub_px = [
                (px0, pmx, pmy, py1_),   # BL (image y flipped)
                (pmx, px1_, pmy, py1_),  # BR
                (pmx, px1_, py0, pmy),   # TR
                (px0, pmx, py0, pmy),    # TL
            ]

            for sub_el, sub_p in zip(sub_els, sub_px):
                sp0, sp1, sp2, sp3 = sub_p
                mask = active_mask[sp2:sp3, sp0:sp1]
                cov = mask.sum() / max(mask.size, 1)
                if cov > 0.05:
                    new_elements.append(sub_el)
                    new_cell_info.append({
                        'ix': info['ix'], 'iy': info['iy'],
                        'px_range': sub_p,
                        'coverage': cov,
                        'level': info['level'] + 1,
                    })
        else:
            new_elements.append(el.copy())
            new_cell_info.append(info.copy())

    # Inject hanging nodes into coarse elements adjacent to refined ones
    final_elements = []
    for el in new_elements:
        el_int = el.astype(int)
        n_v = len(el_int)
        new_el = []

        for k in range(n_v):
            new_el.append(el_int[k])
            va = el_int[k]
            vb = el_int[(k + 1) % n_v]
            edge_key = (min(va, vb), max(va, vb))
            if edge_key in edge_midpoints:
                mid = edge_midpoints[edge_key]
                if mid not in el_int:  # not already in this element
                    new_el.append(mid)

        final_elements.append(np.array(new_el))

    new_vertices = np.array(vert_list)

    # Filter degenerate elements (zero area, duplicate vertices)
    good_elements = []
    good_info = []
    for i, el in enumerate(final_elements):
        el_int = el.astype(int)
        # Remove consecutive duplicate vertices
        clean = [el_int[0]]
        for v in el_int[1:]:
            if v != clean[-1]:
                clean.append(v)
        if clean[-1] == clean[0] and len(clean) > 1:
            clean = clean[:-1]
        if len(set(clean)) < 3:
            continue

        # Check area
        verts = new_vertices[clean]
        area_comp = (verts[:, 0] * np.roll(verts[:, 1], -1)
                     - np.roll(verts[:, 0], -1) * verts[:, 1])
        area = 0.5 * abs(np.sum(area_comp))
        if area < 1e-10:
            continue

        good_elements.append(np.array(clean))
        if i < len(new_cell_info):
            good_info.append(new_cell_info[i])

    # Compact: remove unused vertices
    used_ids = set()
    for el in good_elements:
        used_ids.update(el.astype(int))
    used_ids = sorted(used_ids)
    old_to_new = {old: new for new, old in enumerate(used_ids)}

    compact_verts = new_vertices[used_ids]
    compact_elems = [np.array([old_to_new[int(v)] for v in el])
                     for el in good_elements]

    return compact_verts, compact_elems, good_info


# ── Adaptive loop ────────────────────────────────────────────────────────

def adaptive_confocal_pipeline(image_path, condition_name='adaptive',
                                n_refine=3, nx_initial=16, ny_initial=16,
                                theta=0.3, E_max=1000.0, E_min=10.0,
                                n_hill=2, intensity_threshold=0.05,
                                output_dir=None):
    """
    Adaptive non-conforming VEM on FISH image.

    Parameters
    ----------
    image_path : str
    n_refine : int — number of adaptive refinement steps
    nx_initial, ny_initial : int — initial coarse grid
    theta : float — Dörfler marking fraction
    """
    print(f"\n{'='*60}")
    print(f"Adaptive Non-conforming VEM: {condition_name}")
    print(f"{'='*60}")

    # Load and process image
    img = np.array(Image.open(image_path))
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    H, W = img.shape[:2]
    print(f"  Image: {W}x{H} px")

    # Channel decomposition
    channels = decompose_fish_channels(img[:, :, :3])
    DI_map = compute_pixel_di(channels, method='weighted')
    E_map = compute_pixel_E(DI_map, E_max, E_min, n_hill)

    brightness = (img[:, :, :3].astype(float) / 255.0).mean(axis=2)
    active = brightness > intensity_threshold

    # Largest connected component
    from scipy.ndimage import label as ndlabel
    labeled, n_comp = ndlabel(active)
    if n_comp > 1:
        sizes = [(labeled == k).sum() for k in range(1, n_comp + 1)]
        active = labeled == (np.argmax(sizes) + 1)

    print(f"  Active pixels: {active.sum()} / {H*W}")
    print(f"  DI: mean={DI_map[active].mean():.3f}")
    print(f"  E:  mean={E_map[active].mean():.0f} Pa")

    # Initial coarse mesh
    vertices, elements, cell_info = build_initial_grid(
        active, nx_initial, ny_initial)
    E_per_cell, DI_per_cell = compute_cell_E(
        channels, DI_map, E_map, cell_info, active)

    nu = 0.3
    results = []

    for level in range(n_refine + 1):
        n_el = len(elements)
        n_sides = [len(el) for el in elements]
        levels = [info['level'] for info in cell_info]

        print(f"\n  --- Level {level}: {n_el} elements, "
              f"{len(vertices)} vertices, "
              f"sides={min(n_sides)}-{max(n_sides)}, "
              f"max refinement depth={max(levels)}")

        # Recompute E per cell
        E_per_cell, DI_per_cell = compute_cell_E(
            channels, DI_map, E_map, cell_info, active)

        # BCs
        y_vals = vertices[:, 1]
        y_lo = np.percentile(y_vals, 5)
        y_hi = np.percentile(y_vals, 95)

        bottom = np.where(y_vals <= y_lo)[0]
        top = np.where(y_vals >= y_hi)[0]

        if len(bottom) < 5:
            y_sorted = np.argsort(y_vals)
            n_bc = max(10, len(vertices) // 15)
            bottom = y_sorted[:n_bc]
        if len(top) < 5:
            y_sorted = np.argsort(y_vals)
            n_bc = max(10, len(vertices) // 15)
            top = y_sorted[-n_bc:]

        bc_dofs = np.concatenate([2 * bottom, 2 * bottom + 1])
        bc_vals = np.zeros(len(bc_dofs))
        load_dofs = 2 * top + 1
        load_vals = np.full(len(top), -2.0 / max(len(top), 1))

        # Solve
        try:
            u = vem_elasticity(vertices, elements, E_per_cell, nu,
                               bc_dofs, bc_vals, load_dofs, load_vals)
            ux = u[0::2]
            uy = u[1::2]
            u_mag = np.sqrt(ux**2 + uy**2)
            print(f"    Max |u|: {u_mag.max():.6f}")
        except Exception as e:
            print(f"    Solve failed: {e}")
            u = np.zeros(2 * len(vertices))
            u_mag = np.zeros(len(vertices))

        # Error estimate
        try:
            eta = estimate_element_error(u, vertices, elements,
                                          E_per_cell, nu)
            print(f"    Error: max={eta.max():.2e}, "
                  f"mean={eta.mean():.2e}, "
                  f"total={np.sqrt(np.sum(eta**2)):.2e}")
        except Exception as e:
            print(f"    Error estimation failed: {e}")
            eta = np.ones(n_el)

        results.append({
            'level': level,
            'vertices': vertices.copy(),
            'elements': [el.copy() for el in elements],
            'cell_info': [info.copy() for info in cell_info],
            'E_per_cell': E_per_cell.copy(),
            'DI_per_cell': DI_per_cell.copy(),
            'u': u.copy(),
            'eta': eta.copy(),
            'n_elements': n_el,
            'n_vertices': len(vertices),
        })

        # Refine if not last level
        if level < n_refine:
            threshold = theta * eta.max()
            marked = np.where(eta > threshold)[0]
            print(f"    Marking {len(marked)} / {n_el} elements "
                  f"(θ={theta}, threshold={threshold:.2e})")

            if len(marked) == 0:
                print("    Converged — no elements to refine")
                break

            vertices, elements, cell_info = refine_nonconforming(
                vertices, elements, cell_info,
                marked, active, DI_map, E_map)

    # Visualize
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_adaptive_results(results, img, DI_map, active,
                              condition_name, output_dir)

    return results


# ── Visualization ────────────────────────────────────────────────────────

def plot_adaptive_results(results, img, DI_map, active,
                           condition_name, output_dir):
    """Multi-panel: mesh evolution + error convergence + final result."""
    n_levels = len(results)

    # Figure 1: Mesh evolution
    n_cols = min(n_levels, 4)
    fig, axes = plt.subplots(2, n_cols, figsize=(6 * n_cols, 12))
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    for col in range(n_cols):
        level_idx = col if n_levels <= 4 else col * (n_levels - 1) // max(n_cols - 1, 1)
        res = results[level_idx]
        verts = res['vertices']
        elems = res['elements']
        eta = res['eta']

        # Top row: mesh colored by refinement level
        ax = axes[0, col]
        patches = []
        colors = []
        for i, el in enumerate(elems):
            el_int = el.astype(int)
            poly = MplPolygon(verts[el_int], closed=True)
            patches.append(poly)
            colors.append(res['cell_info'][i]['level'] if i < len(res['cell_info']) else 0)

        pc = PatchCollection(patches, cmap='YlOrRd', edgecolor='k',
                             linewidth=0.3)
        pc.set_array(np.array(colors))
        ax.add_collection(pc)
        ax.set_xlim(verts[:, 0].min() - 1, verts[:, 0].max() + 1)
        ax.set_ylim(verts[:, 1].min() - 1, verts[:, 1].max() + 1)
        ax.set_aspect('equal')
        ax.set_title(f'Level {res["level"]}: {res["n_elements"]} el\n'
                     f'({res["n_vertices"]} vertices)')
        if col == 0:
            fig.colorbar(pc, ax=ax, label='Refinement depth', shrink=0.6)

        # Bottom row: error indicator
        ax = axes[1, col]
        patches2 = []
        for el in elems:
            el_int = el.astype(int)
            poly = MplPolygon(verts[el_int], closed=True)
            patches2.append(poly)

        pc2 = PatchCollection(patches2, cmap='hot_r', edgecolor='k',
                              linewidth=0.3)
        pc2.set_array(eta[:len(patches2)])
        ax.add_collection(pc2)
        ax.set_xlim(verts[:, 0].min() - 1, verts[:, 0].max() + 1)
        ax.set_ylim(verts[:, 1].min() - 1, verts[:, 1].max() + 1)
        ax.set_aspect('equal')
        ax.set_title(f'Error η (max={eta.max():.2e})')
        if col == 0:
            fig.colorbar(pc2, ax=ax, label='η', shrink=0.6)

    fig.suptitle(f'Adaptive Non-conforming VEM: {condition_name}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fname = os.path.join(output_dir, f'adaptive_mesh_evolution_{condition_name}.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")

    # Figure 2: Final result — 6-panel
    final = results[-1]
    verts = final['vertices']
    elems = final['elements']
    u = final['u']
    E_per = final['E_per_cell']
    DI_per = final['DI_per_cell']
    ux = u[0::2]
    uy = u[1::2]
    u_mag_node = np.sqrt(ux**2 + uy**2)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # (0,0) Original image
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('FISH image')
    axes[0, 0].axis('off')

    # (0,1) DI per cell
    _plot_field(axes[0, 1], verts, elems, DI_per, 'RdYlGn_r',
                f'DI (mean={DI_per.mean():.3f})', 'DI', fig)

    # (0,2) E per cell
    _plot_field(axes[0, 2], verts, elems, E_per, 'viridis',
                f'E(DI) (mean={E_per.mean():.0f} Pa)', 'E [Pa]', fig)

    # (1,0) Refinement level
    ref_levels = [info['level'] if i < len(final['cell_info']) else 0
                  for i, info in enumerate(final['cell_info'])]
    _plot_field(axes[1, 0], verts, elems, ref_levels[:len(elems)], 'YlOrRd',
                f'Refinement level (max={max(ref_levels)})', 'Level', fig)

    # (1,1) Displacement
    el_u = [np.mean(u_mag_node[el.astype(int)]) for el in elems]
    _plot_field(axes[1, 1], verts, elems, el_u, 'hot_r',
                f'|u| (max={u_mag_node.max():.4f})', '|u|', fig)

    # (1,2) Error indicator
    eta = final['eta']
    _plot_field(axes[1, 2], verts, elems, eta[:len(elems)], 'inferno',
                f'Error η (total={np.sqrt(np.sum(eta**2)):.2e})', 'η', fig)

    fig.suptitle(f'Adaptive VEM Final: {condition_name}\n'
                 f'{final["n_elements"]} elements, '
                 f'{final["n_vertices"]} vertices',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fname = os.path.join(output_dir, f'adaptive_final_{condition_name}.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")

    # Figure 3: Convergence
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    levels = [r['level'] for r in results]
    n_els = [r['n_elements'] for r in results]
    total_errors = [np.sqrt(np.sum(r['eta']**2)) for r in results]
    max_disps = [np.max(np.sqrt(r['u'][0::2]**2 + r['u'][1::2]**2))
                 for r in results]

    axes[0].semilogy(levels, total_errors, 'b-o', linewidth=2)
    axes[0].set_xlabel('Refinement level')
    axes[0].set_ylabel('Total error ||η||')
    axes[0].set_title('Error convergence')

    axes[1].plot(levels, n_els, 'r-s', linewidth=2)
    axes[1].set_xlabel('Refinement level')
    axes[1].set_ylabel('Number of elements')
    axes[1].set_title('Mesh growth')

    axes[2].plot(levels, max_disps, 'g-^', linewidth=2)
    axes[2].set_xlabel('Refinement level')
    axes[2].set_ylabel('Max |u|')
    axes[2].set_title('Peak displacement')

    fig.suptitle(f'Convergence: {condition_name}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fname = os.path.join(output_dir, f'adaptive_convergence_{condition_name}.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")


def _plot_field(ax, verts, elems, field, cmap, title, label, fig):
    """Helper: plot per-element field on polygon mesh."""
    patches = []
    for el in elems:
        el_int = el.astype(int)
        poly = MplPolygon(verts[el_int], closed=True)
        patches.append(poly)

    pc = PatchCollection(patches, cmap=cmap, edgecolor='k', linewidth=0.3)
    field_arr = np.array(field[:len(patches)])
    pc.set_array(field_arr)
    ax.add_collection(pc)
    margin = 1
    ax.set_xlim(verts[:, 0].min() - margin, verts[:, 0].max() + margin)
    ax.set_ylim(verts[:, 1].min() - margin, verts[:, 1].max() + margin)
    ax.set_aspect('equal')
    ax.set_title(title)
    fig.colorbar(pc, ax=ax, label=label, shrink=0.7)


# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    output_dir = sys.argv[1] if len(sys.argv) > 1 else '/tmp/adaptive_vem'
    img_dir = os.path.join(os.path.dirname(__file__), 'heine_extracted')

    # Run on representative images
    for name in ['fish_commensal_hobic_day21.png',
                 'fish_dysbiotic_hobic_day21.png']:
        cond = name.replace('fish_', '').replace('.png', '')
        adaptive_confocal_pipeline(
            os.path.join(img_dir, name),
            condition_name=cond,
            n_refine=3,
            nx_initial=16, ny_initial=16,
            theta=0.3,
            output_dir=output_dir)
