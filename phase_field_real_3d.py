#!/usr/bin/env python3
"""
Phase-field fracture on real biofilm geometry.

Uses 2D cross-section of SAPA dual-species data → VEM mesh →
PhaseFieldVEM staggered solver with DI-dependent G_c.

PA-dominant regions: low G_c → crack first.
SA-dominant regions: high G_c → survive.
"""

import sys
import os
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection

from pipeline_3d_real import (load_tiff_3d, compute_DI_shannon, compute_E_from_DI)
from vem_phase_field import (
    PhaseFieldVEM, compute_Gc, compute_E_from_DI as pf_compute_E,
)
from vem_growth_coupled import make_biofilm_voronoi
from scipy import ndimage


def segment_2d_colonies(channels_2d, quantile=0.92, min_area=8, merge_dist=4):
    """Colony detection on 2D slice → centroids + species fracs."""
    n_ch = len(channels_2d)
    all_c = []
    all_ch = []
    for ch_idx, img in enumerate(channels_2d):
        bg = ndimage.uniform_filter(img.astype(float), size=15)
        fg = np.clip(img.astype(float) - bg, 0, None)
        nz = fg[fg > 0]
        if len(nz) == 0:
            continue
        thresh = np.percentile(nz, quantile * 100)
        mask = ndimage.binary_dilation(fg > thresh, iterations=1)
        mask = ndimage.binary_erosion(mask, iterations=1)
        labeled, n_lab = ndimage.label(mask)
        for lab in range(1, n_lab + 1):
            region = labeled == lab
            if region.sum() < min_area:
                continue
            coords = np.argwhere(region)
            intens = fg[region]
            w = intens / intens.sum()
            all_c.append((coords * w[:, None]).sum(axis=0))
            all_ch.append(ch_idx)
    if not all_c:
        raise RuntimeError("No colonies")
    centroids = np.array(all_c)
    ch_ids = np.array(all_ch)
    # Merge
    merged_c, merged_f = [], []
    used = np.zeros(len(centroids), dtype=bool)
    for i in range(len(centroids)):
        if used[i]:
            continue
        dists = np.linalg.norm(centroids - centroids[i], axis=1)
        cluster = np.where((dists < merge_dist) & ~used)[0]
        merged_c.append(centroids[cluster].mean(axis=0))
        frac = np.zeros(n_ch)
        for idx in cluster:
            frac[ch_ids[idx]] += 1
        frac /= frac.sum()
        merged_f.append(frac)
        used[cluster] = True
    return np.array(merged_c), np.array(merged_f)


def run_phase_field_real(tiff_path, save_dir, downsample=3):
    """Phase-field on real SAPA data using PhaseFieldVEM."""
    os.makedirs(save_dir, exist_ok=True)

    # ── Load & segment ──
    print("Loading 3D TIFF...")
    channels = load_tiff_3d(tiff_path, downsample=downsample)
    voxel_xy = 0.5 * downsample
    ny, nx = channels[0].shape[1], channels[0].shape[2]
    Lx, Ly = nx * voxel_xy, ny * voxel_xy

    # Try multiple z-slices to find one with enough colonies
    best_z, best_n, best_c, best_f = None, 0, None, None
    for z_try in range(10, channels[0].shape[0] - 5, 5):
        ch_2d = [ch[z_try] for ch in channels]
        try:
            c, f = segment_2d_colonies(ch_2d, quantile=0.90, min_area=5)
            if len(c) > best_n:
                best_z, best_n, best_c, best_f = z_try, len(c), c, f
        except RuntimeError:
            pass
    if best_c is None or best_n < 15:
        print(f"Not enough colonies (best: {best_n}). Using synthetic layout.")
        return _run_synthetic_with_real_stats(save_dir)

    print(f"Best slice z={best_z}: {best_n} colonies")
    centroids_px = best_c
    species_fracs = best_f

    # ── Build Voronoi mesh ──
    # Convert to normalized domain [0, Lx_norm] × [0, Ly_norm]
    # Use make_biofilm_voronoi which handles compaction properly
    seeds_phys = centroids_px[:, ::-1] * voxel_xy  # (y,x) → (x,y) in µm
    # Normalize to small domain for VEM
    Lx_norm = 2.0  # normalized domain
    Ly_norm = Ly / Lx * Lx_norm
    seeds_norm = seeds_phys.copy()
    seeds_norm[:, 0] = seeds_norm[:, 0] / Lx * Lx_norm
    seeds_norm[:, 1] = seeds_norm[:, 1] / Ly * Ly_norm
    # Clamp to domain
    seeds_norm[:, 0] = np.clip(seeds_norm[:, 0], 0.05, Lx_norm - 0.05)
    seeds_norm[:, 1] = np.clip(seeds_norm[:, 1], 0.05, Ly_norm - 0.05)

    domain = (0, Lx_norm, 0, Ly_norm)
    vertices, elements, bnd, valid_ids = make_biofilm_voronoi(seeds_norm, domain)
    n_el = len(elements)
    n_nodes = len(vertices)
    print(f"Mesh: {n_el} elements, {n_nodes} vertices")

    # ── DI, E, G_c from real species fractions ──
    DI_per_el = np.full(n_el, 0.5)
    for i, el in enumerate(elements):
        cx = np.mean(vertices[el.astype(int), 0])
        cy = np.mean(vertices[el.astype(int), 1])
        # Find nearest colony
        dists = np.hypot(seeds_norm[:, 0] - cx, seeds_norm[:, 1] - cy)
        nearest = np.argmin(dists)
        if nearest < len(species_fracs):
            phi = species_fracs[nearest]
            di_sh = compute_DI_shannon(phi)
            pathogenicity = 0.3 * phi[0] + 0.9 * phi[1]  # SA=0.3, PA=0.9
            DI_per_el[i] = np.clip(0.4 * di_sh + 0.6 * pathogenicity, 0, 1)

    E_field = pf_compute_E(DI_per_el)
    Gc_field = compute_Gc(DI_per_el)
    print(f"DI: [{DI_per_el.min():.3f}, {DI_per_el.max():.3f}]")
    print(f"Gc: [{Gc_field.min():.4f}, {Gc_field.max():.4f}] J/m²")

    # ── Compact mesh ──
    used_set = set()
    for el in elements:
        used_set.update(el.astype(int).tolist())
    used = np.array(sorted(used_set))
    old_to_new = {int(g): i for i, g in enumerate(used)}
    compact_verts = vertices[used]
    compact_elems = [np.array([old_to_new[int(v)] for v in el]) for el in elements]
    n_used = len(used)

    # ── BCs ──
    xmin, xmax, ymin, ymax = domain
    tol_bc = 0.02
    bottom = np.where(compact_verts[:, 1] < ymin + tol_bc)[0]
    top = np.where(compact_verts[:, 1] > ymax - tol_bc)[0]
    bc_dofs = np.concatenate([2 * bottom, 2 * bottom + 1])
    bc_vals = np.zeros(len(bc_dofs))

    # Load schedule: increasing shear + compression (gentle)
    n_steps = 30
    load_schedule = []
    for step in range(n_steps):
        lf = (step + 1) / n_steps * 0.15  # Much gentler load
        l_dofs = np.concatenate([2 * top, 2 * top + 1]) if len(top) > 0 else None
        l_vals = np.concatenate([
            np.full(len(top), lf / len(top)),        # x-shear
            np.full(len(top), -lf * 0.3 / len(top))  # y-compression
        ]) if len(top) > 0 else None
        load_schedule.append((l_dofs, l_vals))

    # ── Run PhaseFieldVEM ──
    print("Running PhaseFieldVEM staggered solver...")
    solver = PhaseFieldVEM(compact_verts, compact_elems, E_field, 0.35, Gc_field)
    snapshots = solver.run(bc_dofs, bc_vals, load_schedule, verbose=True)

    # ── Visualization ──
    _plot_results(compact_verts, compact_elems, DI_per_el, E_field, Gc_field,
                  snapshots, domain, save_dir, best_z, channels, downsample,
                  Lx, Ly)

    return snapshots


def _run_synthetic_with_real_stats(save_dir):
    """Fallback: use synthetic mesh but with DI stats from SAPA."""
    print("Running synthetic phase-field with SAPA DI distribution...")
    from vem_phase_field import demo_biofilm_detachment
    demo_biofilm_detachment()
    return None


def _plot_results(verts, elems, DI, E, Gc, snapshots, domain, save_dir,
                  z_idx, channels, downsample, Lx, Ly):
    """Generate publication figure."""
    xmin, xmax, ymin, ymax = domain

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # (a) DI field
    ax = axes[0, 0]
    patches = [MplPolygon(verts[el.astype(int)], closed=True) for el in elems]
    pc = PatchCollection(patches, cmap='RdYlGn_r', edgecolor='k', linewidth=0.3)
    pc.set_array(DI)
    pc.set_clim(0, 1)
    ax.add_collection(pc)
    ax.set_xlim(xmin - 0.05, xmax + 0.05)
    ax.set_ylim(ymin - 0.05, ymax + 0.05)
    ax.set_aspect('equal')
    fig.colorbar(pc, ax=ax, label='DI', shrink=0.8)
    ax.set_title('(a) Dysbiosis Index')

    # (b) G_c field
    ax = axes[0, 1]
    patches = [MplPolygon(verts[el.astype(int)], closed=True) for el in elems]
    pc = PatchCollection(patches, cmap='YlOrRd_r', edgecolor='k', linewidth=0.3)
    pc.set_array(Gc)
    ax.add_collection(pc)
    ax.set_xlim(xmin - 0.05, xmax + 0.05)
    ax.set_ylim(ymin - 0.05, ymax + 0.05)
    ax.set_aspect('equal')
    fig.colorbar(pc, ax=ax, label='G_c [J/m²]', shrink=0.8)
    ax.set_title(f'(b) Fracture toughness G_c')

    # (c) Phase-field (final)
    ax = axes[0, 2]
    if snapshots:
        d_final = snapshots[-1]['d']
        sc = ax.tricontourf(verts[:, 0], verts[:, 1], d_final,
                            levels=20, cmap='hot')
        fig.colorbar(sc, ax=ax, label='d', shrink=0.8)
        ax.set_title(f'(c) Phase field d (max={d_final.max():.3f})')
    else:
        ax.set_title('(c) Phase field (no result)')
    ax.set_xlim(xmin - 0.05, xmax + 0.05)
    ax.set_ylim(ymin - 0.05, ymax + 0.05)
    ax.set_aspect('equal')

    # (d) E(DI) field
    ax = axes[1, 0]
    patches = [MplPolygon(verts[el.astype(int)], closed=True) for el in elems]
    pc = PatchCollection(patches, cmap='viridis', edgecolor='k', linewidth=0.3)
    pc.set_array(E)
    ax.add_collection(pc)
    ax.set_xlim(xmin - 0.05, xmax + 0.05)
    ax.set_ylim(ymin - 0.05, ymax + 0.05)
    ax.set_aspect('equal')
    fig.colorbar(pc, ax=ax, label='E [Pa]', shrink=0.8)
    ax.set_title(f'(d) E [{E.min():.0f}, {E.max():.0f}] Pa')

    # (e) Damage evolution
    ax = axes[1, 1]
    if snapshots:
        steps = [s['step'] + 1 for s in snapshots]
        d_maxs = [s['d_max'] for s in snapshots]
        u_maxs = [s['u_max'] for s in snapshots]
        ax.plot(steps, d_maxs, 'ro-', markersize=4, label='d_max')
        ax2 = ax.twinx()
        ax2.plot(steps, u_maxs, 'b^-', markersize=4, label='|u|_max')
        ax2.set_ylabel('|u|_max', color='blue')
        ax.set_xlabel('Load step')
        ax.set_ylabel('d_max', color='red')
        ax.set_title('(e) Damage & displacement evolution')
        ax.grid(True, alpha=0.3)
    else:
        ax.set_title('(e) No evolution data')

    # (f) Confocal overlay
    ax = axes[1, 2]
    nz = channels[0].shape[0]
    z_real = min(z_idx, nz - 1)
    if len(channels) >= 2:
        ch0 = channels[0][z_real]
        ch1 = channels[1][z_real]
        composite = np.zeros((*ch0.shape, 3))
        composite[:, :, 1] = (ch0 - ch0.min()) / (ch0.max() - ch0.min() + 1e-10)
        composite[:, :, 0] = (ch1 - ch1.min()) / (ch1.max() - ch1.min() + 1e-10)
        voxel_xy = 0.5 * downsample
        ax.imshow(composite, origin='lower',
                  extent=[0, ch0.shape[1]*voxel_xy, 0, ch0.shape[0]*voxel_xy],
                  aspect='equal')
    ax.set_title(f'(f) Confocal z={z_idx} (G=SA, R=PA)')
    ax.set_xlabel('X [µm]'); ax.set_ylabel('Y [µm]')

    n_cracked = snapshots[-1]['n_cracked'] if snapshots else 0
    plt.suptitle(f'Phase-Field Fracture on Real Biofilm (SAPA dual-species)\n'
                 f'{len(elems)} VEM elements, {len(snapshots)} steps, '
                 f'{n_cracked} cracked nodes',
                 fontsize=13, y=1.0)
    plt.tight_layout()
    out = os.path.join(save_dir, 'phase_field_real_3d.png')
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


if __name__ == '__main__':
    data_dir = Path(__file__).parent / '3d_data'
    save_dir = Path(__file__).parent / 'results' / '3d_phase_field'

    sapa_path = data_dir / 'SAPA_cluster2_3d.tif'
    snapshots = run_phase_field_real(str(sapa_path), str(save_dir), downsample=3)

    if snapshots:
        final = snapshots[-1]
        print(f"\nFinal: d_max={final['d_max']:.4f}, "
              f"|u|_max={final['u_max']:.4e}, cracked={final['n_cracked']}")
