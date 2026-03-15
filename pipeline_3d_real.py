#!/usr/bin/env python3
"""
3D Real Confocal → Voronoi → Polyhedral VEM Pipeline
=====================================================

Loads real light-sheet biofilm TIFF z-stacks (Zenodo 10.5281/zenodo.18154035),
segments colonies, builds 3D Voronoi mesh, and runs polyhedral VEM stress analysis.

Datasets:
  - PA_cluster2_3d.tif:   P. aeruginosa single-species  (357, 2, 339, 282)
  - SAPA_cluster2_3d.tif:  S.aureus + P.aeruginosa dual  (199, 2, 428, 404)

Author: Nishioka K., IKM, Leibniz Universität Hannover
"""

import sys
import os
import time
import numpy as np
from pathlib import Path

# ── imports ──────────────────────────────────────────────────────
import tifffile
from scipy import ndimage
from scipy.spatial import Voronoi

# Add VEM modules
sys.path.insert(0, str(Path(__file__).parent))
from vem_3d_advanced import vem_3d_sparse, export_vtk, _merge_vertices, _order_face_vertices
from vem_3d_confocal import vem_3d_solve as vem_3d_solve_confocal

# ── Material model (from Tmcmc paper) ───────────────────────────
E_MAX = 1000.0   # Pa, commensal (low DI)
E_MIN = 30.0     # Pa, dysbiotic (high DI)
N_HILL = 2       # percolation exponent
NU = 0.35        # Poisson's ratio


def compute_DI_shannon(phi):
    """Normalized Shannon entropy DI = -Σ φᵢ ln φᵢ / ln(N)."""
    phi = np.asarray(phi, dtype=float)
    phi = phi[phi > 1e-12]
    if len(phi) <= 1:
        return 0.0
    N = len(phi)
    H = -np.sum(phi * np.log(phi))
    return float(H / np.log(N))


def compute_E_from_DI(DI):
    """E(DI) = E_MIN + (E_MAX - E_MIN)(1 - DI)^n."""
    return E_MIN + (E_MAX - E_MIN) * (1.0 - np.clip(DI, 0, 1)) ** N_HILL


# ═══════════════════════════════════════════════════════════════
# Step 1: Load & preprocess TIFF z-stack
# ═══════════════════════════════════════════════════════════════

def load_tiff_3d(path, downsample=2):
    """
    Load 3D TIFF z-stack and return per-channel volumes.

    Parameters
    ----------
    path : str
        Path to TIFF file. Expected shape: (Z, C, Y, X).
    downsample : int
        Spatial downsample factor (reduce memory).

    Returns
    -------
    channels : list of ndarray, each (nz, ny, nx) float32
    """
    raw = tifffile.imread(path)
    print(f"  Raw shape: {raw.shape}, dtype: {raw.dtype}")

    if raw.ndim == 4:
        # (Z, C, Y, X)
        n_channels = raw.shape[1]
        channels = []
        for c in range(n_channels):
            vol = raw[:, c].astype(np.float32)
            if downsample > 1:
                vol = vol[::downsample, ::downsample, ::downsample]
            channels.append(vol)
    elif raw.ndim == 3:
        vol = raw.astype(np.float32)
        if downsample > 1:
            vol = vol[::downsample, ::downsample, ::downsample]
        channels = [vol]
    else:
        raise ValueError(f"Unexpected TIFF ndim={raw.ndim}")

    for i, ch in enumerate(channels):
        print(f"  Ch{i}: shape={ch.shape}, range=[{ch.min():.1f}, {ch.max():.1f}]")
    return channels


# ═══════════════════════════════════════════════════════════════
# Step 2: Segment colonies via threshold + connected components
# ═══════════════════════════════════════════════════════════════

def segment_colonies_3d(channels, quantile=0.97, min_volume_voxels=50,
                        merge_distance_voxels=8):
    """
    Detect colony centroids from multi-channel 3D fluorescence.

    Parameters
    ----------
    channels : list of (nz, ny, nx) arrays
    quantile : float
        Intensity quantile for foreground threshold.
    min_volume_voxels : int
        Minimum colony volume (in voxels) to keep.
    merge_distance_voxels : float
        Merge centroids closer than this distance.

    Returns
    -------
    centroids : (N, 3) array of (z, y, x) voxel coordinates
    species_fracs : (N, n_ch) array of species fractions per colony
    """
    n_ch = len(channels)
    all_centroids = []
    all_channel_ids = []

    for ch_idx, vol in enumerate(channels):
        # Background subtraction: rolling ball approximation
        bg = ndimage.uniform_filter(vol, size=30)
        fg = np.clip(vol - bg, 0, None)

        # Threshold
        nonzero = fg[fg > 0]
        if len(nonzero) == 0:
            continue
        thresh = np.percentile(nonzero, quantile * 100)
        mask = fg > thresh

        # Morphological cleanup
        mask = ndimage.binary_dilation(mask, iterations=1)
        mask = ndimage.binary_erosion(mask, iterations=1)

        # Connected components
        labeled, n_labels = ndimage.label(mask)
        for lab in range(1, n_labels + 1):
            region = labeled == lab
            vol_size = region.sum()
            if vol_size < min_volume_voxels:
                continue
            # Intensity-weighted centroid
            coords = np.argwhere(region)  # (M, 3)
            intensities = fg[region]
            w = intensities / intensities.sum()
            centroid = (coords * w[:, None]).sum(axis=0)
            all_centroids.append(centroid)
            all_channel_ids.append(ch_idx)

    if len(all_centroids) == 0:
        raise RuntimeError("No colonies detected! Try lowering quantile threshold.")

    centroids = np.array(all_centroids)
    channel_ids = np.array(all_channel_ids)
    print(f"  Raw detections: {len(centroids)} colonies across {n_ch} channels")

    # ── Merge nearby centroids ──
    merged_centroids = []
    merged_fracs = []
    used = np.zeros(len(centroids), dtype=bool)

    for i in range(len(centroids)):
        if used[i]:
            continue
        # Find all within merge distance
        dists = np.linalg.norm(centroids - centroids[i], axis=1)
        cluster = np.where((dists < merge_distance_voxels) & ~used)[0]

        # Merged centroid = mean position
        c_mean = centroids[cluster].mean(axis=0)
        merged_centroids.append(c_mean)

        # Species fractions = count per channel / total
        frac = np.zeros(n_ch)
        for idx in cluster:
            frac[channel_ids[idx]] += 1
        frac /= frac.sum()
        merged_fracs.append(frac)
        used[cluster] = True

    centroids = np.array(merged_centroids)
    species_fracs = np.array(merged_fracs)
    print(f"  After merge: {len(centroids)} colonies")
    return centroids, species_fracs


# ═══════════════════════════════════════════════════════════════
# Step 3: Build 3D Voronoi polyhedral mesh
# ═══════════════════════════════════════════════════════════════

def build_voronoi_mesh_3d(centroids_norm):
    """
    Build bounded 3D Voronoi mesh from normalized centroids in [0,1]³.

    Uses mirror-point approach from vem_3d_confocal.py (proven working).

    Parameters
    ----------
    centroids_norm : (N, 3) array in [0, 1]³

    Returns
    -------
    vertices : (M, 3) array in [0,1]³
    cells : list of arrays (vertex indices per cell)
    cell_faces : list of lists of arrays (face vertex indices per cell)
    seed_to_cell : dict mapping seed index to cell index
    """
    pts_norm = np.clip(centroids_norm.copy(), 0.02, 0.98)
    n_seeds = len(pts_norm)

    # Mirror across 6 faces for bounded Voronoi
    all_pts = [pts_norm]
    for axis in range(3):
        for val in [0.0, 1.0]:
            mirror = pts_norm.copy()
            mirror[:, axis] = 2 * val - mirror[:, axis]
            all_pts.append(mirror)
    all_pts = np.vstack(all_pts)

    vor = Voronoi(all_pts)
    raw_verts = vor.vertices.copy()

    # Build faces per original seed from ridge information
    seed_faces = {i: [] for i in range(n_seeds)}
    for ridge_idx, (p1, p2) in enumerate(vor.ridge_points):
        fv = vor.ridge_vertices[ridge_idx]
        if -1 in fv:
            continue
        if p1 < n_seeds:
            seed_faces[p1].append(np.array(fv))
        if p2 < n_seeds:
            seed_faces[p2].append(np.array(fv))

    # Clip to [0,1]³ and merge duplicate vertices
    raw_verts = np.clip(raw_verts, -0.001, 1.001)
    unique_verts, vert_remap = _merge_vertices(raw_verts, tol=1e-8)

    cells = []
    cell_faces_list = []
    seed_to_cell = {}

    for i in range(n_seeds):
        faces_raw = seed_faces[i]
        if len(faces_raw) < 4:
            continue

        faces = []
        cell_vert_set = set()
        for fv in faces_raw:
            remapped = np.array([vert_remap[v] for v in fv])
            _, idx = np.unique(remapped, return_index=True)
            remapped = remapped[np.sort(idx)]
            if len(remapped) >= 3:
                faces.append(remapped)
                cell_vert_set.update(remapped)

        if len(faces) < 4 or len(cell_vert_set) < 4:
            continue

        cell_verts = np.array(sorted(cell_vert_set))
        cell_center = unique_verts[cell_verts].mean(axis=0)

        ordered_faces = []
        for fv in faces:
            ordered = _order_face_vertices(unique_verts, fv, cell_center)
            if ordered is not None:
                ordered_faces.append(ordered)

        if len(ordered_faces) >= 4:
            seed_to_cell[i] = len(cells)
            cells.append(cell_verts)
            cell_faces_list.append(ordered_faces)

    # Clamp vertices to [0,1]³
    unique_verts = np.clip(unique_verts, 0.0, 1.0)

    print(f"  Voronoi mesh: {len(cells)} cells, {len(unique_verts)} vertices")
    return unique_verts, cells, cell_faces_list, seed_to_cell


# ═══════════════════════════════════════════════════════════════
# Step 4: Assign DI and material properties
# ═══════════════════════════════════════════════════════════════

def assign_material_properties(species_fracs, seed_to_cell, n_cells,
                               dataset_type='single', vertices=None, cells=None):
    """
    Compute DI and E per cell.

    Parameters
    ----------
    species_fracs : (N_seeds, n_ch) array
    seed_to_cell : dict {seed_idx: cell_idx}
    n_cells : int
    dataset_type : 'single' or 'dual'
    vertices : optional, for spatial variation in single-species
    cells : optional

    Returns
    -------
    DI : (n_cells,) array
    E : (n_cells,) array
    """
    DI = np.full(n_cells, 0.5)  # default
    E = np.full(n_cells, 500.0)

    for seed_idx, cell_idx in seed_to_cell.items():
        if seed_idx >= len(species_fracs):
            continue
        phi = species_fracs[seed_idx]

        if dataset_type == 'single':
            # Single species P. aeruginosa: pathogenic, forms soft biofilm
            # Depth-dependent DI: deeper = more mature = higher DI
            if vertices is not None and cells is not None:
                cell_z = vertices[cells[cell_idx]].mean(axis=0)[2]
                z_max = vertices[:, 2].max()
                z_frac = cell_z / z_max if z_max > 0 else 0.5
                DI[cell_idx] = 0.5 + 0.4 * z_frac  # [0.5, 0.9] range
            else:
                DI[cell_idx] = 0.7
            E[cell_idx] = compute_E_from_DI(DI[cell_idx])
        elif dataset_type == 'dual':
            # Dual species: Shannon entropy of species mix
            # S.aureus (Ch0) = commensal-like, P.aeruginosa (Ch1) = pathogenic
            # Map 2ch → DI: weighted by pathogenicity
            # phi = [SA_frac, PA_frac]
            DI_val = compute_DI_shannon(phi)
            # Also add pathogenicity weight: PA is more pathogenic
            pathogenicity = 0.3 * phi[0] + 0.9 * phi[1]  # SA=0.3, PA=0.9
            # Combine: diversity + pathogenicity
            DI_combined = 0.4 * DI_val + 0.6 * pathogenicity
            DI[cell_idx] = np.clip(DI_combined, 0.0, 1.0)
            E[cell_idx] = compute_E_from_DI(DI[cell_idx])

    return DI, E


# ═══════════════════════════════════════════════════════════════
# Step 5: VEM solve with boundary conditions
# ═══════════════════════════════════════════════════════════════

def setup_bc_and_solve(vertices, cells, cell_faces, E_field, nu=NU):
    """
    Set up BCs (fixed bottom, pressure on top) and solve VEM.

    Returns
    -------
    u : (3*N,) displacement vector
    """
    n_verts = len(vertices)
    z_coords = vertices[:, 2]
    z_min, z_max = z_coords.min(), z_coords.max()
    z_range = z_max - z_min

    # Fixed BC: bottom 5%
    bottom_nodes = np.where(z_coords < z_min + 0.05 * z_range)[0]
    bc_fixed_dofs = []
    bc_vals = []
    for n in bottom_nodes:
        for d in range(3):
            bc_fixed_dofs.append(3 * n + d)
            bc_vals.append(0.0)
    bc_fixed_dofs = np.array(bc_fixed_dofs, dtype=int)
    bc_vals = np.array(bc_vals)

    # Load: GCF pressure on top 5% (downward z)
    top_nodes = np.where(z_coords > z_max - 0.05 * z_range)[0]
    pressure = 10.0  # Pa (representative GCF shear)
    load_dofs = np.array([3 * n + 2 for n in top_nodes], dtype=int)  # z-direction
    load_vals = np.full(len(top_nodes), -pressure / len(top_nodes))

    print(f"  BC: {len(bottom_nodes)} fixed nodes, {len(top_nodes)} loaded nodes")
    print(f"  Pressure: {pressure} Pa on top surface")

    u = vem_3d_solve_confocal(vertices, cells, cell_faces, E_field, nu,
                              bc_fixed_dofs, bc_vals, load_dofs, load_vals)
    return u


# ═══════════════════════════════════════════════════════════════
# Step 6: Visualization
# ═══════════════════════════════════════════════════════════════

def plot_results(vertices, cells, cell_faces, DI, E, u, centroids_phys,
                 species_fracs, dataset_name, save_dir):
    """Generate publication-quality figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from matplotlib.colors import Normalize

    os.makedirs(save_dir, exist_ok=True)
    n_verts = len(vertices)

    # Displacement magnitude
    u_mag = np.sqrt(u[0::3]**2 + u[1::3]**2 + u[2::3]**2)

    # ── Figure 1: 4-panel overview ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), subplot_kw={'projection': '3d'})

    # (a) Colony positions colored by dominant species
    ax = axes[0, 0]
    n_ch = species_fracs.shape[1] if species_fracs.ndim > 1 else 1
    if n_ch > 1:
        dominant = np.argmax(species_fracs, axis=1)
        colors_map = ['#e41a1c', '#377eb8', '#4daf4a', '#ff7f00', '#984ea3']
        c_list = [colors_map[d % len(colors_map)] for d in dominant]
    else:
        c_list = '#377eb8'
    ax.scatter(centroids_phys[:, 2], centroids_phys[:, 1], centroids_phys[:, 0],
               c=c_list, s=15, alpha=0.6)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title(f'(a) Colony positions ({len(centroids_phys)} colonies)')

    # (b) DI field (cell centroids)
    ax = axes[0, 1]
    cell_centers = []
    for cell_verts in cells:
        cc = vertices[cell_verts].mean(axis=0)
        cell_centers.append(cc)
    cell_centers = np.array(cell_centers)
    sc = ax.scatter(cell_centers[:, 0], cell_centers[:, 1], cell_centers[:, 2],
                    c=DI, cmap='RdYlGn_r', s=30, alpha=0.7, vmin=0, vmax=1)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title(f'(b) DI field (mean={DI.mean():.3f})')
    fig.colorbar(sc, ax=ax, shrink=0.6, label='DI')

    # (c) E modulus field
    ax = axes[1, 0]
    sc = ax.scatter(cell_centers[:, 0], cell_centers[:, 1], cell_centers[:, 2],
                    c=E, cmap='viridis', s=30, alpha=0.7, vmin=E_MIN, vmax=E_MAX)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title(f'(c) E field [{E.min():.0f}, {E.max():.0f}] Pa')
    fig.colorbar(sc, ax=ax, shrink=0.6, label='E [Pa]')

    # (d) Displacement magnitude
    ax = axes[1, 1]
    sc = ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                    c=u_mag, cmap='hot', s=5, alpha=0.5)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title(f'(d) |u| [{u_mag.min():.4f}, {u_mag.max():.4f}]')
    fig.colorbar(sc, ax=ax, shrink=0.6, label='|u|')

    plt.suptitle(f'3D Confocal → VEM Pipeline: {dataset_name}', fontsize=14, y=0.98)
    plt.tight_layout()
    out = os.path.join(save_dir, f'{dataset_name}_overview.png')
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")

    # ── Figure 2: Cross-section slices ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    z_mid = (vertices[:, 2].min() + vertices[:, 2].max()) / 2
    y_mid = (vertices[:, 1].min() + vertices[:, 1].max()) / 2
    x_mid = (vertices[:, 0].min() + vertices[:, 0].max()) / 2

    # XY slice (at z_mid)
    ax = axes[0]
    tol = (vertices[:, 2].max() - vertices[:, 2].min()) * 0.1
    mask = np.abs(cell_centers[:, 2] - z_mid) < tol
    if mask.sum() > 0:
        sc = ax.scatter(cell_centers[mask, 0], cell_centers[mask, 1],
                        c=E[mask], cmap='viridis', s=50, vmin=E_MIN, vmax=E_MAX)
        fig.colorbar(sc, ax=ax, label='E [Pa]')
    ax.set_xlabel('X'); ax.set_ylabel('Y')
    ax.set_title(f'XY slice (z≈{z_mid:.2f})')
    ax.set_aspect('equal')

    # XZ slice (at y_mid)
    ax = axes[1]
    tol = (vertices[:, 1].max() - vertices[:, 1].min()) * 0.1
    mask = np.abs(cell_centers[:, 1] - y_mid) < tol
    if mask.sum() > 0:
        sc = ax.scatter(cell_centers[mask, 0], cell_centers[mask, 2],
                        c=DI[mask], cmap='RdYlGn_r', s=50, vmin=0, vmax=1)
        fig.colorbar(sc, ax=ax, label='DI')
    ax.set_xlabel('X'); ax.set_ylabel('Z')
    ax.set_title(f'XZ slice (y≈{y_mid:.2f})')

    # YZ slice (at x_mid)
    ax = axes[2]
    tol = (vertices[:, 0].max() - vertices[:, 0].min()) * 0.1
    mask = np.abs(cell_centers[:, 0] - x_mid) < tol
    if mask.sum() > 0:
        u_cell = np.zeros(len(cells))
        for i, cv in enumerate(cells):
            u_cell[i] = u_mag[cv].mean() if len(cv) > 0 else 0
        sc = ax.scatter(cell_centers[mask, 1], cell_centers[mask, 2],
                        c=u_cell[mask], cmap='hot', s=50)
        fig.colorbar(sc, ax=ax, label='|u| mean')
    ax.set_xlabel('Y'); ax.set_ylabel('Z')
    ax.set_title(f'YZ slice (x≈{x_mid:.2f})')

    plt.suptitle(f'{dataset_name} — Cross-sections', fontsize=13)
    plt.tight_layout()
    out = os.path.join(save_dir, f'{dataset_name}_slices.png')
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")

    # ── Figure 3: Histograms ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(DI, bins=30, color='steelblue', edgecolor='white')
    axes[0].set_xlabel('DI'); axes[0].set_title('DI distribution')
    axes[1].hist(E, bins=30, color='forestgreen', edgecolor='white')
    axes[1].set_xlabel('E [Pa]'); axes[1].set_title('E distribution')
    axes[2].hist(u_mag, bins=30, color='firebrick', edgecolor='white')
    axes[2].set_xlabel('|u|'); axes[2].set_title('Displacement distribution')
    plt.suptitle(f'{dataset_name} — Distributions', fontsize=13)
    plt.tight_layout()
    out = os.path.join(save_dir, f'{dataset_name}_histograms.png')
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════════
# Full Pipeline
# ═══════════════════════════════════════════════════════════════

def run_pipeline(tiff_path, dataset_name, dataset_type='single',
                 downsample=4, max_colonies=200,
                 quantile=0.97, min_vol=30, merge_dist=6,
                 save_dir='results/3d_real'):
    """
    End-to-end: TIFF → segment → Voronoi → VEM → visualization.

    Parameters
    ----------
    tiff_path : str
    dataset_name : str
    dataset_type : 'single' or 'dual'
    downsample : int
    max_colonies : int
        Cap on number of colonies (for tractable mesh).
    """
    timings = {}
    os.makedirs(save_dir, exist_ok=True)

    # ── Step 1: Load ──
    print(f"\n{'='*60}")
    print(f"Pipeline: {dataset_name} ({dataset_type})")
    print(f"{'='*60}")
    print("Step 1: Loading TIFF...")
    t0 = time.time()
    channels = load_tiff_3d(tiff_path, downsample=downsample)
    timings['load'] = time.time() - t0

    # Physical domain (approximate from Zenodo metadata)
    # Light sheet: ~0.5 µm/voxel lateral, ~1 µm axial
    voxel_size = np.array([1.0, 0.5, 0.5]) * downsample  # (z, y, x) µm
    shape = channels[0].shape  # (nz, ny, nx)
    Lz = shape[0] * voxel_size[0]
    Ly = shape[1] * voxel_size[1]
    Lx = shape[2] * voxel_size[2]
    print(f"  Physical domain: {Lx:.0f} × {Ly:.0f} × {Lz:.0f} µm")

    # ── Step 2: Segment ──
    print("Step 2: Colony segmentation...")
    t0 = time.time()
    centroids_vox, species_fracs = segment_colonies_3d(
        channels, quantile=quantile, min_volume_voxels=min_vol,
        merge_distance_voxels=merge_dist
    )
    timings['segment'] = time.time() - t0

    # Subsample if too many
    if len(centroids_vox) > max_colonies:
        print(f"  Subsampling {len(centroids_vox)} → {max_colonies} colonies")
        rng = np.random.RandomState(42)
        idx = rng.choice(len(centroids_vox), max_colonies, replace=False)
        centroids_vox = centroids_vox[idx]
        species_fracs = species_fracs[idx]

    # Convert to physical coordinates
    centroids_phys = centroids_vox * voxel_size  # (z, y, x) in µm

    # ── Step 3: Voronoi mesh ──
    print("Step 3: Building 3D Voronoi mesh...")
    t0 = time.time()
    # Normalize to [0, 1]³
    c_norm = np.zeros_like(centroids_phys)
    c_norm[:, 0] = centroids_phys[:, 0] / Lz  # z
    c_norm[:, 1] = centroids_phys[:, 1] / Ly  # y
    c_norm[:, 2] = centroids_phys[:, 2] / Lx  # x

    # Reorder to (x, y, z) for VEM solver convention
    c_norm_xyz = c_norm[:, ::-1]  # (x, y, z)

    vertices_norm, cells, cell_faces, seed_to_cell = build_voronoi_mesh_3d(c_norm_xyz)
    timings['mesh'] = time.time() - t0

    # Scale vertices to physical coordinates (µm)
    vertices_raw = vertices_norm.copy()
    vertices_raw[:, 0] *= Lx
    vertices_raw[:, 1] *= Ly
    vertices_raw[:, 2] *= Lz

    # ── Remove unused vertices (critical for solver) ──
    used_verts = set()
    for cv in cells:
        used_verts.update(cv.tolist())
    for cf_list in cell_faces:
        for face in cf_list:
            used_verts.update(face.astype(int).tolist())
    used_sorted = sorted(used_verts)
    old_to_new = {old: new for new, old in enumerate(used_sorted)}

    vertices = vertices_raw[used_sorted]
    cells = [np.array([old_to_new[int(v)] for v in cv]) for cv in cells]
    cell_faces = [
        [np.array([old_to_new[int(v)] for v in face]) for face in cf_list]
        for cf_list in cell_faces
    ]
    # Update seed_to_cell (cell indices unchanged, only vertex indices remapped)
    print(f"  Vertex cleanup: {len(vertices_raw)} → {len(vertices)} "
          f"(removed {len(vertices_raw) - len(vertices)} unused)")

    if len(cells) < 4:
        print("ERROR: Too few valid cells. Try adjusting segmentation parameters.")
        return None

    # ── Step 4: Material properties ──
    print("Step 4: Assigning material properties...")
    t0 = time.time()
    DI, E = assign_material_properties(species_fracs, seed_to_cell,
                                       len(cells), dataset_type,
                                       vertices=vertices, cells=cells)
    timings['material'] = time.time() - t0
    print(f"  DI: [{DI.min():.3f}, {DI.max():.3f}], mean={DI.mean():.3f}")
    print(f"  E:  [{E.min():.0f}, {E.max():.0f}] Pa, mean={E.mean():.0f}")

    # ── Step 5: VEM solve ──
    print("Step 5: VEM solve...")
    t0 = time.time()
    u = setup_bc_and_solve(vertices, cells, cell_faces, E)
    timings['solve'] = time.time() - t0

    u_mag = np.sqrt(u[0::3]**2 + u[1::3]**2 + u[2::3]**2)
    print(f"  |u| max = {u_mag.max():.6f}, mean = {u_mag.mean():.6f}")

    # ── Step 6: Export VTK ──
    print("Step 6: Exporting VTK...")
    vtk_path = os.path.join(save_dir, f'{dataset_name}.vtk')
    try:
        export_vtk(vtk_path, vertices, cells, cell_faces,
                   point_data={'displacement': u.reshape(-1, 3),
                               'u_magnitude': u_mag},
                   cell_data={'DI': DI, 'E_modulus': E})
        print(f"  Saved: {vtk_path}")
    except Exception as ex:
        print(f"  VTK export skipped: {ex}")

    # ── Step 7: Visualization ──
    print("Step 7: Generating figures...")
    t0 = time.time()
    centroids_phys_xyz = centroids_phys[:, ::-1]  # to (x, y, z)
    plot_results(vertices, cells, cell_faces, DI, E, u,
                 centroids_phys_xyz, species_fracs, dataset_name, save_dir)
    timings['plot'] = time.time() - t0

    # ── Summary ──
    total = sum(timings.values())
    print(f"\n{'─'*40}")
    print(f"Timings:")
    for k, v in timings.items():
        print(f"  {k:>10}: {v:.2f}s")
    print(f"  {'TOTAL':>10}: {total:.2f}s")

    result = {
        'vertices': vertices, 'cells': cells, 'cell_faces': cell_faces,
        'DI': DI, 'E': E, 'u': u,
        'centroids_phys': centroids_phys_xyz,
        'species_fracs': species_fracs,
        'seed_to_cell': seed_to_cell,
        'timings': timings,
        'domain_um': (Lx, Ly, Lz),
    }
    return result


# ═══════════════════════════════════════════════════════════════
# Comparison: Single vs Dual species
# ═══════════════════════════════════════════════════════════════

def compare_single_vs_dual(results_pa, results_sapa, save_dir):
    """Generate comparison figure: single-species vs dual-species."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for row, (res, label) in enumerate([(results_pa, 'PA (single)'),
                                         (results_sapa, 'SAPA (dual)')]):
        if res is None:
            continue
        DI, E, u = res['DI'], res['E'], res['u']
        u_mag = np.sqrt(u[0::3]**2 + u[1::3]**2 + u[2::3]**2)

        axes[row, 0].hist(DI, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
        axes[row, 0].set_xlabel('DI')
        axes[row, 0].set_title(f'{label}: DI (mean={DI.mean():.3f})')
        axes[row, 0].set_xlim(0, 1)

        axes[row, 1].hist(E, bins=30, color='forestgreen', edgecolor='white', alpha=0.8)
        axes[row, 1].set_xlabel('E [Pa]')
        axes[row, 1].set_title(f'{label}: E (mean={E.mean():.0f} Pa)')
        axes[row, 1].set_xlim(0, E_MAX * 1.1)

        axes[row, 2].hist(u_mag, bins=30, color='firebrick', edgecolor='white', alpha=0.8)
        axes[row, 2].set_xlabel('|u|')
        axes[row, 2].set_title(f'{label}: |u| (max={u_mag.max():.5f})')

    plt.suptitle('Single vs Dual Species: 3D VEM Comparison', fontsize=14, y=0.98)
    plt.tight_layout()
    out = os.path.join(save_dir, 'comparison_single_vs_dual.png')
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison: {out}")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    data_dir = Path(__file__).parent / '3d_data'
    save_dir = Path(__file__).parent / 'results' / '3d_real'

    # ── Run PA (single species) ──
    pa_path = data_dir / 'PA_cluster2_3d.tif'
    results_pa = run_pipeline(
        str(pa_path), 'PA_single',
        dataset_type='single', downsample=3,
        max_colonies=200, quantile=0.93, min_vol=20, merge_dist=5,
        save_dir=str(save_dir)
    )

    # ── Run SAPA (dual species) ──
    sapa_path = data_dir / 'SAPA_cluster2_3d.tif'
    results_sapa = run_pipeline(
        str(sapa_path), 'SAPA_dual',
        dataset_type='dual', downsample=3,
        max_colonies=200, quantile=0.93, min_vol=20, merge_dist=5,
        save_dir=str(save_dir)
    )

    # ── Comparison ──
    if results_pa is not None and results_sapa is not None:
        compare_single_vs_dual(results_pa, results_sapa, str(save_dir))

    print("\n✓ All pipelines complete!")
