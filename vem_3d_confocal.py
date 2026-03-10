"""
3D Confocal -> VEM Pipeline: from z-stack to mechanical analysis.

Extends the 2D confocal pipeline (vem_confocal_pipeline.py) to full 3D:
  1. Synthetic 3D confocal z-stack (5-species fluorescence volumes)
  2. 3D colony detection via peak finding
  3. 3D Voronoi -> polyhedral VEM mesh
  4. Per-element E(DI) from species composition
  5. 3D VEM elasticity solve (fixed bottom, GCF pressure top)
  6. Cross-section visualization + pipeline comparison

Domain: 200 x 200 x 50 um (oral biofilm on tooth surface)

References:
  - Heine et al. (2025): 5-species oral biofilm confocal imaging
  - Nishioka thesis: E(DI) constitutive law
  - Gain, Talischi, Paulino (2014): VEM for 3D linear elasticity
"""

import numpy as np
from scipy.spatial import Voronoi
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import os
import time
import sys

sys.path.insert(0, os.path.dirname(__file__))
from vem_3d import isotropic_3d, traction_from_voigt, face_normal_area, polyhedron_volume
from vem_3d_advanced import _merge_vertices, _order_face_vertices, mesh_stats, export_vtk


# ── Species Info ─────────────────────────────────────────────────────────

SPECIES = ['An', 'So', 'Vd', 'Fn', 'Pg']
SPECIES_COLORS_RGB = {
    'An': [0.2, 0.8, 0.2],   # Green
    'So': [1.0, 0.6, 0.0],   # Orange
    'Vd': [0.0, 0.5, 1.0],   # Blue
    'Fn': [0.8, 0.0, 0.8],   # Magenta
    'Pg': [1.0, 0.0, 0.0],   # Red
}
DI_WEIGHTS = np.array([0.0, 0.3, 0.5, 0.7, 1.0])

# Constitutive law parameters
E_MIN = 10.0    # Pa
E_MAX = 1000.0  # Pa
N_HILL = 2
NU = 0.3


# ── Step 1: Synthetic 3D Confocal Z-Stack ────────────────────────────────

def generate_synthetic_3d_confocal(Lx=200.0, Ly=200.0, Lz=50.0,
                                    nx=64, ny=64, nz=20,
                                    n_colonies=200,
                                    condition='dh_baseline', seed=42):
    """
    Generate a synthetic 3D confocal volume with 5 species channels.

    Spatial distribution mimics oral biofilm:
      - An/So: near surface (high z, interface with fluid)
      - Fn: bridging species, mid-depth
      - Pg: deep anaerobic layers (low z)
      - Vd: distributed throughout

    Returns:
      volume: (5, nz, ny, nx) fluorescence per species
      colony_info: list of dicts with colony properties
    """
    rng = np.random.default_rng(seed)
    volume = np.zeros((5, nz, ny, nx))
    dx, dy, dz = Lx / nx, Ly / ny, Lz / nz

    colony_info = []

    for _ in range(n_colonies):
        # Random center in physical coords
        cx = rng.uniform(10, Lx - 10)
        cy = rng.uniform(10, Ly - 10)
        cz = rng.uniform(3, Lz - 3)

        # Colony radius (um)
        r_xy = rng.uniform(5, 18)
        r_z = rng.uniform(2, 8)  # flatter in z (confocal resolution)

        # Depth fraction: 0 = bottom (substratum), 1 = top (fluid interface)
        depth_frac = cz / Lz

        # Species weights depend on depth and condition
        if condition == 'commensal_static':
            base = np.array([0.45, 0.30, 0.15, 0.08, 0.02])
            if depth_frac > 0.7:
                base = np.array([0.55, 0.25, 0.10, 0.07, 0.03])
            elif depth_frac < 0.3:
                base = np.array([0.30, 0.35, 0.20, 0.10, 0.05])
        elif condition == 'dysbiotic_static':
            base = np.array([0.08, 0.20, 0.18, 0.28, 0.26])
            if depth_frac < 0.3:
                base = np.array([0.04, 0.12, 0.14, 0.35, 0.35])
            elif depth_frac > 0.7:
                base = np.array([0.18, 0.35, 0.25, 0.15, 0.07])
        else:  # dh_baseline
            base = np.array([0.22, 0.28, 0.22, 0.16, 0.12])
            if depth_frac < 0.3:
                base = np.array([0.12, 0.18, 0.20, 0.28, 0.22])
            elif depth_frac > 0.7:
                base = np.array([0.35, 0.30, 0.18, 0.12, 0.05])

        # Add stochasticity
        base += rng.uniform(0, 0.06, 5)
        base /= base.sum()

        dominant = int(np.argmax(base))

        # Pixel indices of colony center
        ix = int(cx / dx)
        iy = int(cy / dy)
        iz = int(cz / dz)
        rx = int(r_xy / dx) + 1
        ry = int(r_xy / dy) + 1
        rz = int(r_z / dz) + 1

        # Paint Gaussian blob into each channel
        for sp in range(5):
            if base[sp] < 0.03:
                continue
            for dk in range(-rz * 2, rz * 2 + 1):
                zk = iz + dk
                if zk < 0 or zk >= nz:
                    continue
                for dj in range(-ry * 2, ry * 2 + 1):
                    yj = iy + dj
                    if yj < 0 or yj >= ny:
                        continue
                    for di in range(-rx * 2, rx * 2 + 1):
                        xi = ix + di
                        if xi < 0 or xi >= nx:
                            continue
                        dist2 = ((di * dx)**2 + (dj * dy)**2) / r_xy**2 + \
                                (dk * dz)**2 / r_z**2
                        if dist2 > 9.0:  # 3-sigma cutoff
                            continue
                        val = base[sp] * np.exp(-0.5 * dist2)
                        volume[sp, zk, yj, xi] += val

        colony_info.append({
            'center_um': np.array([cx, cy, cz]),
            'center_vox': np.array([ix, iy, iz]),
            'radius_xy': r_xy,
            'radius_z': r_z,
            'weights': base.copy(),
            'dominant': dominant,
        })

    # Add Poisson-like noise + PSF smoothing
    for ch in range(5):
        volume[ch] += rng.normal(0, 0.015, (nz, ny, nx))
        volume[ch] = np.clip(volume[ch], 0, None)
        volume[ch] = gaussian_filter(volume[ch], sigma=[0.8, 1.2, 1.2])

    return volume, colony_info


# ── Step 2: 3D Colony Detection ──────────────────────────────────────────

def detect_colonies_3d(volume, Lx=200.0, Ly=200.0, Lz=50.0,
                       merge_radius_um=12.0):
    """
    Detect colony centers from 3D multi-channel volume.

    Uses per-channel local maximum detection + merge nearby peaks.

    Returns:
      centers_um: (N, 3) physical coordinates
      species_fracs: (N, 5) species composition per colony
    """
    n_ch, nz, ny, nx = volume.shape
    dx, dy, dz = Lx / nx, Ly / ny, Lz / nz

    all_peaks = []  # (x_um, y_um, z_um)
    peak_channels = []

    for ch in range(n_ch):
        ch_data = volume[ch]
        ch_max = ch_data.max()
        if ch_max < 0.01:
            continue

        # 3D local maximum detection
        local_max = maximum_filter(ch_data, size=5)
        threshold = ch_max * 0.12
        peaks = (ch_data == local_max) & (ch_data > threshold)

        zs, ys, xs = np.where(peaks)
        for z, y, x in zip(zs, ys, xs):
            all_peaks.append([x * dx, y * dy, z * dz])
            peak_channels.append(ch)

    if len(all_peaks) == 0:
        return np.zeros((0, 3)), np.zeros((0, 5))

    all_peaks = np.array(all_peaks)

    # Merge nearby peaks
    used = np.zeros(len(all_peaks), dtype=bool)
    merged = []

    for i in range(len(all_peaks)):
        if used[i]:
            continue
        cluster = [i]
        for j in range(i + 1, len(all_peaks)):
            if used[j]:
                continue
            if np.linalg.norm(all_peaks[i] - all_peaks[j]) < merge_radius_um:
                cluster.append(j)
                used[j] = True
        used[i] = True
        pts = all_peaks[cluster]
        merged.append(pts.mean(axis=0))

    centers_um = np.array(merged)

    # Compute species fractions from local neighborhood
    species_fracs = []
    r_vox_xy = max(1, int(6.0 / dx))
    r_vox_z = max(1, int(3.0 / dz))

    for cx, cy, cz in centers_um:
        ix = int(cx / dx)
        iy = int(cy / dy)
        iz = int(cz / dz)
        fracs = np.zeros(5)

        for dk in range(-r_vox_z, r_vox_z + 1):
            zk = iz + dk
            if zk < 0 or zk >= nz:
                continue
            for dj in range(-r_vox_xy, r_vox_xy + 1):
                yj = iy + dj
                if yj < 0 or yj >= ny:
                    continue
                for di in range(-r_vox_xy, r_vox_xy + 1):
                    xi = ix + di
                    if xi < 0 or xi >= nx:
                        continue
                    if di**2 + dj**2 + dk**2 > (r_vox_xy + r_vox_z)**2:
                        continue
                    for ch in range(5):
                        fracs[ch] += volume[ch, zk, yj, xi]

        if fracs.sum() > 1e-10:
            fracs /= fracs.sum()
        else:
            fracs = np.ones(5) / 5
        species_fracs.append(fracs)

    return centers_um, np.array(species_fracs)


# ── Step 3: 3D Voronoi Mesh from Colony Centers ─────────────────────────

def build_voronoi_mesh_3d(centers_um, Lx=200.0, Ly=200.0, Lz=50.0):
    """
    Build a 3D Voronoi polyhedral mesh from colony centers.

    Uses mirror points for clean boundary clipping.

    Returns:
      vertices, cells, cell_faces, seed_to_cell map
    """
    n_seeds = len(centers_um)

    # Normalize to [0,1]^3 for mesh generation, then scale back
    pts_norm = centers_um.copy()
    pts_norm[:, 0] /= Lx
    pts_norm[:, 1] /= Ly
    pts_norm[:, 2] /= Lz

    # Clamp to safe interior
    pts_norm = np.clip(pts_norm, 0.02, 0.98)

    # Mirror across 6 faces
    all_pts = [pts_norm]
    for axis in range(3):
        for val in [0.0, 1.0]:
            mirror = pts_norm.copy()
            mirror[:, axis] = 2 * val - mirror[:, axis]
            all_pts.append(mirror)
    all_pts = np.vstack(all_pts)

    vor = Voronoi(all_pts)
    raw_verts = vor.vertices.copy()

    # Build faces per original seed
    seed_faces = {i: [] for i in range(n_seeds)}
    for ridge_idx, (p1, p2) in enumerate(vor.ridge_points):
        fv = vor.ridge_vertices[ridge_idx]
        if -1 in fv:
            continue
        if p1 < n_seeds:
            seed_faces[p1].append(np.array(fv))
        if p2 < n_seeds:
            seed_faces[p2].append(np.array(fv))

    # Clip to [0,1]^3
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

    # Scale vertices to physical coords
    phys_verts = unique_verts.copy()
    phys_verts[:, 0] *= Lx
    phys_verts[:, 1] *= Ly
    phys_verts[:, 2] *= Lz

    return phys_verts, cells, cell_faces_list, seed_to_cell


# ── Step 4: Material Assignment ──────────────────────────────────────────

def compute_DI(phi):
    """Pathogenicity-weighted Dysbiosis Index."""
    return float(np.clip(np.dot(DI_WEIGHTS, phi), 0.0, 1.0))


def compute_E(DI):
    """E(DI) constitutive law: E_min + (E_max - E_min) * (1 - DI)^n."""
    return E_MIN + (E_MAX - E_MIN) * (1.0 - DI) ** N_HILL


# ── Step 5: 3D VEM Solve (sparse, scaled for um-Pa) ─────────────────────

def vem_3d_solve(vertices, cells, cell_faces, E_field, nu,
                 bc_fixed_dofs, bc_vals,
                 load_dofs=None, load_vals=None):
    """
    3D VEM elasticity solver with sparse assembly.

    Same algorithm as vem_3d_advanced.vem_3d_sparse but kept self-contained
    for the pipeline.
    """
    n_nodes = len(vertices)
    n_dofs = 3 * n_nodes
    n_polys = 12

    rows_list = []
    cols_list = []
    vals_list = []
    F_global = np.zeros(n_dofs)

    strain_ids = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 2, 0, 0],
        [0, 0, 0, 0, 2, 0],
        [0, 0, 0, 0, 0, 2],
    ], dtype=float)

    skipped = 0
    for el_id in range(len(cells)):
        vert_ids = cells[el_id].astype(int)
        coords = vertices[vert_ids]
        faces = cell_faces[el_id]
        n_v = len(vert_ids)
        n_el = 3 * n_v

        vmap = {int(g): loc for loc, g in enumerate(vert_ids)}

        E_el = E_field[el_id] if hasattr(E_field, '__len__') else E_field
        C = isotropic_3d(E_el, nu)

        centroid = coords.mean(axis=0)
        h = max(np.linalg.norm(coords[i] - coords[j])
                for i in range(n_v) for j in range(i + 1, n_v))
        if h < 1e-15:
            skipped += 1
            continue

        vol = polyhedron_volume(vertices, faces)
        if vol < 1e-20:
            skipped += 1
            continue

        xc, yc, zc = centroid

        # D matrix
        D = np.zeros((n_el, n_polys))
        for i in range(n_v):
            dx_h = (coords[i, 0] - xc) / h
            dy_h = (coords[i, 1] - yc) / h
            dz_h = (coords[i, 2] - zc) / h
            r = 3 * i
            D[r,   :] = [1, 0, 0, 0,    dz_h, -dy_h, dx_h, 0,    0,    0,    dz_h, dy_h]
            D[r+1, :] = [0, 1, 0, -dz_h, 0,    dx_h,  0,    dy_h, 0,    dz_h, 0,    dx_h]
            D[r+2, :] = [0, 0, 1, dy_h, -dx_h, 0,     0,    0,    dz_h, dy_h, dx_h, 0]

        # B matrix
        B = np.zeros((n_polys, n_el))
        for i in range(n_v):
            B[0, 3*i]     = 1.0 / n_v
            B[1, 3*i + 1] = 1.0 / n_v
            B[2, 3*i + 2] = 1.0 / n_v

        for face in faces:
            face_int = face.astype(int)
            pts = vertices[face_int]
            n_f, A_f = face_normal_area(pts)
            fc = pts.mean(axis=0)
            if np.dot(n_f, fc - centroid) < 0:
                n_f = -n_f
            k_f = len(face_int)

            for gv in face_int:
                if gv not in vmap:
                    continue
                li = vmap[gv]
                w = A_f / k_f

                wrot = w / (2.0 * vol)
                B[3, 3*li + 1] += -wrot * n_f[2]
                B[3, 3*li + 2] +=  wrot * n_f[1]
                B[4, 3*li + 0] +=  wrot * n_f[2]
                B[4, 3*li + 2] += -wrot * n_f[0]
                B[5, 3*li + 0] += -wrot * n_f[1]
                B[5, 3*li + 1] +=  wrot * n_f[0]

                for alpha in range(6):
                    eps_a = strain_ids[alpha] / h
                    sigma_a = C @ eps_a
                    t_f = traction_from_voigt(sigma_a, n_f)
                    B[6 + alpha, 3*li + 0] += w * t_f[0]
                    B[6 + alpha, 3*li + 1] += w * t_f[1]
                    B[6 + alpha, 3*li + 2] += w * t_f[2]

        G = B @ D
        try:
            projector = np.linalg.solve(G, B)
        except np.linalg.LinAlgError:
            skipped += 1
            continue

        G_tilde = G.copy()
        G_tilde[:6, :] = 0.0
        K_cons = projector.T @ G_tilde @ projector

        I_minus_PiD = np.eye(n_el) - D @ projector
        trace_k = np.trace(K_cons)
        stab_param = trace_k / n_el if trace_k > 0 else E_el
        K_stab = stab_param * (I_minus_PiD.T @ I_minus_PiD)

        K_local = K_cons + K_stab

        gdofs = np.zeros(n_el, dtype=int)
        for i in range(n_v):
            gdofs[3*i]     = 3 * vert_ids[i]
            gdofs[3*i + 1] = 3 * vert_ids[i] + 1
            gdofs[3*i + 2] = 3 * vert_ids[i] + 2

        gi, gj = np.meshgrid(gdofs, gdofs, indexing='ij')
        rows_list.append(gi.ravel())
        cols_list.append(gj.ravel())
        vals_list.append(K_local.ravel())

    if skipped > 0:
        print(f"    (skipped {skipped} degenerate cells)")

    rows = np.concatenate(rows_list)
    cols = np.concatenate(cols_list)
    vals = np.concatenate(vals_list)
    K_global = sparse.coo_matrix((vals, (rows, cols)),
                                 shape=(n_dofs, n_dofs)).tocsr()

    if load_dofs is not None and load_vals is not None:
        F_global[load_dofs] += load_vals

    u = np.zeros(n_dofs)
    bc_set = set(bc_fixed_dofs.tolist())
    internal = np.array([i for i in range(n_dofs) if i not in bc_set])

    u[bc_fixed_dofs] = bc_vals
    F_global -= K_global[:, bc_fixed_dofs].toarray() @ bc_vals

    K_ii = K_global[np.ix_(internal, internal)]

    try:
        u[internal] = spsolve(K_ii, F_global[internal])
        if np.any(np.isnan(u)):
            raise ValueError("NaN in solution")
    except Exception:
        reg = 1e-10 * sparse.eye(K_ii.shape[0])
        u[internal] = spsolve(K_ii + reg, F_global[internal])

    return u


# ── Visualization ────────────────────────────────────────────────────────

def plot_cross_sections(vertices, cells, cell_faces, DI_per_el, E_per_el,
                        u, Lx, Ly, Lz, save_dir, condition):
    """Generate cross-section plots (xy, xz, yz) of fields."""
    ux = u[0::3]
    uy = u[1::3]
    uz = u[2::3]
    u_mag = np.sqrt(ux**2 + uy**2 + uz**2)

    # Cell centroids
    centroids = np.zeros((len(cells), 3))
    u_mag_cell = np.zeros(len(cells))
    for i, cell in enumerate(cells):
        ci = cell.astype(int)
        centroids[i] = vertices[ci].mean(axis=0)
        u_mag_cell[i] = np.mean(u_mag[ci])

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # Row 1: DI, E, |u| in XY plane (mid-z slice)
    z_mid = Lz / 2
    z_tol = Lz * 0.25
    xy_mask = np.abs(centroids[:, 2] - z_mid) < z_tol

    fields = [
        (DI_per_el, 'RdYlGn_r', 'DI', 'Dysbiosis Index (XY, z-mid)'),
        (E_per_el, 'viridis', 'E [Pa]', "Young's Modulus (XY, z-mid)"),
        (u_mag_cell, 'hot_r', '|u| [um]', 'Displacement (XY, z-mid)'),
    ]

    for col, (data, cmap, label, title) in enumerate(fields):
        ax = axes[0, col]
        sel = np.where(xy_mask)[0]
        if len(sel) > 0:
            sc = ax.scatter(centroids[sel, 0], centroids[sel, 1],
                           c=data[sel], cmap=cmap, s=60, edgecolors='k',
                           linewidth=0.3)
            fig.colorbar(sc, ax=ax, label=label, shrink=0.8)
        ax.set_xlim(0, Lx)
        ax.set_ylim(0, Ly)
        ax.set_xlabel('x [um]')
        ax.set_ylabel('y [um]')
        ax.set_title(title)
        ax.set_aspect('equal')

    # Row 2: XZ and YZ cross-sections + 3D scatter
    # XZ slice at y-mid
    y_mid = Ly / 2
    y_tol = Ly * 0.25
    xz_mask = np.abs(centroids[:, 1] - y_mid) < y_tol

    ax_xz = axes[1, 0]
    sel = np.where(xz_mask)[0]
    if len(sel) > 0:
        sc = ax_xz.scatter(centroids[sel, 0], centroids[sel, 2],
                          c=DI_per_el[sel], cmap='RdYlGn_r', s=60,
                          edgecolors='k', linewidth=0.3)
        fig.colorbar(sc, ax=ax_xz, label='DI', shrink=0.8)
    ax_xz.set_xlim(0, Lx)
    ax_xz.set_ylim(0, Lz)
    ax_xz.set_xlabel('x [um]')
    ax_xz.set_ylabel('z [um]')
    ax_xz.set_title('DI (XZ, y-mid)')
    ax_xz.set_aspect('auto')

    # YZ slice at x-mid
    x_mid = Lx / 2
    x_tol = Lx * 0.25
    yz_mask = np.abs(centroids[:, 0] - x_mid) < x_tol

    ax_yz = axes[1, 1]
    sel = np.where(yz_mask)[0]
    if len(sel) > 0:
        sc = ax_yz.scatter(centroids[sel, 1], centroids[sel, 2],
                          c=E_per_el[sel], cmap='viridis', s=60,
                          edgecolors='k', linewidth=0.3)
        fig.colorbar(sc, ax=ax_yz, label='E [Pa]', shrink=0.8)
    ax_yz.set_xlim(0, Ly)
    ax_yz.set_ylim(0, Lz)
    ax_yz.set_xlabel('y [um]')
    ax_yz.set_ylabel('z [um]')
    ax_yz.set_title('E(DI) (YZ, x-mid)')
    ax_yz.set_aspect('auto')

    # 3D scatter of all cell centroids colored by displacement
    ax3d = fig.add_subplot(2, 3, 6, projection='3d')
    sc3 = ax3d.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
                       c=u_mag_cell, cmap='hot_r', s=30, edgecolors='k',
                       linewidth=0.2, alpha=0.8)
    ax3d.set_xlabel('x [um]')
    ax3d.set_ylabel('y [um]')
    ax3d.set_zlabel('z [um]')
    ax3d.set_title('3D Displacement')
    fig.colorbar(sc3, ax=ax3d, label='|u| [um]', shrink=0.6)

    fig.suptitle(f'3D Confocal -> VEM: {condition}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(save_dir, f'vem_3d_confocal_{condition}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close()
    return path


def plot_3d_di_scatter(centers_um, species_fracs, save_dir, condition):
    """3D scatter plot of colony positions colored by DI."""
    DI_vals = np.array([compute_DI(phi) for phi in species_fracs])
    dominant = np.argmax(species_fracs, axis=1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7),
                                    subplot_kw={'projection': '3d'})

    # Left: DI
    sc1 = ax1.scatter(centers_um[:, 0], centers_um[:, 1], centers_um[:, 2],
                      c=DI_vals, cmap='RdYlGn_r', s=40, edgecolors='k',
                      linewidth=0.3, vmin=0, vmax=1)
    ax1.set_xlabel('x [um]')
    ax1.set_ylabel('y [um]')
    ax1.set_zlabel('z [um]')
    ax1.set_title(f'Colony DI ({condition})')
    fig.colorbar(sc1, ax=ax1, label='DI', shrink=0.6)

    # Right: Dominant species
    colors_map = plt.cm.Set1(np.linspace(0, 1, 5))
    sp_colors = colors_map[dominant]
    ax2.scatter(centers_um[:, 0], centers_um[:, 1], centers_um[:, 2],
                c=sp_colors, s=40, edgecolors='k', linewidth=0.3)
    # Legend
    for sp_i, sp_name in enumerate(SPECIES):
        ax2.scatter([], [], [], c=[colors_map[sp_i]], label=sp_name, s=40)
    ax2.legend(loc='upper left', fontsize=8)
    ax2.set_xlabel('x [um]')
    ax2.set_ylabel('y [um]')
    ax2.set_zlabel('z [um]')
    ax2.set_title(f'Dominant Species ({condition})')

    fig.suptitle('3D Colony Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(save_dir, f'vem_3d_confocal_colonies_{condition}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close()
    return path


# ── Full Pipeline ────────────────────────────────────────────────────────

def run_3d_pipeline(condition='dh_baseline', save_dir='results/confocal_3d',
                    n_colonies=150, seed=42):
    """
    Complete 3D pipeline: synthetic confocal z-stack -> VEM.
    """
    Lx, Ly, Lz = 200.0, 200.0, 50.0
    nx_vol, ny_vol, nz_vol = 64, 64, 20

    timings = {}

    print(f"\n{'='*65}")
    print(f"  3D Confocal -> VEM Pipeline: {condition}")
    print(f"{'='*65}")

    # ── Step 1: Synthetic confocal ──
    print("\n  [Step 1] Generating synthetic 3D confocal z-stack...")
    t0 = time.time()
    volume, colony_info = generate_synthetic_3d_confocal(
        Lx=Lx, Ly=Ly, Lz=Lz, nx=nx_vol, ny=ny_vol, nz=nz_vol,
        n_colonies=n_colonies, condition=condition, seed=seed)
    timings['1_confocal'] = time.time() - t0
    print(f"    Volume: ({nx_vol}, {ny_vol}, {nz_vol}) voxels, "
          f"5 channels, {n_colonies} colonies")
    print(f"    Time: {timings['1_confocal']:.2f}s")

    # ── Step 2: Colony detection ──
    print("\n  [Step 2] Detecting 3D colonies...")
    t0 = time.time()
    centers_um, species_fracs = detect_colonies_3d(
        volume, Lx=Lx, Ly=Ly, Lz=Lz, merge_radius_um=10.0)
    timings['2_detection'] = time.time() - t0
    print(f"    Detected: {len(centers_um)} colonies")
    print(f"    Time: {timings['2_detection']:.2f}s")

    if len(centers_um) < 10:
        print("  ERROR: Too few colonies detected, aborting.")
        return None

    # Plot 3D colony scatter
    plot_3d_di_scatter(centers_um, species_fracs, save_dir, condition)

    # ── Step 3: 3D Voronoi mesh ──
    print("\n  [Step 3] Building 3D Voronoi polyhedral mesh...")
    t0 = time.time()
    vertices, cells, cell_faces, seed_to_cell = build_voronoi_mesh_3d(
        centers_um, Lx=Lx, Ly=Ly, Lz=Lz)
    timings['3_mesh'] = time.time() - t0
    print(f"    Mesh statistics:")
    mesh_stats(vertices, cells, cell_faces)
    print(f"    Mapped seeds: {len(seed_to_cell)}/{len(centers_um)}")
    print(f"    Time: {timings['3_mesh']:.2f}s")

    # ── Step 4: Material assignment ──
    print("\n  [Step 4] Computing DI and E(DI)...")
    t0 = time.time()
    n_cells = len(cells)
    DI_per_el = np.full(n_cells, 0.5)
    E_per_el = np.full(n_cells, compute_E(0.5))
    phi_per_el = np.zeros((n_cells, 5))

    for seed_idx, cell_idx in seed_to_cell.items():
        if seed_idx < len(species_fracs):
            phi = species_fracs[seed_idx]
            phi_per_el[cell_idx] = phi
            DI_per_el[cell_idx] = compute_DI(phi)
            E_per_el[cell_idx] = compute_E(DI_per_el[cell_idx])

    timings['4_material'] = time.time() - t0
    print(f"    DI range: [{DI_per_el.min():.3f}, {DI_per_el.max():.3f}]")
    print(f"    E  range: [{E_per_el.min():.0f}, {E_per_el.max():.0f}] Pa")
    print(f"    Time: {timings['4_material']:.4f}s")

    # ── Step 5: VEM solve ──
    print("\n  [Step 5] Solving 3D VEM elasticity...")
    t0 = time.time()

    # Identify boundary nodes used by cells
    used_nodes = set()
    for cell in cells:
        used_nodes.update(cell.astype(int).tolist())
    used_nodes = np.array(sorted(used_nodes))

    z_min = vertices[used_nodes, 2].min()
    z_max = vertices[used_nodes, 2].max()
    z_range = z_max - z_min

    # Fixed bottom (substratum)
    tol_bot = z_min + 0.05 * z_range
    bottom = used_nodes[vertices[used_nodes, 2] < tol_bot]
    bc_dofs = np.concatenate([3*bottom, 3*bottom+1, 3*bottom+2])
    bc_vals = np.zeros(len(bc_dofs))

    # GCF pressure on top surface (z-direction, downward)
    tol_top = z_max - 0.05 * z_range
    top = used_nodes[vertices[used_nodes, 2] > tol_top]

    # Pressure = 1 Pa total distributed over top nodes
    p_per_node = -1.0 / max(len(top), 1)
    load_dofs = 3 * top + 2  # z-DOFs
    load_vals = np.full(len(top), p_per_node)

    print(f"    Fixed (bottom): {len(bottom)} nodes")
    print(f"    Loaded (top):   {len(top)} nodes")

    u = vem_3d_solve(vertices, cells, cell_faces, E_per_el, NU,
                     bc_dofs, bc_vals, load_dofs, load_vals)
    timings['5_solve'] = time.time() - t0

    ux = u[0::3]
    uy = u[1::3]
    uz = u[2::3]
    u_mag = np.sqrt(ux**2 + uy**2 + uz**2)
    u_max = np.max(u_mag[used_nodes]) if len(used_nodes) > 0 else 0
    print(f"    Max |u|: {u_max:.6e} um")
    print(f"    Time: {timings['5_solve']:.2f}s")

    # ── Visualization ──
    print("\n  [Step 6] Generating visualizations...")
    t0 = time.time()
    plot_cross_sections(vertices, cells, cell_faces, DI_per_el, E_per_el,
                        u, Lx, Ly, Lz, save_dir, condition)

    # VTK export
    disp_vec = np.column_stack([ux, uy, uz])
    vtk_path = os.path.join(save_dir, f'vem_3d_confocal_{condition}.vtk')
    export_vtk(vtk_path, vertices, cells, cell_faces,
               point_data={'displacement': disp_vec, 'u_magnitude': u_mag},
               cell_data={'DI': DI_per_el, 'E_modulus': E_per_el})
    timings['6_viz'] = time.time() - t0

    total_time = sum(timings.values())
    timings['total'] = total_time

    return {
        'vertices': vertices,
        'cells': cells,
        'cell_faces': cell_faces,
        'DI': DI_per_el,
        'E': E_per_el,
        'u': u,
        'phi': phi_per_el,
        'centers': centers_um,
        'species_fracs': species_fracs,
        'timings': timings,
        'n_colonies_detected': len(centers_um),
        'condition': condition,
    }


# ── Pipeline Comparison Table ────────────────────────────────────────────

def print_comparison_table(results_list):
    """Print pipeline comparison: Traditional FEM (5-step) vs VEM (2-step)."""
    print("\n" + "=" * 75)
    print("  PIPELINE COMPARISON: Traditional FEM vs VEM")
    print("=" * 75)

    print("\n  Traditional Pipeline (5 steps):")
    print("  " + "-" * 55)
    print("    1. Confocal z-stack acquisition")
    print("    2. Voxel segmentation (Otsu/watershed)")
    print("    3. Surface extraction (marching cubes)")
    print("    4. Tetrahedral meshing (TetGen/Gmsh)")
    print("    5. FEM solve (Abaqus C3D4)")
    print("  Typical time: ~hours (manual steps + commercial software)")
    print("  Mesh conversion: lossy (surface smoothing, remeshing)")
    print("  Species info: lost at step 3 (geometry only)")

    print("\n  VEM Pipeline (2 steps):")
    print("  " + "-" * 55)
    print("    1. Colony detection from z-stack -> Voronoi seeds")
    print("    2. Voronoi polyhedral VEM (direct solve)")
    print("  Advantages:")
    print("    - No mesh conversion (colony = element)")
    print("    - Species composition preserved per element")
    print("    - Arbitrary polyhedra (no tet quality issues)")
    print("    - Fully automated, no manual intervention")

    if results_list:
        r0 = results_list[0]
        t = r0['timings']
        print(f"\n  VEM Pipeline Timing ({r0['condition']}):")
        print("  " + "-" * 55)
        print(f"    Step 1 (confocal gen):   {t['1_confocal']:>8.2f} s")
        print(f"    Step 2 (detection):      {t['2_detection']:>8.2f} s")
        print(f"    Step 3 (Voronoi mesh):   {t['3_mesh']:>8.2f} s")
        print(f"    Step 4 (material):       {t['4_material']:>8.4f} s")
        print(f"    Step 5 (VEM solve):      {t['5_solve']:>8.2f} s")
        print(f"    Step 6 (visualization):  {t['6_viz']:>8.2f} s")
        print(f"    {'Total':>28s}: {t['total']:>8.2f} s")

    # Multi-condition comparison
    if len(results_list) > 1:
        print(f"\n  {'Condition':<22s} {'Cells':>6s} {'DI_mean':>8s} "
              f"{'E_min':>8s} {'E_max':>8s} {'E_ratio':>8s} {'|u|_max':>10s}")
        print("  " + "-" * 72)
        for r in results_list:
            ratio = r['E'].max() / max(r['E'].min(), 1)
            u_mag = np.sqrt(r['u'][0::3]**2 + r['u'][1::3]**2 + r['u'][2::3]**2)
            print(f"  {r['condition']:<22s} {len(r['cells']):>6d} "
                  f"{r['DI'].mean():>8.3f} {r['E'].min():>8.0f} "
                  f"{r['E'].max():>8.0f} {ratio:>8.1f}x {u_mag.max():>10.2e}")

    print("=" * 75)


# ── Main ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'results', 'confocal_3d')
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 65)
    print("  3D Confocal -> VEM Pipeline")
    print("  Domain: 200 x 200 x 50 um (oral biofilm)")
    print("  Species: An, So, Vd, Fn, Pg")
    print(f"  E(DI) = {E_MIN} + ({E_MAX}-{E_MIN}) * (1-DI)^{N_HILL}")
    print(f"  nu = {NU}")
    print("=" * 65)

    conditions = ['commensal_static', 'dh_baseline', 'dysbiotic_static']
    results_all = []

    for cond in conditions:
        res = run_3d_pipeline(condition=cond, save_dir=save_dir,
                              n_colonies=150, seed=42)
        if res is not None:
            results_all.append(res)

    # Pipeline comparison
    print_comparison_table(results_all)

    print(f"\n  All outputs saved to: {save_dir}/")
    print("  3D Confocal -> VEM Pipeline complete.")
