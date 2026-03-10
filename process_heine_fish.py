"""
Process Heine 2025 FISH confocal images → VEM pipeline.

Extracts 5-species fluorescence channels from composite FISH images
(published in Heine et al. 2025, Fig 3B) and runs the VEM confocal pipeline.

FISH probes (from Table S5):
  An (A. naeslundii): Alexa Fluor 488 → green
  So (S. oralis):     Alexa Fluor 405 → blue/violet (appears blue in composite)
  Vd (V. dispar/parvula): Alexa Fluor 568 → orange/yellow
  Fn (F. nucleatum):  AF405 + AF647 → dual-label (appears cyan/white)
  Pg (P. gingivalis): Alexa Fluor 647 → red/far-red

In the composite images:
  - Blue-dominant pixels → An (488) or So (405)
  - Green pixels → An (488)
  - Yellow/orange → Vd (568)
  - Red → Pg (647)
  - Magenta/white → Fn (dual) or multi-species overlap

Note: PDF-extracted JPEG images have limited resolution (~250x240 px)
and channel bleed-through. This is a best-effort color decomposition.
"""

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, label, center_of_mass
from scipy.spatial import Voronoi
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from vem_elasticity import vem_elasticity


# ── Color decomposition: composite RGB → 5-species channels ──────────────

def decompose_fish_channels(rgb, method='heuristic'):
    """
    Decompose a composite FISH RGB image into 5 pseudo-channels.

    The Heine 2025 FISH images use 5 fluorophores mapped to a single
    RGB composite. We use color-space heuristics to approximate
    the individual channels.

    Parameters:
      rgb: (H, W, 3) uint8 array
      method: 'heuristic' (color-based) or 'nmf' (non-negative matrix factorization)

    Returns:
      channels: (5, H, W) float array, each in [0, 1]
        Index: 0=An, 1=So, 2=Vd, 3=Fn, 4=Pg
    """
    img = rgb.astype(np.float64) / 255.0
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    H, W = R.shape
    channels = np.zeros((5, H, W))

    # Background mask
    brightness = (R + G + B) / 3
    bg_mask = brightness < 0.05

    if method == 'heuristic':
        # Heine 2025 FISH probe → fluorophore → approximate RGB rendering:
        #   So: Alexa 405 (λ_em=421nm) → blue/violet channel
        #   An: Alexa 488 (λ_em=519nm) → green channel
        #   Vd: Alexa 568 (λ_em=603nm) → yellow/orange
        #   Fn: AF405+AF647 dual → cyan/white (both B and R high)
        #   Pg: Alexa 647 (λ_em=668nm) → red/far-red
        #
        # Reference spectra matrix S (3×5) based on fluorophore emission:
        S = np.array([
            # R     G     B     ← RGB contribution
            [0.05, 0.60, 0.10],  # An (488: green)
            [0.10, 0.05, 0.85],  # So (405: blue)
            [0.75, 0.65, 0.05],  # Vd (568: yellow-orange)
            [0.45, 0.30, 0.55],  # Fn (405+647: blue+red → magenta/white)
            [0.90, 0.05, 0.10],  # Pg (647: red)
        ]).T  # → (3, 5)

        # Non-negative least squares per pixel (spectral unmixing)
        from scipy.optimize import nnls
        pixels = np.stack([R.ravel(), G.ravel(), B.ravel()], axis=0)  # (3, N)
        N = pixels.shape[1]
        abundances = np.zeros((5, N))

        # Vectorized: solve S @ h = pixel for each pixel using pseudoinverse + clip
        # (faster than per-pixel NNLS for 250x240 images)
        S_pinv = np.linalg.pinv(S)  # (5, 3)
        abundances = S_pinv @ pixels  # (5, N)
        abundances = np.clip(abundances, 0, None)

        for ch in range(5):
            channels[ch] = abundances[ch].reshape(H, W)
            cmax = channels[ch].max()
            if cmax > 1e-10:
                channels[ch] /= cmax
            channels[ch][bg_mask] = 0
            channels[ch] = gaussian_filter(channels[ch], sigma=1.0)

    elif method == 'nmf':
        from scipy.optimize import nnls

        # Reference spectra (same as heuristic)
        S = np.array([
            [0.05, 0.60, 0.10],  # An
            [0.10, 0.05, 0.85],  # So
            [0.75, 0.65, 0.05],  # Vd
            [0.45, 0.30, 0.55],  # Fn
            [0.90, 0.05, 0.10],  # Pg
        ]).T  # (3, 5)

        pixels_flat = np.stack([R.ravel(), G.ravel(), B.ravel()], axis=0)
        N = pixels_flat.shape[1]

        # Per-pixel NNLS (slower but more accurate)
        abundances = np.zeros((5, N))
        for i in range(N):
            if brightness.ravel()[i] > 0.03:
                abundances[:, i], _ = nnls(S, pixels_flat[:, i])

        for ch in range(5):
            channels[ch] = abundances[ch].reshape(H, W)
            cmax = channels[ch].max()
            if cmax > 1e-10:
                channels[ch] /= cmax
            channels[ch] = gaussian_filter(channels[ch], sigma=1.0)

    return channels


# ── Colony detection from decomposed channels ────────────────────────────

def detect_colonies_from_channels(channels, min_intensity=0.08, merge_radius=8):
    """
    Detect colony centroids from 5-channel pseudo-fluorescence.

    Returns:
      seeds_px: (N, 2) pixel coordinates [x, y]
      species_fracs: (N, 5) species composition per colony
    """
    from scipy.ndimage import maximum_filter

    n_ch, H, W = channels.shape
    all_peaks = []
    peak_ch = []

    for ch in range(n_ch):
        data = channels[ch]
        if data.max() < 0.01:
            continue

        # Local max detection
        local_max = maximum_filter(data, size=6)
        threshold = max(data.max() * 0.2, min_intensity)
        peaks = (data == local_max) & (data > threshold)
        ys, xs = np.where(peaks)

        for y, x in zip(ys, xs):
            all_peaks.append([x, y])
            peak_ch.append(ch)

    if not all_peaks:
        return np.zeros((0, 2)), np.zeros((0, 5))

    all_peaks = np.array(all_peaks, dtype=float)

    # Merge nearby peaks
    merged = []
    used = np.zeros(len(all_peaks), dtype=bool)
    for i in range(len(all_peaks)):
        if used[i]:
            continue
        cluster = [i]
        for j in range(i + 1, len(all_peaks)):
            if used[j]:
                continue
            if np.linalg.norm(all_peaks[i] - all_peaks[j]) < merge_radius:
                cluster.append(j)
                used[j] = True
        used[i] = True
        merged.append(all_peaks[cluster].mean(axis=0))

    seeds = np.array(merged)

    # Species composition per colony (local neighborhood)
    species_fracs = []
    r = 4  # pixel radius for sampling
    for sx, sy in seeds:
        ix, iy = int(sx), int(sy)
        fracs = np.zeros(5)
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                yi, xi = iy + di, ix + dj
                if 0 <= yi < H and 0 <= xi < W and di**2 + dj**2 <= r**2:
                    for ch in range(5):
                        fracs[ch] += channels[ch, yi, xi]
        total = fracs.sum()
        if total > 1e-10:
            fracs /= total
        else:
            fracs = np.ones(5) / 5
        species_fracs.append(fracs)

    return seeds, np.array(species_fracs)


# ── DI and E(DI) computation ─────────────────────────────────────────────

def compute_di(species_fracs):
    """
    Compute Dysbiosis Index per colony.
    DI = 1 - (1/ln(5)) * sum(phi_i * ln(phi_i))  (Shannon entropy normalized)
    Higher DI = more pathogenic.
    """
    n = len(species_fracs)
    di = np.zeros(n)
    for i in range(n):
        phi = species_fracs[i]
        phi = np.clip(phi, 1e-15, None)
        entropy = -np.sum(phi * np.log(phi))
        max_entropy = np.log(5)
        # Low diversity (one species dominates) can be either healthy or pathogenic
        # Use Pg fraction as pathogenic indicator
        pg_frac = phi[4]  # Pg is index 4
        fn_frac = phi[3]  # Fn is index 3
        # DI: pathogenic species weighted
        di[i] = 0.5 * (1.0 - entropy / max_entropy) + 0.3 * pg_frac + 0.2 * fn_frac
    return np.clip(di, 0, 1)


def compute_E_from_di(di, E_max=1000.0, E_min=10.0, n_exp=2):
    """E(DI) = E_min + (E_max - E_min) * (1 - DI)^n"""
    return E_min + (E_max - E_min) * (1 - di) ** n_exp


# ── Voronoi mesh generation ──────────────────────────────────────────────

def build_voronoi_mesh(seeds_px, img_shape, scale_um=25.0, scale_px=None):
    """
    Build a VEM-compatible Voronoi mesh from colony centroids.

    Parameters:
      seeds_px: (N, 2) pixel coordinates
      img_shape: (H, W) of image
      scale_um: physical scale (µm) corresponding to scale_px pixels
      scale_px: scale bar length in pixels (auto-detected if None)

    Returns:
      vertices, elements, boundary_nodes, px_to_um scale factor
    """
    H, W = img_shape

    # Scale: assume 25 µm scale bar is ~50 px (from the image)
    if scale_px is None:
        scale_px = 50.0  # approximate from image
    um_per_px = scale_um / scale_px

    # Physical domain
    Lx = W * um_per_px
    Ly = H * um_per_px

    seeds_phys = seeds_px.copy().astype(float)
    seeds_phys[:, 0] *= um_per_px
    seeds_phys[:, 1] *= um_per_px

    n_seeds = len(seeds_phys)

    # Mirror seeds for bounded Voronoi
    all_pts = [seeds_phys]
    for axis, bounds in [(0, [0.0, Lx]), (1, [0.0, Ly])]:
        for val in bounds:
            mirror = seeds_phys.copy()
            mirror[:, axis] = 2 * val - mirror[:, axis]
            all_pts.append(mirror)
    all_pts = np.vstack(all_pts)

    vor = Voronoi(all_pts)

    # Extract bounded cells for original seeds only
    vertices_list = list(vor.vertices)
    elements = []
    valid_seeds = []

    for i in range(n_seeds):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]

        if -1 in region or len(region) < 3:
            continue

        verts = vor.vertices[region]

        # Clip to domain
        if (np.any(verts[:, 0] < -0.1 * Lx) or np.any(verts[:, 0] > 1.1 * Lx) or
            np.any(verts[:, 1] < -0.1 * Ly) or np.any(verts[:, 1] > 1.1 * Ly)):
            continue

        verts[:, 0] = np.clip(verts[:, 0], 0, Lx)
        verts[:, 1] = np.clip(verts[:, 1], 0, Ly)

        elements.append(np.array(region))
        valid_seeds.append(i)

    if not elements:
        return None, None, None, um_per_px, None

    # Reindex: only keep vertices actually used by elements
    used_set = set()
    for el in elements:
        for vi in el:
            used_set.add(vi)
    used_sorted = sorted(used_set)
    old_to_new = {old: new for new, old in enumerate(used_sorted)}

    vertices_compact = vor.vertices[used_sorted]
    elements_compact = []
    for el in elements:
        elements_compact.append(np.array([old_to_new[vi] for vi in el]))

    # Boundary nodes (on domain edges)
    tol = 1e-6
    boundary = set()
    for el in elements_compact:
        for vi in el:
            v = vertices_compact[vi]
            if v[0] < tol or v[0] > Lx - tol or v[1] < tol or v[1] > Ly - tol:
                boundary.add(vi)
    boundary_nodes = np.array(sorted(boundary)) if boundary else np.array([], dtype=int)

    return vertices_compact, elements_compact, boundary_nodes, um_per_px, valid_seeds


# ── Full pipeline: FISH image → VEM solution ─────────────────────────────

def process_fish_image(image_path, condition_name='unknown',
                       scale_um=25.0, E_max=1000.0, E_min=10.0,
                       output_dir=None):
    """
    Full pipeline: FISH image → color decomposition → colony detection →
    Voronoi mesh → DI → E(DI) → VEM elasticity.

    Returns dict with all intermediate and final results.
    """
    print(f"\n{'='*60}")
    print(f"Processing: {condition_name}")
    print(f"Image: {image_path}")
    print(f"{'='*60}")

    # Load image
    img = np.array(Image.open(image_path))
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    H, W = img.shape[:2]
    print(f"  Image size: {W}x{H} px")

    # Step 1: Color decomposition
    print("  Step 1: Decomposing fluorescence channels...")
    channels = decompose_fish_channels(img[:, :, :3])
    for i, sp in enumerate(['An', 'So', 'Vd', 'Fn', 'Pg']):
        intensity = channels[i].sum()
        print(f"    {sp}: total intensity = {intensity:.1f}, "
              f"max = {channels[i].max():.3f}")

    # Step 2: Colony detection
    print("  Step 2: Detecting colonies...")
    seeds, species_fracs = detect_colonies_from_channels(channels)
    n_colonies = len(seeds)
    print(f"    Found {n_colonies} colonies")

    if n_colonies < 3:
        print("  WARNING: Too few colonies detected. Skipping VEM.")
        return {'condition': condition_name, 'n_colonies': n_colonies,
                'channels': channels, 'error': 'too_few_colonies'}

    # Step 3: DI and E(DI)
    print("  Step 3: Computing DI and E(DI)...")
    di = compute_di(species_fracs)
    E_vals = compute_E_from_di(di, E_max, E_min)
    print(f"    DI: mean={di.mean():.3f}, range=[{di.min():.3f}, {di.max():.3f}]")
    print(f"    E:  mean={E_vals.mean():.0f} Pa, range=[{E_vals.min():.0f}, {E_vals.max():.0f}] Pa")

    # Mean species composition
    mean_phi = species_fracs.mean(axis=0)
    print(f"    Mean species: An={mean_phi[0]:.2f} So={mean_phi[1]:.2f} "
          f"Vd={mean_phi[2]:.2f} Fn={mean_phi[3]:.2f} Pg={mean_phi[4]:.2f}")

    # Step 4: Build Voronoi mesh
    print("  Step 4: Building Voronoi mesh...")
    result = build_voronoi_mesh(seeds, (H, W), scale_um=scale_um)
    if result[0] is None:
        print("  WARNING: Mesh construction failed.")
        return {'condition': condition_name, 'n_colonies': n_colonies,
                'channels': channels, 'di': di, 'E': E_vals,
                'error': 'mesh_failed'}

    vertices, elements, boundary_nodes, um_per_px, valid_seeds = result
    print(f"    Mesh: {len(vertices)} vertices, {len(elements)} cells")
    print(f"    Scale: {um_per_px:.2f} µm/px")

    # Step 5: VEM elasticity
    print("  Step 5: Solving VEM elasticity...")
    n_nodes = len(vertices)
    nu = 0.3

    # Per-element E values
    E_per_cell = E_vals[valid_seeds] if len(valid_seeds) <= len(E_vals) else \
        np.full(len(elements), E_vals.mean())



    # Boundary conditions: fixed bottom, pressure top
    # Use percentile-based selection since Voronoi vertices don't land
    # exactly on domain boundaries
    Ly = H * um_per_px
    Lx = W * um_per_px

    # Only use vertices that are part of elements
    used_verts = set()
    for el in elements:
        for vi in el:
            used_verts.add(vi)
    used_verts = np.array(sorted(used_verts))

    # Bottom 5% of used vertices (by y-coordinate)
    y_vals = vertices[used_verts, 1]
    y_lo = np.percentile(y_vals, 5)
    y_hi = np.percentile(y_vals, 95)

    bottom = used_verts[y_vals <= y_lo]
    top = used_verts[y_vals >= y_hi]

    # Ensure enough BC nodes
    if len(bottom) < 3:
        bottom = used_verts[np.argsort(y_vals)[:max(5, len(used_verts)//20)]]
    if len(top) < 3:
        top = used_verts[np.argsort(y_vals)[-max(5, len(used_verts)//20):]]

    bc_dofs = np.concatenate([2 * bottom, 2 * bottom + 1])
    bc_vals = np.zeros(len(bc_dofs))

    load_dofs = 2 * top + 1  # y-direction
    # GCF pressure ~2 Pa, distributed
    load_vals = np.full(len(top), -2.0 / max(len(top), 1))

    try:
        u = vem_elasticity(vertices, elements, E_per_cell, nu,
                           bc_dofs, bc_vals, load_dofs, load_vals)
        ux = u[0::2]
        uy = u[1::2]
        max_disp = np.sqrt(ux**2 + uy**2).max()
        print(f"    Max displacement: {max_disp:.4f} µm")
        print(f"    E per cell: min={E_per_cell.min():.0f}, max={E_per_cell.max():.0f}, mean={E_per_cell.mean():.0f} Pa")
    except Exception as e:
        print(f"  VEM solve failed: {e}")
        u = None
        max_disp = None

    results = {
        'condition': condition_name,
        'image': img,
        'channels': channels,
        'seeds_px': seeds,
        'species_fracs': species_fracs,
        'di': di,
        'E': E_vals,
        'E_per_cell': E_per_cell,
        'vertices': vertices,
        'elements': elements,
        'boundary_nodes': boundary_nodes,
        'valid_seeds': valid_seeds,
        'um_per_px': um_per_px,
        'u': u,
        'max_disp': max_disp,
        'n_colonies': n_colonies,
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_results(results, output_dir)

    return results


# ── Visualization ────────────────────────────────────────────────────────

def plot_results(results, output_dir):
    """Generate comprehensive visualization of pipeline results."""
    cond = results['condition']
    img = results['image']
    channels = results['channels']
    seeds = results['seeds_px']
    di = results['di']
    E = results['E']
    vertices = results.get('vertices')
    elements = results.get('elements')
    u = results.get('u')
    um_per_px = results.get('um_per_px', 0.5)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Heine 2025 FISH → VEM: {cond}', fontsize=14, fontweight='bold')

    # (0,0) Original image with detected colonies
    axes[0, 0].imshow(img)
    if len(seeds) > 0:
        axes[0, 0].scatter(seeds[:, 0], seeds[:, 1], c='white',
                          s=15, marker='+', linewidths=0.5, alpha=0.8)
    axes[0, 0].set_title(f'FISH image + {len(seeds)} colonies')
    axes[0, 0].axis('off')

    # (0,1) Channel decomposition (composite of An=green, So=blue, Pg=red)
    composite = np.zeros((*channels.shape[1:], 3))
    composite[:, :, 1] = np.clip(channels[0], 0, 1)  # An → green
    composite[:, :, 2] = np.clip(channels[1], 0, 1)  # So → blue
    composite[:, :, 0] = np.clip(channels[4], 0, 1)  # Pg → red
    # Add Vd as yellow
    composite[:, :, 0] += 0.5 * np.clip(channels[2], 0, 1)
    composite[:, :, 1] += 0.5 * np.clip(channels[2], 0, 1)
    composite = np.clip(composite, 0, 1)
    axes[0, 1].imshow(composite)
    axes[0, 1].set_title('Decomposed (R=Pg, G=An, B=So, Y=Vd)')
    axes[0, 1].axis('off')

    # (0,2) Species fractions bar
    if len(seeds) > 0:
        mean_phi = results['species_fracs'].mean(axis=0)
        bars = axes[0, 2].bar(['An', 'So', 'Vd', 'Fn', 'Pg'], mean_phi,
                             color=['#33cc33', '#0088ff', '#ffaa00', '#cc00cc', '#ff0000'])
        axes[0, 2].set_ylabel('Mean fraction')
        axes[0, 2].set_title(f'Species composition (N={len(seeds)})')
        axes[0, 2].set_ylim(0, 1)

    # (1,0) DI map
    if len(seeds) > 0:
        sc = axes[1, 0].scatter(seeds[:, 0], seeds[:, 1], c=di,
                               cmap='RdYlGn_r', vmin=0, vmax=1, s=30)
        plt.colorbar(sc, ax=axes[1, 0], label='DI')
        axes[1, 0].set_xlim(0, img.shape[1])
        axes[1, 0].set_ylim(img.shape[0], 0)
        axes[1, 0].set_title(f'DI map (mean={di.mean():.2f})')
        axes[1, 0].set_aspect('equal')

    # (1,1) E(DI) map
    if len(seeds) > 0:
        sc = axes[1, 1].scatter(seeds[:, 0], seeds[:, 1], c=E,
                               cmap='viridis', s=30)
        plt.colorbar(sc, ax=axes[1, 1], label='E [Pa]')
        axes[1, 1].set_xlim(0, img.shape[1])
        axes[1, 1].set_ylim(img.shape[0], 0)
        axes[1, 1].set_title(f'E(DI) map (mean={E.mean():.0f} Pa)')
        axes[1, 1].set_aspect('equal')

    # (1,2) Displacement field
    if u is not None and vertices is not None:
        ux = u[0::2]
        uy = u[1::2]
        mag = np.sqrt(ux**2 + uy**2)
        sc = axes[1, 2].scatter(vertices[:, 0], vertices[:, 1], c=mag,
                               cmap='hot', s=5, alpha=0.6)
        plt.colorbar(sc, ax=axes[1, 2], label='|u| [µm]')
        axes[1, 2].set_title(f'Displacement (max={mag.max():.4f} µm)')
        axes[1, 2].set_aspect('equal')
        axes[1, 2].invert_yaxis()
    else:
        axes[1, 2].text(0.5, 0.5, 'VEM solve\nnot available',
                       ha='center', va='center', transform=axes[1, 2].transAxes)

    plt.tight_layout()
    fname = os.path.join(output_dir, f'heine_fish_vem_{cond}.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")


def plot_comparison(all_results, output_dir):
    """Compare Commensal vs Dysbiotic across timepoints."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Heine 2025 FISH → VEM: Commensal vs Dysbiotic HOBIC',
                fontsize=14, fontweight='bold')

    commensal = [r for r in all_results if 'commensal' in r['condition']]
    dysbiotic = [r for r in all_results if 'dysbiotic' in r['condition']]

    # Sort by day
    for group in [commensal, dysbiotic]:
        group.sort(key=lambda r: int(r['condition'].split('day')[1]) if 'day' in r['condition'] else 0)

    # Row 0: DI evolution
    days_c = [int(r['condition'].split('day')[1]) for r in commensal if 'day' in r['condition']]
    days_d = [int(r['condition'].split('day')[1]) for r in dysbiotic if 'day' in r['condition']]
    di_c = [r['di'].mean() for r in commensal if r.get('di') is not None and len(r['di']) > 0]
    di_d = [r['di'].mean() for r in dysbiotic if r.get('di') is not None and len(r['di']) > 0]

    if days_c and di_c:
        axes[0, 0].plot(days_c[:len(di_c)], di_c, 'b-o', label='Commensal', linewidth=2)
    if days_d and di_d:
        axes[0, 0].plot(days_d[:len(di_d)], di_d, 'r-s', label='Dysbiotic', linewidth=2)
    axes[0, 0].set_xlabel('Day')
    axes[0, 0].set_ylabel('Mean DI')
    axes[0, 0].set_title('DI evolution')
    axes[0, 0].legend()
    axes[0, 0].set_ylim(0, 1)

    # Row 0: E evolution
    E_c = [r['E'].mean() for r in commensal if r.get('E') is not None and len(r['E']) > 0]
    E_d = [r['E'].mean() for r in dysbiotic if r.get('E') is not None and len(r['E']) > 0]

    if days_c and E_c:
        axes[0, 1].plot(days_c[:len(E_c)], E_c, 'b-o', label='Commensal', linewidth=2)
    if days_d and E_d:
        axes[0, 1].plot(days_d[:len(E_d)], E_d, 'r-s', label='Dysbiotic', linewidth=2)
    axes[0, 1].set_xlabel('Day')
    axes[0, 1].set_ylabel('Mean E [Pa]')
    axes[0, 1].set_title('Stiffness evolution')
    axes[0, 1].legend()

    # Row 0: Colony count
    nc = [r['n_colonies'] for r in commensal]
    nd = [r['n_colonies'] for r in dysbiotic]
    if days_c and nc:
        axes[0, 2].plot(days_c[:len(nc)], nc, 'b-o', label='Commensal')
    if days_d and nd:
        axes[0, 2].plot(days_d[:len(nd)], nd, 'r-s', label='Dysbiotic')
    axes[0, 2].set_xlabel('Day')
    axes[0, 2].set_ylabel('N colonies')
    axes[0, 2].set_title('Colony count')
    axes[0, 2].legend()

    # Row 0: Species stacked bar for last timepoint
    for i, (group, label, color) in enumerate(
            [(commensal, 'Commensal', 'blue'), (dysbiotic, 'Dysbiotic', 'red')]):
        if group and group[-1].get('species_fracs') is not None and len(group[-1]['species_fracs']) > 0:
            mean_phi = group[-1]['species_fracs'].mean(axis=0)
            bottom = 0
            colors = ['#33cc33', '#0088ff', '#ffaa00', '#cc00cc', '#ff0000']
            for j, (sp, c) in enumerate(zip(['An', 'So', 'Vd', 'Fn', 'Pg'], colors)):
                axes[0, 3].bar(label, mean_phi[j], bottom=bottom, color=c, label=sp if i == 0 else '')
                bottom += mean_phi[j]
    axes[0, 3].set_ylabel('Fraction')
    axes[0, 3].set_title('Species (last day)')
    axes[0, 3].legend(loc='upper right', fontsize=8)

    # Row 1: Representative images (day 1, 6, 21 for each)
    for col, day_target in enumerate([1, 6, 21]):
        ax = axes[1, col]
        # Find matching results
        for r in commensal + dysbiotic:
            if f'day{day_target:02d}' in r['condition'] and 'commensal' in r['condition']:
                ax.imshow(r['image'])
                if r.get('seeds_px') is not None and len(r['seeds_px']) > 0:
                    ax.scatter(r['seeds_px'][:, 0], r['seeds_px'][:, 1],
                             c='white', s=10, marker='+', linewidths=0.3)
                ax.set_title(f'Commensal Day {day_target}')
                ax.axis('off')
                break

    # Last panel: displacement comparison
    ax = axes[1, 3]
    disp_c = [r.get('max_disp', 0) or 0 for r in commensal]
    disp_d = [r.get('max_disp', 0) or 0 for r in dysbiotic]
    if any(d > 0 for d in disp_c + disp_d):
        if days_c:
            ax.plot(days_c[:len(disp_c)], disp_c, 'b-o', label='Commensal')
        if days_d:
            ax.plot(days_d[:len(disp_d)], disp_d, 'r-s', label='Dysbiotic')
        ax.set_xlabel('Day')
        ax.set_ylabel('Max |u| [µm]')
        ax.set_title('Peak displacement')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No displacement\ndata', ha='center', va='center',
               transform=ax.transAxes)

    plt.tight_layout()
    fname = os.path.join(output_dir, 'heine_fish_vem_comparison.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved comparison: {fname}")


# ── Hybrid pipeline: TMCMC-calibrated DI + FISH spatial layout ────────

# Heine 2025 experimental days and corresponding ODE timesteps.
# Growth rate ~0.41/h → dt=1e-5 dimensionless, maxtimestep maps to
# real time via t_phys ≈ maxtimestep * dt / growth_rate.
# The TMCMC calibration used maxtimestep=2500. Heine 2025 runs 21 days.
# We linearly interpolate: day d → step = int(d/21 * 2500).
HEINE_DAYS = [1, 6, 10, 15, 21]

# MAP theta paths (relative to Tmcmc202601 _runs directory)
_TMCMC_ROOT = os.path.join(os.path.dirname(__file__), '..', 'Tmcmc202601',
                            'data_5species', '_runs')
_MAP_PATHS = {
    'commensal_hobic': os.path.join(_TMCMC_ROOT, 'commensal_hobic', 'theta_MAP.json'),
    'dysbiotic_hobic': os.path.join(_TMCMC_ROOT, 'dh_baseline', 'theta_MAP.json'),
}

# DI weights for the weighted-sum formula (alternative to Shannon DI).
# w = [An, So, Vd, Fn, Pg] — pathogenicity weights.
_DI_WEIGHTS = np.array([0.0, 0.3, 0.5, 0.7, 1.0])


def _load_map_theta(condition):
    """Load MAP theta_full (20-vector) for a condition."""
    import json
    path = _MAP_PATHS[condition]
    with open(path) as f:
        data = json.load(f)
    return np.array(data['theta_full'], dtype=np.float64)


def _run_hamilton_ode(theta, maxtimestep=2500, dt=1e-5,
                      K_hill=0.05, n_hill=4.0):
    """
    Run the 0D Hamilton ODE (Numba solver) and return full trajectory.

    Returns:
      t_arr: (T+1,) time array
      g_arr: (T+1, 12) state array
        g[:5] = phi (species fractions)
        g[6:11] = psi (viability)
        g[5] = phi0, g[11] = gamma
    """
    solver_dir = os.path.join(os.path.dirname(__file__), '..', 'Tmcmc202601',
                               'tmcmc', 'program2602')
    if solver_dir not in sys.path:
        sys.path.insert(0, solver_dir)
    from improved_5species_jit import BiofilmNewtonSolver5S

    solver = BiofilmNewtonSolver5S(
        dt=dt, maxtimestep=maxtimestep, eps=1e-8,
        K_hill=K_hill, n_hill=n_hill
    )
    t_arr, g_arr = solver.run_deterministic(theta[:20])
    return t_arr, g_arr


def _get_phi_at_days(theta, days=None, maxtimestep=2500):
    """
    Run Hamilton ODE and extract species fractions phi at specified days.

    Days are mapped linearly to timesteps: step = int(day/21 * maxtimestep).

    Returns:
      phi_dict: {day: (5,) array of species fractions}
    """
    if days is None:
        days = HEINE_DAYS
    t_arr, g_arr = _run_hamilton_ode(theta, maxtimestep=maxtimestep)
    n_steps = len(t_arr) - 1

    phi_dict = {}
    for day in days:
        step = min(int(day / 21.0 * n_steps), n_steps)
        phi = g_arr[step, :5].copy()
        # Normalize to sum to 1 (phi0 is the "void" phase)
        phi_sum = phi.sum()
        if phi_sum > 1e-10:
            phi /= phi_sum
        else:
            phi = np.ones(5) / 5.0
        phi_dict[day] = phi

    return phi_dict


def _compute_di_shannon(phi):
    """Shannon entropy based DI: DI = 1 - H/H_max. Consistent with material_models.py."""
    phi = np.asarray(phi, dtype=np.float64)
    phi_sum = phi.sum()
    if phi_sum < 1e-15:
        return 0.5
    p = phi / phi_sum
    with np.errstate(divide='ignore', invalid='ignore'):
        log_p = np.where(p > 1e-15, np.log(p), 0.0)
    H = -(p * log_p).sum()
    return float(1.0 - H / np.log(5.0))


def _compute_di_weighted(phi):
    """Weighted DI: DI = w^T * phi_normalized."""
    phi = np.asarray(phi, dtype=np.float64)
    phi_sum = phi.sum()
    if phi_sum < 1e-15:
        return 0.5
    p = phi / phi_sum
    return float(_DI_WEIGHTS @ p)


def process_fish_image_hybrid(image_path, condition_name, model_phi,
                               di_method='shannon', spatial_noise_std=0.02,
                               scale_um=25.0, E_max=1000.0, E_min=10.0,
                               n_exp=2, output_dir=None):
    """
    Hybrid pipeline: FISH image for colony positions, TMCMC model for DI.

    Parameters:
      image_path: path to FISH image
      condition_name: e.g. 'commensal_hobic_day06'
      model_phi: (5,) species fractions from TMCMC-calibrated ODE
      di_method: 'shannon' or 'weighted'
      spatial_noise_std: std of DI noise for spatial variation (realism)
      scale_um: physical scale
      E_max, E_min, n_exp: E(DI) parameters

    Returns:
      dict with results (same structure as process_fish_image, plus 'model_di')
    """
    print(f"\n{'='*60}")
    print(f"[HYBRID] Processing: {condition_name}")
    print(f"  Image: {image_path}")
    print(f"  Model phi: {np.array2string(model_phi, precision=3)}")
    print(f"{'='*60}")

    # Load image
    img = np.array(Image.open(image_path))
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    H, W = img.shape[:2]

    # Step 1: Color decomposition (still needed for colony detection)
    print("  Step 1: Decomposing channels (for colony detection)...")
    channels = decompose_fish_channels(img[:, :, :3])

    # Step 2: Colony detection from image
    print("  Step 2: Detecting colonies from FISH image...")
    seeds, species_fracs_image = detect_colonies_from_channels(channels)
    n_colonies = len(seeds)
    print(f"    Found {n_colonies} colonies")

    if n_colonies < 3:
        print("  WARNING: Too few colonies. Skipping VEM.")
        return {'condition': condition_name, 'n_colonies': n_colonies,
                'channels': channels, 'error': 'too_few_colonies'}

    # Step 3: Compute DI from MODEL (not from image)
    if di_method == 'weighted':
        base_di = _compute_di_weighted(model_phi)
    else:
        base_di = _compute_di_shannon(model_phi)

    print(f"  Step 3: Model-based DI = {base_di:.4f} (method={di_method})")
    print(f"    Model phi: An={model_phi[0]:.3f} So={model_phi[1]:.3f} "
          f"Vd={model_phi[2]:.3f} Fn={model_phi[3]:.3f} Pg={model_phi[4]:.3f}")

    # Add spatial noise: each colony gets DI = base_di + N(0, noise_std)
    # This preserves the spatial heterogeneity visible in images while using
    # the correct global DI from the calibrated model.
    rng = np.random.default_rng(hash(condition_name) % (2**31))
    di_model = base_di + rng.normal(0, spatial_noise_std, n_colonies)
    di_model = np.clip(di_model, 0.0, 1.0)

    # Also compute image-based DI for comparison
    di_image = compute_di(species_fracs_image)

    # E(DI) from model-based DI
    E_model = compute_E_from_di(di_model, E_max, E_min, n_exp)
    E_image = compute_E_from_di(di_image, E_max, E_min, n_exp)

    print(f"    Model DI: mean={di_model.mean():.4f}, range=[{di_model.min():.4f}, {di_model.max():.4f}]")
    print(f"    Image DI: mean={di_image.mean():.4f}, range=[{di_image.min():.4f}, {di_image.max():.4f}]")
    print(f"    Model E:  mean={E_model.mean():.0f} Pa, range=[{E_model.min():.0f}, {E_model.max():.0f}] Pa")
    print(f"    Image E:  mean={E_image.mean():.0f} Pa, range=[{E_image.min():.0f}, {E_image.max():.0f}] Pa")

    # Step 4: Build Voronoi mesh (from image colony positions)
    print("  Step 4: Building Voronoi mesh...")
    result = build_voronoi_mesh(seeds, (H, W), scale_um=scale_um)
    if result[0] is None:
        print("  WARNING: Mesh construction failed.")
        return {'condition': condition_name, 'n_colonies': n_colonies,
                'channels': channels, 'di_model': di_model, 'di_image': di_image,
                'E_model': E_model, 'E_image': E_image, 'error': 'mesh_failed'}

    vertices, elements, boundary_nodes, um_per_px, valid_seeds = result
    print(f"    Mesh: {len(vertices)} vertices, {len(elements)} cells")

    # Step 5: VEM with model-based E
    print("  Step 5: Solving VEM (model-based E)...")
    n_nodes = len(vertices)
    nu = 0.3
    Ly = H * um_per_px
    Lx = W * um_per_px

    E_per_cell = E_model[valid_seeds] if len(valid_seeds) <= len(E_model) else \
        np.full(len(elements), E_model.mean())

    # BC setup (same as original)
    used_verts = set()
    for el in elements:
        for vi in el:
            used_verts.add(vi)
    used_verts = np.array(sorted(used_verts))

    y_vals = vertices[used_verts, 1]
    y_lo = np.percentile(y_vals, 5)
    y_hi = np.percentile(y_vals, 95)

    bottom = used_verts[y_vals <= y_lo]
    top = used_verts[y_vals >= y_hi]

    if len(bottom) < 3:
        bottom = used_verts[np.argsort(y_vals)[:max(5, len(used_verts)//20)]]
    if len(top) < 3:
        top = used_verts[np.argsort(y_vals)[-max(5, len(used_verts)//20):]]

    bc_dofs = np.concatenate([2 * bottom, 2 * bottom + 1])
    bc_vals = np.zeros(len(bc_dofs))
    load_dofs = 2 * top + 1
    load_vals = np.full(len(top), -2.0 / max(len(top), 1))

    u_model = None
    max_disp_model = None
    try:
        u_model = vem_elasticity(vertices, elements, E_per_cell, nu,
                                  bc_dofs, bc_vals, load_dofs, load_vals)
        ux = u_model[0::2]
        uy = u_model[1::2]
        max_disp_model = np.sqrt(ux**2 + uy**2).max()
        print(f"    Max displacement (model): {max_disp_model:.4f} um")
    except Exception as e:
        print(f"  VEM solve failed: {e}")

    # Also solve with image-based E for comparison
    E_per_cell_image = E_image[valid_seeds] if len(valid_seeds) <= len(E_image) else \
        np.full(len(elements), E_image.mean())
    u_image = None
    max_disp_image = None
    try:
        u_image = vem_elasticity(vertices, elements, E_per_cell_image, nu,
                                  bc_dofs, bc_vals, load_dofs, load_vals)
        ux = u_image[0::2]
        uy = u_image[1::2]
        max_disp_image = np.sqrt(ux**2 + uy**2).max()
        print(f"    Max displacement (image): {max_disp_image:.4f} um")
    except Exception as e:
        print(f"  VEM solve (image E) failed: {e}")

    results = {
        'condition': condition_name,
        'image': img,
        'channels': channels,
        'seeds_px': seeds,
        'species_fracs_image': species_fracs_image,
        'model_phi': model_phi,
        'di_model': di_model,
        'di_image': di_image,
        'E_model': E_model,
        'E_image': E_image,
        'E_per_cell_model': E_per_cell,
        'E_per_cell_image': E_per_cell_image,
        'vertices': vertices,
        'elements': elements,
        'boundary_nodes': boundary_nodes,
        'valid_seeds': valid_seeds,
        'um_per_px': um_per_px,
        'u_model': u_model,
        'u_image': u_image,
        'max_disp_model': max_disp_model,
        'max_disp_image': max_disp_image,
        'n_colonies': n_colonies,
        'base_di': base_di,
    }

    return results


def plot_hybrid_comparison(all_results, output_dir):
    """
    Generate comparison figure: Image-based DI vs Model-based DI for all conditions.

    Layout: 5 rows (days) x 4 cols:
      [0] FISH image + colonies
      [1] Image-based DI map
      [2] Model-based DI map
      [3] E difference |E_model - E_image|
    Two groups: commensal (top) and dysbiotic (bottom).
    """
    os.makedirs(output_dir, exist_ok=True)

    commensal = sorted([r for r in all_results if 'commensal' in r['condition']],
                       key=lambda r: int(r['condition'].split('day')[1]) if 'day' in r['condition'] else 0)
    dysbiotic = sorted([r for r in all_results if 'dysbiotic' in r['condition']],
                       key=lambda r: int(r['condition'].split('day')[1]) if 'day' in r['condition'] else 0)

    for group, group_name in [(commensal, 'Commensal_HOBIC'), (dysbiotic, 'Dysbiotic_HOBIC')]:
        if not group:
            continue

        n_rows = len(group)
        fig, axes = plt.subplots(n_rows, 4, figsize=(16, 3.5 * n_rows))
        if n_rows == 1:
            axes = axes[np.newaxis, :]
        fig.suptitle(f'Hybrid DI Pipeline: {group_name}\n'
                     f'Left=FISH image | Mid-left=Image DI | Mid-right=Model DI | Right=|E_model - E_image|',
                     fontsize=13, fontweight='bold')

        for row, r in enumerate(group):
            cond = r['condition']
            day_str = cond.split('day')[1] if 'day' in cond else '?'
            seeds = r['seeds_px']
            img = r['image']

            # Col 0: FISH image + colonies
            axes[row, 0].imshow(img)
            if len(seeds) > 0:
                axes[row, 0].scatter(seeds[:, 0], seeds[:, 1], c='white',
                                     s=15, marker='+', linewidths=0.5, alpha=0.8)
            axes[row, 0].set_title(f'Day {day_str} ({len(seeds)} col.)', fontsize=10)
            axes[row, 0].axis('off')

            # Col 1: Image-based DI
            if r.get('di_image') is not None and len(seeds) > 0:
                sc = axes[row, 1].scatter(seeds[:, 0], seeds[:, 1],
                                          c=r['di_image'], cmap='RdYlGn_r',
                                          vmin=0, vmax=1, s=25)
                plt.colorbar(sc, ax=axes[row, 1], fraction=0.046)
                axes[row, 1].set_xlim(0, img.shape[1])
                axes[row, 1].set_ylim(img.shape[0], 0)
                axes[row, 1].set_aspect('equal')
                axes[row, 1].set_title(f'Image DI (mean={r["di_image"].mean():.3f})', fontsize=10)
            else:
                axes[row, 1].text(0.5, 0.5, 'N/A', ha='center', va='center',
                                  transform=axes[row, 1].transAxes)

            # Col 2: Model-based DI
            if r.get('di_model') is not None and len(seeds) > 0:
                sc = axes[row, 2].scatter(seeds[:, 0], seeds[:, 1],
                                          c=r['di_model'], cmap='RdYlGn_r',
                                          vmin=0, vmax=1, s=25)
                plt.colorbar(sc, ax=axes[row, 2], fraction=0.046)
                axes[row, 2].set_xlim(0, img.shape[1])
                axes[row, 2].set_ylim(img.shape[0], 0)
                axes[row, 2].set_aspect('equal')
                axes[row, 2].set_title(f'Model DI (mean={r["di_model"].mean():.3f})', fontsize=10)
            else:
                axes[row, 2].text(0.5, 0.5, 'N/A', ha='center', va='center',
                                  transform=axes[row, 2].transAxes)

            # Col 3: |E_model - E_image| difference
            if (r.get('E_model') is not None and r.get('E_image') is not None
                    and len(seeds) > 0):
                dE = np.abs(r['E_model'] - r['E_image'])
                sc = axes[row, 3].scatter(seeds[:, 0], seeds[:, 1],
                                          c=dE, cmap='hot', s=25,
                                          vmin=0, vmax=max(dE.max(), 1))
                plt.colorbar(sc, ax=axes[row, 3], fraction=0.046, label='Pa')
                axes[row, 3].set_xlim(0, img.shape[1])
                axes[row, 3].set_ylim(img.shape[0], 0)
                axes[row, 3].set_aspect('equal')
                axes[row, 3].set_title(f'|dE| (max={dE.max():.0f} Pa)', fontsize=10)
            else:
                axes[row, 3].text(0.5, 0.5, 'N/A', ha='center', va='center',
                                  transform=axes[row, 3].transAxes)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fname = os.path.join(output_dir, f'hybrid_di_comparison_{group_name.lower()}.png')
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {fname}")


def plot_hybrid_summary(all_results, output_dir):
    """
    Summary figure: DI and E evolution over days, model vs image.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle('Hybrid Pipeline Summary: TMCMC-calibrated DI vs Image-decomposed DI',
                 fontsize=13, fontweight='bold')

    commensal = sorted([r for r in all_results if 'commensal' in r['condition']],
                       key=lambda r: int(r['condition'].split('day')[1]) if 'day' in r['condition'] else 0)
    dysbiotic = sorted([r for r in all_results if 'dysbiotic' in r['condition']],
                       key=lambda r: int(r['condition'].split('day')[1]) if 'day' in r['condition'] else 0)

    def _extract_days(group):
        return [int(r['condition'].split('day')[1]) for r in group if 'day' in r['condition']]

    # (0,0) DI evolution: model vs image
    ax = axes[0, 0]
    for group, label, cs, ms in [(commensal, 'Comm', 'blue', 'o'), (dysbiotic, 'Dysb', 'red', 's')]:
        days = _extract_days(group)
        di_m = [r['di_model'].mean() for r in group if r.get('di_model') is not None]
        di_i = [r['di_image'].mean() for r in group if r.get('di_image') is not None]
        if days and di_m:
            ax.plot(days[:len(di_m)], di_m, f'-{ms}', color=cs, linewidth=2,
                    label=f'{label} Model')
            ax.plot(days[:len(di_i)], di_i, f'--{ms}', color=cs, linewidth=1.5,
                    alpha=0.5, label=f'{label} Image')
    ax.set_xlabel('Day')
    ax.set_ylabel('Mean DI')
    ax.set_title('DI Evolution')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    # (0,1) E evolution: model vs image
    ax = axes[0, 1]
    for group, label, cs, ms in [(commensal, 'Comm', 'blue', 'o'), (dysbiotic, 'Dysb', 'red', 's')]:
        days = _extract_days(group)
        E_m = [r['E_model'].mean() for r in group if r.get('E_model') is not None]
        E_i = [r['E_image'].mean() for r in group if r.get('E_image') is not None]
        if days and E_m:
            ax.plot(days[:len(E_m)], E_m, f'-{ms}', color=cs, linewidth=2,
                    label=f'{label} Model')
            ax.plot(days[:len(E_i)], E_i, f'--{ms}', color=cs, linewidth=1.5,
                    alpha=0.5, label=f'{label} Image')
    ax.set_xlabel('Day')
    ax.set_ylabel('Mean E [Pa]')
    ax.set_title('Stiffness Evolution')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (0,2) Displacement comparison
    ax = axes[0, 2]
    for group, label, cs, ms in [(commensal, 'Comm', 'blue', 'o'), (dysbiotic, 'Dysb', 'red', 's')]:
        days = _extract_days(group)
        d_m = [r.get('max_disp_model', 0) or 0 for r in group]
        d_i = [r.get('max_disp_image', 0) or 0 for r in group]
        if days and any(d > 0 for d in d_m + d_i):
            ax.plot(days[:len(d_m)], d_m, f'-{ms}', color=cs, linewidth=2,
                    label=f'{label} Model')
            ax.plot(days[:len(d_i)], d_i, f'--{ms}', color=cs, linewidth=1.5,
                    alpha=0.5, label=f'{label} Image')
    ax.set_xlabel('Day')
    ax.set_ylabel('Max |u| [um]')
    ax.set_title('Peak Displacement')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (1,0) Species fractions from model (bar chart, last day)
    ax = axes[1, 0]
    sp_names = ['An', 'So', 'Vd', 'Fn', 'Pg']
    sp_colors = ['#33cc33', '#0088ff', '#ffaa00', '#cc00cc', '#ff0000']
    x_pos = np.arange(5)
    bar_w = 0.35
    for i, (group, label, offset) in enumerate(
            [(commensal, 'Comm', -bar_w/2), (dysbiotic, 'Dysb', bar_w/2)]):
        if group and group[-1].get('model_phi') is not None:
            phi = group[-1]['model_phi']
            bars = ax.bar(x_pos + offset, phi, bar_w, label=label,
                          color=[sp_colors[j] for j in range(5)],
                          alpha=0.8 if i == 0 else 0.5,
                          edgecolor='black', linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(sp_names)
    ax.set_ylabel('Model phi')
    ax.set_title('Species Fractions (Day 21, Model)')
    ax.legend(fontsize=8)

    # (1,1) DI scatter: model vs image per colony (all conditions pooled)
    ax = axes[1, 1]
    for group, label, cs in [(commensal, 'Commensal', 'blue'), (dysbiotic, 'Dysbiotic', 'red')]:
        all_di_m = []
        all_di_i = []
        for r in group:
            if r.get('di_model') is not None and r.get('di_image') is not None:
                all_di_m.extend(r['di_model'])
                all_di_i.extend(r['di_image'])
        if all_di_m:
            ax.scatter(all_di_i, all_di_m, c=cs, s=8, alpha=0.4, label=label)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='1:1')
    ax.set_xlabel('Image-based DI')
    ax.set_ylabel('Model-based DI')
    ax.set_title('DI: Model vs Image (per colony)')
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)

    # (1,2) DI delta histogram
    ax = axes[1, 2]
    for group, label, cs in [(commensal, 'Commensal', 'blue'), (dysbiotic, 'Dysbiotic', 'red')]:
        deltas = []
        for r in group:
            if r.get('di_model') is not None and r.get('di_image') is not None:
                deltas.extend(r['di_model'] - r['di_image'])
        if deltas:
            ax.hist(deltas, bins=30, color=cs, alpha=0.5, label=label, density=True)
    ax.axvline(0, color='black', linestyle='--', alpha=0.3)
    ax.set_xlabel('DI_model - DI_image')
    ax.set_ylabel('Density')
    ax.set_title('DI Correction Distribution')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fname = os.path.join(output_dir, 'hybrid_di_summary.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")


def process_all_heine_hybrid(output_dir=None):
    """
    Main hybrid pipeline: process all 10 Heine 2025 FISH conditions.

    Uses TMCMC-calibrated Hamilton ODE for species fractions (-> DI),
    and FISH images only for colony spatial positions.

    Generates:
      - Per-condition comparison figures (image vs model DI/E maps)
      - Summary figure with temporal evolution
    """
    base_dir = os.path.join(os.path.dirname(__file__), 'heine_extracted')
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), 'results', 'heine_fish')
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("HYBRID DI PIPELINE: TMCMC-Calibrated Model + FISH Spatial Layout")
    print("=" * 70)

    # ── 1. Load MAP theta for each condition ──────────────────────────────
    print("\n[1] Loading TMCMC MAP theta...")
    theta_ch = _load_map_theta('commensal_hobic')
    theta_dh = _load_map_theta('dysbiotic_hobic')
    print(f"  Commensal HOBIC: theta[0:5] = {theta_ch[:5].round(3)}")
    print(f"  Dysbiotic HOBIC: theta[0:5] = {theta_dh[:5].round(3)}")

    # ── 2. Run Hamilton ODE for each condition → phi at each day ──────────
    print("\n[2] Running Hamilton ODE for species fractions at each day...")
    phi_ch = _get_phi_at_days(theta_ch, HEINE_DAYS)
    phi_dh = _get_phi_at_days(theta_dh, HEINE_DAYS)

    print("  Commensal HOBIC:")
    for day, phi in phi_ch.items():
        di = _compute_di_shannon(phi)
        print(f"    Day {day:2d}: phi={np.array2string(phi, precision=3, floatmode='fixed')}, DI={di:.4f}")

    print("  Dysbiotic HOBIC:")
    for day, phi in phi_dh.items():
        di = _compute_di_shannon(phi)
        print(f"    Day {day:2d}: phi={np.array2string(phi, precision=3, floatmode='fixed')}, DI={di:.4f}")

    # ── 3. Process each FISH image with hybrid approach ───────────────────
    print("\n[3] Processing FISH images with hybrid DI...")
    images = {
        'commensal_hobic_day01': ('fish_commensal_hobic_day01.png', 'commensal_hobic', 1),
        'commensal_hobic_day06': ('fish_commensal_hobic_day06.png', 'commensal_hobic', 6),
        'commensal_hobic_day10': ('fish_commensal_hobic_day10.png', 'commensal_hobic', 10),
        'commensal_hobic_day15': ('fish_commensal_hobic_day15.png', 'commensal_hobic', 15),
        'commensal_hobic_day21': ('fish_commensal_hobic_day21.png', 'commensal_hobic', 21),
        'dysbiotic_hobic_day01': ('fish_dysbiotic_hobic_day01.png', 'dysbiotic_hobic', 1),
        'dysbiotic_hobic_day06': ('fish_dysbiotic_hobic_day06.png', 'dysbiotic_hobic', 6),
        'dysbiotic_hobic_day10': ('fish_dysbiotic_hobic_day10.png', 'dysbiotic_hobic', 10),
        'dysbiotic_hobic_day15': ('fish_dysbiotic_hobic_day15.png', 'dysbiotic_hobic', 15),
        'dysbiotic_hobic_day21': ('fish_dysbiotic_hobic_day21.png', 'dysbiotic_hobic', 21),
    }

    phi_maps = {'commensal_hobic': phi_ch, 'dysbiotic_hobic': phi_dh}

    all_results = []
    for cond_name in sorted(images.keys()):
        fname, cond_key, day = images[cond_name]
        path = os.path.join(base_dir, fname)
        if not os.path.exists(path):
            print(f"  Skipping {cond_name}: {fname} not found")
            continue

        model_phi = phi_maps[cond_key][day]
        r = process_fish_image_hybrid(
            path, cond_name, model_phi,
            di_method='shannon', spatial_noise_std=0.02,
            output_dir=output_dir
        )
        all_results.append(r)

    # ── 4. Generate comparison figures ────────────────────────────────────
    print("\n[4] Generating comparison figures...")
    valid_results = [r for r in all_results if 'error' not in r]

    if len(valid_results) >= 2:
        plot_hybrid_comparison(valid_results, output_dir)
        plot_hybrid_summary(valid_results, output_dir)

    # ── 5. Print summary table ────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("HYBRID PIPELINE SUMMARY")
    print("=" * 90)
    print(f"{'Condition':<30s} {'N_col':>5s} {'DI_model':>10s} {'DI_image':>10s} "
          f"{'E_model':>10s} {'E_image':>10s} {'|u|_model':>10s} {'|u|_image':>10s}")
    print("-" * 90)
    for r in all_results:
        if 'error' in r:
            print(f"  {r['condition']:<30s}: ERROR ({r['error']})")
            continue
        di_m = f"{r['di_model'].mean():.4f}" if r.get('di_model') is not None else "N/A"
        di_i = f"{r['di_image'].mean():.4f}" if r.get('di_image') is not None else "N/A"
        E_m = f"{r['E_model'].mean():.0f}" if r.get('E_model') is not None else "N/A"
        E_i = f"{r['E_image'].mean():.0f}" if r.get('E_image') is not None else "N/A"
        u_m = f"{r['max_disp_model']:.4f}" if r.get('max_disp_model') else "N/A"
        u_i = f"{r['max_disp_image']:.4f}" if r.get('max_disp_image') else "N/A"
        print(f"  {r['condition']:<30s} {r['n_colonies']:>5d} {di_m:>10s} {di_i:>10s} "
              f"{E_m:>10s} {E_i:>10s} {u_m:>10s} {u_i:>10s}")

    return all_results


# ── Main ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Heine 2025 FISH -> VEM pipeline')
    parser.add_argument('--hybrid', action='store_true',
                        help='Run hybrid pipeline (TMCMC model DI + FISH spatial)')
    parser.add_argument('--original', action='store_true',
                        help='Run original image-only pipeline')
    parser.add_argument('--both', action='store_true',
                        help='Run both pipelines (default if no flag)')
    args = parser.parse_args()

    # Default: run both if no specific flag
    if not (args.hybrid or args.original or args.both):
        args.both = True

    base_dir = os.path.join(os.path.dirname(__file__), 'heine_extracted')
    out_dir = os.path.join(os.path.dirname(__file__), 'results', 'heine_fish')

    if args.original or args.both:
        print("\n" + "#" * 70)
        print("# ORIGINAL PIPELINE (image-only DI)")
        print("#" * 70)

        images = {
            'commensal_hobic_day01': 'fish_commensal_hobic_day01.png',
            'commensal_hobic_day06': 'fish_commensal_hobic_day06.png',
            'commensal_hobic_day10': 'fish_commensal_hobic_day10.png',
            'commensal_hobic_day15': 'fish_commensal_hobic_day15.png',
            'commensal_hobic_day21': 'fish_commensal_hobic_day21.png',
            'dysbiotic_hobic_day01': 'fish_dysbiotic_hobic_day01.png',
            'dysbiotic_hobic_day06': 'fish_dysbiotic_hobic_day06.png',
            'dysbiotic_hobic_day10': 'fish_dysbiotic_hobic_day10.png',
            'dysbiotic_hobic_day15': 'fish_dysbiotic_hobic_day15.png',
            'dysbiotic_hobic_day21': 'fish_dysbiotic_hobic_day21.png',
        }

        all_results = []
        for cond, fname in sorted(images.items()):
            path = os.path.join(base_dir, fname)
            if not os.path.exists(path):
                print(f"  Skipping {cond}: {fname} not found")
                continue
            r = process_fish_image(path, condition_name=cond,
                                  output_dir=out_dir)
            all_results.append(r)

        if len(all_results) >= 2:
            plot_comparison(all_results, out_dir)

        print("\n" + "=" * 60)
        print("ORIGINAL PIPELINE SUMMARY")
        print("=" * 60)
        for r in all_results:
            di_str = f"DI={r['di'].mean():.2f}" if r.get('di') is not None and len(r.get('di', [])) > 0 else "N/A"
            E_str = f"E={r['E'].mean():.0f} Pa" if r.get('E') is not None and len(r.get('E', [])) > 0 else "N/A"
            disp_str = f"|u|={r['max_disp']:.4f} um" if r.get('max_disp') else "N/A"
            print(f"  {r['condition']:30s}: N={r['n_colonies']:3d}  {di_str:12s}  {E_str:12s}  {disp_str}")

    if args.hybrid or args.both:
        print("\n" + "#" * 70)
        print("# HYBRID PIPELINE (TMCMC model DI + FISH spatial)")
        print("#" * 70)
        process_all_heine_hybrid(output_dir=out_dir)
