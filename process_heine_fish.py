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

    if method == 'heuristic':
        # Background mask (very dark pixels)
        brightness = (R + G + B) / 3
        bg_mask = brightness < 0.05

        # An (Alexa 488 → green): high G, moderate B, low R
        # In composite appears as green-to-cyan
        channels[0] = np.clip(G - 0.5 * R - 0.3 * B, 0, 1)
        # Boost where blue+green but not red (cyan = An)
        cyan_boost = np.clip(np.minimum(G, B) - R, 0, 1)
        channels[0] = np.clip(channels[0] + 0.5 * cyan_boost, 0, 1)

        # So (Alexa 405 → blue/violet): high B, low R, low G
        channels[1] = np.clip(B - 0.4 * G - 0.3 * R, 0, 1)

        # Vd (Alexa 568 → yellow/orange): high R+G, low B
        channels[2] = np.clip(np.minimum(R, G) - 0.5 * B, 0, 1)
        # Orange: R > G, low B
        orange = np.clip(R - 0.8 * G, 0, 1) * (1 - B)
        channels[2] = np.clip(channels[2] + 0.3 * orange, 0, 1)

        # Fn (dual AF405+AF647 → appears white/magenta in composite)
        # White = all channels high
        white_component = np.minimum(np.minimum(R, G), B)
        channels[3] = np.clip(white_component - 0.15, 0, 1)
        # Also magenta: high R+B, low G
        magenta = np.clip(np.minimum(R, B) - G, 0, 1)
        channels[3] = np.clip(channels[3] + 0.5 * magenta, 0, 1)

        # Pg (Alexa 647 → red/far-red): high R, low G, low B
        channels[4] = np.clip(R - 0.5 * G - 0.5 * B, 0, 1)

        # Zero out background
        for ch in range(5):
            channels[ch][bg_mask] = 0
            channels[ch] = gaussian_filter(channels[ch], sigma=1.0)

    elif method == 'nmf':
        from sklearn.decomposition import NMF
        # Reshape to (N_pixels, 3)
        pixels = img.reshape(-1, 3)
        # Reference spectra for each fluorophore (approximate RGB rendering)
        W_init = np.array([
            [0.1, 0.8, 0.5],   # An: green-cyan
            [0.1, 0.2, 0.9],   # So: blue
            [0.9, 0.8, 0.1],   # Vd: yellow
            [0.7, 0.5, 0.7],   # Fn: white/magenta
            [0.9, 0.1, 0.1],   # Pg: red
        ]).T  # (3, 5)
        model = NMF(n_components=5, init='custom', max_iter=500, random_state=42)
        H_init = np.linalg.lstsq(W_init, pixels.T, rcond=None)[0]
        H_init = np.clip(H_init, 0, None)
        model.fit(pixels, W=W_init, H=H_init)
        H_result = model.transform(pixels)
        for ch in range(5):
            channels[ch] = H_result[:, ch].reshape(H, W)
            channels[ch] /= max(channels[ch].max(), 1e-10)

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

    # Spatially varying E: use mean E for now (VEM needs single E per solve,
    # or we use the weighted approach)
    E_mean = E_per_cell.mean()

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
        u = vem_elasticity(vertices, elements, E_mean, nu,
                           bc_dofs, bc_vals, load_dofs, load_vals)
        ux = u[0::2]
        uy = u[1::2]
        max_disp = np.sqrt(ux**2 + uy**2).max()
        print(f"    Max displacement: {max_disp:.4f} µm")
        print(f"    Mean E used: {E_mean:.0f} Pa")
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


# ── Main ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    base_dir = os.path.join(os.path.dirname(__file__), 'heine_extracted')
    out_dir = os.path.join(os.path.dirname(__file__), 'results', 'heine_fish')

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
    print("SUMMARY")
    print("=" * 60)
    for r in all_results:
        di_str = f"DI={r['di'].mean():.2f}" if r.get('di') is not None and len(r.get('di', [])) > 0 else "N/A"
        E_str = f"E={r['E'].mean():.0f} Pa" if r.get('E') is not None and len(r.get('E', [])) > 0 else "N/A"
        disp_str = f"|u|={r['max_disp']:.4f} µm" if r.get('max_disp') else "N/A"
        print(f"  {r['condition']:30s}: N={r['n_colonies']:3d}  {di_str:12s}  {E_str:12s}  {disp_str}")
