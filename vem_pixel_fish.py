"""
Pixel-Direct VEM: Heine 2025 FISH image → pixel mesh → E(DI) → VEM.

Zero mesh generation. Each active pixel becomes a VEM quad element.
Channel decomposition → per-pixel species fraction → DI → E(DI) → elasticity.

Comparison with existing Voronoi pipeline (process_heine_fish.py):
  Voronoi: image → colony detection → seeds → Voronoi → per-cell DI → VEM
  Pixel:   image → per-pixel channel unmixing → per-pixel DI → VEM  (this file)

Advantage: no segmentation error, full spatial resolution, 1-line mesh generation.
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
from vem_exotic_meshes import pixel_mesh_from_array


# ── Per-pixel DI computation ──────────────────────────────────────────────

# Pathogenicity weights: An=commensal, Pg=pathogenic
_DI_WEIGHTS = np.array([0.0, 0.3, 0.5, 0.7, 1.0])  # An, So, Vd, Fn, Pg


def compute_pixel_di(channels, method='weighted'):
    """
    Compute per-pixel Dysbiosis Index from 5-channel fluorescence.

    Parameters
    ----------
    channels : (5, H, W) float — species abundance per pixel
    method : 'weighted' (pathogenicity-weighted) or 'shannon' (entropy-based)

    Returns
    -------
    DI : (H, W) float in [0, 1]
    """
    _, H, W = channels.shape
    total = channels.sum(axis=0)  # (H, W)
    total = np.where(total > 1e-10, total, 1.0)

    # Normalize to fractions per pixel
    phi = channels / total[None, :, :]  # (5, H, W)

    if method == 'weighted':
        # DI = sum(w_i * phi_i)
        DI = np.tensordot(_DI_WEIGHTS, phi, axes=([0], [0]))  # (H, W)
    else:
        # Shannon entropy: low entropy with pathogen dominance → high DI
        phi_safe = np.clip(phi, 1e-15, None)
        entropy = -np.sum(phi * np.log(phi_safe), axis=0)
        max_entropy = np.log(5)
        normalized_entropy = entropy / max_entropy

        # Pathogenic fraction
        pg_frac = phi[4]
        fn_frac = phi[3]
        DI = 0.5 * (1.0 - normalized_entropy) + 0.3 * pg_frac + 0.2 * fn_frac

    return np.clip(DI, 0, 1)


def compute_pixel_E(DI, E_max=1000.0, E_min=10.0, n=2):
    """E(DI) = E_min + (E_max - E_min) * (1 - DI)^n"""
    return E_min + (E_max - E_min) * (1.0 - DI) ** n


# ── Pixel-direct pipeline ────────────────────────────────────────────────

def pixel_fish_pipeline(image_path, condition_name='unknown',
                        scale_um=25.0, scale_px=50.0,
                        E_max=1000.0, E_min=10.0, n_hill=2,
                        intensity_threshold=0.05,
                        downsample=1,
                        output_dir=None):
    """
    Full pixel-direct pipeline: FISH image → VEM elasticity.

    Parameters
    ----------
    image_path : str
    condition_name : str
    scale_um : float — physical scale in µm
    scale_px : float — scale bar in pixels
    E_max, E_min : float — E(DI) bounds [Pa]
    n_hill : int — Hill exponent
    intensity_threshold : float — minimum brightness for active pixel
    downsample : int — downsample factor (1=full res, 2=half, etc.)
    output_dir : str — save figures here

    Returns
    -------
    results : dict with all intermediate and final data
    """
    print(f"\n{'='*60}")
    print(f"Pixel-Direct VEM: {condition_name}")
    print(f"Image: {image_path}")
    print(f"{'='*60}")

    # Load image
    img = np.array(Image.open(image_path))
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    H, W = img.shape[:2]
    print(f"  Image: {W}x{H} px")

    # Downsample if needed
    if downsample > 1:
        img_ds = img[::downsample, ::downsample]
        H_ds, W_ds = img_ds.shape[:2]
        print(f"  Downsampled: {W_ds}x{H_ds} px (factor {downsample})")
    else:
        img_ds = img
        H_ds, W_ds = H, W

    # Step 1: Channel decomposition
    print("  Step 1: Spectral unmixing (5 channels)...")
    channels = decompose_fish_channels(img_ds[:, :, :3])
    species = ['An', 'So', 'Vd', 'Fn', 'Pg']
    for i, sp in enumerate(species):
        print(f"    {sp}: max={channels[i].max():.3f}, "
              f"mean={channels[i].mean():.4f}")

    # Step 2: Per-pixel DI and E
    print("  Step 2: Per-pixel DI and E(DI)...")
    DI = compute_pixel_di(channels, method='weighted')
    E_map = compute_pixel_E(DI, E_max, E_min, n_hill)

    # Active pixel mask (where biofilm exists)
    brightness = (img_ds[:, :, :3].astype(float) / 255.0).mean(axis=2)
    active = brightness > intensity_threshold

    # Keep only the largest connected component (avoid singular stiffness matrix)
    from scipy.ndimage import label as ndlabel
    labeled, n_components = ndlabel(active)
    if n_components > 1:
        sizes = [(labeled == i).sum() for i in range(1, n_components + 1)]
        largest = np.argmax(sizes) + 1
        active = labeled == largest
        print(f"    Connected components: {n_components}, "
              f"keeping largest ({sizes[largest-1]} px)")

    n_active = active.sum()
    print(f"    Active pixels: {n_active} / {H_ds * W_ds} "
          f"({100.0 * n_active / (H_ds * W_ds):.1f}%)")
    print(f"    DI: mean={DI[active].mean():.3f}, "
          f"range=[{DI[active].min():.3f}, {DI[active].max():.3f}]")
    print(f"    E:  mean={E_map[active].mean():.0f} Pa, "
          f"range=[{E_map[active].min():.0f}, {E_map[active].max():.0f}] Pa")

    # Step 3: Build pixel mesh
    print("  Step 3: Building pixel mesh (zero cost)...")
    vertices, elements, E_field = pixel_mesh_from_array(active, E_map)
    n_el = len(elements)
    print(f"    Mesh: {len(vertices)} vertices, {n_el} elements (all quads)")

    # Physical scale
    um_per_px = scale_um / scale_px * downsample
    vertices_um = vertices * um_per_px
    Lx_um = W_ds * um_per_px
    Ly_um = H_ds * um_per_px
    print(f"    Physical domain: {Lx_um:.1f} x {Ly_um:.1f} µm "
          f"({um_per_px:.2f} µm/px)")

    # Step 4: Solve VEM
    print("  Step 4: VEM elasticity solve...")
    nu = 0.3

    # BCs: percentile-based (robust for irregular shapes)
    # pixel_mesh_from_array flips y, so high y = image top = substratum
    y_vals = vertices_um[:, 1]
    y_lo = np.percentile(y_vals, 5)
    y_hi = np.percentile(y_vals, 95)

    # Substratum (bottom in physical = low y) is fixed
    bottom = np.where(y_vals <= y_lo)[0]
    # GCF pressure on top (high y)
    top = np.where(y_vals >= y_hi)[0]

    # Ensure enough BC nodes
    if len(bottom) < 5:
        y_sorted = np.argsort(y_vals)
        n_bc = max(10, len(vertices_um) // 15)
        bottom = y_sorted[:n_bc]
    if len(top) < 5:
        y_sorted = np.argsort(y_vals)
        n_bc = max(10, len(vertices_um) // 15)
        top = y_sorted[-n_bc:]

    print(f"    BC: {len(bottom)} fixed (bottom), {len(top)} loaded (top)")

    bc_dofs = np.concatenate([2 * bottom, 2 * bottom + 1])
    bc_vals = np.zeros(len(bc_dofs))

    # GCF pressure ~2 Pa
    load_dofs = 2 * top + 1
    load_vals = np.full(len(top), -2.0 / max(len(top), 1))

    try:
        u = vem_elasticity(vertices_um, elements, E_field, nu,
                           bc_dofs, bc_vals, load_dofs, load_vals)
        ux = u[0::2]
        uy = u[1::2]
        u_mag = np.sqrt(ux**2 + uy**2)
        max_disp = u_mag.max()
        print(f"    Max displacement: {max_disp:.6f} µm")

        # Approximate mean von Mises
        vm_mean = _approx_mean_vm(vertices_um, elements, u, E_field, nu)
        print(f"    Mean von Mises: {vm_mean:.2f} Pa")
    except Exception as e:
        print(f"    VEM solve failed: {e}")
        u = None
        max_disp = None
        vm_mean = None

    results = {
        'condition': condition_name,
        'image': img,
        'image_ds': img_ds,
        'channels': channels,
        'DI_map': DI,
        'E_map': E_map,
        'active_mask': active,
        'vertices': vertices,         # pixel coords
        'vertices_um': vertices_um,   # physical coords
        'elements': elements,
        'E_field': E_field,
        'u': u,
        'max_disp': max_disp,
        'vm_mean': vm_mean,
        'um_per_px': um_per_px,
        'n_active': n_active,
        'downsample': downsample,
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_pixel_results(results, output_dir)

    return results


def _approx_mean_vm(vertices, elements, u, E_field, nu):
    """Approximate mean von Mises stress from element-wise gradient."""
    ux = u[0::2]
    uy = u[1::2]
    vm_list = []

    for i, el in enumerate(elements):
        el_int = el.astype(int)
        verts = vertices[el_int]
        n_v = len(el_int)
        if n_v < 3:
            continue

        A = np.column_stack([verts - verts.mean(axis=0), np.ones(n_v)])
        try:
            grad_ux = np.linalg.lstsq(A, ux[el_int], rcond=None)[0][:2]
            grad_uy = np.linalg.lstsq(A, uy[el_int], rcond=None)[0][:2]
        except Exception:
            continue

        exx = grad_ux[0]
        eyy = grad_uy[1]
        exy = 0.5 * (grad_ux[1] + grad_uy[0])

        E_el = E_field[i]
        C = E_el / (1.0 - nu**2)
        sxx = C * (exx + nu * eyy)
        syy = C * (nu * exx + eyy)
        sxy = C * (1.0 - nu) / 2.0 * 2 * exy
        vm = np.sqrt(max(0, sxx**2 - sxx * syy + syy**2 + 3 * sxy**2))
        vm_list.append(vm)

    return np.mean(vm_list) if vm_list else 0.0


# ── Visualization ────────────────────────────────────────────────────────

def plot_pixel_results(results, output_dir):
    """6-panel visualization: image, channels, DI, E, mesh+displacement, stress."""
    cond = results['condition']
    img = results['image_ds']
    channels = results['channels']
    DI = results['DI_map']
    E_map = results['E_map']
    active = results['active_mask']
    vertices = results['vertices_um']
    elements = results['elements']
    E_field = results['E_field']
    u = results['u']
    um_per_px = results['um_per_px']

    H, W = img.shape[:2]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Pixel-Direct VEM: {cond}\n'
                 f'({len(elements)} pixel elements, {um_per_px:.2f} µm/px)',
                 fontsize=14, fontweight='bold')

    # (0,0) Original FISH image
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('FISH composite')
    axes[0, 0].axis('off')

    # (0,1) Decomposed channels (RGB composite)
    composite = np.zeros((H, W, 3))
    composite[:, :, 1] = np.clip(channels[0], 0, 1)  # An → green
    composite[:, :, 2] = np.clip(channels[1], 0, 1)  # So → blue
    composite[:, :, 0] = np.clip(channels[4], 0, 1)  # Pg → red
    composite[:, :, 0] += 0.5 * np.clip(channels[2], 0, 1)  # Vd → yellow
    composite[:, :, 1] += 0.5 * np.clip(channels[2], 0, 1)
    composite = np.clip(composite, 0, 1)
    axes[0, 1].imshow(composite)
    axes[0, 1].set_title('Unmixed (R=Pg, G=An, B=So, Y=Vd)')
    axes[0, 1].axis('off')

    # (0,2) Per-pixel DI map
    DI_masked = np.where(active, DI, np.nan)
    im = axes[0, 2].imshow(DI_masked, cmap='RdYlGn_r', vmin=0, vmax=1,
                            interpolation='nearest')
    fig.colorbar(im, ax=axes[0, 2], label='DI', shrink=0.8)
    axes[0, 2].set_title(f'Pixel DI (mean={DI[active].mean():.3f})')
    axes[0, 2].axis('off')

    # (1,0) Per-pixel E(DI) map
    E_masked = np.where(active, E_map, np.nan)
    im = axes[1, 0].imshow(E_masked, cmap='viridis', interpolation='nearest')
    fig.colorbar(im, ax=axes[1, 0], label='E [Pa]', shrink=0.8)
    axes[1, 0].set_title(f'E(DI) (mean={E_map[active].mean():.0f} Pa)')
    axes[1, 0].axis('off')

    # (1,1) Displacement on pixel mesh
    if u is not None:
        ux = u[0::2]
        uy = u[1::2]
        u_mag = np.sqrt(ux**2 + uy**2)

        # Reconstruct displacement image
        disp_img = np.full((H, W), np.nan)
        # Map element centroids back to pixel coordinates
        for i, el in enumerate(elements):
            el_int = el.astype(int)
            centroid = vertices[el_int].mean(axis=0)  # in µm
            # Convert back to pixel
            px = int(round(centroid[0] / um_per_px))
            py = H - 1 - int(round(centroid[1] / um_per_px))
            if 0 <= py < H and 0 <= px < W:
                disp_img[py, px] = np.mean(u_mag[el_int])

        im = axes[1, 1].imshow(disp_img, cmap='hot_r', interpolation='nearest')
        fig.colorbar(im, ax=axes[1, 1], label='|u| [µm]', shrink=0.8)
        axes[1, 1].set_title(f'Displacement (max={u_mag.max():.6f} µm)')
    else:
        axes[1, 1].text(0.5, 0.5, 'VEM solve failed',
                        ha='center', va='center',
                        transform=axes[1, 1].transAxes, fontsize=14)
    axes[1, 1].axis('off')

    # (1,2) Von Mises stress
    if u is not None:
        vm_img = np.full((H, W), np.nan)
        nu = 0.3
        for i, el in enumerate(elements):
            el_int = el.astype(int)
            verts = vertices[el_int]
            n_v = len(el_int)
            if n_v < 3:
                continue
            A = np.column_stack([verts - verts.mean(axis=0), np.ones(n_v)])
            try:
                grad_ux = np.linalg.lstsq(A, ux[el_int], rcond=None)[0][:2]
                grad_uy = np.linalg.lstsq(A, uy[el_int], rcond=None)[0][:2]
            except Exception:
                continue
            exx = grad_ux[0]
            eyy = grad_uy[1]
            exy = 0.5 * (grad_ux[1] + grad_uy[0])
            E_el = E_field[i]
            C_val = E_el / (1.0 - nu**2)
            sxx = C_val * (exx + nu * eyy)
            syy = C_val * (nu * exx + eyy)
            sxy = C_val * (1.0 - nu) / 2.0 * 2 * exy
            vm = np.sqrt(max(0, sxx**2 - sxx * syy + syy**2 + 3 * sxy**2))

            centroid = verts.mean(axis=0)
            px = int(round(centroid[0] / um_per_px))
            py = H - 1 - int(round(centroid[1] / um_per_px))
            if 0 <= py < H and 0 <= px < W:
                vm_img[py, px] = vm

        im = axes[1, 2].imshow(vm_img, cmap='inferno', interpolation='nearest')
        fig.colorbar(im, ax=axes[1, 2], label='σ_vm [Pa]', shrink=0.8)
        vm_valid = vm_img[~np.isnan(vm_img)]
        axes[1, 2].set_title(f'von Mises (mean={vm_valid.mean():.1f} Pa)')
    else:
        axes[1, 2].text(0.5, 0.5, 'N/A', ha='center', va='center',
                        transform=axes[1, 2].transAxes, fontsize=14)
    axes[1, 2].axis('off')

    plt.tight_layout()
    fname = os.path.join(output_dir, f'pixel_vem_{cond}.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")


# ── Comparison: Pixel vs Voronoi ─────────────────────────────────────────

def compare_pixel_vs_voronoi(image_path, condition_name='comparison',
                              output_dir='/tmp'):
    """
    Run both pipelines on the same image and compare results.
    """
    from process_heine_fish import process_fish_image

    print(f"\n{'='*60}")
    print(f"COMPARISON: Pixel vs Voronoi — {condition_name}")
    print(f"{'='*60}")

    # Pixel pipeline
    res_pixel = pixel_fish_pipeline(
        image_path, condition_name=f'{condition_name}_pixel',
        downsample=4, output_dir=output_dir)

    # Voronoi pipeline
    res_voronoi = process_fish_image(
        image_path, condition_name=f'{condition_name}_voronoi',
        output_dir=output_dir)

    # Summary comparison
    print(f"\n  {'Metric':<25} {'Pixel':>12} {'Voronoi':>12}")
    print(f"  {'-'*49}")
    print(f"  {'Elements':<25} {len(res_pixel['elements']):>12} "
          f"{len(res_voronoi.get('elements', [])):>12}")
    print(f"  {'Vertices':<25} {len(res_pixel['vertices']):>12} "
          f"{len(res_voronoi.get('vertices', np.array([]))):>12}")

    if res_pixel.get('max_disp') and res_voronoi.get('max_disp'):
        print(f"  {'Max |u| [µm]':<25} {res_pixel['max_disp']:>12.6f} "
              f"{res_voronoi['max_disp']:>12.6f}")

    di_p = res_pixel['DI_map'][res_pixel['active_mask']].mean()
    di_v = res_voronoi.get('di')
    if di_v is not None and len(di_v) > 0:
        print(f"  {'Mean DI':<25} {di_p:>12.3f} {di_v.mean():>12.3f}")

    E_p = res_pixel['E_map'][res_pixel['active_mask']].mean()
    E_v = res_voronoi.get('E')
    if E_v is not None and len(E_v) > 0:
        print(f"  {'Mean E [Pa]':<25} {E_p:>12.0f} {E_v.mean():>12.0f}")

    return res_pixel, res_voronoi


# ── Batch processing: all Heine 2025 images ──────────────────────────────

def batch_heine_pixel(output_dir='/tmp/pixel_fish'):
    """Process all 10 Heine 2025 FISH images with pixel-direct VEM."""
    img_dir = os.path.join(os.path.dirname(__file__), 'heine_extracted')

    files = sorted([
        f for f in os.listdir(img_dir)
        if f.startswith('fish_') and f.endswith('.png')
    ])

    print(f"\n{'='*60}")
    print(f"Batch Pixel-Direct VEM: {len(files)} FISH images")
    print(f"{'='*60}")

    all_results = []
    summary_rows = []

    for f in files:
        # Parse condition and day
        parts = f.replace('.png', '').split('_')
        # fish_commensal_hobic_day01.png → commensal_hobic_day01
        cond = '_'.join(parts[1:])

        res = pixel_fish_pipeline(
            os.path.join(img_dir, f),
            condition_name=cond,
            downsample=4,  # 4x downsample for tractable dense solve
            output_dir=output_dir)

        all_results.append(res)

        summary_rows.append({
            'condition': cond,
            'n_pixels': res['n_active'],
            'mean_DI': res['DI_map'][res['active_mask']].mean(),
            'mean_E': res['E_map'][res['active_mask']].mean(),
            'max_disp': res.get('max_disp', 0) or 0,
            'vm_mean': res.get('vm_mean', 0) or 0,
        })

    # Summary plot
    plot_batch_summary(all_results, summary_rows, output_dir)

    return all_results


def plot_batch_summary(all_results, summary_rows, output_dir):
    """Plot summary comparison across all timepoints and conditions."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Pixel-Direct VEM: Heine 2025 Batch Results',
                 fontsize=14, fontweight='bold')

    # Separate commensal and dysbiotic
    comm = [r for r in summary_rows if 'commensal' in r['condition']]
    dysb = [r for r in summary_rows if 'dysbiotic' in r['condition']]

    def get_day(r):
        for part in r['condition'].split('_'):
            if part.startswith('day'):
                return int(part[3:])
        return 0

    comm.sort(key=get_day)
    dysb.sort(key=get_day)

    days_c = [get_day(r) for r in comm]
    days_d = [get_day(r) for r in dysb]

    # (0,0) DI evolution
    if days_c:
        axes[0, 0].plot(days_c, [r['mean_DI'] for r in comm],
                        'b-o', linewidth=2, label='Commensal')
    if days_d:
        axes[0, 0].plot(days_d, [r['mean_DI'] for r in dysb],
                        'r-s', linewidth=2, label='Dysbiotic')
    axes[0, 0].set_xlabel('Day')
    axes[0, 0].set_ylabel('Mean pixel DI')
    axes[0, 0].set_title('DI evolution')
    axes[0, 0].legend()
    axes[0, 0].set_ylim(0, 1)

    # (0,1) E evolution
    if days_c:
        axes[0, 1].plot(days_c, [r['mean_E'] for r in comm],
                        'b-o', linewidth=2, label='Commensal')
    if days_d:
        axes[0, 1].plot(days_d, [r['mean_E'] for r in dysb],
                        'r-s', linewidth=2, label='Dysbiotic')
    axes[0, 1].set_xlabel('Day')
    axes[0, 1].set_ylabel('Mean E [Pa]')
    axes[0, 1].set_title('Stiffness evolution')
    axes[0, 1].legend()

    # (0,2) Active pixel count
    if days_c:
        axes[0, 2].plot(days_c, [r['n_pixels'] for r in comm],
                        'b-o', linewidth=2, label='Commensal')
    if days_d:
        axes[0, 2].plot(days_d, [r['n_pixels'] for r in dysb],
                        'r-s', linewidth=2, label='Dysbiotic')
    axes[0, 2].set_xlabel('Day')
    axes[0, 2].set_ylabel('Active pixels')
    axes[0, 2].set_title('Biofilm coverage')
    axes[0, 2].legend()

    # (1,0) Displacement
    if days_c:
        axes[1, 0].plot(days_c, [r['max_disp'] for r in comm],
                        'b-o', linewidth=2, label='Commensal')
    if days_d:
        axes[1, 0].plot(days_d, [r['max_disp'] for r in dysb],
                        'r-s', linewidth=2, label='Dysbiotic')
    axes[1, 0].set_xlabel('Day')
    axes[1, 0].set_ylabel('Max |u| [µm]')
    axes[1, 0].set_title('Peak displacement')
    axes[1, 0].legend()

    # (1,1) Von Mises stress
    if days_c:
        axes[1, 1].plot(days_c, [r['vm_mean'] for r in comm],
                        'b-o', linewidth=2, label='Commensal')
    if days_d:
        axes[1, 1].plot(days_d, [r['vm_mean'] for r in dysb],
                        'r-s', linewidth=2, label='Dysbiotic')
    axes[1, 1].set_xlabel('Day')
    axes[1, 1].set_ylabel('Mean σ_vm [Pa]')
    axes[1, 1].set_title('Von Mises stress')
    axes[1, 1].legend()

    # (1,2) Summary table
    axes[1, 2].axis('off')
    table_data = []
    for r in summary_rows:
        table_data.append([
            r['condition'].replace('_hobic_', ' ').replace('_', ' '),
            f"{r['n_pixels']}",
            f"{r['mean_DI']:.3f}",
            f"{r['mean_E']:.0f}",
            f"{r['max_disp']:.4f}" if r['max_disp'] else 'N/A',
        ])

    if table_data:
        table = axes[1, 2].table(
            cellText=table_data,
            colLabels=['Condition', 'Pixels', 'DI', 'E [Pa]', '|u| [µm]'],
            loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.2, 1.5)

    plt.tight_layout()
    fname = os.path.join(output_dir, 'pixel_vem_batch_summary.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Batch summary saved: {fname}")


# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    output_dir = sys.argv[1] if len(sys.argv) > 1 else '/tmp/pixel_fish'

    if len(sys.argv) > 2 and sys.argv[2] == '--compare':
        # Compare pixel vs voronoi on one image
        img = os.path.join(os.path.dirname(__file__), 'heine_extracted',
                           'fish_dysbiotic_hobic_day21.png')
        compare_pixel_vs_voronoi(img, 'dysbiotic_day21', output_dir)
    else:
        # Batch all images
        batch_heine_pixel(output_dir)
