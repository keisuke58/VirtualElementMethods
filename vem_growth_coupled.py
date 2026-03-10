"""
Growth-Coupled VEM: Biofilm growth dynamics + 2D VEM elasticity.

Prototype for Future Work Idea #2: staggered coupling of
species dynamics (Hamilton ODE → DI → E(DI)) with VEM on Voronoi mesh.

Key features:
  - 2D Voronoi mesh where each cell = one biofilm micro-colony
  - Simplified 5-species Hamilton ODE per cell (logistic + interaction)
  - DI(φ) → E(DI) constitutive law
  - Staggered loop: grow → update E → solve VEM → (optional) cell division
  - Voronoi re-meshing when cells divide (VEM handles arbitrary polygons)

References:
  - Klempt et al. (2024): staggered coupling FEM + growth
  - Nishioka thesis: E(DI) = E_min + (E_max - E_min)·(1 - DI)^n
"""

import numpy as np
from scipy.spatial import Voronoi
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
import matplotlib.animation as animation
import os

from vem_elasticity import vem_elasticity


# ── Species Dynamics (simplified Hamilton) ────────────────────────────────

SPECIES_NAMES = ['An', 'So', 'Vd', 'Fn', 'Pg']

def make_interaction_matrix(condition='dh_baseline'):
    """
    Simplified 5-species interaction matrix A.
    Diagonal = intrinsic growth rate, off-diagonal = cross-feeding/competition.
    Based on MAP θ from TMCMC calibration.
    """
    if condition == 'commensal_static':
        # Commensal: An dominates, Pg suppressed
        A = np.array([
            [ 0.50, -0.05,  0.02,  0.01, -0.10],
            [-0.03,  0.35,  0.02,  0.01, -0.05],
            [ 0.05,  0.03,  0.30, -0.02, -0.08],
            [ 0.02,  0.01,  0.05,  0.25, -0.05],
            [-0.15, -0.10, -0.05, -0.02,  0.10],
        ])
    elif condition == 'dysbiotic_static':
        # Dysbiotic: Fn dominates via cross-feeding, Pg elevated
        A = np.array([
            [ 0.15, -0.15, -0.08, -0.05, -0.10],
            [-0.05,  0.35,  0.05,  0.03,  0.01],
            [ 0.02,  0.08,  0.40,  0.06,  0.03],
            [ 0.01,  0.05,  0.12,  0.55,  0.08],
            [ 0.00,  0.02,  0.08,  0.12,  0.42],
        ])
    else:  # dh_baseline (intermediate)
        A = np.array([
            [ 0.35, -0.10,  0.00,  0.00, -0.05],
            [-0.03,  0.45,  0.05,  0.03,  0.00],
            [ 0.04,  0.06,  0.38,  0.03,  0.02],
            [ 0.01,  0.03,  0.10,  0.40,  0.05],
            [-0.05,  0.00,  0.05,  0.08,  0.25],
        ])
    return A


def hamilton_step(phi, A, dt=0.1, stress_feedback=0.0):
    """
    One step of replicator equation on the simplex.
    dφ_i/dt = φ_i · [(Aφ)_i - φ^T A φ] + stress_effect

    Two-way coupling: mechanical stress modulates growth rates.
    High stress suppresses growth of fragile (pathogenic) species
    and favors robust (commensal) species.

    Parameters
    ----------
    phi : (5,) species fractions
    A : (5,5) interaction matrix
    dt : time step
    stress_feedback : scalar ≥ 0, von Mises stress magnitude (normalized)
        Higher stress favors commensal species (index 0) over pathogenic (index 4)

    Uses RK2 (midpoint) for better stability.
    """
    # Stress-dependent fitness modification:
    # High stress → commensal advantage, pathogenic disadvantage
    # Robustness weights: An=1.0, So=0.7, Vd=0.5, Fn=0.3, Pg=0.1
    robustness = np.array([1.0, 0.7, 0.5, 0.3, 0.1])
    stress_mod = stress_feedback * (robustness - 0.5) * 0.1

    def rhs(p):
        fitness = A @ p + stress_mod
        avg_fitness = p @ fitness
        return p * (fitness - avg_fitness)

    # RK2 midpoint
    k1 = rhs(phi)
    phi_mid = phi + 0.5 * dt * k1
    phi_mid = np.clip(phi_mid, 1e-10, None)
    phi_mid /= phi_mid.sum()

    k2 = rhs(phi_mid)
    phi_new = phi + dt * k2
    phi_new = np.clip(phi_new, 1e-10, None)
    phi_new /= phi_new.sum()
    return phi_new


def compute_DI(phi):
    """
    Dysbiosis Index from species fractions.
    Weighted sum: pathogenic species contribute more.
    DI = w^T · φ where w = pathogenicity weights.
    An(commensal)=0.0, So(acidogenic)=0.3, Vd(bridge)=0.5, Fn(bridge)=0.7, Pg(pathogen)=1.0
    Normalized so DI ∈ [0, 1].
    """
    # Pathogenicity weights for [An, So, Vd, Fn, Pg]
    w = np.array([0.0, 0.3, 0.5, 0.7, 1.0])
    DI = np.dot(w, phi)
    return np.clip(DI, 0.0, 1.0)


def compute_E(DI, E_max=1000.0, E_min=30.0, n=2):
    """E(DI) = E_min + (E_max - E_min) · (1 - DI)^n"""
    return E_min + (E_max - E_min) * (1.0 - DI) ** n


# ── 2D Voronoi Mesh with Growth ──────────────────────────────────────────

def make_biofilm_voronoi(seeds, domain=(0, 2, 0, 1)):
    """
    Generate 2D Voronoi mesh from seed points, clipped to domain.
    Uses mirror points for clean boundary treatment.

    Returns: vertices, elements (list of arrays), boundary_nodes
    """
    xmin, xmax, ymin, ymax = domain
    Lx = xmax - xmin
    Ly = ymax - ymin

    # Mirror across 4 edges
    all_pts = [seeds]
    for axis, vals in [(0, [xmin, xmax]), (1, [ymin, ymax])]:
        for v in vals:
            mirror = seeds.copy()
            mirror[:, axis] = 2 * v - mirror[:, axis]
            all_pts.append(mirror)
    all_pts = np.vstack(all_pts)

    n_orig = len(seeds)
    vor = Voronoi(all_pts)

    vertices = vor.vertices.copy()
    # Clip to domain
    vertices[:, 0] = np.clip(vertices[:, 0], xmin - 0.001, xmax + 0.001)
    vertices[:, 1] = np.clip(vertices[:, 1], ymin - 0.001, ymax + 0.001)

    # Merge close vertices
    unique_verts, remap = _merge_verts_2d(vertices, tol=1e-8)

    # Extract elements for original seeds only
    elements = []
    valid_cell_ids = []

    for cell_idx in range(n_orig):
        region_idx = vor.point_region[cell_idx]
        region = vor.regions[region_idx]
        if -1 in region or len(region) < 3:
            continue

        face = np.array([remap[v] for v in region])
        # Remove duplicates preserving order
        _, idx = np.unique(face, return_index=True)
        face = face[np.sort(idx)]
        if len(face) < 3:
            continue

        # Check cell is inside domain
        cell_c = unique_verts[face].mean(axis=0)
        if (xmin - 0.1 <= cell_c[0] <= xmax + 0.1 and
                ymin - 0.1 <= cell_c[1] <= ymax + 0.1):
            elements.append(face)
            valid_cell_ids.append(cell_idx)

    # Find boundary nodes
    tol = 0.02
    bnd = np.where(
        (unique_verts[:, 0] < xmin + tol) | (unique_verts[:, 0] > xmax - tol) |
        (unique_verts[:, 1] < ymin + tol) | (unique_verts[:, 1] > ymax - tol)
    )[0]

    return unique_verts, elements, bnd, np.array(valid_cell_ids)


def _merge_verts_2d(verts, tol=1e-10):
    """Merge close 2D vertices."""
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

    final_remap = np.array([old_to_new[i] for i in range(n)])
    return np.array(new_verts), final_remap


def cell_area_2d(vertices, element):
    """Compute area of 2D polygon."""
    verts = vertices[element.astype(int)]
    n = len(verts)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += verts[i, 0] * verts[j, 1] - verts[j, 0] * verts[i, 1]
    return 0.5 * abs(area)


# ── Growth-Coupled VEM Simulation ─────────────────────────────────────────

class BiofilmGrowthVEM:
    """
    Staggered coupling: species dynamics → DI → E(DI) → VEM elasticity.

    Growth steps:
      1. Advance species ODE for each cell
      2. Compute DI and E per cell
      3. Solve VEM elasticity (gravity + GCF pressure)
      4. Check for cell division (if cell biomass > threshold)
      5. Re-mesh if cells were added
    """

    def __init__(self, n_cells=40, condition='dh_baseline',
                 domain=(0, 2, 0, 1), seed=42):
        self.condition = condition
        self.domain = domain
        self.rng = np.random.default_rng(seed)
        self.A = make_interaction_matrix(condition)
        self.nu = 0.35

        # Initial seeds on a regular-ish grid with perturbation
        xmin, xmax, ymin, ymax = domain
        nx = int(np.sqrt(n_cells * (xmax - xmin) / (ymax - ymin)))
        ny = max(int(n_cells / nx), 2)
        xx = np.linspace(xmin + 0.05, xmax - 0.05, nx)
        yy = np.linspace(ymin + 0.05, ymax - 0.05, ny)
        gx, gy = np.meshgrid(xx, yy)
        seeds = np.column_stack([gx.ravel(), gy.ravel()])
        # Add small perturbation
        seeds += self.rng.uniform(-0.03, 0.03, seeds.shape)

        self.seeds = seeds[:n_cells]
        self._init_species()
        self._build_mesh()
        self.history = []

    def _init_species(self):
        """Initialize species fractions per cell."""
        n = len(self.seeds)
        self.phi = np.zeros((n, 5))

        for i in range(n):
            x, y = self.seeds[i]
            xmid = (self.domain[0] + self.domain[1]) / 2
            ymid = (self.domain[2] + self.domain[3]) / 2

            # Spatial gradient: commensal at edges, dysbiotic at center
            r = np.sqrt((x - xmid)**2 + (y - ymid)**2)
            r_max = np.sqrt((xmid - self.domain[0])**2 +
                            (ymid - self.domain[2])**2)
            proximity = 1.0 - r / r_max  # 1 at center, 0 at corner

            if proximity > 0.6:
                # Dysbiotic core: more So, Fn, Pg
                self.phi[i] = [0.10, 0.35, 0.20, 0.20, 0.15]
            elif proximity > 0.3:
                # Transition zone
                self.phi[i] = [0.25, 0.30, 0.20, 0.15, 0.10]
            else:
                # Commensal periphery: more An, less Pg
                self.phi[i] = [0.40, 0.25, 0.20, 0.12, 0.03]

            # Add small random noise
            noise = self.rng.uniform(0, 0.02, 5)
            self.phi[i] += noise
            self.phi[i] /= self.phi[i].sum()

    def _build_mesh(self):
        """Build Voronoi mesh from current seeds."""
        self.vertices, self.elements, self.boundary, self.valid_ids = \
            make_biofilm_voronoi(self.seeds, self.domain)
        self.n_cells = len(self.elements)

    def grow_step(self, dt=0.5, n_substeps=5):
        """
        Advance species dynamics by dt (with n_substeps sub-intervals).

        Two-way coupling: if self.u exists, compute per-cell stress
        and feed back to hamilton_step. High stress suppresses pathogenic
        species and favors commensal species.
        """
        dt_sub = dt / n_substeps

        # Compute per-cell stress feedback from displacement field
        stress_per_cell = np.zeros(len(self.phi))
        if hasattr(self, 'u') and self.u is not None and np.any(self.u != 0):
            ux = self.u[0::2]
            uy = self.u[1::2]
            n_ux = len(ux)
            for i in range(min(self.n_cells, len(self.valid_ids))):
                cell_id = self.valid_ids[i]
                if cell_id < len(self.phi):
                    el = self.elements[i]
                    el_int = el.astype(int)
                    if np.any(el_int >= n_ux):
                        continue
                    # von Mises proxy: RMS displacement magnitude
                    u_mag = np.sqrt(ux[el_int]**2 + uy[el_int]**2)
                    stress_per_cell[cell_id] = np.mean(u_mag) * 100  # normalize

        for _ in range(n_substeps):
            for i in range(len(self.phi)):
                cell_id = self.valid_ids[i] if i < len(self.valid_ids) else i
                if cell_id < len(self.phi):
                    sf = stress_per_cell[cell_id] if cell_id < len(stress_per_cell) else 0.0
                    self.phi[cell_id] = hamilton_step(
                        self.phi[cell_id], self.A, dt=dt_sub,
                        stress_feedback=sf)

    def compute_properties(self):
        """Compute DI, E for all valid cells."""
        self.DI = np.zeros(self.n_cells)
        self.E = np.zeros(self.n_cells)

        for i in range(self.n_cells):
            cell_id = self.valid_ids[i] if i < len(self.valid_ids) else i
            if cell_id < len(self.phi):
                self.DI[i] = compute_DI(self.phi[cell_id])
                self.E[i] = compute_E(self.DI[i])
            else:
                self.DI[i] = 0.5
                self.E[i] = compute_E(0.5)

    def solve_vem(self):
        """Solve VEM elasticity with current E field.
        Re-indexes to used nodes only to avoid singular K.
        """
        xmin, xmax, ymin, ymax = self.domain

        # Collect used nodes and build compact re-index
        used_set = set()
        for el in self.elements:
            used_set.update(el.astype(int).tolist())
        used = np.array(sorted(used_set))
        n_used = len(used)

        # Map: old global index → new compact index
        old_to_new = {int(g): i for i, g in enumerate(used)}

        # Compact vertices and elements
        compact_verts = self.vertices[used]
        compact_elems = []
        for el in self.elements:
            compact_elems.append(np.array([old_to_new[int(v)] for v in el]))

        # BC: fix bottom edge
        tol = 0.02
        bottom_mask = compact_verts[:, 1] < ymin + tol
        bottom_new = np.where(bottom_mask)[0]
        bc_dofs = np.concatenate([2 * bottom_new, 2 * bottom_new + 1])
        bc_vals = np.zeros(len(bc_dofs))

        # Load: gravity (downward) + GCF pressure on top
        top_mask = compact_verts[:, 1] > ymax - tol
        top_new = np.where(top_mask)[0]
        all_new = np.arange(n_used)

        load_dofs_list = []
        load_vals_list = []

        # Gravity on all nodes (y-direction)
        gravity = -0.005 / max(n_used, 1)
        load_dofs_list.append(2 * all_new + 1)
        load_vals_list.append(np.full(n_used, gravity))

        # GCF pressure on top (downward)
        if len(top_new) > 0:
            gcf_pressure = -0.01 / len(top_new)
            load_dofs_list.append(2 * top_new + 1)
            load_vals_list.append(np.full(len(top_new), gcf_pressure))

        load_dofs = np.concatenate(load_dofs_list)
        load_vals = np.concatenate(load_vals_list)

        try:
            u_compact = vem_elasticity(
                compact_verts, compact_elems, self.E, self.nu,
                bc_dofs, bc_vals, load_dofs, load_vals)
        except np.linalg.LinAlgError:
            u_compact = np.zeros(2 * n_used)

        # Map back to full vertex array
        self.u = np.zeros(2 * len(self.vertices))
        for new_i, old_i in enumerate(used):
            self.u[2 * old_i]     = u_compact[2 * new_i]
            self.u[2 * old_i + 1] = u_compact[2 * new_i + 1]

    def try_cell_division(self, area_threshold=0.06):
        """
        Check for cell division: if cell area > threshold,
        split by adding a new seed near the old one.
        Returns True if any division occurred.
        """
        new_seeds = list(self.seeds)
        new_phi = list(self.phi)
        divided = False

        for i in range(self.n_cells):
            cell_id = self.valid_ids[i] if i < len(self.valid_ids) else i
            if cell_id >= len(new_seeds):
                continue

            area = cell_area_2d(self.vertices, self.elements[i])
            if area > area_threshold:
                # Split: add new seed near original
                parent = new_seeds[cell_id]
                offset = self.rng.uniform(-0.05, 0.05, 2)
                child_pos = parent + offset

                # Check inside domain
                xmin, xmax, ymin, ymax = self.domain
                child_pos[0] = np.clip(child_pos[0], xmin + 0.02, xmax - 0.02)
                child_pos[1] = np.clip(child_pos[1], ymin + 0.02, ymax - 0.02)

                new_seeds.append(child_pos)
                # Child inherits parent species with small mutation
                child_phi = self.phi[cell_id].copy()
                child_phi += self.rng.uniform(-0.01, 0.01, 5)
                child_phi = np.clip(child_phi, 1e-8, None)
                child_phi /= child_phi.sum()
                new_phi.append(child_phi)
                divided = True

        if divided:
            self.seeds = np.array(new_seeds)
            self.phi = np.array(new_phi)
            self._build_mesh()

        return divided

    def run(self, n_steps=20, dt=0.5, division_interval=5, verbose=True):
        """Run growth-coupled simulation."""
        if verbose:
            print("=" * 60)
            print(f"Growth-Coupled VEM: {self.condition}")
            print(f"  Initial cells: {self.n_cells}")
            print("=" * 60)

        for step in range(n_steps):
            # 1. Grow species
            self.grow_step(dt=dt)

            # 2. Update material
            self.compute_properties()

            # 3. Solve VEM
            self.solve_vem()

            # Store snapshot (before division, while u is still valid)
            ux = self.u[0::2]
            uy = self.u[1::2]
            n_verts = len(self.vertices)
            used = set()
            for el in self.elements:
                used.update(el.astype(int).tolist())
            used = np.array([v for v in sorted(used) if v < n_verts and v < len(ux)])
            u_mag_max = np.max(np.sqrt(ux[used]**2 + uy[used]**2)) if len(used) > 0 else 0

            # 4. Cell division (after snapshot)
            divided = False
            if (step + 1) % division_interval == 0:
                divided = self.try_cell_division()

            snapshot = {
                'step': step,
                'n_cells': self.n_cells,
                'DI_mean': np.mean(self.DI),
                'DI_std': np.std(self.DI),
                'E_mean': np.mean(self.E),
                'E_min': np.min(self.E),
                'E_max': np.max(self.E),
                'u_max': u_mag_max,
                'divided': divided,
                'phi_mean': self.phi[:len(self.valid_ids)].mean(axis=0).copy()
                           if len(self.valid_ids) <= len(self.phi) else np.zeros(5),
            }
            self.history.append(snapshot)

            if verbose and (step % 5 == 0 or divided):
                div_str = " [DIVISION]" if divided else ""
                print(f"  t={step*dt:5.1f} | cells={self.n_cells:3d} | "
                      f"DI={snapshot['DI_mean']:.3f}±{snapshot['DI_std']:.3f} | "
                      f"E=[{snapshot['E_min']:.0f},{snapshot['E_max']:.0f}] Pa | "
                      f"|u|_max={snapshot['u_max']:.6f}{div_str}")

        if verbose:
            print(f"\n  Final: {self.n_cells} cells, "
                  f"DI={np.mean(self.DI):.3f}, E=[{np.min(self.E):.0f},{np.max(self.E):.0f}] Pa")

        return self.history


# ── Visualization ─────────────────────────────────────────────────────────

def plot_growth_snapshot(sim, step_label='', save=None):
    """Plot current state: DI, E, species, displacement."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    verts = sim.vertices
    elems = sim.elements

    # Helper
    def plot_field(ax, data_per_cell, cmap, label, coords=None):
        if coords is None:
            coords = verts
        patches = []
        colors = []
        for i, el in enumerate(elems):
            el_int = el.astype(int)
            patches.append(MplPolygon(coords[el_int], closed=True))
            colors.append(data_per_cell[i] if i < len(data_per_cell) else 0)
        pc = PatchCollection(patches, cmap=cmap, edgecolor='k', linewidth=0.3)
        pc.set_array(np.array(colors))
        ax.add_collection(pc)
        xmin, xmax, ymin, ymax = sim.domain
        ax.set_xlim(xmin - 0.05, xmax + 0.05)
        ax.set_ylim(ymin - 0.05, ymax + 0.05)
        ax.set_aspect('equal')
        fig.colorbar(pc, ax=ax, label=label, shrink=0.8)

    # 1. DI field
    plot_field(axes[0, 0], sim.DI, 'RdYlGn_r', 'DI')
    axes[0, 0].set_title('Dysbiosis Index')

    # 2. E field
    plot_field(axes[0, 1], sim.E, 'viridis', 'E [Pa]')
    axes[0, 1].set_title("Young's Modulus E(DI)")

    # 3. Displacement magnitude
    ux = sim.u[0::2]
    uy = sim.u[1::2]
    deform_scale = 100.0
    deformed = verts + deform_scale * np.column_stack([ux, uy])
    u_mag_per_cell = []
    for el in elems:
        el_int = el.astype(int)
        u_mag_per_cell.append(np.mean(np.sqrt(ux[el_int]**2 + uy[el_int]**2)))
    plot_field(axes[0, 2], u_mag_per_cell, 'hot_r', '|u|', coords=deformed)
    axes[0, 2].set_title(f'Deformed (x{deform_scale:.0f})')

    # 4-5. Dominant species and Pg fraction
    dominant = []
    pg_frac = []
    for i in range(sim.n_cells):
        cell_id = sim.valid_ids[i] if i < len(sim.valid_ids) else i
        if cell_id < len(sim.phi):
            dominant.append(np.argmax(sim.phi[cell_id]))
            pg_frac.append(sim.phi[cell_id, 4])
        else:
            dominant.append(0)
            pg_frac.append(0)

    # Dominant species map
    patches_dom = []
    for el in elems:
        patches_dom.append(MplPolygon(verts[el.astype(int)], closed=True))

    species_cmap = plt.cm.Set1
    pc_dom = PatchCollection(patches_dom, cmap=species_cmap, edgecolor='k',
                             linewidth=0.3)
    pc_dom.set_array(np.array(dominant, dtype=float))
    pc_dom.set_clim(0, 4)
    axes[1, 0].add_collection(pc_dom)
    xmin, xmax, ymin, ymax = sim.domain
    axes[1, 0].set_xlim(xmin - 0.05, xmax + 0.05)
    axes[1, 0].set_ylim(ymin - 0.05, ymax + 0.05)
    axes[1, 0].set_aspect('equal')
    cb = fig.colorbar(pc_dom, ax=axes[1, 0], label='Species', shrink=0.8,
                      ticks=[0, 1, 2, 3, 4])
    cb.ax.set_yticklabels(SPECIES_NAMES)
    axes[1, 0].set_title('Dominant Species')

    # Pg fraction
    plot_field(axes[1, 1], pg_frac, 'Reds', 'φ_Pg')
    axes[1, 1].set_title('P. gingivalis Fraction')

    # 6. Time history
    ax_hist = axes[1, 2]
    if sim.history:
        steps = [h['step'] for h in sim.history]
        DIs = [h['DI_mean'] for h in sim.history]
        Es = [h['E_mean'] for h in sim.history]

        ax_hist.plot(steps, DIs, 'r-o', markersize=3, label='DI (mean)')
        ax_hist.set_xlabel('Growth Step')
        ax_hist.set_ylabel('DI', color='r')
        ax_hist.tick_params(axis='y', labelcolor='r')

        ax2 = ax_hist.twinx()
        ax2.plot(steps, Es, 'b-s', markersize=3, label='E (mean)')
        ax2.set_ylabel('E [Pa]', color='b')
        ax2.tick_params(axis='y', labelcolor='b')

        # Mark divisions
        for h in sim.history:
            if h['divided']:
                ax_hist.axvline(h['step'], color='green', alpha=0.3,
                                linestyle='--')
        ax_hist.set_title('Growth History')
        ax_hist.legend(loc='upper left')
        ax2.legend(loc='upper right')

    fig.suptitle(f'Growth-Coupled VEM: {sim.condition} {step_label}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save}")
    plt.close()


def plot_species_evolution(sim, save=None):
    """Plot species fraction evolution over time."""
    if not sim.history:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    steps = [h['step'] for h in sim.history]
    phi_means = np.array([h['phi_mean'] for h in sim.history])

    # Stacked area plot
    ax = axes[0]
    colors = ['#2ca02c', '#ff7f0e', '#1f77b4', '#9467bd', '#d62728']
    ax.stackplot(steps, phi_means.T, labels=SPECIES_NAMES, colors=colors,
                 alpha=0.8)
    ax.set_xlabel('Growth Step')
    ax.set_ylabel('Mean Species Fraction')
    ax.set_title('Species Composition Over Time')
    ax.legend(loc='center right')
    ax.set_ylim(0, 1)

    # DI and E evolution with cell count
    ax2 = axes[1]
    DIs = [h['DI_mean'] for h in sim.history]
    n_cells = [h['n_cells'] for h in sim.history]
    u_maxs = [h['u_max'] for h in sim.history]

    ax2.plot(steps, DIs, 'r-o', markersize=3, label='DI mean')
    ax2.set_xlabel('Growth Step')
    ax2.set_ylabel('DI / |u|_max × 1000', color='r')

    ax2.plot(steps, [u * 1000 for u in u_maxs], 'b--', alpha=0.7,
             label='|u|_max × 1000')

    ax3 = ax2.twinx()
    ax3.plot(steps, n_cells, 'g-^', markersize=3, label='# cells')
    ax3.set_ylabel('Cell Count', color='g')

    ax2.legend(loc='upper left')
    ax3.legend(loc='upper right')
    ax2.set_title('DI, Displacement, Cell Count')

    fig.suptitle(f'Growth-Coupled VEM: {sim.condition}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save}")
    plt.close()


# ── Multi-Condition Comparison ────────────────────────────────────────────

def compare_conditions(save_dir='/tmp'):
    """Run growth-coupled VEM for 3 conditions and compare."""
    print("\n" + "=" * 60)
    print("Multi-Condition Growth-Coupled VEM Comparison")
    print("=" * 60)

    conditions = ['commensal_static', 'dh_baseline', 'dysbiotic_static']
    labels = ['Commensal (CS)', 'DH Baseline', 'Dysbiotic (DS)']
    results = {}

    for cond, label in zip(conditions, labels):
        print(f"\n--- {label} ---")
        sim = BiofilmGrowthVEM(n_cells=30, condition=cond, seed=42)
        sim.run(n_steps=40, dt=1.0, division_interval=15, verbose=True)
        results[cond] = sim

        plot_growth_snapshot(
            sim, step_label=f'(final)',
            save=f'{save_dir}/vem_growth_{cond}.png')

        plot_species_evolution(
            sim, save=f'{save_dir}/vem_growth_{cond}_evolution.png')

    # Comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, (cond, label) in enumerate(zip(conditions, labels)):
        sim = results[cond]
        ax = axes[i]

        patches = []
        colors = []
        for j, el in enumerate(sim.elements):
            patches.append(MplPolygon(sim.vertices[el.astype(int)], closed=True))
            colors.append(sim.E[j])

        pc = PatchCollection(patches, cmap='viridis', edgecolor='k',
                             linewidth=0.3)
        pc.set_array(np.array(colors))
        pc.set_clim(30, 1000)
        ax.add_collection(pc)
        xmin, xmax, ymin, ymax = sim.domain
        ax.set_xlim(xmin - 0.05, xmax + 0.05)
        ax.set_ylim(ymin - 0.05, ymax + 0.05)
        ax.set_aspect('equal')
        fig.colorbar(pc, ax=ax, label='E [Pa]', shrink=0.8)
        ax.set_title(f'{label}\nDI={np.mean(sim.DI):.3f}, '
                     f'E=[{np.min(sim.E):.0f},{np.max(sim.E):.0f}] Pa\n'
                     f'{sim.n_cells} cells')

    fig.suptitle('Growth-Coupled VEM: Condition Comparison (Final State)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = f'{save_dir}/vem_growth_comparison.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"\n  Saved: {path}")
    plt.close()

    # Summary table
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  {'Condition':<20s} {'Cells':>6s} {'DI':>8s} {'E_min':>8s} "
          f"{'E_max':>8s} {'|u|_max':>10s}")
    print("-" * 60)
    for cond, label in zip(conditions, labels):
        sim = results[cond]
        h = sim.history[-1]
        print(f"  {label:<20s} {h['n_cells']:>6d} {h['DI_mean']:>8.3f} "
              f"{h['E_min']:>8.0f} {h['E_max']:>8.0f} {h['u_max']:>10.6f}")
    print("=" * 60)

    return results


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    save_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(save_dir, exist_ok=True)

    # Single condition demo
    print("=" * 60)
    print("Growth-Coupled VEM Prototype")
    print("=" * 60)

    sim = BiofilmGrowthVEM(n_cells=40, condition='dh_baseline', seed=42)
    sim.run(n_steps=40, dt=1.0, division_interval=15, verbose=True)

    plot_growth_snapshot(
        sim, step_label='(final)',
        save=f'{save_dir}/vem_growth_dh_final.png')

    plot_species_evolution(
        sim, save=f'{save_dir}/vem_growth_dh_evolution.png')

    # Multi-condition comparison
    results = compare_conditions(save_dir=save_dir)

    print("\nGrowth-Coupled VEM prototype complete!")
