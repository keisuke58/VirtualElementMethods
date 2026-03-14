"""
Adaptive h-Refinement Coupled with Phase-Field Fracture VEM for Biofilm Detachment.

Combines:
  - vem_phase_field.py: staggered phase-field fracture with g(d)=(1-d)^2+k degradation
  - vem_error_estimator.py: h-adaptive mesh refinement with ZZ-type error estimator

Key idea: crack tip = high |nabla d| + high psi^+/G_c = high error indicator
  -> automatic mesh refinement at crack front
  -> coarse mesh far from crack, fine mesh at crack tip

Architecture:
  1. crack_tip_indicator: combined refinement indicator eta_E = w1*|grad d| + w2*psi/Gc
  2. refine_at_crack_tip: Voronoi re-meshing at marked elements
  3. transfer_fields: nearest-neighbor interpolation of d and psi to new mesh
  4. AdaptivePhaseFieldVEM: incremental loading with periodic adaptive refinement

References:
  - Aldakheel, Hudobivnik, Hussein, Wriggers (2018) CMAME 341
  - Nguyen-Thanh et al. (2018) CMAME 340 — VEM for 2D fracture at IKM
  - Beirão da Veiga et al. (2015) "A posteriori error estimation for VEM"
"""

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection

from vem_phase_field import (
    PhaseFieldVEM,
    compute_Gc,
    compute_E_from_DI,
    _element_geometry,
)
from vem_error_estimator import (
    estimate_element_error,
    refine_mesh_adaptive,
    compute_mesh_quality,
    _merge_verts,
)
from vem_growth_coupled import make_biofilm_voronoi
from vem_elasticity import vem_elasticity


# ── Crack-Tip Refinement Indicator ───────────────────────────────────────


def crack_tip_indicator(d_field, vertices, elements, psi_history, Gc_field,
                        w1=1.0, w2=1.0):
    """
    Combined refinement indicator for phase-field fracture.

    eta_E = w1 * |grad d|_E + w2 * psi^+ / G_c

    |grad d| is computed via the boundary integral (H^1 semi-norm pattern):
      grad d ≈ (1/|E|) * sum_edges d_mid * n_edge * |edge|

    High near crack tip (0 < d < 1 transition zone), low in intact/fully
    cracked regions.

    Parameters
    ----------
    d_field : (n_nodes,) phase-field values at nodes
    vertices : (n_nodes, 2) coordinates
    elements : list of int arrays, element connectivity
    psi_history : (n_el,) max tensile energy density per element
    Gc_field : (n_el,) fracture toughness per element
    w1, w2 : weights for gradient and energy terms

    Returns
    -------
    eta_crack : (n_el,) combined crack-tip indicator per element
    """
    n_el = len(elements)
    eta_crack = np.zeros(n_el)

    for i, el in enumerate(elements):
        el_int = el.astype(int)
        verts = vertices[el_int]
        n_v = len(el_int)

        # Element area via shoelace
        area_comp = (
            verts[:, 0] * np.roll(verts[:, 1], -1)
            - np.roll(verts[:, 0], -1) * verts[:, 1]
        )
        area = 0.5 * abs(np.sum(area_comp))
        if area < 1e-20:
            continue

        # Gradient of d via boundary integral
        grad_d = np.zeros(2)
        for k in range(n_v):
            j = (k + 1) % n_v
            dx = verts[j, 0] - verts[k, 0]
            dy = verts[j, 1] - verts[k, 1]
            normal = np.array([dy, -dx])
            d_mid = 0.5 * (d_field[el_int[k]] + d_field[el_int[j]])
            grad_d += d_mid * normal
        grad_d /= area
        grad_d_mag = np.linalg.norm(grad_d)

        # Energy ratio: psi^+ / G_c
        Gc_i = Gc_field[i] if Gc_field[i] > 1e-15 else 1e-15
        energy_ratio = psi_history[i] / Gc_i

        eta_crack[i] = w1 * grad_d_mag + w2 * energy_ratio

    return eta_crack


# ── Adaptive Mesh Refinement at Crack Tip ────────────────────────────────


def refine_at_crack_tip(vertices, elements, eta_crack, theta=0.3,
                        domain=(0, 2, 0, 1)):
    """
    h-adaptive refinement driven by crack-tip indicator.

    Marks elements where eta > theta * max(eta), adds Voronoi seeds at
    edge midpoints of marked elements, and rebuilds the mesh.

    Uses the same Voronoi re-meshing pattern as refine_mesh_adaptive.

    Parameters
    ----------
    vertices : (n_nodes, 2) current mesh vertices
    elements : list of int arrays, element connectivity
    eta_crack : (n_el,) crack-tip indicator per element
    theta : Dorfler marking fraction
    domain : (xmin, xmax, ymin, ymax)

    Returns
    -------
    new_vertices, new_elements, new_boundary, marked
    """
    xmin, xmax, ymin, ymax = domain

    # Current seeds ~ element centroids
    seeds = []
    for el in elements:
        el_int = el.astype(int)
        seeds.append(vertices[el_int].mean(axis=0))
    seeds = np.array(seeds)

    # Mark elements with largest indicator
    eta_max = eta_crack.max()
    if eta_max < 1e-15:
        # Nothing to refine
        tol_bnd = 0.02
        bnd = np.where(
            (vertices[:, 0] < xmin + tol_bnd)
            | (vertices[:, 0] > xmax - tol_bnd)
            | (vertices[:, 1] < ymin + tol_bnd)
            | (vertices[:, 1] > ymax - tol_bnd)
        )[0]
        return vertices, elements, bnd, np.array([], dtype=int)

    threshold = theta * eta_max
    marked = np.where(eta_crack > threshold)[0]

    # Add new seeds at edge midpoints of marked elements
    new_seeds = list(seeds)
    for idx in marked:
        el_int = elements[idx].astype(int)
        verts = vertices[el_int]
        n_v = len(el_int)
        for k in range(n_v):
            mid = 0.5 * (verts[k] + verts[(k + 1) % n_v])
            new_seeds.append(mid)

    new_seeds = np.array(new_seeds)

    # Clip to domain interior
    new_seeds[:, 0] = np.clip(new_seeds[:, 0], xmin + 0.01, xmax - 0.01)
    new_seeds[:, 1] = np.clip(new_seeds[:, 1], ymin + 0.01, ymax - 0.01)

    # Remove duplicates
    unique = [new_seeds[0]]
    for s in new_seeds[1:]:
        if all(np.linalg.norm(s - u) > 0.005 for u in unique):
            unique.append(s)
    new_seeds = np.array(unique)

    # Rebuild Voronoi mesh
    from scipy.spatial import Voronoi

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
        if (
            xmin - 0.1 <= cell_c[0] <= xmax + 0.1
            and ymin - 0.1 <= cell_c[1] <= ymax + 0.1
        ):
            new_elements.append(face)

    tol_bnd = 0.02
    bnd = np.where(
        (unique_verts[:, 0] < xmin + tol_bnd)
        | (unique_verts[:, 0] > xmax - tol_bnd)
        | (unique_verts[:, 1] < ymin + tol_bnd)
        | (unique_verts[:, 1] > ymax - tol_bnd)
    )[0]

    return unique_verts, new_elements, bnd, marked


# ── Field Transfer ───────────────────────────────────────────────────────


def transfer_fields(old_verts, old_elems, old_d, old_psi,
                    new_verts, new_elems):
    """
    Transfer phase-field d and psi history from old mesh to new mesh.

    - d (node-based): nearest-neighbor interpolation from old nodes
    - psi (element-based): nearest old element centroid for each new element

    Parameters
    ----------
    old_verts : (n_old_nodes, 2)
    old_elems : list of int arrays
    old_d : (n_old_nodes,) phase-field at old nodes
    old_psi : (n_old_el,) psi history at old elements
    new_verts : (n_new_nodes, 2)
    new_elems : list of int arrays

    Returns
    -------
    new_d : (n_new_nodes,) transferred phase-field
    new_psi : (n_new_el,) transferred psi history
    """
    n_new_nodes = len(new_verts)
    n_new_el = len(new_elems)

    # ── d: nearest-neighbor from old nodes ──
    new_d = np.zeros(n_new_nodes)
    for i in range(n_new_nodes):
        dists = np.linalg.norm(old_verts - new_verts[i], axis=1)
        nearest = np.argmin(dists)
        new_d[i] = old_d[nearest]

    # ── psi: nearest old element centroid ──
    old_centroids = np.zeros((len(old_elems), 2))
    for j, el in enumerate(old_elems):
        old_centroids[j] = old_verts[el.astype(int)].mean(axis=0)

    new_psi = np.zeros(n_new_el)
    for i, el in enumerate(new_elems):
        cx = new_verts[el.astype(int)].mean(axis=0)
        dists = np.linalg.norm(old_centroids - cx, axis=1)
        nearest = np.argmin(dists)
        new_psi[i] = old_psi[nearest]

    return new_d, new_psi


# ── Adaptive Phase-Field VEM Solver ──────────────────────────────────────


class AdaptivePhaseFieldVEM:
    """
    Phase-field fracture VEM with automatic h-refinement at crack tip.

    Crack tip = high |grad d| + high psi^+/G_c -> mesh refinement.
    Coarse mesh far from crack, fine mesh at crack tip.

    Staggered solve per load step:
      1. Fix d -> solve displacement u (degraded VEM stiffness)
      2. Fix u -> compute psi^+ -> solve phase-field d
      3. Enforce irreversibility d_new >= d_old
    Periodically: compute crack_tip_indicator, refine mesh, transfer fields.
    """

    def __init__(
        self,
        vertices,
        elements,
        DI_per_cell,
        nu=0.35,
        Gc_max=0.5,
        Gc_min=0.01,
        l0=None,
        refine_interval=5,
        max_refine_levels=3,
        theta_refine=0.3,
        domain=(0, 2, 0, 1),
    ):
        self.vertices = np.array(vertices, dtype=float)
        self.elements = [np.asarray(el, dtype=int) for el in elements]
        self.DI_per_cell = np.array(DI_per_cell, dtype=float)
        self.nu = nu
        self.Gc_max = Gc_max
        self.Gc_min = Gc_min
        self.l0 = l0
        self.refine_interval = refine_interval
        self.max_refine_levels = max_refine_levels
        self.theta_refine = theta_refine
        self.domain = domain

        # Derived fields
        self.E_field = compute_E_from_DI(self.DI_per_cell)
        self.Gc_field = compute_Gc(self.DI_per_cell, Gc_max=Gc_max, Gc_min=Gc_min)

        # Phase-field solver
        self._build_solver()

        # Tracking
        self.refine_count = 0
        self.mesh_history = []

    def _build_solver(self):
        """Create PhaseFieldVEM solver on current mesh."""
        self.solver = PhaseFieldVEM(
            self.vertices,
            self.elements,
            self.E_field,
            self.nu,
            self.Gc_field,
            l0=self.l0,
        )

    def _setup_bcs(self, vertices):
        """Bottom fixed, top loaded."""
        xmin, xmax, ymin, ymax = self.domain
        tol_bc = 0.02

        bottom = np.where(vertices[:, 1] < ymin + tol_bc)[0]
        bc_dofs = np.concatenate([2 * bottom, 2 * bottom + 1])
        bc_vals = np.zeros(len(bc_dofs))

        top = np.where(vertices[:, 1] > ymax - tol_bc)[0]
        return bc_dofs, bc_vals, top

    def _compute_DI_spatial(self, vertices, elements):
        """Compute DI per element from spatial position."""
        xmin, xmax, ymin, ymax = self.domain
        xmid = (xmin + xmax) / 2
        ymid = (ymin + ymax) / 2
        r_max = np.sqrt((xmid - xmin) ** 2 + (ymid - ymin) ** 2)

        DI = np.zeros(len(elements))
        for i, el in enumerate(elements):
            el_int = el.astype(int)
            cx = np.mean(vertices[el_int, 0])
            cy = np.mean(vertices[el_int, 1])
            r = np.sqrt((cx - xmid) ** 2 + (cy - ymid) ** 2)
            proximity = 1.0 - r / r_max
            DI[i] = np.clip(0.15 + 0.65 * proximity, 0.0, 1.0)

        return DI

    def _do_refinement(self, verbose=False):
        """Perform one refinement step based on crack-tip indicator."""
        if self.refine_count >= self.max_refine_levels:
            if verbose:
                print(f"    [Refine] Max levels ({self.max_refine_levels}) reached, skipping.")
            return False

        eta = crack_tip_indicator(
            self.solver.d,
            self.vertices,
            self.elements,
            self.solver.psi_history,
            self.Gc_field,
        )

        if eta.max() < 1e-10:
            if verbose:
                print("    [Refine] Indicator too small, skipping.")
            return False

        new_verts, new_elems, new_bnd, marked = refine_at_crack_tip(
            self.vertices,
            self.elements,
            eta,
            theta=self.theta_refine,
            domain=self.domain,
        )

        if len(marked) == 0:
            if verbose:
                print("    [Refine] No elements marked, skipping.")
            return False

        # Transfer fields
        old_d = self.solver.d.copy()
        old_psi = self.solver.psi_history.copy()

        new_d, new_psi = transfer_fields(
            self.vertices, self.elements, old_d, old_psi,
            new_verts, new_elems,
        )

        # Compact mesh
        used_set = set()
        for el in new_elems:
            used_set.update(el.astype(int).tolist())
        used = np.array(sorted(used_set))
        old_to_new = {int(g): i for i, g in enumerate(used)}

        compact_verts = new_verts[used]
        compact_elems = [
            np.array([old_to_new[int(v)] for v in el]) for el in new_elems
        ]

        compact_d = np.zeros(len(used))
        for old_idx, new_idx in old_to_new.items():
            if old_idx < len(new_d):
                compact_d[new_idx] = new_d[old_idx]

        new_DI = self._compute_DI_spatial(compact_verts, compact_elems)
        new_E = compute_E_from_DI(new_DI)
        new_Gc = compute_Gc(new_DI, Gc_max=self.Gc_max, Gc_min=self.Gc_min)

        n_old_cells = len(self.elements)
        self.vertices = compact_verts
        self.elements = compact_elems
        self.DI_per_cell = new_DI
        self.E_field = new_E
        self.Gc_field = new_Gc

        self._build_solver()
        self.solver.d = np.clip(compact_d, 0.0, 1.0)
        self.solver.psi_history = new_psi.copy()

        self.refine_count += 1
        self.mesh_history.append(
            (len(self.elements), len(self.vertices))
        )

        if verbose:
            print(
                f"    [Refine] Level {self.refine_count}: "
                f"marked {len(marked)}/{n_old_cells} -> "
                f"{len(self.elements)} cells, {len(self.vertices)} nodes"
            )

        return True

    def run(
        self,
        n_load_steps=30,
        load_factor_max=3.0,
        max_stagger=30,
        tol=1e-4,
        verbose=True,
    ):
        """
        Incremental loading with staggered phase-field solve and
        periodic adaptive refinement.

        Returns list of snapshot dicts per load step.
        """
        snapshots = []
        self.mesh_history.append(
            (len(self.elements), len(self.vertices))
        )

        for step in range(n_load_steps):
            bc_dofs, bc_vals, top = self._setup_bcs(self.vertices)
            n_top = len(top)

            lf = (step + 1) / n_load_steps * load_factor_max
            l_dofs_list = []
            l_vals_list = []
            if n_top > 0:
                l_dofs_list.append(2 * top)
                l_vals_list.append(np.full(n_top, lf / n_top))
                l_dofs_list.append(2 * top + 1)
                l_vals_list.append(np.full(n_top, -lf * 0.3 / n_top))
            l_dofs = np.concatenate(l_dofs_list) if l_dofs_list else None
            l_vals = np.concatenate(l_vals_list) if l_vals_list else None

            # Staggered solve
            d_old = self.solver.d.copy()
            for stag_iter in range(max_stagger):
                self.solver.solve_displacement(bc_dofs, bc_vals, l_dofs, l_vals)
                psi_plus = self.solver.compute_psi_plus_field()
                d_new = self.solver.solve_phase_field(psi_plus)

                d_change = np.linalg.norm(d_new - d_old) / max(
                    np.linalg.norm(d_new), 1e-10
                )

                if verbose and stag_iter % 5 == 0:
                    print(
                        f"  Step {step+1}/{n_load_steps}, stagger {stag_iter}: "
                        f"|Dd|/|d| = {d_change:.2e}, "
                        f"max(d) = {np.max(self.solver.d):.4f}, "
                        f"max(psi+) = {np.max(psi_plus):.2e}"
                    )

                if d_change < tol:
                    break
                d_old = d_new.copy()

            ux = self.solver.u[0::2]
            uy = self.solver.u[1::2]
            mag = np.sqrt(ux**2 + uy**2)

            snap = {
                "step": step,
                "u": self.solver.u.copy(),
                "d": self.solver.d.copy(),
                "psi_history": self.solver.psi_history.copy(),
                "u_max": np.max(mag),
                "d_max": np.max(self.solver.d),
                "d_mean": np.mean(self.solver.d),
                "psi_max": np.max(psi_plus),
                "stagger_iters": stag_iter + 1,
                "n_cracked": int(np.sum(self.solver.d > 0.9)),
                "n_cells": len(self.elements),
                "n_nodes": len(self.vertices),
                "vertices": self.vertices.copy(),
                "elements": [el.copy() for el in self.elements],
                "DI_per_cell": self.DI_per_cell.copy(),
                "E_field": self.E_field.copy(),
                "Gc_field": self.Gc_field.copy(),
                "refine_level": self.refine_count,
            }
            snapshots.append(snap)

            if verbose:
                print(
                    f"  -> Step {step+1} done: |u|_max={snap['u_max']:.4e}, "
                    f"d_max={snap['d_max']:.4f}, cracked={snap['n_cracked']}, "
                    f"cells={snap['n_cells']}"
                )

            # Periodic adaptive refinement
            if (
                (step + 1) % self.refine_interval == 0
                and self.refine_count < self.max_refine_levels
            ):
                if verbose:
                    print(f"\n  --- Adaptive refinement at step {step+1} ---")
                self._do_refinement(verbose=verbose)
                if verbose:
                    print()

        return snapshots


# ── Demo ─────────────────────────────────────────────────────────────────


def demo_adaptive_fracture():
    """
    Demo: adaptive h-refinement coupled with phase-field fracture VEM.

    40-cell biofilm, dysbiotic center (high DI, low Gc) cracks first.
    Mesh automatically refines at crack tip every 5 steps.

    Generates 2x3 figure:
      Row 1: Initial mesh with DI, mesh after 1st refinement, mesh after 2nd refinement
      Row 2: Final phase-field d, deformed mesh, convergence plot
    """
    print("=" * 70)
    print("Adaptive Phase-Field Fracture VEM: Biofilm Detachment")
    print("=" * 70)

    rng = np.random.default_rng(42)
    domain = (0, 2, 0, 1)
    n_cells = 40
    xmin, xmax, ymin, ymax = domain
    nu = 0.35

    # ── Generate initial mesh ──
    nx = int(np.sqrt(n_cells * 2))
    ny = max(n_cells // nx, 2)
    xx = np.linspace(xmin + 0.1, xmax - 0.1, nx)
    yy = np.linspace(ymin + 0.05, ymax - 0.05, ny)
    gx, gy = np.meshgrid(xx, yy)
    seeds = np.column_stack([gx.ravel(), gy.ravel()])[:n_cells]
    seeds += rng.uniform(-0.03, 0.03, seeds.shape)

    vertices, elements, bnd, valid_ids = make_biofilm_voronoi(seeds, domain)
    n_el = len(elements)

    # ── Spatial DI gradient ──
    xmid = (xmin + xmax) / 2
    ymid = (ymin + ymax) / 2
    DI_per_cell = np.zeros(n_el)
    for i, el in enumerate(elements):
        el_int = el.astype(int)
        cx = np.mean(vertices[el_int, 0])
        cy = np.mean(vertices[el_int, 1])
        r = np.sqrt((cx - xmid) ** 2 + (cy - ymid) ** 2)
        r_max = np.sqrt((xmid - xmin) ** 2 + (ymid - ymin) ** 2)
        proximity = 1.0 - r / r_max
        DI_per_cell[i] = np.clip(0.15 + 0.65 * proximity, 0.0, 1.0)

    # ── Compact mesh ──
    used_set = set()
    for el in elements:
        used_set.update(el.astype(int).tolist())
    used = np.array(sorted(used_set))
    old_to_new = {int(g): i for i, g in enumerate(used)}

    compact_verts = vertices[used]
    compact_elems = [np.array([old_to_new[int(v)] for v in el]) for el in elements]

    print(f"  Initial mesh: {len(compact_elems)} cells, {len(compact_verts)} nodes")
    print(f"  DI range: [{DI_per_cell.min():.3f}, {DI_per_cell.max():.3f}]")
    print(f"  E range: [{compute_E_from_DI(DI_per_cell).max():.0f}, "
          f"{compute_E_from_DI(DI_per_cell).min():.0f}] Pa")
    print()

    # ── Run adaptive solver ──
    solver = AdaptivePhaseFieldVEM(
        compact_verts,
        compact_elems,
        DI_per_cell,
        nu=nu,
        Gc_max=0.5,
        Gc_min=0.01,
        refine_interval=5,
        max_refine_levels=3,
        theta_refine=0.3,
        domain=domain,
    )

    snapshots = solver.run(
        n_load_steps=30,
        load_factor_max=3.0,
        max_stagger=30,
        tol=1e-4,
        verbose=True,
    )

    # ── Identify key snapshots ──
    snap_initial = snapshots[0]
    snap_final = snapshots[-1]

    refine_snaps = []
    prev_level = 0
    for s in snapshots:
        if s["refine_level"] > prev_level:
            refine_snaps.append(s)
            prev_level = s["refine_level"]
    while len(refine_snaps) < 2:
        refine_snaps.append(snap_final)

    # ── Plot 2x3 figure ──
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    def _plot_mesh_field(ax, verts, elems, values, cmap, label, title,
                         clim=None):
        patches = [MplPolygon(verts[el.astype(int)], closed=True) for el in elems]
        pc = PatchCollection(patches, cmap=cmap, edgecolor="k", linewidth=0.3)
        pc.set_array(np.array(values))
        if clim is not None:
            pc.set_clim(*clim)
        ax.add_collection(pc)
        ax.set_xlim(xmin - 0.05, xmax + 0.05)
        ax.set_ylim(ymin - 0.05, ymax + 0.05)
        ax.set_aspect("equal")
        fig.colorbar(pc, ax=ax, label=label, shrink=0.8)
        ax.set_title(title)

    # Row 1: Mesh evolution
    _plot_mesh_field(
        axes[0, 0],
        snap_initial["vertices"],
        snap_initial["elements"],
        snap_initial["DI_per_cell"],
        "RdYlGn_r", "DI",
        f"(a) Initial: {snap_initial['n_cells']} cells\nDI field",
    )

    s1 = refine_snaps[0]
    _plot_mesh_field(
        axes[0, 1],
        s1["vertices"], s1["elements"], s1["DI_per_cell"],
        "RdYlGn_r", "DI",
        f"(b) After refine 1: {s1['n_cells']} cells\nDI field",
    )

    s2 = refine_snaps[1]
    _plot_mesh_field(
        axes[0, 2],
        s2["vertices"], s2["elements"], s2["DI_per_cell"],
        "RdYlGn_r", "DI",
        f"(c) After refine 2: {s2['n_cells']} cells\nDI field",
    )

    # Row 2: Final results
    d_per_cell_final = np.array(
        [np.mean(snap_final["d"][el.astype(int)]) for el in snap_final["elements"]]
    )
    _plot_mesh_field(
        axes[1, 0],
        snap_final["vertices"], snap_final["elements"],
        d_per_cell_final, "inferno", "d (damage)",
        f"(d) Phase-field d, max={snap_final['d_max']:.3f}",
        clim=(0, 1),
    )

    # Deformed mesh
    final_verts = snap_final["vertices"]
    final_u = snap_final["u"]
    ux = final_u[0::2]
    uy = final_u[1::2]
    mag = np.sqrt(ux**2 + uy**2)
    scale = 20.0
    deformed = final_verts + scale * np.column_stack([ux, uy])

    colors_disp = [np.mean(mag[el.astype(int)]) for el in snap_final["elements"]]
    patches_def = [
        MplPolygon(deformed[el.astype(int)], closed=True)
        for el in snap_final["elements"]
    ]
    pc_def = PatchCollection(patches_def, cmap="hot_r", edgecolor="k", linewidth=0.3)
    pc_def.set_array(np.array(colors_disp))
    axes[1, 1].add_collection(pc_def)
    axes[1, 1].set_xlim(xmin - 0.2, xmax + 0.5)
    axes[1, 1].set_ylim(ymin - 0.2, ymax + 0.2)
    axes[1, 1].set_aspect("equal")
    fig.colorbar(pc_def, ax=axes[1, 1], label="|u|", shrink=0.8)
    axes[1, 1].set_title(
        f"(e) Deformed (x{scale:.0f}), |u|_max={snap_final['u_max']:.3e}"
    )

    # Convergence plot
    ax_conv = axes[1, 2]
    steps_arr = []
    eta_totals = []
    n_cells_arr = []
    for s in snapshots:
        steps_arr.append(s["step"] + 1)
        n_cells_arr.append(s["n_cells"])
        eta_s = crack_tip_indicator(
            s["d"], s["vertices"], s["elements"],
            s["psi_history"], s["Gc_field"],
        )
        eta_totals.append(np.sqrt(np.sum(eta_s**2)))

    color_eta = "tab:red"
    ax_conv.plot(steps_arr, eta_totals, "o-", color=color_eta, ms=3, lw=1.5,
                 label=r"$\eta_{total}$")
    ax_conv.set_xlabel("Load Step")
    ax_conv.set_ylabel(r"$\eta_{total}$ (crack indicator)", color=color_eta)
    ax_conv.tick_params(axis="y", labelcolor=color_eta)

    for rs in refine_snaps:
        ax_conv.axvline(rs["step"] + 1, color="green", alpha=0.4,
                        linestyle="--")

    ax_conv2 = ax_conv.twinx()
    color_cells = "tab:blue"
    ax_conv2.plot(steps_arr, n_cells_arr, "s-", color=color_cells, ms=3,
                  lw=1.5, label="# cells")
    ax_conv2.set_ylabel("Number of cells", color=color_cells)
    ax_conv2.tick_params(axis="y", labelcolor=color_cells)

    ax_conv.legend(loc="upper left")
    ax_conv2.legend(loc="upper right")
    ax_conv.set_title(r"(f) $\eta_{total}$ and cell count")
    ax_conv.grid(True, alpha=0.3)

    fig.suptitle(
        "Adaptive Phase-Field VEM: Biofilm Detachment\n"
        "(h-refinement at crack tip — dysbiotic center cracks first)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()

    import os
    save_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "vem_adaptive_fracture_demo.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {path}")
    plt.close()

    # ── Summary ──
    print("\n" + "=" * 70)
    print("Summary:")
    print(f"  Load steps:        {len(snapshots)}")
    print(f"  Refinement levels: {solver.refine_count}")
    print(f"  Initial cells:     {snapshots[0]['n_cells']}")
    print(f"  Final cells:       {snapshots[-1]['n_cells']}")
    print(f"  Final d_max:       {snapshots[-1]['d_max']:.4f}")
    print(f"  Final |u|_max:     {snapshots[-1]['u_max']:.4e}")
    print(f"  Cracked nodes:     {snapshots[-1]['n_cracked']}")
    print("=" * 70)


if __name__ == "__main__":
    demo_adaptive_fracture()
