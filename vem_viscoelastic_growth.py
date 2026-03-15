#!/usr/bin/env python3
"""
vem_viscoelastic_growth.py -- Growth-Coupled Viscoelastic VEM
=============================================================

Staggered coupling: Hamilton ODE species dynamics → DI(t) → SLS params(t) → VE-VEM.

At each macro time step:
  1. Advance species ODE (Hamilton replicator, n_substeps)
  2. Update DI, SLS parameters per cell
  3. Solve VE-VEM with updated algorithmic tangent + internal variable history
  4. Optional: cell division + re-meshing

Key difference from static growth-coupled:
  - Internal variable h(t) carries memory of deformation history
  - Material parameters E_inf, E_1, tau evolve as DI changes
  - Simo 1987 exponential integrator handles time-varying constitutive law

Reference: Klempt et al. (2024) — adiabatic scale separation for staggered coupling.
"""

import numpy as np
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from pathlib import Path
import sys, os

sys.path.insert(0, os.path.dirname(__file__))

from vem_growth_coupled import (
    make_interaction_matrix, hamilton_step, compute_DI, compute_E,
    make_biofilm_voronoi, SPECIES_NAMES,
)
from vem_viscoelastic import (
    sls_params_from_di, _plane_stress_C, _compute_element_vem,
    _assemble_viscoelastic_step,
)


class ViscoelasticGrowthVEM:
    """
    Staggered growth-coupled viscoelastic VEM.

    Combines:
    - Hamilton ODE for 5-species replicator dynamics
    - DI-dependent SLS (Standard Linear Solid) material model
    - P₁ VEM spatial discretization
    - Simo 1987 exponential integrator for internal variables
    """

    def __init__(self, n_cells=30, condition='dh_baseline',
                 domain=(0, 2, 0, 1), seed=42, nu=0.35):
        self.condition = condition
        self.domain = domain
        self.nu = nu
        self.rng = np.random.default_rng(seed)
        self.A = make_interaction_matrix(condition)

        xmin, xmax, ymin, ymax = domain
        nx = int(np.sqrt(n_cells * (xmax - xmin) / (ymax - ymin)))
        ny = max(int(n_cells / nx), 2)
        xx = np.linspace(xmin + 0.05, xmax - 0.05, nx)
        yy = np.linspace(ymin + 0.05, ymax - 0.05, ny)
        gx, gy = np.meshgrid(xx, yy)
        seeds = np.column_stack([gx.ravel(), gy.ravel()])
        seeds += self.rng.uniform(-0.03, 0.03, seeds.shape)
        self.seeds = seeds[:n_cells]

        self._init_species()
        self._build_mesh()

        # Viscoelastic state
        self.h_all = None  # internal variable per element (n_el, 3)
        self.eps_prev = None  # previous strain per element (n_el, 3)
        self.t_current = 0.0

        self.history = []

    def _init_species(self):
        """Initialize species fractions per cell."""
        n = len(self.seeds)
        self.phi = np.zeros((n, 5))

        xmid = (self.domain[0] + self.domain[1]) / 2
        ymid = (self.domain[2] + self.domain[3]) / 2

        for i in range(n):
            x, y = self.seeds[i]
            r = np.sqrt((x - xmid)**2 + (y - ymid)**2)
            r_max = np.sqrt((xmid - self.domain[0])**2 + (ymid - self.domain[2])**2)
            proximity = 1.0 - r / r_max

            if proximity > 0.6:
                self.phi[i] = [0.10, 0.35, 0.20, 0.20, 0.15]
            elif proximity > 0.3:
                self.phi[i] = [0.25, 0.30, 0.20, 0.15, 0.10]
            else:
                self.phi[i] = [0.40, 0.25, 0.20, 0.12, 0.03]

    def _build_mesh(self):
        """Build Voronoi mesh and compact node indexing."""
        self.vertices, self.elements, self.boundary, self.valid_ids = \
            make_biofilm_voronoi(self.seeds, self.domain)
        self.n_cells = len(self.elements)

        # Compact indexing
        used_set = set()
        for el in self.elements:
            used_set.update(el.astype(int).tolist())
        self.used = np.array(sorted(used_set))
        self.old_to_new = {int(g): i for i, g in enumerate(self.used)}
        self.n_used = len(self.used)
        self.compact_verts = self.vertices[self.used]
        self.compact_elems = [np.array([self.old_to_new[int(v)] for v in el])
                              for el in self.elements]

    def compute_properties(self):
        """Compute DI, SLS params for all cells."""
        self.DI = np.zeros(self.n_cells)
        for i in range(self.n_cells):
            cell_id = self.valid_ids[i] if i < len(self.valid_ids) else i
            if cell_id < len(self.phi):
                self.DI[i] = compute_DI(self.phi[cell_id])
            else:
                self.DI[i] = 0.5

        self.sls_params = sls_params_from_di(self.DI)

    def grow_step(self, dt=0.5, n_substeps=5):
        """Advance species ODE."""
        dt_sub = dt / n_substeps
        for _ in range(n_substeps):
            for i in range(min(self.n_cells, len(self.valid_ids))):
                cell_id = self.valid_ids[i]
                if cell_id < len(self.phi):
                    self.phi[cell_id] = hamilton_step(
                        self.phi[cell_id], self.A, dt=dt_sub)

    def solve_ve_vem_step(self, dt):
        """
        One viscoelastic VEM time step with current SLS parameters.

        Updates internal variables h_all and eps_prev.
        """
        n_el = self.n_cells
        n_dofs = 2 * self.n_used

        # Compute per-element constitutive matrices
        E_inf = self.sls_params["E_inf"]
        E_1 = self.sls_params["E_1"]
        tau = self.sls_params["tau"]

        C_inf_all = np.zeros((n_el, 3, 3))
        C_1_all = np.zeros((n_el, 3, 3))
        for k in range(n_el):
            C_inf_all[k] = _plane_stress_C(E_inf[k], self.nu)
            C_1_all[k] = _plane_stress_C(E_1[k], self.nu)

        # Precompute VEM element data
        elem_data = []
        for el_id in range(n_el):
            verts = self.compact_verts[self.compact_elems[el_id].astype(int)]
            elem_data.append(_compute_element_vem(verts, self.nu))

        # Initialize state if needed
        if self.h_all is None:
            self.h_all = np.zeros((n_el, 3))
            self.eps_prev = np.zeros((n_el, 3))

        # Simo coefficients
        if dt > 1e-15:
            exp_dt = np.exp(-dt / tau)
            gamma = (tau / dt) * (1.0 - exp_dt)
        else:
            exp_dt = np.ones(n_el)
            gamma = np.ones(n_el)

        # Algorithmic tangent
        C_alg_all = C_inf_all + gamma[:, None, None] * C_1_all

        # Effective history stress
        h_star = np.zeros((n_el, 3))
        for k in range(n_el):
            h_star[k] = exp_dt[k] * self.h_all[k] - gamma[k] * C_1_all[k] @ self.eps_prev[k]

        # BCs: fix bottom, gravity + GCF on top
        xmin, xmax, ymin, ymax = self.domain
        tol = 0.02
        bottom = np.where(self.compact_verts[:, 1] < ymin + tol)[0]
        top = np.where(self.compact_verts[:, 1] > ymax - tol)[0]

        bc_dofs = np.concatenate([2 * bottom, 2 * bottom + 1])
        bc_vals = np.zeros(len(bc_dofs))

        # External load: gravity + GCF pressure
        F_ext = np.zeros(n_dofs)
        gravity = -0.005 / max(self.n_used, 1)
        F_ext[1::2] += gravity  # y-component for all nodes
        if len(top) > 0:
            gcf = -0.01 / len(top)
            F_ext[2 * top + 1] += gcf

        # Solve
        u = _assemble_viscoelastic_step(
            self.compact_verts, self.compact_elems, elem_data,
            C_alg_all, h_star, bc_dofs, bc_vals, F_ext,
        )

        # Post-process: update strains and internal variables
        sigma_el = np.zeros((n_el, 3))
        for k in range(n_el):
            el_int = self.compact_elems[k].astype(int)
            ed = elem_data[k]
            n_v = ed["n_v"]
            gdofs = np.zeros(ed["n_el_dofs"], dtype=int)
            for i in range(n_v):
                gdofs[2 * i] = 2 * el_int[i]
                gdofs[2 * i + 1] = 2 * el_int[i] + 1

            u_el = u[gdofs]
            eps_new = ed["strain_proj"] @ u_el

            self.h_all[k] = exp_dt[k] * self.h_all[k] + \
                gamma[k] * C_1_all[k] @ (eps_new - self.eps_prev[k])
            sigma_el[k] = C_inf_all[k] @ eps_new + self.h_all[k]
            self.eps_prev[k] = eps_new.copy()

        self.u_compact = u
        self.sigma = sigma_el

        # Map back to full vertex array
        self.u = np.zeros(2 * len(self.vertices))
        for new_i, old_i in enumerate(self.used):
            self.u[2 * old_i] = u[2 * new_i]
            self.u[2 * old_i + 1] = u[2 * new_i + 1]

    def run(self, n_steps=30, dt_growth=0.5, dt_ve=1.0,
            ve_substeps=5, verbose=True):
        """
        Run growth-coupled viscoelastic simulation.

        Parameters
        ----------
        n_steps : int -- number of macro growth steps
        dt_growth : float -- ODE time step per growth step
        dt_ve : float -- total viscoelastic time per growth step
        ve_substeps : int -- VE-VEM time steps per growth step
        """
        if verbose:
            print("=" * 65)
            print(f"Viscoelastic Growth-Coupled VEM: {self.condition}")
            print(f"  Cells: {self.n_cells}, Growth dt={dt_growth}, "
                  f"VE dt={dt_ve} ({ve_substeps} substeps)")
            print("=" * 65)

        dt_sub = dt_ve / ve_substeps

        for step in range(n_steps):
            # 1. Grow species
            self.grow_step(dt=dt_growth)

            # 2. Update material (DI → SLS params)
            self.compute_properties()

            # 3. Viscoelastic solve (multiple substeps)
            for _ in range(ve_substeps):
                self.solve_ve_vem_step(dt_sub)
                self.t_current += dt_sub

            # von Mises stress (mean)
            vm = np.sqrt(self.sigma[:, 0]**2 + self.sigma[:, 1]**2
                         - self.sigma[:, 0] * self.sigma[:, 1]
                         + 3 * self.sigma[:, 2]**2)

            snapshot = {
                'step': step,
                't': self.t_current,
                'n_cells': self.n_cells,
                'DI_mean': np.mean(self.DI),
                'DI_std': np.std(self.DI),
                'E_inf_mean': np.mean(self.sls_params["E_inf"]),
                'E_inf_range': (np.min(self.sls_params["E_inf"]),
                                np.max(self.sls_params["E_inf"])),
                'tau_mean': np.mean(self.sls_params["tau"]),
                'sigma_vm_mean': np.mean(vm),
                'sigma_vm_max': np.max(vm),
                'u_max': np.max(np.sqrt(self.u_compact[0::2]**2 +
                                         self.u_compact[1::2]**2)),
                'h_norm': np.mean(np.linalg.norm(self.h_all, axis=1)),
            }
            self.history.append(snapshot)

            if verbose and (step % 5 == 0 or step == n_steps - 1):
                print(f"  step={step:3d} t={self.t_current:6.1f}s | "
                      f"DI={snapshot['DI_mean']:.3f} | "
                      f"E_inf=[{snapshot['E_inf_range'][0]:.0f},"
                      f"{snapshot['E_inf_range'][1]:.0f}] Pa | "
                      f"tau={snapshot['tau_mean']:.1f}s | "
                      f"σ_vm={snapshot['sigma_vm_mean']:.4f} Pa | "
                      f"|h|={snapshot['h_norm']:.4f}")

        if verbose:
            print(f"\n  Final: {self.n_cells} cells, t={self.t_current:.1f}s")

        return self.history


def demo_viscoelastic_growth():
    """Demo: 3-condition viscoelastic growth comparison."""
    print("=" * 65)
    print("Demo: Viscoelastic Growth-Coupled VEM — 3 Conditions")
    print("=" * 65)

    conditions = ["commensal_static", "dh_baseline", "dysbiotic_static"]
    labels = ["Commensal (CS)", "DH Baseline", "Dysbiotic (DS)"]
    colors = ["#2166ac", "#e08214", "#d73027"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    sims = {}

    for idx, (cond, lab, col) in enumerate(zip(conditions, labels, colors)):
        print(f"\n--- {lab} ---")
        sim = ViscoelasticGrowthVEM(n_cells=25, condition=cond, seed=42)
        sim.run(n_steps=20, dt_growth=0.5, dt_ve=2.0, ve_substeps=4, verbose=True)
        sims[cond] = sim

        # Row 1: DI field
        ax = axes[0, idx]
        patches = [MplPolygon(sim.compact_verts[el.astype(int)], closed=True)
                   for el in sim.compact_elems]
        pc = PatchCollection(patches, cmap="RdYlGn_r", edgecolor="k", linewidth=0.3)
        pc.set_array(sim.DI)
        pc.set_clim(0, 0.8)
        ax.add_collection(pc)
        ax.set_xlim(-0.05, 2.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal")
        fig.colorbar(pc, ax=ax, label="DI", shrink=0.7)
        ax.set_title(f"{lab}\nDI={np.mean(sim.DI):.3f}, "
                     f"E_inf=[{np.min(sim.sls_params['E_inf']):.0f},"
                     f"{np.max(sim.sls_params['E_inf']):.0f}] Pa",
                     fontsize=10, fontweight="bold")

    # Row 2: time histories
    # (a) DI evolution
    ax = axes[1, 0]
    for cond, lab, col in zip(conditions, labels, colors):
        DIs = [h["DI_mean"] for h in sims[cond].history]
        ts = [h["t"] for h in sims[cond].history]
        ax.plot(ts, DIs, "-o", color=col, ms=2, lw=1.5, label=lab)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Mean DI")
    ax.set_title("(d) DI Evolution", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (b) sigma_vm evolution
    ax = axes[1, 1]
    for cond, lab, col in zip(conditions, labels, colors):
        svm = [h["sigma_vm_mean"] for h in sims[cond].history]
        ts = [h["t"] for h in sims[cond].history]
        ax.plot(ts, svm, "-s", color=col, ms=2, lw=1.5, label=lab)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Mean σ_vm [Pa]")
    ax.set_title("(e) Stress Relaxation", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (c) |h| internal variable norm
    ax = axes[1, 2]
    for cond, lab, col in zip(conditions, labels, colors):
        hn = [h["h_norm"] for h in sims[cond].history]
        ts = [h["t"] for h in sims[cond].history]
        ax.plot(ts, hn, "-^", color=col, ms=2, lw=1.5, label=lab)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("|h| (internal variable)")
    ax.set_title("(f) Memory Decay", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("Viscoelastic Growth-Coupled VEM: Hamilton ODE → DI(t) → SLS → VE-VEM",
                 fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "vem_viscoelastic_growth.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")

    # Summary table
    print("\n  Summary:")
    print(f"  {'Condition':<20s} {'DI':>6s} {'E_inf [Pa]':>12s} {'tau [s]':>8s} "
          f"{'σ_vm [Pa]':>10s} {'|h|':>8s}")
    print("  " + "-" * 68)
    for cond, lab in zip(conditions, labels):
        h = sims[cond].history[-1]
        print(f"  {lab:<20s} {h['DI_mean']:6.3f} "
              f"[{h['E_inf_range'][0]:>4.0f},{h['E_inf_range'][1]:>4.0f}] "
              f"{h['tau_mean']:8.1f} {h['sigma_vm_mean']:10.4f} {h['h_norm']:8.4f}")


if __name__ == "__main__":
    demo_viscoelastic_growth()
