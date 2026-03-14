#!/usr/bin/env python3
"""
Generate showcase figures for VEM biofilm mechanics.
Produces publication-quality visuals for README / wiki / presentation.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from vem_growth_coupled import (
    make_biofilm_voronoi, compute_DI, compute_E, BiofilmGrowthVEM,
    SPECIES_NAMES,
)
from vem_elasticity import vem_elasticity
from vem_nonlinear import vem_nonlinear, neo_hookean_params, _build_projector, _compute_deformation_gradient
from vem_phase_field import (
    PhaseFieldVEM, compute_Gc, compute_E_from_DI,
    compute_element_strains, assemble_degraded_elasticity_vem,
)

SAVE_DIR = os.path.join(os.path.dirname(__file__), "results", "showcase")
os.makedirs(SAVE_DIR, exist_ok=True)


def _make_mesh(n_cells=50, seed=42):
    """Generate a nice Voronoi mesh."""
    rng = np.random.default_rng(seed)
    domain = (0, 2, 0, 1)
    xmin, xmax, ymin, ymax = domain

    nx = int(np.sqrt(n_cells * 2))
    ny = max(n_cells // nx, 2)
    xx = np.linspace(xmin + 0.08, xmax - 0.08, nx)
    yy = np.linspace(ymin + 0.05, ymax - 0.05, ny)
    gx, gy = np.meshgrid(xx, yy)
    seeds = np.column_stack([gx.ravel(), gy.ravel()])[:n_cells]
    seeds += rng.uniform(-0.04, 0.04, seeds.shape)

    vertices, elements, bnd, valid_ids = make_biofilm_voronoi(seeds, domain)
    return vertices, elements, bnd, valid_ids, domain


def _compact_mesh(vertices, elements, domain):
    """Compact mesh: remove unused nodes, build BC arrays."""
    xmin, xmax, ymin, ymax = domain
    used_set = set()
    for el in elements:
        used_set.update(el.astype(int).tolist())
    used = np.array(sorted(used_set))
    old_to_new = {int(g): i for i, g in enumerate(used)}
    n_used = len(used)

    compact_verts = vertices[used]
    compact_elems = [np.array([old_to_new[int(v)] for v in el]) for el in elements]

    tol = 0.02
    bottom = np.where(compact_verts[:, 1] < ymin + tol)[0]
    top = np.where(compact_verts[:, 1] > ymax - tol)[0]
    bc_dofs = np.concatenate([2 * bottom, 2 * bottom + 1])
    bc_vals = np.zeros(len(bc_dofs))

    return compact_verts, compact_elems, bc_dofs, bc_vals, top, n_used


# ── Figure 1: Growth-Coupled VEM — 3 Conditions ──────────────────────────

def fig1_growth_coupled():
    """3-condition growth-coupled VEM comparison: CS, DH, DS."""
    print("Generating Fig 1: Growth-Coupled VEM...")

    conditions = ["commensal_static", "dh_baseline", "dysbiotic_static"]
    labels = ["Commensal (CS)", "DH Baseline", "Dysbiotic (DS)"]
    colors_cond = ["#2196F3", "#FF9800", "#F44336"]

    fig = plt.figure(figsize=(20, 14))

    # Run simulations
    sims = {}
    for cond in conditions:
        sim = BiofilmGrowthVEM(n_cells=35, condition=cond, seed=42)
        sim.run(n_steps=30, dt=0.8, division_interval=10, verbose=False)
        sims[cond] = sim

    # Row 1: DI fields (3 panels)
    for col, (cond, label) in enumerate(zip(conditions, labels)):
        ax = fig.add_subplot(3, 3, col + 1)
        sim = sims[cond]
        patches = [MplPolygon(sim.vertices[el.astype(int)], closed=True) for el in sim.elements]
        pc = PatchCollection(patches, cmap="RdYlGn_r", edgecolor="k", linewidth=0.4)
        pc.set_array(sim.DI)
        pc.set_clim(0, 0.8)
        ax.add_collection(pc)
        ax.set_xlim(-0.05, 2.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal")
        fig.colorbar(pc, ax=ax, label="DI", shrink=0.7)
        ax.set_title(f"{label}\nDI = {np.mean(sim.DI):.3f}", fontsize=12, fontweight="bold")

    # Row 2: E fields (3 panels)
    for col, (cond, label) in enumerate(zip(conditions, labels)):
        ax = fig.add_subplot(3, 3, col + 4)
        sim = sims[cond]
        patches = [MplPolygon(sim.vertices[el.astype(int)], closed=True) for el in sim.elements]
        pc = PatchCollection(patches, cmap="viridis", edgecolor="k", linewidth=0.4)
        pc.set_array(sim.E)
        pc.set_clim(30, 1000)
        ax.add_collection(pc)
        ax.set_xlim(-0.05, 2.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal")
        fig.colorbar(pc, ax=ax, label="E [Pa]", shrink=0.7)
        ax.set_title(f"E = [{np.min(sim.E):.0f}, {np.max(sim.E):.0f}] Pa", fontsize=11)

    # Row 3: Deformed mesh + species evolution
    # (a) Deformed overlay
    ax = fig.add_subplot(3, 3, 7)
    for cond, label, col_c in zip(conditions, labels, colors_cond):
        sim = sims[cond]
        ux = sim.u[0::2]
        uy = sim.u[1::2]
        scale = 200.0
        deformed = sim.vertices + scale * np.column_stack([ux, uy])
        for el in sim.elements:
            el_int = el.astype(int)
            poly = deformed[el_int]
            poly_closed = np.vstack([poly, poly[0]])
            ax.plot(poly_closed[:, 0], poly_closed[:, 1], color=col_c, linewidth=0.5, alpha=0.6)
    ax.set_xlim(-0.1, 2.3)
    ax.set_ylim(-0.1, 1.1)
    ax.set_aspect("equal")
    ax.set_title(f"Deformed (x{scale:.0f})", fontsize=11)
    ax.legend([plt.Line2D([0],[0],color=c,lw=2) for c in colors_cond], labels, fontsize=8)

    # (b) DI evolution
    ax = fig.add_subplot(3, 3, 8)
    for cond, label, col_c in zip(conditions, labels, colors_cond):
        sim = sims[cond]
        DIs = [h["DI_mean"] for h in sim.history]
        ax.plot(DIs, "-o", color=col_c, ms=2, lw=1.5, label=label)
    ax.set_xlabel("Growth Step")
    ax.set_ylabel("Mean DI")
    ax.set_title("DI Evolution", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (c) Summary table
    ax = fig.add_subplot(3, 3, 9)
    ax.axis("off")
    summary = "Condition    Cells  DI     E_range [Pa]    |u|_max\n"
    summary += "─" * 55 + "\n"
    for cond, label in zip(conditions, labels):
        sim = sims[cond]
        h = sim.history[-1]
        summary += f"{label:<16s} {h['n_cells']:>3d}   {h['DI_mean']:.3f}  [{h['E_min']:>4.0f}, {h['E_max']:>4.0f}]  {h['u_max']:.2e}\n"
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    fig.suptitle("Growth-Coupled VEM: Biofilm Staggered Simulation\n"
                 "Hamilton ODE → DI → E(DI) → VEM Elasticity → Stress Feedback",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(SAVE_DIR, "fig1_growth_coupled.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close()


# ── Figure 2: Neo-Hookean vs Linear VEM ───────────────────────────────────

def fig2_nonlinear():
    """Neo-Hookean vs Linear VEM under increasing load."""
    print("Generating Fig 2: Neo-Hookean vs Linear...")

    vertices, elements, bnd, valid_ids, domain = _make_mesh(n_cells=40, seed=42)
    compact_verts, compact_elems, bc_dofs, bc_vals, top, n_used = _compact_mesh(
        vertices, elements, domain
    )
    n_el = len(compact_elems)
    nu = 0.35

    # Dysbiotic biofilm (soft, large deformation)
    DI_val = 0.70
    E_val = compute_E(DI_val)
    E_field = np.full(n_el, E_val)

    # Sweep load magnitudes
    load_factors = np.linspace(0.1, 5.0, 15)
    u_max_linear = []
    u_max_nonlinear = []
    max_strains = []

    for lf in load_factors:
        l_dofs = np.concatenate([2 * top, 2 * top + 1]) if len(top) > 0 else np.array([], dtype=int)
        l_vals = np.concatenate([
            np.full(len(top), lf / max(len(top), 1)),
            np.full(len(top), -lf * 0.3 / max(len(top), 1)),
        ]) if len(top) > 0 else np.array([])

        u_lin = vem_elasticity(compact_verts, compact_elems, E_field, nu, bc_dofs, bc_vals, l_dofs, l_vals)
        u_nl, _ = vem_nonlinear(compact_verts, compact_elems, E_field, nu, bc_dofs, bc_vals,
                                l_dofs, l_vals, n_load_steps=8, verbose=False)

        mag_lin = np.max(np.sqrt(u_lin[0::2]**2 + u_lin[1::2]**2))
        mag_nl = np.max(np.sqrt(u_nl[0::2]**2 + u_nl[1::2]**2))
        u_max_linear.append(mag_lin)
        u_max_nonlinear.append(mag_nl)

        # Estimate max strain
        max_strain = 0.0
        for el in compact_elems:
            el_int = el.astype(int)
            verts_el = compact_verts[el_int]
            n_v = len(el_int)
            C_mat = (E_val / (1.0 - nu**2)) * np.array(
                [[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, (1.0 - nu) / 2.0]]
            )
            proj, _, _, area, h = _build_projector(verts_el, C_mat)
            gdofs_l = np.zeros(2 * n_v, dtype=int)
            for i in range(n_v):
                gdofs_l[2 * i] = 2 * el_int[i]
                gdofs_l[2 * i + 1] = 2 * el_int[i] + 1
            F_def = _compute_deformation_gradient(proj, u_nl[gdofs_l], h)
            E_gl = 0.5 * (F_def.T @ F_def - np.eye(2))
            max_strain = max(max_strain, np.max(np.abs(E_gl)))
        max_strains.append(max_strain)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # (a) Load-displacement curves
    ax = axes[0]
    ax.plot(load_factors, u_max_linear, "b-o", ms=4, lw=2, label="Linear VEM")
    ax.plot(load_factors, u_max_nonlinear, "r-s", ms=4, lw=2, label="Neo-Hookean VEM")
    ax.set_xlabel("Load Factor", fontsize=12)
    ax.set_ylabel("|u|_max", fontsize=12)
    ax.set_title("(a) Load-Displacement", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # (b) Relative difference
    ax = axes[1]
    rel_diff = [abs(nl - lin) / max(lin, 1e-15) * 100 for lin, nl in zip(u_max_linear, u_max_nonlinear)]
    ax.plot(max_strains, rel_diff, "g-^", ms=5, lw=2)
    ax.axhline(5, color="gray", linestyle="--", alpha=0.5, label="5% threshold")
    ax.axvline(0.05, color="orange", linestyle="--", alpha=0.5, label="ε = 5%")
    ax.set_xlabel("Max Green-Lagrange Strain", fontsize=12)
    ax.set_ylabel("Linear vs NL Difference [%]", fontsize=12)
    ax.set_title("(b) When Does Nonlinearity Matter?", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (c) Deformed mesh at highest load
    ax = axes[2]
    # Use the last load
    lf = load_factors[-1]
    l_dofs = np.concatenate([2 * top, 2 * top + 1]) if len(top) > 0 else np.array([], dtype=int)
    l_vals = np.concatenate([
        np.full(len(top), lf / max(len(top), 1)),
        np.full(len(top), -lf * 0.3 / max(len(top), 1)),
    ]) if len(top) > 0 else np.array([])

    u_lin = vem_elasticity(compact_verts, compact_elems, E_field, nu, bc_dofs, bc_vals, l_dofs, l_vals)
    u_nl, _ = vem_nonlinear(compact_verts, compact_elems, E_field, nu, bc_dofs, bc_vals,
                            l_dofs, l_vals, n_load_steps=10, verbose=False)

    scale = 30.0
    for u_vec, color, label in [(u_lin, "blue", "Linear"), (u_nl, "red", "Neo-Hookean")]:
        deformed = compact_verts + scale * np.column_stack([u_vec[0::2], u_vec[1::2]])
        for el in compact_elems:
            el_int = el.astype(int)
            poly = deformed[el_int]
            poly_c = np.vstack([poly, poly[0]])
            ax.plot(poly_c[:, 0], poly_c[:, 1], color=color, linewidth=0.5, alpha=0.7)
    ax.set_xlim(-0.2, 2.8)
    ax.set_ylim(-0.2, 1.2)
    ax.set_aspect("equal")
    ax.set_title(f"(c) Deformed (×{scale:.0f}), Load={lf:.1f}", fontsize=13, fontweight="bold")
    ax.legend([plt.Line2D([0],[0],color="blue",lw=2), plt.Line2D([0],[0],color="red",lw=2)],
              ["Linear", "Neo-Hookean"], fontsize=10)

    fig.suptitle(f"Neo-Hookean vs Linear VEM — Dysbiotic Biofilm (DI={DI_val}, E={E_val:.0f} Pa)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(SAVE_DIR, "fig2_nonlinear_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close()


# ── Figure 3: Phase-field Detachment Sequence ─────────────────────────────

def fig3_phase_field():
    """Phase-field detachment: time sequence of crack evolution."""
    print("Generating Fig 3: Phase-field Detachment...")

    vertices, elements, bnd, valid_ids, domain = _make_mesh(n_cells=50, seed=42)
    compact_verts, compact_elems, bc_dofs, bc_vals, top, n_used = _compact_mesh(
        vertices, elements, domain
    )
    xmin, xmax, ymin, ymax = domain
    n_el = len(compact_elems)
    nu = 0.35

    # Spatial DI gradient
    xmid, ymid = (xmin + xmax) / 2, (ymin + ymax) / 2
    DI_per_cell = np.zeros(n_el)
    for i, el in enumerate(compact_elems):
        el_int = el.astype(int)
        cx = np.mean(compact_verts[el_int, 0])
        cy = np.mean(compact_verts[el_int, 1])
        r = np.sqrt((cx - xmid)**2 + (cy - ymid)**2)
        r_max = np.sqrt((xmid - xmin)**2 + (ymid - ymin)**2)
        proximity = 1.0 - r / r_max
        DI_per_cell[i] = np.clip(0.15 + 0.60 * proximity, 0.0, 1.0)

    E_field = compute_E_from_DI(DI_per_cell)
    Gc_field = compute_Gc(DI_per_cell)

    # Run phase-field
    n_steps = 30
    load_schedule = []
    for step in range(n_steps):
        lf = (step + 1) / n_steps * 4.0
        l_dofs_list, l_vals_list = [], []
        if len(top) > 0:
            l_dofs_list.append(2 * top)
            l_vals_list.append(np.full(len(top), lf / len(top)))
            l_dofs_list.append(2 * top + 1)
            l_vals_list.append(np.full(len(top), -lf * 0.2 / len(top)))
        l_dofs = np.concatenate(l_dofs_list) if l_dofs_list else None
        l_vals = np.concatenate(l_vals_list) if l_vals_list else None
        load_schedule.append((l_dofs, l_vals))

    solver = PhaseFieldVEM(compact_verts, compact_elems, E_field, nu, Gc_field)
    snapshots = solver.run(bc_dofs, bc_vals, load_schedule, verbose=False)

    # Find key frames: pre-crack, onset, propagation, failure
    d_maxs = [s["d_max"] for s in snapshots]
    # Find onset (first step with d_max > 0.1)
    onset_idx = next((i for i, d in enumerate(d_maxs) if d > 0.1), len(d_maxs) // 3)
    # Find failure (first step with d_max > 0.95)
    failure_idx = next((i for i, d in enumerate(d_maxs) if d > 0.95), len(d_maxs) - 1)
    # Pre-crack: halfway to onset
    pre_idx = max(0, onset_idx // 2)
    # Propagation: halfway between onset and failure
    prop_idx = (onset_idx + failure_idx) // 2

    key_frames = [pre_idx, onset_idx, prop_idx, failure_idx]
    frame_labels = ["Pre-crack", "Crack Onset", "Propagation", "Full Failure"]

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))

    for col, (idx, label) in enumerate(zip(key_frames, frame_labels)):
        snap = snapshots[idx]

        # Row 1: Phase-field d
        ax = axes[0, col]
        d_per_cell = np.array([np.mean(snap["d"][el.astype(int)]) for el in compact_elems])
        patches = [MplPolygon(compact_verts[el.astype(int)], closed=True) for el in compact_elems]
        pc = PatchCollection(patches, cmap="inferno", edgecolor="k", linewidth=0.3)
        pc.set_array(d_per_cell)
        pc.set_clim(0, 1)
        ax.add_collection(pc)
        ax.set_xlim(xmin - 0.05, xmax + 0.05)
        ax.set_ylim(ymin - 0.05, ymax + 0.05)
        ax.set_aspect("equal")
        fig.colorbar(pc, ax=ax, label="d", shrink=0.7)
        ax.set_title(f"{label}\nStep {idx+1}, d_max={snap['d_max']:.3f}",
                     fontsize=11, fontweight="bold")

        # Row 2: Deformed + displacement
        ax = axes[1, col]
        ux = snap["u"][0::2]
        uy = snap["u"][1::2]
        mag = np.sqrt(ux**2 + uy**2)

        # Adaptive scale
        max_mag = np.max(mag)
        scale = min(50.0, 0.3 / max(max_mag, 1e-10))

        deformed = compact_verts + scale * np.column_stack([ux, uy])
        patches = [MplPolygon(deformed[el.astype(int)], closed=True) for el in compact_elems]
        colors = [np.mean(mag[el.astype(int)]) for el in compact_elems]
        pc = PatchCollection(patches, cmap="hot_r", edgecolor="k", linewidth=0.3)
        pc.set_array(np.array(colors))
        ax.add_collection(pc)
        ax.set_xlim(xmin - 0.2, xmax + 0.5)
        ax.set_ylim(ymin - 0.2, ymax + 0.2)
        ax.set_aspect("equal")
        fig.colorbar(pc, ax=ax, label="|u|", shrink=0.7)
        ax.set_title(f"|u|_max = {snap['u_max']:.2e}", fontsize=10)

    fig.suptitle("Phase-Field VEM: Biofilm Detachment Sequence\n"
                 "DI → G_c(DI): Dysbiotic center cracks first (low fracture toughness)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(SAVE_DIR, "fig3_phase_field_sequence.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close()

    # Also plot load-displacement + damage curve
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    steps = [s["step"] + 1 for s in snapshots]
    u_maxs = [s["u_max"] for s in snapshots]
    d_maxs_plot = [s["d_max"] for s in snapshots]

    ax.plot(steps, u_maxs, "b-o", ms=4, lw=2, label="|u|_max")
    ax.set_xlabel("Load Step", fontsize=12)
    ax.set_ylabel("|u|_max", color="b", fontsize=12)
    ax.tick_params(axis="y", labelcolor="b")

    ax2 = ax.twinx()
    ax2.plot(steps, d_maxs_plot, "r-s", ms=4, lw=2, label="d_max")
    ax2.set_ylabel("d_max (damage)", color="r", fontsize=12)
    ax2.tick_params(axis="y", labelcolor="r")
    ax2.fill_between(steps, 0, d_maxs_plot, color="red", alpha=0.1)

    # Mark key frames
    for idx, label in zip(key_frames, frame_labels):
        ax.axvline(idx + 1, color="gray", linestyle=":", alpha=0.5)
        ax.text(idx + 1.2, ax.get_ylim()[1] * 0.95, label, fontsize=7,
                rotation=90, va="top", color="gray")

    ax.legend(loc="upper left", fontsize=10)
    ax2.legend(loc="center right", fontsize=10)
    ax.set_title("Phase-Field VEM: Load-Displacement + Damage", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "fig3b_load_displacement.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close()


# ── Figure 4: Constitutive Laws Overview ──────────────────────────────────

def fig4_constitutive():
    """Visualize E(DI), G_c(DI), and material property landscape."""
    print("Generating Fig 4: Constitutive Laws...")

    DI = np.linspace(0, 1, 200)
    E = compute_E_from_DI(DI)
    Gc = compute_Gc(DI)

    # Lamé parameters
    nu = 0.35
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) E(DI)
    ax = axes[0]
    ax.plot(DI, E, "b-", lw=2.5)
    ax.fill_between(DI, E, alpha=0.15, color="blue")
    ax.axhspan(20, 14000, alpha=0.05, color="green", label="Literature range (20-14000 Pa)")
    ax.set_xlabel("Dysbiosis Index (DI)", fontsize=12)
    ax.set_ylabel("E [Pa]", fontsize=12)
    ax.set_title("(a) Young's Modulus E(DI)", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1100)
    # Mark conditions
    for di, label, col in [(0.10, "CS", "#2196F3"), (0.25, "DH", "#FF9800"), (0.75, "DS", "#F44336")]:
        e = compute_E_from_DI(di)
        ax.plot(di, e, "o", color=col, ms=10, zorder=5)
        ax.annotate(f"{label}\n{e:.0f} Pa", (di, e), textcoords="offset points",
                    xytext=(10, 10), fontsize=9, color=col, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (b) G_c(DI)
    ax = axes[1]
    ax.plot(DI, Gc, "r-", lw=2.5)
    ax.fill_between(DI, Gc, alpha=0.15, color="red")
    ax.set_xlabel("Dysbiosis Index (DI)", fontsize=12)
    ax.set_ylabel("G_c [J/m²]", fontsize=12)
    ax.set_title("(b) Fracture Toughness G_c(DI)", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 1)
    for di, label, col in [(0.10, "CS", "#2196F3"), (0.25, "DH", "#FF9800"), (0.75, "DS", "#F44336")]:
        gc = compute_Gc(di)
        ax.plot(di, gc, "o", color=col, ms=10, zorder=5)
        ax.annotate(f"{label}\n{gc:.3f} J/m²", (di, gc), textcoords="offset points",
                    xytext=(10, 5), fontsize=9, color=col, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # (c) E vs G_c scatter (material landscape)
    ax = axes[2]
    sc = ax.scatter(E, Gc, c=DI, cmap="RdYlGn_r", s=10, zorder=3)
    fig.colorbar(sc, ax=ax, label="DI", shrink=0.8)
    for di, label, col in [(0.10, "CS", "#2196F3"), (0.25, "DH", "#FF9800"), (0.75, "DS", "#F44336")]:
        e = compute_E_from_DI(di)
        gc = compute_Gc(di)
        ax.plot(e, gc, "o", color=col, ms=12, zorder=5, markeredgecolor="k")
        ax.annotate(label, (e, gc), textcoords="offset points",
                    xytext=(8, 8), fontsize=11, color=col, fontweight="bold")
    ax.set_xlabel("E [Pa]", fontsize=12)
    ax.set_ylabel("G_c [J/m²]", fontsize=12)
    ax.set_title("(c) Material Landscape", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 1100)
    # Add quadrant labels
    ax.text(800, 0.45, "Stiff + Tough\n(Commensal)", fontsize=9, ha="center", color="#2196F3", alpha=0.7)
    ax.text(150, 0.05, "Soft + Fragile\n(Dysbiotic)", fontsize=9, ha="center", color="#F44336", alpha=0.7)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Biofilm Constitutive Laws: DI-Dependent Material Properties",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(SAVE_DIR, "fig4_constitutive_laws.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("VEM Biofilm Mechanics — Showcase Figure Generation")
    print("=" * 60)

    fig4_constitutive()
    fig1_growth_coupled()
    fig2_nonlinear()
    fig3_phase_field()

    print("\n" + "=" * 60)
    print(f"All showcase figures saved to: {SAVE_DIR}")
    print("=" * 60)
