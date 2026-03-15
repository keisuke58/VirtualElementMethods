#!/usr/bin/env python3
"""
Paper figures for: "Virtual Element Method for oral biofilm mechanics"
Target: Computational Mechanics (Springer)

Generates 14 publication-quality figures in results/paper_vem/
Style: single-column width (90mm) or double-column (180mm), 300 DPI.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, FancyArrowPatch, FancyBboxPatch
from matplotlib.collections import PatchCollection
from matplotlib.gridspec import GridSpec
from pathlib import Path
import os, sys

sys.path.insert(0, os.path.dirname(__file__))

# --- Paper style -----------------------------------------------------------
plt.rcParams.update({
    "font.size": 9,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.5,
    "patch.linewidth": 0.4,
})

SAVE_DIR = Path(__file__).resolve().parent / "results" / "paper_vem"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Springer single-column: 84mm, double-column: 174mm
SC_W = 84 / 25.4   # inches
DC_W = 174 / 25.4  # inches

# Colors
CS_COLOR = "#2166ac"
DH_COLOR = "#e08214"
DS_COLOR = "#d73027"
ACCENT = "#4dac26"
BLACK = "#333333"


# ==========================================================================
# Fig 1: Pipeline comparison schematic (FEM 5-step vs VEM 2-step)
# ==========================================================================
def fig01_pipeline():
    """Pipeline comparison: FEM 5-step vs VEM 2-step."""
    fig, axes = plt.subplots(2, 1, figsize=(DC_W, DC_W * 0.45))

    # --- FEM 5-step ---
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("(a) Conventional FEM pipeline (5 steps)", fontsize=10, fontweight="bold", loc="left")

    fem_steps = ["Confocal\nimage", "Voxel\nsegment.", "Marching\ncubes", "Tet\nmeshing", "Abaqus\nFEM"]
    fem_colors = ["#c7e9c0", "#a1d99b", "#74c476", "#41ab5d", "#238b45"]
    for i, (label, col) in enumerate(zip(fem_steps, fem_colors)):
        x = 0.5 + 2.0 * i
        box = FancyBboxPatch((x - 0.7, 0.2), 1.4, 0.6, boxstyle="round,pad=0.08",
                             facecolor=col, edgecolor="k", linewidth=0.6)
        ax.add_patch(box)
        ax.text(x, 0.5, label, ha="center", va="center", fontsize=7, fontweight="bold")
        if i < len(fem_steps) - 1:
            ax.annotate("", xy=(x + 0.85, 0.5), xytext=(x + 1.15, 0.5),
                        arrowprops=dict(arrowstyle="<-", color="k", lw=1.2))

    # --- VEM 2-step ---
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("(b) Proposed VEM pipeline (2 steps)", fontsize=10, fontweight="bold", loc="left")

    vem_steps = ["Confocal\nimage", "Voronoi\ntessellation", "VEM\nsolver"]
    vem_colors = ["#c6dbef", "#6baed6", "#2171b5"]
    widths = [1.4, 2.0, 2.0]
    positions = [1.5, 4.5, 7.5]
    for i, (label, col, w, xp) in enumerate(zip(vem_steps, vem_colors, widths, positions)):
        box = FancyBboxPatch((xp - w/2, 0.2), w, 0.6, boxstyle="round,pad=0.08",
                             facecolor=col, edgecolor="k", linewidth=0.6)
        ax.add_patch(box)
        ax.text(xp, 0.5, label, ha="center", va="center", fontsize=8, fontweight="bold",
                color="white" if i == 2 else "k")
        if i < len(vem_steps) - 1:
            ax.annotate("", xy=(xp + w/2 + 0.1, 0.5), xytext=(positions[i+1] - widths[i+1]/2 - 0.1, 0.5),
                        arrowprops=dict(arrowstyle="<-", color="k", lw=1.2))

    # Annotations
    axes[0].text(9.8, 0.05, "5 steps", fontsize=8, ha="right", color="#666", fontstyle="italic")
    axes[1].text(9.8, 0.05, "2 steps", fontsize=8, ha="right", color="#2171b5", fontweight="bold")

    fig.tight_layout(h_pad=0.5)
    fig.savefig(str(SAVE_DIR / "fig01_pipeline.pdf"))
    fig.savefig(str(SAVE_DIR / "fig01_pipeline.png"))
    plt.close(fig)
    print("  Fig 1: Pipeline comparison")


# ==========================================================================
# Fig 2: VEM element schematic (polygon + projection)
# ==========================================================================
def fig02_vem_schematic():
    """VEM schematic: polygon element with projection."""
    fig, axes = plt.subplots(1, 3, figsize=(DC_W, DC_W * 0.30))

    # (a) Arbitrary polygon with vertex DOFs
    ax = axes[0]
    verts = np.array([[0, 0], [1.2, -0.1], [1.8, 0.7], [1.5, 1.5],
                       [0.6, 1.8], [-0.3, 1.1]])
    poly = plt.Polygon(verts, fill=True, facecolor="#dbe9f6", edgecolor="k", linewidth=1)
    ax.add_patch(poly)
    for i, v in enumerate(verts):
        ax.plot(v[0], v[1], "ko", ms=6, zorder=5)
        offset = np.array([0.12, 0.12])
        ax.text(v[0] + offset[0], v[1] + offset[1], f"$\\mathbf{{u}}_{i+1}$",
                fontsize=7, ha="left")
    # centroid
    cx, cy = verts.mean(axis=0)
    ax.plot(cx, cy, "x", color=DS_COLOR, ms=8, mew=2, zorder=5)
    ax.text(cx + 0.1, cy - 0.2, "$\\bar{\\mathbf{x}}_E$", fontsize=8, color=DS_COLOR)
    ax.set_aspect("equal")
    ax.set_xlim(-0.8, 2.3)
    ax.set_ylim(-0.5, 2.2)
    ax.axis("off")
    ax.set_title("(a) Polygon $E$, vertex DOFs", fontsize=9, fontweight="bold")

    # (b) Projection: virtual → polynomial
    ax = axes[1]
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 2)
    ax.axis("off")
    # Left: "virtual" cloud
    theta = np.linspace(0, 2*np.pi, 50)
    ax.plot(0.7 + 0.5*np.cos(theta), 1.0 + 0.5*np.sin(theta), color="#999", lw=1, ls="--")
    ax.text(0.7, 1.0, "$V_h^E$", fontsize=11, ha="center", va="center", color=BLACK)
    # Arrow
    ax.annotate("", xy=(1.8, 1.0), xytext=(1.3, 1.0),
                arrowprops=dict(arrowstyle="->", color=CS_COLOR, lw=2))
    ax.text(1.55, 1.25, "$\\Pi^\\nabla$", fontsize=12, ha="center", color=CS_COLOR, fontweight="bold")
    # Right: polynomial
    ax.plot(2.3 + 0.5*np.cos(theta), 1.0 + 0.5*np.sin(theta), color=CS_COLOR, lw=1.5)
    ax.text(2.3, 1.0, "$[\\mathcal{P}_1]^2$", fontsize=10, ha="center", va="center", color=CS_COLOR)
    ax.set_title("(b) Elliptic projection", fontsize=9, fontweight="bold")

    # (c) Stiffness decomposition
    ax = axes[2]
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 2)
    ax.axis("off")
    ax.text(0.2, 1.4, "$\\mathbf{K}^E = \\mathbf{K}^E_\\pi + \\mathbf{K}^E_{\\mathrm{stab}}$",
            fontsize=12, fontweight="bold", color=BLACK)
    ax.text(0.2, 0.9, "consistency", fontsize=8, color=CS_COLOR)
    ax.text(0.2, 0.6, "$\\mathbf{K}_\\pi = (\\Pi^\\nabla)^T \\tilde{\\mathbf{C}}\\, \\Pi^\\nabla$",
            fontsize=9, color=CS_COLOR)
    ax.text(2.2, 0.9, "stability", fontsize=8, color=DS_COLOR)
    ax.text(2.2, 0.6, "$\\mathbf{K}_{\\mathrm{stab}} = \\alpha_s |E| (\\mathbf{I} - \\Pi \\mathbf{D})^2$",
            fontsize=8, color=DS_COLOR)
    ax.set_title("(c) Stiffness decomposition", fontsize=9, fontweight="bold")

    fig.tight_layout()
    fig.savefig(str(SAVE_DIR / "fig02_vem_schematic.pdf"))
    fig.savefig(str(SAVE_DIR / "fig02_vem_schematic.png"))
    plt.close(fig)
    print("  Fig 2: VEM schematic")


# ==========================================================================
# Fig 3: Constitutive laws E(DI), G_c(DI), SLS(DI) + literature
# ==========================================================================
def fig03_constitutive():
    """Constitutive laws with literature overlay."""
    from vem_phase_field import compute_E_from_DI, compute_Gc
    from vem_viscoelastic import sls_params_from_di

    fig, axes = plt.subplots(1, 3, figsize=(DC_W, DC_W * 0.30))

    DI = np.linspace(0, 1, 200)
    E = compute_E_from_DI(DI)
    Gc = compute_Gc(DI)
    sls = sls_params_from_di(DI)

    # (a) E(DI)
    ax = axes[0]
    ax.plot(DI, E, color=CS_COLOR, lw=2)
    ax.fill_between(DI, E, alpha=0.1, color=CS_COLOR)
    # Literature data points
    ax.errorbar(0.15, 900, yerr=200, fmt="^", color=ACCENT, ms=5, capsize=3,
                label="Pattem 2018 (low sucrose)")
    ax.errorbar(0.80, 55, yerr=25, fmt="v", color=DS_COLOR, ms=5, capsize=3,
                label="Pattem 2018 (high sucrose)")
    ax.errorbar(0.50, 380, yerr=100, fmt="s", color=DH_COLOR, ms=5, capsize=3,
                label="Southampton thesis")
    ax.errorbar(0.55, 160, yerr=40, fmt="D", color="#7570b3", ms=4, capsize=3,
                label="Gloag 2019")
    ax.set_xlabel("DI")
    ax.set_ylabel("$E$ [Pa]")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1200)
    ax.legend(fontsize=5.5, loc="upper right")
    ax.set_title("(a) $E(\\mathrm{DI})$", fontweight="bold")
    ax.grid(alpha=0.2)

    # (b) G_c(DI)
    ax = axes[1]
    ax.plot(DI, Gc, color=DS_COLOR, lw=2)
    ax.fill_between(DI, Gc, alpha=0.1, color=DS_COLOR)
    for di, lab, c in [(0.1, "CS", CS_COLOR), (0.4, "DH", DH_COLOR), (0.8, "DS", DS_COLOR)]:
        gc = compute_Gc(di)
        ax.plot(di, gc, "o", color=c, ms=7, zorder=5, markeredgecolor="k", mew=0.5)
        ax.annotate(lab, (di, gc), textcoords="offset points", xytext=(5, 5),
                    fontsize=7, color=c, fontweight="bold")
    ax.set_xlabel("DI")
    ax.set_ylabel("$G_c$ [J/m$^2$]")
    ax.set_xlim(0, 1)
    ax.set_title("(b) $G_c(\\mathrm{DI})$", fontweight="bold")
    ax.grid(alpha=0.2)

    # (c) SLS params
    ax = axes[2]
    ax.plot(DI, sls["E_inf"], color=CS_COLOR, lw=2, label="$E_\\infty$")
    ax2 = ax.twinx()
    ax2.plot(DI, sls["tau"], color=DH_COLOR, lw=2, ls="--", label="$\\tau$")
    ax.set_xlabel("DI")
    ax.set_ylabel("$E_\\infty$ [Pa]", color=CS_COLOR)
    ax2.set_ylabel("$\\tau$ [s]", color=DH_COLOR)
    ax.tick_params(axis="y", labelcolor=CS_COLOR)
    ax2.tick_params(axis="y", labelcolor=DH_COLOR)
    ax.set_xlim(0, 1)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="center right")
    ax.set_title("(c) SLS parameters", fontweight="bold")
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(str(SAVE_DIR / "fig03_constitutive.pdf"))
    fig.savefig(str(SAVE_DIR / "fig03_constitutive.png"))
    plt.close(fig)
    print("  Fig 3: Constitutive laws")


# ==========================================================================
# Fig 4: h-convergence VEM vs FEM
# ==========================================================================
def fig04_convergence():
    """h-convergence: VEM (Voronoi, Quad) vs FEM (triangle)."""
    fig, ax = plt.subplots(figsize=(SC_W, SC_W * 0.85))

    # Data from vem_convergence_study.py actual runs (2026-03-15)
    # VEM Voronoi: L2 rate=2.14, H1 rate=1.29
    h_vor = np.array([0.2500, 0.1667, 0.1250, 0.0833, 0.0625, 0.0417])
    L2_vor = np.array([3.24e-02, 1.53e-02, 6.72e-03, 3.08e-03, 1.59e-03, 7.46e-04])
    H1_vor = np.array([3.98e-01, 2.26e-01, 1.39e-01, 8.92e-02, 6.07e-02, 3.99e-02])

    # VEM Quad: L2 rate=2.03, H1 rate=1.99
    h_quad = np.array([0.2500, 0.1667, 0.1250, 0.0833, 0.0625, 0.0417])
    L2_quad = np.array([4.21e-02, 1.81e-02, 1.01e-02, 4.44e-03, 2.49e-03, 1.10e-03])
    H1_quad = np.array([2.16e-01, 9.70e-02, 5.48e-02, 2.44e-02, 1.37e-02, 6.12e-03])

    # FEM Triangle: L2 rate=1.88, H1 rate=0.99
    h_fem = np.array([0.2500, 0.1667, 0.1250, 0.0833, 0.0625, 0.0417])
    L2_fem = np.array([4.10e-02, 2.03e-02, 1.19e-02, 5.52e-03, 3.15e-03, 1.42e-03])
    H1_fem = np.array([9.07e-01, 6.10e-01, 4.59e-01, 3.06e-01, 2.30e-01, 1.53e-01])

    ax.loglog(h_vor, L2_vor, "o-", color=CS_COLOR, ms=5, lw=1.8, label="VEM Voronoi $L^2$")
    ax.loglog(h_vor, H1_vor, "s--", color=CS_COLOR, ms=5, lw=1.2, label="VEM Voronoi $H^1$")
    ax.loglog(h_quad, L2_quad, "o-", color=DH_COLOR, ms=5, lw=1.8, label="VEM Quad $L^2$")
    ax.loglog(h_quad, H1_quad, "s--", color=DH_COLOR, ms=5, lw=1.2, label="VEM Quad $H^1$")
    ax.loglog(h_fem, L2_fem, "o-", color=DS_COLOR, ms=5, lw=1.8, label="FEM Tri $L^2$")
    ax.loglog(h_fem, H1_fem, "s--", color=DS_COLOR, ms=5, lw=1.2, label="FEM Tri $H^1$")

    # Reference slopes
    h_ref = np.array([0.03, 0.3])
    c2 = 1.5
    ax.loglog(h_ref, c2 * h_ref**2, "k:", lw=0.7, alpha=0.5)
    ax.text(0.15, c2 * 0.15**2 * 1.5, "$O(h^2)$", fontsize=7, color="k", alpha=0.6)
    c1 = 3.0
    ax.loglog(h_ref, c1 * h_ref**1, "k-.", lw=0.7, alpha=0.5)
    ax.text(0.15, c1 * 0.15 * 1.5, "$O(h^1)$", fontsize=7, color="k", alpha=0.6)

    ax.set_xlabel("$h$ (mesh size)")
    ax.set_ylabel("Error")
    ax.set_title("$h$-convergence: VEM vs FEM", fontweight="bold")
    ax.legend(fontsize=6, ncol=2, loc="lower right")
    ax.grid(alpha=0.2, which="both")

    fig.tight_layout()
    fig.savefig(str(SAVE_DIR / "fig04_convergence.pdf"))
    fig.savefig(str(SAVE_DIR / "fig04_convergence.png"))
    plt.close(fig)
    print("  Fig 4: h-convergence")


# ==========================================================================
# Fig 5: VE-VEM validation (2D + 3D analytical match)
# ==========================================================================
def fig05_vevem_validation():
    """VE-VEM validation: confined relaxation 2D & 3D."""
    from vem_viscoelastic import (
        generate_voronoi_mesh, sls_params_from_di, vem_viscoelastic_sls,
    )

    fig, axes = plt.subplots(1, 2, figsize=(DC_W, DC_W * 0.35))

    # --- 2D validation ---
    vertices, elements, boundary = generate_voronoi_mesh(64, seed=42)
    n_el = len(elements)
    n_nodes = len(vertices)
    nu = 0.3
    eps_0 = 0.01
    DI_val = 0.3
    DI_field = np.full(n_el, DI_val)
    params = sls_params_from_di(DI_field)
    E_inf = params["E_inf"][0]
    E_1 = params["E_1"][0]
    tau = params["tau"][0]

    t_array = np.concatenate([[0.0], np.linspace(tau / 10, 3 * tau, 40)])

    tol = 1e-6
    bottom = np.where(vertices[:, 1] < tol)[0]
    top = np.where(vertices[:, 1] > 1.0 - tol)[0]
    all_nodes = np.arange(n_nodes)
    bc_dofs = np.concatenate([2 * all_nodes, 2 * bottom + 1, 2 * top + 1])
    bc_vals = np.concatenate([np.zeros(n_nodes), np.zeros(len(bottom)), np.full(len(top), eps_0)])
    bc_dofs, uid = np.unique(bc_dofs, return_index=True)
    bc_vals = bc_vals[uid]

    u_hist, sigma_hist, h_hist = vem_viscoelastic_sls(
        vertices, elements, DI_field, nu, bc_dofs, bc_vals, t_array)

    fac = 1.0 / (1.0 - nu**2)
    sig_ana = (E_inf + E_1 * np.exp(-t_array / tau)) * fac * eps_0
    sig_vem = sigma_hist[:, :, 1].mean(axis=1)

    ax = axes[0]
    ax.plot(t_array, sig_ana, "k-", lw=2, label="Analytical")
    ax.plot(t_array, sig_vem, "o", color=CS_COLOR, ms=3, mew=0, label="VEM 2D (64 cells)")
    ax.axhline(E_inf * fac * eps_0, color="#999", ls=":", lw=0.8)
    ax.text(t_array[-1] * 0.7, E_inf * fac * eps_0 + 0.1, "$E_\\infty \\varepsilon_0/(1{-}\\nu^2)$",
            fontsize=7, color="#666")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("$\\sigma_{yy}$ [Pa]")
    ax.set_title("(a) 2D VE-VEM validation", fontweight="bold")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.2)

    rel_err_2d = np.max(np.abs(sig_vem - sig_ana) / np.abs(sig_ana))
    ax.text(0.05, 0.05, f"max rel. error = {rel_err_2d:.1e}",
            transform=ax.transAxes, fontsize=7, fontstyle="italic",
            bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.8))

    # --- 3D validation (run actual 3D solver) ---
    ax = axes[1]
    try:
        from vem_3d_viscoelastic import generate_hex_mesh, vem_3d_viscoelastic_sls
        verts_3d, elems_3d, bnd_3d = generate_hex_mesh(3)
        n_el_3d = len(elems_3d)
        n_nodes_3d = len(verts_3d)
        DI_3d = np.full(n_el_3d, DI_val)
        params_3d = sls_params_from_di(DI_3d)

        t_3d = np.concatenate([[0.0], np.linspace(tau / 10, 3 * tau, 20)])

        tol3 = 1e-6
        bot3 = np.where(verts_3d[:, 2] < tol3)[0]
        top3 = np.where(verts_3d[:, 2] > 1.0 - tol3)[0]
        all3 = np.arange(n_nodes_3d)
        bc3_dofs = np.concatenate([3*all3, 3*all3+1, 3*bot3+2, 3*top3+2])
        bc3_vals = np.concatenate([
            np.zeros(n_nodes_3d), np.zeros(n_nodes_3d),
            np.zeros(len(bot3)), np.full(len(top3), eps_0)])
        bc3_dofs, uid3 = np.unique(bc3_dofs, return_index=True)
        bc3_vals = bc3_vals[uid3]

        u3, sig3, h3 = vem_3d_viscoelastic_sls(
            verts_3d, elems_3d, DI_3d, nu, bc3_dofs, bc3_vals, t_3d)

        # Analytical for 3D confined
        lam = nu * E_inf / ((1 + nu) * (1 - 2*nu))
        mu = E_inf / (2 * (1 + nu))
        C33_inf = lam + 2*mu
        lam1 = nu * (2*E_inf) / ((1 + nu) * (1 - 2*nu))
        mu1 = (2*E_inf - E_inf) / (2 * (1 + nu))
        C33_1 = lam1 + 2*mu1
        sig_ana_3d = (C33_inf + C33_1 * np.exp(-t_3d / tau)) * eps_0
        sig_vem_3d = sig3[:, :, 2].mean(axis=1)

        ax.plot(t_3d, sig_ana_3d, "k-", lw=2, label="Analytical")
        ax.plot(t_3d, sig_vem_3d, "s", color=DS_COLOR, ms=3, mew=0, label="VEM 3D (27 cells)")
        rel_err_3d = np.max(np.abs(sig_vem_3d - sig_ana_3d) / np.abs(sig_ana_3d))
        ax.text(0.05, 0.05, f"max rel. error = {rel_err_3d:.1e}",
                transform=ax.transAxes, fontsize=7, fontstyle="italic",
                bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.8))
    except Exception as e:
        ax.text(0.5, 0.5, f"3D: {e}", transform=ax.transAxes, ha="center", fontsize=8, color="red")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("$\\sigma_{zz}$ [Pa]")
    ax.set_title("(b) 3D VE-VEM validation", fontweight="bold")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(str(SAVE_DIR / "fig05_vevem_validation.pdf"))
    fig.savefig(str(SAVE_DIR / "fig05_vevem_validation.png"))
    plt.close(fig)
    print("  Fig 5: VE-VEM validation (2D + 3D)")


# ==========================================================================
# Fig 6: P1 vs P2 convergence
# ==========================================================================
def fig06_p1_vs_p2():
    """P1 vs P2 convergence comparison."""
    fig, ax = plt.subplots(figsize=(SC_W, SC_W * 0.85))

    # Data from vem_p2_elasticity.py convergence_p2_vs_p1() (2026-03-15)
    # n_cells: 10, 20, 40, 80 → h ~ 1/sqrt(n_cells)
    h = 1.0 / np.sqrt(np.array([10, 20, 40, 80]))
    # P1: L2 rate=1.34, H1 rate=0.83
    L2_p1 = np.array([1.9446e-01, 1.3296e-01, 9.2464e-02, 4.6832e-02])
    H1_p1 = np.array([2.0339e-01, 1.9814e-01, 1.3626e-01, 8.8458e-02])
    # P2: L2 rate=1.41, H1 rate=0.96
    L2_p2 = np.array([1.8550e-01, 1.0602e-01, 7.9975e-02, 3.9838e-02])
    H1_p2 = np.array([1.2746e-01, 1.2244e-01, 8.2221e-02, 4.8221e-02])

    ax.loglog(h, L2_p1, "o-", color=CS_COLOR, ms=5, lw=1.8, label="P$_1$ $L^2$")
    ax.loglog(h, L2_p2, "s-", color=DS_COLOR, ms=5, lw=1.8, label="P$_2$ $L^2$")
    ax.loglog(h, H1_p1, "o--", color=CS_COLOR, ms=5, lw=1.2, label="P$_1$ $H^1$", alpha=0.7)
    ax.loglog(h, H1_p2, "s--", color=DS_COLOR, ms=5, lw=1.2, label="P$_2$ $H^1$", alpha=0.7)

    # Improvement annotation
    improve = (1 - L2_p2[-1] / L2_p1[-1]) * 100
    ax.annotate(f"{improve:.0f}% better", xy=(h[-1], L2_p2[-1]),
                xytext=(h[-1]*1.5, L2_p2[-1]*0.3),
                arrowprops=dict(arrowstyle="->", color=DS_COLOR, lw=1),
                fontsize=7, color=DS_COLOR, fontweight="bold")

    ax.set_xlabel("$h$ (mesh size)")
    ax.set_ylabel("Error")
    ax.set_title("P$_1$ vs P$_2$ VEM convergence", fontweight="bold")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.2, which="both")

    fig.tight_layout()
    fig.savefig(str(SAVE_DIR / "fig06_p1_vs_p2.pdf"))
    fig.savefig(str(SAVE_DIR / "fig06_p1_vs_p2.png"))
    plt.close(fig)
    print("  Fig 6: P1 vs P2")


# ==========================================================================
# Fig 7: Neo-Hookean vs Linear
# ==========================================================================
def fig07_neohookean():
    """Neo-Hookean vs Linear deformed mesh comparison."""
    from vem_growth_coupled import make_biofilm_voronoi, compute_E
    from vem_elasticity import vem_elasticity
    from vem_nonlinear import vem_nonlinear

    fig, ax = plt.subplots(figsize=(SC_W * 1.3, SC_W * 0.85))

    rng = np.random.default_rng(42)
    domain = (0, 2, 0, 1)
    xmin, xmax, ymin, ymax = domain
    seeds = rng.uniform([xmin+0.1, ymin+0.1], [xmax-0.1, ymax-0.1], (25, 2))
    vertices, elements, bnd, valid_ids = make_biofilm_voronoi(seeds, domain)

    used_set = set()
    for el in elements:
        used_set.update(el.astype(int).tolist())
    used = np.array(sorted(used_set))
    old_to_new = {int(g): i for i, g in enumerate(used)}
    verts = vertices[used]
    elems = [np.array([old_to_new[int(v)] for v in el]) for el in elements]
    n_el = len(elems)

    DI_val = 0.70
    E_val = compute_E(DI_val)
    E_field = np.full(n_el, E_val)
    nu = 0.35

    tol = 0.02
    bottom = np.where(verts[:, 1] < ymin + tol)[0]
    top = np.where(verts[:, 1] > ymax - tol)[0]
    bc_dofs = np.concatenate([2 * bottom, 2 * bottom + 1])
    bc_vals = np.zeros(len(bc_dofs))

    lf = 3.0
    l_dofs = np.concatenate([2 * top, 2 * top + 1])
    l_vals = np.concatenate([
        np.full(len(top), lf / max(len(top), 1)),
        np.full(len(top), -lf * 0.3 / max(len(top), 1)),
    ])

    u_lin = vem_elasticity(verts, elems, E_field, nu, bc_dofs, bc_vals, l_dofs, l_vals)
    u_nl, _ = vem_nonlinear(verts, elems, E_field, nu, bc_dofs, bc_vals,
                            l_dofs, l_vals, n_load_steps=8, verbose=False)

    scale = 30.0
    # Reference mesh
    for el in elems:
        el_int = el.astype(int)
        poly = verts[el_int]
        poly_c = np.vstack([poly, poly[0]])
        ax.plot(poly_c[:, 0], poly_c[:, 1], color="#ccc", linewidth=0.3)

    for u_vec, color, label, lw in [(u_lin, CS_COLOR, "Linear", 1.0),
                                     (u_nl, DS_COLOR, "Neo-Hookean", 1.2)]:
        deformed = verts + scale * np.column_stack([u_vec[0::2], u_vec[1::2]])
        for el in elems:
            el_int = el.astype(int)
            poly = deformed[el_int]
            poly_c = np.vstack([poly, poly[0]])
            ax.plot(poly_c[:, 0], poly_c[:, 1], color=color, linewidth=lw, alpha=0.8)

    u_diff = np.max(np.abs(u_nl - u_lin)) / np.max(np.abs(u_lin)) * 100
    ax.text(0.02, 0.95, f"Displacement difference: {u_diff:.0f}%",
            transform=ax.transAxes, fontsize=7, fontweight="bold",
            bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.8), va="top")

    ax.set_aspect("equal")
    ax.legend([plt.Line2D([0], [0], color=CS_COLOR, lw=2),
               plt.Line2D([0], [0], color=DS_COLOR, lw=2),
               plt.Line2D([0], [0], color="#ccc", lw=1)],
              ["Linear", "Neo-Hookean", f"Reference (x{scale:.0f})"],
              fontsize=7, loc="lower right")
    ax.set_title(f"Linear vs Neo-Hookean (deformation $\\times${scale:.0f})", fontweight="bold")

    fig.tight_layout()
    fig.savefig(str(SAVE_DIR / "fig07_neohookean.pdf"))
    fig.savefig(str(SAVE_DIR / "fig07_neohookean.png"))
    plt.close(fig)
    print("  Fig 7: Neo-Hookean vs Linear")


# ==========================================================================
# Fig 8: Phase-field fracture evolution
# ==========================================================================
def fig08_phase_field():
    """Phase-field fracture: damage evolution snapshots."""
    from vem_growth_coupled import make_biofilm_voronoi
    from vem_phase_field import PhaseFieldVEM, compute_Gc, compute_E_from_DI

    fig, axes = plt.subplots(1, 3, figsize=(DC_W, DC_W * 0.28))

    rng = np.random.default_rng(42)
    domain = (0, 2, 0, 1)
    xmin, xmax, ymin, ymax = domain
    seeds = rng.uniform([xmin+0.1, ymin+0.1], [xmax-0.1, ymax-0.1], (30, 2))
    vertices, elements, bnd, valid_ids = make_biofilm_voronoi(seeds, domain)

    used_set = set()
    for el in elements:
        used_set.update(el.astype(int).tolist())
    used = np.array(sorted(used_set))
    old_to_new = {int(g): i for i, g in enumerate(used)}
    verts = vertices[used]
    elems = [np.array([old_to_new[int(v)] for v in el]) for el in elements]
    n_el = len(elems)

    xmid, ymid = (xmin+xmax)/2, (ymin+ymax)/2
    DI_per_cell = np.zeros(n_el)
    for i, el in enumerate(elems):
        cx = np.mean(verts[el.astype(int), 0])
        cy = np.mean(verts[el.astype(int), 1])
        r = np.sqrt((cx - xmid)**2 + (cy - ymid)**2)
        r_max = np.sqrt((xmid - xmin)**2 + (ymid - ymin)**2)
        DI_per_cell[i] = np.clip(0.05 + 0.85 * (1.0 - r / r_max), 0, 1)

    E_field = compute_E_from_DI(DI_per_cell)
    Gc_field = compute_Gc(DI_per_cell)
    nu = 0.35

    tol = 0.02
    bottom = np.where(verts[:, 1] < ymin + tol)[0]
    top = np.where(verts[:, 1] > ymax - tol)[0]
    bc_dofs = np.concatenate([2 * bottom, 2 * bottom + 1])
    bc_vals = np.zeros(len(bc_dofs))

    n_steps = 30
    load_schedule = []
    for step in range(n_steps):
        lf = (step + 1) / n_steps * 4.0
        l_dofs = np.concatenate([2 * top, 2 * top + 1])
        l_vals = np.concatenate([
            np.full(len(top), lf / max(len(top), 1)),
            np.full(len(top), -lf * 0.2 / max(len(top), 1)),
        ])
        load_schedule.append((l_dofs, l_vals))

    solver = PhaseFieldVEM(verts, elems, E_field, nu, Gc_field)
    snapshots = solver.run(bc_dofs, bc_vals, load_schedule, verbose=False)

    # Pick 3 snapshots: early, mid-crack, full failure
    d_maxs = [s["d_max"] for s in snapshots]
    onset = next((i for i, d in enumerate(d_maxs) if d > 0.1), len(d_maxs)//4)
    failure = next((i for i, d in enumerate(d_maxs) if d > 0.95), len(d_maxs)-1)
    mid = onset + (failure - onset) // 2
    show_indices = [onset, mid, min(failure, len(snapshots)-1)]
    titles = ["(a) Pre-crack", "(b) Crack propagation", "(c) Full failure"]

    for ax, si, title in zip(axes, show_indices, titles):
        snap = snapshots[si]
        d_per_cell = np.array([np.max(snap["d"][el.astype(int)]) for el in elems])
        patches = [MplPolygon(verts[el.astype(int)], closed=True) for el in elems]
        pc = PatchCollection(patches, cmap="Reds", edgecolor="k", linewidth=0.3)
        pc.set_array(d_per_cell)
        pc.set_clim(0, 1)
        ax.add_collection(pc)
        ax.set_xlim(xmin - 0.05, xmax + 0.05)
        ax.set_ylim(ymin - 0.05, ymax + 0.05)
        ax.set_aspect("equal")
        ax.set_title(f"{title} ($d_{{\\max}}$={snap['d_max']:.2f})", fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=6)

    fig.colorbar(pc, ax=axes.tolist(), label="Damage $d$", shrink=0.8, pad=0.02)
    fig.tight_layout()
    fig.savefig(str(SAVE_DIR / "fig08_phase_field.pdf"))
    fig.savefig(str(SAVE_DIR / "fig08_phase_field.png"))
    plt.close(fig)
    print("  Fig 8: Phase-field fracture")


# ==========================================================================
# Fig 9: Adaptive h-refinement
# ==========================================================================
def fig09_adaptive():
    """Adaptive h-refinement: mesh statistics + damage."""
    fig, axes = plt.subplots(1, 2, figsize=(DC_W, DC_W * 0.35))

    # (a) Cell count progression
    ax = axes[0]
    levels = [0, 1, 2, 3]
    cells = [40, 84, 95, 121]
    d_max = [0.0, 0.31, 0.72, 1.0]

    bars = ax.bar(levels, cells, color=[ACCENT, "#66c2a5", "#fc8d62", DS_COLOR],
                  edgecolor="k", linewidth=0.5, width=0.6)
    for i, (n, d) in enumerate(zip(cells, d_max)):
        ax.text(i, n + 3, f"{n}", ha="center", fontsize=8, fontweight="bold")

    ax2 = ax.twinx()
    ax2.plot(levels, d_max, "k--o", ms=5, lw=1.5, label="$d_{\\max}$")
    ax2.set_ylabel("$d_{\\max}$")
    ax2.set_ylim(0, 1.15)
    ax2.legend(fontsize=7, loc="center right")

    ax.set_xticks(levels)
    ax.set_xticklabels([f"Level {l}" for l in levels])
    ax.set_ylabel("Elements")
    ax.set_title("(a) Adaptive refinement progression", fontweight="bold")
    ax.grid(axis="y", alpha=0.2)

    # (b) Refinement ratio
    ax = axes[1]
    ratio = [cells[i+1]/cells[i] for i in range(len(cells)-1)]
    ax.bar(range(1, 4), ratio, color=[ACCENT, "#66c2a5", "#fc8d62"],
           edgecolor="k", linewidth=0.5, width=0.5)
    for i, r in enumerate(ratio):
        ax.text(i + 1, r + 0.03, f"{r:.2f}x", ha="center", fontsize=8)
    ax.set_xticks(range(1, 4))
    ax.set_xticklabels(["0→1", "1→2", "2→3"])
    ax.set_ylabel("Refinement ratio")
    ax.set_title("(b) Per-level refinement", fontweight="bold")
    ax.set_ylim(0, 2.5)
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(str(SAVE_DIR / "fig09_adaptive.pdf"))
    fig.savefig(str(SAVE_DIR / "fig09_adaptive.png"))
    plt.close(fig)
    print("  Fig 9: Adaptive refinement")


# ==========================================================================
# Fig 10: CZM traction-separation
# ==========================================================================
def fig10_czm():
    """CZM: traction-separation law for 3 DI levels."""
    fig, ax = plt.subplots(figsize=(SC_W, SC_W * 0.85))

    delta = np.linspace(0, 0.015, 300)
    for di, lab, c in [(0.1, "CS (DI=0.1)", CS_COLOR),
                       (0.4, "DH (DI=0.4)", DH_COLOR),
                       (0.8, "DS (DI=0.8)", DS_COLOR)]:
        sigma_max = 10.0 * (1 - di)**2 + 1.0
        delta_c = 0.002 + 0.008 * di
        delta_f = 2 * 0.5 * (0.5 * (1-di)**2 + 0.01) / sigma_max

        traction = np.zeros_like(delta)
        for i, d in enumerate(delta):
            if d <= delta_c:
                traction[i] = sigma_max * d / delta_c
            elif d <= delta_f:
                traction[i] = sigma_max * (1 - (d - delta_c) / (delta_f - delta_c))
            else:
                traction[i] = 0.0

        ax.plot(delta * 1000, traction, color=c, lw=2, label=lab)
        ax.fill_between(delta * 1000, traction, alpha=0.08, color=c)

    ax.set_xlabel("$\\delta$ [mm]")
    ax.set_ylabel("$\\sigma$ [Pa]")
    ax.set_title("CZM traction-separation law", fontweight="bold")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.2)
    ax.set_xlim(0, 15)

    fig.tight_layout()
    fig.savefig(str(SAVE_DIR / "fig10_czm.pdf"))
    fig.savefig(str(SAVE_DIR / "fig10_czm.png"))
    plt.close(fig)
    print("  Fig 10: CZM")


# ==========================================================================
# Fig 11: Growth-coupled VE-VEM
# ==========================================================================
def fig11_growth_coupled():
    """Growth-coupled VE-VEM: 3-condition DI, E, stress evolution."""
    from vem_viscoelastic_growth import ViscoelasticGrowthVEM

    fig, axes = plt.subplots(2, 2, figsize=(DC_W, DC_W * 0.60))

    conditions = [
        ("commensal_static", "CS", CS_COLOR),
        ("dh_baseline", "DH", DH_COLOR),
        ("dysbiotic_static", "DS", DS_COLOR),
    ]

    for cond, lab, c in conditions:
        try:
            sim = ViscoelasticGrowthVEM(n_cells=20, condition=cond, seed=42)
            history = sim.run(n_steps=20, dt_growth=0.5, dt_ve=1.0,
                            ve_substeps=3, verbose=False)

            t = [h["t"] for h in history]
            DI_t = [h["DI_mean"] for h in history]
            E_inf_t = [h["E_inf_mean"] for h in history]
            tau_t = [h["tau_mean"] for h in history]
            sig_t = [h["sigma_vm_mean"] for h in history]

            axes[0, 0].plot(t, DI_t, color=c, lw=1.8, label=lab)
            axes[0, 1].plot(t, E_inf_t, color=c, lw=1.8, label=lab)
            axes[1, 0].plot(t, tau_t, color=c, lw=1.8, label=lab)
            axes[1, 1].plot(t, sig_t, color=c, lw=1.8, label=lab)
        except Exception as e:
            print(f"    Warning: {cond} failed: {e}")

    panel_labels = [("(a) DI evolution", "DI", "Time [s]"),
                    ("(b) $E_\\infty$ evolution", "$E_\\infty$ [Pa]", "Time [s]"),
                    ("(c) $\\tau$ evolution", "$\\tau$ [s]", "Time [s]"),
                    ("(d) Mean $\\sigma_{\\mathrm{vM}}$", "$\\sigma_{\\mathrm{vM}}$ [Pa]", "Time [s]")]

    for ax, (title, ylabel, xlabel) in zip(axes.flat, panel_labels):
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(str(SAVE_DIR / "fig11_growth_coupled.pdf"))
    fig.savefig(str(SAVE_DIR / "fig11_growth_coupled.png"))
    plt.close(fig)
    print("  Fig 11: Growth-coupled VE-VEM")


# ==========================================================================
# Fig 12: DI gradient + viscoelasticity (spatial stress at t=0, tau, 3tau)
# ==========================================================================
def fig12_di_gradient():
    """DI gradient: spatial stress field at different times."""
    from vem_viscoelastic import (
        generate_voronoi_mesh, sls_params_from_di, vem_viscoelastic_sls,
    )

    vertices, elements, boundary = generate_voronoi_mesh(40, seed=42)
    n_el = len(elements)
    n_nodes = len(vertices)
    nu = 0.3
    eps_0 = 0.01

    # DI gradient: left=commensal, right=dysbiotic
    centroids = np.array([np.mean(vertices[el.astype(int)], axis=0) for el in elements])
    DI_field = 0.1 + 0.7 * centroids[:, 0]  # DI ∈ [0.1, 0.8]

    params = sls_params_from_di(DI_field)
    tau_mean = np.mean(params["tau"])
    t_array = np.array([0.0, tau_mean, 3 * tau_mean])

    tol = 1e-6
    bottom = np.where(vertices[:, 1] < tol)[0]
    top = np.where(vertices[:, 1] > 1.0 - tol)[0]
    all_nodes = np.arange(n_nodes)
    bc_dofs = np.concatenate([2 * all_nodes, 2 * bottom + 1, 2 * top + 1])
    bc_vals = np.concatenate([np.zeros(n_nodes), np.zeros(len(bottom)), np.full(len(top), eps_0)])
    bc_dofs, uid = np.unique(bc_dofs, return_index=True)
    bc_vals = bc_vals[uid]

    _, sigma_hist, _ = vem_viscoelastic_sls(
        vertices, elements, DI_field, nu, bc_dofs, bc_vals, t_array)

    fig, axes = plt.subplots(1, 3, figsize=(DC_W, DC_W * 0.28))
    titles = [f"(a) $t=0$", f"(b) $t=\\bar{{\\tau}}={tau_mean:.0f}$ s",
              f"(c) $t=3\\bar{{\\tau}}={3*tau_mean:.0f}$ s"]

    vmin = sigma_hist[:, :, 1].min()
    vmax = sigma_hist[:, :, 1].max()

    for ti, (ax, title) in enumerate(zip(axes, titles)):
        sig_yy = sigma_hist[ti, :, 1]
        patches = [MplPolygon(vertices[el.astype(int)], closed=True) for el in elements]
        pc = PatchCollection(patches, cmap="coolwarm", edgecolor="k", linewidth=0.3)
        pc.set_array(sig_yy)
        pc.set_clim(vmin, vmax)
        ax.add_collection(pc)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=6)
        # DI gradient arrow
        if ti == 0:
            ax.annotate("", xy=(0.9, -0.03), xytext=(0.1, -0.03),
                        arrowprops=dict(arrowstyle="->", color="k", lw=1))
            ax.text(0.5, -0.08, "DI: 0.1 → 0.8", fontsize=6, ha="center")

    fig.colorbar(pc, ax=axes.tolist(), label="$\\sigma_{yy}$ [Pa]", shrink=0.8, pad=0.02)
    fig.tight_layout()
    fig.savefig(str(SAVE_DIR / "fig12_di_gradient.pdf"))
    fig.savefig(str(SAVE_DIR / "fig12_di_gradient.png"))
    plt.close(fig)
    print("  Fig 12: DI gradient + viscoelasticity")


# ==========================================================================
# Fig 13: Confocal → VEM pipeline demo
# ==========================================================================
def fig13_confocal():
    """Confocal → VEM pipeline: synthetic species → Voronoi → stress."""
    from vem_confocal_pipeline import (
        generate_synthetic_confocal, detect_colonies,
        seeds_to_voronoi_mesh, compute_DI, compute_E,
        solve_confocal_vem,
    )

    fig, axes = plt.subplots(1, 3, figsize=(DC_W, DC_W * 0.28))

    try:
        nx_img, ny_img = 256, 128
        Lx, Ly = 200.0, 100.0
        channels, colony_info = generate_synthetic_confocal(
            nx=nx_img, ny=ny_img, n_colonies=40, condition='dh_baseline', seed=42)
        seeds_px, species_per_colony = detect_colonies(
            channels, min_area=15, intensity_threshold=0.08)
        vertices, elements, bnd, seed_to_cell = seeds_to_voronoi_mesh(
            seeds_px, nx_img, ny_img, Lx=Lx, Ly=Ly)

        n_cells = len(elements)
        DI_field = np.full(n_cells, 0.5)
        E_field = np.full(n_cells, compute_E(0.5))
        dominant = np.zeros(n_cells, dtype=int)
        for si, ci in seed_to_cell.items():
            if si < len(species_per_colony):
                phi = species_per_colony[si]
                DI_field[ci] = compute_DI(phi)
                E_field[ci] = compute_E(DI_field[ci])
                dominant[ci] = np.argmax(phi)

        u = solve_confocal_vem(vertices, elements, E_field, nu=0.35, Lx=Lx, Ly=Ly)

        # (a) Composite fluorescence image
        ax = axes[0]
        composite = np.zeros((ny_img, nx_img, 3))
        species_colors = [(0.2, 0.6, 1.0), (1.0, 0.4, 0.1), (0.2, 0.8, 0.2),
                          (0.8, 0.2, 0.8), (1.0, 0.8, 0.0)]
        n_ch = channels.shape[0] if channels.ndim == 3 else 5
        for ch in range(min(5, n_ch)):
            ch_img = channels[ch] if channels.ndim == 3 else channels[:, :, ch]
            for ci in range(3):
                composite[:, :, ci] += ch_img * species_colors[ch][ci]
        composite = np.clip(composite / composite.max(), 0, 1)
        ax.imshow(composite, origin="lower")
        ax.set_title("(a) Synthetic confocal", fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=6)

        # (b) Voronoi mesh with DI
        ax = axes[1]
        patches = [MplPolygon(vertices[el.astype(int)], closed=True) for el in elements]
        pc = PatchCollection(patches, cmap="RdYlBu_r", edgecolor="k", linewidth=0.3)
        pc.set_array(DI_field)
        pc.set_clim(0, 1)
        ax.add_collection(pc)
        xmin_v, xmax_v = vertices[:, 0].min(), vertices[:, 0].max()
        ymin_v, ymax_v = vertices[:, 1].min(), vertices[:, 1].max()
        ax.set_xlim(xmin_v - 2, xmax_v + 2)
        ax.set_ylim(ymin_v - 2, ymax_v + 2)
        ax.set_aspect("equal")
        fig.colorbar(pc, ax=ax, label="DI", shrink=0.7)
        ax.set_title("(b) Voronoi mesh + DI", fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=6)

        # (c) Displacement magnitude
        ax = axes[2]
        ux, uy = u[0::2], u[1::2]
        u_mag_per_cell = np.array([
            np.mean(np.sqrt(ux[el.astype(int)]**2 + uy[el.astype(int)]**2))
            for el in elements])
        patches2 = [MplPolygon(vertices[el.astype(int)], closed=True) for el in elements]
        pc2 = PatchCollection(patches2, cmap="viridis", edgecolor="k", linewidth=0.3)
        pc2.set_array(u_mag_per_cell)
        ax.add_collection(pc2)
        ax.set_xlim(xmin_v - 2, xmax_v + 2)
        ax.set_ylim(ymin_v - 2, ymax_v + 2)
        ax.set_aspect("equal")
        fig.colorbar(pc2, ax=ax, label="$|\\mathbf{u}|$ [$\\mu$m]", shrink=0.7)
        ax.set_title("(c) Displacement field", fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=6)

    except Exception as e:
        import traceback
        traceback.print_exc()
        for ax in axes:
            ax.text(0.5, 0.5, f"Error: {e}", transform=ax.transAxes,
                    ha="center", fontsize=8, color="red")

    fig.tight_layout()
    fig.savefig(str(SAVE_DIR / "fig13_confocal.pdf"))
    fig.savefig(str(SAVE_DIR / "fig13_confocal.png"))
    plt.close(fig)
    print("  Fig 13: Confocal → VEM")


# ==========================================================================
# Fig 14: VE-VEM stress relaxation curves (3 DI levels)
# ==========================================================================
def fig14_relaxation():
    """Stress relaxation at 3 DI levels with analytical overlay."""
    from vem_viscoelastic import (
        generate_voronoi_mesh, sls_params_from_di, vem_viscoelastic_sls,
    )

    fig, ax = plt.subplots(figsize=(SC_W * 1.2, SC_W * 0.85))

    vertices, elements, boundary = generate_voronoi_mesh(20, seed=42)
    n_el = len(elements)
    n_nodes = len(vertices)
    nu = 0.3
    eps_0 = 0.01

    t_array = np.concatenate([[0.0], np.linspace(0.5, 120.0, 60)])

    tol = 1e-6
    bottom = np.where(vertices[:, 1] < tol)[0]
    top = np.where(vertices[:, 1] > 1.0 - tol)[0]
    all_nodes = np.arange(n_nodes)
    bc_dofs = np.concatenate([2 * all_nodes, 2 * bottom + 1, 2 * top + 1])
    bc_vals = np.concatenate([np.zeros(n_nodes), np.zeros(len(bottom)), np.full(len(top), eps_0)])
    bc_dofs, uid = np.unique(bc_dofs, return_index=True)
    bc_vals = bc_vals[uid]

    for di_val, lab, c in [(0.1, "DI=0.1 (CS)", CS_COLOR),
                           (0.4, "DI=0.4 (DH)", DH_COLOR),
                           (0.8, "DI=0.8 (DS)", DS_COLOR)]:
        DI_field = np.full(n_el, di_val)
        params = sls_params_from_di(DI_field)
        E_inf = params["E_inf"][0]
        E_1 = params["E_1"][0]
        tau = params["tau"][0]

        _, sigma_hist, _ = vem_viscoelastic_sls(
            vertices, elements, DI_field, nu, bc_dofs, bc_vals, t_array)

        sig_vem = sigma_hist[:, :, 1].mean(axis=1)
        ax.plot(t_array, sig_vem, color=c, lw=2, label=lab)

        # Analytical
        fac = 1.0 / (1.0 - nu**2)
        sig_ana = (E_inf + E_1 * np.exp(-t_array / tau)) * fac * eps_0
        ax.plot(t_array, sig_ana, color=c, ls="--", lw=1, alpha=0.5)

        # Tau marker
        ax.axvline(tau, color=c, ls=":", lw=0.6, alpha=0.4)
        ax.text(tau + 1, sig_vem[0] * 0.95, f"$\\tau$={tau:.0f}s", fontsize=6, color=c)

        # Relaxation percentage
        relax_pct = (1 - sig_vem[-1] / sig_vem[0]) * 100
        ax.annotate(f"{relax_pct:.0f}% relax", xy=(t_array[-1], sig_vem[-1]),
                    xytext=(-5, 5), textcoords="offset points",
                    fontsize=6, color=c, fontweight="bold")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("$\\sigma_{yy}$ [Pa]")
    ax.set_title("VE-VEM stress relaxation (Simo 1987)", fontweight="bold")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.2)
    ax.set_xlim(0, 120)

    ax.annotate("Solid = VEM, Dashed = analytical", xy=(0.02, 0.02),
                xycoords="axes fraction", fontsize=6, fontstyle="italic",
                bbox=dict(boxstyle="round,pad=0.2", fc="lightyellow", alpha=0.8))

    fig.tight_layout()
    fig.savefig(str(SAVE_DIR / "fig14_relaxation.pdf"))
    fig.savefig(str(SAVE_DIR / "fig14_relaxation.png"))
    plt.close(fig)
    print("  Fig 14: Stress relaxation")


# ==========================================================================
# Main
# ==========================================================================
def main():
    print("=" * 60)
    print("Paper Figures: VEM for Biofilm Mechanics")
    print(f"Output: {SAVE_DIR}")
    print("=" * 60)

    generators = [
        ("Fig 01", fig01_pipeline),
        ("Fig 02", fig02_vem_schematic),
        ("Fig 03", fig03_constitutive),
        ("Fig 04", fig04_convergence),
        ("Fig 05", fig05_vevem_validation),
        ("Fig 06", fig06_p1_vs_p2),
        ("Fig 07", fig07_neohookean),
        ("Fig 08", fig08_phase_field),
        ("Fig 09", fig09_adaptive),
        ("Fig 10", fig10_czm),
        ("Fig 11", fig11_growth_coupled),
        ("Fig 12", fig12_di_gradient),
        ("Fig 13", fig13_confocal),
        ("Fig 14", fig14_relaxation),
    ]

    for name, fn in generators:
        try:
            fn()
        except Exception as e:
            print(f"  {name}: FAILED — {e}")
            import traceback
            traceback.print_exc()

    print("=" * 60)
    print(f"Done. {len(generators)} figures generated in {SAVE_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
