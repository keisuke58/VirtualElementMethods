"""
Phase-Field Fracture VEM for Biofilm Detachment Modeling.

Staggered (alternating minimization) phase-field approach on VEM polygonal mesh.
Models biofilm detachment as a fracture process where dysbiotic regions
(high DI, low G_c) crack first under mechanical loading.

Phase-field model (Aldakheel et al. 2018, adapted):
  Momentum:  ∇·[g(d)·σ₀] = 0,  g(d) = (1-d)² + k
  Phase-field: G_c·l₀·Δd - G_c/l₀·d + 2(1-d)·ψ⁺ = 0

Staggered solve:
  1. Fix d → solve displacement u (degraded VEM stiffness)
  2. Fix u → solve phase-field d (scalar VEM, reaction-diffusion)
  Repeat with irreversibility d_new ≥ d_old.

References:
  - Aldakheel, Hudobivnik, Hussein, Wriggers (2018) CMAME 341
  - Miehe, Welschinger, Hofacker (2010) "Thermodynamically consistent phase-field"
  - Nguyen-Thanh et al. (2018) CMAME 340 — VEM for 2D fracture at IKM
"""

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection


# ── Biofilm-specific constitutive parameters ──────────────────────────────


def compute_Gc(DI, Gc_max=0.5, Gc_min=0.01, n=2):
    """
    Fracture toughness from Dysbiosis Index.
    Dysbiotic biofilm (high DI) → low G_c → easy detachment.

    G_c(DI) = G_c_min + (G_c_max - G_c_min)·(1 - DI)^n
    """
    DI = np.clip(DI, 0.0, 1.0)
    return Gc_min + (Gc_max - Gc_min) * (1.0 - DI) ** n


def compute_E_from_DI(DI, E_max=1000.0, E_min=30.0, n=2):
    """E(DI) = E_min + (E_max - E_min)·(1 - DI)^n"""
    return E_min + (E_max - E_min) * (1.0 - DI) ** n


# ── Strain decomposition ──────────────────────────────────────────────────


def spectral_decomposition_2d(eps_voigt):
    """
    Spectral decomposition of 2D strain tensor.
    Split into tensile ε⁺ (positive eigenvalues) and compressive ε⁻.

    Parameters
    ----------
    eps_voigt : (3,) array [ε_xx, ε_yy, γ_xy] (engineering shear)

    Returns
    -------
    eps_plus, eps_minus : (3,) arrays in Voigt form
    """
    eps_tensor = np.array(
        [
            [eps_voigt[0], 0.5 * eps_voigt[2]],
            [0.5 * eps_voigt[2], eps_voigt[1]],
        ]
    )

    eigvals, eigvecs = np.linalg.eigh(eps_tensor)

    eps_plus_tensor = np.zeros((2, 2))
    eps_minus_tensor = np.zeros((2, 2))

    for i in range(2):
        n = eigvecs[:, i : i + 1]
        if eigvals[i] > 0:
            eps_plus_tensor += eigvals[i] * (n @ n.T)
        else:
            eps_minus_tensor += eigvals[i] * (n @ n.T)

    eps_plus = np.array(
        [eps_plus_tensor[0, 0], eps_plus_tensor[1, 1], 2.0 * eps_plus_tensor[0, 1]]
    )
    eps_minus = np.array(
        [eps_minus_tensor[0, 0], eps_minus_tensor[1, 1], 2.0 * eps_minus_tensor[0, 1]]
    )
    return eps_plus, eps_minus


def compute_psi_plus(eps_voigt, E, nu):
    """
    Tensile elastic energy density (crack driving force).

    ψ⁺ = λ/2·<tr(ε)>₊² + μ·tr(ε⁺·ε⁺)
    """
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    eps_plus, _ = spectral_decomposition_2d(eps_voigt)

    # Tensile trace
    tr_eps = eps_voigt[0] + eps_voigt[1]
    tr_plus = max(tr_eps, 0.0)

    # ψ⁺ = λ/2·<tr(ε)>₊² + μ·(ε⁺_xx² + ε⁺_yy² + 2·(ε⁺_xy)²)
    eps_plus_tensor = np.array(
        [
            [eps_plus[0], 0.5 * eps_plus[2]],
            [0.5 * eps_plus[2], eps_plus[1]],
        ]
    )
    psi = lam / 2.0 * tr_plus**2 + mu * np.trace(eps_plus_tensor @ eps_plus_tensor)
    return max(psi, 0.0)


# ── Element-level VEM helpers ─────────────────────────────────────────────


def _element_geometry(verts):
    """Compute area, centroid, diameter for polygon."""
    n_v = len(verts)
    area_comp = (
        verts[:, 0] * np.roll(verts[:, 1], -1)
        - np.roll(verts[:, 0], -1) * verts[:, 1]
    )
    area = 0.5 * abs(np.sum(area_comp))
    centroid = np.sum(
        (np.roll(verts, -1, axis=0) + verts) * area_comp[:, None], axis=0
    ) / (6.0 * max(area, 1e-15))
    h = max(
        np.linalg.norm(verts[i] - verts[j])
        for i in range(n_v)
        for j in range(i + 1, n_v)
    )
    return area, centroid, h


def _vertex_normals(verts):
    """Compute vertex normals (integral weighting) for polygon."""
    n_v = len(verts)
    normals = np.zeros((n_v, 2))
    for i in range(n_v):
        prev_v = verts[(i - 1) % n_v]
        next_v = verts[(i + 1) % n_v]
        normals[i] = [next_v[1] - prev_v[1], prev_v[0] - next_v[0]]
    return normals


# ── Scalar VEM assembly (for phase-field equation) ────────────────────────


def assemble_scalar_vem(vertices, elements, diffusion_field, reaction_field, source_field):
    """
    Assemble scalar VEM system (Poisson-like with reaction and source).

    Solves: -κ·Δd + r·d = s  on each element
    where κ = diffusion, r = reaction, s = source (per element).

    Uses the Sutton (2017) scalar VEM pattern.

    Returns: K_global (sparse), F_global (dense)
    """
    n_nodes = vertices.shape[0]
    n_polys = 3

    row_idx = []
    col_idx = []
    val_data = []
    F_global = np.zeros(n_nodes)

    for el_id in range(len(elements)):
        vert_ids = elements[el_id].astype(int)
        verts = vertices[vert_ids]
        n_v = len(vert_ids)

        area, centroid, h = _element_geometry(verts)
        vnormals = _vertex_normals(verts)

        kappa = diffusion_field[el_id] if hasattr(diffusion_field, "__len__") else diffusion_field
        react = reaction_field[el_id] if hasattr(reaction_field, "__len__") else reaction_field
        source = source_field[el_id] if hasattr(source_field, "__len__") else source_field

        # D matrix (n_v × 3)
        D = np.zeros((n_v, n_polys))
        D[:, 0] = 1.0
        for i in range(n_v):
            D[i, 1] = (verts[i, 0] - centroid[0]) / h
            D[i, 2] = (verts[i, 1] - centroid[1]) / h

        # B matrix (3 × n_v)
        B = np.zeros((n_polys, n_v))
        B[0, :] = 1.0 / n_v
        for i in range(n_v):
            for poly_id in range(1, n_polys):
                grad_p = np.zeros(2)
                if poly_id == 1:
                    grad_p[0] = 1.0 / h
                else:
                    grad_p[1] = 1.0 / h
                B[poly_id, i] = 0.5 * np.dot(grad_p, vnormals[i])

        G = B @ D
        G_inv = np.linalg.inv(G)
        projector = G_inv @ B

        # Diffusion stiffness
        G_tilde = G.copy()
        G_tilde[0, :] = 0.0
        K_cons = projector.T @ G_tilde @ projector * kappa

        # Stabilization
        I_minus_PiD = np.eye(n_v) - D @ projector
        trace_k = np.trace(K_cons)
        stab_param = 0.5 * abs(trace_k) / max(n_v, 1) if trace_k > 0 else kappa * 0.01
        K_stab = stab_param * (I_minus_PiD.T @ I_minus_PiD)

        K_diff = K_cons + K_stab

        # Mass matrix (lumped): M_ii = area / n_v
        M_lumped = (area / n_v) * np.eye(n_v)

        # Element stiffness: K_diff + react · M
        K_el = K_diff + react * M_lumped

        # Element source: source · M · 1
        F_el = source * (area / n_v) * np.ones(n_v)

        # Assemble
        ii, jj = np.meshgrid(vert_ids, vert_ids, indexing="ij")
        row_idx.append(ii.ravel())
        col_idx.append(jj.ravel())
        val_data.append(K_el.ravel())
        F_global[vert_ids] += F_el

    K_global = sp.csr_matrix(
        (np.concatenate(val_data), (np.concatenate(row_idx), np.concatenate(col_idx))),
        shape=(n_nodes, n_nodes),
    )
    return K_global, F_global


# ── Degraded elasticity VEM assembly ──────────────────────────────────────


def assemble_degraded_elasticity_vem(
    vertices, elements, E_field, nu, d_field, k_residual=1e-6, stab_alpha=0.5
):
    """
    Assemble VEM elasticity with degradation g(d) = (1-d)² + k.

    Returns: K_global (sparse), element strain data for ψ⁺ computation
    """
    n_nodes = vertices.shape[0]
    n_dofs = 2 * n_nodes
    n_polys = 6

    row_idx = []
    col_idx = []
    val_data = []

    elem_strain_data = []

    for el_id in range(len(elements)):
        vert_ids = elements[el_id].astype(int)
        verts = vertices[vert_ids]
        n_v = len(vert_ids)
        n_el_dofs = 2 * n_v

        E_el = E_field[el_id] if hasattr(E_field, "__len__") else E_field

        # Average d over element nodes
        d_avg = np.mean(d_field[vert_ids]) if len(d_field) > 0 else 0.0
        g_d = (1.0 - d_avg) ** 2 + k_residual

        # Degraded modulus
        E_degraded = E_el * g_d

        C_mat = (E_degraded / (1.0 - nu**2)) * np.array(
            [
                [1.0, nu, 0.0],
                [nu, 1.0, 0.0],
                [0.0, 0.0, (1.0 - nu) / 2.0],
            ]
        )

        area, centroid, h = _element_geometry(verts)
        xc, yc = centroid
        vnormals = _vertex_normals(verts)

        # D matrix
        D = np.zeros((n_el_dofs, n_polys))
        for i in range(n_v):
            dx = (verts[i, 0] - xc) / h
            dy = (verts[i, 1] - yc) / h
            D[2 * i, :] = [1.0, 0.0, -dy, dx, 0.0, dy]
            D[2 * i + 1, :] = [0.0, 1.0, dx, 0.0, dy, dx]

        # B matrix
        B = np.zeros((n_polys, n_el_dofs))
        for i in range(n_v):
            B[0, 2 * i] = 1.0 / n_v
            B[1, 2 * i + 1] = 1.0 / n_v
        for i in range(n_v):
            B[2, 2 * i] = -vnormals[i, 1] / (4.0 * area)
            B[2, 2 * i + 1] = vnormals[i, 0] / (4.0 * area)

        strain_basis = np.array(
            [[1.0 / h, 0.0, 0.0], [0.0, 1.0 / h, 0.0], [0.0, 0.0, 2.0 / h]]
        )
        for i in range(n_v):
            vn = vnormals[i]
            for alpha in range(3):
                sigma = C_mat @ strain_basis[alpha]
                tx = sigma[0] * vn[0] + sigma[2] * vn[1]
                ty = sigma[2] * vn[0] + sigma[1] * vn[1]
                B[3 + alpha, 2 * i] += 0.5 * tx
                B[3 + alpha, 2 * i + 1] += 0.5 * ty

        G = B @ D
        projector = np.linalg.solve(G, B)

        G_tilde = G.copy()
        G_tilde[:3, :] = 0.0

        K_cons = projector.T @ G_tilde @ projector
        I_minus_PiD = np.eye(n_el_dofs) - D @ projector
        trace_k = np.trace(K_cons)
        stab_param = stab_alpha * abs(trace_k) / n_el_dofs if trace_k > 0 else E_degraded * 0.01
        K_stab = stab_param * (I_minus_PiD.T @ I_minus_PiD)
        K_local = K_cons + K_stab

        # Store projector for strain recovery
        elem_strain_data.append(
            {
                "vert_ids": vert_ids,
                "projector": projector,
                "h": h,
                "area": area,
                "E_el": E_el,  # undegraded E for ψ⁺
            }
        )

        # Assemble
        gdofs = np.zeros(n_el_dofs, dtype=int)
        for i in range(n_v):
            gdofs[2 * i] = 2 * vert_ids[i]
            gdofs[2 * i + 1] = 2 * vert_ids[i] + 1

        ii, jj = np.meshgrid(gdofs, gdofs, indexing="ij")
        row_idx.append(ii.ravel())
        col_idx.append(jj.ravel())
        val_data.append(K_local.ravel())

    K_global = sp.csr_matrix(
        (np.concatenate(val_data), (np.concatenate(row_idx), np.concatenate(col_idx))),
        shape=(n_dofs, n_dofs),
    )
    return K_global, elem_strain_data


def compute_element_strains(u, elem_strain_data):
    """
    Recover element-average strains from displacement via VEM projector.

    Returns: list of (3,) Voigt strain arrays [ε_xx, ε_yy, γ_xy]
    """
    strains = []
    for ed in elem_strain_data:
        vert_ids = ed["vert_ids"]
        n_v = len(vert_ids)
        proj = ed["projector"]
        h = ed["h"]

        gdofs = np.zeros(2 * n_v, dtype=int)
        for i in range(n_v):
            gdofs[2 * i] = 2 * vert_ids[i]
            gdofs[2 * i + 1] = 2 * vert_ids[i] + 1

        u_local = u[gdofs]
        c = proj @ u_local

        eps_xx = c[3] / h
        eps_yy = c[4] / h
        gamma_xy = c[5] / h  # engineering shear

        strains.append(np.array([eps_xx, eps_yy, gamma_xy]))

    return strains


# ── Phase-Field VEM Solver ────────────────────────────────────────────────


class PhaseFieldVEM:
    """
    Staggered phase-field fracture solver on VEM polygonal mesh.

    Alternating minimization:
      1. Fix d → solve displacement u (degraded VEM stiffness)
      2. Fix u → solve phase-field d (scalar reaction-diffusion VEM)
      3. Enforce irreversibility: d_new = max(d_new, d_old)
    """

    def __init__(self, vertices, elements, E_field, nu, Gc_field, l0=None):
        self.vertices = vertices
        self.elements = elements
        self.E_field = np.asarray(E_field, dtype=float)
        self.nu = nu
        self.Gc_field = np.asarray(Gc_field, dtype=float)

        n_el = len(elements)
        # Default l0: average element diameter
        if l0 is None:
            diams = []
            for el in elements:
                verts = vertices[el.astype(int)]
                _, _, h = _element_geometry(verts)
                diams.append(h)
            self.l0 = np.mean(diams) * 0.5
        else:
            self.l0 = l0

        self.n_nodes = vertices.shape[0]
        self.n_dofs = 2 * self.n_nodes
        self.n_el = n_el

        # State
        self.d = np.zeros(self.n_nodes)  # phase-field (damage)
        self.u = np.zeros(self.n_dofs)  # displacement
        self.psi_history = np.zeros(n_el)  # max ψ⁺ history

    def solve_displacement(self, bc_fixed_dofs, bc_vals, load_dofs=None, load_vals=None):
        """Solve displacement with degraded stiffness g(d)·K."""
        K, elem_data = assemble_degraded_elasticity_vem(
            self.vertices, self.elements, self.E_field, self.nu, self.d
        )

        F = np.zeros(self.n_dofs)
        if load_dofs is not None and load_vals is not None:
            F[load_dofs] += load_vals

        # Apply BCs
        u = np.zeros(self.n_dofs)
        bc_set = set(bc_fixed_dofs.tolist())
        internal = np.array([i for i in range(self.n_dofs) if i not in bc_set])
        u[bc_fixed_dofs] = bc_vals
        F -= K[:, bc_fixed_dofs].toarray() @ bc_vals

        K_ii = K[np.ix_(internal, internal)]
        try:
            u[internal] = sp.linalg.spsolve(K_ii, F[internal])
        except Exception:
            pass

        self.u = u
        self._elem_strain_data = elem_data
        return u

    def compute_psi_plus_field(self):
        """Compute tensile energy density ψ⁺ per element and update history."""
        strains = compute_element_strains(self.u, self._elem_strain_data)

        psi_plus = np.zeros(self.n_el)
        for i, eps in enumerate(strains):
            E_el = self.E_field[i] if hasattr(self.E_field, "__len__") else self.E_field
            psi = compute_psi_plus(eps, E_el, self.nu)
            # History: ψ_history = max(ψ⁺, ψ_history)
            psi_plus[i] = max(psi, self.psi_history[i])

        self.psi_history = psi_plus.copy()
        return psi_plus

    def solve_phase_field(self, psi_plus):
        """
        Solve phase-field equation via scalar VEM.

        G_c·l₀·∇d·∇δd + (G_c/l₀ + 2ψ⁺)·d·δd = 2ψ⁺·δd
        → diffusion = Gc·l0, reaction = Gc/l0 + 2ψ⁺, source = 2ψ⁺
        """
        diffusion = self.Gc_field * self.l0
        reaction = self.Gc_field / self.l0 + 2.0 * psi_plus
        source = 2.0 * psi_plus

        K, F = assemble_scalar_vem(
            self.vertices, self.elements, diffusion, reaction, source
        )

        # Solve: no Dirichlet BC on d (natural BC = no flux)
        try:
            d_new = sp.linalg.spsolve(K, F)
        except Exception:
            d_new = self.d.copy()

        # Clamp to [0, 1]
        d_new = np.clip(d_new, 0.0, 1.0)

        # Irreversibility: d can only grow
        d_new = np.maximum(d_new, self.d)

        self.d = d_new
        return d_new

    def run(
        self,
        bc_fixed_dofs,
        bc_vals,
        load_schedule,
        load_dofs=None,
        max_stagger=30,
        tol=1e-4,
        verbose=False,
    ):
        """
        Incremental loading with staggered phase-field solve.

        Parameters
        ----------
        bc_fixed_dofs, bc_vals: displacement BCs
        load_schedule: list of (load_dofs, load_vals) for each step
        max_stagger: max staggered iterations per step
        tol: convergence tolerance for staggered iteration

        Returns: list of snapshots
        """
        snapshots = []

        for step, (l_dofs, l_vals) in enumerate(load_schedule):
            d_old = self.d.copy()

            for stag_iter in range(max_stagger):
                # Step 1: solve displacement
                self.solve_displacement(bc_fixed_dofs, bc_vals, l_dofs, l_vals)

                # Step 2: compute crack driving force
                psi_plus = self.compute_psi_plus_field()

                # Step 3: solve phase-field
                d_new = self.solve_phase_field(psi_plus)

                # Check convergence
                d_change = np.linalg.norm(d_new - d_old) / max(
                    np.linalg.norm(d_new), 1e-10
                )

                if verbose and stag_iter % 5 == 0:
                    print(
                        f"  Step {step+1}, stagger {stag_iter}: "
                        f"|Δd|/|d| = {d_change:.2e}, "
                        f"max(d) = {np.max(self.d):.4f}, "
                        f"max(ψ⁺) = {np.max(psi_plus):.2e}"
                    )

                if d_change < tol:
                    break
                d_old = d_new.copy()

            # Snapshot
            ux, uy = self.u[0::2], self.u[1::2]
            mag = np.sqrt(ux**2 + uy**2)

            snapshots.append(
                {
                    "step": step,
                    "u": self.u.copy(),
                    "d": self.d.copy(),
                    "u_max": np.max(mag),
                    "d_max": np.max(self.d),
                    "d_mean": np.mean(self.d),
                    "psi_max": np.max(psi_plus),
                    "stagger_iters": stag_iter + 1,
                    "n_cracked": np.sum(self.d > 0.9),
                }
            )

            if verbose:
                s = snapshots[-1]
                print(
                    f"  → Step {step+1} done: |u|_max={s['u_max']:.4e}, "
                    f"d_max={s['d_max']:.4f}, cracked_nodes={s['n_cracked']}"
                )

        return snapshots


# ── Demo ──────────────────────────────────────────────────────────────────


def demo_biofilm_detachment():
    """
    Demo: biofilm detachment under increasing GCF shear load.
    Dysbiotic center (high DI, low G_c) cracks first.
    """
    from vem_growth_coupled import make_biofilm_voronoi

    rng = np.random.default_rng(42)
    domain = (0, 2, 0, 1)
    n_cells = 40
    xmin, xmax, ymin, ymax = domain
    nu = 0.35

    # Generate seeds
    nx = int(np.sqrt(n_cells * 2))
    ny = max(n_cells // nx, 2)
    xx = np.linspace(xmin + 0.1, xmax - 0.1, nx)
    yy = np.linspace(ymin + 0.05, ymax - 0.05, ny)
    gx, gy = np.meshgrid(xx, yy)
    seeds = np.column_stack([gx.ravel(), gy.ravel()])[:n_cells]
    seeds += rng.uniform(-0.03, 0.03, seeds.shape)

    vertices, elements, bnd, valid_ids = make_biofilm_voronoi(seeds, domain)
    n_el = len(elements)
    n_nodes = vertices.shape[0]

    # Spatial DI gradient: dysbiotic center, commensal edges
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

    E_field = compute_E_from_DI(DI_per_cell)
    Gc_field = compute_Gc(DI_per_cell)

    # Compact mesh (remove unused nodes)
    used_set = set()
    for el in elements:
        used_set.update(el.astype(int).tolist())
    used = np.array(sorted(used_set))
    old_to_new = {int(g): i for i, g in enumerate(used)}
    n_used = len(used)

    compact_verts = vertices[used]
    compact_elems = [np.array([old_to_new[int(v)] for v in el]) for el in elements]

    # BCs: bottom fixed
    tol_bc = 0.02
    bottom = np.where(compact_verts[:, 1] < ymin + tol_bc)[0]
    bc_dofs = np.concatenate([2 * bottom, 2 * bottom + 1])
    bc_vals = np.zeros(len(bc_dofs))

    # Load: increasing shear on top
    top = np.where(compact_verts[:, 1] > ymax - tol_bc)[0]
    n_steps = 25

    load_schedule = []
    for step in range(n_steps):
        lf = (step + 1) / n_steps * 3.0  # Load factor up to 3.0
        l_dofs_list = []
        l_vals_list = []
        if len(top) > 0:
            l_dofs_list.append(2 * top)  # x-shear
            l_vals_list.append(np.full(len(top), lf / len(top)))
            l_dofs_list.append(2 * top + 1)  # y-compression
            l_vals_list.append(np.full(len(top), -lf * 0.3 / len(top)))
        l_dofs = np.concatenate(l_dofs_list) if l_dofs_list else None
        l_vals = np.concatenate(l_vals_list) if l_vals_list else None
        load_schedule.append((l_dofs, l_vals))

    # Run phase-field solver
    solver = PhaseFieldVEM(compact_verts, compact_elems, E_field, nu, Gc_field)
    print("Running phase-field VEM biofilm detachment...")
    snapshots = solver.run(bc_dofs, bc_vals, load_schedule, verbose=True)

    # ── Plot ──
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Row 1: Initial state
    # (a) DI field
    ax = axes[0, 0]
    patches = [MplPolygon(compact_verts[el.astype(int)], closed=True) for el in compact_elems]
    pc = PatchCollection(patches, cmap="RdYlGn_r", edgecolor="k", linewidth=0.3)
    pc.set_array(DI_per_cell)
    ax.add_collection(pc)
    ax.set_xlim(xmin - 0.05, xmax + 0.05)
    ax.set_ylim(ymin - 0.05, ymax + 0.05)
    ax.set_aspect("equal")
    fig.colorbar(pc, ax=ax, label="DI", shrink=0.8)
    ax.set_title("(a) Dysbiosis Index")

    # (b) E field
    ax = axes[0, 1]
    patches = [MplPolygon(compact_verts[el.astype(int)], closed=True) for el in compact_elems]
    pc = PatchCollection(patches, cmap="viridis", edgecolor="k", linewidth=0.3)
    pc.set_array(E_field)
    ax.add_collection(pc)
    ax.set_xlim(xmin - 0.05, xmax + 0.05)
    ax.set_ylim(ymin - 0.05, ymax + 0.05)
    ax.set_aspect("equal")
    fig.colorbar(pc, ax=ax, label="E [Pa]", shrink=0.8)
    ax.set_title("(b) Young's Modulus E(DI)")

    # (c) G_c field
    ax = axes[0, 2]
    patches = [MplPolygon(compact_verts[el.astype(int)], closed=True) for el in compact_elems]
    pc = PatchCollection(patches, cmap="Blues", edgecolor="k", linewidth=0.3)
    pc.set_array(Gc_field)
    ax.add_collection(pc)
    ax.set_xlim(xmin - 0.05, xmax + 0.05)
    ax.set_ylim(ymin - 0.05, ymax + 0.05)
    ax.set_aspect("equal")
    fig.colorbar(pc, ax=ax, label="G_c [J/m²]", shrink=0.8)
    ax.set_title("(c) Fracture Toughness G_c(DI)")

    # Row 2: Final state
    final = snapshots[-1]

    # (d) Displacement at failure
    ax = axes[1, 0]
    ux = final["u"][0::2]
    uy = final["u"][1::2]
    mag = np.sqrt(ux**2 + uy**2)
    scale = 20.0
    deformed = compact_verts + scale * np.column_stack([ux, uy])
    patches = [MplPolygon(deformed[el.astype(int)], closed=True) for el in compact_elems]
    colors = [np.mean(mag[el.astype(int)]) for el in compact_elems]
    pc = PatchCollection(patches, cmap="hot_r", edgecolor="k", linewidth=0.3)
    pc.set_array(np.array(colors))
    ax.add_collection(pc)
    ax.set_xlim(xmin - 0.2, xmax + 0.5)
    ax.set_ylim(ymin - 0.2, ymax + 0.2)
    ax.set_aspect("equal")
    fig.colorbar(pc, ax=ax, label="|u|", shrink=0.8)
    ax.set_title(f"(d) Deformed (×{scale:.0f}), |u|_max={final['u_max']:.3e}")

    # (e) Phase-field (crack pattern)
    ax = axes[1, 1]
    d_per_cell = np.array(
        [np.mean(final["d"][el.astype(int)]) for el in compact_elems]
    )
    patches = [MplPolygon(compact_verts[el.astype(int)], closed=True) for el in compact_elems]
    pc = PatchCollection(patches, cmap="inferno", edgecolor="k", linewidth=0.3)
    pc.set_array(d_per_cell)
    pc.set_clim(0, 1)
    ax.add_collection(pc)
    ax.set_xlim(xmin - 0.05, xmax + 0.05)
    ax.set_ylim(ymin - 0.05, ymax + 0.05)
    ax.set_aspect("equal")
    fig.colorbar(pc, ax=ax, label="d (damage)", shrink=0.8)
    ax.set_title(f"(e) Phase-field d, max={final['d_max']:.3f}")

    # (f) Load-displacement + damage evolution
    ax = axes[1, 2]
    steps = [s["step"] + 1 for s in snapshots]
    u_maxs = [s["u_max"] for s in snapshots]
    d_maxs = [s["d_max"] for s in snapshots]

    ax.plot(steps, u_maxs, "b-o", ms=3, lw=1.5, label="|u|_max")
    ax.set_xlabel("Load Step")
    ax.set_ylabel("|u|_max", color="b")
    ax.tick_params(axis="y", labelcolor="b")

    ax2 = ax.twinx()
    ax2.plot(steps, d_maxs, "r-s", ms=3, lw=1.5, label="d_max")
    ax2.set_ylabel("d_max (damage)", color="r")
    ax2.tick_params(axis="y", labelcolor="r")

    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax.set_title("(f) Load-Displacement + Damage")
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Phase-Field VEM: Biofilm Detachment\n"
        "(Dysbiotic center cracks first — low G_c)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    import os

    save_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "vem_phase_field_demo.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


if __name__ == "__main__":
    demo_biofilm_detachment()
