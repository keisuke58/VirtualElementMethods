"""
Cohesive Zone Model (CZM) on VEM for Tooth-Biofilm Interface Detachment.

CZM models the tooth-biofilm interface as a zero-thickness layer with
bilinear traction-separation law. As GCF shear loads the biofilm,
interface tractions increase until reaching peak strength (sigma_max),
then soften linearly to zero at critical separation (delta_c) =>
complete detachment.

Complementary to phase-field (vem_phase_field.py):
  - Phase-field captures bulk fracture *within* the biofilm
  - CZM captures interface delamination *at* the tooth surface

Traction-Separation Law (Park-Paulino-Roesler style, bilinear):
  - Loading branch:  t = K_penalty * delta        (delta < delta_0)
  - Softening branch: t = sigma_max*(delta_c - delta)/(delta_c - delta_0)
  - Fully debonded:  t = 0                        (delta >= delta_c)
  - Mixed-mode: effective delta_eff = sqrt(delta_n^2 + delta_t^2)
  - Irreversible damage: D_new >= D_old

DI-dependent interface strength:
  sigma_max(DI) = sigma_min + (sigma_max - sigma_min)*(1 - DI)^n
  Dysbiotic biofilm has weaker adhesion to tooth surface.

References:
  - Park, Paulino, Roesler (2009) JMPS 57(6): "A unified potential-based
    cohesive model of mixed-mode fracture"
  - Alfano, Crisfield (2001) IJNME 50(7): "Finite element interface models
    for the delamination analysis of laminated composites"
  - Camanho, Davila (2002) NASA/TM-2002-211737: mixed-mode decohesion
"""

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
import os

from vem_elasticity import vem_elasticity
from vem_growth_coupled import make_biofilm_voronoi
from vem_phase_field import compute_E_from_DI, _element_geometry


# -- Traction-Separation Law ------------------------------------------------


def bilinear_tsl(delta_n, delta_t, sigma_max, tau_max, delta_c_n, delta_c_t):
    """
    Bilinear traction-separation law (mixed mode).

    Parameters
    ----------
    delta_n : float
        Normal separation (positive = opening away from tooth).
    delta_t : float
        Tangential separation (sliding along interface).
    sigma_max : float
        Peak normal traction [Pa].
    tau_max : float
        Peak shear traction [Pa].
    delta_c_n : float
        Critical normal separation (full debond).
    delta_c_t : float
        Critical tangential separation (full debond).

    Returns
    -------
    t_n : float
        Normal traction.
    t_t : float
        Tangential traction.
    D : float
        Scalar damage variable, 0 (intact) to 1 (fully debonded).
    """
    ratio = 0.1  # onset at 10% of critical separation
    delta_0_n = delta_c_n * ratio
    delta_0_t = delta_c_t * ratio

    # Penalty stiffnesses (initial slope)
    K_n = sigma_max / max(delta_0_n, 1e-15)
    K_t = tau_max / max(delta_0_t, 1e-15)

    # Normalised separations for mixed-mode coupling
    lambda_n = abs(delta_n) / max(delta_c_n, 1e-15)
    lambda_t = abs(delta_t) / max(delta_c_t, 1e-15)
    lambda_eff = np.sqrt(lambda_n**2 + lambda_t**2)

    lambda_0 = ratio
    lambda_c = 1.0

    # Damage from effective separation
    if lambda_eff <= lambda_0:
        D = 0.0
    elif lambda_eff >= lambda_c:
        D = 1.0
    else:
        D = (lambda_eff - lambda_0) / (lambda_c - lambda_0)
        D = np.clip(D, 0.0, 1.0)

    # Tractions: (1 - D) * K * delta
    if delta_n > 0:
        t_n = (1.0 - D) * K_n * delta_n
    else:
        # Compression: penalty contact, no damage effect
        t_n = K_n * delta_n

    t_t = (1.0 - D) * K_t * delta_t

    return t_n, t_t, D


def compute_interface_strength(DI, sigma_max_healthy=10.0,
                               sigma_max_dysbiotic=1.0, n=2):
    """
    DI-dependent interface strength.

    sigma_max(DI) = sigma_min + (sigma_max - sigma_min) * (1 - DI)^n

    Dysbiotic biofilm (high DI) has weaker adhesion to tooth surface.
    Values scaled to be comparable with bulk E (~30-1000 Pa).
    """
    DI = np.clip(DI, 0.0, 1.0)
    return sigma_max_dysbiotic + (sigma_max_healthy - sigma_max_dysbiotic) * (1.0 - DI) ** n


# -- Cohesive Interface -----------------------------------------------------


class CohesiveInterface:
    """
    Zero-thickness cohesive interface at the tooth-biofilm boundary.

    Interface nodes sit on the bottom boundary (y ~ y_min).
    The tooth surface is treated as rigid at y = y_min.
    Separation = displacement of interface node relative to tooth.
    """

    def __init__(self, interface_nodes, vertices, sigma_max, tau_max,
                 delta_c_n=0.01, delta_c_t=0.02, penalty_stiffness=1e6):
        self.interface_nodes = np.asarray(interface_nodes, dtype=int)
        self.vertices = vertices
        self.n_interface = len(self.interface_nodes)

        self.sigma_max = np.asarray(sigma_max, dtype=float)
        self.tau_max = np.asarray(tau_max, dtype=float)
        self.delta_c_n = delta_c_n
        self.delta_c_t = delta_c_t
        self.penalty_stiffness = penalty_stiffness

        # Per-node state
        self.damage = np.zeros(self.n_interface)
        self.max_lambda_eff = np.zeros(self.n_interface)

        self.n_dofs = 2 * len(vertices)

    def compute_traction_stiffness(self, u):
        """
        Compute interface stiffness matrix and residual force vector.

        For each interface node i:
          - Separation: delta_n = u_y(i), delta_t = u_x(i)
          - Apply TSL to get traction and tangent stiffness
          - Assemble into global K_interface and F_interface
        """
        rows = []
        cols = []
        vals = []
        F_interface = np.zeros(self.n_dofs)

        ratio = 0.1

        for idx in range(self.n_interface):
            node = self.interface_nodes[idx]
            dof_x = 2 * node
            dof_y = 2 * node + 1

            delta_t = u[dof_x]
            delta_n = u[dof_y]

            sig_max = self.sigma_max[idx]
            tau_max_i = self.tau_max[idx]

            t_n, t_t, D_new = bilinear_tsl(
                delta_n, delta_t, sig_max, tau_max_i,
                self.delta_c_n, self.delta_c_t
            )

            # Irreversibility
            D_new = max(D_new, self.damage[idx])
            self.damage[idx] = D_new

            # Tangent stiffness: secant approximation
            delta_0_n = self.delta_c_n * ratio
            delta_0_t = self.delta_c_t * ratio
            K_n_full = sig_max / max(delta_0_n, 1e-15)
            K_t_full = tau_max_i / max(delta_0_t, 1e-15)

            K_n_eff = (1.0 - D_new) * K_n_full
            K_t_eff = (1.0 - D_new) * K_t_full

            if delta_n <= 0:
                K_n_eff = K_n_full

            rows.append(dof_x)
            cols.append(dof_x)
            vals.append(K_t_eff)

            rows.append(dof_y)
            cols.append(dof_y)
            vals.append(K_n_eff)

            F_interface[dof_x] -= t_t
            F_interface[dof_y] -= t_n

        K_interface = sp.csr_matrix(
            (np.array(vals), (np.array(rows), np.array(cols))),
            shape=(self.n_dofs, self.n_dofs)
        )

        return K_interface, F_interface

    def get_damage(self):
        """Return per-interface-node damage array."""
        return self.damage.copy()

    def get_debonded_length(self, threshold=0.95):
        """Fraction of interface nodes that are fully debonded."""
        n_debonded = np.sum(self.damage > threshold)
        return n_debonded / max(self.n_interface, 1)

    def get_debonded_physical_length(self, threshold=0.95):
        """Physical x-extent of debonded region."""
        debonded_mask = self.damage > threshold
        if not np.any(debonded_mask):
            return 0.0
        debonded_nodes = self.interface_nodes[debonded_mask]
        x_coords = self.vertices[debonded_nodes, 0]
        return np.max(x_coords) - np.min(x_coords)


# -- CZM VEM Solver ---------------------------------------------------------


class CZM_VEM_Solver:
    """
    Coupled VEM elasticity + cohesive zone model solver.

    Incremental loading with interface nonlinearity:
      1. Assemble VEM bulk stiffness
      2. Add CZM interface stiffness K_interface
      3. Solve coupled system K_total * u = F
      4. Update interface damage (irreversible)
    """

    def __init__(self, vertices, elements, E_field, nu, interface,
                 bc_fixed_dofs=None, bc_vals=None):
        self.vertices = vertices
        self.elements = elements
        self.E_field = np.asarray(E_field, dtype=float)
        self.nu = nu
        self.interface = interface

        self.n_nodes = len(vertices)
        self.n_dofs = 2 * self.n_nodes
        self.n_el = len(elements)

        self.bc_fixed_dofs = np.asarray(bc_fixed_dofs, dtype=int) if bc_fixed_dofs is not None else np.array([], dtype=int)
        self.bc_vals = np.asarray(bc_vals, dtype=float) if bc_vals is not None else np.array([], dtype=float)

        self.u = np.zeros(self.n_dofs)

    def _assemble_bulk_stiffness(self):
        """Assemble VEM bulk elasticity stiffness matrix."""
        n_polys = 6

        row_idx = []
        col_idx = []
        val_data = []

        for el_id in range(self.n_el):
            vert_ids = self.elements[el_id].astype(int)
            verts = self.vertices[vert_ids]
            n_v = len(vert_ids)
            n_el_dofs = 2 * n_v

            E_el = self.E_field[el_id] if self.E_field.ndim > 0 and len(self.E_field) > 1 else float(self.E_field)

            C_mat = (E_el / (1.0 - self.nu**2)) * np.array([
                [1.0,      self.nu, 0.0],
                [self.nu,  1.0,     0.0],
                [0.0,      0.0,     (1.0 - self.nu) / 2.0],
            ])

            area, centroid, h = _element_geometry(verts)
            xc, yc = centroid

            vnormals = np.zeros((n_v, 2))
            for i in range(n_v):
                prev_v = verts[(i - 1) % n_v]
                next_v = verts[(i + 1) % n_v]
                vnormals[i] = [next_v[1] - prev_v[1], prev_v[0] - next_v[0]]

            D = np.zeros((n_el_dofs, n_polys))
            for i in range(n_v):
                dx = (verts[i, 0] - xc) / h
                dy = (verts[i, 1] - yc) / h
                D[2 * i, :]     = [1.0, 0.0, -dy, dx, 0.0, dy]
                D[2 * i + 1, :] = [0.0, 1.0,  dx, 0.0, dy, dx]

            B = np.zeros((n_polys, n_el_dofs))
            for i in range(n_v):
                B[0, 2 * i] = 1.0 / n_v
                B[1, 2 * i + 1] = 1.0 / n_v
            for i in range(n_v):
                B[2, 2 * i]     = -vnormals[i, 1] / (4.0 * area)
                B[2, 2 * i + 1] =  vnormals[i, 0] / (4.0 * area)

            strain_basis = np.array([
                [1.0 / h, 0.0,     0.0],
                [0.0,     1.0 / h, 0.0],
                [0.0,     0.0,     2.0 / h],
            ])
            for i in range(n_v):
                vn = vnormals[i]
                for alpha in range(3):
                    sigma = C_mat @ strain_basis[alpha]
                    tx = sigma[0] * vn[0] + sigma[2] * vn[1]
                    ty = sigma[2] * vn[0] + sigma[1] * vn[1]
                    B[3 + alpha, 2 * i]     += 0.5 * tx
                    B[3 + alpha, 2 * i + 1] += 0.5 * ty

            G = B @ D
            projector = np.linalg.solve(G, B)

            G_tilde = G.copy()
            G_tilde[:3, :] = 0.0

            K_cons = projector.T @ G_tilde @ projector
            I_minus_PiD = np.eye(n_el_dofs) - D @ projector
            trace_k = np.trace(K_cons)
            stab_param = 0.5 * abs(trace_k) / n_el_dofs if trace_k > 0 else E_el * 0.01
            K_stab = stab_param * (I_minus_PiD.T @ I_minus_PiD)
            K_local = K_cons + K_stab

            gdofs = np.zeros(n_el_dofs, dtype=int)
            for i in range(n_v):
                gdofs[2 * i]     = 2 * vert_ids[i]
                gdofs[2 * i + 1] = 2 * vert_ids[i] + 1

            ii, jj = np.meshgrid(gdofs, gdofs, indexing='ij')
            row_idx.append(ii.ravel())
            col_idx.append(jj.ravel())
            val_data.append(K_local.ravel())

        K_bulk = sp.csr_matrix(
            (np.concatenate(val_data),
             (np.concatenate(row_idx), np.concatenate(col_idx))),
            shape=(self.n_dofs, self.n_dofs)
        )
        return K_bulk

    def solve_step(self, load_dofs, load_vals):
        """
        Solve one load step: bulk VEM + CZM interface.

        Returns u, interface_damage.
        """
        K_bulk = self._assemble_bulk_stiffness()
        K_interface, F_interface = self.interface.compute_traction_stiffness(self.u)
        K_total = K_bulk + K_interface

        F_ext = np.zeros(self.n_dofs)
        if load_dofs is not None and load_vals is not None:
            F_ext[load_dofs] += load_vals

        F_rhs = F_ext.copy()

        u = np.zeros(self.n_dofs)
        bc_set = set(self.bc_fixed_dofs.tolist())
        internal = np.array([i for i in range(self.n_dofs) if i not in bc_set])

        u[self.bc_fixed_dofs] = self.bc_vals
        F_rhs -= K_total[:, self.bc_fixed_dofs].toarray() @ self.bc_vals

        K_ii = K_total[np.ix_(internal, internal)]

        try:
            u[internal] = sp.linalg.spsolve(K_ii, F_rhs[internal])
        except Exception:
            u = self.u.copy()

        self.u = u
        self.interface.compute_traction_stiffness(self.u)

        return u, self.interface.get_damage()

    def run(self, n_steps=30, load_factor_max=3.0, load_dofs=None,
            load_vals_unit=None, verbose=True):
        """Run incremental loading simulation."""
        if verbose:
            print("=" * 65)
            print("CZM-VEM Solver: Tooth-Biofilm Interface Detachment")
            print(f"  Elements: {self.n_el}, Nodes: {self.n_nodes}, "
                  f"Interface nodes: {self.interface.n_interface}")
            print(f"  Load steps: {n_steps}, max load factor: {load_factor_max}")
            print("=" * 65)

        snapshots = []

        for step in range(n_steps):
            lf = (step + 1) / n_steps * load_factor_max

            if load_dofs is not None and load_vals_unit is not None:
                l_vals = lf * load_vals_unit
            else:
                l_vals = None

            u, damage = self.solve_step(load_dofs, l_vals)

            ux = u[0::2]
            uy = u[1::2]
            u_mag = np.sqrt(ux**2 + uy**2)

            debond_frac = self.interface.get_debonded_length(threshold=0.95)
            debond_len = self.interface.get_debonded_physical_length(threshold=0.95)

            snapshot = {
                'step': step,
                'load_factor': lf,
                'u': u.copy(),
                'u_max': np.max(u_mag),
                'damage': damage.copy(),
                'damage_max': np.max(damage),
                'damage_mean': np.mean(damage),
                'debond_fraction': debond_frac,
                'debond_length': debond_len,
                'n_debonded': int(np.sum(damage > 0.95)),
            }
            snapshots.append(snapshot)

            if verbose:
                print(f"  Step {step+1:3d}/{n_steps} | LF={lf:.3f} | "
                      f"|u|_max={np.max(u_mag):.4e} | "
                      f"D_max={np.max(damage):.4f} | "
                      f"debond={debond_frac*100:.1f}% "
                      f"({snapshot['n_debonded']}/{self.interface.n_interface} nodes)")

        if verbose:
            print("-" * 65)
            final = snapshots[-1]
            print(f"  Final: |u|_max={final['u_max']:.4e}, "
                  f"D_max={final['damage_max']:.4f}, "
                  f"debond={final['debond_fraction']*100:.1f}%")
            print("=" * 65)

        return snapshots


# -- Demo -------------------------------------------------------------------


def demo_biofilm_czm():
    """
    Demo: CZM-based tooth-biofilm interface detachment.

    40-cell Voronoi biofilm, domain (0,2)x(0,1).
    Dysbiotic center => weak interface at center of bottom boundary.
    Increasing GCF shear load on top surface.

    Generates 2x3 figure:
      (a) DI spatial field
      (b) Interface strength sigma_max(DI) along bottom boundary
      (c) Deformed mesh at failure (colored by |u|)
      (d) Interface damage D along bottom vs load step (heatmap)
      (e) Load-displacement curve with debond annotation
      (f) Traction-separation curve: center node vs edge node
    """
    rng = np.random.default_rng(42)
    domain = (0, 2, 0, 1)
    n_cells = 40
    xmin, xmax, ymin, ymax = domain
    nu = 0.35

    # -- Generate Voronoi mesh --
    nx = int(np.sqrt(n_cells * 2))
    ny = max(n_cells // nx, 2)
    xx = np.linspace(xmin + 0.1, xmax - 0.1, nx)
    yy = np.linspace(ymin + 0.05, ymax - 0.05, ny)
    gx, gy = np.meshgrid(xx, yy)
    seeds = np.column_stack([gx.ravel(), gy.ravel()])[:n_cells]
    seeds += rng.uniform(-0.03, 0.03, seeds.shape)

    vertices, elements, bnd, valid_ids = make_biofilm_voronoi(seeds, domain)
    n_el = len(elements)

    # -- Compact mesh --
    used_set = set()
    for el in elements:
        used_set.update(el.astype(int).tolist())
    used = np.array(sorted(used_set))
    old_to_new = {int(g): i for i, g in enumerate(used)}

    compact_verts = vertices[used]
    compact_elems = [np.array([old_to_new[int(v)] for v in el]) for el in elements]

    # -- Spatial DI gradient --
    xmid = (xmin + xmax) / 2
    ymid = (ymin + ymax) / 2
    DI_per_cell = np.zeros(n_el)
    for i, el in enumerate(compact_elems):
        el_int = el.astype(int)
        cx = np.mean(compact_verts[el_int, 0])
        cy = np.mean(compact_verts[el_int, 1])
        r = np.sqrt((cx - xmid)**2 + (cy - ymid)**2)
        r_max = np.sqrt((xmid - xmin)**2 + (ymid - ymin)**2)
        proximity = 1.0 - r / r_max
        DI_per_cell[i] = np.clip(0.15 + 0.65 * proximity, 0.0, 1.0)

    E_field = compute_E_from_DI(DI_per_cell)

    # -- Identify bottom interface nodes --
    tol_bc = 0.02
    bottom_mask = compact_verts[:, 1] < ymin + tol_bc
    interface_nodes = np.where(bottom_mask)[0]
    x_order = np.argsort(compact_verts[interface_nodes, 0])
    interface_nodes = interface_nodes[x_order]

    # -- DI at interface nodes --
    DI_at_interface = np.zeros(len(interface_nodes))
    for idx, node in enumerate(interface_nodes):
        nx_coord = compact_verts[node, 0]
        ny_coord = compact_verts[node, 1]
        min_dist = np.inf
        best_di = 0.5
        for i, el in enumerate(compact_elems):
            el_int = el.astype(int)
            cx = np.mean(compact_verts[el_int, 0])
            cy = np.mean(compact_verts[el_int, 1])
            dist = np.sqrt((nx_coord - cx)**2 + (ny_coord - cy)**2)
            if dist < min_dist:
                min_dist = dist
                best_di = DI_per_cell[i]
        DI_at_interface[idx] = best_di

    # -- Interface strength from DI --
    sigma_max_per_node = compute_interface_strength(DI_at_interface)
    tau_max_per_node = sigma_max_per_node * 0.8

    # -- Build CZM interface --
    # delta_c ~ expected displacement scale, sigma_max ~ E scale
    interface = CohesiveInterface(
        interface_nodes, compact_verts,
        sigma_max=sigma_max_per_node,
        tau_max=tau_max_per_node,
        delta_c_n=0.05,
        delta_c_t=0.05,
    )

    # -- BCs: pin one corner to prevent rigid body motion --
    # CZM at bottom provides the main constraint; only pin left-bottom corner
    left_bottom = np.where(
        (compact_verts[:, 0] < xmin + tol_bc) &
        (compact_verts[:, 1] < ymin + tol_bc)
    )[0]
    if len(left_bottom) == 0:
        left_bottom = np.array([interface_nodes[0]])
    bc_dofs = np.array([2 * left_bottom[0]])  # fix x of one node only
    bc_vals = np.zeros(1)

    # -- Load: shear (x) on top surface --
    top_mask = compact_verts[:, 1] > ymax - tol_bc
    top_nodes = np.where(top_mask)[0]

    load_dofs_list = []
    load_vals_list = []
    if len(top_nodes) > 0:
        load_dofs_list.append(2 * top_nodes)
        load_vals_list.append(np.full(len(top_nodes), 1.0 / len(top_nodes)))
        load_dofs_list.append(2 * top_nodes + 1)
        load_vals_list.append(np.full(len(top_nodes), -0.2 / len(top_nodes)))

    load_dofs = np.concatenate(load_dofs_list)
    load_vals_unit = np.concatenate(load_vals_list)

    # -- Run CZM solver --
    solver = CZM_VEM_Solver(
        compact_verts, compact_elems, E_field, nu, interface,
        bc_fixed_dofs=bc_dofs, bc_vals=bc_vals,
    )

    n_steps = 50
    snapshots = solver.run(
        n_steps=n_steps, load_factor_max=30.0,
        load_dofs=load_dofs, load_vals_unit=load_vals_unit,
        verbose=True,
    )

    # -- TSL history for center and edge nodes --
    print("\nRecording traction-separation history for center/edge nodes...")

    interface_x = compact_verts[interface_nodes, 0]
    center_idx = np.argmin(np.abs(interface_x - xmid))
    edge_idx = 0

    center_node = interface_nodes[center_idx]
    edge_node = interface_nodes[edge_idx]

    tsl_center = {'delta_t': [], 'delta_n': [], 't_t': [], 't_n': [], 'D': []}
    tsl_edge = {'delta_t': [], 'delta_n': [], 't_t': [], 't_n': [], 'D': []}

    for snap in snapshots:
        u = snap['u']
        for node, tsl_dict, nidx in [
            (center_node, tsl_center, center_idx),
            (edge_node, tsl_edge, edge_idx),
        ]:
            dt = u[2 * node]
            dn = u[2 * node + 1]
            sig_m = sigma_max_per_node[nidx]
            tau_m = tau_max_per_node[nidx]
            tn, tt, D = bilinear_tsl(dn, dt, sig_m, tau_m, 0.05, 0.05)
            tsl_dict['delta_t'].append(dt)
            tsl_dict['delta_n'].append(dn)
            tsl_dict['t_t'].append(tt)
            tsl_dict['t_n'].append(tn)
            tsl_dict['D'].append(D)

    # ── Plot 2x3 figure ──
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # (a) DI field
    ax = axes[0, 0]
    patches = [MplPolygon(compact_verts[el.astype(int)], closed=True)
               for el in compact_elems]
    pc = PatchCollection(patches, cmap='RdYlGn_r', edgecolor='k', linewidth=0.3)
    pc.set_array(DI_per_cell)
    ax.add_collection(pc)
    ax.set_xlim(xmin - 0.05, xmax + 0.05)
    ax.set_ylim(ymin - 0.05, ymax + 0.05)
    ax.set_aspect('equal')
    fig.colorbar(pc, ax=ax, label='DI', shrink=0.8)
    ax.set_title('(a) Dysbiosis Index')

    # (b) Interface strength along bottom
    ax = axes[0, 1]
    x_interface = compact_verts[interface_nodes, 0]
    ax.fill_between(x_interface, 0, sigma_max_per_node, alpha=0.3, color='steelblue')
    ax.plot(x_interface, sigma_max_per_node, 'b-o', ms=4, lw=1.5,
            label=r'$\sigma_{max}$(DI)')
    ax.plot(x_interface, tau_max_per_node, 'r--s', ms=3, lw=1.2,
            label=r'$\tau_{max}$(DI)')
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('Interface Strength [Pa]')
    ax.set_title(r'(b) Interface Strength $\sigma_{max}$(DI)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(xmin, xmax)

    # (c) Deformed mesh at failure
    ax = axes[0, 2]
    final = snapshots[-1]
    ux = final['u'][0::2]
    uy = final['u'][1::2]
    mag = np.sqrt(ux**2 + uy**2)
    scale = 20.0
    deformed = compact_verts + scale * np.column_stack([ux, uy])

    patches = [MplPolygon(deformed[el.astype(int)], closed=True)
               for el in compact_elems]
    colors = [np.mean(mag[el.astype(int)]) for el in compact_elems]
    pc = PatchCollection(patches, cmap='hot_r', edgecolor='k', linewidth=0.3)
    pc.set_array(np.array(colors))
    ax.add_collection(pc)
    ax.set_xlim(xmin - 0.2, xmax + 0.5)
    ax.set_ylim(ymin - 0.2, ymax + 0.2)
    ax.set_aspect('equal')
    fig.colorbar(pc, ax=ax, label='|u|', shrink=0.8)
    ax.set_title(f'(c) Deformed (x{scale:.0f}), |u|_max={final["u_max"]:.3e}')

    # (d) Interface damage heatmap
    ax = axes[1, 0]
    damage_history = np.array([s['damage'] for s in snapshots])
    load_factors = [s['load_factor'] for s in snapshots]

    im = ax.imshow(
        damage_history.T, aspect='auto', origin='lower',
        cmap='inferno', vmin=0, vmax=1,
        extent=[load_factors[0], load_factors[-1],
                x_interface[0], x_interface[-1]],
    )
    ax.set_xlabel('Load Factor')
    ax.set_ylabel('x along interface [mm]')
    fig.colorbar(im, ax=ax, label='Damage D', shrink=0.8)
    ax.set_title('(d) Interface Damage Evolution')

    # (e) Load-displacement + debond
    ax = axes[1, 1]
    lf_arr = np.array([s['load_factor'] for s in snapshots])
    u_max_arr = np.array([s['u_max'] for s in snapshots])
    debond_arr = np.array([s['debond_fraction'] for s in snapshots])

    ax.plot(lf_arr, u_max_arr, 'b-o', ms=4, lw=1.5, label='|u|_max')
    ax.set_xlabel('Load Factor')
    ax.set_ylabel('|u|_max', color='b')
    ax.tick_params(axis='y', labelcolor='b')

    ax2 = ax.twinx()
    ax2.plot(lf_arr, debond_arr * 100, 'r-s', ms=3, lw=1.2, label='Debond %')
    ax2.set_ylabel('Debonded [%]', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.set_ylim(-5, 105)

    debond_onset = None
    for s in snapshots:
        if s['debond_fraction'] > 0.01:
            debond_onset = s
            break
    if debond_onset is not None:
        ax.axvline(debond_onset['load_factor'], color='gray', ls='--', alpha=0.5)
        ax.annotate(f"onset LF={debond_onset['load_factor']:.2f}",
                    xy=(debond_onset['load_factor'], debond_onset['u_max']),
                    xytext=(debond_onset['load_factor'] + 0.3, debond_onset['u_max']),
                    arrowprops=dict(arrowstyle='->', color='gray'),
                    fontsize=9, color='gray')

    ax.legend(loc='upper left')
    ax2.legend(loc='center right')
    ax.set_title('(e) Load-Displacement + Debond Length')
    ax.grid(True, alpha=0.3)

    # (f) Traction-separation: center vs edge
    ax = axes[1, 2]
    ax.plot(np.abs(tsl_center['delta_t']), np.abs(tsl_center['t_t']),
            'r-o', ms=3, lw=1.5,
            label=f'Center (x={compact_verts[center_node, 0]:.2f})')
    ax.plot(np.abs(tsl_edge['delta_t']), np.abs(tsl_edge['t_t']),
            'b-s', ms=3, lw=1.5,
            label=f'Edge (x={compact_verts[edge_node, 0]:.2f})')

    ax.axhline(sigma_max_per_node[center_idx] * 0.8, color='r',
               ls=':', alpha=0.4, label=f'Center peak={sigma_max_per_node[center_idx]*0.8:.0f} Pa')
    ax.axhline(sigma_max_per_node[edge_idx] * 0.8, color='b',
               ls=':', alpha=0.4, label=f'Edge peak={sigma_max_per_node[edge_idx]*0.8:.0f} Pa')

    ax.set_xlabel(r'Tangential Separation $|\delta_t|$')
    ax.set_ylabel(r'Shear Traction $|t_t|$ [Pa]')
    ax.set_title('(f) Traction-Separation (Shear)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        'CZM-VEM: Tooth-Biofilm Interface Detachment\n'
        '(Dysbiotic center debonds first — weak DI-dependent adhesion)',
        fontsize=14, fontweight='bold',
    )
    plt.tight_layout()

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, 'vem_czm_demo.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {path}")
    plt.close()

    return snapshots


if __name__ == '__main__':
    demo_biofilm_czm()
