"""
VEM for 3D Linear Elasticity on Polyhedral Meshes.

Extension of 2D VEM elasticity to 3D:
  - 3 DOFs/node (u_x, u_y, u_z)
  - 12 polynomial basis: 3 translations + 3 rotations + 6 strain modes
  - Face integrals replace edge integrals

Polynomial basis P_1^3 (dim=12):
  Rigid body (6):
    p1  = (1, 0, 0)                          — translation x
    p2  = (0, 1, 0)                          — translation y
    p3  = (0, 0, 1)                          — translation z
    p4  = (0, -(z-zc)/h, (y-yc)/h)          — rotation about x
    p5  = ((z-zc)/h, 0, -(x-xc)/h)          — rotation about y
    p6  = (-(y-yc)/h, (x-xc)/h, 0)          — rotation about z
  Strain modes (6):
    p7  = ((x-xc)/h, 0, 0)                  — ε_xx
    p8  = (0, (y-yc)/h, 0)                  — ε_yy
    p9  = (0, 0, (z-zc)/h)                  — ε_zz
    p10 = (0, (z-zc)/h, (y-yc)/h)           — ε_yz (symmetric shear)
    p11 = ((z-zc)/h, 0, (x-xc)/h)           — ε_xz (symmetric shear)
    p12 = ((y-yc)/h, (x-xc)/h, 0)           — ε_xy (symmetric shear)

References:
  - Gain, Talischi, Paulino (2014) "VEM for 3D linear elasticity"
  - Beirao da Veiga et al. (2013) "Basic principles of VEM"
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ── Mesh Generation ───────────────────────────────────────────────────────

def make_hex_mesh(nx=3, ny=3, nz=3, perturb=0.15, seed=42):
    """
    Create a (nx×ny×nz) grid of perturbed hexahedral elements in [0,1]^3.
    Interior nodes are randomly perturbed to create irregular polyhedra.
    """
    rng = np.random.default_rng(seed)

    xs = np.linspace(0, 1, nx + 1)
    ys = np.linspace(0, 1, ny + 1)
    zs = np.linspace(0, 1, nz + 1)

    vertices = []
    node_map = {}
    idx = 0
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                v = np.array([xs[i], ys[j], zs[k]])
                # Perturb interior nodes
                on_boundary = (i in (0, nx) or j in (0, ny) or k in (0, nz))
                if not on_boundary:
                    scale = perturb / max(nx, ny, nz)
                    v += rng.uniform(-scale, scale, 3)
                node_map[(i, j, k)] = idx
                vertices.append(v)
                idx += 1

    vertices = np.array(vertices)

    #   7----6       Hex vertex ordering (standard):
    #  /|   /|       Bottom: 0,1,2,3  Top: 4,5,6,7
    # 4----5 |
    # | 3--|-2
    # |/   |/
    # 0----1

    cells = []        # list of arrays of vertex indices
    cell_faces = []   # list of lists of face arrays (ordered for outward normal)

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                v = [node_map[(i, j, k)],     node_map[(i+1, j, k)],
                     node_map[(i+1, j+1, k)], node_map[(i, j+1, k)],
                     node_map[(i, j, k+1)],   node_map[(i+1, j, k+1)],
                     node_map[(i+1, j+1, k+1)], node_map[(i, j+1, k+1)]]

                cells.append(np.array(v))

                # 6 faces with outward normals (right-hand rule)
                faces = [
                    np.array([v[0], v[3], v[2], v[1]]),  # bottom z-
                    np.array([v[4], v[5], v[6], v[7]]),  # top    z+
                    np.array([v[0], v[1], v[5], v[4]]),  # front  y-
                    np.array([v[2], v[3], v[7], v[6]]),  # back   y+
                    np.array([v[0], v[4], v[7], v[3]]),  # left   x-
                    np.array([v[1], v[2], v[6], v[5]]),  # right  x+
                ]
                cell_faces.append(faces)

    return vertices, cells, cell_faces


# ── Geometry Helpers ──────────────────────────────────────────────────────

def face_normal_area(pts):
    """
    Compute unit outward normal and area for a planar polygon.
    pts: (k, 3) ordered vertices.
    """
    normal = np.zeros(3)
    for i in range(1, len(pts) - 1):
        normal += np.cross(pts[i] - pts[0], pts[i + 1] - pts[0])
    area = np.linalg.norm(normal) / 2.0
    unit_n = normal / (np.linalg.norm(normal) + 1e-30)
    return unit_n, area


def polyhedron_volume(vertices, faces):
    """Compute volume using divergence theorem: V = (1/6) Σ v0·(v1×v2)."""
    vol = 0.0
    for face in faces:
        pts = vertices[face]
        for i in range(1, len(pts) - 1):
            vol += np.dot(pts[0], np.cross(pts[i], pts[i + 1]))
    return abs(vol) / 6.0


# ── 3D Constitutive Matrix ───────────────────────────────────────────────

def isotropic_3d(E, nu):
    """3D isotropic elasticity matrix (Voigt: [σxx, σyy, σzz, σyz, σxz, σxy])."""
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    C = np.array([
        [lam + 2*mu, lam,       lam,       0,  0,  0],
        [lam,        lam + 2*mu, lam,       0,  0,  0],
        [lam,        lam,       lam + 2*mu, 0,  0,  0],
        [0,          0,         0,          mu, 0,  0],
        [0,          0,         0,          0,  mu, 0],
        [0,          0,         0,          0,  0,  mu],
    ])
    return C


def traction_from_voigt(sigma_voigt, n):
    """Compute traction t = σ·n from Voigt stress and normal vector."""
    sxx, syy, szz, syz, sxz, sxy = sigma_voigt
    tx = sxx * n[0] + sxy * n[1] + sxz * n[2]
    ty = sxy * n[0] + syy * n[1] + syz * n[2]
    tz = sxz * n[0] + syz * n[1] + szz * n[2]
    return np.array([tx, ty, tz])


# ── VEM 3D Solver ─────────────────────────────────────────────────────────

def vem_3d_elasticity(vertices, cells, cell_faces, E_field, nu,
                      bc_fixed_dofs, bc_vals,
                      load_dofs=None, load_vals=None):
    """
    Lowest-order VEM for 3D linear elasticity on polyhedral meshes.

    Parameters
    ----------
    vertices    : (N, 3) node coordinates
    cells       : list of int arrays — vertex indices per cell
    cell_faces  : list of lists of int arrays — face vertex indices per cell
    E_field     : float or (N_el,) Young's modulus per element
    nu          : float Poisson's ratio
    bc_fixed_dofs : int array — constrained DOF indices
    bc_vals     : float array — prescribed displacement values
    load_dofs   : int array — DOFs with point loads
    load_vals   : float array — load magnitudes

    Returns
    -------
    u : (3*N,) displacement vector
    """
    n_nodes = len(vertices)
    n_dofs = 3 * n_nodes
    n_polys = 12

    K_global = np.zeros((n_dofs, n_dofs))
    F_global = np.zeros(n_dofs)

    # Strain basis in Voigt [ε_xx, ε_yy, ε_zz, 2ε_yz, 2ε_xz, 2ε_xy]
    # (multiplied by h later)
    strain_ids = np.array([
        [1, 0, 0, 0, 0, 0],   # p7:  ε_xx
        [0, 1, 0, 0, 0, 0],   # p8:  ε_yy
        [0, 0, 1, 0, 0, 0],   # p9:  ε_zz
        [0, 0, 0, 2, 0, 0],   # p10: 2ε_yz
        [0, 0, 0, 0, 2, 0],   # p11: 2ε_xz
        [0, 0, 0, 0, 0, 2],   # p12: 2ε_xy
    ], dtype=float)

    for el_id in range(len(cells)):
        vert_ids = cells[el_id].astype(int)
        coords = vertices[vert_ids]
        faces = cell_faces[el_id]
        n_v = len(vert_ids)
        n_el = 3 * n_v

        # Local vertex index map
        vmap = {int(g): loc for loc, g in enumerate(vert_ids)}

        E_el = E_field[el_id] if hasattr(E_field, '__len__') else E_field
        C = isotropic_3d(E_el, nu)

        # ── Geometry ──
        centroid = coords.mean(axis=0)
        h = max(np.linalg.norm(coords[i] - coords[j])
                for i in range(n_v) for j in range(i + 1, n_v))
        vol = polyhedron_volume(vertices, faces)

        xc, yc, zc = centroid

        # ── D matrix (3·n_v × 12) ──
        D = np.zeros((n_el, n_polys))
        for i in range(n_v):
            dx = (coords[i, 0] - xc) / h
            dy = (coords[i, 1] - yc) / h
            dz = (coords[i, 2] - zc) / h
            r = 3 * i
            #        p1  p2  p3  p4   p5  p6   p7  p8  p9  p10 p11 p12
            D[r,   :] = [1, 0, 0, 0,   dz, -dy, dx, 0,  0,  0,  dz, dy]
            D[r+1, :] = [0, 1, 0, -dz, 0,  dx,  0,  dy, 0,  dz, 0,  dx]
            D[r+2, :] = [0, 0, 1, dy,  -dx, 0,  0,  0,  dz, dy, dx, 0]

        # ── B matrix (12 × 3·n_v) ──
        B = np.zeros((n_polys, n_el))

        # Rows 0-2: translations (average displacement)
        for i in range(n_v):
            B[0, 3 * i]     = 1.0 / n_v
            B[1, 3 * i + 1] = 1.0 / n_v
            B[2, 3 * i + 2] = 1.0 / n_v

        # Process each face for rotation and strain rows
        for face in faces:
            face_int = face.astype(int)
            pts = vertices[face_int]
            n_f, A_f = face_normal_area(pts)

            # Orient normal outward (away from cell centroid)
            fc = pts.mean(axis=0)
            if np.dot(n_f, fc - centroid) < 0:
                n_f = -n_f

            k_f = len(face_int)

            for gv in face_int:
                if gv not in vmap:
                    continue
                li = vmap[gv]  # local index
                w = A_f / k_f

                # Rows 3-5: rotations from ∫_∂V (u×n) dS
                # ω_x = (1/2)(∂uz/∂y - ∂uy/∂z) → ∫(uz·ny - uy·nz)
                wrot = w / (2.0 * vol)
                B[3, 3*li + 1] += -wrot * n_f[2]   # -nz · uy
                B[3, 3*li + 2] +=  wrot * n_f[1]   # +ny · uz
                B[4, 3*li + 0] +=  wrot * n_f[2]   # +nz · ux
                B[4, 3*li + 2] += -wrot * n_f[0]   # -nx · uz
                B[5, 3*li + 0] += -wrot * n_f[1]   # -ny · ux
                B[5, 3*li + 1] +=  wrot * n_f[0]   # +nx · uy

                # Rows 6-11: strain modes via traction integrals
                for alpha in range(6):
                    eps_a = strain_ids[alpha] / h
                    sigma_a = C @ eps_a
                    t_f = traction_from_voigt(sigma_a, n_f)
                    B[6 + alpha, 3*li + 0] += w * t_f[0]
                    B[6 + alpha, 3*li + 1] += w * t_f[1]
                    B[6 + alpha, 3*li + 2] += w * t_f[2]

        # ── Projector ──
        G = B @ D  # 12 × 12
        projector = np.linalg.solve(G, B)  # 12 × 3·n_v

        # Consistency: zero out rigid body rows
        G_tilde = G.copy()
        G_tilde[:6, :] = 0.0

        K_cons = projector.T @ G_tilde @ projector

        # Stabilization
        I_minus_PiD = np.eye(n_el) - D @ projector
        trace_k = np.trace(K_cons)
        stab_param = trace_k / n_el if trace_k > 0 else E_el
        K_stab = stab_param * (I_minus_PiD.T @ I_minus_PiD)

        K_local = K_cons + K_stab

        # ── Assemble ──
        gdofs = np.zeros(n_el, dtype=int)
        for i in range(n_v):
            gdofs[3*i]     = 3 * vert_ids[i]
            gdofs[3*i + 1] = 3 * vert_ids[i] + 1
            gdofs[3*i + 2] = 3 * vert_ids[i] + 2

        for i in range(n_el):
            for j in range(n_el):
                K_global[gdofs[i], gdofs[j]] += K_local[i, j]

    # ── Point loads ──
    if load_dofs is not None and load_vals is not None:
        F_global[load_dofs] += load_vals

    # ── Solve ──
    u = np.zeros(n_dofs)
    bc_set = set(bc_fixed_dofs)
    internal = np.array([i for i in range(n_dofs) if i not in bc_set])

    u[bc_fixed_dofs] = bc_vals
    F_global -= K_global[:, bc_fixed_dofs] @ bc_vals

    K_ii = K_global[np.ix_(internal, internal)]
    u[internal] = np.linalg.solve(K_ii, F_global[internal])

    return u


# ── Visualization ─────────────────────────────────────────────────────────

def plot_3d_vem(vertices, cells, cell_faces, u, deform_scale=1.0,
                title=None, save=None):
    """Plot 3D VEM result: deformed mesh colored by displacement magnitude."""
    ux = u[0::3]
    uy = u[1::3]
    uz = u[2::3]
    u_mag = np.sqrt(ux**2 + uy**2 + uz**2)
    deformed = vertices + deform_scale * np.column_stack([ux, uy, uz])

    fig = plt.figure(figsize=(16, 6))

    for plot_idx, (coords, label) in enumerate([
        (vertices, 'Undeformed'),
        (deformed, f'Deformed (x{deform_scale})')
    ]):
        ax = fig.add_subplot(1, 2, plot_idx + 1, projection='3d')

        # Collect all faces as polygons
        all_polys = []
        all_colors = []
        for el_id in range(len(cells)):
            for face in cell_faces[el_id]:
                face_int = face.astype(int)
                poly = coords[face_int]
                all_polys.append(poly)
                all_colors.append(np.mean(u_mag[face_int]))

        all_colors = np.array(all_colors)
        norm = plt.Normalize(all_colors.min(), all_colors.max())
        cmap = plt.cm.viridis

        pc = Poly3DCollection(all_polys, alpha=0.7, edgecolor='k',
                              linewidth=0.3)
        face_colors = cmap(norm(all_colors))
        pc.set_facecolor(face_colors)
        ax.add_collection3d(pc)

        ax.set_xlim(coords[:, 0].min() - 0.05, coords[:, 0].max() + 0.05)
        ax.set_ylim(coords[:, 1].min() - 0.05, coords[:, 1].max() + 0.05)
        ax.set_zlim(coords[:, 2].min() - 0.05, coords[:, 2].max() + 0.05)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(label)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        fig.colorbar(sm, ax=ax, label='|u|', shrink=0.6)

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save}")
    plt.close()


# ── Demo 1: 3D Patch Test ────────────────────────────────────────────────

def demo_3d_patch_test():
    """
    Uniform tension σ_xx = 1.
    Exact: u_x = x/E, u_y = -ν·x ... wait, for 3D:
    ε_xx = σ/E, ε_yy = ε_zz = -ν·σ/E
    u_x = σ·x/E, u_y = -ν·σ·y/E, u_z = -ν·σ·z/E
    """
    print("=" * 60)
    print("Demo 1: 3D Patch Test (uniform tension)")
    print("=" * 60)

    vertices, cells, cell_faces = make_hex_mesh(nx=3, ny=3, nz=3, perturb=0.15)
    n_nodes = len(vertices)

    E_mod = 1000.0
    nu = 0.3

    # Exact solution for σ_xx = 1
    sigma = 1.0
    exact_ux = sigma * vertices[:, 0] / E_mod
    exact_uy = -nu * sigma * vertices[:, 1] / E_mod
    exact_uz = -nu * sigma * vertices[:, 2] / E_mod

    # Fix all boundary nodes to exact displacement
    tol = 1e-6
    boundary = np.where(
        (vertices[:, 0] < tol) | (vertices[:, 0] > 1 - tol) |
        (vertices[:, 1] < tol) | (vertices[:, 1] > 1 - tol) |
        (vertices[:, 2] < tol) | (vertices[:, 2] > 1 - tol)
    )[0]

    bc_dofs = np.concatenate([3 * boundary, 3 * boundary + 1, 3 * boundary + 2])
    bc_vals = np.concatenate([exact_ux[boundary], exact_uy[boundary],
                              exact_uz[boundary]])

    u = vem_3d_elasticity(vertices, cells, cell_faces, E_mod, nu,
                          bc_dofs, bc_vals)

    ux = u[0::3]
    uy = u[1::3]
    uz = u[2::3]

    err_x = np.max(np.abs(ux - exact_ux))
    err_y = np.max(np.abs(uy - exact_uy))
    err_z = np.max(np.abs(uz - exact_uz))
    print(f"  Nodes: {n_nodes}, Elements: {len(cells)}")
    print(f"  Max error u_x: {err_x:.2e}")
    print(f"  Max error u_y: {err_y:.2e}")
    print(f"  Max error u_z: {err_z:.2e}")
    passed = max(err_x, err_y, err_z) < 1e-10
    print(f"  PASS: {passed}")
    return passed


# ── Demo 2: 3D Compression ───────────────────────────────────────────────

def demo_3d_compression(save_dir='/tmp'):
    """Compress a cube: fix bottom (z=0), uniform pressure on top (z=1)."""
    print("\n" + "=" * 60)
    print("Demo 2: 3D Cube Compression (perturbed hex mesh)")
    print("=" * 60)

    vertices, cells, cell_faces = make_hex_mesh(nx=4, ny=4, nz=4, perturb=0.2)
    n_nodes = len(vertices)

    E_mod = 1000.0
    nu = 0.3
    tol = 1e-6

    # Fix bottom face (z ≈ 0)
    bottom = np.where(vertices[:, 2] < tol)[0]
    bc_dofs = np.concatenate([3 * bottom, 3 * bottom + 1, 3 * bottom + 2])
    bc_vals = np.zeros(len(bc_dofs))

    # Pressure on top face (z ≈ 1): downward force
    top = np.where(vertices[:, 2] > 1 - tol)[0]
    load_per_node = -1.0 / len(top)
    load_dofs = 3 * top + 2  # z-DOF
    load_vals = np.full(len(top), load_per_node)

    print(f"  Nodes: {n_nodes}, Elements: {len(cells)}")
    print(f"  Fixed (bottom): {len(bottom)}, Loaded (top): {len(top)}")

    u = vem_3d_elasticity(vertices, cells, cell_faces, E_mod, nu,
                          bc_dofs, bc_vals, load_dofs, load_vals)

    ux = u[0::3]
    uy = u[1::3]
    uz = u[2::3]
    print(f"  Max |u_x|: {np.max(np.abs(ux)):.6f}")
    print(f"  Max |u_y|: {np.max(np.abs(uy)):.6f}")
    print(f"  Max |u_z|: {np.max(np.abs(uz)):.6f}")
    print(f"  Top deflection (avg u_z): {np.mean(uz[top]):.6f}")

    plot_3d_vem(vertices, cells, cell_faces, u, deform_scale=200,
                title='3D VEM: Cube Compression (perturbed hex)',
                save=f'{save_dir}/vem_3d_compression.png')

    return u


# ── Demo 3: 3D Biofilm E(DI) ─────────────────────────────────────────────

def demo_3d_biofilm(save_dir='/tmp'):
    """
    3D cube with spatially varying E(DI).
    DI high at center (soft), low at boundary (stiff).
    """
    print("\n" + "=" * 60)
    print("Demo 3: 3D Biofilm E(DI) on Polyhedral Mesh")
    print("=" * 60)

    vertices, cells, cell_faces = make_hex_mesh(nx=4, ny=4, nz=4, perturb=0.2)
    n_nodes = len(vertices)

    E_max = 1000.0
    E_min = 30.0
    n_hill = 2
    nu = 0.3
    center = np.array([0.5, 0.5, 0.5])
    max_dist = 0.5 * np.sqrt(3)

    E_per_el = np.zeros(len(cells))
    DI_per_el = np.zeros(len(cells))
    for i, cell in enumerate(cells):
        cell_int = cell.astype(int)
        el_c = vertices[cell_int].mean(axis=0)
        dist = np.linalg.norm(el_c - center)
        DI = np.clip(0.9 - 0.8 * dist / max_dist, 0.05, 0.95)
        DI_per_el[i] = DI
        E_per_el[i] = E_min + (E_max - E_min) * (1.0 - DI) ** n_hill

    print(f"  DI range: [{DI_per_el.min():.2f}, {DI_per_el.max():.2f}]")
    print(f"  E  range: [{E_per_el.min():.0f}, {E_per_el.max():.0f}] Pa")
    print(f"  E ratio:  {E_per_el.max() / E_per_el.min():.1f}x")

    tol = 1e-6

    # Fix bottom
    bottom = np.where(vertices[:, 2] < tol)[0]
    bc_dofs = np.concatenate([3 * bottom, 3 * bottom + 1, 3 * bottom + 2])
    bc_vals = np.zeros(len(bc_dofs))

    # Pressure on top
    top = np.where(vertices[:, 2] > 1 - tol)[0]
    load_per_node = -0.5 / len(top)
    load_dofs = 3 * top + 2
    load_vals = np.full(len(top), load_per_node)

    u = vem_3d_elasticity(vertices, cells, cell_faces, E_per_el, nu,
                          bc_dofs, bc_vals, load_dofs, load_vals)

    ux = u[0::3]
    uy = u[1::3]
    uz = u[2::3]
    u_mag = np.sqrt(ux**2 + uy**2 + uz**2)
    print(f"  Max |u|: {np.max(u_mag):.6f}")
    print(f"  Center deflection: "
          f"{uz[np.argmin(np.linalg.norm(vertices - center, axis=1))]:.6f}")

    # ── 3-panel plot: DI, E, deformed ──
    fig = plt.figure(figsize=(20, 6))

    for plot_idx, (data, cmap_name, label) in enumerate([
        (DI_per_el, 'RdYlGn_r', 'Dysbiosis Index'),
        (E_per_el, 'viridis', 'E [Pa]'),
        (None, 'hot_r', '|u|'),
    ]):
        ax = fig.add_subplot(1, 3, plot_idx + 1, projection='3d')

        all_polys = []
        all_colors = []

        if plot_idx < 2:
            coords = vertices
        else:
            coords = vertices + 200 * np.column_stack([ux, uy, uz])

        for el_id in range(len(cells)):
            for face in cell_faces[el_id]:
                face_int = face.astype(int)
                pts = coords[face_int]
                all_polys.append(pts)
                if plot_idx == 0:
                    all_colors.append(data[el_id])
                elif plot_idx == 1:
                    all_colors.append(data[el_id])
                else:
                    all_colors.append(np.mean(u_mag[face_int]))

        all_colors = np.array(all_colors)
        norm = plt.Normalize(all_colors.min(), all_colors.max())
        cmap = plt.get_cmap(cmap_name)

        pc = Poly3DCollection(all_polys, alpha=0.6, edgecolor='k',
                              linewidth=0.15)
        pc.set_facecolor(cmap(norm(all_colors)))
        ax.add_collection3d(pc)

        ax.set_xlim(coords[:, 0].min() - 0.05, coords[:, 0].max() + 0.05)
        ax.set_ylim(coords[:, 1].min() - 0.05, coords[:, 1].max() + 0.05)
        ax.set_zlim(coords[:, 2].min() - 0.05, coords[:, 2].max() + 0.05)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        fig.colorbar(sm, ax=ax, label=label, shrink=0.5)

    fig.suptitle('3D VEM + E(DI): Biofilm Mechanical Response',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = f'{save_dir}/vem_3d_biofilm.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close()

    return u


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    demo_3d_patch_test()
    demo_3d_compression()
    demo_3d_biofilm()
    print("\n" + "=" * 60)
    print("All 3D demos complete.")
    print("=" * 60)
