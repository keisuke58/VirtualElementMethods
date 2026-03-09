"""
Advanced 3D VEM: Voronoi polyhedral mesh + VTK export + convergence study.

Improvements over vem_3d.py:
  1. True 3D Voronoi mesh (arbitrary polyhedra, not just hex)
  2. VTK export for ParaView visualization
  3. h-refinement convergence study
  4. Sparse assembly for larger meshes
"""

import numpy as np
from scipy.spatial import Voronoi
from scipy import sparse
from scipy.sparse.linalg import spsolve
import struct
import os

from vem_3d import (isotropic_3d, traction_from_voigt, face_normal_area,
                    polyhedron_volume)


# ── 3D Voronoi Mesh Generator ─────────────────────────────────────────────

def make_voronoi_mesh_3d(n_seeds=30, seed=42):
    """
    Generate 3D Voronoi polyhedral mesh in [0,1]^3.

    Uses mirror points across all 6 faces for clean boundary treatment.
    Returns vertices, cells, cell_faces with properly ordered face vertices.
    """
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0.12, 0.88, (n_seeds, 3))

    # Mirror across 6 faces
    all_pts = [pts]
    for axis in range(3):
        for val in [0.0, 1.0]:
            mirror = pts.copy()
            mirror[:, axis] = 2 * val - mirror[:, axis]
            all_pts.append(mirror)
    all_pts = np.vstack(all_pts)

    vor = Voronoi(all_pts)
    raw_verts = vor.vertices.copy()

    # Build face list per original seed cell
    # ridge_points[i] = (p1, p2), ridge_vertices[i] = face vertex list
    seed_faces = {i: [] for i in range(n_seeds)}
    for ridge_idx, (p1, p2) in enumerate(vor.ridge_points):
        fv = vor.ridge_vertices[ridge_idx]
        if -1 in fv:
            continue
        if p1 < n_seeds:
            seed_faces[p1].append(np.array(fv))
        if p2 < n_seeds:
            seed_faces[p2].append(np.array(fv))

    # Clip vertices to [0, 1]^3 with small epsilon to avoid degeneracy
    raw_verts = np.clip(raw_verts, -0.001, 1.001)

    # Merge duplicate vertices (from clipping)
    unique_verts, vert_remap = _merge_vertices(raw_verts, tol=1e-8)

    cells = []
    cell_faces = []

    for i in range(n_seeds):
        faces_raw = seed_faces[i]
        if len(faces_raw) < 4:
            continue

        # Remap vertex indices and remove degenerate faces
        faces = []
        cell_vert_set = set()
        for fv in faces_raw:
            remapped = np.array([vert_remap[v] for v in fv])
            # Remove duplicate vertices in face
            _, idx = np.unique(remapped, return_index=True)
            remapped = remapped[np.sort(idx)]
            if len(remapped) >= 3:
                faces.append(remapped)
                cell_vert_set.update(remapped)

        if len(faces) < 4 or len(cell_vert_set) < 4:
            continue

        cell_verts = np.array(sorted(cell_vert_set))

        # Order face vertices for consistent outward normals
        cell_center = unique_verts[cell_verts].mean(axis=0)
        ordered_faces = []
        for fv in faces:
            ordered = _order_face_vertices(unique_verts, fv, cell_center)
            if ordered is not None:
                ordered_faces.append(ordered)

        if len(ordered_faces) >= 4:
            cells.append(cell_verts)
            cell_faces.append(ordered_faces)

    return unique_verts, cells, cell_faces


def _merge_vertices(verts, tol=1e-10):
    """Merge vertices closer than tol. Return unique verts and remap."""
    n = len(verts)
    remap = np.arange(n)
    unique = list(range(n))

    # Simple O(n^2) merge — fine for < 10k vertices
    for i in range(n):
        if remap[i] != i:
            continue
        for j in range(i + 1, n):
            if remap[j] != j:
                continue
            if np.linalg.norm(verts[i] - verts[j]) < tol:
                remap[j] = i

    # Compact
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


def _order_face_vertices(vertices, face_verts, cell_center):
    """Order polygon vertices CCW when viewed from outside the cell."""
    pts = vertices[face_verts]
    if len(pts) < 3:
        return None

    fc = pts.mean(axis=0)

    # Face normal from first triangle
    v1 = pts[1] - pts[0]
    v2 = pts[2] - pts[0]
    normal = np.cross(v1, v2)
    nlen = np.linalg.norm(normal)
    if nlen < 1e-15:
        return None
    normal /= nlen

    # Orient outward
    if np.dot(normal, fc - cell_center) < 0:
        normal = -normal

    # Project onto face plane and sort by angle
    d = pts - fc
    u_ax = d[0].copy()
    u_ax -= np.dot(u_ax, normal) * normal
    u_len = np.linalg.norm(u_ax)
    if u_len < 1e-15:
        # Try next vertex
        for k in range(1, len(pts)):
            u_ax = d[k].copy()
            u_ax -= np.dot(u_ax, normal) * normal
            u_len = np.linalg.norm(u_ax)
            if u_len > 1e-15:
                break
    if u_len < 1e-15:
        return None
    u_ax /= u_len
    v_ax = np.cross(normal, u_ax)

    angles = np.arctan2(d @ v_ax, d @ u_ax)
    order = np.argsort(angles)
    return face_verts[order]


# ── Mesh Statistics ───────────────────────────────────────────────────────

def mesh_stats(vertices, cells, cell_faces):
    """Print mesh statistics."""
    n_verts = len(vertices)
    n_cells = len(cells)
    faces_per_cell = [len(f) for f in cell_faces]
    verts_per_cell = [len(c) for c in cells]
    verts_per_face = [len(f) for faces in cell_faces for f in faces]

    print(f"  Vertices:       {n_verts}")
    print(f"  Cells:          {n_cells}")
    print(f"  Verts/cell:     {np.min(verts_per_cell)}-{np.max(verts_per_cell)} "
          f"(avg {np.mean(verts_per_cell):.1f})")
    print(f"  Faces/cell:     {np.min(faces_per_cell)}-{np.max(faces_per_cell)} "
          f"(avg {np.mean(faces_per_cell):.1f})")
    print(f"  Verts/face:     {np.min(verts_per_face)}-{np.max(verts_per_face)} "
          f"(avg {np.mean(verts_per_face):.1f})")


# ── VEM 3D Solver (Sparse) ────────────────────────────────────────────────

def vem_3d_sparse(vertices, cells, cell_faces, E_field, nu,
                  bc_fixed_dofs, bc_vals,
                  load_dofs=None, load_vals=None):
    """
    3D VEM with sparse assembly. Same algorithm as vem_3d.vem_3d_elasticity
    but uses COO sparse format for the global stiffness matrix.
    """
    n_nodes = len(vertices)
    n_dofs = 3 * n_nodes
    n_polys = 12

    # COO format accumulators
    rows_list = []
    cols_list = []
    vals_list = []
    F_global = np.zeros(n_dofs)

    strain_ids = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 2, 0, 0],
        [0, 0, 0, 0, 2, 0],
        [0, 0, 0, 0, 0, 2],
    ], dtype=float)

    for el_id in range(len(cells)):
        vert_ids = cells[el_id].astype(int)
        coords = vertices[vert_ids]
        faces = cell_faces[el_id]
        n_v = len(vert_ids)
        n_el = 3 * n_v

        vmap = {int(g): loc for loc, g in enumerate(vert_ids)}

        E_el = E_field[el_id] if hasattr(E_field, '__len__') else E_field
        C = isotropic_3d(E_el, nu)

        centroid = coords.mean(axis=0)
        h = max(np.linalg.norm(coords[i] - coords[j])
                for i in range(n_v) for j in range(i + 1, n_v))
        vol = polyhedron_volume(vertices, faces)
        if vol < 1e-20:
            continue

        xc, yc, zc = centroid

        # D matrix
        D = np.zeros((n_el, n_polys))
        for i in range(n_v):
            dx = (coords[i, 0] - xc) / h
            dy = (coords[i, 1] - yc) / h
            dz = (coords[i, 2] - zc) / h
            r = 3 * i
            D[r,   :] = [1, 0, 0, 0,   dz, -dy, dx, 0,  0,  0,  dz, dy]
            D[r+1, :] = [0, 1, 0, -dz, 0,  dx,  0,  dy, 0,  dz, 0,  dx]
            D[r+2, :] = [0, 0, 1, dy,  -dx, 0,  0,  0,  dz, dy, dx, 0]

        # B matrix
        B = np.zeros((n_polys, n_el))
        for i in range(n_v):
            B[0, 3*i]     = 1.0 / n_v
            B[1, 3*i + 1] = 1.0 / n_v
            B[2, 3*i + 2] = 1.0 / n_v

        for face in faces:
            face_int = face.astype(int)
            pts = vertices[face_int]
            n_f, A_f = face_normal_area(pts)
            fc = pts.mean(axis=0)
            if np.dot(n_f, fc - centroid) < 0:
                n_f = -n_f
            k_f = len(face_int)

            for gv in face_int:
                if gv not in vmap:
                    continue
                li = vmap[gv]
                w = A_f / k_f

                wrot = w / (2.0 * vol)
                B[3, 3*li + 1] += -wrot * n_f[2]
                B[3, 3*li + 2] +=  wrot * n_f[1]
                B[4, 3*li + 0] +=  wrot * n_f[2]
                B[4, 3*li + 2] += -wrot * n_f[0]
                B[5, 3*li + 0] += -wrot * n_f[1]
                B[5, 3*li + 1] +=  wrot * n_f[0]

                for alpha in range(6):
                    eps_a = strain_ids[alpha] / h
                    sigma_a = C @ eps_a
                    t_f = traction_from_voigt(sigma_a, n_f)
                    B[6 + alpha, 3*li + 0] += w * t_f[0]
                    B[6 + alpha, 3*li + 1] += w * t_f[1]
                    B[6 + alpha, 3*li + 2] += w * t_f[2]

        # Projector
        G = B @ D
        try:
            projector = np.linalg.solve(G, B)
        except np.linalg.LinAlgError:
            continue

        G_tilde = G.copy()
        G_tilde[:6, :] = 0.0
        K_cons = projector.T @ G_tilde @ projector

        I_minus_PiD = np.eye(n_el) - D @ projector
        trace_k = np.trace(K_cons)
        stab_param = trace_k / n_el if trace_k > 0 else E_el
        K_stab = stab_param * (I_minus_PiD.T @ I_minus_PiD)

        K_local = K_cons + K_stab

        # COO assembly
        gdofs = np.zeros(n_el, dtype=int)
        for i in range(n_v):
            gdofs[3*i]     = 3 * vert_ids[i]
            gdofs[3*i + 1] = 3 * vert_ids[i] + 1
            gdofs[3*i + 2] = 3 * vert_ids[i] + 2

        gi, gj = np.meshgrid(gdofs, gdofs, indexing='ij')
        rows_list.append(gi.ravel())
        cols_list.append(gj.ravel())
        vals_list.append(K_local.ravel())

    # Assemble sparse
    rows = np.concatenate(rows_list)
    cols = np.concatenate(cols_list)
    vals = np.concatenate(vals_list)
    K_global = sparse.coo_matrix((vals, (rows, cols)),
                                 shape=(n_dofs, n_dofs)).tocsr()

    if load_dofs is not None and load_vals is not None:
        F_global[load_dofs] += load_vals

    # Solve
    u = np.zeros(n_dofs)
    bc_set = set(bc_fixed_dofs.tolist())
    internal = np.array([i for i in range(n_dofs) if i not in bc_set])

    u[bc_fixed_dofs] = bc_vals
    F_global -= K_global[:, bc_fixed_dofs].toarray() @ bc_vals

    K_ii = K_global[np.ix_(internal, internal)]

    # Add small regularization if needed
    try:
        u[internal] = spsolve(K_ii, F_global[internal])
        if np.any(np.isnan(u)):
            raise ValueError("NaN in solution")
    except Exception:
        # Regularize
        reg = 1e-10 * sparse.eye(K_ii.shape[0])
        u[internal] = spsolve(K_ii + reg, F_global[internal])

    return u


# ── VTK Export ─────────────────────────────────────────────────────────────

def export_vtk(filename, vertices, cells, cell_faces,
               point_data=None, cell_data=None):
    """
    Export to VTK unstructured grid (legacy ASCII format).
    Supports arbitrary polyhedra (VTK_POLYHEDRON = 42).
    """
    n_nodes = len(vertices)
    n_cells = len(cells)

    with open(filename, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("VEM 3D Result\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n\n")

        # Points
        f.write(f"POINTS {n_nodes} double\n")
        for v in vertices:
            f.write(f"{v[0]:.10f} {v[1]:.10f} {v[2]:.10f}\n")
        f.write("\n")

        # Cells — use VTK_POLYHEDRON (type 42) format
        # Each cell: nFaces, (nPtsInFace, pt0, pt1, ...), ...
        cell_entries = []
        total_size = 0
        for el_id in range(n_cells):
            faces = cell_faces[el_id]
            entry = [len(faces)]
            for face in faces:
                face_int = face.astype(int).tolist()
                entry.append(len(face_int))
                entry.extend(face_int)
            cell_entries.append(entry)
            total_size += len(entry)

        f.write(f"CELLS {n_cells} {total_size + n_cells}\n")
        for entry in cell_entries:
            f.write(f"{len(entry)} " + " ".join(str(x) for x in entry) + "\n")
        f.write("\n")

        f.write(f"CELL_TYPES {n_cells}\n")
        for _ in range(n_cells):
            f.write("42\n")  # VTK_POLYHEDRON
        f.write("\n")

        # Point data
        if point_data:
            f.write(f"POINT_DATA {n_nodes}\n")
            for name, data in point_data.items():
                if data.ndim == 1:
                    f.write(f"SCALARS {name} double 1\n")
                    f.write("LOOKUP_TABLE default\n")
                    for val in data:
                        f.write(f"{val:.10e}\n")
                else:
                    f.write(f"VECTORS {name} double\n")
                    for row in data:
                        f.write(f"{row[0]:.10e} {row[1]:.10e} {row[2]:.10e}\n")
                f.write("\n")

        # Cell data
        if cell_data:
            f.write(f"CELL_DATA {n_cells}\n")
            for name, data in cell_data.items():
                f.write(f"SCALARS {name} double 1\n")
                f.write("LOOKUP_TABLE default\n")
                for val in data:
                    f.write(f"{val:.10e}\n")
                f.write("\n")

    print(f"  VTK saved: {filename}")


# ── Convergence Study ──────────────────────────────────────────────────────

def convergence_study(save_dir='/tmp'):
    """
    h-refinement convergence on perturbed hex meshes.
    Perturbation scales with h to ensure proper convergence.
    """
    from vem_3d import make_hex_mesh

    print("=" * 60)
    print("Convergence Study (h-refinement, patch test)")
    print("=" * 60)

    ns = [2, 3, 4, 6, 8, 10]
    hs = []
    errors = []

    E_mod = 1000.0
    nu = 0.3

    for n in ns:
        # Perturbation proportional to h² for clean convergence
        perturb = 0.3 * (1.0 / n)
        vertices, cells, cell_faces = make_hex_mesh(
            nx=n, ny=n, nz=n, perturb=perturb, seed=42)
        h = 1.0 / n
        hs.append(h)

        # Exact: uniform tension σ_xx = 1
        exact_ux = vertices[:, 0] / E_mod
        exact_uy = -nu * vertices[:, 1] / E_mod
        exact_uz = -nu * vertices[:, 2] / E_mod

        tol = 1e-6
        boundary = np.where(
            (vertices[:, 0] < tol) | (vertices[:, 0] > 1 - tol) |
            (vertices[:, 1] < tol) | (vertices[:, 1] > 1 - tol) |
            (vertices[:, 2] < tol) | (vertices[:, 2] > 1 - tol)
        )[0]

        bc_dofs = np.concatenate([3*boundary, 3*boundary+1, 3*boundary+2])
        bc_vals = np.concatenate([exact_ux[boundary], exact_uy[boundary],
                                  exact_uz[boundary]])

        u = vem_3d_sparse(vertices, cells, cell_faces, E_mod, nu,
                          bc_dofs, bc_vals)

        ux, uy, uz = u[0::3], u[1::3], u[2::3]
        err = max(np.max(np.abs(ux - exact_ux)),
                  np.max(np.abs(uy - exact_uy)),
                  np.max(np.abs(uz - exact_uz)))
        errors.append(err)
        print(f"  n={n:2d}, h={h:.4f}, cells={len(cells):4d}, "
              f"nodes={len(vertices):5d}, error={err:.2e}")

    # Convergence rate
    hs = np.array(hs)
    errors = np.array(errors)
    # Skip first point (n=2 is too coarse, often exact)
    valid = errors > 1e-15
    if np.sum(valid) >= 2:
        h_v = hs[valid]
        e_v = errors[valid]
        rates = np.log(e_v[:-1] / e_v[1:]) / np.log(h_v[:-1] / h_v[1:])
        print(f"\n  Convergence rates: {', '.join(f'{r:.2f}' for r in rates)}")
        print(f"  Average rate: {np.mean(rates):.2f} (expected ~1-2 for k=1 VEM)")
    else:
        rates = np.array([])
        print("\n  Not enough data points for rate computation")

    # Plot
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.loglog(hs, errors, 'bo-', linewidth=2, markersize=8, label='VEM error')
    if np.any(valid):
        ref_h = hs[valid]
        ref_e = errors[valid]
        ax.loglog(ref_h, ref_e[0] * (ref_h / ref_h[0])**1, 'r--', alpha=0.5,
                  label='O($h$) reference')
        ax.loglog(ref_h, ref_e[0] * (ref_h / ref_h[0])**2, 'k--', alpha=0.5,
                  label='O($h^2$) reference')
    ax.set_xlabel('Element size $h$')
    ax.set_ylabel('Max displacement error')
    ax.set_title('3D VEM Convergence (perturbed hex, patch test)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = f'{save_dir}/vem_3d_convergence.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close()

    return hs, errors, rates


# ── Demo: Voronoi Mesh + Biofilm ───────────────────────────────────────────

def demo_voronoi_biofilm(save_dir='/tmp'):
    """
    3D Voronoi polyhedral mesh with E(DI) spatial variation.
    Export to VTK for ParaView visualization.
    """
    print("\n" + "=" * 60)
    print("Demo: 3D Voronoi Biofilm E(DI)")
    print("=" * 60)

    vertices, cells, cell_faces = make_voronoi_mesh_3d(n_seeds=50, seed=42)
    mesh_stats(vertices, cells, cell_faces)

    # E(DI)
    E_max, E_min, n_hill, nu = 1000.0, 30.0, 2, 0.3
    center = np.array([0.5, 0.5, 0.5])
    max_dist = 0.5 * np.sqrt(3)

    E_per_el = np.zeros(len(cells))
    DI_per_el = np.zeros(len(cells))
    for i, cell in enumerate(cells):
        el_c = vertices[cell.astype(int)].mean(axis=0)
        dist = np.linalg.norm(el_c - center)
        DI = np.clip(0.9 - 0.8 * dist / max_dist, 0.05, 0.95)
        DI_per_el[i] = DI
        E_per_el[i] = E_min + (E_max - E_min) * (1.0 - DI) ** n_hill

    print(f"  DI range: [{DI_per_el.min():.2f}, {DI_per_el.max():.2f}]")
    print(f"  E  range: [{E_per_el.min():.0f}, {E_per_el.max():.0f}] Pa")

    # Identify boundary nodes (only nodes actually used by cells)
    used_nodes = set()
    for cell in cells:
        used_nodes.update(cell.astype(int).tolist())
    used_nodes = np.array(sorted(used_nodes))

    z_min = vertices[used_nodes, 2].min()
    z_max = vertices[used_nodes, 2].max()
    z_range = z_max - z_min
    tol_bot = z_min + 0.05 * z_range
    tol_top = z_max - 0.05 * z_range

    bottom = used_nodes[vertices[used_nodes, 2] < tol_bot]
    top = used_nodes[vertices[used_nodes, 2] > tol_top]

    bc_dofs = np.concatenate([3*bottom, 3*bottom+1, 3*bottom+2])
    bc_vals = np.zeros(len(bc_dofs))

    load_per_node = -1.0 / max(len(top), 1)
    load_dofs = 3 * top + 2
    load_vals = np.full(len(top), load_per_node)

    print(f"  Fixed (bottom): {len(bottom)}, Loaded (top): {len(top)}")

    u = vem_3d_sparse(vertices, cells, cell_faces, E_per_el, nu,
                      bc_dofs, bc_vals, load_dofs, load_vals)

    ux, uy, uz = u[0::3], u[1::3], u[2::3]
    u_mag = np.sqrt(ux**2 + uy**2 + uz**2)
    print(f"  Max |u|: {np.max(u_mag):.6f}")

    # Export to VTK
    disp_vec = np.column_stack([ux, uy, uz])
    export_vtk(
        f'{save_dir}/vem_3d_voronoi_biofilm.vtk',
        vertices, cells, cell_faces,
        point_data={
            'displacement': disp_vec,
            'u_magnitude': u_mag,
        },
        cell_data={
            'DI': DI_per_el,
            'E_modulus': E_per_el,
        }
    )

    # Also export deformed
    deformed = vertices + 200 * disp_vec
    export_vtk(
        f'{save_dir}/vem_3d_voronoi_biofilm_deformed.vtk',
        deformed, cells, cell_faces,
        point_data={
            'displacement': disp_vec,
            'u_magnitude': u_mag,
        },
        cell_data={
            'DI': DI_per_el,
            'E_modulus': E_per_el,
        }
    )

    # matplotlib fallback plot
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(18, 6))

    for plot_idx, (data_per_el, cmap_name, label) in enumerate([
        (DI_per_el, 'RdYlGn_r', 'Dysbiosis Index'),
        (E_per_el, 'viridis', 'E [Pa]'),
        (None, 'hot_r', '|u|'),
    ]):
        ax = fig.add_subplot(1, 3, plot_idx + 1, projection='3d')
        all_polys = []
        all_colors = []

        coords = vertices if plot_idx < 2 else deformed

        for el_id in range(len(cells)):
            for face in cell_faces[el_id]:
                fi = face.astype(int)
                all_polys.append(coords[fi])
                if plot_idx < 2:
                    all_colors.append(data_per_el[el_id])
                else:
                    all_colors.append(np.mean(u_mag[fi]))

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
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        fig.colorbar(sm, ax=ax, label=label, shrink=0.5)

    fig.suptitle('3D VEM on Voronoi Polyhedra + E(DI)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = f'{save_dir}/vem_3d_voronoi_biofilm.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close()

    return u


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import time

    t0 = time.time()
    hs, errors, rates = convergence_study()

    t1 = time.time()
    demo_voronoi_biofilm()

    t2 = time.time()
    print(f"\n  Convergence study: {t1-t0:.1f}s")
    print(f"  Voronoi biofilm:   {t2-t1:.1f}s")
    print(f"  Total:             {t2-t0:.1f}s")

    print("\n" + "=" * 60)
    print("All advanced demos complete.")
    print("=" * 60)
