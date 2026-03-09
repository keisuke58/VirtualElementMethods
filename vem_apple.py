"""
VEM on an Apple-shaped 3D polyhedral mesh.

Generates an apple geometry, fills with Voronoi polyhedra,
and solves 3D elasticity under gravity loading.
"""

import numpy as np
from scipy.spatial import Voronoi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from vem_3d import isotropic_3d, traction_from_voigt, face_normal_area, polyhedron_volume
from vem_3d_advanced import (vem_3d_sparse, export_vtk, mesh_stats,
                             _merge_vertices, _order_face_vertices)


# ── Apple Geometry ─────────────────────────────────────────────────────────

def apple_radius(theta, phi):
    """
    Apple shape in spherical coordinates.
    theta: polar angle (0=top, pi=bottom)
    phi: azimuthal angle
    Returns radius at (theta, phi).
    """
    R = 1.0

    # Base sphere with slight vertical squash
    r = R * 0.9

    # Bulge in the middle (apple belly)
    r += 0.15 * np.sin(theta) ** 2

    # Dent at top (stem dimple)
    r -= 0.35 * np.exp(-((theta / 0.4) ** 2))

    # Slight dent at bottom
    r -= 0.08 * np.exp(-(((theta - np.pi) / 0.5) ** 2))

    return r


def point_inside_apple(xyz):
    """Check if point (x, y, z) is inside the apple."""
    x, y, z = xyz
    r_pt = np.sqrt(x**2 + y**2 + z**2)
    if r_pt < 1e-10:
        return True
    theta = np.arccos(np.clip(z / r_pt, -1, 1))
    phi = np.arctan2(y, x)
    r_apple = apple_radius(theta, phi)
    return r_pt < r_apple * 0.95  # slight margin


def generate_apple_seeds(n_seeds=80, seed=42):
    """Generate random seed points inside the apple shape."""
    rng = np.random.default_rng(seed)
    seeds = []

    # Rejection sampling
    attempts = 0
    while len(seeds) < n_seeds and attempts < n_seeds * 100:
        pt = rng.uniform(-1.2, 1.2, 3)
        if point_inside_apple(pt):
            seeds.append(pt)
        attempts += 1

    return np.array(seeds)


def apple_surface_points(n_theta=30, n_phi=40):
    """Generate surface points for visualization."""
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    T, P = np.meshgrid(theta, phi)

    R = apple_radius(T, P)
    X = R * np.sin(T) * np.cos(P)
    Y = R * np.sin(T) * np.sin(P)
    Z = R * np.cos(T)
    return X, Y, Z


# ── Voronoi Mesh in Apple ──────────────────────────────────────────────────

def make_apple_mesh(n_seeds=60, seed=42):
    """
    Generate Voronoi polyhedral mesh inside apple shape.
    Uses mirror points for boundary treatment.
    """
    print("  Generating apple seeds...")
    seeds = generate_apple_seeds(n_seeds, seed)
    n_actual = len(seeds)
    print(f"  Seeds inside apple: {n_actual}")

    # Mirror points across apple surface (reflect radially)
    mirror_pts = []
    for pt in seeds:
        r_pt = np.linalg.norm(pt)
        if r_pt < 1e-10:
            continue
        theta = np.arccos(np.clip(pt[2] / r_pt, -1, 1))
        phi = np.arctan2(pt[1], pt[0])
        r_apple = apple_radius(theta, phi)
        # Reflect: place mirror point at 2*r_apple - r_pt along same direction
        r_mirror = 2 * r_apple - r_pt
        direction = pt / r_pt
        mirror_pts.append(r_mirror * direction)

    mirror_pts = np.array(mirror_pts)
    all_pts = np.vstack([seeds, mirror_pts])

    print(f"  Total Voronoi seeds (with mirrors): {len(all_pts)}")
    vor = Voronoi(all_pts)
    raw_verts = vor.vertices.copy()

    # Build face list for original seed cells only
    seed_faces = {i: [] for i in range(n_actual)}
    for ridge_idx, (p1, p2) in enumerate(vor.ridge_points):
        fv = vor.ridge_vertices[ridge_idx]
        if -1 in fv:
            continue
        fv_arr = np.array(fv)
        if p1 < n_actual:
            seed_faces[p1].append(fv_arr)
        if p2 < n_actual:
            seed_faces[p2].append(fv_arr)

    # Merge duplicate vertices
    unique_verts, vert_remap = _merge_vertices(raw_verts, tol=1e-8)

    cells = []
    cell_faces = []

    for i in range(n_actual):
        faces_raw = seed_faces[i]
        if len(faces_raw) < 4:
            continue

        faces = []
        cell_vert_set = set()
        for fv in faces_raw:
            remapped = np.array([vert_remap[v] for v in fv])
            _, idx = np.unique(remapped, return_index=True)
            remapped = remapped[np.sort(idx)]
            if len(remapped) >= 3:
                faces.append(remapped)
                cell_vert_set.update(remapped)

        if len(faces) < 4 or len(cell_vert_set) < 4:
            continue

        cell_verts = np.array(sorted(cell_vert_set))
        cell_center = unique_verts[cell_verts].mean(axis=0)

        # Check if cell center is inside apple
        if not point_inside_apple(cell_center):
            continue

        ordered_faces = []
        for fv in faces:
            ordered = _order_face_vertices(unique_verts, fv, cell_center)
            if ordered is not None:
                ordered_faces.append(ordered)

        if len(ordered_faces) >= 4:
            cells.append(cell_verts)
            cell_faces.append(ordered_faces)

    print(f"  Valid cells inside apple: {len(cells)}")
    return unique_verts, cells, cell_faces


# ── Apple VEM Demo ─────────────────────────────────────────────────────────

def demo_apple(save_dir='/tmp'):
    """
    Solve 3D elasticity on apple-shaped Voronoi mesh.
    - Soft core (like real apple flesh), stiffer skin
    - Gravity loading (apple sitting on a table)
    """
    print("=" * 60)
    print("Apple VEM: 3D Elasticity on Apple-shaped Polyhedra")
    print("=" * 60)

    vertices, cells, cell_faces = make_apple_mesh(n_seeds=80, seed=42)
    mesh_stats(vertices, cells, cell_faces)

    if len(cells) == 0:
        print("  ERROR: No valid cells generated!")
        return None

    # Material: stiffer near surface (skin), softer inside (flesh)
    # Apple flesh: E ~ 0.3-1.0 MPa, skin: E ~ 5-10 MPa
    E_skin = 8.0   # MPa (normalized)
    E_flesh = 0.5   # MPa
    nu = 0.35       # nearly incompressible fruit

    E_per_el = np.zeros(len(cells))
    dist_to_surface = np.zeros(len(cells))

    for i, cell in enumerate(cells):
        cell_int = cell.astype(int)
        el_c = vertices[cell_int].mean(axis=0)
        r_pt = np.linalg.norm(el_c)
        if r_pt > 1e-10:
            theta = np.arccos(np.clip(el_c[2] / r_pt, -1, 1))
            phi = np.arctan2(el_c[1], el_c[0])
            r_apple = apple_radius(theta, phi)
            rel_depth = r_pt / r_apple  # 0=center, 1=surface
        else:
            rel_depth = 0.0

        dist_to_surface[i] = rel_depth
        # Smooth transition: skin layer at rel_depth > 0.7
        skin_factor = np.clip((rel_depth - 0.6) / 0.3, 0, 1)
        E_per_el[i] = E_flesh + (E_skin - E_flesh) * skin_factor

    print(f"  E range: [{E_per_el.min():.2f}, {E_per_el.max():.2f}] MPa")

    # Boundary conditions: fix bottom nodes (apple sitting on table)
    used_nodes = set()
    for cell in cells:
        used_nodes.update(cell.astype(int).tolist())
    used_nodes = np.array(sorted(used_nodes))

    z_vals = vertices[used_nodes, 2]
    z_min = z_vals.min()
    z_range = z_vals.max() - z_min

    # Fix bottom 5% of nodes
    bottom = used_nodes[z_vals < z_min + 0.08 * z_range]
    bc_dofs = np.concatenate([3*bottom, 3*bottom+1, 3*bottom+2])
    bc_vals = np.zeros(len(bc_dofs))

    # Gravity load on all nodes (downward)
    gravity = -0.01  # normalized gravity
    load_dofs = 3 * used_nodes + 2  # z-DOF
    load_vals = np.full(len(used_nodes), gravity / len(used_nodes))

    print(f"  Fixed (bottom): {len(bottom)}")
    print(f"  Loaded (gravity): {len(used_nodes)} nodes")

    u = vem_3d_sparse(vertices, cells, cell_faces, E_per_el, nu,
                      bc_dofs, bc_vals, load_dofs, load_vals)

    ux, uy, uz = u[0::3], u[1::3], u[2::3]
    u_mag = np.sqrt(ux**2 + uy**2 + uz**2)
    print(f"  Max |u|: {np.max(u_mag[used_nodes]):.6f}")

    # ── Export VTK ──
    disp_vec = np.column_stack([ux, uy, uz])
    export_vtk(
        f'{save_dir}/vem_apple.vtk',
        vertices, cells, cell_faces,
        point_data={'displacement': disp_vec, 'u_magnitude': u_mag},
        cell_data={'E_modulus': E_per_el, 'depth': dist_to_surface}
    )

    # ── Visualization ──
    fig = plt.figure(figsize=(20, 6))

    deform_scale = 50.0
    deformed = vertices + deform_scale * disp_vec

    titles = ['Apple Stiffness (skin vs flesh)',
              'Undeformed Apple',
              f'Deformed (x{deform_scale:.0f})']
    data_sources = [
        (vertices, E_per_el, 'YlOrRd', 'E [MPa]'),
        (vertices, None, 'Greens', '|u|'),
        (deformed, None, 'hot_r', '|u|'),
    ]

    for plot_idx, (coords, cell_data, cmap_name, clabel) in enumerate(data_sources):
        ax = fig.add_subplot(1, 3, plot_idx + 1, projection='3d')

        all_polys = []
        all_colors = []

        for el_id in range(len(cells)):
            for face in cell_faces[el_id]:
                fi = face.astype(int)
                pts = coords[fi]
                if np.any(np.isnan(pts)):
                    continue
                all_polys.append(pts)
                if cell_data is not None:
                    all_colors.append(cell_data[el_id])
                else:
                    all_colors.append(np.mean(u_mag[fi]))

        if not all_polys:
            continue

        all_colors = np.array(all_colors)
        norm = plt.Normalize(all_colors.min(), all_colors.max() + 1e-15)
        cmap = plt.get_cmap(cmap_name)

        pc = Poly3DCollection(all_polys, alpha=0.7, edgecolor='k',
                              linewidth=0.15)
        pc.set_facecolor(cmap(norm(all_colors)))
        ax.add_collection3d(pc)

        # Also draw apple wireframe for reference
        if plot_idx == 1:
            X, Y, Z = apple_surface_points(15, 20)
            ax.plot_wireframe(X, Y, Z, color='green', alpha=0.08,
                              linewidth=0.3)

        pad = 0.2
        ax.set_xlim(coords[list(used_nodes), 0].min() - pad,
                     coords[list(used_nodes), 0].max() + pad)
        ax.set_ylim(coords[list(used_nodes), 1].min() - pad,
                     coords[list(used_nodes), 1].max() + pad)
        ax.set_zlim(coords[list(used_nodes), 2].min() - pad,
                     coords[list(used_nodes), 2].max() + pad)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(titles[plot_idx])

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        fig.colorbar(sm, ax=ax, label=clabel, shrink=0.5)

    fig.suptitle('VEM on Apple: 3D Elasticity with Skin/Flesh Stiffness',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = f'{save_dir}/vem_apple.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close()

    # ── Cross-section plot (slice at y≈0) ──
    fig2, ax2 = plt.subplots(figsize=(8, 8))

    # Draw apple outline
    theta_line = np.linspace(0, np.pi, 200)
    r_line = np.array([apple_radius(t, 0) for t in theta_line])
    x_outline = r_line * np.sin(theta_line)
    z_outline = r_line * np.cos(theta_line)
    ax2.plot(x_outline, z_outline, 'g-', linewidth=2, label='Apple surface')
    ax2.plot(-x_outline, z_outline, 'g-', linewidth=2)

    # Plot cell centroids colored by E, only near y=0
    for i, cell in enumerate(cells):
        cell_int = cell.astype(int)
        el_c = vertices[cell_int].mean(axis=0)
        if abs(el_c[1]) < 0.3:  # near y=0 slice
            color = plt.cm.YlOrRd(E_per_el[i] / E_per_el.max())
            # Draw cell polygon (project to xz plane)
            for face in cell_faces[i]:
                fi = face.astype(int)
                face_c = vertices[fi].mean(axis=0)
                if abs(face_c[1]) < 0.4:
                    xs = vertices[fi, 0]
                    zs = vertices[fi, 2]
                    ax2.fill(xs, zs, color=color, alpha=0.5,
                             edgecolor='k', linewidth=0.3)

    sm = plt.cm.ScalarMappable(cmap='YlOrRd',
                                norm=plt.Normalize(E_per_el.min(), E_per_el.max()))
    fig2.colorbar(sm, ax=ax2, label='E [MPa]')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')
    ax2.set_title('Apple Cross-Section (y=0): Skin vs Flesh Stiffness')
    ax2.set_aspect('equal')
    ax2.legend()
    path2 = f'{save_dir}/vem_apple_cross_section.png'
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path2}")
    plt.close()

    return u


if __name__ == '__main__':
    demo_apple()
    print("\nApple VEM complete!")
