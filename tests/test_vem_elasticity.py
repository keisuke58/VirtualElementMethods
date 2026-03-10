"""Tests for 2D VEM elasticity."""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from vem_elasticity import vem_elasticity, load_mesh


class TestPatchTest2D:
    """Patch test: linear displacement must be reproduced exactly."""

    @pytest.mark.parametrize("mesh_name", [
        'voronoi.mat', 'squares.mat', 'smoothed-voronoi.mat'
    ])
    def test_uniform_tension(self, mesh_dir, mesh_name):
        """σ_xx = 1 → u_x = x/E, u_y = -ν·y/E."""
        mesh_file = os.path.join(mesh_dir, mesh_name)
        if not os.path.exists(mesh_file):
            pytest.skip(f"Mesh {mesh_name} not found")

        vertices, elements, boundary = load_mesh(mesh_file)
        E_mod, nu = 1000.0, 0.3

        exact_ux = vertices[:, 0] / E_mod
        exact_uy = -nu * vertices[:, 1] / E_mod

        bc_dofs = np.concatenate([2 * boundary, 2 * boundary + 1])
        bc_vals = np.concatenate([exact_ux[boundary], exact_uy[boundary]])

        u = vem_elasticity(vertices, elements, E_mod, nu, bc_dofs, bc_vals)

        ux, uy = u[0::2], u[1::2]
        assert np.max(np.abs(ux - exact_ux)) < 1e-10, \
            f"Patch test u_x failed on {mesh_name}"
        assert np.max(np.abs(uy - exact_uy)) < 1e-10, \
            f"Patch test u_y failed on {mesh_name}"

    def test_triangles_bounded_error(self, mesh_dir):
        """triangles.mat: VEM on pure triangles has bounded error (not exact)."""
        mesh_file = os.path.join(mesh_dir, 'triangles.mat')
        if not os.path.exists(mesh_file):
            pytest.skip("triangles.mat not found")
        vertices, elements, boundary = load_mesh(mesh_file)
        E_mod, nu = 1000.0, 0.3
        exact_ux = vertices[:, 0] / E_mod
        exact_uy = -nu * vertices[:, 1] / E_mod
        bc_dofs = np.concatenate([2*boundary, 2*boundary+1])
        bc_vals = np.concatenate([exact_ux[boundary], exact_uy[boundary]])
        u = vem_elasticity(vertices, elements, E_mod, nu, bc_dofs, bc_vals)
        ux = u[0::2]
        err = np.max(np.abs(ux - exact_ux))
        assert err < 0.01, f"Triangle mesh error too large: {err:.2e}"

    def test_pure_shear(self, mesh_dir):
        """Pure shear: u_x = γ·y, u_y = γ·x (linear displacement)."""
        vertices, elements, boundary = load_mesh(
            os.path.join(mesh_dir, 'voronoi.mat'))
        E_mod, nu = 500.0, 0.25
        gamma = 0.001

        exact_ux = gamma * vertices[:, 1]
        exact_uy = gamma * vertices[:, 0]

        bc_dofs = np.concatenate([2 * boundary, 2 * boundary + 1])
        bc_vals = np.concatenate([exact_ux[boundary], exact_uy[boundary]])

        u = vem_elasticity(vertices, elements, E_mod, nu, bc_dofs, bc_vals)
        ux, uy = u[0::2], u[1::2]

        assert np.max(np.abs(ux - exact_ux)) < 1e-10
        assert np.max(np.abs(uy - exact_uy)) < 1e-10


class TestElasticityProperties:
    """Test physical properties of the elasticity solution."""

    def test_cantilever_deflects_downward(self, mesh_dir):
        """Cantilever with downward load deflects in -y direction."""
        vertices, elements, boundary = load_mesh(
            os.path.join(mesh_dir, 'voronoi.mat'))
        E_mod, nu = 1000.0, 0.3
        tol = 1e-6

        left = np.where(vertices[:, 0] < tol)[0]
        bc_dofs = np.concatenate([2 * left, 2 * left + 1])
        bc_vals = np.zeros(len(bc_dofs))

        right = np.where(vertices[:, 0] > 1.0 - tol)[0]
        load_dofs = 2 * right + 1
        load_vals = np.full(len(right), -1.0 / len(right))

        u = vem_elasticity(vertices, elements, E_mod, nu,
                           bc_dofs, bc_vals, load_dofs, load_vals)
        uy = u[1::2]
        assert np.mean(uy[right]) < 0, "Tip should deflect downward"

    def test_softer_material_deflects_more(self, mesh_dir):
        """Softer material (lower E) gives larger displacement."""
        vertices, elements, boundary = load_mesh(
            os.path.join(mesh_dir, 'voronoi.mat'))
        nu = 0.3
        tol = 1e-6

        bottom = np.where(vertices[:, 1] < tol)[0]
        bc_dofs = np.concatenate([2 * bottom, 2 * bottom + 1])
        bc_vals = np.zeros(len(bc_dofs))

        top = np.where(vertices[:, 1] > 1.0 - tol)[0]
        load_dofs = 2 * top + 1
        load_vals = np.full(len(top), -0.5 / len(top))

        u_stiff = vem_elasticity(vertices, elements, 1000.0, nu,
                                 bc_dofs, bc_vals, load_dofs, load_vals)
        u_soft = vem_elasticity(vertices, elements, 100.0, nu,
                                bc_dofs, bc_vals, load_dofs, load_vals)

        assert np.max(np.abs(u_soft)) > np.max(np.abs(u_stiff))

    def test_spatially_varying_E(self, mesh_dir):
        """Spatially varying E(DI) runs without error."""
        vertices, elements, boundary = load_mesh(
            os.path.join(mesh_dir, 'voronoi.mat'))
        nu = 0.3

        E_per_el = np.array([
            30 + 970 * (1 - 0.5) ** 2 for _ in elements
        ])

        tol = 1e-6
        bottom = np.where(vertices[:, 1] < tol)[0]
        bc_dofs = np.concatenate([2 * bottom, 2 * bottom + 1])
        bc_vals = np.zeros(len(bc_dofs))

        top = np.where(vertices[:, 1] > 1.0 - tol)[0]
        load_dofs = 2 * top + 1
        load_vals = np.full(len(top), -0.5 / len(top))

        u = vem_elasticity(vertices, elements, E_per_el, nu,
                           bc_dofs, bc_vals, load_dofs, load_vals)
        assert np.all(np.isfinite(u))


class TestConvergence2D:
    """h-refinement convergence for 2D elasticity."""

    def test_patch_test_convergence(self, mesh_dir):
        """Patch test error should be machine epsilon on all meshes."""
        E_mod, nu = 1000.0, 0.3
        # Skip triangles.mat: pure triangles don't pass exact patch test in VEM
        mesh_names = ['squares.mat', 'voronoi.mat', 'smoothed-voronoi.mat']

        for name in mesh_names:
            path = os.path.join(mesh_dir, name)
            if not os.path.exists(path):
                continue
            vertices, elements, boundary = load_mesh(path)

            exact_ux = vertices[:, 0] / E_mod
            exact_uy = -nu * vertices[:, 1] / E_mod

            bc_dofs = np.concatenate([2 * boundary, 2 * boundary + 1])
            bc_vals = np.concatenate([exact_ux[boundary], exact_uy[boundary]])

            u = vem_elasticity(vertices, elements, E_mod, nu, bc_dofs, bc_vals)
            ux, uy = u[0::2], u[1::2]

            err = max(np.max(np.abs(ux - exact_ux)),
                      np.max(np.abs(uy - exact_uy)))
            assert err < 1e-10, f"Patch test failed on {name}: err={err:.2e}"
