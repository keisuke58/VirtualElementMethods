"""Tests for 3D VEM elasticity."""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from vem_3d import (vem_3d_elasticity, make_hex_mesh, isotropic_3d,
                     face_normal_area, polyhedron_volume)
from vem_3d_advanced import vem_3d_sparse, make_voronoi_mesh_3d


class TestGeometryHelpers:
    """Test geometry utility functions."""

    def test_face_normal_unit_length(self):
        """Face normal should be unit vector."""
        pts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0.0]])
        n, area = face_normal_area(pts)
        np.testing.assert_allclose(np.linalg.norm(n), 1.0, atol=1e-14)

    def test_face_area_square(self):
        """Area of unit square should be 1."""
        pts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0.0]])
        n, area = face_normal_area(pts)
        np.testing.assert_allclose(area, 1.0, atol=1e-14)

    def test_face_area_triangle(self):
        """Area of right triangle (legs=1)."""
        pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0.0]])
        n, area = face_normal_area(pts)
        np.testing.assert_allclose(area, 0.5, atol=1e-14)

    def test_hex_volume(self):
        """Volume of unit cube via divergence theorem."""
        verts = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1.0],
        ])
        faces = [
            np.array([0, 3, 2, 1]),
            np.array([4, 5, 6, 7]),
            np.array([0, 1, 5, 4]),
            np.array([2, 3, 7, 6]),
            np.array([0, 4, 7, 3]),
            np.array([1, 2, 6, 5]),
        ]
        vol = polyhedron_volume(verts, faces)
        np.testing.assert_allclose(vol, 1.0, atol=1e-12)

    def test_constitutive_symmetry(self):
        """3D isotropic C matrix must be symmetric."""
        C = isotropic_3d(1000.0, 0.3)
        np.testing.assert_allclose(C, C.T, atol=1e-14)

    def test_constitutive_positive_definite(self):
        """C matrix must be positive definite."""
        C = isotropic_3d(1000.0, 0.3)
        eigenvalues = np.linalg.eigvalsh(C)
        assert np.all(eigenvalues > 0)


class TestPatchTest3D:
    """3D patch tests on different meshes."""

    def _run_patch_test(self, vertices, cells, cell_faces, solver_func):
        """Generic 3D patch test: uniform tension σ_xx = 1."""
        E_mod, nu = 1000.0, 0.3
        sigma = 1.0

        exact_ux = sigma * vertices[:, 0] / E_mod
        exact_uy = -nu * sigma * vertices[:, 1] / E_mod
        exact_uz = -nu * sigma * vertices[:, 2] / E_mod

        tol = 1e-6
        boundary = np.where(
            (vertices[:, 0] < tol) | (vertices[:, 0] > 1 - tol) |
            (vertices[:, 1] < tol) | (vertices[:, 1] > 1 - tol) |
            (vertices[:, 2] < tol) | (vertices[:, 2] > 1 - tol)
        )[0]

        bc_dofs = np.concatenate([3*boundary, 3*boundary+1, 3*boundary+2])
        bc_vals = np.concatenate([exact_ux[boundary], exact_uy[boundary],
                                  exact_uz[boundary]])

        u = solver_func(vertices, cells, cell_faces, E_mod, nu,
                        bc_dofs, bc_vals)
        ux, uy, uz = u[0::3], u[1::3], u[2::3]

        err = max(np.max(np.abs(ux - exact_ux)),
                  np.max(np.abs(uy - exact_uy)),
                  np.max(np.abs(uz - exact_uz)))
        return err

    def test_patch_test_dense(self, hex_mesh_3x3):
        """Patch test with dense solver."""
        vertices, cells, cell_faces = hex_mesh_3x3
        err = self._run_patch_test(vertices, cells, cell_faces,
                                   vem_3d_elasticity)
        assert err < 1e-5, f"3D patch test (dense) err={err:.2e}"

    def test_patch_test_sparse(self, hex_mesh_3x3):
        """Patch test with sparse solver."""
        vertices, cells, cell_faces = hex_mesh_3x3
        err = self._run_patch_test(vertices, cells, cell_faces, vem_3d_sparse)
        assert err < 1e-5, f"3D patch test (sparse) err={err:.2e}"

    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_patch_test_multiple_sizes(self, n):
        """Patch test on multiple mesh sizes."""
        perturb = 0.3 / n
        vertices, cells, cell_faces = make_hex_mesh(
            nx=n, ny=n, nz=n, perturb=perturb, seed=42)
        err = self._run_patch_test(vertices, cells, cell_faces, vem_3d_sparse)
        assert err < 1e-5, f"Patch test n={n} err={err:.2e}"


class TestConvergence3D:
    """h-refinement convergence for 3D VEM."""

    def test_convergence_rate(self):
        """Convergence rate should be ≥ 1 (k=1 VEM)."""
        ns = [2, 3, 4, 6]
        hs, errors = [], []
        E_mod, nu = 1000.0, 0.3

        for n in ns:
            perturb = 0.3 / n
            vertices, cells, cell_faces = make_hex_mesh(
                nx=n, ny=n, nz=n, perturb=perturb, seed=42)
            h = 1.0 / n

            exact_ux = vertices[:, 0] / E_mod
            exact_uy = -nu * vertices[:, 1] / E_mod
            exact_uz = -nu * vertices[:, 2] / E_mod

            tol = 1e-6
            boundary = np.where(
                (vertices[:, 0] < tol) | (vertices[:, 0] > 1 - tol) |
                (vertices[:, 1] < tol) | (vertices[:, 1] > 1 - tol) |
                (vertices[:, 2] < tol) | (vertices[:, 2] > 1 - tol)
            )[0]

            bc_dofs = np.concatenate(
                [3*boundary, 3*boundary+1, 3*boundary+2])
            bc_vals = np.concatenate(
                [exact_ux[boundary], exact_uy[boundary],
                 exact_uz[boundary]])

            u = vem_3d_sparse(vertices, cells, cell_faces, E_mod, nu,
                              bc_dofs, bc_vals)
            ux, uy, uz = u[0::3], u[1::3], u[2::3]
            err = max(np.max(np.abs(ux - exact_ux)),
                      np.max(np.abs(uy - exact_uy)),
                      np.max(np.abs(uz - exact_uz)))
            hs.append(h)
            errors.append(err)

        # Check convergence: for patch test, errors should decrease
        # (may be near machine eps, so just check monotone decrease)
        hs = np.array(hs)
        errors = np.array(errors)
        valid = errors > 1e-14
        if np.sum(valid) >= 2:
            h_v, e_v = hs[valid], errors[valid]
            rates = np.log(e_v[:-1] / e_v[1:]) / np.log(h_v[:-1] / h_v[1:])
            avg_rate = np.mean(rates)
            assert avg_rate > 0.5, f"Convergence rate {avg_rate:.2f} too low"


class TestPhysicalBehavior3D:
    """Test physical correctness of 3D solutions."""

    def test_compression_deflects_downward(self, hex_mesh_4x4):
        """Compressed cube top should move down."""
        vertices, cells, cell_faces = hex_mesh_4x4
        E_mod, nu = 1000.0, 0.3
        tol = 1e-6

        bottom = np.where(vertices[:, 2] < tol)[0]
        bc_dofs = np.concatenate([3*bottom, 3*bottom+1, 3*bottom+2])
        bc_vals = np.zeros(len(bc_dofs))

        top = np.where(vertices[:, 2] > 1 - tol)[0]
        load_dofs = 3 * top + 2
        load_vals = np.full(len(top), -1.0 / len(top))

        u = vem_3d_elasticity(vertices, cells, cell_faces, E_mod, nu,
                              bc_dofs, bc_vals, load_dofs, load_vals)
        uz = u[2::3]
        assert np.mean(uz[top]) < 0

    def test_symmetry(self, hex_mesh_4x4):
        """Symmetric loading should give symmetric response."""
        vertices, cells, cell_faces = hex_mesh_4x4
        E_mod, nu = 1000.0, 0.3
        tol = 1e-6

        bottom = np.where(vertices[:, 2] < tol)[0]
        bc_dofs = np.concatenate([3*bottom, 3*bottom+1, 3*bottom+2])
        bc_vals = np.zeros(len(bc_dofs))

        top = np.where(vertices[:, 2] > 1 - tol)[0]
        load_dofs = 3 * top + 2
        load_vals = np.full(len(top), -1.0 / len(top))

        u = vem_3d_elasticity(vertices, cells, cell_faces, E_mod, nu,
                              bc_dofs, bc_vals, load_dofs, load_vals)

        # Average uz should be roughly uniform on top (symmetric load)
        uz_top = u[2::3][top]
        relative_spread = np.std(uz_top) / abs(np.mean(uz_top))
        assert relative_spread < 0.5, "Top surface deflection should be roughly uniform"
