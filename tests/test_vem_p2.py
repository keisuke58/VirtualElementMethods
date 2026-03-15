"""Tests for P₂ VEM: 2nd-order virtual element elasticity."""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from vem_p2_elasticity import (
    generate_voronoi_mesh,
    add_edge_midpoints,
    _eval_p2_vector_basis,
    _eval_p2_strain,
    _compute_div_sigma,
    _compute_strain_energy_matrix,
    vem_p2_elasticity,
)


def _plane_stress_C(E, nu):
    """Plane-stress constitutive matrix."""
    fac = E / (1.0 - nu ** 2)
    return fac * np.array([
        [1.0, nu, 0.0],
        [nu, 1.0, 0.0],
        [0.0, 0.0, (1.0 - nu) / 2.0],
    ])


# ---------------------------------------------------------------------------
# Basis function tests
# ---------------------------------------------------------------------------

class TestP2Basis:
    """Test P₂ vector polynomial basis functions."""

    def test_basis_shape(self):
        """12 basis functions, each 2D vector."""
        B = _eval_p2_vector_basis(0.5, 0.3, 0.0, 0.0, 1.0)
        assert B.shape == (12, 2)

    def test_basis_at_origin(self):
        """At centroid (xhat=yhat=0): only constant modes nonzero."""
        B = _eval_p2_vector_basis(0.0, 0.0, 0.0, 0.0, 1.0)
        # Basis 0: (1,0), Basis 1: (0,1), Basis 2: (0,0) at origin
        assert B[0, 0] == pytest.approx(1.0)
        assert B[0, 1] == pytest.approx(0.0)
        assert B[1, 0] == pytest.approx(0.0)
        assert B[1, 1] == pytest.approx(1.0)
        # Linear and quadratic modes vanish at centroid
        for i in range(3, 12):
            assert abs(B[i, 0]) < 1e-14 and abs(B[i, 1]) < 1e-14

    def test_rigid_body_modes(self):
        """First 3 basis: translation (1,0), (0,1), rotation (-y,x)."""
        x, y, xc, yc, h = 0.3, 0.7, 0.0, 0.0, 1.0
        B = _eval_p2_vector_basis(x, y, xc, yc, h)
        # Mode 0: (1,0)
        np.testing.assert_allclose(B[0], [1, 0])
        # Mode 1: (0,1)
        np.testing.assert_allclose(B[1], [0, 1])
        # Mode 2: (-yhat, xhat) rotation
        xhat, yhat = (x - xc) / h, (y - yc) / h
        np.testing.assert_allclose(B[2], [-yhat, xhat], atol=1e-14)


class TestP2Strain:
    """Test strain computation from P₂ basis."""

    def test_strain_shape(self):
        """12 basis → 12 Voigt strain vectors (3,)."""
        S = _eval_p2_strain(0.5, 0.3, 0.0, 0.0, 1.0)
        assert S.shape == (12, 3)

    def test_rigid_modes_zero_strain(self):
        """Rigid body modes (0,1,2) must have zero strain everywhere."""
        S = _eval_p2_strain(0.7, 0.2, 0.5, 0.5, 0.8)
        for i in range(3):
            np.testing.assert_allclose(S[i], [0, 0, 0], atol=1e-14)

    def test_linear_strain_modes(self):
        """Linear strain modes (3,4,5) have constant strain."""
        S1 = _eval_p2_strain(0.1, 0.2, 0.0, 0.0, 1.0)
        S2 = _eval_p2_strain(0.8, 0.9, 0.0, 0.0, 1.0)
        for i in range(3, 6):
            np.testing.assert_allclose(S1[i], S2[i], atol=1e-14,
                                       err_msg=f"Strain mode {i} not constant")


# ---------------------------------------------------------------------------
# Div(sigma) volume correction tests
# ---------------------------------------------------------------------------

class TestDivSigma:
    """Test div(sigma) computation for volume correction."""

    def test_shape(self):
        C = _plane_stress_C(100.0, 0.3)
        ds = _compute_div_sigma(C, 1.0)
        assert ds.shape == (12, 2)

    def test_linear_modes_zero(self):
        """For linear basis (modes 0-5), div(sigma) = 0."""
        C = _plane_stress_C(100.0, 0.3)
        ds = _compute_div_sigma(C, 1.0)
        for i in range(6):
            np.testing.assert_allclose(ds[i], [0, 0], atol=1e-14,
                                       err_msg=f"Mode {i} should have zero div(sigma)")

    def test_quadratic_modes_nonzero(self):
        """Quadratic modes (6-11) generally have nonzero div(sigma)."""
        C = _plane_stress_C(100.0, 0.3)
        ds = _compute_div_sigma(C, 1.0)
        # At least some quadratic modes should have nonzero div
        has_nonzero = any(np.linalg.norm(ds[i]) > 1e-10 for i in range(6, 12))
        assert has_nonzero, "All quadratic div(sigma) are zero — unlikely"


# ---------------------------------------------------------------------------
# Strain energy matrix tests
# ---------------------------------------------------------------------------

class TestStrainEnergyMatrix:
    """Test analytical strain energy matrix computation."""

    def test_symmetric(self):
        """Strain energy matrix must be symmetric."""
        verts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        xc, yc = 0.5, 0.5
        h = np.sqrt(2)
        C = _plane_stress_C(100.0, 0.3)
        a_K = _compute_strain_energy_matrix(verts, xc, yc, h, C)
        np.testing.assert_allclose(a_K, a_K.T, atol=1e-10)

    def test_positive_semidefinite(self):
        """Strain energy matrix must be PSD."""
        verts = np.array([[0, 0], [1, 0], [0.8, 0.9], [0.1, 1.1]], dtype=float)
        xc = verts[:, 0].mean()
        yc = verts[:, 1].mean()
        h = max(np.linalg.norm(verts[i] - verts[j])
                for i in range(4) for j in range(i+1, 4))
        C = _plane_stress_C(100.0, 0.3)
        a_K = _compute_strain_energy_matrix(verts, xc, yc, h, C)
        eigvals = np.linalg.eigvalsh(a_K)
        assert np.all(eigvals >= -1e-10), f"Negative eigenvalue: {eigvals.min()}"

    def test_rigid_body_kernel(self):
        """Rigid body modes (0,1,2) should have zero strain energy."""
        verts = np.array([[0, 0], [2, 0], [2, 1], [0, 1]], dtype=float)
        xc, yc = 1.0, 0.5
        h = np.sqrt(5)
        C = _plane_stress_C(100.0, 0.3)
        a_K = _compute_strain_energy_matrix(verts, xc, yc, h, C)
        # Rows/columns 0,1,2 (rigid body modes) should have negligible entries
        for i in range(3):
            assert np.linalg.norm(a_K[i, :]) < 1e-10
            assert np.linalg.norm(a_K[:, i]) < 1e-10

    def test_scales_with_E(self):
        """Doubling E doubles strain energy matrix."""
        verts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        xc, yc, h = 0.5, 0.5, np.sqrt(2)
        a1 = _compute_strain_energy_matrix(verts, xc, yc, h, _plane_stress_C(100.0, 0.3))
        a2 = _compute_strain_energy_matrix(verts, xc, yc, h, _plane_stress_C(200.0, 0.3))
        np.testing.assert_allclose(a2, 2.0 * a1, atol=1e-10)


# ---------------------------------------------------------------------------
# Edge midpoint augmentation tests
# ---------------------------------------------------------------------------

class TestEdgeMidpoints:
    """Test add_edge_midpoints mesh augmentation."""

    def test_adds_midpoints(self):
        """Each edge gets exactly one midpoint node."""
        verts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        elems = [np.array([0, 1, 2, 3])]
        new_v, new_e, emap = add_edge_midpoints(verts, elems)
        # 4 vertices + 4 midpoints
        assert len(new_v) == 8
        # Element has 4 vertices + 4 midpoints
        assert len(new_e[0]) == 8

    def test_shared_edge_single_midpoint(self):
        """Two elements sharing an edge get the same midpoint node."""
        verts = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [2, 0], [2, 1]], dtype=float)
        elems = [np.array([0, 1, 2, 3]), np.array([1, 4, 5, 2])]
        new_v, new_e, emap = add_edge_midpoints(verts, elems)
        # Shared edge (1,2) should produce one midpoint
        shared_key = (min(1, 2), max(1, 2))
        assert shared_key in emap

    def test_midpoint_coordinates(self):
        """Midpoint coordinates are averages of edge endpoints."""
        verts = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=float)
        elems = [np.array([0, 1, 2, 3])]
        new_v, new_e, emap = add_edge_midpoints(verts, elems)
        for (v1, v2), mid_idx in emap.items():
            expected = (verts[v1] + verts[v2]) / 2
            np.testing.assert_allclose(new_v[mid_idx], expected)


# ---------------------------------------------------------------------------
# Solver integration tests
# ---------------------------------------------------------------------------

class TestP2Solver:
    """Integration tests for P₂ VEM solver."""

    @pytest.fixture
    def p2_mesh(self):
        """Small P₂ mesh for testing."""
        verts, elems, bnd = generate_voronoi_mesh(16, seed=42)
        new_v, new_e, _ = add_edge_midpoints(np.array(verts), elems)
        new_v = np.array(new_v)
        # Recompute boundary
        tol = 1e-10
        boundary = np.where(
            (np.abs(new_v[:, 0]) < tol) | (np.abs(new_v[:, 0] - 1) < tol) |
            (np.abs(new_v[:, 1]) < tol) | (np.abs(new_v[:, 1] - 1) < tol)
        )[0]
        return new_v, new_e, boundary

    def test_linear_solution_exact(self, p2_mesh):
        """P₂ VEM must reproduce linear displacement exactly."""
        verts, elems, bnd = p2_mesh
        n_el = len(elems)
        E_mod, nu = 100.0, 0.3

        # Linear solution: u = (x, -nu*y) / E
        exact_ux = verts[:, 0] / E_mod
        exact_uy = -nu * verts[:, 1] / E_mod

        bc_dofs = np.concatenate([2 * bnd, 2 * bnd + 1])
        bc_vals = np.concatenate([exact_ux[bnd], exact_uy[bnd]])

        u = vem_p2_elasticity(verts, elems, E_mod, nu, bc_dofs, bc_vals)

        ux, uy = u[0::2], u[1::2]
        err_x = np.max(np.abs(ux - exact_ux))
        err_y = np.max(np.abs(uy - exact_uy))

        assert err_x < 1e-10, f"P2 linear u_x error: {err_x:.2e}"
        assert err_y < 1e-10, f"P2 linear u_y error: {err_y:.2e}"

    def test_pure_shear_exact(self, p2_mesh):
        """Pure shear: u = (gamma*y, gamma*x) must be reproduced exactly."""
        verts, elems, bnd = p2_mesh
        n_el = len(elems)
        E_mod, nu = 200.0, 0.25
        gamma = 0.001

        exact_ux = gamma * verts[:, 1]
        exact_uy = gamma * verts[:, 0]

        bc_dofs = np.concatenate([2 * bnd, 2 * bnd + 1])
        bc_vals = np.concatenate([exact_ux[bnd], exact_uy[bnd]])

        u = vem_p2_elasticity(verts, elems, E_mod, nu, bc_dofs, bc_vals)

        assert np.max(np.abs(u[0::2] - exact_ux)) < 1e-10
        assert np.max(np.abs(u[1::2] - exact_uy)) < 1e-10

    def test_output_size(self, p2_mesh):
        """Output displacement vector has correct size."""
        verts, elems, bnd = p2_mesh
        n_dofs = 2 * len(verts)
        bc_dofs = np.concatenate([2 * bnd, 2 * bnd + 1])
        bc_vals = np.zeros(len(bc_dofs))

        u = vem_p2_elasticity(verts, elems, 100.0, 0.3, bc_dofs, bc_vals)
        assert len(u) == n_dofs

    def test_stiffer_material_less_displacement(self, p2_mesh):
        """Higher E → smaller displacement."""
        verts, elems, bnd = p2_mesh
        n_el = len(elems)

        bottom = np.where(verts[:, 1] < 1e-6)[0]
        top = np.where(verts[:, 1] > 1.0 - 1e-6)[0]
        bc_dofs = np.concatenate([2 * bottom, 2 * bottom + 1])
        bc_vals = np.zeros(len(bc_dofs))
        l_dofs = 2 * top + 1
        l_vals = np.full(len(top), -0.1 / max(len(top), 1))

        u_soft = vem_p2_elasticity(verts, elems, 50.0, 0.3,
                                    bc_dofs, bc_vals, l_dofs, l_vals)
        u_stiff = vem_p2_elasticity(verts, elems, 500.0, 0.3,
                                     bc_dofs, bc_vals, l_dofs, l_vals)

        mag_soft = np.max(np.abs(u_soft))
        mag_stiff = np.max(np.abs(u_stiff))
        assert mag_soft > mag_stiff, \
            f"Soft ({mag_soft:.4e}) should deform more than stiff ({mag_stiff:.4e})"

    def test_p2_better_than_p1(self):
        """P₂ error should be smaller than P₁ for quadratic displacement."""
        from vem_elasticity import vem_elasticity

        verts_p1, elems_p1, bnd_p1 = generate_voronoi_mesh(32, seed=42)
        verts_p2, elems_p2, _ = add_edge_midpoints(np.array(verts_p1), elems_p1)
        verts_p2 = np.array(verts_p2)

        bnd_p2 = np.where(
            (np.abs(verts_p2[:, 0]) < 1e-10) | (np.abs(verts_p2[:, 0] - 1) < 1e-10) |
            (np.abs(verts_p2[:, 1]) < 1e-10) | (np.abs(verts_p2[:, 1] - 1) < 1e-10)
        )[0]

        E_mod, nu = 100.0, 0.3

        # Quadratic displacement: u = (x^2, 0)
        # This is NOT exactly representable by P1
        exact_ux_p1 = verts_p1[:, 0] ** 2
        exact_uy_p1 = np.zeros(len(verts_p1))
        exact_ux_p2 = verts_p2[:, 0] ** 2
        exact_uy_p2 = np.zeros(len(verts_p2))

        # P1 solve
        bc_dofs_p1 = np.concatenate([2 * bnd_p1, 2 * bnd_p1 + 1])
        bc_vals_p1 = np.concatenate([exact_ux_p1[bnd_p1], exact_uy_p1[bnd_p1]])
        n_el_p1 = len(elems_p1)
        u_p1 = vem_elasticity(verts_p1, elems_p1,
                              np.full(n_el_p1, E_mod), nu,
                              bc_dofs_p1, bc_vals_p1)
        err_p1 = np.sqrt(np.mean((u_p1[0::2] - exact_ux_p1)**2 +
                                  (u_p1[1::2] - exact_uy_p1)**2))

        # P2 solve
        bc_dofs_p2 = np.concatenate([2 * bnd_p2, 2 * bnd_p2 + 1])
        bc_vals_p2 = np.concatenate([exact_ux_p2[bnd_p2], exact_uy_p2[bnd_p2]])
        u_p2 = vem_p2_elasticity(verts_p2, elems_p2, E_mod, nu,
                                  bc_dofs_p2, bc_vals_p2)
        err_p2 = np.sqrt(np.mean((u_p2[0::2] - exact_ux_p2)**2 +
                                  (u_p2[1::2] - exact_uy_p2)**2))

        assert err_p2 < err_p1, \
            f"P2 error ({err_p2:.4e}) should be smaller than P1 ({err_p1:.4e})"


# ---------------------------------------------------------------------------
# Voronoi mesh generation (from P2 module)
# ---------------------------------------------------------------------------

class TestP2VoronoiMesh:
    """Test Voronoi mesh from vem_p2_elasticity module."""

    def test_ccw_ordering(self):
        """All elements should have CCW vertex ordering (positive area)."""
        verts, elems, _ = generate_voronoi_mesh(25, seed=42)
        for k, el in enumerate(elems):
            el_int = el.astype(int)
            v = verts[el_int]
            area_comp = (v[:, 0] * np.roll(v[:, 1], -1) -
                         np.roll(v[:, 0], -1) * v[:, 1])
            area = 0.5 * np.sum(area_comp)
            assert area > 0, f"Element {k} has non-positive area {area}"

    def test_no_degenerate_elements(self):
        """No elements with < 3 vertices."""
        verts, elems, _ = generate_voronoi_mesh(30, seed=42)
        for k, el in enumerate(elems):
            assert len(el) >= 3, f"Element {k} has only {len(el)} vertices"
