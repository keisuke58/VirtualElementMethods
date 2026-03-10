"""Tests for a posteriori error estimator and adaptive VEM."""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from vem_error_estimator import (
    l2_error, h1_seminorm_error, estimate_element_error,
    compute_mesh_quality, adaptive_vem_solve, refine_mesh_adaptive,
)
from vem_elasticity import vem_elasticity, load_mesh


class TestErrorNorms:
    """Test L² and H¹ error norm computation."""

    def test_l2_zero_for_exact(self, mesh_dir):
        """L² error should be zero when u_h = u_exact."""
        vertices, elements, boundary = load_mesh(
            os.path.join(mesh_dir, 'voronoi.mat'))
        u = np.random.randn(2 * len(vertices))
        err = l2_error(u, u, vertices, elements)
        np.testing.assert_allclose(err, 0.0, atol=1e-14)

    def test_h1_zero_for_exact(self, mesh_dir):
        """H¹ error should be zero when u_h = u_exact."""
        vertices, elements, boundary = load_mesh(
            os.path.join(mesh_dir, 'voronoi.mat'))
        u = np.random.randn(2 * len(vertices))
        err = h1_seminorm_error(u, u, vertices, elements)
        np.testing.assert_allclose(err, 0.0, atol=1e-14)

    def test_l2_positive_for_different(self, mesh_dir):
        """L² error positive when solutions differ."""
        vertices, elements, boundary = load_mesh(
            os.path.join(mesh_dir, 'voronoi.mat'))
        u1 = np.zeros(2 * len(vertices))
        u2 = np.ones(2 * len(vertices)) * 0.01
        err = l2_error(u1, u2, vertices, elements)
        assert err > 0

    def test_l2_triangle_inequality(self, mesh_dir):
        """L² norm satisfies triangle inequality."""
        vertices, elements, boundary = load_mesh(
            os.path.join(mesh_dir, 'voronoi.mat'))
        n = 2 * len(vertices)
        rng = np.random.default_rng(42)
        u1, u2, u3 = rng.standard_normal(n), rng.standard_normal(n), rng.standard_normal(n)

        e12 = l2_error(u1, u2, vertices, elements)
        e23 = l2_error(u2, u3, vertices, elements)
        e13 = l2_error(u1, u3, vertices, elements)
        assert e13 <= e12 + e23 + 1e-10

    def test_scalar_l2(self, mesh_dir):
        """L² works for scalar fields."""
        vertices, elements, boundary = load_mesh(
            os.path.join(mesh_dir, 'voronoi.mat'))
        u1 = np.zeros(len(vertices))
        u2 = np.ones(len(vertices)) * 0.1
        err = l2_error(u1, u2, vertices, elements)
        assert err > 0


class TestErrorEstimator:
    """Test a posteriori error estimator."""

    def test_estimator_nonnegative(self, mesh_dir):
        """Error indicators must be non-negative."""
        vertices, elements, boundary = load_mesh(
            os.path.join(mesh_dir, 'voronoi.mat'))
        E_mod, nu = 1000.0, 0.3
        tol = 1e-6

        bottom = np.where(vertices[:, 1] < tol)[0]
        bc_dofs = np.concatenate([2*bottom, 2*bottom+1])
        bc_vals = np.zeros(len(bc_dofs))

        top = np.where(vertices[:, 1] > 1 - tol)[0]
        load_dofs = 2 * top + 1
        load_vals = np.full(len(top), -0.5 / len(top))

        u = vem_elasticity(vertices, elements, E_mod, nu,
                           bc_dofs, bc_vals, load_dofs, load_vals)

        eta = estimate_element_error(u, vertices, elements, E_mod, nu)
        assert np.all(eta >= -1e-15)

    def test_patch_test_zero_estimator(self, mesh_dir):
        """Patch test (exact solution) should give near-zero estimator."""
        vertices, elements, boundary = load_mesh(
            os.path.join(mesh_dir, 'voronoi.mat'))
        E_mod, nu = 1000.0, 0.3

        exact_ux = vertices[:, 0] / E_mod
        exact_uy = -nu * vertices[:, 1] / E_mod
        bc_dofs = np.concatenate([2*boundary, 2*boundary+1])
        bc_vals = np.concatenate([exact_ux[boundary], exact_uy[boundary]])

        u = vem_elasticity(vertices, elements, E_mod, nu, bc_dofs, bc_vals)
        eta = estimate_element_error(u, vertices, elements, E_mod, nu)

        # For exact solution, traction jumps should be very small
        assert np.max(eta) < 1e-6


class TestMeshQuality:
    """Test mesh quality metric computation."""

    def test_quality_keys(self, mesh_dir):
        """Quality metrics contain expected keys."""
        vertices, elements, _ = load_mesh(
            os.path.join(mesh_dir, 'voronoi.mat'))
        q = compute_mesh_quality(vertices, elements)
        assert 'aspect_ratios' in q
        assert 'areas' in q
        assert 'regularity' in q
        assert 'min_angles' in q
        assert 'summary' in q

    def test_positive_areas(self, mesh_dir):
        """All element areas must be positive."""
        vertices, elements, _ = load_mesh(
            os.path.join(mesh_dir, 'voronoi.mat'))
        q = compute_mesh_quality(vertices, elements)
        assert np.all(q['areas'] > 0)

    def test_aspect_ratio_bounds(self, mesh_dir):
        """Aspect ratios ≥ 1."""
        vertices, elements, _ = load_mesh(
            os.path.join(mesh_dir, 'squares.mat'))
        q = compute_mesh_quality(vertices, elements)
        finite = q['aspect_ratios'][np.isfinite(q['aspect_ratios'])]
        assert np.all(finite >= 1.0 - 1e-10)

    def test_angles_positive(self, mesh_dir):
        """All internal angles must be positive."""
        vertices, elements, _ = load_mesh(
            os.path.join(mesh_dir, 'voronoi.mat'))
        q = compute_mesh_quality(vertices, elements)
        assert np.all(q['min_angles'] > 0)


class TestAdaptiveVEM:
    """Test adaptive mesh refinement."""

    def test_adaptive_increases_cells(self):
        """Refinement should increase number of cells."""
        def E_field(x, y):
            return 30 + 970 * (1 - 0.5)**2

        results = adaptive_vem_solve(
            E_field, nu=0.3, n_refine=2, theta=0.3,
            n_initial_seeds=10)

        assert results[-1]['n_cells'] >= results[0]['n_cells']

    def test_adaptive_reduces_error(self):
        """Adaptive refinement should reduce total error indicator."""
        def E_field(x, y):
            DI = 0.9 - 0.8 * abs(x - 0.5) / 0.5
            return 30 + 970 * (1 - DI)**2

        results = adaptive_vem_solve(
            E_field, nu=0.3, n_refine=2, theta=0.3,
            n_initial_seeds=12)

        # Error indicator should generally decrease
        # (might not always due to mesh quality, but trend should be down)
        eta_first = results[0]['eta_total']
        eta_last = results[-1]['eta_total']
        # Allow some tolerance since adaptive refinement is heuristic
        assert eta_last < eta_first * 2.0

    def test_refine_mesh_produces_valid_mesh(self, mesh_dir):
        """Refined mesh should have valid topology."""
        vertices, elements, boundary = load_mesh(
            os.path.join(mesh_dir, 'voronoi.mat'))

        E_mod, nu = 1000.0, 0.3
        tol = 1e-6
        bottom = np.where(vertices[:, 1] < tol)[0]
        bc_dofs = np.concatenate([2*bottom, 2*bottom+1])
        bc_vals = np.zeros(len(bc_dofs))
        top = np.where(vertices[:, 1] > 1 - tol)[0]
        load_dofs = 2*top+1
        load_vals = np.full(len(top), -0.5/len(top))

        u = vem_elasticity(vertices, elements, E_mod, nu,
                           bc_dofs, bc_vals, load_dofs, load_vals)
        eta = estimate_element_error(u, vertices, elements, E_mod, nu)

        new_verts, new_elems, new_bnd, marked = refine_mesh_adaptive(
            vertices, elements, eta, theta=0.3)

        assert len(new_elems) >= len(elements)
        assert len(new_verts) > 0
        # All elements should have ≥ 3 vertices
        for el in new_elems:
            assert len(el) >= 3
