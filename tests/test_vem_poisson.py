"""Tests for scalar 2D VEM (Poisson equation)."""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from vem import vem, square_domain_rhs, square_domain_boundary_condition


class TestPoissonVEM:
    """Test scalar VEM on Poisson equation."""

    def test_square_domain_solves(self, mesh_dir):
        """VEM produces a non-trivial solution on square domain."""
        u = vem(os.path.join(mesh_dir, 'voronoi.mat'),
                square_domain_rhs, square_domain_boundary_condition)
        assert u is not None
        assert len(u) > 0
        assert np.max(np.abs(u)) > 0

    def test_boundary_values_satisfied(self, mesh_dir):
        """Boundary conditions are exactly satisfied."""
        import scipy.io
        mesh_file = os.path.join(mesh_dir, 'voronoi.mat')
        mesh = scipy.io.loadmat(mesh_file)
        vertices = mesh['vertices']
        boundary = mesh['boundary'].T[0] - 1

        u = vem(mesh_file, square_domain_rhs, square_domain_boundary_condition)
        expected = square_domain_boundary_condition(vertices[boundary])
        np.testing.assert_allclose(u[boundary], expected, atol=1e-12)

    @pytest.mark.parametrize("mesh_name", [
        'voronoi.mat', 'squares.mat', 'triangles.mat', 'smoothed-voronoi.mat'
    ])
    def test_solves_on_all_meshes(self, mesh_dir, mesh_name):
        """VEM works on all available mesh types."""
        mesh_file = os.path.join(mesh_dir, mesh_name)
        if not os.path.exists(mesh_file):
            pytest.skip(f"Mesh file {mesh_name} not found")
        u = vem(mesh_file, square_domain_rhs, square_domain_boundary_condition)
        assert np.all(np.isfinite(u))

    def test_zero_rhs_gives_harmonic(self, mesh_dir):
        """Zero RHS with non-trivial BCs gives harmonic function."""
        def zero_rhs(point):
            return 0.0

        def linear_bc(points):
            return points[:, 0] + points[:, 1]

        mesh_file = os.path.join(mesh_dir, 'voronoi.mat')
        u = vem(mesh_file, zero_rhs, linear_bc)
        # Harmonic function with linear BC should be linear → check it's bounded
        assert np.all(np.isfinite(u))
        assert np.max(u) <= 2.5  # max of x+y on [0,1]^2 is 2

    def test_constant_solution(self, mesh_dir):
        """Zero RHS + constant BC → constant solution."""
        def zero_rhs(point):
            return 0.0

        def const_bc(points):
            return np.ones(len(points)) * 3.14

        mesh_file = os.path.join(mesh_dir, 'squares.mat')
        u = vem(mesh_file, zero_rhs, const_bc)
        np.testing.assert_allclose(u, 3.14, atol=1e-10)
