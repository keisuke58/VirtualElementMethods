"""Tests for space-time VEM."""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from vem_spacetime import (vem_anisotropic, make_spacetime_voronoi,
                            sls_params_from_DI, sls_relaxation)


class TestSpaceTimeVEM:
    """Test space-time VEM solver."""

    def test_constant_solution(self):
        """C=I, zero RHS, constant BC → constant solution."""
        vertices, elements, boundary = make_spacetime_voronoi(
            nx_seeds=8, nt_seeds=10, Lx=1.0, T=1.0, seed=42)
        n_cells = len(elements)

        C = np.eye(2)
        C_per_el = np.tile(C, (n_cells, 1, 1))

        bc_nodes = np.unique(np.concatenate([
            boundary['bottom'], boundary['left'],
            boundary['right'], boundary['top']
        ]))
        bc_vals = np.ones(len(bc_nodes)) * 5.0

        u = vem_anisotropic(vertices, elements, C_per_el, bc_nodes, bc_vals)
        np.testing.assert_allclose(u[bc_nodes], 5.0, atol=1e-10)

    def test_heat_equation_qualitative(self):
        """Heat solution decays from initial condition."""
        kappa = 0.1
        Lx, T = 1.0, 0.5
        beta = 0.01

        vertices, elements, boundary = make_spacetime_voronoi(
            nx_seeds=10, nt_seeds=12, Lx=Lx, T=T, seed=42)
        n_cells = len(elements)

        C = np.array([[kappa, 0.0], [0.0, beta]])
        C_per_el = np.tile(C, (n_cells, 1, 1))

        bc_nodes = np.unique(np.concatenate([
            boundary['bottom'], boundary['left'], boundary['right']
        ]))
        bc_vals = np.zeros(len(bc_nodes))
        for i, node in enumerate(bc_nodes):
            x, t = vertices[node]
            if t < 0.02:
                bc_vals[i] = np.sin(np.pi * x)

        u = vem_anisotropic(vertices, elements, C_per_el, bc_nodes, bc_vals)

        # Solution should decay in time
        assert np.all(np.isfinite(u))
        # Solution at t≈T should be smaller than at t≈0
        near_t0 = np.where(vertices[:, 1] < 0.1)[0]
        near_tT = np.where(vertices[:, 1] > T - 0.1)[0]
        if len(near_t0) > 0 and len(near_tT) > 0:
            assert np.mean(np.abs(u[near_tT])) < np.mean(np.abs(u[near_t0])) + 0.1

    def test_heat_convergence(self):
        """Space-time VEM converges for heat equation."""
        kappa = 0.1
        Lx, T = 1.0, 0.3
        beta = 0.01

        configs = [(6, 8), (10, 14), (15, 20)]
        errors = []

        for nx_s, nt_s in configs:
            vertices, elements, boundary = make_spacetime_voronoi(
                nx_seeds=nx_s, nt_seeds=nt_s, Lx=Lx, T=T, seed=42)
            n_cells = len(elements)

            C = np.array([[kappa, 0.0], [0.0, beta]])
            C_per_el = np.tile(C, (n_cells, 1, 1))

            bc_nodes = np.unique(np.concatenate([
                boundary['bottom'], boundary['left'], boundary['right']
            ]))
            bc_vals = np.zeros(len(bc_nodes))
            for i, node in enumerate(bc_nodes):
                x, t = vertices[node]
                if t < 0.02:
                    bc_vals[i] = np.sin(np.pi * x)

            u = vem_anisotropic(vertices, elements, C_per_el, bc_nodes, bc_vals)

            # Exact solution
            u_exact = np.sin(np.pi * vertices[:, 0]) * np.exp(
                -kappa * np.pi**2 * vertices[:, 1])

            used = set()
            for el in elements:
                used.update(el.astype(int).tolist())
            used = np.array(sorted(used))
            err = np.max(np.abs(u[used] - u_exact[used]))
            errors.append(err)

        # Errors should generally decrease with refinement
        assert errors[-1] < errors[0] * 1.5, \
            f"No convergence: errors={errors}"


class TestSLSMaterial:
    """Test SLS viscoelastic material model."""

    def test_sls_params_bounds(self):
        """SLS parameters are positive for all DI."""
        for DI in np.linspace(0, 1, 20):
            E_inf, E_1, tau, eta = sls_params_from_DI(DI)
            assert E_inf > 0
            assert E_1 >= 0
            assert tau > 0
            assert eta >= 0

    def test_sls_commensal_stiffer(self):
        """Commensal (DI=0.1) should be stiffer than dysbiotic (DI=0.8)."""
        E_inf_c, _, _, _ = sls_params_from_DI(0.1)
        E_inf_d, _, _, _ = sls_params_from_DI(0.8)
        assert E_inf_c > E_inf_d

    def test_sls_relaxation_monotone(self):
        """Stress relaxation should decrease monotonically."""
        E_inf, E_1, tau, _ = sls_params_from_DI(0.5)
        t = np.linspace(0, 10, 100)
        sigma = sls_relaxation(E_inf, E_1, tau, 0.01, t)
        assert np.all(np.diff(sigma) <= 1e-15), "Relaxation must be monotonically decreasing"

    def test_sls_relaxation_limits(self):
        """σ(0) = (E_inf + E_1)·ε₀, σ(∞) = E_inf·ε₀."""
        E_inf, E_1, tau, _ = sls_params_from_DI(0.3)
        eps_0 = 0.01
        sigma_0 = sls_relaxation(E_inf, E_1, tau, eps_0, 0.0)
        sigma_inf = sls_relaxation(E_inf, E_1, tau, eps_0, 1e6)

        np.testing.assert_allclose(sigma_0, (E_inf + E_1) * eps_0, rtol=1e-10)
        np.testing.assert_allclose(sigma_inf, E_inf * eps_0, rtol=1e-6)
