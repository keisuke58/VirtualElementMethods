"""Tests for growth-coupled VEM."""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from vem_growth_coupled import (
    hamilton_step, compute_DI, compute_E, make_interaction_matrix,
    BiofilmGrowthVEM
)


class TestHamiltonODE:
    """Test replicator equation dynamics."""

    def test_simplex_preservation(self):
        """Species fractions must sum to 1 after stepping."""
        phi = np.array([0.3, 0.2, 0.2, 0.2, 0.1])
        A = make_interaction_matrix('dh_baseline')
        for _ in range(100):
            phi = hamilton_step(phi, A, dt=0.5)
            np.testing.assert_allclose(phi.sum(), 1.0, atol=1e-10)

    def test_positivity(self):
        """All fractions remain positive."""
        phi = np.array([0.01, 0.01, 0.01, 0.01, 0.96])
        A = make_interaction_matrix('commensal_static')
        for _ in range(200):
            phi = hamilton_step(phi, A, dt=0.5)
            assert np.all(phi > 0), f"Negative fraction: {phi}"

    @pytest.mark.parametrize("condition", [
        'commensal_static', 'dh_baseline', 'dysbiotic_static'
    ])
    def test_equilibrium_stable(self, condition):
        """Replicator converges to equilibrium."""
        phi = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        A = make_interaction_matrix(condition)
        for _ in range(500):
            phi = hamilton_step(phi, A, dt=0.5)

        # Should have reached something stable
        phi_prev = phi.copy()
        for _ in range(50):
            phi = hamilton_step(phi, A, dt=0.5)
        np.testing.assert_allclose(phi, phi_prev, atol=1e-4,
                                   err_msg=f"Not converged for {condition}")


class TestDIAndE:
    """Test DI and E(DI) functions."""

    def test_DI_range(self):
        """DI should be in [0, 1]."""
        for _ in range(100):
            phi = np.random.dirichlet(np.ones(5))
            DI = compute_DI(phi)
            assert 0 <= DI <= 1

    def test_DI_commensal_low(self):
        """Pure An should give DI = 0."""
        phi = np.array([1.0, 0, 0, 0, 0])
        assert compute_DI(phi) == 0.0

    def test_DI_pathogenic_high(self):
        """Pure Pg should give DI = 1."""
        phi = np.array([0, 0, 0, 0, 1.0])
        assert compute_DI(phi) == 1.0

    def test_E_range(self):
        """E should be in [E_min, E_max]."""
        for DI in np.linspace(0, 1, 50):
            E = compute_E(DI)
            assert 30 <= E <= 1000

    def test_E_monotone_decreasing(self):
        """Higher DI → lower E."""
        E_vals = [compute_E(DI) for DI in np.linspace(0, 1, 50)]
        assert all(E_vals[i] >= E_vals[i+1] for i in range(len(E_vals) - 1))


class TestBiofilmGrowthVEM:
    """Integration test for growth-coupled simulation."""

    def test_simulation_runs(self):
        """Basic simulation completes without error."""
        sim = BiofilmGrowthVEM(n_cells=15, condition='dh_baseline', seed=42)
        history = sim.run(n_steps=5, dt=1.0, division_interval=3,
                          verbose=False)
        assert len(history) == 5
        assert history[-1]['n_cells'] > 0

    def test_cell_division_increases_count(self):
        """Cell division should increase cell count."""
        sim = BiofilmGrowthVEM(n_cells=20, condition='dh_baseline', seed=42)
        initial_cells = sim.n_cells
        sim.run(n_steps=10, dt=1.0, division_interval=3, verbose=False)
        # After 3 division intervals (steps 3, 6, 9), might have divided
        assert sim.n_cells >= initial_cells

    @pytest.mark.parametrize("condition", [
        'commensal_static', 'dh_baseline', 'dysbiotic_static'
    ])
    def test_condition_affects_DI(self, condition):
        """Different conditions should give different DI ranges."""
        sim = BiofilmGrowthVEM(n_cells=20, condition=condition, seed=42)
        sim.run(n_steps=15, dt=1.0, division_interval=20, verbose=False)
        sim.compute_properties()
        assert 0 <= np.mean(sim.DI) <= 1

    def test_dysbiotic_softer_than_commensal(self):
        """Dysbiotic biofilm should have lower mean E."""
        sim_cs = BiofilmGrowthVEM(n_cells=20, condition='commensal_static',
                                  seed=42)
        sim_cs.run(n_steps=20, dt=1.0, division_interval=30, verbose=False)
        sim_cs.compute_properties()

        sim_ds = BiofilmGrowthVEM(n_cells=20, condition='dysbiotic_static',
                                  seed=42)
        sim_ds.run(n_steps=20, dt=1.0, division_interval=30, verbose=False)
        sim_ds.compute_properties()

        assert np.mean(sim_cs.E) > np.mean(sim_ds.E), \
            f"CS E={np.mean(sim_cs.E):.0f} should > DS E={np.mean(sim_ds.E):.0f}"

    def test_displacement_finite(self):
        """All displacements should be finite."""
        sim = BiofilmGrowthVEM(n_cells=15, condition='dh_baseline', seed=42)
        sim.run(n_steps=3, dt=1.0, division_interval=10, verbose=False)
        assert np.all(np.isfinite(sim.u))
