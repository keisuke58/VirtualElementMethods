"""Tests for VE-VEM: SLS viscoelastic VEM with Simo 1987 integrator."""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from vem_viscoelastic import (
    sls_params_from_di,
    _plane_stress_C,
    _compute_element_vem,
    generate_voronoi_mesh,
    vem_viscoelastic_sls,
)


# ---------------------------------------------------------------------------
# SLS parameter model tests
# ---------------------------------------------------------------------------

class TestSLSParams:
    """Test DI → SLS parameter mapping."""

    def test_commensal_stiff(self):
        """DI=0 → max stiffness, max relaxation time."""
        p = sls_params_from_di(np.array([0.0]))
        assert p["E_inf"][0] == pytest.approx(1000.0, rel=1e-10)
        assert p["tau"][0] == pytest.approx(60.0, rel=1e-10)

    def test_dysbiotic_soft(self):
        """DI=1 → min stiffness, min relaxation time."""
        p = sls_params_from_di(np.array([1.0]))
        assert p["E_inf"][0] == pytest.approx(10.0, rel=1e-10)
        assert p["tau"][0] == pytest.approx(2.0, rel=1e-10)

    def test_monotonic_E_inf(self):
        """E_inf must decrease monotonically with DI."""
        di = np.linspace(0, 1, 50)
        p = sls_params_from_di(di)
        assert np.all(np.diff(p["E_inf"]) <= 0)

    def test_monotonic_tau(self):
        """tau must decrease monotonically with DI."""
        di = np.linspace(0, 1, 50)
        p = sls_params_from_di(di)
        assert np.all(np.diff(p["tau"]) <= 0)

    def test_E1_positive(self):
        """E_1 = E_0 - E_inf must be positive everywhere."""
        di = np.linspace(0, 1, 100)
        p = sls_params_from_di(di)
        assert np.all(p["E_1"] > 0)

    def test_eta_equals_E1_times_tau(self):
        """Viscosity eta = E_1 * tau."""
        di = np.array([0.0, 0.3, 0.6, 1.0])
        p = sls_params_from_di(di)
        np.testing.assert_allclose(p["eta"], p["E_1"] * p["tau"], rtol=1e-12)

    def test_clipping(self):
        """DI values outside [0,1] should be clipped."""
        p_neg = sls_params_from_di(np.array([-0.5]))
        p_zero = sls_params_from_di(np.array([0.0]))
        np.testing.assert_allclose(p_neg["E_inf"], p_zero["E_inf"])

        p_over = sls_params_from_di(np.array([1.5]))
        p_one = sls_params_from_di(np.array([1.0]))
        np.testing.assert_allclose(p_over["E_inf"], p_one["E_inf"])

    def test_vectorized(self):
        """Vectorized computation matches element-wise."""
        di = np.array([0.1, 0.5, 0.9])
        p_vec = sls_params_from_di(di)
        for i, d in enumerate(di):
            p_i = sls_params_from_di(np.array([d]))
            assert p_vec["E_inf"][i] == pytest.approx(p_i["E_inf"][0])
            assert p_vec["tau"][i] == pytest.approx(p_i["tau"][0])


# ---------------------------------------------------------------------------
# Constitutive matrix tests
# ---------------------------------------------------------------------------

class TestConstitutiveMatrix:
    """Test plane-stress C matrix."""

    def test_symmetry(self):
        C = _plane_stress_C(100.0, 0.3)
        np.testing.assert_allclose(C, C.T, atol=1e-15)

    def test_positive_definite(self):
        C = _plane_stress_C(100.0, 0.3)
        eigvals = np.linalg.eigvalsh(C)
        assert np.all(eigvals > 0)

    def test_scales_with_E(self):
        C1 = _plane_stress_C(100.0, 0.3)
        C2 = _plane_stress_C(200.0, 0.3)
        np.testing.assert_allclose(C2, 2.0 * C1, rtol=1e-12)


# ---------------------------------------------------------------------------
# Element computation tests
# ---------------------------------------------------------------------------

class TestElementVEM:
    """Test P1 VEM element projector computation."""

    def test_square_element(self):
        """Square element: area=1, well-conditioned G."""
        verts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        ed = _compute_element_vem(verts, 0.3)
        assert ed["area"] == pytest.approx(1.0, rel=1e-10)
        assert ed["n_v"] == 4
        assert ed["n_el_dofs"] == 8

    def test_G_invertible(self):
        """G = B @ D must be non-singular."""
        verts = np.array([[0, 0], [1, 0], [0.8, 0.9], [0.1, 1.1]], dtype=float)
        ed = _compute_element_vem(verts, 0.3)
        assert abs(np.linalg.det(ed["G"])) > 1e-10

    def test_projector_identity_on_polynomials(self):
        """Pi @ D = I_6 (projector recovers polynomial coefficients)."""
        verts = np.array([[0, 0], [2, 0], [2, 1], [0, 1]], dtype=float)
        ed = _compute_element_vem(verts, 0.3)
        PiD = ed["projector"] @ ed["D"]
        np.testing.assert_allclose(PiD, np.eye(6), atol=1e-10)

    def test_strain_proj_shape(self):
        """strain_proj is (3, n_el_dofs)."""
        verts = np.array([[0, 0], [1, 0], [1, 1], [0.5, 1.2], [0, 1]], dtype=float)
        ed = _compute_element_vem(verts, 0.3)
        assert ed["strain_proj"].shape == (3, 10)


# ---------------------------------------------------------------------------
# Mesh generation tests
# ---------------------------------------------------------------------------

class TestVoronoiMesh:
    """Test Voronoi mesh generation."""

    def test_mesh_valid(self):
        v, e, b = generate_voronoi_mesh(16, seed=42)
        assert len(e) >= 14  # might lose a couple cells
        assert v.shape[1] == 2
        assert len(b) > 0

    def test_deterministic(self):
        v1, e1, b1 = generate_voronoi_mesh(20, seed=123)
        v2, e2, b2 = generate_voronoi_mesh(20, seed=123)
        np.testing.assert_array_equal(v1, v2)
        assert len(e1) == len(e2)

    def test_boundary_on_edges(self):
        v, e, b = generate_voronoi_mesh(16, seed=42)
        tol = 1e-10
        for i in b:
            on_edge = (abs(v[i, 0]) < tol or abs(v[i, 0] - 1) < tol or
                       abs(v[i, 1]) < tol or abs(v[i, 1] - 1) < tol)
            assert on_edge, f"Boundary node {i} at {v[i]} not on domain edge"


# ---------------------------------------------------------------------------
# Solver validation tests
# ---------------------------------------------------------------------------

class TestVEVEMSolver:
    """Integration tests for the full VE-VEM solver."""

    @pytest.fixture
    def confined_setup(self):
        """Laterally confined step displacement on uniform mesh."""
        vertices, elements, boundary = generate_voronoi_mesh(32, seed=42)
        n_el = len(elements)
        n_nodes = len(vertices)
        DI_val = 0.3
        DI_field = np.full(n_el, DI_val)
        nu = 0.3
        eps_0 = 0.01

        bottom = np.where(vertices[:, 1] < 1e-6)[0]
        top = np.where(vertices[:, 1] > 1.0 - 1e-6)[0]
        all_nodes = np.arange(n_nodes)

        bc_dofs = np.concatenate([2 * all_nodes, 2 * bottom + 1, 2 * top + 1])
        bc_vals = np.concatenate([
            np.zeros(n_nodes),
            np.zeros(len(bottom)),
            np.full(len(top), eps_0),
        ])
        bc_dofs, unique_idx = np.unique(bc_dofs, return_index=True)
        bc_vals = bc_vals[unique_idx]

        params = sls_params_from_di(DI_field)
        tau_val = params["tau"][0]
        t_array = np.concatenate([[0.0], np.linspace(tau_val / 10, 3 * tau_val, 30)])

        return {
            "vertices": vertices, "elements": elements,
            "DI_field": DI_field, "nu": nu, "eps_0": eps_0,
            "bc_dofs": bc_dofs, "bc_vals": bc_vals,
            "t_array": t_array, "params": params,
        }

    def test_machine_precision_confined(self, confined_setup):
        """Laterally confined uniform: VEM matches analytical to machine precision."""
        s = confined_setup
        u_hist, sigma_hist, h_hist = vem_viscoelastic_sls(
            s["vertices"], s["elements"], s["DI_field"], s["nu"],
            s["bc_dofs"], s["bc_vals"], s["t_array"],
        )

        E_inf = s["params"]["E_inf"][0]
        E_1 = s["params"]["E_1"][0]
        tau = s["params"]["tau"][0]
        fac = 1.0 / (1.0 - s["nu"]**2)

        sigma_ana = (E_inf + E_1 * np.exp(-s["t_array"] / tau)) * fac * s["eps_0"]
        sigma_vem = sigma_hist[:, :, 1].mean(axis=1)

        np.testing.assert_allclose(sigma_vem, sigma_ana, rtol=1e-12)

    def test_stress_relaxation_decreasing(self, confined_setup):
        """sigma_yy must decrease monotonically (stress relaxation)."""
        s = confined_setup
        _, sigma_hist, _ = vem_viscoelastic_sls(
            s["vertices"], s["elements"], s["DI_field"], s["nu"],
            s["bc_dofs"], s["bc_vals"], s["t_array"],
        )
        sigma_yy = sigma_hist[:, :, 1].mean(axis=1)
        # After t=0 (step application), stress decreases
        assert np.all(np.diff(sigma_yy[1:]) <= 1e-15)

    def test_long_time_limit(self, confined_setup):
        """Long-time stress converges to E_inf * eps_0 / (1-nu^2)."""
        s = confined_setup
        tau = s["params"]["tau"][0]
        t_long = np.concatenate([[0.0], np.linspace(1, 20 * tau, 50)])

        _, sigma_hist, _ = vem_viscoelastic_sls(
            s["vertices"], s["elements"], s["DI_field"], s["nu"],
            s["bc_dofs"], s["bc_vals"], t_long,
        )

        E_inf = s["params"]["E_inf"][0]
        fac = 1.0 / (1.0 - s["nu"]**2)
        sigma_inf_expected = E_inf * fac * s["eps_0"]
        sigma_inf_vem = sigma_hist[-1, :, 1].mean()

        assert sigma_inf_vem == pytest.approx(sigma_inf_expected, rel=1e-6)

    def test_instantaneous_response(self, confined_setup):
        """At t=0, stress = (E_inf + E_1) * eps_0 / (1-nu^2) = E_0 * eps_0 / (1-nu^2)."""
        s = confined_setup
        _, sigma_hist, _ = vem_viscoelastic_sls(
            s["vertices"], s["elements"], s["DI_field"], s["nu"],
            s["bc_dofs"], s["bc_vals"], s["t_array"],
        )

        E_0 = s["params"]["E_0"][0]
        fac = 1.0 / (1.0 - s["nu"]**2)
        sigma_0_expected = E_0 * fac * s["eps_0"]
        sigma_0_vem = sigma_hist[0, :, 1].mean()

        assert sigma_0_vem == pytest.approx(sigma_0_expected, rel=1e-10)

    def test_output_shapes(self, confined_setup):
        """Check output array shapes."""
        s = confined_setup
        n_t = len(s["t_array"])
        n_el = len(s["elements"])
        n_dofs = 2 * len(s["vertices"])

        u_hist, sigma_hist, h_hist = vem_viscoelastic_sls(
            s["vertices"], s["elements"], s["DI_field"], s["nu"],
            s["bc_dofs"], s["bc_vals"], s["t_array"],
        )

        assert u_hist.shape == (n_t, n_dofs)
        assert sigma_hist.shape == (n_t, n_el, 3)
        assert h_hist.shape == (n_t, n_el, 3)

    def test_di_gradient_ordering(self):
        """Higher DI → softer → lower stress at all times."""
        vertices, elements, boundary = generate_voronoi_mesh(20, seed=42)
        n_el = len(elements)
        n_nodes = len(vertices)
        nu = 0.3
        eps_0 = 0.01

        bottom = np.where(vertices[:, 1] < 1e-6)[0]
        top = np.where(vertices[:, 1] > 1.0 - 1e-6)[0]
        all_nodes = np.arange(n_nodes)

        bc_dofs = np.concatenate([2 * all_nodes, 2 * bottom + 1, 2 * top + 1])
        bc_vals = np.concatenate([
            np.zeros(n_nodes), np.zeros(len(bottom)), np.full(len(top), eps_0),
        ])
        bc_dofs, unique_idx = np.unique(bc_dofs, return_index=True)
        bc_vals = bc_vals[unique_idx]

        t_array = np.array([0.0, 10.0, 50.0])

        # DI=0.1 (stiff)
        _, sig_stiff, _ = vem_viscoelastic_sls(
            vertices, elements, np.full(n_el, 0.1), nu,
            bc_dofs, bc_vals, t_array,
        )
        # DI=0.8 (soft)
        _, sig_soft, _ = vem_viscoelastic_sls(
            vertices, elements, np.full(n_el, 0.8), nu,
            bc_dofs, bc_vals, t_array,
        )

        # Stiff material has higher stress at all times
        for ti in range(len(t_array)):
            assert sig_stiff[ti, :, 1].mean() > sig_soft[ti, :, 1].mean()

    def test_internal_variable_h_decays(self, confined_setup):
        """Internal variable h should decay toward C_inf contribution."""
        s = confined_setup
        _, _, h_hist = vem_viscoelastic_sls(
            s["vertices"], s["elements"], s["DI_field"], s["nu"],
            s["bc_dofs"], s["bc_vals"], s["t_array"],
        )
        # h magnitude should decrease after initial loading
        h_mag = np.linalg.norm(h_hist, axis=2).mean(axis=1)
        # After step 1, h decreases (exponential decay)
        assert h_mag[-1] < h_mag[1]
