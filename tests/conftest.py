"""Shared fixtures for VEM test suite."""

import sys
import os
import numpy as np
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

MESH_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'meshes')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')


@pytest.fixture
def mesh_dir():
    return MESH_DIR


@pytest.fixture
def results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return RESULTS_DIR


@pytest.fixture
def voronoi_mesh():
    """Load voronoi.mat mesh for 2D tests."""
    import scipy.io
    mesh = scipy.io.loadmat(os.path.join(MESH_DIR, 'voronoi.mat'))
    vertices = mesh['vertices']
    elements = np.array(
        [i[0].reshape(-1) - 1 for i in mesh['elements']], dtype=object)
    boundary = mesh['boundary'].T[0] - 1
    return vertices, elements, boundary


@pytest.fixture
def smoothed_voronoi_mesh():
    """Load smoothed-voronoi.mat mesh."""
    import scipy.io
    mesh = scipy.io.loadmat(os.path.join(MESH_DIR, 'smoothed-voronoi.mat'))
    vertices = mesh['vertices']
    elements = np.array(
        [i[0].reshape(-1) - 1 for i in mesh['elements']], dtype=object)
    boundary = mesh['boundary'].T[0] - 1
    return vertices, elements, boundary


@pytest.fixture
def hex_mesh_3x3():
    """3x3x3 perturbed hex mesh for 3D tests."""
    from vem_3d import make_hex_mesh
    return make_hex_mesh(nx=3, ny=3, nz=3, perturb=0.15, seed=42)


@pytest.fixture
def hex_mesh_4x4():
    """4x4x4 perturbed hex mesh."""
    from vem_3d import make_hex_mesh
    return make_hex_mesh(nx=4, ny=4, nz=4, perturb=0.2, seed=42)


@pytest.fixture
def voronoi_mesh_3d():
    """3D Voronoi polyhedral mesh."""
    from vem_3d_advanced import make_voronoi_mesh_3d
    return make_voronoi_mesh_3d(n_seeds=30, seed=42)
