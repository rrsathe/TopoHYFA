"""
Shared fixtures for the TopoHYFA test suite.
"""

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def _seed():
    """Fix all RNG seeds for reproducible numerical tests."""
    torch.manual_seed(0)
    np.random.seed(0)


BATCH = 8
GENES = 16
LATENT = 4


@pytest.fixture
def batch_size() -> int:
    return BATCH


@pytest.fixture
def n_genes() -> int:
    return GENES


@pytest.fixture
def n_latent() -> int:
    return LATENT


@pytest.fixture
def positive_rate() -> torch.Tensor:
    """(BATCH, GENES) positive float tensor – valid mean parameter for NB/Poisson."""
    return torch.rand(BATCH, GENES) + 0.5


@pytest.fixture
def dispersion() -> torch.Tensor:
    """(BATCH, GENES) positive float tensor – valid dispersion (theta)."""
    return torch.rand(BATCH, GENES) + 0.1


@pytest.fixture
def count_obs() -> torch.Tensor:
    """(BATCH, GENES) non-negative integer observations (floats for log_prob)."""
    return torch.randint(0, 10, (BATCH, GENES)).float()


@pytest.fixture
def numpy_matrix() -> np.ndarray:
    """(BATCH, GENES) float array of 'gene expression' values."""
    rng = np.random.default_rng(42)
    return rng.random((BATCH, GENES)).astype(np.float32)
