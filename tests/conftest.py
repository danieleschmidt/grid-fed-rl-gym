"""Shared test fixtures and configuration for pytest."""

import pytest
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any


@pytest.fixture
def test_data_dir() -> Path:
    """Get path to test data directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_grid_config() -> Dict[str, Any]:
    """Sample grid configuration for testing."""
    return {
        "feeder": "IEEE13Bus",
        "timestep": 1.0,
        "episode_length": 100,
        "stochastic_loads": True,
        "renewable_sources": ["solar", "wind"]
    }


@pytest.fixture
def sample_training_config() -> Dict[str, Any]:
    """Sample training configuration for testing."""
    return {
        "algorithm": "CQL",
        "batch_size": 32,
        "learning_rate": 1e-4,
        "episodes": 10,
        "hidden_dims": [64, 64]
    }


@pytest.fixture
def mock_grid_data() -> Dict[str, np.ndarray]:
    """Mock grid measurement data."""
    np.random.seed(42)
    return {
        "bus_voltages": np.random.uniform(0.95, 1.05, size=(13,)),
        "line_currents": np.random.uniform(0, 100, size=(12,)),
        "power_demands": np.random.uniform(10, 500, size=(13,)),
        "der_outputs": np.random.uniform(0, 200, size=(5,))
    }


@pytest.fixture
def torch_device() -> torch.device:
    """Get appropriate torch device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def test_output_dir(tmp_path_factory) -> Path:
    """Create temporary directory for test outputs."""
    return tmp_path_factory.mktemp("test_outputs")