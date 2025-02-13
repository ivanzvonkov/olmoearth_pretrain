"""Conftest for the tests."""

import random

import numpy as np
import pytest
import torch


@pytest.fixture(scope="session", autouse=True)
def set_random_seeds() -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)


@pytest.fixture
def supported_modalities() -> list[str]:
    """Create a list of supported modalities for testing."""
    return ["sentinel2", "latlon"]


# TODO: add some create mock data factory functions for all the contracts and different steps
