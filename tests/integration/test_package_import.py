"""Test package imports work correctly."""

import pytest


def test_main_package_import():
    """Test main package can be imported."""
    import grid_fed_rl
    assert hasattr(grid_fed_rl, "__version__")


def test_subpackage_imports():
    """Test all subpackages can be imported."""
    import grid_fed_rl.environments
    import grid_fed_rl.feeders  
    import grid_fed_rl.algorithms
    import grid_fed_rl.federated
    import grid_fed_rl.controllers
    import grid_fed_rl.evaluation
    import grid_fed_rl.utils


def test_cli_import():
    """Test CLI module can be imported."""
    from grid_fed_rl.cli import main
    assert callable(main)