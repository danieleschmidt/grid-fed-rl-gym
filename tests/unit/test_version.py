"""Test version information."""

import pytest
from grid_fed_rl.version import __version__, get_version


def test_version_format():
    """Test that version follows semantic versioning."""
    version_parts = __version__.split(".")
    assert len(version_parts) >= 2, "Version should have at least major.minor"
    
    for part in version_parts[:2]:
        assert part.isdigit(), f"Version part '{part}' should be numeric"


def test_get_version():
    """Test version retrieval function."""
    version = get_version()
    assert isinstance(version, str)
    assert len(version) > 0
    assert version == __version__