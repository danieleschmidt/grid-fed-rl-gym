"""Version information for grid-fed-rl-gym."""

import os
from pathlib import Path

def get_version() -> str:
    """Get version from VERSION file."""
    version_file = Path(__file__).parent.parent / "VERSION"
    if version_file.exists():
        return version_file.read_text().strip()
    return "0.1.0"

__version__ = get_version()