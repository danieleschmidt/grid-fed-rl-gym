"""IEEE test feeders and network topology definitions."""

from .ieee_feeders import IEEE13Bus, IEEE34Bus, IEEE123Bus
from .base import BaseFeeder, CustomFeeder, SimpleRadialFeeder

# Try to import synthetic feeders gracefully
try:
    from .synthetic import SyntheticFeeder, RandomFeeder
    _SYNTHETIC_AVAILABLE = True
except ImportError:
    _SYNTHETIC_AVAILABLE = False
    # Create stub classes
    class SyntheticFeeder:
        pass
    class RandomFeeder:
        pass

__all__ = [
    "IEEE13Bus",
    "IEEE34Bus", 
    "IEEE123Bus",
    "BaseFeeder",
    "CustomFeeder",
    "SimpleRadialFeeder",
    "SyntheticFeeder",
    "RandomFeeder"
]