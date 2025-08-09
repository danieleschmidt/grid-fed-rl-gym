"""
Grid-Fed-RL-Gym: Federated Reinforcement Learning for Power Grids

A comprehensive framework for training and deploying reinforcement learning agents
on power distribution networks with federated learning capabilities.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragonlabs.com"

# Core exports for basic functionality
try:
    from .environments.grid_env import GridEnvironment
    from .environments.base import BaseGridEnvironment
    from .feeders.ieee_feeders import IEEE13Bus, IEEE34Bus, IEEE123Bus
    from .feeders.base import CustomFeeder
    from .algorithms.offline import CQL, IQL
    from .federated.core import FederatedOfflineRL
    _IMPORTS_AVAILABLE = True
except ImportError:
    # Graceful degradation if dependencies missing
    _IMPORTS_AVAILABLE = False
    GridEnvironment = None
    BaseGridEnvironment = None

__all__ = [
    "__version__",
    "__author__", 
    "__email__",
    "GridEnvironment",
    "BaseGridEnvironment",
    "IEEE13Bus", 
    "IEEE34Bus", 
    "IEEE123Bus",
    "CustomFeeder",
    "CQL",
    "IQL", 
    "FederatedOfflineRL"
]