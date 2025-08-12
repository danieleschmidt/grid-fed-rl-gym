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
except ImportError as e:
    # Graceful degradation if dependencies missing
    import warnings
    warnings.warn(f"Failed to import core components: {e}. Some functionality may be limited.")
    _IMPORTS_AVAILABLE = False
    
    # Create dummy classes for basic functionality
    class GridEnvironment:
        def __init__(self, *args, **kwargs):
            raise ImportError("GridEnvironment requires numpy and other dependencies")
    
    class BaseGridEnvironment:
        def __init__(self, *args, **kwargs):
            raise ImportError("BaseGridEnvironment requires numpy and other dependencies")
    
    class IEEE13Bus:
        def __init__(self, *args, **kwargs):
            raise ImportError("IEEE13Bus requires numpy and other dependencies")
    
    IEEE34Bus = IEEE13Bus
    IEEE123Bus = IEEE13Bus
    CustomFeeder = IEEE13Bus
    CQL = IEEE13Bus
    IQL = IEEE13Bus
    FederatedOfflineRL = IEEE13Bus

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