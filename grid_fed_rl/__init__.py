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
    # First ensure numpy is available in virtual environment
    import sys
    import os
    
    # Add virtual environment to path if activated
    if 'VIRTUAL_ENV' in os.environ:
        venv_path = os.environ['VIRTUAL_ENV']
        sys.path.insert(0, os.path.join(venv_path, 'lib', 'python3.12', 'site-packages'))
    
    # Also check for venv in current directory
    venv_local = os.path.join(os.path.dirname(__file__), '..', 'venv', 'lib', 'python3.12', 'site-packages')
    if os.path.exists(venv_local):
        sys.path.insert(0, venv_local)
    
    import numpy as np  # Test numpy availability
    
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
    
    # Create working fallback classes for basic functionality
    from .environments.grid_env import GridEnvironment as _GridEnvironment
    from .environments.base import BaseGridEnvironment as _BaseGridEnvironment
    from .feeders.ieee_feeders import IEEE13Bus as _IEEE13Bus, IEEE34Bus as _IEEE34Bus, IEEE123Bus as _IEEE123Bus
    from .feeders.base import CustomFeeder as _CustomFeeder
    
    # Use the working implementations with fallback
    GridEnvironment = _GridEnvironment
    BaseGridEnvironment = _BaseGridEnvironment
    IEEE13Bus = _IEEE13Bus
    IEEE34Bus = _IEEE34Bus
    IEEE123Bus = _IEEE123Bus
    CustomFeeder = _CustomFeeder
    
    # Create dummy RL algorithm classes
    class CQL:
        def __init__(self, *args, **kwargs):
            raise ImportError("CQL requires torch and other ML dependencies")
    
    class IQL:
        def __init__(self, *args, **kwargs):
            raise ImportError("IQL requires torch and other ML dependencies")
    
    class FederatedOfflineRL:
        def __init__(self, *args, **kwargs):
            raise ImportError("FederatedOfflineRL requires torch and other ML dependencies")

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