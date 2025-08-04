"""Reinforcement learning algorithms for grid control."""

try:
    from .offline import CQL, IQL, AWR
    from .base import BaseAlgorithm, OfflineRLAlgorithm
    offline_available = True
except ImportError:
    from .base import BaseAlgorithm
    offline_available = False
    print("Warning: Offline RL algorithms require torch. Install torch to use CQL, IQL, AWR.")

# Placeholder classes for missing dependencies
try:
    from .safe import SafeRL, ConstrainedPolicyOptimization
except ImportError:
    class SafeRL: pass
    class ConstrainedPolicyOptimization: pass

try:
    from .multi_agent import MADDPG, QMIX
except ImportError:
    class MADDPG: pass
    class QMIX: pass

__all__ = [
    "BaseAlgorithm"
]

if offline_available:
    __all__.extend(["CQL", "IQL", "AWR", "OfflineRLAlgorithm"])
    
__all__.extend(["SafeRL", "ConstrainedPolicyOptimization", "MADDPG", "QMIX"])