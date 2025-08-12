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

# Novel research algorithms for federated RL in power systems
try:
    from .physics_informed import PIFRL, PIFRLClient, PhysicsConstraint
    from .multi_objective import MOFRL, Objective, ParetoSolution
    from .uncertainty_aware import UAFRL, UAFRLClient, UncertaintyMetrics
    from .graph_neural import GNFRL, GNFRLClient, GraphTopology, PowerSystemGraph
    from .continual_learning import ContinualFederatedRL, ContinualFederatedClient, Task
    research_algorithms_available = True
except ImportError as e:
    print(f"Warning: Research algorithms require additional dependencies: {e}")
    research_algorithms_available = False
    # Create placeholder classes
    class PIFRL: pass
    class PIFRLClient: pass
    class PhysicsConstraint: pass
    class MOFRL: pass
    class Objective: pass
    class ParetoSolution: pass
    class UAFRL: pass
    class UAFRLClient: pass
    class UncertaintyMetrics: pass
    class GNFRL: pass
    class GNFRLClient: pass
    class GraphTopology: pass
    class PowerSystemGraph: pass
    class ContinualFederatedRL: pass
    class ContinualFederatedClient: pass
    class Task: pass

__all__ = [
    "BaseAlgorithm"
]

if offline_available:
    __all__.extend(["CQL", "IQL", "AWR", "OfflineRLAlgorithm"])
    
__all__.extend(["SafeRL", "ConstrainedPolicyOptimization", "MADDPG", "QMIX"])

if research_algorithms_available:
    __all__.extend([
        # Physics-Informed Federated RL
        "PIFRL", "PIFRLClient", "PhysicsConstraint",
        # Multi-Objective Federated RL
        "MOFRL", "Objective", "ParetoSolution",
        # Uncertainty-Aware Federated RL
        "UAFRL", "UAFRLClient", "UncertaintyMetrics",
        # Graph Neural Federated RL
        "GNFRL", "GNFRLClient", "GraphTopology", "PowerSystemGraph",
        # Continual Federated RL
        "ContinualFederatedRL", "ContinualFederatedClient", "Task"
    ])
else:
    __all__.extend([
        "PIFRL", "PIFRLClient", "PhysicsConstraint",
        "MOFRL", "Objective", "ParetoSolution", 
        "UAFRL", "UAFRLClient", "UncertaintyMetrics",
        "GNFRL", "GNFRLClient", "GraphTopology", "PowerSystemGraph",
        "ContinualFederatedRL", "ContinualFederatedClient", "Task"
    ])