"""
grid-fed-rl-gym: Federated Reinforcement Learning for Power Grid Control

Multiple grid operators train RL agents locally on their own grid segments,
then federate their policies via FedAvg to improve collective grid stability.
"""

from .env import GridEnv
from .policy import PolicyNetwork
from .agent import LocalRLAgent
from .federated import FederatedGridCoordinator
from .simulator import GridSimulator

__version__ = "1.0.0"
__all__ = [
    "GridEnv",
    "PolicyNetwork",
    "LocalRLAgent",
    "FederatedGridCoordinator",
    "GridSimulator",
]
