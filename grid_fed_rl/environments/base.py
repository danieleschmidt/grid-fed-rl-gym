"""Base classes for grid environments."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Space


class BaseGridEnvironment(gym.Env, ABC):
    """Base class for all grid simulation environments.
    
    Provides common functionality for power system environments including
    observation spaces, action spaces, and basic environment lifecycle.
    """
    
    def __init__(
        self,
        timestep: float = 1.0,
        episode_length: int = 86400,
        voltage_limits: Tuple[float, float] = (0.95, 1.05),
        frequency_limits: Tuple[float, float] = (59.5, 60.5),
        **kwargs
    ) -> None:
        """Initialize base grid environment.
        
        Args:
            timestep: Simulation timestep in seconds
            episode_length: Episode length in timesteps
            voltage_limits: Min/max voltage in per unit
            frequency_limits: Min/max frequency in Hz
        """
        super().__init__()
        
        self.timestep = timestep
        self.episode_length = episode_length
        self.voltage_limits = voltage_limits
        self.frequency_limits = frequency_limits
        
        self.current_step = 0
        self.episode_reward = 0.0
        self.constraint_violations = 0
        
        # Will be set by subclasses
        self.observation_space: Optional[Space] = None
        self.action_space: Optional[Space] = None
        
    @abstractmethod
    def reset(
        self, 
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state."""
        pass
        
    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one environment step."""
        pass
        
    @abstractmethod
    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render the environment."""
        pass
        
    def close(self) -> None:
        """Clean up environment resources."""
        pass
        
    def get_observation(self) -> np.ndarray:
        """Get current observation from grid state."""
        raise NotImplementedError
        
    def get_reward(self, action: np.ndarray, next_obs: np.ndarray) -> float:
        """Calculate reward for current state and action."""
        raise NotImplementedError
        
    def is_done(self) -> bool:
        """Check if episode is complete."""
        return self.current_step >= self.episode_length
        
    def check_constraints(self, state: Dict[str, Any]) -> Dict[str, bool]:
        """Check operational constraints.
        
        Args:
            state: Current grid state
            
        Returns:
            Dictionary of constraint violations
        """
        violations = {}
        
        # Voltage constraints
        if "bus_voltages" in state:
            voltages = state["bus_voltages"]
            violations["voltage_high"] = np.any(voltages > self.voltage_limits[1])
            violations["voltage_low"] = np.any(voltages < self.voltage_limits[0])
            
        # Frequency constraints
        if "frequency" in state:
            freq = state["frequency"]
            violations["frequency_high"] = freq > self.frequency_limits[1]
            violations["frequency_low"] = freq < self.frequency_limits[0]
            
        return violations
        
    def get_info(self) -> Dict[str, Any]:
        """Get additional environment information."""
        return {
            "current_step": self.current_step,
            "episode_reward": self.episode_reward,
            "constraint_violations": self.constraint_violations,
            "timestep": self.timestep
        }


class GridComponent(ABC):
    """Base class for grid components like buses, lines, transformers."""
    
    def __init__(self, id: Union[int, str], **kwargs) -> None:
        self.id = id
        self.parameters = kwargs
        
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Get component state."""
        pass
        
    @abstractmethod
    def update_state(self, **kwargs) -> None:
        """Update component state."""
        pass


class Bus(GridComponent):
    """Electrical bus component."""
    
    def __init__(
        self,
        id: Union[int, str],
        voltage_level: float,
        bus_type: str = "pq",
        base_voltage: float = 1.0,
        **kwargs
    ) -> None:
        super().__init__(id, **kwargs)
        self.voltage_level = voltage_level
        self.bus_type = bus_type  # slack, pv, pq
        self.base_voltage = base_voltage
        self.voltage_magnitude = 1.0
        self.voltage_angle = 0.0
        
    def get_state(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "voltage_magnitude": self.voltage_magnitude,
            "voltage_angle": self.voltage_angle,
            "bus_type": self.bus_type
        }
        
    def update_state(self, voltage_magnitude: float = None, voltage_angle: float = None) -> None:
        if voltage_magnitude is not None:
            self.voltage_magnitude = voltage_magnitude
        if voltage_angle is not None:
            self.voltage_angle = voltage_angle


class Line(GridComponent):
    """Transmission/distribution line component."""
    
    def __init__(
        self,
        id: Union[int, str],
        from_bus: Union[int, str],
        to_bus: Union[int, str],
        resistance: float,
        reactance: float,
        rating: float,
        **kwargs
    ) -> None:
        super().__init__(id, **kwargs)
        self.from_bus = from_bus
        self.to_bus = to_bus
        self.resistance = resistance
        self.reactance = reactance
        self.rating = rating
        self.power_flow = 0.0
        self.loading = 0.0
        
    def get_state(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "from_bus": self.from_bus,
            "to_bus": self.to_bus,
            "power_flow": self.power_flow,
            "loading": self.loading
        }
        
    def update_state(self, power_flow: float = None) -> None:
        if power_flow is not None:
            self.power_flow = power_flow
            self.loading = abs(power_flow) / self.rating if self.rating > 0 else 0.0


class Load(GridComponent):
    """Load component with time-varying consumption."""
    
    def __init__(
        self,
        id: Union[int, str],
        bus: Union[int, str],
        base_power: float,
        power_factor: float = 0.95,
        **kwargs
    ) -> None:
        super().__init__(id, **kwargs)
        self.bus = bus
        self.base_power = base_power
        self.power_factor = power_factor
        self.active_power = base_power
        self.reactive_power = base_power * np.tan(np.arccos(power_factor))
        
    def get_state(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "bus": self.bus,
            "active_power": self.active_power,
            "reactive_power": self.reactive_power
        }
        
    def update_state(self, multiplier: float = 1.0) -> None:
        self.active_power = self.base_power * multiplier
        self.reactive_power = self.active_power * np.tan(np.arccos(self.power_factor))