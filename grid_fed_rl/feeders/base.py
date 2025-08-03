"""Base classes for network feeders and topologies."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from dataclasses import dataclass

from ..environments.base import Bus, Line, Load


@dataclass
class FeederParameters:
    """Standard parameters for feeder definition."""
    base_voltage: float  # kV
    base_power: float   # MVA
    frequency: float    # Hz


class BaseFeeder(ABC):
    """Abstract base class for all network feeders."""
    
    def __init__(self, name: str, parameters: Optional[FeederParameters] = None):
        self.name = name
        if parameters is None:
            self.parameters = FeederParameters(
                base_voltage=12.47,  # kV
                base_power=10.0,     # MVA
                frequency=60.0       # Hz
            )
        else:
            self.parameters = parameters
            
        self.buses: List[Bus] = []
        self.lines: List[Line] = []
        self.loads: List[Load] = []
        self.generators: Dict[str, Dict[str, Any]] = {}
        
    @abstractmethod
    def build_network(self) -> None:
        """Build the network topology."""
        pass
        
    def get_bus_by_id(self, bus_id: Union[int, str]) -> Optional[Bus]:
        """Get bus by ID."""
        for bus in self.buses:
            if bus.id == bus_id:
                return bus
        return None
        
    def get_line_by_id(self, line_id: Union[int, str]) -> Optional[Line]:
        """Get line by ID."""
        for line in self.lines:
            if line.id == line_id:
                return line
        return None
        
    def get_load_by_id(self, load_id: Union[int, str]) -> Optional[Load]:
        """Get load by ID."""
        for load in self.loads:
            if load.id == load_id:
                return load
        return None
        
    def add_bus(self, bus: Bus) -> None:
        """Add bus to feeder."""
        self.buses.append(bus)
        
    def add_line(self, line: Line) -> None:
        """Add line to feeder."""
        self.lines.append(line)
        
    def add_load(self, load: Load) -> None:
        """Add load to feeder."""
        self.loads.append(load)
        
    def add_generator(self, gen_id: str, gen_info: Dict[str, Any]) -> None:
        """Add generator to feeder."""
        self.generators[gen_id] = gen_info
        
    def get_adjacency_matrix(self) -> np.ndarray:
        """Get network adjacency matrix."""
        n_buses = len(self.buses)
        adj_matrix = np.zeros((n_buses, n_buses), dtype=int)
        
        bus_map = {bus.id: i for i, bus in enumerate(self.buses)}
        
        for line in self.lines:
            i = bus_map.get(line.from_bus)
            j = bus_map.get(line.to_bus)
            if i is not None and j is not None:
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1
                
        return adj_matrix
        
    def get_network_stats(self) -> Dict[str, Any]:
        """Get network statistics."""
        return {
            "name": self.name,
            "num_buses": len(self.buses),
            "num_lines": len(self.lines),
            "num_loads": len(self.loads),
            "num_generators": len(self.generators),
            "total_load": sum(load.base_power for load in self.loads),
            "total_generation_capacity": sum(
                gen.get("capacity", 0) for gen in self.generators.values()
            ),
            "base_voltage_kv": self.parameters.base_voltage,
            "base_power_mva": self.parameters.base_power
        }
        
    def validate_network(self) -> List[str]:
        """Validate network connectivity and parameters."""
        errors = []
        
        # Check for orphaned buses
        connected_buses = set()
        for line in self.lines:
            connected_buses.add(line.from_bus)
            connected_buses.add(line.to_bus)
            
        for bus in self.buses:
            if bus.id not in connected_buses and len(self.buses) > 1:
                errors.append(f"Bus {bus.id} is not connected to any line")
                
        # Check for missing buses in lines
        bus_ids = {bus.id for bus in self.buses}
        for line in self.lines:
            if line.from_bus not in bus_ids:
                errors.append(f"Line {line.id} references non-existent bus {line.from_bus}")
            if line.to_bus not in bus_ids:
                errors.append(f"Line {line.id} references non-existent bus {line.to_bus}")
                
        # Check for loads on non-existent buses
        for load in self.loads:
            if load.bus not in bus_ids:
                errors.append(f"Load {load.id} references non-existent bus {load.bus}")
                
        # Check for at least one slack bus
        slack_buses = [bus for bus in self.buses if bus.bus_type == "slack"]
        if len(slack_buses) == 0:
            errors.append("No slack bus found - at least one bus must be slack type")
        elif len(slack_buses) > 1:
            errors.append(f"Multiple slack buses found: {[bus.id for bus in slack_buses]}")
            
        return errors


class CustomFeeder(BaseFeeder):
    """Custom feeder for user-defined networks."""
    
    def __init__(self, name: str = "Custom", parameters: Optional[FeederParameters] = None):
        super().__init__(name, parameters)
        
    def build_network(self) -> None:
        """Build empty network - components added manually."""
        pass
        
    def from_dict(self, network_dict: Dict[str, Any]) -> None:
        """Build network from dictionary specification."""
        # Clear existing components
        self.buses.clear()
        self.lines.clear()
        self.loads.clear()
        self.generators.clear()
        
        # Add buses
        for bus_data in network_dict.get("buses", []):
            bus = Bus(
                id=bus_data["id"],
                voltage_level=bus_data.get("voltage_level", self.parameters.base_voltage * 1000),
                bus_type=bus_data.get("type", "pq"),
                base_voltage=bus_data.get("base_voltage", 1.0)
            )
            self.add_bus(bus)
            
        # Add lines
        for line_data in network_dict.get("lines", []):
            line = Line(
                id=line_data["id"],
                from_bus=line_data["from_bus"],
                to_bus=line_data["to_bus"],
                resistance=line_data["resistance"],
                reactance=line_data["reactance"],
                rating=line_data.get("rating", 1e6)
            )
            self.add_line(line)
            
        # Add loads
        for load_data in network_dict.get("loads", []):
            load = Load(
                id=load_data["id"],
                bus=load_data["bus"],
                base_power=load_data["power"],
                power_factor=load_data.get("power_factor", 0.95)
            )
            self.add_load(load)
            
        # Add generators
        for gen_data in network_dict.get("generators", []):
            self.add_generator(gen_data["id"], gen_data)
            
    def to_dict(self) -> Dict[str, Any]:
        """Export network to dictionary format."""
        return {
            "name": self.name,
            "parameters": {
                "base_voltage": self.parameters.base_voltage,
                "base_power": self.parameters.base_power,
                "frequency": self.parameters.frequency
            },
            "buses": [
                {
                    "id": bus.id,
                    "voltage_level": bus.voltage_level,
                    "type": bus.bus_type,
                    "base_voltage": bus.base_voltage
                }
                for bus in self.buses
            ],
            "lines": [
                {
                    "id": line.id,
                    "from_bus": line.from_bus,
                    "to_bus": line.to_bus,
                    "resistance": line.resistance,
                    "reactance": line.reactance,
                    "rating": line.rating
                }
                for line in self.lines
            ],
            "loads": [
                {
                    "id": load.id,
                    "bus": load.bus,
                    "power": load.base_power,
                    "power_factor": load.power_factor
                }
                for load in self.loads
            ],
            "generators": list(self.generators.values())
        }


class SimpleRadialFeeder(BaseFeeder):
    """Simple radial feeder for testing."""
    
    def __init__(
        self,
        num_buses: int = 5,
        line_impedance: Tuple[float, float] = (0.01, 0.02),
        load_power: float = 1e6,
        name: str = "SimpleRadial"
    ):
        super().__init__(name)
        self.num_buses = num_buses
        self.line_impedance = line_impedance
        self.load_power = load_power
        self.build_network()
        
    def build_network(self) -> None:
        """Build simple radial network."""
        # Create buses
        for i in range(1, self.num_buses + 1):
            bus_type = "slack" if i == 1 else "pq"
            bus = Bus(
                id=i,
                voltage_level=self.parameters.base_voltage * 1000,
                bus_type=bus_type
            )
            self.add_bus(bus)
            
        # Create lines (radial topology)
        for i in range(1, self.num_buses):
            line = Line(
                id=f"line_{i}_{i+1}",
                from_bus=i,
                to_bus=i + 1,
                resistance=self.line_impedance[0],
                reactance=self.line_impedance[1],
                rating=5e6  # 5 MVA rating
            )
            self.add_line(line)
            
        # Add loads to all buses except slack
        for i in range(2, self.num_buses + 1):
            load = Load(
                id=f"load_{i}",
                bus=i,
                base_power=self.load_power,
                power_factor=0.95
            )
            self.add_load(load)