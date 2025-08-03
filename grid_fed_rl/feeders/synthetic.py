"""Synthetic network generators for testing and experimentation."""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass

from .base import BaseFeeder, FeederParameters
from ..environments.base import Bus, Line, Load


@dataclass
class NetworkConfig:
    """Configuration for synthetic network generation."""
    num_buses: int = 20
    connectivity: float = 0.3  # 0-1, higher = more connected
    load_probability: float = 0.7  # Probability a bus has load
    min_load_kw: float = 50
    max_load_kw: float = 500
    line_length_range: Tuple[float, float] = (0.1, 2.0)  # km
    dg_probability: float = 0.2  # Probability of distributed generation
    
    
class SyntheticFeeder(BaseFeeder):
    """Generate synthetic distribution feeders."""
    
    def __init__(
        self,
        config: Optional[NetworkConfig] = None,
        seed: Optional[int] = None,
        name: str = "Synthetic"
    ):
        super().__init__(name)
        self.config = config or NetworkConfig()
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
        self.build_network()
        
    def build_network(self) -> None:
        """Generate synthetic network topology."""
        # Create buses
        self._create_buses()
        
        # Create network topology
        self._create_topology()
        
        # Add loads
        self._add_loads()
        
        # Add distributed generation
        self._add_distributed_generation()
        
    def _create_buses(self) -> None:
        """Create buses for synthetic network."""
        for i in range(self.config.num_buses):
            bus_type = "slack" if i == 0 else "pq"
            
            bus = Bus(
                id=i + 1,
                voltage_level=self.parameters.base_voltage * 1000,
                bus_type=bus_type
            )
            self.add_bus(bus)
            
    def _create_topology(self) -> None:
        """Create network topology."""
        n = self.config.num_buses
        
        # Start with spanning tree to ensure connectivity
        self._create_spanning_tree()
        
        # Add additional connections based on connectivity parameter
        self._add_random_connections()
        
    def _create_spanning_tree(self) -> None:
        """Create spanning tree for basic connectivity."""
        connected = {1}  # Start with slack bus
        unconnected = set(range(2, self.config.num_buses + 1))
        
        line_id = 1
        
        while unconnected:
            # Pick random connected and unconnected buses
            from_bus = np.random.choice(list(connected))
            to_bus = np.random.choice(list(unconnected))
            
            # Create line
            line = self._create_random_line(line_id, from_bus, to_bus)
            self.add_line(line)
            
            # Update connected sets
            connected.add(to_bus)
            unconnected.remove(to_bus)
            
            line_id += 1
            
    def _add_random_connections(self) -> None:
        """Add random connections to increase network connectivity."""
        n = self.config.num_buses
        max_possible_lines = n * (n - 1) // 2
        current_lines = len(self.lines)
        
        target_lines = int(current_lines + 
                          self.config.connectivity * (max_possible_lines - current_lines))
        
        line_id = len(self.lines) + 1
        existing_connections = {(line.from_bus, line.to_bus) for line in self.lines}
        existing_connections.update({(line.to_bus, line.from_bus) for line in self.lines})
        
        attempts = 0
        while len(self.lines) < target_lines and attempts < 1000:
            from_bus = np.random.randint(1, n + 1)
            to_bus = np.random.randint(1, n + 1)
            
            if (from_bus != to_bus and 
                (from_bus, to_bus) not in existing_connections):
                
                line = self._create_random_line(line_id, from_bus, to_bus)
                self.add_line(line)
                
                existing_connections.add((from_bus, to_bus))
                existing_connections.add((to_bus, from_bus))
                
                line_id += 1
                
            attempts += 1
            
    def _create_random_line(self, line_id: int, from_bus: int, to_bus: int) -> Line:
        """Create line with random parameters."""
        # Random line length
        length_km = (self.config.line_length_range[0] + 
                    (self.config.line_length_range[1] - self.config.line_length_range[0]) * 
                    np.random.random())
        
        # Typical distribution line parameters (per km)
        r_per_km = 0.2 + 0.3 * np.random.random()  # 0.2-0.5 ohm/km
        x_per_km = 0.3 + 0.4 * np.random.random()  # 0.3-0.7 ohm/km
        
        # Convert to per unit
        base_impedance = (self.parameters.base_voltage ** 2) / self.parameters.base_power
        
        resistance_pu = (r_per_km * length_km) / base_impedance
        reactance_pu = (x_per_km * length_km) / base_impedance
        
        # Random line rating
        rating_mva = 2 + 8 * np.random.random()  # 2-10 MVA
        
        return Line(
            id=f"line_{line_id}",
            from_bus=from_bus,
            to_bus=to_bus,
            resistance=resistance_pu,
            reactance=reactance_pu,
            rating=rating_mva * 1e6
        )
        
    def _add_loads(self) -> None:
        """Add loads to buses."""
        for bus in self.buses[1:]:  # Skip slack bus
            if np.random.random() < self.config.load_probability:
                # Random load size
                load_kw = (self.config.min_load_kw + 
                          (self.config.max_load_kw - self.config.min_load_kw) * 
                          np.random.random())
                
                # Random power factor
                power_factor = 0.85 + 0.15 * np.random.random()  # 0.85-1.0
                
                load = Load(
                    id=f"load_{bus.id}",
                    bus=bus.id,
                    base_power=load_kw * 1000,  # Convert to watts
                    power_factor=power_factor
                )
                self.add_load(load)
                
    def _add_distributed_generation(self) -> None:
        """Add distributed generation to random buses."""
        candidate_buses = [bus.id for bus in self.buses[1:]]  # Exclude slack
        
        for bus_id in candidate_buses:
            if np.random.random() < self.config.dg_probability:
                # Random DG type
                dg_type = np.random.choice(["solar", "wind", "battery"])
                
                if dg_type == "solar":
                    capacity_kw = 100 + 400 * np.random.random()  # 100-500 kW
                    self.add_generator(f"solar_{bus_id}", {
                        "type": "solar",
                        "bus": bus_id,
                        "capacity": capacity_kw * 1000,
                        "efficiency": 0.15 + 0.10 * np.random.random()  # 15-25%
                    })
                elif dg_type == "wind":
                    capacity_kw = 500 + 1500 * np.random.random()  # 0.5-2 MW
                    self.add_generator(f"wind_{bus_id}", {
                        "type": "wind",
                        "bus": bus_id,
                        "capacity": capacity_kw * 1000,
                        "cut_in_speed": 2.5 + 1.0 * np.random.random(),
                        "rated_speed": 10 + 5 * np.random.random(),
                        "cut_out_speed": 20 + 10 * np.random.random()
                    })
                else:  # battery
                    capacity_kwh = 200 + 800 * np.random.random()  # 0.2-1 MWh
                    power_kw = capacity_kwh * 0.5  # C/2 rate
                    self.add_generator(f"battery_{bus_id}", {
                        "type": "battery",
                        "bus": bus_id,
                        "capacity_kwh": capacity_kwh,
                        "power_rating_kw": power_kw,
                        "efficiency": 0.85 + 0.10 * np.random.random()  # 85-95%
                    })


class RandomFeeder(SyntheticFeeder):
    """Completely random feeder for stress testing."""
    
    def __init__(self, seed: Optional[int] = None):
        # Random configuration
        config = NetworkConfig(
            num_buses=np.random.randint(10, 50),
            connectivity=np.random.uniform(0.1, 0.8),
            load_probability=np.random.uniform(0.4, 0.9),
            min_load_kw=np.random.uniform(10, 100),
            max_load_kw=np.random.uniform(200, 1000),
            dg_probability=np.random.uniform(0.1, 0.5)
        )
        
        super().__init__(config, seed, "Random")


class ScalableFeeder(SyntheticFeeder):
    """Scalable feeder that maintains realistic properties at different sizes."""
    
    def __init__(self, num_buses: int, seed: Optional[int] = None):
        # Scale properties with network size
        connectivity = max(0.1, min(0.6, 20.0 / num_buses))  # Sparser for larger networks
        load_prob = min(0.9, 0.5 + 0.01 * num_buses)  # More loads in larger networks
        dg_prob = min(0.4, 0.1 + 0.005 * num_buses)  # More DG in larger networks
        
        config = NetworkConfig(
            num_buses=num_buses,
            connectivity=connectivity,
            load_probability=load_prob,
            dg_probability=dg_prob,
            min_load_kw=20,
            max_load_kw=300,
            line_length_range=(0.05, 1.5)
        )
        
        super().__init__(config, seed, f"Scalable{num_buses}")


class MicrogridFeeder(SyntheticFeeder):
    """Microgrid-style feeder with high DG penetration."""
    
    def __init__(self, seed: Optional[int] = None):
        config = NetworkConfig(
            num_buses=15,
            connectivity=0.4,  # Well connected
            load_probability=0.8,  # Most buses have loads
            min_load_kw=20,
            max_load_kw=200,  # Smaller loads
            dg_probability=0.6,  # High DG penetration
            line_length_range=(0.05, 0.5)  # Shorter lines
        )
        
        super().__init__(config, seed, "Microgrid")
        
    def _add_distributed_generation(self) -> None:
        """Add DG with microgrid characteristics."""
        super()._add_distributed_generation()
        
        # Add central battery storage
        central_bus = np.random.choice([bus.id for bus in self.buses[1:]])
        self.add_generator(f"central_battery_{central_bus}", {
            "type": "battery",
            "bus": central_bus,
            "capacity_kwh": 1000,  # 1 MWh
            "power_rating_kw": 500,  # 500 kW
            "efficiency": 0.95
        })
        
        # Ensure solar on at least 3 buses
        solar_buses = np.random.choice(
            [bus.id for bus in self.buses[1:]], 
            size=min(3, len(self.buses) - 1), 
            replace=False
        )
        
        for bus_id in solar_buses:
            if f"solar_{bus_id}" not in self.generators:
                self.add_generator(f"solar_{bus_id}", {
                    "type": "solar",
                    "bus": bus_id,
                    "capacity": 150e3,  # 150 kW
                    "efficiency": 0.20
                })