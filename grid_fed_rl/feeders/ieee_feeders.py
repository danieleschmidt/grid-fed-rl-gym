"""IEEE standard test feeders implementation."""

from typing import Dict, List, Optional, Tuple, Any

# Minimal NumPy replacement for basic functionality
try:
    import numpy as np
except ImportError:
    # Minimal numpy-like functionality for basic operation
    class MinimalNumPy:
        ndarray = list
        def array(self, data, dtype=None):
            return list(data) if hasattr(data, '__iter__') else [data]
        def sqrt(self, x): import math; return math.sqrt(x)
        float32 = float
    np = MinimalNumPy()

from .base import BaseFeeder, FeederParameters
from ..environments.base import Bus, Line, Load


class IEEE13Bus(BaseFeeder):
    """IEEE 13-bus test feeder."""
    
    def __init__(self):
        parameters = FeederParameters(
            base_voltage=4.16,  # kV
            base_power=10.0,    # MVA
            frequency=60.0      # Hz
        )
        super().__init__("IEEE13Bus", parameters)
        self.build_network()
        
    def build_network(self) -> None:
        """Build IEEE 13-bus test feeder network."""
        # Create buses
        bus_data = [
            (650, "slack"),  # Substation/slack bus
            (632, "pq"), (633, "pq"), (634, "pq"),
            (645, "pq"), (646, "pq"), (671, "pq"),
            (680, "pq"), (684, "pq"), (611, "pq"),
            (652, "pq"), (692, "pq"), (675, "pq")
        ]
        
        for bus_id, bus_type in bus_data:
            bus = Bus(
                id=bus_id,
                voltage_level=self.parameters.base_voltage * 1000,
                bus_type=bus_type,
                base_voltage=1.0
            )
            self.add_bus(bus)
            
        # Line data: (from, to, length_ft, config)
        # Simplified impedances for this implementation
        line_data = [
            (650, 632, 2000, "601"),  # Overhead line
            (632, 633, 500, "602"),   # Overhead line  
            (632, 645, 500, "603"),   # Overhead line
            (632, 671, 2000, "601"),  # Overhead line
            (645, 646, 300, "603"),   # Overhead line
            (671, 680, 1000, "601"),  # Overhead line
            (671, 684, 300, "604"),   # Overhead line
            (633, 634, 0, "XFM1"),    # Transformer
            (684, 611, 300, "603"),   # Overhead line
            (684, 652, 800, "607"),   # Underground cable
            (671, 692, 0, "SWITCH"), # Switch
            (692, 675, 500, "606")    # Overhead line
        ]
        
        # Simplified line impedances (R + jX in ohms/mile, converted to pu)
        config_impedances = {
            "601": (0.3465, 1.0179),  # Overhead
            "602": (0.7526, 1.1814),  # Overhead
            "603": (1.3238, 1.3569),  # Overhead
            "604": (1.3238, 1.3569),  # Overhead
            "606": (0.7982, 0.4463),  # Overhead
            "607": (1.3425, 0.5124),  # Underground
            "XFM1": (0.0, 0.06),      # Transformer
            "SWITCH": (0.0001, 0.0001) # Switch
        }
        
        base_impedance = (self.parameters.base_voltage ** 2) / self.parameters.base_power
        
        for from_bus, to_bus, length_ft, config in line_data:
            r_ohm_mile, x_ohm_mile = config_impedances[config]
            length_miles = length_ft / 5280.0
            
            # Convert to per unit
            resistance_pu = (r_ohm_mile * length_miles) / base_impedance
            reactance_pu = (x_ohm_mile * length_miles) / base_impedance
            
            line = Line(
                id=f"line_{from_bus}_{to_bus}",
                from_bus=from_bus,
                to_bus=to_bus,
                resistance=resistance_pu,
                reactance=reactance_pu,
                rating=5e6  # 5 MVA rating
            )
            self.add_line(line)
            
        # Load data: (bus, phase, kW, kVAR)
        load_data = [
            (634, "ABC", 400, 290),    # Balanced load
            (645, "BC", 170, 125),     # Unbalanced load
            (646, "ABC", 230, 132),    # Balanced load
            (652, "A", 128, 86),       # Single phase
            (671, "ABC", 1155, 660),   # Balanced load
            (675, "ABC", 843, 462),    # Balanced load
            (692, "ABC", 170, 151),    # Balanced load
            (611, "C", 170, 80)        # Single phase
        ]
        
        for bus_id, phase, kw, kvar in load_data:
            # Convert to VA base
            power_mw = kw / 1000.0
            
            load = Load(
                id=f"load_{bus_id}",
                bus=bus_id,
                base_power=power_mw * 1e6,  # Convert to watts
                power_factor=kw / np.sqrt(kw**2 + kvar**2) if kvar != 0 else 0.95
            )
            self.add_load(load)
            
        # Add distributed generators (for demonstration)
        self.add_generator("solar_671", {
            "type": "solar",
            "bus": 671,
            "capacity": 500e3,  # 500 kW
            "efficiency": 0.18
        })
        
        self.add_generator("wind_675", {
            "type": "wind", 
            "bus": 675,
            "capacity": 1e6,  # 1 MW
            "cut_in_speed": 3.0,
            "rated_speed": 12.0,
            "cut_out_speed": 25.0
        })


class IEEE34Bus(BaseFeeder):
    """IEEE 34-bus test feeder."""
    
    def __init__(self):
        parameters = FeederParameters(
            base_voltage=24.9,  # kV
            base_power=10.0,    # MVA
            frequency=60.0      # Hz
        )
        super().__init__("IEEE34Bus", parameters)
        self.build_network()
        
    def build_network(self) -> None:
        """Build IEEE 34-bus test feeder network."""
        # Simplified 34-bus system (subset for demonstration)
        # In practice, would have complete 34-bus data
        
        # Create main buses
        bus_data = [
            (800, "slack"),  # Substation
            (802, "pq"), (806, "pq"), (808, "pq"), (810, "pq"),
            (812, "pq"), (814, "pq"), (850, "pq"), (816, "pq"),
            (818, "pq"), (820, "pq"), (822, "pq"), (824, "pq"),
            (826, "pq"), (828, "pq"), (830, "pq"), (854, "pq"),
            (856, "pq"), (858, "pq"), (864, "pq"), (834, "pq"),
            (860, "pq"), (836, "pq"), (840, "pq"), (842, "pq"),
            (844, "pq"), (846, "pq"), (848, "pq"), (832, "pq"),
            (888, "pq"), (890, "pq"), (838, "pq"), (862, "pq"),
            (868, "pq")
        ]
        
        for bus_id, bus_type in bus_data:
            bus = Bus(
                id=bus_id,
                voltage_level=self.parameters.base_voltage * 1000,
                bus_type=bus_type
            )
            self.add_bus(bus)
            
        # Simplified line connections (main trunk)
        line_connections = [
            (800, 802), (802, 806), (806, 808), (808, 810),
            (810, 812), (812, 814), (814, 850), (816, 818),
            (816, 824), (818, 820), (820, 822), (824, 826),
            (824, 828), (828, 830), (854, 856), (832, 858),
            (858, 864), (858, 834), (834, 860), (860, 836),
            (836, 840), (840, 842), (842, 844), (844, 846),
            (846, 848), (832, 888), (888, 890), (890, 838),
            (834, 862), (862, 838), (842, 868)
        ]
        
        base_impedance = (self.parameters.base_voltage ** 2) / self.parameters.base_power
        
        for i, (from_bus, to_bus) in enumerate(line_connections):
            # Typical distribution line impedances
            resistance_pu = 0.005 + 0.002 * np.random.random()
            reactance_pu = 0.01 + 0.005 * np.random.random()
            
            line = Line(
                id=f"line_{from_bus}_{to_bus}",
                from_bus=from_bus,
                to_bus=to_bus,
                resistance=resistance_pu,
                reactance=reactance_pu,
                rating=10e6  # 10 MVA rating
            )
            self.add_line(line)
            
        # Add representative loads
        load_buses = [806, 810, 820, 822, 826, 830, 854, 858, 864, 840, 844, 848, 890]
        for bus_id in load_buses:
            # Random load between 100-500 kW
            load_kw = 100 + 400 * np.random.random()
            
            load = Load(
                id=f"load_{bus_id}",
                bus=bus_id,
                base_power=load_kw * 1000,  # Convert to watts
                power_factor=0.95
            )
            self.add_load(load)
            
        # Add renewable generation
        self.add_generator("solar_farm_830", {
            "type": "solar",
            "bus": 830,
            "capacity": 2e6,  # 2 MW
            "efficiency": 0.20
        })


class IEEE123Bus(BaseFeeder):
    """IEEE 123-bus test feeder."""
    
    def __init__(self):
        parameters = FeederParameters(
            base_voltage=4.16,  # kV
            base_power=10.0,    # MVA
            frequency=60.0      # Hz
        )
        super().__init__("IEEE123Bus", parameters)
        self.build_network()
        
    def build_network(self) -> None:
        """Build IEEE 123-bus test feeder network."""
        # This is a simplified version - the full 123-bus system
        # would require extensive data tables
        
        # Create representative buses
        bus_ids = list(range(1, 124))  # 123 buses
        
        for bus_id in bus_ids:
            bus_type = "slack" if bus_id == 1 else "pq"
            bus = Bus(
                id=bus_id,
                voltage_level=self.parameters.base_voltage * 1000,
                bus_type=bus_type
            )
            self.add_bus(bus)
            
        # Create a realistic distribution network topology
        # Main feeder backbone
        backbone_buses = [1, 3, 7, 13, 18, 25, 35, 49, 64, 78, 97, 114]
        
        for i in range(len(backbone_buses) - 1):
            from_bus = backbone_buses[i]
            to_bus = backbone_buses[i + 1]
            
            line = Line(
                id=f"main_{from_bus}_{to_bus}",
                from_bus=from_bus,
                to_bus=to_bus,
                resistance=0.003 + 0.002 * np.random.random(),
                reactance=0.006 + 0.004 * np.random.random(),
                rating=15e6  # 15 MVA
            )
            self.add_line(line)
            
        # Add lateral feeders
        lateral_connections = []
        np.random.seed(42)  # For reproducibility
        
        for backbone_bus in backbone_buses[1:]:  # Skip slack bus
            # Each backbone bus has 2-5 lateral connections
            num_laterals = np.random.randint(2, 6)
            
            available_buses = [b for b in bus_ids if b not in backbone_buses and b > backbone_bus]
            if len(available_buses) >= num_laterals:
                lateral_buses = np.random.choice(available_buses, num_laterals, replace=False)
                
                for lateral_bus in lateral_buses:
                    lateral_connections.append((backbone_bus, lateral_bus))
                    
        # Create lateral lines
        for from_bus, to_bus in lateral_connections:
            line = Line(
                id=f"lateral_{from_bus}_{to_bus}",
                from_bus=from_bus,
                to_bus=to_bus,
                resistance=0.008 + 0.005 * np.random.random(),
                reactance=0.012 + 0.008 * np.random.random(),
                rating=5e6  # 5 MVA
            )
            self.add_line(line)
            
        # Add secondary connections (mesh some laterals)
        secondary_connections = []
        connected_buses = set([conn[1] for conn in lateral_connections])
        
        for _ in range(20):  # Add 20 secondary connections
            available = list(connected_buses)
            if len(available) >= 2:
                from_bus, to_bus = np.random.choice(available, 2, replace=False)
                if from_bus != to_bus:
                    secondary_connections.append((from_bus, to_bus))
                    
        for from_bus, to_bus in secondary_connections:
            line = Line(
                id=f"secondary_{from_bus}_{to_bus}",
                from_bus=from_bus,
                to_bus=to_bus,
                resistance=0.010 + 0.008 * np.random.random(),
                reactance=0.015 + 0.010 * np.random.random(),
                rating=3e6  # 3 MVA
            )
            self.add_line(line)
            
        # Add loads to most buses (except slack)
        load_probability = 0.7  # 70% of buses have loads
        np.random.seed(123)
        
        for bus_id in bus_ids[1:]:  # Skip slack bus
            if np.random.random() < load_probability:
                # Load varies from 10 kW to 200 kW
                load_kw = 10 + 190 * np.random.random()
                
                load = Load(
                    id=f"load_{bus_id}",
                    bus=bus_id,
                    base_power=load_kw * 1000,  # Convert to watts
                    power_factor=0.92 + 0.06 * np.random.random()  # 0.92-0.98
                )
                self.add_load(load)
                
        # Add distributed generation
        dg_buses = [25, 49, 78, 97, 114]  # Strategic locations
        
        for i, bus_id in enumerate(dg_buses):
            if i % 2 == 0:  # Solar
                self.add_generator(f"solar_{bus_id}", {
                    "type": "solar",
                    "bus": bus_id,
                    "capacity": (200 + 300 * np.random.random()) * 1000,  # 200-500 kW
                    "efficiency": 0.18 + 0.04 * np.random.random()  # 18-22%
                })
            else:  # Wind
                self.add_generator(f"wind_{bus_id}", {
                    "type": "wind",
                    "bus": bus_id,
                    "capacity": (500 + 1000 * np.random.random()) * 1000,  # 0.5-1.5 MW
                    "cut_in_speed": 3.0,
                    "rated_speed": 12.0,
                    "cut_out_speed": 25.0
                })
                
        # Add energy storage
        storage_buses = [35, 64, 97]
        for bus_id in storage_buses:
            self.add_generator(f"battery_{bus_id}", {
                "type": "battery",
                "bus": bus_id,
                "capacity_kwh": 500 + 500 * np.random.random(),  # 0.5-1 MWh
                "power_rating_kw": 250 + 250 * np.random.random(),  # 250-500 kW
                "efficiency": 0.90 + 0.05 * np.random.random()  # 90-95%
            })