"""Configuration validation and schema checking."""

import json
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of configuration validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    
    def add_error(self, message: str) -> None:
        self.errors.append(message)
        self.is_valid = False
        
    def add_warning(self, message: str) -> None:
        self.warnings.append(message)


class ConfigValidator:
    """Validate configuration dictionaries against schemas."""
    
    def __init__(self):
        self.schemas = {
            'environment': self._get_environment_schema(),
            'feeder': self._get_feeder_schema(),
            'training': self._get_training_schema()
        }
        
    def validate_environment_config(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate environment configuration."""
        result = ValidationResult(True, [], [])
        
        # Required fields
        required_fields = ['timestep', 'episode_length']
        for field in required_fields:
            if field not in config:
                result.add_error(f"Missing required field: {field}")
        
        # Timestep validation
        timestep = config.get('timestep', 1.0)
        if not isinstance(timestep, (int, float)) or timestep <= 0:
            result.add_error("timestep must be a positive number")
        elif timestep > 60.0:
            result.add_warning("Large timestep may affect simulation accuracy")
            
        # Episode length validation
        episode_length = config.get('episode_length', 86400)
        if not isinstance(episode_length, int) or episode_length <= 0:
            result.add_error("episode_length must be a positive integer")
        elif episode_length > 1000000:
            result.add_warning("Very long episodes may affect performance")
        
        # Voltage limits
        voltage_limits = config.get('voltage_limits', (0.95, 1.05))
        if not isinstance(voltage_limits, (tuple, list)) or len(voltage_limits) != 2:
            result.add_error("voltage_limits must be a tuple/list of 2 values")
        else:
            v_min, v_max = voltage_limits
            if not (0.5 <= v_min < v_max <= 1.5):
                result.add_error("Invalid voltage limits - must be in range [0.5, 1.5] with min < max")
                
        # Frequency limits
        frequency_limits = config.get('frequency_limits', (59.5, 60.5))
        if not isinstance(frequency_limits, (tuple, list)) or len(frequency_limits) != 2:
            result.add_error("frequency_limits must be a tuple/list of 2 values")
        else:
            f_min, f_max = frequency_limits
            if not (50.0 <= f_min < f_max <= 70.0):
                result.add_error("Invalid frequency limits - must be in range [50, 70] Hz")
        
        # Safety penalty
        safety_penalty = config.get('safety_penalty', 100.0)
        if not isinstance(safety_penalty, (int, float)) or safety_penalty < 0:
            result.add_error("safety_penalty must be a non-negative number")
        elif safety_penalty == 0:
            result.add_warning("Zero safety penalty may lead to unsafe operation")
            
        return result
        
    def validate_feeder_config(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate feeder configuration."""
        result = ValidationResult(True, [], [])
        
        # Feeder type
        feeder_type = config.get('type', 'ieee13')
        valid_types = ['ieee13', 'ieee34', 'ieee123', 'simple', 'custom']
        if feeder_type not in valid_types:
            result.add_error(f"Invalid feeder type: {feeder_type}. Must be one of {valid_types}")
            
        # Base voltage
        base_voltage = config.get('base_voltage', 12.47)
        if not isinstance(base_voltage, (int, float)) or base_voltage <= 0:
            result.add_error("base_voltage must be a positive number")
        elif base_voltage > 500:
            result.add_warning("Very high base voltage - ensure units are correct (kV)")
            
        # Base power
        base_power = config.get('base_power', 10.0)
        if not isinstance(base_power, (int, float)) or base_power <= 0:
            result.add_error("base_power must be a positive number")
        elif base_power > 1000:
            result.add_warning("Very high base power - ensure units are correct (MVA)")
            
        # Custom feeder validation
        if feeder_type == 'custom':
            result = self._validate_custom_feeder(config, result)
            
        return result
        
    def _validate_custom_feeder(self, config: Dict[str, Any], result: ValidationResult) -> ValidationResult:
        """Validate custom feeder configuration."""
        
        # Required sections
        required_sections = ['buses', 'lines']
        for section in required_sections:
            if section not in config:
                result.add_error(f"Custom feeder missing required section: {section}")
                
        # Validate buses
        buses = config.get('buses', [])
        if not isinstance(buses, list) or len(buses) == 0:
            result.add_error("Custom feeder must have at least one bus")
        else:
            bus_ids = set()
            slack_buses = 0
            
            for i, bus in enumerate(buses):
                if not isinstance(bus, dict):
                    result.add_error(f"Bus {i} must be a dictionary")
                    continue
                    
                # Bus ID
                bus_id = bus.get('id')
                if bus_id is None:
                    result.add_error(f"Bus {i} missing required 'id' field")
                elif bus_id in bus_ids:
                    result.add_error(f"Duplicate bus ID: {bus_id}")
                else:
                    bus_ids.add(bus_id)
                    
                # Bus type
                bus_type = bus.get('type', 'pq')
                if bus_type not in ['slack', 'pv', 'pq']:
                    result.add_error(f"Bus {bus_id} has invalid type: {bus_type}")
                elif bus_type == 'slack':
                    slack_buses += 1
                    
                # Voltage level
                voltage_level = bus.get('voltage_level')
                if voltage_level is not None and (not isinstance(voltage_level, (int, float)) or voltage_level <= 0):
                    result.add_error(f"Bus {bus_id} has invalid voltage_level")
                    
            # Check slack bus count
            if slack_buses == 0:
                result.add_error("Custom feeder must have exactly one slack bus")
            elif slack_buses > 1:
                result.add_error(f"Custom feeder has {slack_buses} slack buses - only one allowed")
                
        # Validate lines
        lines = config.get('lines', [])
        if isinstance(lines, list):
            for i, line in enumerate(lines):
                if not isinstance(line, dict):
                    result.add_error(f"Line {i} must be a dictionary")
                    continue
                    
                # Required fields
                required_line_fields = ['from_bus', 'to_bus', 'resistance', 'reactance']
                for field in required_line_fields:
                    if field not in line:
                        result.add_error(f"Line {i} missing required field: {field}")
                        
                # Impedance values
                for param in ['resistance', 'reactance']:
                    value = line.get(param)
                    if value is not None and (not isinstance(value, (int, float)) or value < 0):
                        result.add_error(f"Line {i} has invalid {param}: {value}")
                        
                # Rating
                rating = line.get('rating')
                if rating is not None and (not isinstance(rating, (int, float)) or rating <= 0):
                    result.add_error(f"Line {i} has invalid rating: {rating}")
                    
        return result
        
    def validate_training_config(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate training configuration."""
        result = ValidationResult(True, [], [])
        
        # Algorithm
        algorithm = config.get('algorithm')
        if algorithm is not None:
            valid_algorithms = ['cql', 'iql', 'awr', 'ppo', 'sac']
            if algorithm.lower() not in valid_algorithms:
                result.add_warning(f"Unknown algorithm: {algorithm}")
                
        # Learning rate
        lr = config.get('learning_rate', 3e-4)
        if not isinstance(lr, (int, float)) or lr <= 0 or lr > 1:
            result.add_error("learning_rate must be between 0 and 1")
        elif lr > 0.01:
            result.add_warning("High learning rate may cause training instability")
            
        # Batch size
        batch_size = config.get('batch_size', 256)
        if not isinstance(batch_size, int) or batch_size <= 0:
            result.add_error("batch_size must be a positive integer")
        elif batch_size < 32:
            result.add_warning("Small batch size may cause training instability")
        elif batch_size > 2048:
            result.add_warning("Large batch size may require significant memory")
            
        # Training steps
        training_steps = config.get('training_steps', 100000)
        if not isinstance(training_steps, int) or training_steps <= 0:
            result.add_error("training_steps must be a positive integer")
        elif training_steps > 10000000:
            result.add_warning("Very long training may be inefficient")
            
        return result
        
    def _get_environment_schema(self) -> Dict[str, Any]:
        """Get environment configuration schema."""
        return {
            "type": "object",
            "properties": {
                "timestep": {"type": "number", "minimum": 0.001, "maximum": 60.0},
                "episode_length": {"type": "integer", "minimum": 1},
                "voltage_limits": {
                    "type": "array", 
                    "items": {"type": "number"},
                    "minItems": 2,
                    "maxItems": 2
                },
                "frequency_limits": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 2, 
                    "maxItems": 2
                },
                "safety_penalty": {"type": "number", "minimum": 0}
            },
            "required": ["timestep", "episode_length"]
        }
        
    def _get_feeder_schema(self) -> Dict[str, Any]:
        """Get feeder configuration schema."""
        return {
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": ["ieee13", "ieee34", "ieee123", "simple", "custom"]},
                "base_voltage": {"type": "number", "minimum": 0.1},
                "base_power": {"type": "number", "minimum": 0.1}
            }
        }
        
    def _get_training_schema(self) -> Dict[str, Any]:
        """Get training configuration schema."""
        return {
            "type": "object",
            "properties": {
                "algorithm": {"type": "string"},
                "learning_rate": {"type": "number", "minimum": 0, "maximum": 1},
                "batch_size": {"type": "integer", "minimum": 1},
                "training_steps": {"type": "integer", "minimum": 1}
            }
        }


def validate_action_bounds(action: Union[np.ndarray, List, float], action_space) -> Tuple[bool, str]:
    """Validate action is within action space bounds."""
    
    # Convert to numpy array
    if not isinstance(action, np.ndarray):
        action = np.array([action] if np.isscalar(action) else action)
    
    # Check shape
    expected_shape = action_space.shape
    if action.shape != expected_shape:
        return False, f"Action shape {action.shape} doesn't match expected {expected_shape}"
    
    # Check bounds
    if hasattr(action_space, 'low') and hasattr(action_space, 'high'):
        low = np.atleast_1d(action_space.low)
        high = np.atleast_1d(action_space.high)
        action_1d = np.atleast_1d(action)
        
        if np.any(action_1d < low) or np.any(action_1d > high):
            violations = []
            for i, (a, l, h) in enumerate(zip(action_1d, low, high)):
                if a < l:
                    violations.append(f"action[{i}]={a:.3f} < low={l:.3f}")
                elif a > h:
                    violations.append(f"action[{i}]={a:.3f} > high={h:.3f}")
            return False, f"Action bounds violated: {'; '.join(violations)}"
    
    return True, "Action is valid"


def validate_network_connectivity(buses: List, lines: List) -> ValidationResult:
    """Validate network connectivity and topology."""
    result = ValidationResult(True, [], [])
    
    if len(buses) == 0:
        result.add_error("Network has no buses")
        return result
        
    if len(lines) == 0 and len(buses) > 1:
        result.add_error("Multi-bus network has no lines")
        return result
    
    # Build adjacency list
    bus_ids = {getattr(bus, 'id', i) for i, bus in enumerate(buses)}
    adjacency = {bus_id: set() for bus_id in bus_ids}
    
    for line in lines:
        from_bus = getattr(line, 'from_bus', None)
        to_bus = getattr(line, 'to_bus', None)
        
        if from_bus not in bus_ids:
            result.add_error(f"Line references non-existent from_bus: {from_bus}")
            continue
        if to_bus not in bus_ids:
            result.add_error(f"Line references non-existent to_bus: {to_bus}")
            continue
            
        adjacency[from_bus].add(to_bus)
        adjacency[to_bus].add(from_bus)
    
    # Check connectivity using BFS
    if len(buses) > 1:
        visited = set()
        queue = [next(iter(bus_ids))]  # Start from first bus
        
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            queue.extend(adjacency[current] - visited)
        
        if len(visited) < len(bus_ids):
            isolated_buses = bus_ids - visited
            result.add_error(f"Network has isolated buses: {isolated_buses}")
    
    return result


# Global validator instance
config_validator = ConfigValidator()