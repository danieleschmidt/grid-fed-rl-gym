"""Input validation and sanitization utilities."""

import numpy as np
from typing import Any, Dict, List, Optional, Union
from .exceptions import DataValidationError, InvalidActionError


def validate_action(action: np.ndarray, action_space) -> np.ndarray:
    """Validate and sanitize action input."""
    try:
        # Ensure it's a numpy array
        if not isinstance(action, np.ndarray):
            action = np.array(action)
        
        # Check shape
        if action.shape != action_space.shape:
            if action.size == action_space.shape[0]:
                action = action.reshape(action_space.shape)
            else:
                raise InvalidActionError(
                    f"Action shape {action.shape} doesn't match action space {action_space.shape}"
                )
        
        # Check for NaN or inf FIRST (before bounds checking)
        if np.any(np.isnan(action)):
            raise InvalidActionError("Action contains NaN values")
        if np.any(np.isinf(action)):
            raise InvalidActionError("Action contains infinite values")
        
        # Check for extremely large values that could cause numerical issues
        if np.any(np.abs(action) > 1e6):
            raise InvalidActionError("Action contains extremely large values")
        
        # Check bounds
        if hasattr(action_space, 'low') and hasattr(action_space, 'high'):
            if np.any(action < action_space.low) or np.any(action > action_space.high):
                # Clip to bounds with warning
                action = np.clip(action, action_space.low, action_space.high)
        
        return action
        
    except Exception as e:
        raise InvalidActionError(f"Action validation failed: {e}")


def validate_power_value(power: float, max_power: float = 1e9) -> float:
    """Validate power value."""
    if not isinstance(power, (int, float)):
        raise DataValidationError(f"Power must be numeric, got {type(power)}")
    
    if not np.isfinite(power):
        raise DataValidationError("Power value must be finite")
    
    if abs(power) > max_power:
        raise DataValidationError(f"Power value {power} exceeds maximum {max_power}")
    
    return float(power)


def validate_voltage(voltage: float, min_voltage: float = 0.1, max_voltage: float = 2.0) -> float:
    """Validate voltage value in per unit."""
    if not isinstance(voltage, (int, float)):
        raise DataValidationError(f"Voltage must be numeric, got {type(voltage)}")
    
    if not np.isfinite(voltage):
        raise DataValidationError("Voltage value must be finite")
    
    if voltage < min_voltage or voltage > max_voltage:
        raise DataValidationError(f"Voltage {voltage} outside valid range [{min_voltage}, {max_voltage}]")
    
    return float(voltage)


def validate_frequency(frequency: float, min_freq: float = 55.0, max_freq: float = 65.0) -> float:
    """Validate frequency value in Hz."""
    if not isinstance(frequency, (int, float)):
        raise DataValidationError(f"Frequency must be numeric, got {type(frequency)}")
    
    if not np.isfinite(frequency):
        raise DataValidationError("Frequency value must be finite")
    
    if frequency < min_freq or frequency > max_freq:
        raise DataValidationError(f"Frequency {frequency} outside valid range [{min_freq}, {max_freq}]")
    
    return float(frequency)


def validate_network_parameters(buses: List, lines: List, loads: List) -> None:
    """Validate network parameters."""
    if not buses:
        raise DataValidationError("Network must have at least one bus")
    
    if not lines and len(buses) > 1:
        raise DataValidationError("Multi-bus network must have transmission lines")
    
    # Check for slack bus
    slack_buses = [bus for bus in buses if bus.bus_type == "slack"]
    if len(slack_buses) != 1:
        raise DataValidationError(f"Network must have exactly one slack bus, found {len(slack_buses)}")
    
    # Validate line connections
    bus_ids = {bus.id for bus in buses}
    for line in lines:
        if line.from_bus not in bus_ids:
            raise DataValidationError(f"Line {line.id} connects to non-existent bus {line.from_bus}")
        if line.to_bus not in bus_ids:
            raise DataValidationError(f"Line {line.id} connects to non-existent bus {line.to_bus}")
        if line.from_bus == line.to_bus:
            raise DataValidationError(f"Line {line.id} connects bus to itself")
    
    # Validate load connections
    for load in loads:
        if load.bus not in bus_ids:
            raise DataValidationError(f"Load {load.id} connected to non-existent bus {load.bus}")


def sanitize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize configuration dictionary."""
    sanitized = {}
    
    # Define expected types and defaults
    expected_config = {
        'timestep': (float, 1.0),
        'episode_length': (int, 86400),
        'voltage_limits': (tuple, (0.95, 1.05)),
        'frequency_limits': (tuple, (59.5, 60.5)),
        'safety_penalty': (float, 100.0),
        'stochastic_loads': (bool, True),
        'weather_variation': (bool, True),
        'renewable_sources': (list, ['solar', 'wind'])
    }
    
    for key, (expected_type, default_value) in expected_config.items():
        if key in config:
            value = config[key]
            
            # Type checking
            if expected_type == tuple and isinstance(value, (list, tuple)):
                value = tuple(value)
            elif not isinstance(value, expected_type):
                try:
                    value = expected_type(value)
                except (ValueError, TypeError):
                    raise DataValidationError(
                        f"Config parameter '{key}' must be {expected_type.__name__}, got {type(value).__name__}"
                    )
            
            # Range validation
            if key == 'timestep' and (value <= 0 or value > 3600):
                raise DataValidationError("Timestep must be between 0 and 3600 seconds")
            elif key == 'episode_length' and (value <= 0 or value > 86400 * 7):
                raise DataValidationError("Episode length must be between 0 and 7 days")
            elif key == 'voltage_limits' and (len(value) != 2 or value[0] >= value[1]):
                raise DataValidationError("Voltage limits must be (min, max) with min < max")
            elif key == 'frequency_limits' and (len(value) != 2 or value[0] >= value[1]):
                raise DataValidationError("Frequency limits must be (min, max) with min < max")
            
            sanitized[key] = value
        else:
            sanitized[key] = default_value
    
    # Add any additional config values (with warning)
    for key, value in config.items():
        if key not in expected_config:
            sanitized[key] = value
    
    return sanitized


def validate_constraints(constraints: List, state_dim: int, action_dim: int) -> None:
    """Validate safety constraints."""
    if not isinstance(constraints, list):
        raise DataValidationError("Constraints must be a list")
    
    for i, constraint in enumerate(constraints):
        if not hasattr(constraint, 'constraint_function'):
            raise DataValidationError(f"Constraint {i} must have constraint_function")
        if not hasattr(constraint, 'name'):
            raise DataValidationError(f"Constraint {i} must have name")
        if not callable(constraint.constraint_function):
            raise DataValidationError(f"Constraint {i} constraint_function must be callable")


def sanitize_config(config: Union[Dict[str, Any], Any], required_fields: Optional[List[str]] = None) -> Dict[str, Any]:
    """Sanitize configuration dictionary with required fields validation."""
    if not isinstance(config, dict):
        if hasattr(config, '__dict__'):
            config = config.__dict__
        else:
            raise DataValidationError(f"Config must be dict or have __dict__, got {type(config)}")
    
    # Check required fields
    if required_fields:
        missing = [field for field in required_fields if field not in config]
        if missing:
            raise DataValidationError(f"Missing required config fields: {missing}")
    
    sanitized = {}
    
    # Define expected types and defaults  
    expected_config = {
        'timestep': (float, 1.0),
        'episode_length': (int, 86400),
        'voltage_limits': (tuple, (0.95, 1.05)),
        'frequency_limits': (tuple, (59.5, 60.5)),
        'safety_penalty': (float, 100.0),
        'stochastic_loads': (bool, True),
        'weather_variation': (bool, True),
        'renewable_sources': (list, ['solar', 'wind']),
        'num_clients': (int, 5),
        'rounds': (int, 100),
        'local_epochs': (int, 5),
        'batch_size': (int, 256),
        'learning_rate': (float, 1e-3),
        'privacy_budget': (float, 1.0)
    }
    
    for key, (expected_type, default_value) in expected_config.items():
        if key in config:
            value = config[key]
            
            # Type checking
            if expected_type == tuple and isinstance(value, (list, tuple)):
                value = tuple(value)
            elif not isinstance(value, expected_type):
                try:
                    value = expected_type(value)
                except (ValueError, TypeError):
                    raise DataValidationError(
                        f"Config parameter '{key}' must be {expected_type.__name__}, got {type(value).__name__}"
                    )
            
            # Range validation
            if key == 'timestep' and (value <= 0 or value > 3600):
                raise DataValidationError("Timestep must be between 0 and 3600 seconds")
            elif key == 'episode_length' and (value <= 0 or value > 86400 * 7):
                raise DataValidationError("Episode length must be between 0 and 7 days")
            elif key == 'voltage_limits' and (len(value) != 2 or value[0] >= value[1]):
                raise DataValidationError("Voltage limits must be (min, max) with min < max")
            elif key == 'frequency_limits' and (len(value) != 2 or value[0] >= value[1]):
                raise DataValidationError("Frequency limits must be (min, max) with min < max")
            elif key in ['num_clients', 'rounds', 'local_epochs'] and value <= 0:
                raise DataValidationError(f"{key} must be positive")
            elif key == 'learning_rate' and (value <= 0 or value > 1):
                raise DataValidationError("Learning rate must be between 0 and 1")
            
            sanitized[key] = value
        else:
            sanitized[key] = default_value
    
    # Add any additional config values
    for key, value in config.items():
        if key not in expected_config:
            sanitized[key] = value
    
    return sanitized


def validate_privacy_parameters(epsilon: float, delta: float) -> None:
    """Validate differential privacy parameters."""
    if not isinstance(epsilon, (int, float)) or epsilon <= 0:
        raise DataValidationError("Epsilon must be positive number")
    if not isinstance(delta, (int, float)) or delta <= 0 or delta >= 1:
        raise DataValidationError("Delta must be between 0 and 1")