"""Comprehensive input validation and sanitization."""

import re
import json
from typing import Any, Dict, List, Optional, Union, Type, Tuple
from dataclasses import dataclass
from enum import Enum
import logging


class ValidationLevel(Enum):
    """Validation strictness levels."""
    LENIENT = "lenient"      # Basic validation, warnings only
    STRICT = "strict"        # Standard validation, errors on violations
    PARANOID = "paranoid"    # Maximum validation, strict type checking


@dataclass
class ValidationResult:
    """Result of input validation."""
    valid: bool
    value: Any
    errors: List[str]
    warnings: List[str]
    sanitized: bool = False


class GridInputValidator:
    """Comprehensive input validation for grid simulation parameters."""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STRICT):
        self.validation_level = validation_level
        self.logger = logging.getLogger(__name__)
        
        # Define validation schemas
        self._setup_schemas()
    
    def _setup_schemas(self):
        """Setup validation schemas for different input types."""
        
        self.voltage_limits = {
            "min": 0.1,   # 0.1 pu minimum
            "max": 2.0,   # 2.0 pu maximum
            "nominal_min": 0.90,  # Typical minimum
            "nominal_max": 1.10   # Typical maximum
        }
        
        self.frequency_limits = {
            "min": 45.0,  # Hz minimum
            "max": 65.0,  # Hz maximum
            "nominal_min": 59.0,  # Typical minimum
            "nominal_max": 61.0   # Typical maximum
        }
        
        self.power_limits = {
            "min": -1e9,   # -1 GW (generation)
            "max": 1e9,    # 1 GW (load)
            "small_threshold": 1e3,   # 1 kW
            "large_threshold": 1e8    # 100 MW
        }
        
        # String patterns
        self.patterns = {
            "bus_id": r"^[a-zA-Z0-9_-]{1,50}$",
            "line_id": r"^[a-zA-Z0-9_-]{1,50}$",
            "safe_filename": r"^[a-zA-Z0-9._-]{1,100}$",
            "version": r"^\d+\.\d+\.\d+(?:-[a-zA-Z0-9]+)?$"
        }
    
    def validate_numeric(
        self, 
        value: Any, 
        name: str,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        integer_only: bool = False
    ) -> ValidationResult:
        """Validate numeric values with range checking."""
        
        errors = []
        warnings = []
        sanitized = False
        
        # Type validation
        if not isinstance(value, (int, float)):
            try:
                if integer_only:
                    value = int(value)
                else:
                    value = float(value)
                sanitized = True
                warnings.append(f"{name}: Converted to {'int' if integer_only else 'float'}")
            except (ValueError, TypeError):
                errors.append(f"{name}: Must be numeric, got {type(value).__name__}")
                return ValidationResult(False, value, errors, warnings, sanitized)
        
        # Integer validation
        if integer_only and not isinstance(value, int):
            try:
                value = int(value)
                sanitized = True
                warnings.append(f"{name}: Converted to integer")
            except ValueError:
                errors.append(f"{name}: Must be integer")
                return ValidationResult(False, value, errors, warnings, sanitized)
        
        # NaN/Infinity checks
        if isinstance(value, float):
            import math
            if math.isnan(value):
                errors.append(f"{name}: NaN values not allowed")
                return ValidationResult(False, value, errors, warnings, sanitized)
            if math.isinf(value):
                errors.append(f"{name}: Infinite values not allowed")
                return ValidationResult(False, value, errors, warnings, sanitized)
        
        # Range validation
        if min_val is not None and value < min_val:
            if self.validation_level == ValidationLevel.PARANOID:
                errors.append(f"{name}: {value} below minimum {min_val}")
            else:
                warnings.append(f"{name}: {value} below recommended minimum {min_val}")
                
        if max_val is not None and value > max_val:
            if self.validation_level == ValidationLevel.PARANOID:
                errors.append(f"{name}: {value} above maximum {max_val}")
            else:
                warnings.append(f"{name}: {value} above recommended maximum {max_val}")
        
        return ValidationResult(len(errors) == 0, value, errors, warnings, sanitized)
    
    def validate_string(
        self, 
        value: Any, 
        name: str,
        pattern: Optional[str] = None,
        max_length: int = 1000,
        allow_empty: bool = False
    ) -> ValidationResult:
        """Validate string values with pattern matching."""
        
        errors = []
        warnings = []
        sanitized = False
        
        # Type conversion
        if not isinstance(value, str):
            try:
                value = str(value)
                sanitized = True
                warnings.append(f"{name}: Converted to string")
            except Exception:
                errors.append(f"{name}: Cannot convert to string")
                return ValidationResult(False, value, errors, warnings, sanitized)
        
        # Empty check
        if not allow_empty and len(value) == 0:
            errors.append(f"{name}: Empty string not allowed")
        
        # Length check
        if len(value) > max_length:
            if self.validation_level == ValidationLevel.LENIENT:
                value = value[:max_length]
                sanitized = True
                warnings.append(f"{name}: Truncated to {max_length} characters")
            else:
                errors.append(f"{name}: Exceeds maximum length {max_length}")
        
        # Pattern validation
        if pattern and not re.match(pattern, value):
            errors.append(f"{name}: Does not match required pattern")
        
        # Security checks
        dangerous_patterns = [
            r'<script.*?>',
            r'javascript:',
            r'on\w+\s*=',
            r'\.\./\.\./',
            r'[;\|&`]'
        ]
        
        for dangerous in dangerous_patterns:
            if re.search(dangerous, value, re.IGNORECASE):
                if self.validation_level == ValidationLevel.PARANOID:
                    errors.append(f"{name}: Contains potentially dangerous content")
                else:
                    warnings.append(f"{name}: Contains suspicious patterns")
        
        return ValidationResult(len(errors) == 0, value, errors, warnings, sanitized)
    
    def validate_voltage(self, value: Any, name: str = "voltage") -> ValidationResult:
        """Validate voltage values in per unit."""
        result = self.validate_numeric(
            value, name, 
            self.voltage_limits["min"], 
            self.voltage_limits["max"]
        )
        
        # Additional voltage-specific checks
        if result.valid and isinstance(result.value, (int, float)):
            if result.value < self.voltage_limits["nominal_min"]:
                result.warnings.append(f"{name}: Low voltage {result.value:.3f} pu")
            elif result.value > self.voltage_limits["nominal_max"]:
                result.warnings.append(f"{name}: High voltage {result.value:.3f} pu")
        
        return result
    
    def validate_frequency(self, value: Any, name: str = "frequency") -> ValidationResult:
        """Validate frequency values in Hz."""
        result = self.validate_numeric(
            value, name,
            self.frequency_limits["min"],
            self.frequency_limits["max"]
        )
        
        # Additional frequency-specific checks
        if result.valid and isinstance(result.value, (int, float)):
            if not (self.frequency_limits["nominal_min"] <= result.value <= self.frequency_limits["nominal_max"]):
                result.warnings.append(f"{name}: Off-nominal frequency {result.value:.2f} Hz")
        
        return result
    
    def validate_power(self, value: Any, name: str = "power") -> ValidationResult:
        """Validate power values in Watts."""
        result = self.validate_numeric(
            value, name,
            self.power_limits["min"],
            self.power_limits["max"]
        )
        
        # Additional power-specific checks
        if result.valid and isinstance(result.value, (int, float)):
            abs_power = abs(result.value)
            
            if abs_power < self.power_limits["small_threshold"]:
                result.warnings.append(f"{name}: Very small power {abs_power:.0f} W")
            elif abs_power > self.power_limits["large_threshold"]:
                result.warnings.append(f"{name}: Very large power {abs_power/1e6:.1f} MW")
        
        return result
    
    def validate_bus_config(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate bus configuration."""
        errors = []
        warnings = []
        sanitized_config = config.copy()
        sanitized = False
        
        required_fields = ["id", "voltage_level"]
        
        # Check required fields
        for field in required_fields:
            if field not in config:
                errors.append(f"Bus config missing required field: {field}")
        
        if len(errors) > 0:
            return ValidationResult(False, config, errors, warnings, sanitized)
        
        # Validate individual fields
        id_result = self.validate_string(
            config["id"], "bus_id", 
            self.patterns["bus_id"]
        )
        if not id_result.valid:
            errors.extend(id_result.errors)
        else:
            sanitized_config["id"] = id_result.value
            if id_result.sanitized:
                sanitized = True
        
        voltage_result = self.validate_numeric(
            config["voltage_level"], "voltage_level", 
            100, 1000000  # 100V to 1MV
        )
        if not voltage_result.valid:
            errors.extend(voltage_result.errors)
        else:
            sanitized_config["voltage_level"] = voltage_result.value
            if voltage_result.sanitized:
                sanitized = True
        
        # Validate bus type
        if "bus_type" in config:
            valid_types = ["slack", "pv", "pq"]
            if config["bus_type"] not in valid_types:
                errors.append(f"Invalid bus_type: {config['bus_type']}, must be one of {valid_types}")
        
        return ValidationResult(len(errors) == 0, sanitized_config, errors, warnings, sanitized)
    
    def validate_environment_config(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate complete environment configuration."""
        errors = []
        warnings = []
        sanitized_config = config.copy()
        sanitized = False
        
        # Validate timestep
        if "timestep" in config:
            timestep_result = self.validate_numeric(
                config["timestep"], "timestep", 0.001, 3600
            )
            if not timestep_result.valid:
                errors.extend(timestep_result.errors)
            else:
                sanitized_config["timestep"] = timestep_result.value
                if timestep_result.sanitized:
                    sanitized = True
        
        # Validate episode length
        if "episode_length" in config:
            episode_result = self.validate_numeric(
                config["episode_length"], "episode_length", 1, 1000000, integer_only=True
            )
            if not episode_result.valid:
                errors.extend(episode_result.errors)
            else:
                sanitized_config["episode_length"] = episode_result.value
                if episode_result.sanitized:
                    sanitized = True
        
        # Validate renewable sources
        if "renewable_sources" in config:
            if not isinstance(config["renewable_sources"], list):
                errors.append("renewable_sources must be a list")
            else:
                valid_sources = ["solar", "wind", "hydro", "battery"]
                invalid_sources = [s for s in config["renewable_sources"] if s not in valid_sources]
                if invalid_sources:
                    warnings.append(f"Unknown renewable sources: {invalid_sources}")
        
        # Validate voltage limits
        if "voltage_limits" in config:
            if isinstance(config["voltage_limits"], (list, tuple)) and len(config["voltage_limits"]) == 2:
                low, high = config["voltage_limits"]
                
                low_result = self.validate_voltage(low, "voltage_limit_low")
                high_result = self.validate_voltage(high, "voltage_limit_high") 
                
                if not (low_result.valid and high_result.valid):
                    errors.extend(low_result.errors + high_result.errors)
                elif low_result.value >= high_result.value:
                    errors.append("voltage_limits: low limit must be less than high limit")
                else:
                    sanitized_config["voltage_limits"] = (low_result.value, high_result.value)
                    if low_result.sanitized or high_result.sanitized:
                        sanitized = True
            else:
                errors.append("voltage_limits must be a 2-element tuple/list")
        
        return ValidationResult(len(errors) == 0, sanitized_config, errors, warnings, sanitized)
    
    def sanitize_dict(self, data: Dict[str, Any], max_depth: int = 10) -> ValidationResult:
        """Sanitize dictionary data recursively."""
        if max_depth <= 0:
            return ValidationResult(False, data, ["Maximum recursion depth exceeded"], [], False)
        
        sanitized = {}
        errors = []
        warnings = []
        has_changes = False
        
        for key, value in data.items():
            # Sanitize key
            key_result = self.validate_string(key, f"key_{key}", max_length=100)
            if not key_result.valid:
                warnings.extend(key_result.warnings)
                continue  # Skip invalid keys
                
            clean_key = key_result.value
            if key_result.sanitized:
                has_changes = True
            
            # Sanitize value based on type
            if isinstance(value, dict):
                value_result = self.sanitize_dict(value, max_depth - 1)
                if value_result.valid:
                    sanitized[clean_key] = value_result.value
                    if value_result.sanitized:
                        has_changes = True
                else:
                    warnings.append(f"Skipped invalid nested dict at key {clean_key}")
                    
            elif isinstance(value, (list, tuple)):
                # Sanitize lists (basic validation)
                if len(value) > 1000:
                    warnings.append(f"Large list at key {clean_key} (length: {len(value)})")
                sanitized[clean_key] = value
                
            elif isinstance(value, str):
                str_result = self.validate_string(value, f"value_{clean_key}")
                sanitized[clean_key] = str_result.value
                if str_result.sanitized:
                    has_changes = True
                warnings.extend(str_result.warnings)
                
            else:
                # Keep other types as-is
                sanitized[clean_key] = value
        
        return ValidationResult(True, sanitized, errors, warnings, has_changes)


def validate_grid_input(data: Any, validation_level: str = "strict") -> Dict[str, Any]:
    """Standalone function for grid input validation."""
    
    try:
        level = ValidationLevel(validation_level.lower())
    except ValueError:
        level = ValidationLevel.STRICT
    
    validator = GridInputValidator(level)
    
    if isinstance(data, dict):
        result = validator.sanitize_dict(data)
    else:
        result = ValidationResult(False, data, ["Input must be a dictionary"], [], False)
    
    return {
        "valid": result.valid,
        "data": result.value,
        "errors": result.errors,
        "warnings": result.warnings,
        "sanitized": result.sanitized
    }


if __name__ == "__main__":
    # Test validation
    test_data = {
        "timestep": "1.5",
        "voltage_limits": [0.95, 1.05],
        "renewable_sources": ["solar", "wind"],
        "episode_length": 86400,
        "malicious_script": "<script>alert('test')</script>"
    }
    
    result = validate_grid_input(test_data, "strict")
    print("Validation Result:")
    print(json.dumps(result, indent=2))