"""Robust validation and error handling for grid operations."""

import logging
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation strictness levels."""
    STRICT = "strict"
    NORMAL = "normal"
    PERMISSIVE = "permissive"


class SafetyLevel(Enum):
    """Safety criticality levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ValidationResult:
    """Result of validation check."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    severity: SafetyLevel
    suggested_action: Optional[str] = None


class RobustValidator:
    """Comprehensive validation for grid operations."""
    
    def __init__(self, level: ValidationLevel = ValidationLevel.NORMAL):
        self.level = level
        self.validation_history: List[ValidationResult] = []
        
    def validate_action(self, action: Any, action_space: Any) -> ValidationResult:
        """Validate RL action with comprehensive checks."""
        errors = []
        warnings = []
        
        try:
            # Type validation
            if not hasattr(action, '__iter__') and not isinstance(action, (int, float)):
                errors.append(f"Action must be numeric or iterable, got {type(action)}")
                
            # Convert to list for validation
            if hasattr(action, '__iter__'):
                action_list = list(action)
            else:
                action_list = [action]
                
            # Range validation
            if hasattr(action_space, 'low') and hasattr(action_space, 'high'):
                for i, val in enumerate(action_list):
                    if hasattr(action_space.low, '__iter__'):
                        low = action_space.low[i] if i < len(action_space.low) else -1
                        high = action_space.high[i] if i < len(action_space.high) else 1
                    else:
                        low, high = action_space.low, action_space.high
                        
                    if val < low:
                        if self.level == ValidationLevel.STRICT:
                            errors.append(f"Action[{i}] = {val} below minimum {low}")
                        else:
                            warnings.append(f"Action[{i}] = {val} below minimum {low}, will be clipped")
                            
                    if val > high:
                        if self.level == ValidationLevel.STRICT:
                            errors.append(f"Action[{i}] = {val} above maximum {high}")
                        else:
                            warnings.append(f"Action[{i}] = {val} above maximum {high}, will be clipped")
                            
            # NaN/Inf validation
            for i, val in enumerate(action_list):
                if str(val).lower() in ['nan', 'inf', '-inf']:
                    errors.append(f"Action[{i}] contains invalid value: {val}")
                    
            severity = SafetyLevel.CRITICAL if errors else SafetyLevel.LOW
            suggested_action = "Reject action and use safe default" if errors else "Proceed with action"
            
        except Exception as e:
            errors.append(f"Validation failed with exception: {e}")
            severity = SafetyLevel.CRITICAL
            suggested_action = "Reject action and investigate"
            
        result = ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            severity=severity,
            suggested_action=suggested_action
        )
        
        self.validation_history.append(result)
        return result
        
    def validate_grid_state(self, state: Dict[str, Any]) -> ValidationResult:
        """Validate grid state for safety and consistency."""
        errors = []
        warnings = []
        
        try:
            # Voltage validation
            if "bus_voltages" in state:
                voltages = state["bus_voltages"]
                for i, v in enumerate(voltages):
                    if v <= 0:
                        errors.append(f"Bus {i} voltage {v} is non-positive")
                    elif v < 0.8:
                        errors.append(f"Bus {i} voltage {v} critically low (<0.8 pu)")
                    elif v < 0.95:
                        warnings.append(f"Bus {i} voltage {v} below normal range")
                    elif v > 1.2:
                        errors.append(f"Bus {i} voltage {v} critically high (>1.2 pu)")
                    elif v > 1.05:
                        warnings.append(f"Bus {i} voltage {v} above normal range")
                        
            # Frequency validation
            if "frequency" in state:
                freq = state["frequency"]
                if freq <= 0:
                    errors.append(f"Frequency {freq} is non-positive")
                elif freq < 58.0:
                    errors.append(f"Frequency {freq} Hz critically low")
                elif freq < 59.5:
                    warnings.append(f"Frequency {freq} Hz below normal range")
                elif freq > 62.0:
                    errors.append(f"Frequency {freq} Hz critically high")
                elif freq > 60.5:
                    warnings.append(f"Frequency {freq} Hz above normal range")
                    
            # Line loading validation
            if "line_loadings" in state:
                loadings = state["line_loadings"]
                for i, loading in enumerate(loadings):
                    if loading < 0:
                        warnings.append(f"Line {i} negative loading {loading}")
                    elif loading > 1.2:
                        errors.append(f"Line {i} critically overloaded: {loading*100:.1f}%")
                    elif loading > 1.0:
                        errors.append(f"Line {i} overloaded: {loading*100:.1f}%")
                    elif loading > 0.9:
                        warnings.append(f"Line {i} heavily loaded: {loading*100:.1f}%")
                        
            severity = SafetyLevel.CRITICAL if errors else SafetyLevel.LOW
            suggested_action = "Emergency shutdown required" if any("critically" in e for e in errors) else "Monitor closely"
            
        except Exception as e:
            errors.append(f"State validation failed: {e}")
            severity = SafetyLevel.CRITICAL
            suggested_action = "System check required"
            
        result = ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            severity=severity,
            suggested_action=suggested_action
        )
        
        self.validation_history.append(result)
        return result
        
    def validate_power_flow_convergence(self, solution: Any) -> ValidationResult:
        """Validate power flow solution."""
        errors = []
        warnings = []
        
        try:
            if not hasattr(solution, 'converged'):
                errors.append("Power flow solution missing convergence status")
            elif not solution.converged:
                errors.append("Power flow did not converge")
                
            if hasattr(solution, 'iterations'):
                if solution.iterations > 50:
                    warnings.append(f"Power flow required {solution.iterations} iterations")
                    
            if hasattr(solution, 'losses') and solution.losses < 0:
                errors.append(f"Negative system losses: {solution.losses}")
                
            severity = SafetyLevel.HIGH if errors else SafetyLevel.LOW
            suggested_action = "Use backup solver" if errors else "Accept solution"
            
        except Exception as e:
            errors.append(f"Power flow validation failed: {e}")
            severity = SafetyLevel.CRITICAL
            suggested_action = "Check solver configuration"
            
        result = ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            severity=severity,
            suggested_action=suggested_action
        )
        
        self.validation_history.append(result)
        return result
        
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of validation history."""
        if not self.validation_history:
            return {"total_validations": 0}
            
        total = len(self.validation_history)
        valid = sum(1 for r in self.validation_history if r.is_valid)
        
        severity_counts = {}
        for level in SafetyLevel:
            severity_counts[level.value] = sum(1 for r in self.validation_history if r.severity == level)
            
        return {
            "total_validations": total,
            "valid_count": valid,
            "error_rate": (total - valid) / total if total > 0 else 0,
            "severity_distribution": severity_counts,
            "recent_errors": [r.errors for r in self.validation_history[-10:] if r.errors]
        }


class RobustErrorHandler:
    """Enhanced error handling with recovery strategies."""
    
    def __init__(self):
        self.error_count = 0
        self.recovery_attempts = 0
        self.critical_errors = []
        
    def handle_action_error(self, error: Exception, action: Any) -> Any:
        """Handle action-related errors with safe fallbacks."""
        self.error_count += 1
        logger.error(f"Action error: {error} for action {action}")
        
        try:
            # Return safe neutral action
            if hasattr(action, '__iter__'):
                return [0.0] * len(action)
            else:
                return 0.0
        except:
            return 0.0
            
    def handle_simulation_error(self, error: Exception, state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle simulation errors with state recovery."""
        self.error_count += 1
        self.recovery_attempts += 1
        
        logger.error(f"Simulation error: {error}")
        
        # Return safe default state
        safe_state = {
            "bus_voltages": [1.0] * state.get("num_buses", 3),
            "line_loadings": [0.0] * state.get("num_lines", 2),
            "frequency": 60.0
        }
        
        return safe_state
        
    def handle_critical_error(self, error: Exception, context: str):
        """Handle critical system errors."""
        self.critical_errors.append({
            "error": str(error),
            "context": context,
            "timestamp": str(__import__('datetime').datetime.now())
        })
        
        logger.critical(f"Critical error in {context}: {error}")
        
        # In a real system, this might trigger emergency shutdown
        if len(self.critical_errors) > 10:
            logger.critical("Too many critical errors - system requires manual intervention")
            
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error handling summary."""
        return {
            "total_errors": self.error_count,
            "recovery_attempts": self.recovery_attempts,
            "critical_errors": len(self.critical_errors),
            "recent_critical": self.critical_errors[-5:] if self.critical_errors else []
        }


# Global validator and error handler instances
global_validator = RobustValidator()
global_error_handler = RobustErrorHandler()