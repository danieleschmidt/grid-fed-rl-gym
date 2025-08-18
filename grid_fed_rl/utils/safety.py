"""Safety constraints and monitoring for grid operations."""

import numpy as np
from typing import Dict, List, Tuple, Any
from .exceptions import SafetyLimitExceededError
import logging

logger = logging.getLogger(__name__)


class ConstraintViolation:
    """Represents a constraint violation in the grid."""
    
    def __init__(self, violation_type: str, location: Any, value: float, limit: float):
        self.violation_type = violation_type
        self.location = location
        self.value = value
        self.limit = limit
        self.timestamp = None
        
    def __str__(self) -> str:
        return f"{self.violation_type} violation at {self.location}: {self.value} > {self.limit}"


class SafetyMonitor:
    """Safety monitoring system for grid operations."""
    
    def __init__(self):
        self.constraints = []
        self.violation_history = []
        
    def add_constraint(self, constraint):
        """Add a safety constraint to monitor."""
        self.constraints.append(constraint)
    
    def check_violations(self, state: Dict[str, Any]) -> List[ConstraintViolation]:
        """Check all constraints against current state."""
        violations = []
        for constraint in self.constraints:
            violation = constraint.check(state)
            if violation:
                violations.append(violation)
                self.violation_history.append(violation)
        return violations


class VoltageConstraint:
    """Voltage constraint checker."""
    
    def __init__(self, min_voltage: float = 0.95, max_voltage: float = 1.05):
        self.min_voltage = min_voltage
        self.max_voltage = max_voltage
    
    def check(self, state: Dict[str, Any]) -> ConstraintViolation:
        """Check voltage constraints."""
        voltage = state.get('voltage', 1.0)
        if voltage < self.min_voltage:
            return ConstraintViolation("voltage_low", "bus", voltage, self.min_voltage)
        elif voltage > self.max_voltage:
            return ConstraintViolation("voltage_high", "bus", voltage, self.max_voltage)
        return None


class ThermalConstraint:
    """Thermal constraint checker."""
    
    def __init__(self, max_current: float = 100.0):
        self.max_current = max_current
    
    def check(self, state: Dict[str, Any]) -> ConstraintViolation:
        """Check thermal constraints."""
        current = state.get('current', 0.0)
        if current > self.max_current:
            return ConstraintViolation("thermal", "line", current, self.max_current)
        return None


class SafetyChecker:
    """Safety constraint checker for grid operations."""
    
    def __init__(
        self,
        voltage_limits: Tuple[float, float] = (0.95, 1.05),
        frequency_limits: Tuple[float, float] = (59.5, 60.5),
        line_loading_limit: float = 1.0,
        thermal_limits: Dict[str, float] = None,
        rate_of_change_limits: Dict[str, float] = None
    ):
        self.voltage_limits = voltage_limits
        self.frequency_limits = frequency_limits
        self.line_loading_limit = line_loading_limit
        self.thermal_limits = thermal_limits or {'transformer': 100.0, 'generator': 150.0}
        self.rate_of_change_limits = rate_of_change_limits or {'voltage': 0.1, 'frequency': 0.5}
        self._previous_state = None
        self._violation_history = []
        
    def check_constraints(
        self,
        bus_voltages: np.ndarray,
        frequency: float,
        line_loadings: np.ndarray,
        thermal_data: Dict[str, float] = None,
        timestep: float = 1.0
    ) -> Dict[str, List[ConstraintViolation]]:
        """Check all safety constraints and return violations."""
        
        violations = {
            'voltage': [],
            'frequency': [],
            'line_loading': [],
            'thermal': [],
            'rate_of_change': []
        }
        
        # Check voltage constraints
        for i, voltage in enumerate(bus_voltages):
            if voltage < self.voltage_limits[0]:
                violations['voltage'].append(
                    ConstraintViolation('voltage_low', i, voltage, self.voltage_limits[0])
                )
            elif voltage > self.voltage_limits[1]:
                violations['voltage'].append(
                    ConstraintViolation('voltage_high', i, voltage, self.voltage_limits[1])
                )
                
        # Check frequency constraints
        if frequency < self.frequency_limits[0]:
            violations['frequency'].append(
                ConstraintViolation('frequency_low', 'system', frequency, self.frequency_limits[0])
            )
        elif frequency > self.frequency_limits[1]:
            violations['frequency'].append(
                ConstraintViolation('frequency_high', 'system', frequency, self.frequency_limits[1])
            )
            
        # Check line loading constraints
        for i, loading in enumerate(line_loadings):
            if loading > self.line_loading_limit:
                violations['line_loading'].append(
                    ConstraintViolation('line_overload', i, loading, self.line_loading_limit)
                )
        
        # Check thermal constraints
        if thermal_data:
            for component, temperature in thermal_data.items():
                component_type = component.split('_')[0]
                if component_type in self.thermal_limits:
                    if temperature > self.thermal_limits[component_type]:
                        violations['thermal'].append(
                            ConstraintViolation('thermal_overload', component, temperature, self.thermal_limits[component_type])
                        )
        
        # Check rate of change constraints
        if self._previous_state is not None:
            voltage_change = np.max(np.abs(bus_voltages - self._previous_state['voltages']) / timestep)
            if voltage_change > self.rate_of_change_limits['voltage']:
                violations['rate_of_change'].append(
                    ConstraintViolation('voltage_rate', 'system', voltage_change, self.rate_of_change_limits['voltage'])
                )
            
            freq_change = abs(frequency - self._previous_state['frequency']) / timestep
            if freq_change > self.rate_of_change_limits['frequency']:
                violations['rate_of_change'].append(
                    ConstraintViolation('frequency_rate', 'system', freq_change, self.rate_of_change_limits['frequency'])
                )
        
        # Store current state for next iteration
        self._previous_state = {
            'voltages': bus_voltages.copy(),
            'frequency': frequency
        }
        
        # Update violation history
        total_violations = sum(len(v) for v in violations.values())
        if total_violations > 0:
            self._violation_history.append({
                'timestamp': None,  # Will be set externally
                'violations': violations,
                'total': total_violations
            })
        
        return violations
        
    def is_safe(self, violations: Dict[str, List[ConstraintViolation]]) -> bool:
        """Check if system is in safe state."""
        return all(len(v) == 0 for v in violations.values())
    
    def get_violation_severity(self, violations: Dict[str, List[ConstraintViolation]]) -> str:
        """Get overall violation severity level."""
        thermal_violations = len(violations.get('thermal', []))
        rate_violations = len(violations.get('rate_of_change', []))
        total_violations = sum(len(v) for v in violations.values())
        
        if thermal_violations > 0 or rate_violations > 2:
            return 'critical'
        elif total_violations > 5:
            return 'high'
        elif total_violations > 2:
            return 'medium'
        elif total_violations > 0:
            return 'low'
        else:
            return 'safe'


class SafetyShield:
    """Advanced safety shield with predictive intervention."""
    
    def __init__(
        self,
        safety_checker: SafetyChecker,
        intervention_threshold: float = 0.95,
        prediction_horizon: int = 3,
        backup_controller = None
    ):
        self.safety_checker = safety_checker
        self.intervention_threshold = intervention_threshold
        self.prediction_horizon = prediction_horizon
        self.backup_controller = backup_controller
        self.intervention_count = 0
        self.intervention_history = []
        
    def should_intervene(
        self,
        current_state: Dict[str, Any],
        proposed_action: np.ndarray,
        confidence: float = 1.0
    ) -> Tuple[bool, str]:
        """Determine if safety intervention is needed."""
        
        # Check current violations
        violations = self.safety_checker.check_constraints(
            current_state['bus_voltages'],
            current_state['frequency'],
            current_state['line_loadings'],
            current_state.get('thermal_data'),
            current_state.get('timestep', 1.0)
        )
        
        severity = self.safety_checker.get_violation_severity(violations)
        
        # Immediate intervention for critical violations
        if severity == 'critical':
            return True, 'Critical violations detected'
        
        # Predictive intervention based on confidence
        if confidence < self.intervention_threshold and severity in ['high', 'medium']:
            return True, f'Low confidence ({confidence:.3f}) with {severity} violations'
        
        # Rate-based intervention
        total_violations = sum(len(v) for v in violations.values())
        if total_violations > 3 and confidence < 0.8:
            return True, f'Multiple violations ({total_violations}) with low confidence'
        
        return False, 'Safe to proceed'
    
    def get_safe_action(
        self,
        current_state: Dict[str, Any],
        proposed_action: np.ndarray,
        confidence: float = 1.0
    ) -> Tuple[np.ndarray, bool]:
        """Get safe action, potentially overriding proposed action."""
        
        should_intervene, reason = self.should_intervene(current_state, proposed_action, confidence)
        
        if should_intervene:
            self.intervention_count += 1
            logger.warning(f"Safety intervention #{self.intervention_count}: {reason}")
            
            # Record intervention
            self.intervention_history.append({
                'timestamp': None,  # Will be set externally
                'reason': reason,
                'original_action': proposed_action.copy(),
                'confidence': confidence
            })
            
            # Use backup controller if available
            if self.backup_controller is not None:
                safe_action = self.backup_controller.get_action(current_state)
                return safe_action, True
            else:
                # Conservative fallback: reduce action magnitude
                safe_action = proposed_action * 0.1
                return safe_action, True
        
        return proposed_action, False


class SafetyMonitor:
    """Monitor grid safety constraints and trigger protective actions."""
    
    def __init__(
        self,
        voltage_limits: Tuple[float, float] = (0.90, 1.10),
        frequency_limits: Tuple[float, float] = (59.0, 61.0),
        line_loading_limit: float = 1.0,
        emergency_voltage_limits: Tuple[float, float] = (0.80, 1.20),
        emergency_frequency_limits: Tuple[float, float] = (57.0, 63.0)
    ):
        self.voltage_limits = voltage_limits
        self.frequency_limits = frequency_limits
        self.line_loading_limit = line_loading_limit
        self.emergency_voltage_limits = emergency_voltage_limits
        self.emergency_frequency_limits = emergency_frequency_limits
        
        # Violation tracking
        self.violation_history = []
        self.consecutive_violations = 0
        self.emergency_mode = False
        
    def check_constraints(
        self,
        bus_voltages: np.ndarray,
        frequency: float,
        line_loadings: np.ndarray,
        timestep: int
    ) -> Dict[str, Any]:
        """Check all safety constraints and return violation information."""
        
        violations = {
            'voltage_high': [],
            'voltage_low': [],
            'voltage_emergency': [],
            'frequency_high': False,
            'frequency_low': False,
            'frequency_emergency': False,
            'line_overload': [],
            'total_violations': 0,
            'emergency_action_required': False
        }
        
        # Voltage constraints
        high_voltage_mask = bus_voltages > self.voltage_limits[1]
        low_voltage_mask = bus_voltages < self.voltage_limits[0]
        
        violations['voltage_high'] = np.where(high_voltage_mask)[0].tolist()
        violations['voltage_low'] = np.where(low_voltage_mask)[0].tolist()
        
        # Emergency voltage constraints
        emergency_high = bus_voltages > self.emergency_voltage_limits[1]
        emergency_low = bus_voltages < self.emergency_voltage_limits[0]
        
        if np.any(emergency_high) or np.any(emergency_low):
            violations['voltage_emergency'] = (
                np.where(emergency_high)[0].tolist() + 
                np.where(emergency_low)[0].tolist()
            )
            violations['emergency_action_required'] = True
            logger.critical(f"Emergency voltage violations at buses: {violations['voltage_emergency']}")
        
        # Frequency constraints
        if frequency > self.frequency_limits[1]:
            violations['frequency_high'] = True
        elif frequency < self.frequency_limits[0]:
            violations['frequency_low'] = True
            
        # Emergency frequency constraints
        if frequency > self.emergency_frequency_limits[1] or frequency < self.emergency_frequency_limits[0]:
            violations['frequency_emergency'] = True
            violations['emergency_action_required'] = True
            logger.critical(f"Emergency frequency violation: {frequency} Hz")
        
        # Line loading constraints
        overloaded_lines = np.where(line_loadings > self.line_loading_limit)[0]
        violations['line_overload'] = overloaded_lines.tolist()
        
        # Count total violations
        violations['total_violations'] = (
            len(violations['voltage_high']) +
            len(violations['voltage_low']) +
            len(violations['voltage_emergency']) +
            int(violations['frequency_high']) +
            int(violations['frequency_low']) +
            int(violations['frequency_emergency']) +
            len(violations['line_overload'])
        )
        
        # Update violation tracking
        if violations['total_violations'] > 0:
            self.consecutive_violations += 1
            self.violation_history.append((timestep, violations.copy()))
        else:
            self.consecutive_violations = 0
            
        # Trigger emergency mode if needed
        if (violations['emergency_action_required'] or 
            self.consecutive_violations > 5 or
            violations['total_violations'] > 10):
            self.emergency_mode = True
            violations['emergency_action_required'] = True
            
        return violations
        
    def get_corrective_actions(self, violations: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest corrective actions for violations."""
        actions = {
            'load_shedding': [],
            'generation_adjustment': [],
            'reactive_power_injection': [],
            'emergency_shutdown': False
        }
        
        # Voltage violations
        if violations['voltage_low']:
            actions['reactive_power_injection'].extend(violations['voltage_low'])
            
        if violations['voltage_high']:
            actions['reactive_power_injection'].extend(violations['voltage_high'])
            
        # Emergency voltage violations
        if violations['voltage_emergency']:
            actions['load_shedding'].extend(violations['voltage_emergency'])
            if len(violations['voltage_emergency']) > 3:
                actions['emergency_shutdown'] = True
        
        # Frequency violations
        if violations['frequency_low']:
            actions['generation_adjustment'].append('increase')
        elif violations['frequency_high']:
            actions['generation_adjustment'].append('decrease')
            
        # Emergency frequency violations
        if violations['frequency_emergency']:
            actions['load_shedding'] = list(range(len(violations.get('bus_count', 10))))
            if abs(60.0 - violations.get('frequency', 60.0)) > 3.0:
                actions['emergency_shutdown'] = True
        
        # Line overloading
        if violations['line_overload']:
            actions['load_shedding'].extend(violations['line_overload'])
            
        return actions
        
    def reset(self):
        """Reset safety monitor state."""
        self.violation_history.clear()
        self.consecutive_violations = 0
        self.emergency_mode = False
        
    def get_safety_metrics(self) -> Dict[str, float]:
        """Get safety performance metrics."""
        if not self.violation_history:
            return {
                'violation_rate': 0.0,
                'max_consecutive_violations': 0,
                'emergency_events': 0,
                'total_violations': 0
            }
            
        total_violations = sum(v[1]['total_violations'] for v in self.violation_history)
        emergency_events = sum(1 for v in self.violation_history if v[1]['emergency_action_required'])
        
        return {
            'violation_rate': len(self.violation_history) / max(1, len(self.violation_history)),
            'max_consecutive_violations': max(self.consecutive_violations, 
                                           max((v[1]['total_violations'] for v in self.violation_history), default=0)),
            'emergency_events': emergency_events,
            'total_violations': total_violations
        }


class CircuitBreaker:
    """Circuit breaker for emergency protection."""
    
    def __init__(self, trip_threshold: float = 2.0, reset_time: float = 30.0):
        self.trip_threshold = trip_threshold
        self.reset_time = reset_time
        self.is_tripped = False
        self.trip_time = 0
        self.trip_count = 0
        
    def check_trip(self, current_loading: float, current_time: float) -> bool:
        """Check if circuit breaker should trip."""
        if self.is_tripped:
            # Check if enough time has passed to reset
            if current_time - self.trip_time > self.reset_time:
                self.is_tripped = False
                logger.info(f"Circuit breaker reset after {self.reset_time}s")
            return True
            
        if current_loading > self.trip_threshold:
            self.is_tripped = True
            self.trip_time = current_time
            self.trip_count += 1
            logger.warning(f"Circuit breaker tripped due to overload: {current_loading:.2f}")
            return True
            
        return False
        
    def manual_reset(self):
        """Manually reset circuit breaker."""
        self.is_tripped = False
        logger.info("Circuit breaker manually reset")
        
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        return {
            'is_tripped': self.is_tripped,
            'trip_count': self.trip_count,
            'trip_time': self.trip_time
        }