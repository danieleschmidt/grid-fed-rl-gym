"""Safety constraints and monitoring for grid operations."""

import numpy as np
from typing import Dict, List, Tuple, Any
from .exceptions import SafetyLimitExceededError
import logging

logger = logging.getLogger(__name__)


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
        
        if np.any(emergency_high) or np.any(emergency_low):\n            violations['voltage_emergency'] = (\n                np.where(emergency_high)[0].tolist() + \n                np.where(emergency_low)[0].tolist()\n            )\n            violations['emergency_action_required'] = True\n            logger.critical(f\"Emergency voltage violations at buses: {violations['voltage_emergency']}\")\n        \n        # Frequency constraints\n        if frequency > self.frequency_limits[1]:\n            violations['frequency_high'] = True\n        elif frequency < self.frequency_limits[0]:\n            violations['frequency_low'] = True\n            \n        # Emergency frequency constraints\n        if frequency > self.emergency_frequency_limits[1] or frequency < self.emergency_frequency_limits[0]:\n            violations['frequency_emergency'] = True\n            violations['emergency_action_required'] = True\n            logger.critical(f\"Emergency frequency violation: {frequency} Hz\")\n        \n        # Line loading constraints\n        overloaded_lines = np.where(line_loadings > self.line_loading_limit)[0]\n        violations['line_overload'] = overloaded_lines.tolist()\n        \n        # Count total violations\n        violations['total_violations'] = (\n            len(violations['voltage_high']) +\n            len(violations['voltage_low']) +\n            len(violations['voltage_emergency']) +\n            int(violations['frequency_high']) +\n            int(violations['frequency_low']) +\n            int(violations['frequency_emergency']) +\n            len(violations['line_overload'])\n        )\n        \n        # Update violation tracking\n        if violations['total_violations'] > 0:\n            self.consecutive_violations += 1\n            self.violation_history.append((timestep, violations.copy()))\n        else:\n            self.consecutive_violations = 0\n            \n        # Trigger emergency mode if needed\n        if (violations['emergency_action_required'] or \n            self.consecutive_violations > 5 or\n            violations['total_violations'] > 10):\n            self.emergency_mode = True\n            violations['emergency_action_required'] = True\n            \n        return violations\n        \n    def get_corrective_actions(self, violations: Dict[str, Any]) -> Dict[str, Any]:\n        \"\"\"Suggest corrective actions for violations.\"\"\"\n        actions = {\n            'load_shedding': [],\n            'generation_adjustment': [],\n            'reactive_power_injection': [],\n            'emergency_shutdown': False\n        }\n        \n        # Voltage violations\n        if violations['voltage_low']:\n            actions['reactive_power_injection'].extend(violations['voltage_low'])\n            \n        if violations['voltage_high']:\n            actions['reactive_power_injection'].extend(violations['voltage_high'])\n            \n        # Emergency voltage violations\n        if violations['voltage_emergency']:\n            actions['load_shedding'].extend(violations['voltage_emergency'])\n            if len(violations['voltage_emergency']) > 3:\n                actions['emergency_shutdown'] = True\n        \n        # Frequency violations\n        if violations['frequency_low']:\n            actions['generation_adjustment'].append('increase')\n        elif violations['frequency_high']:\n            actions['generation_adjustment'].append('decrease')\n            \n        # Emergency frequency violations\n        if violations['frequency_emergency']:\n            actions['load_shedding'] = list(range(len(violations.get('bus_count', 10))))\n            if abs(60.0 - violations.get('frequency', 60.0)) > 3.0:\n                actions['emergency_shutdown'] = True\n        \n        # Line overloading\n        if violations['line_overload']:\n            actions['load_shedding'].extend(violations['line_overload'])\n            \n        return actions\n        \n    def reset(self):\n        \"\"\"Reset safety monitor state.\"\"\"\n        self.violation_history.clear()\n        self.consecutive_violations = 0\n        self.emergency_mode = False\n        \n    def get_safety_metrics(self) -> Dict[str, float]:\n        \"\"\"Get safety performance metrics.\"\"\"\n        if not self.violation_history:\n            return {\n                'violation_rate': 0.0,\n                'max_consecutive_violations': 0,\n                'emergency_events': 0,\n                'total_violations': 0\n            }\n            \n        total_violations = sum(v[1]['total_violations'] for v in self.violation_history)\n        emergency_events = sum(1 for v in self.violation_history if v[1]['emergency_action_required'])\n        \n        return {\n            'violation_rate': len(self.violation_history) / max(1, len(self.violation_history)),\n            'max_consecutive_violations': max(self.consecutive_violations, \n                                           max((v[1]['total_violations'] for v in self.violation_history), default=0)),\n            'emergency_events': emergency_events,\n            'total_violations': total_violations\n        }\n\n\nclass CircuitBreaker:\n    \"\"\"Circuit breaker for emergency protection.\"\"\"\n    \n    def __init__(self, trip_threshold: float = 2.0, reset_time: float = 30.0):\n        self.trip_threshold = trip_threshold\n        self.reset_time = reset_time\n        self.is_tripped = False\n        self.trip_time = 0\n        self.trip_count = 0\n        \n    def check_trip(self, current_loading: float, current_time: float) -> bool:\n        \"\"\"Check if circuit breaker should trip.\"\"\"\n        if self.is_tripped:\n            # Check if enough time has passed to reset\n            if current_time - self.trip_time > self.reset_time:\n                self.is_tripped = False\n                logger.info(f\"Circuit breaker reset after {self.reset_time}s\")\n            return True\n            \n        if current_loading > self.trip_threshold:\n            self.is_tripped = True\n            self.trip_time = current_time\n            self.trip_count += 1\n            logger.warning(f\"Circuit breaker tripped due to overload: {current_loading:.2f}\")\n            return True\n            \n        return False\n        \n    def manual_reset(self):\n        \"\"\"Manually reset circuit breaker.\"\"\"\n        self.is_tripped = False\n        logger.info(\"Circuit breaker manually reset\")\n        \n    def get_status(self) -> Dict[str, Any]:\n        \"\"\"Get circuit breaker status.\"\"\"\n        return {\n            'is_tripped': self.is_tripped,\n            'trip_count': self.trip_count,\n            'trip_time': self.trip_time\n        }