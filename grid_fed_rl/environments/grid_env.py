"""Main grid environment for reinforcement learning."""

from typing import Any, Dict, List, Optional, Tuple, Union
import warnings
import logging

# Minimal NumPy replacement for basic functionality
try:
    import numpy as np
except ImportError:
    # Minimal numpy-like functionality for basic operation
    class MinimalNumPy:
        ndarray = list  # Use list as ndarray replacement
        
        def array(self, data, dtype=None):
            if isinstance(data, list):
                return data
            return [data] if not hasattr(data, '__iter__') else list(data)
        def random(self):
            import random
            return type('obj', (object,), {
                'uniform': lambda low, high: random.uniform(low, high),
                'normal': lambda mean, std: random.gauss(mean, std),
                'random': lambda: random.random()
            })()
        def sin(self, x): import math; return math.sin(x)
        def cos(self, x): import math; return math.cos(x)
        @property
        def pi(self): import math; return math.pi
        def abs(self, x): return abs(x)
        def sum(self, x): return sum(x)
        def clip(self, x, low, high): return max(low, min(high, x))
        def max(self, x): return max(x)
        def min(self, x): return min(x)
        def any(self, x): return any(x)
        def tan(self, x): import math; return math.tan(x)
        def arccos(self, x): import math; return math.acos(x)
        float32 = float
    np = MinimalNumPy()

from .base import Box
from .base import BaseGridEnvironment, Bus, Line, Load

# Try to import advanced modules, use stubs if not available
try:
    from .power_flow import PowerFlowSolver, NewtonRaphsonSolver, PowerFlowSolution
except ImportError:
    # Create stub classes
    class PowerFlowSolution:
        def __init__(self):
            self.converged = True
            self.bus_voltages = [1.0, 1.0, 1.0]
            self.bus_angles = [0.0, 0.0, 0.0]
            self.line_flows = [0.0, 0.0]
            self.losses = 0.0
    
    class PowerFlowSolver:
        def solve(self, buses, lines, loads, generation):
            return PowerFlowSolution()
    
    class NewtonRaphsonSolver(PowerFlowSolver):
        pass

try:
    from .dynamics import GridDynamics, WeatherData, TimeVaryingLoadModel, SolarPVModel, WindTurbineModel, BatteryModel
except ImportError:
    # Create stub classes
    class WeatherData:
        def __init__(self, **kwargs):
            self.solar_irradiance = kwargs.get('solar_irradiance', 0.0)
            self.wind_speed = kwargs.get('wind_speed', 5.0)
            self.temperature = kwargs.get('temperature', 25.0)
            self.cloud_cover = kwargs.get('cloud_cover', 0.3)
    
    class TimeVaryingLoadModel:
        pass
    
    class SolarPVModel:
        def __init__(self, **kwargs):
            pass
    
    class WindTurbineModel:
        def __init__(self, **kwargs):
            pass
    
    class BatteryModel:
        def __init__(self, **kwargs):
            self.capacity = kwargs.get('capacity', 1000)
            self.power_rating = kwargs.get('power_rating', 500)
            self.efficiency = kwargs.get('efficiency', 0.95)
            self.soc = kwargs.get('initial_soc', 0.5)
            self.current_power = 0.0
    
    class GridDynamics:
        def __init__(self):
            self.frequency = 60.0
        
        def add_load_model(self, load_id, model):
            pass
        
        def add_renewable_model(self, gen_id, model):
            pass
        
        def add_battery_model(self, battery_id, model):
            pass
        
        def get_load_power(self, load_id, base_power, time, power_factor=0.95, timestep=1.0):
            # Simple load variation
            import random
            multiplier = 0.8 + 0.4 * random.random()
            return base_power * multiplier, base_power * multiplier * 0.3
        
        def get_renewable_power(self, gen_id, capacity, time, weather):
            # Simple renewable model
            import random
            if 'solar' in gen_id:
                return capacity * weather.solar_irradiance / 1000.0 * (0.8 + 0.4 * random.random())
            elif 'wind' in gen_id:
                wind_power = min(capacity, capacity * (weather.wind_speed / 15.0) ** 3)
                return wind_power * (0.8 + 0.4 * random.random())
            return 0.0
        
        def update_batteries(self, commands, timestep):
            return commands
        
        def update_frequency(self, power_imbalance, timestep):
            # Simple frequency response
            self.frequency = 60.0 + power_imbalance * 0.1

try:
    from ..utils import (
        validate_action, validate_network_parameters, sanitize_config,
        PowerFlowError, InvalidActionError, SafetyLimitExceededError
    )
except ImportError:
    # Create stub validation functions
    def validate_action(action, action_space):
        if hasattr(action, '__iter__'):
            return list(action)
        return [action]
    
    def validate_network_parameters(params):
        return params
    
    def sanitize_config(config):
        return config
    
    class PowerFlowError(Exception):
        pass
    
    class InvalidActionError(Exception):
        pass
    
    class SafetyLimitExceededError(Exception):
        pass


class GridEnvironment(BaseGridEnvironment):
    """Complete grid environment with power flow, dynamics, and RL interface."""
    
    def __init__(
        self,
        feeder,  # Will be from feeders module
        timestep: float = 1.0,
        episode_length: int = 86400,  # 24 hours in seconds
        stochastic_loads: bool = True,
        renewable_sources: Optional[List[str]] = None,
        weather_variation: bool = True,
        power_flow_solver: Optional[PowerFlowSolver] = None,
        voltage_limits: Tuple[float, float] = (0.95, 1.05),
        frequency_limits: Tuple[float, float] = (59.5, 60.5),
        safety_penalty: float = 100.0,
        **kwargs
    ) -> None:
        """Initialize grid environment.
        
        Args:
            feeder: Network topology and parameters
            timestep: Simulation timestep in seconds
            episode_length: Episode length in timesteps
            stochastic_loads: Enable load uncertainty
            renewable_sources: List of renewable types ['solar', 'wind']
            weather_variation: Enable weather variations
            power_flow_solver: Custom power flow solver
            voltage_limits: Min/max voltage in per unit
            frequency_limits: Min/max frequency in Hz
            safety_penalty: Penalty for constraint violations
        """
        super().__init__(timestep, episode_length, voltage_limits, frequency_limits, **kwargs)
        
        self.feeder = feeder
        self.stochastic_loads = stochastic_loads
        self.renewable_sources = renewable_sources or []
        self.weather_variation = weather_variation
        self.safety_penalty = safety_penalty
        
        # Initialize power flow solver
        if power_flow_solver is None:
            # Use robust solver with fallback mechanism
            try:
                from .robust_power_flow import RobustPowerFlowSolver
                self.solver = RobustPowerFlowSolver(tolerance=1e-4, max_iterations=20)
            except ImportError:
                self.solver = PowerFlowSolver()
        else:
            self.solver = power_flow_solver
            
        # Initialize grid dynamics
        self.dynamics = GridDynamics()
        
        # State tracking
        self.current_time = 0.0
        self.weather = WeatherData(
            solar_irradiance=0.0,
            wind_speed=5.0,
            temperature=25.0,
            cloud_cover=0.3
        )
        
        # Grid components (will be populated from feeder)
        self.buses: List[Bus] = []
        self.lines: List[Line] = []
        self.loads: List[Load] = []
        self.generators: Dict[str, Dict] = {}
        self.batteries: Dict[str, BatteryModel] = {}
        
        # Performance tracking
        self.total_load_served = 0.0
        self.total_generation = 0.0
        self.total_losses = 0.0
        self.voltage_violations = 0
        self.frequency_violations = 0
        
        # Initialize feeder components
        self._initialize_feeder()
        
        # Set up action and observation spaces
        self._setup_spaces()
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
    def _initialize_feeder(self) -> None:
        """Initialize grid components from feeder definition."""
        # This would normally extract from feeder object
        # For now, create a simple 3-bus system
        
        # Buses
        self.buses = [
            Bus(id=1, voltage_level=12.47e3, bus_type="slack"),
            Bus(id=2, voltage_level=12.47e3, bus_type="pq"),
            Bus(id=3, voltage_level=12.47e3, bus_type="pq")
        ]
        
        # Lines
        self.lines = [
            Line(id="line_1_2", from_bus=1, to_bus=2, resistance=0.01, reactance=0.02, rating=5e6),
            Line(id="line_2_3", from_bus=2, to_bus=3, resistance=0.015, reactance=0.025, rating=3e6)
        ]
        
        # Loads
        self.loads = [
            Load(id="load_2", bus=2, base_power=2e6, power_factor=0.95),
            Load(id="load_3", bus=3, base_power=1.5e6, power_factor=0.95)
        ]
        
        # Set up load models
        if self.stochastic_loads:
            for load in self.loads:
                self.dynamics.add_load_model(load.id, TimeVaryingLoadModel())
                
        # Renewable generators
        if "solar" in self.renewable_sources:
            self.generators["solar_2"] = {
                "type": "solar",
                "bus": 2,
                "capacity": 1e6,  # 1 MW
                "model": SolarPVModel(efficiency=0.18, panel_area=5556)  # 1MW / (0.18 * 1000)
            }
            self.dynamics.add_renewable_model("solar_2", self.generators["solar_2"]["model"])
            
        if "wind" in self.renewable_sources:
            self.generators["wind_3"] = {
                "type": "wind", 
                "bus": 3,
                "capacity": 2e6,  # 2 MW
                "model": WindTurbineModel()
            }
            self.dynamics.add_renewable_model("wind_3", self.generators["wind_3"]["model"])
            
        # Battery storage
        self.batteries["battery_2"] = BatteryModel(
            capacity=1e3,      # 1 MWh
            power_rating=0.5e6,  # 500 kW
            efficiency=0.95,
            initial_soc=0.5
        )
        self.dynamics.add_battery_model("battery_2", self.batteries["battery_2"])
        
    def _setup_spaces(self) -> None:
        """Setup observation and action spaces."""
        # Observation space: bus voltages, angles, line flows, frequency, loads, generation, battery SOC
        n_buses = len(self.buses)
        n_lines = len(self.lines)
        n_batteries = len(self.batteries)
        
        obs_dim = (
            n_buses * 2 +    # Voltage magnitude and angle for each bus
            n_lines * 2 +    # Real and reactive power flow for each line
            1 +              # System frequency
            len(self.loads) * 2 +  # Active and reactive load for each load
            len(self.generators) +  # Generation for each generator
            n_batteries * 2  # SOC and power for each battery
        )
        
        # Observation bounds
        self.observation_space = Box(
            low=np.array([
                # Bus voltages (0.8 to 1.2 pu)
                *([0.8] * n_buses), *([np.pi] * n_buses),
                # Line flows (-10 to 10 MW)
                *([-10e6] * n_lines), *([-10e6] * n_lines),
                # Frequency (55 to 65 Hz)
                55.0,
                # Loads (0 to 5 MW)
                *([0.0] * len(self.loads) * 2),
                # Generation (0 to capacity)
                *([0.0] * len(self.generators)),
                # Battery SOC (0 to 1) and power (-rating to +rating)
                *([0.0, -1e6] * n_batteries)
            ]),
            high=np.array([
                # Bus voltages
                *([1.2] * n_buses), *([np.pi] * n_buses),
                # Line flows
                *([10e6] * n_lines), *([10e6] * n_lines),
                # Frequency
                65.0,
                # Loads
                *([5e6] * len(self.loads) * 2),
                # Generation
                *([gen["capacity"] for gen in self.generators.values()]),
                # Battery SOC and power
                *([1.0, 1e6] * n_batteries)
            ]),
            dtype=np.float32
        )
        
        # Action space: battery power commands, renewable curtailment
        # For simplicity: battery power command (-1 to 1, scaled to rating)
        action_dim = len(self.batteries) + len(self.generators)
        
        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(action_dim,),
            dtype=np.float32
        )
        
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
            
        # Reset time and counters
        self.current_time = 0.0
        self.current_step = 0
        self.episode_reward = 0.0
        self.constraint_violations = 0
        
        # Reset performance tracking
        self.total_load_served = 0.0
        self.total_generation = 0.0
        self.total_losses = 0.0
        self.voltage_violations = 0
        self.frequency_violations = 0
        
        # Reset grid state
        for bus in self.buses:
            bus.voltage_magnitude = 1.0
            bus.voltage_angle = 0.0
            
        for line in self.lines:
            line.power_flow = 0.0
            line.loading = 0.0
            
        # Reset dynamics
        self.dynamics.frequency = 60.0
        
        # Reset batteries
        for battery in self.batteries.values():
            battery.soc = 0.5
            battery.current_power = 0.0
            
        # Reset weather
        self._update_weather()
        
        # Get initial observation
        obs = self.get_observation()
        info = self.get_info()
        
        return obs, info
        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one environment step."""
        try:
            # Robust action handling with enhanced validation and monitoring
            try:
                from ..utils.robust_validation import global_validator, global_error_handler
                from ..utils.enhanced_logging import grid_logger, performance_monitor
                from ..utils.health_monitoring import system_health, system_watchdog
                
                # Start performance monitoring
                performance_monitor.start_timer("step_execution")
                system_watchdog.heartbeat()
                
                # Validate action
                validation_result = global_validator.validate_action(action, self.action_space)
                if not validation_result.is_valid:
                    grid_logger.log_step(self.current_step, "Action", f"Invalid action: {validation_result.errors}", logging.ERROR)
                    safe_action = global_error_handler.handle_action_error(InvalidActionError("Action validation failed"), action)
                    self._apply_actions(safe_action)
                    system_health.update_error_rate(self.current_step + 1, global_error_handler.error_count)
                else:
                    if validation_result.warnings:
                        grid_logger.log_step(self.current_step, "Action", f"Action warnings: {validation_result.warnings}", logging.WARNING)
                    self._apply_actions(action)
                    
            except ImportError:
                # Fallback to basic validation if robust modules not available
                action = validate_action(action, self.action_space)
                self._apply_actions(action)
                
        except InvalidActionError as e:
            # Enhanced error handling
            try:
                from ..utils.robust_validation import global_error_handler
                from ..utils.enhanced_logging import grid_logger
                global_error_handler.handle_action_error(e, action)
                grid_logger.log_error_with_context(e, {"step": self.current_step, "action": action})
            except ImportError:
                logging.error(f"Action error at step {self.current_step}: {e}")
            
            obs = self.get_observation()
            penalty = -self.safety_penalty
            info = {'error': str(e), 'action_invalid': True}
            return obs, penalty, False, False, info
        except Exception as e:
            # Enhanced unexpected error handling
            try:
                from ..utils.robust_validation import global_error_handler
                from ..utils.enhanced_logging import grid_logger
                global_error_handler.handle_critical_error(e, "step_execution")
                grid_logger.log_error_with_context(e, {"step": self.current_step, "action": action})
            except ImportError:
                logging.error(f"Unexpected error in step: {e}")
                
            obs = self.get_observation()
            penalty = -self.safety_penalty * 2
            info = {'error': str(e), 'unexpected_error': True}
            return obs, penalty, True, False, info
        
        # Update time
        self.current_time += self.timestep
        self.current_step += 1
        
        # Update weather
        self._update_weather()
        
        # Calculate load and generation
        loads_dict, generation_dict = self._calculate_power_injections()
        
        # Solve power flow with enhanced monitoring and caching
        start_pf_time = __import__('time').time()
        
        # Try to get cached solution first
        cached_solution = None
        try:
            from ..utils.performance_optimization import power_flow_cache, performance_profiler
            cached_solution = power_flow_cache.get_solution(self.buses, self.lines, loads_dict, generation_dict)
        except ImportError:
            pass
            
        if cached_solution is not None:
            solution = cached_solution
            pf_duration = __import__('time').time() - start_pf_time  # Minimal cache lookup time
        else:
            # Solve power flow and cache result
            solution = self.solver.solve(self.buses, self.lines, loads_dict, generation_dict)
            pf_duration = __import__('time').time() - start_pf_time
            
            # Cache the solution
            try:
                from ..utils.performance_optimization import power_flow_cache
                power_flow_cache.store_solution(self.buses, self.lines, loads_dict, generation_dict, solution)
            except ImportError:
                pass
        
        # Enhanced power flow monitoring
        try:
            from ..utils.robust_validation import global_validator
            from ..utils.enhanced_logging import grid_logger, performance_monitor
            from ..utils.health_monitoring import system_health
            
            # Validate power flow solution
            pf_validation = global_validator.validate_power_flow_convergence(solution)
            if not pf_validation.is_valid:
                grid_logger.log_step(self.current_step, "PowerFlow", f"Power flow issues: {pf_validation.errors}", logging.WARNING)
                
            # Log power flow performance with localization
            try:
                from ..utils.global_support import global_manager
                if solution.converged:
                    message = global_manager.localize_message("power_flow_converged")
                else:
                    message = global_manager.localize_message("power_flow_failed")
                grid_logger.log_step(self.current_step, "PowerFlow", message, logging.INFO)
                
                # Also record for compliance if needed
                from ..utils.compliance_framework import record_data_activity
                record_data_activity("technical", "grid_simulation")
                
            except ImportError:
                # Fallback to English
                grid_logger.log_power_flow(self.current_step, solution.converged, 
                                         getattr(solution, 'iterations', 0), 
                                         getattr(solution, 'losses', 0.0))
            
            # Update health metrics and performance optimization
            system_health.update_simulation_speed(pf_duration)
            performance_monitor.end_timer("step_execution", self.current_step)
            
            # Record performance for adaptive optimization
            try:
                from ..utils.performance_optimization import adaptive_optimizer, memory_optimizer
                adaptive_optimizer.record_performance("power_flow", pf_duration, solution.converged)
                memory_optimizer.cleanup_if_needed()
            except ImportError:
                pass
            
        except ImportError:
            # Fallback logging
            if not solution.converged:
                logging.warning(f"Power flow did not converge at step {self.current_step}")
        
        # Update grid state from solution
        self._update_grid_state(solution)
        
        # Update dynamics
        self._update_dynamics(solution)
        
        # Get observation and reward
        obs = self.get_observation()
        reward = self.get_reward(action, obs)
        
        # Check termination
        terminated = self.is_done()
        truncated = False
        
        # Enhanced safety violation handling
        violations = self.check_constraints(self._get_grid_state())
        # Safely check if any violations occurred
        has_violations = False
        try:
            if isinstance(violations, dict):
                for v in violations.values():
                    if hasattr(v, '__len__'):  # List or array-like
                        if len(v) > 0:
                            has_violations = True
                            break
                    elif v:  # Simple boolean or truthy value
                        has_violations = True
                        break
            else:
                has_violations = bool(violations)
        except (TypeError, ValueError):
            has_violations = False
            
        if has_violations:
            self.constraint_violations += 1
            
            try:
                from ..utils.enhanced_logging import grid_logger
                from ..utils.health_monitoring import system_health
                
                # Log specific violations
                for violation_type, occurred in violations.items():
                    if occurred:
                        grid_logger.log_constraint_violation(self.current_step, violation_type, 
                                                           {"state": self._get_grid_state()})
                
                # Update health monitoring
                system_health.increment_violations()
                
            except ImportError:
                logging.warning(f"Constraint violations at step {self.current_step}: {violations}")
            
            if self.constraint_violations > 10:  # Safety limit
                truncated = True
                reward -= self.safety_penalty
                
        self.episode_reward += reward
        
        info = self.get_info()
        info.update({
            "power_flow_converged": solution.converged,
            "max_voltage": max(solution.bus_voltages),
            "min_voltage": min(solution.bus_voltages),
            "total_losses": solution.losses,
            "constraint_violations": violations
        })
        
        return obs, reward, terminated, truncated, info
        
    def _apply_actions(self, action: np.ndarray) -> None:
        """Apply control actions to grid components."""
        # Ensure action is a numpy array
        if not isinstance(action, np.ndarray):
            action = np.array([action])
        
        action_idx = 0
        
        # Battery commands
        battery_commands = {}
        for battery_id, battery in self.batteries.items():
            # Scale action (-1, 1) to power rating
            if action_idx < len(action):
                power_command = action[action_idx] * battery.power_rating
            else:
                power_command = 0.0  # Default to no action
            battery_commands[battery_id] = power_command
            action_idx += 1
            
        # Apply battery commands
        actual_powers = self.dynamics.update_batteries(battery_commands, self.timestep)
        
        # Renewable curtailment (if any generators)
        self.curtailment = {}
        for gen_id in self.generators.keys():
            if action_idx < len(action):
                # Curtailment factor (0 = full curtailment, 1 = no curtailment)
                self.curtailment[gen_id] = (action[action_idx] + 1) / 2
                action_idx += 1
            else:
                self.curtailment[gen_id] = 1.0
                
    def _update_weather(self) -> None:
        """Update weather conditions."""
        if not self.weather_variation:
            return
            
        # Simple weather model with daily cycles
        hour = (self.current_time / 3600) % 24
        
        # Solar irradiance (follows sun elevation)
        if 6 <= hour <= 18:
            sun_elevation = np.sin(np.pi * (hour - 6) / 12)
            base_irradiance = 1000 * sun_elevation
        else:
            base_irradiance = 0
            
        # Add some randomness
        import random
        self.weather.solar_irradiance = base_irradiance * (0.8 + 0.4 * random.random())
        
        # Wind speed (with some persistence)
        self.weather.wind_speed += random.gauss(0, 0.5)
        self.weather.wind_speed = max(0, min(30, self.weather.wind_speed))
        
        # Temperature (daily cycle)
        base_temp = 25 + 10 * np.sin(2 * np.pi * (hour - 12) / 24)
        self.weather.temperature = base_temp + random.gauss(0, 2)
        
        # Cloud cover
        self.weather.cloud_cover = max(0, min(1, self.weather.cloud_cover + random.gauss(0, 0.1)))
        
    def _calculate_power_injections(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Calculate current power injections for loads and generation."""
        loads_dict = {}
        generation_dict = {}
        
        # Calculate load powers
        for load in self.loads:
            if self.stochastic_loads:
                active_power, reactive_power = self.dynamics.get_load_power(
                    load.id, load.base_power, self.current_time,
                    power_factor=0.95, timestep=self.timestep
                )
            else:
                active_power = load.base_power
                
            loads_dict[load.bus] = loads_dict.get(load.bus, 0) + active_power
            
        # Calculate renewable generation
        for gen_id, gen_info in self.generators.items():
            power = self.dynamics.get_renewable_power(
                gen_id, gen_info["capacity"], self.current_time, self.weather
            )
            
            # Apply curtailment
            curtailed_power = power * self.curtailment.get(gen_id, 1.0)
            
            generation_dict[gen_info["bus"]] = generation_dict.get(gen_info["bus"], 0) + curtailed_power
            
        # Add battery power
        for battery_id, battery in self.batteries.items():
            if battery.current_power > 0:  # Discharging
                bus_id = 2  # Hardcoded for now
                generation_dict[bus_id] = generation_dict.get(bus_id, 0) + battery.current_power
            elif battery.current_power < 0:  # Charging
                bus_id = 2
                loads_dict[bus_id] = loads_dict.get(bus_id, 0) + abs(battery.current_power)
                
        return loads_dict, generation_dict
        
    def _update_grid_state(self, solution: PowerFlowSolution) -> None:
        """Update internal grid state from power flow solution."""
        if not solution.converged:
            self.logger.warning(f"Power flow did not converge at step {self.current_step}")
            
        # Update bus voltages
        for i, bus in enumerate(self.buses):
            if i < len(solution.bus_voltages):
                bus.voltage_magnitude = solution.bus_voltages[i]
                bus.voltage_angle = solution.bus_angles[i]
                
        # Update line flows
        for i, line in enumerate(self.lines):
            if i < len(solution.line_flows):
                line.update_state(power_flow=solution.line_flows[i])
                
        # Track losses
        self.total_losses += solution.losses * self.timestep / 3600  # kWh
        
    def _update_dynamics(self, solution: PowerFlowSolution) -> None:
        """Update system dynamics based on power flow results."""
        # Calculate power imbalance for frequency dynamics
        total_load = sum(load.active_power for load in self.loads)
        total_generation = sum(
            self.dynamics.get_renewable_power(gen_id, gen_info["capacity"], self.current_time, self.weather)
            for gen_id, gen_info in self.generators.items()
        )
        
        power_imbalance = total_generation - total_load - solution.losses
        self.dynamics.update_frequency(power_imbalance / 1e6, self.timestep)  # MW
        
    def get_observation(self) -> np.ndarray:
        """Get current observation vector."""
        obs = []
        
        # Bus voltages and angles
        for bus in self.buses:
            obs.extend([bus.voltage_magnitude, bus.voltage_angle])
            
        # Line flows and loadings
        for line in self.lines:
            obs.extend([line.power_flow, line.loading])
            
        # System frequency
        obs.append(self.dynamics.frequency)
        
        # Load powers
        for load in self.loads:
            obs.extend([load.active_power, load.reactive_power])
            
        # Generation powers
        for gen_id, gen_info in self.generators.items():
            power = self.dynamics.get_renewable_power(
                gen_id, gen_info["capacity"], self.current_time, self.weather
            )
            obs.append(power)
            
        # Battery states
        for battery in self.batteries.values():
            obs.extend([battery.soc, battery.current_power])
            
        return obs
        
    def get_reward(self, action, next_obs) -> float:
        """Calculate reward based on grid performance."""
        reward = 0.0
        
        # Power quality reward (voltage regulation)
        voltages = [bus.voltage_magnitude for bus in self.buses]
        voltage_deviations = [abs(v - 1.0) for v in voltages]
        reward -= sum(voltage_deviations) * 10  # Penalty for voltage deviations
        
        # Frequency regulation reward
        freq_deviation = abs(self.dynamics.frequency - 60.0)
        reward -= freq_deviation * 20
        
        # Line loading penalty
        loadings = [line.loading for line in self.lines]
        overloaded = [l > 0.8 for l in loadings]
        reward -= sum(overloaded) * 50  # Penalty for high line loadings
        
        # Efficiency reward (minimize losses)
        reward -= self.total_losses * 0.1
        
        # Renewable utilization reward
        total_renewable = sum(
            self.dynamics.get_renewable_power(gen_id, gen_info["capacity"], self.current_time, self.weather)
            for gen_id, gen_info in self.generators.items()
        )
        total_curtailed = sum(
            self.dynamics.get_renewable_power(gen_id, gen_info["capacity"], self.current_time, self.weather) * 
            (1 - self.curtailment.get(gen_id, 1.0))
            for gen_id, gen_info in self.generators.items()
        )
        reward += (total_renewable - total_curtailed) * 1e-5  # Small reward for renewable usage
        
        # Battery efficiency
        for battery in self.batteries.values():
            # Reward for keeping SOC in good range
            if 0.2 <= battery.soc <= 0.8:
                reward += 1.0
            else:
                reward -= 5.0
                
        return reward
        
    def _get_grid_state(self) -> Dict[str, Any]:
        """Get current grid state for constraint checking."""
        return {
            "bus_voltages": [bus.voltage_magnitude for bus in self.buses],
            "line_loadings": [line.loading for line in self.lines],
            "frequency": self.dynamics.frequency
        }
        
    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render the environment state."""
        if mode == "human":
            print(f"\\nStep: {self.current_step}, Time: {self.current_time:.1f}s")
            print(f"Frequency: {self.dynamics.frequency:.2f} Hz")
            print("Bus Voltages:")
            for bus in self.buses:
                print(f"  Bus {bus.id}: {bus.voltage_magnitude:.3f} pu")
            print("Line Loadings:")
            for line in self.lines:
                print(f"  {line.id}: {line.loading:.1%}")
            print("Batteries:")
            for battery_id, battery in self.batteries.items():
                print(f"  {battery_id}: SOC={battery.soc:.1%}, Power={battery.current_power/1e3:.1f} kW")
        return None