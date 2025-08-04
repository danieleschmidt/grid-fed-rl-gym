"""Main grid environment for reinforcement learning."""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from .base import Box
import warnings
import logging

from ..utils import (
    validate_action, validate_network_parameters, sanitize_config,
    PowerFlowError, InvalidActionError, SafetyLimitExceededError
)

from .base import BaseGridEnvironment, Bus, Line, Load
from .power_flow import PowerFlowSolver, NewtonRaphsonSolver, PowerFlowSolution
from .dynamics import GridDynamics, WeatherData, TimeVaryingLoadModel, SolarPVModel, WindTurbineModel, BatteryModel


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
            from .robust_power_flow import RobustPowerFlowSolver
            self.solver = RobustPowerFlowSolver(tolerance=1e-4, max_iterations=20)
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
            # Validate and sanitize action
            action = validate_action(action, self.action_space)
            
            # Apply actions
            self._apply_actions(action)
        except InvalidActionError as e:
            # Return safe observation with penalty
            obs = self.get_observation()
            penalty = -self.safety_penalty
            info = {'error': str(e), 'action_invalid': True}
            return obs, penalty, False, False, info
        except Exception as e:
            # Log unexpected error and return safe state
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
        
        # Solve power flow
        solution = self.solver.solve(self.buses, self.lines, loads_dict, generation_dict)
        
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
        
        # Check for safety violations
        violations = self.check_constraints(self._get_grid_state())
        if any(violations.values()):
            self.constraint_violations += 1
            if self.constraint_violations > 10:  # Safety limit
                truncated = True
                reward -= self.safety_penalty
                
        self.episode_reward += reward
        
        info = self.get_info()
        info.update({
            "power_flow_converged": solution.converged,
            "max_voltage": np.max(solution.bus_voltages),
            "min_voltage": np.min(solution.bus_voltages),
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
        self.weather.solar_irradiance = base_irradiance * (0.8 + 0.4 * np.random.random())
        
        # Wind speed (with some persistence)
        self.weather.wind_speed += np.random.normal(0, 0.5)
        self.weather.wind_speed = np.clip(self.weather.wind_speed, 0, 30)
        
        # Temperature (daily cycle)
        base_temp = 25 + 10 * np.sin(2 * np.pi * (hour - 12) / 24)
        self.weather.temperature = base_temp + np.random.normal(0, 2)
        
        # Cloud cover
        self.weather.cloud_cover = np.clip(self.weather.cloud_cover + np.random.normal(0, 0.1), 0, 1)
        
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
            
        return np.array(obs, dtype=np.float32)
        
    def get_reward(self, action: np.ndarray, next_obs: np.ndarray) -> float:
        """Calculate reward based on grid performance."""
        reward = 0.0
        
        # Power quality reward (voltage regulation)
        voltages = np.array([bus.voltage_magnitude for bus in self.buses])
        voltage_deviations = np.abs(voltages - 1.0)
        reward -= np.sum(voltage_deviations) * 10  # Penalty for voltage deviations
        
        # Frequency regulation reward
        freq_deviation = abs(self.dynamics.frequency - 60.0)
        reward -= freq_deviation * 20
        
        # Line loading penalty
        loadings = np.array([line.loading for line in self.lines])
        overloaded = loadings > 0.8
        reward -= np.sum(overloaded) * 50  # Penalty for high line loadings
        
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
            "bus_voltages": np.array([bus.voltage_magnitude for bus in self.buses]),
            "line_loadings": np.array([line.loading for line in self.lines]),
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