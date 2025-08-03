"""Grid dynamics models for load, generation, and system behavior."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
import math


@dataclass
class WeatherData:
    """Weather conditions for renewable generation modeling."""
    solar_irradiance: float  # W/m²
    wind_speed: float       # m/s
    temperature: float      # °C
    cloud_cover: float     # 0-1


class LoadModel(ABC):
    """Abstract base class for load models."""
    
    @abstractmethod
    def get_power(self, time: float, base_power: float, **kwargs) -> Tuple[float, float]:
        """Get active and reactive power at given time."""
        pass


class ConstantPowerModel(LoadModel):
    """Constant power load model."""
    
    def get_power(self, time: float, base_power: float, **kwargs) -> Tuple[float, float]:
        power_factor = kwargs.get('power_factor', 0.95)
        reactive_power = base_power * math.tan(math.acos(power_factor))
        return base_power, reactive_power


class TimeVaryingLoadModel(LoadModel):
    """Time-varying load model with daily/seasonal patterns."""
    
    def __init__(self, daily_profile: Optional[np.ndarray] = None, seasonal_factor: float = 1.0):
        if daily_profile is None:
            # Default residential load profile (24 hours)
            self.daily_profile = np.array([
                0.5, 0.4, 0.4, 0.4, 0.4, 0.5,  # 0-5 AM
                0.7, 0.9, 0.8, 0.7, 0.6, 0.6,  # 6-11 AM
                0.7, 0.7, 0.6, 0.6, 0.7, 0.9,  # 12-5 PM
                1.0, 0.9, 0.8, 0.7, 0.6, 0.5   # 6-11 PM
            ])
        else:
            self.daily_profile = daily_profile
            
        self.seasonal_factor = seasonal_factor
        
    def get_power(self, time: float, base_power: float, **kwargs) -> Tuple[float, float]:
        # Time in hours
        hour = (time / 3600) % 24
        hour_idx = int(hour)
        
        # Interpolate between hours
        next_hour_idx = (hour_idx + 1) % 24
        fraction = hour - hour_idx
        
        multiplier = (self.daily_profile[hour_idx] * (1 - fraction) + 
                     self.daily_profile[next_hour_idx] * fraction)
        
        # Add random variation (±10%)
        noise = kwargs.get('noise_factor', 0.1)
        if noise > 0:
            multiplier *= (1 + np.random.normal(0, noise))
            
        active_power = base_power * multiplier * self.seasonal_factor
        power_factor = kwargs.get('power_factor', 0.95)
        reactive_power = active_power * math.tan(math.acos(power_factor))
        
        return max(0, active_power), reactive_power


class StochasticLoadModel(LoadModel):
    """Stochastic load model with Ornstein-Uhlenbeck process."""
    
    def __init__(self, mean_reversion: float = 0.1, volatility: float = 0.2):
        self.mean_reversion = mean_reversion
        self.volatility = volatility
        self.current_deviation = 0.0
        
    def get_power(self, time: float, base_power: float, **kwargs) -> Tuple[float, float]:
        dt = kwargs.get('timestep', 1.0)
        
        # Ornstein-Uhlenbeck process for load variation
        dW = np.random.normal(0, math.sqrt(dt))
        self.current_deviation += (-self.mean_reversion * self.current_deviation * dt + 
                                  self.volatility * dW)
        
        multiplier = 1.0 + self.current_deviation
        multiplier = max(0.1, min(2.0, multiplier))  # Clamp between 10%-200%
        
        active_power = base_power * multiplier
        power_factor = kwargs.get('power_factor', 0.95)
        reactive_power = active_power * math.tan(math.acos(power_factor))
        
        return active_power, reactive_power


class RenewableModel(ABC):
    """Abstract base class for renewable generation models."""
    
    @abstractmethod
    def get_power(self, time: float, weather: WeatherData, capacity: float, **kwargs) -> float:
        """Get power output given weather conditions."""
        pass


class SolarPVModel(RenewableModel):
    """Solar PV generation model."""
    
    def __init__(self, efficiency: float = 0.18, panel_area: float = 1000.0):
        self.efficiency = efficiency
        self.panel_area = panel_area  # m²
        
    def get_power(self, time: float, weather: WeatherData, capacity: float, **kwargs) -> float:
        # Solar irradiance model based on time of day
        hour = (time / 3600) % 24
        
        # Sun elevation angle (simplified)
        if 6 <= hour <= 18:
            sun_elevation = math.sin(math.pi * (hour - 6) / 12)
        else:
            sun_elevation = 0
            
        # Clear sky irradiance
        clear_sky_irradiance = 1000 * sun_elevation  # W/m²
        
        # Apply cloud cover
        actual_irradiance = clear_sky_irradiance * (1 - 0.8 * weather.cloud_cover)
        
        # Temperature derating (0.4%/°C above 25°C)
        temp_factor = 1 - 0.004 * max(0, weather.temperature - 25)
        
        # Power output
        power = (actual_irradiance * self.panel_area * self.efficiency * temp_factor)
        
        return min(power, capacity)


class WindTurbineModel(RenewableModel):
    """Wind turbine generation model."""
    
    def __init__(
        self,
        cut_in_speed: float = 3.0,
        rated_speed: float = 12.0,
        cut_out_speed: float = 25.0
    ):
        self.cut_in_speed = cut_in_speed
        self.rated_speed = rated_speed
        self.cut_out_speed = cut_out_speed
        
    def get_power(self, time: float, weather: WeatherData, capacity: float, **kwargs) -> float:
        wind_speed = weather.wind_speed
        
        if wind_speed < self.cut_in_speed or wind_speed > self.cut_out_speed:
            return 0.0
        elif wind_speed <= self.rated_speed:
            # Cubic relationship below rated speed
            power_ratio = ((wind_speed - self.cut_in_speed) / 
                          (self.rated_speed - self.cut_in_speed)) ** 3
            return capacity * power_ratio
        else:
            # Constant at rated power
            return capacity


class BatteryModel:
    """Battery energy storage system model."""
    
    def __init__(
        self,
        capacity: float,      # kWh
        power_rating: float,  # kW
        efficiency: float = 0.95,
        initial_soc: float = 0.5
    ):
        self.capacity = capacity
        self.power_rating = power_rating
        self.efficiency = efficiency
        self.soc = initial_soc  # State of charge (0-1)
        self.current_power = 0.0
        
    def charge(self, power: float, timestep: float) -> float:
        """Charge battery with given power for timestep duration.
        
        Returns actual power consumed.
        """
        max_charge_power = min(power, self.power_rating)
        max_energy = (1.0 - self.soc) * self.capacity
        max_charge_energy = max_energy / self.efficiency
        
        actual_energy = min(max_charge_power * timestep / 3600, max_charge_energy)
        actual_power = actual_energy * 3600 / timestep
        
        self.soc += actual_energy * self.efficiency / self.capacity
        self.current_power = -actual_power  # Negative for charging
        
        return actual_power
        
    def discharge(self, power: float, timestep: float) -> float:
        """Discharge battery with given power for timestep duration.
        
        Returns actual power delivered.
        """
        max_discharge_power = min(power, self.power_rating)
        max_energy = self.soc * self.capacity * self.efficiency
        
        actual_energy = min(max_discharge_power * timestep / 3600, max_energy)
        actual_power = actual_energy * 3600 / timestep
        
        self.soc -= actual_energy / (self.capacity * self.efficiency)
        self.current_power = actual_power  # Positive for discharging
        
        return actual_power
        
    def get_state(self) -> Dict[str, float]:
        return {
            "soc": self.soc,
            "power": self.current_power,
            "energy": self.soc * self.capacity
        }


class GridDynamics:
    """Overall grid dynamics coordinator."""
    
    def __init__(
        self,
        frequency_nominal: float = 60.0,
        inertia_constant: float = 5.0,
        damping_coefficient: float = 1.0
    ):
        self.frequency_nominal = frequency_nominal
        self.frequency = frequency_nominal
        self.inertia_constant = inertia_constant
        self.damping_coefficient = damping_coefficient
        
        self.load_models: Dict[str, LoadModel] = {}
        self.renewable_models: Dict[str, RenewableModel] = {}
        self.battery_models: Dict[str, BatteryModel] = {}
        
    def add_load_model(self, load_id: str, model: LoadModel) -> None:
        """Add load model for specific load."""
        self.load_models[load_id] = model
        
    def add_renewable_model(self, gen_id: str, model: RenewableModel) -> None:
        """Add renewable generation model."""
        self.renewable_models[gen_id] = model
        
    def add_battery_model(self, battery_id: str, model: BatteryModel) -> None:
        """Add battery storage model."""
        self.battery_models[battery_id] = model
        
    def update_frequency(self, power_imbalance: float, timestep: float) -> None:
        """Update system frequency based on power imbalance."""
        # Simplified swing equation
        # P_imbalance = 2H * f * df/dt + D * df
        # df = (P_imbalance - D * df_prev) / (2H * f) * dt
        
        df_dt = ((power_imbalance - self.damping_coefficient * 
                 (self.frequency - self.frequency_nominal)) / 
                (2 * self.inertia_constant * self.frequency_nominal))
        
        self.frequency += df_dt * timestep
        
        # Limit frequency excursions
        self.frequency = max(55.0, min(65.0, self.frequency))
        
    def get_load_power(
        self,
        load_id: str,
        base_power: float,
        time: float,
        **kwargs
    ) -> Tuple[float, float]:
        """Get load power using assigned model."""
        if load_id in self.load_models:
            return self.load_models[load_id].get_power(time, base_power, **kwargs)
        else:
            # Default constant power
            model = ConstantPowerModel()
            return model.get_power(time, base_power, **kwargs)
            
    def get_renewable_power(
        self,
        gen_id: str,
        capacity: float,
        time: float,
        weather: WeatherData,
        **kwargs
    ) -> float:
        """Get renewable power using assigned model."""
        if gen_id in self.renewable_models:
            return self.renewable_models[gen_id].get_power(time, weather, capacity, **kwargs)
        else:
            return 0.0
            
    def update_batteries(self, battery_commands: Dict[str, float], timestep: float) -> Dict[str, float]:
        """Update all batteries with power commands.
        
        Returns actual power delivered/consumed by each battery.
        """
        actual_powers = {}
        
        for battery_id, command_power in battery_commands.items():
            if battery_id in self.battery_models:
                battery = self.battery_models[battery_id]
                
                if command_power > 0:  # Discharge
                    actual_power = battery.discharge(command_power, timestep)
                elif command_power < 0:  # Charge
                    actual_power = -battery.charge(-command_power, timestep)
                else:
                    actual_power = 0.0
                    
                actual_powers[battery_id] = actual_power
                
        return actual_powers
        
    def get_system_state(self) -> Dict[str, Any]:
        """Get overall system state."""
        state = {
            "frequency": self.frequency,
            "frequency_deviation": self.frequency - self.frequency_nominal
        }
        
        # Add battery states
        for battery_id, battery in self.battery_models.items():
            battery_state = battery.get_state()
            state[f"battery_{battery_id}_soc"] = battery_state["soc"]
            state[f"battery_{battery_id}_power"] = battery_state["power"]
            
        return state