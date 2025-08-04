"""Custom exceptions for grid-fed-rl-gym."""


class GridEnvironmentError(Exception):
    """Base exception for grid environment errors."""
    pass


class PowerFlowError(GridEnvironmentError):
    """Power flow calculation failed."""
    pass


class ConstraintViolationError(GridEnvironmentError):
    """Grid operational constraints violated."""
    pass


class NetworkTopologyError(GridEnvironmentError):
    """Invalid network topology."""
    pass


class InvalidActionError(GridEnvironmentError):
    """Invalid action provided to environment."""
    pass


class SafetyLimitExceededError(GridEnvironmentError):
    """Safety limits exceeded during operation."""
    pass


class DataValidationError(GridEnvironmentError):
    """Data validation failed."""
    pass


class ConfigurationError(GridEnvironmentError):
    """Invalid configuration provided."""
    pass