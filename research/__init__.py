"""
Research framework for novel federated RL algorithm validation.

This package provides statistical validation, experimental design, and
benchmarking capabilities for research-grade algorithm evaluation.
"""

from .statistical_validation import StatisticalValidator, ExperimentResult, ExperimentalDesign
from .experimental_framework import ExperimentRunner, AlgorithmConfig, EnvironmentConfig

__all__ = [
    'StatisticalValidator',
    'ExperimentResult', 
    'ExperimentalDesign',
    'ExperimentRunner',
    'AlgorithmConfig',
    'EnvironmentConfig'
]