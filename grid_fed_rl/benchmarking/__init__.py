"""Comprehensive benchmarking framework for federated RL in power systems.

This module provides tools for rigorous experimental evaluation, statistical
significance testing, and reproducible research in federated power grid control.
"""

from .benchmark_suite import (
    BenchmarkSuite, BenchmarkExperiment, ExperimentResult,
    IEEE_TEST_CASES, RENEWABLE_SCENARIOS, LOAD_PROFILES
)
from .statistical_analysis import (
    StatisticalAnalyzer, SignificanceTest, EffectSizeCalculator,
    MultipleComparison, BootstrapAnalysis
)
from .performance_metrics import (
    PowerSystemMetrics, FederatedLearningMetrics, 
    EconomicMetrics, SafetyMetrics, EnvironmentalMetrics
)
from .reproducibility import (
    ReproducibilityManager, ExperimentConfig, ResultsValidator,
    SeedManager, EnvironmentValidator
)
from .visualization import (
    BenchmarkVisualizer, PerformanceComparison, StatisticalPlots,
    FederatedAnalysisPlots, GridTopologyVisualizer
)

__all__ = [
    # Benchmark Suite
    "BenchmarkSuite",
    "BenchmarkExperiment", 
    "ExperimentResult",
    "IEEE_TEST_CASES",
    "RENEWABLE_SCENARIOS",
    "LOAD_PROFILES",
    
    # Statistical Analysis
    "StatisticalAnalyzer",
    "SignificanceTest",
    "EffectSizeCalculator", 
    "MultipleComparison",
    "BootstrapAnalysis",
    
    # Performance Metrics
    "PowerSystemMetrics",
    "FederatedLearningMetrics",
    "EconomicMetrics",
    "SafetyMetrics",
    "EnvironmentalMetrics",
    
    # Reproducibility
    "ReproducibilityManager",
    "ExperimentConfig",
    "ResultsValidator",
    "SeedManager",
    "EnvironmentValidator",
    
    # Visualization
    "BenchmarkVisualizer",
    "PerformanceComparison",
    "StatisticalPlots",
    "FederatedAnalysisPlots",
    "GridTopologyVisualizer"
]