"""Research framework for reproducible federated RL experiments in power systems.

This module provides comprehensive tools for conducting reproducible research,
including experiment management, result validation, and publication-ready outputs.
"""

from .experiment_manager import (
    ExperimentManager, ResearchExperiment, ExperimentPipeline,
    ResearchConfig, ResultsAggregator
)
from .reproducibility import (
    ReproducibilityManager, SeedManager, EnvironmentValidator,
    ExperimentTracker, ResultsValidator
)
from .paper_generator import (
    PaperGenerator, LatexGenerator, ResultsFormatter,
    FigureGenerator, TableGenerator
)
from .research_utils import (
    DatasetGenerator, BaselineComparison, AlgorithmFactory,
    MetricsCollector, ExperimentLogger
)

__all__ = [
    # Experiment Management
    "ExperimentManager",
    "ResearchExperiment", 
    "ExperimentPipeline",
    "ResearchConfig",
    "ResultsAggregator",
    
    # Reproducibility
    "ReproducibilityManager",
    "SeedManager",
    "EnvironmentValidator",
    "ExperimentTracker",
    "ResultsValidator",
    
    # Paper Generation
    "PaperGenerator",
    "LatexGenerator",
    "ResultsFormatter",
    "FigureGenerator",
    "TableGenerator",
    
    # Research Utilities
    "DatasetGenerator",
    "BaselineComparison",
    "AlgorithmFactory",
    "MetricsCollector",
    "ExperimentLogger"
]