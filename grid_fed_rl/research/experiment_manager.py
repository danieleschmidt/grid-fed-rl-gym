"""Comprehensive experiment management for reproducible federated RL research."""

from typing import Any, Dict, List, Optional, Tuple, Callable, Union
import numpy as np
import pandas as pd
import json
import yaml
import pickle
import hashlib
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import git

from ..algorithms.physics_informed import PIFRL, PIFRLClient
from ..algorithms.multi_objective import MOFRL
from ..algorithms.uncertainty_aware import UAFRL, UAFRLClient
from ..algorithms.graph_neural import GNFRL, GNFRLClient
from ..algorithms.continual_learning import ContinualFederatedRL, ContinualFederatedClient
from ..algorithms.offline import CQL, IQL, AWR
from ..algorithms.safe import SafeRL, ConstrainedPolicyOptimization
from ..benchmarking.benchmark_suite import BenchmarkSuite, ExperimentResult
from ..benchmarking.statistical_analysis import StatisticalAnalyzer
from ..federated.core import FederatedOfflineRL, FedLearningConfig


@dataclass
class ResearchConfig:
    """Configuration for research experiments."""
    # Experiment metadata
    experiment_name: str
    description: str
    authors: List[str]
    institution: str
    contact_email: str
    
    # Research parameters
    research_questions: List[str]
    hypotheses: List[str]
    objectives: List[str]
    
    # Experimental design
    algorithms: List[str]
    test_cases: List[str]
    scenarios: List[str]
    num_seeds: int = 10
    num_trials: int = 5
    
    # Computational resources
    parallel_execution: bool = True
    max_workers: int = -1
    timeout_minutes: int = 120
    memory_limit_gb: int = 16
    
    # Output settings
    output_directory: str = "research_results"
    save_raw_data: bool = True
    save_models: bool = True
    generate_plots: bool = True
    generate_tables: bool = True
    
    # Publication settings
    target_venue: Optional[str] = None
    paper_template: str = "ieee"
    include_appendix: bool = True
    
    # Reproducibility
    random_seed: int = 42
    version_control: bool = True
    environment_snapshot: bool = True
    
    # Additional metadata
    created_at: Optional[datetime] = None
    version: str = "1.0"
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    def save(self, filepath: Union[str, Path]):
        """Save configuration to file."""
        filepath = Path(filepath)
        
        if filepath.suffix == '.json':
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=2, default=str)
        elif filepath.suffix in ['.yml', '.yaml']:
            with open(filepath, 'w') as f:
                yaml.dump(self.to_dict(), f, indent=2, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'ResearchConfig':
        """Load configuration from file."""
        filepath = Path(filepath)
        
        if filepath.suffix == '.json':
            with open(filepath, 'r') as f:
                data = json.load(f)
        elif filepath.suffix in ['.yml', '.yaml']:
            with open(filepath, 'r') as f:
                data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        # Convert datetime string back to datetime object
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        
        return cls(**data)


@dataclass
class ResearchExperiment:
    """Individual research experiment definition."""
    name: str
    algorithm: str
    test_case: str
    scenario: str
    config_overrides: Dict[str, Any] = field(default_factory=dict)
    expected_runtime_minutes: Optional[float] = None
    priority: int = 1  # Higher = higher priority
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    
    def get_experiment_id(self) -> str:
        """Generate unique experiment ID."""
        content = f"{self.algorithm}_{self.test_case}_{self.scenario}"
        return hashlib.md5(content.encode()).hexdigest()[:8]


class AlgorithmFactory:
    """Factory for creating algorithm instances."""
    
    ALGORITHM_REGISTRY = {
        'pifrl': PIFRL,
        'mofrl': MOFRL, 
        'uafrl': UAFRL,
        'gnfrl': GNFRL,
        'continual_frl': ContinualFederatedRL,
        'cql': CQL,
        'iql': IQL,
        'awr': AWR,
        'safe_rl': SafeRL,
        'cpo': ConstrainedPolicyOptimization
    }
    
    CLIENT_REGISTRY = {
        'pifrl': PIFRLClient,
        'uafrl': UAFRLClient,
        'gnfrl': GNFRLClient,
        'continual_frl': ContinualFederatedClient
    }
    
    @classmethod
    def create_algorithm(
        self,
        algorithm_name: str,
        state_dim: int,
        action_dim: int,
        **kwargs
    ):
        """Create algorithm instance."""
        if algorithm_name not in self.ALGORITHM_REGISTRY:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
        
        algorithm_class = self.ALGORITHM_REGISTRY[algorithm_name]
        
        # Handle special cases for different algorithms
        if algorithm_name == 'pifrl':
            # Physics-informed RL needs additional parameters
            kwargs.setdefault('num_buses', 13)
            kwargs.setdefault('physics_constraints', [])
            
        elif algorithm_name == 'mofrl':
            # Multi-objective RL needs objectives
            from ..algorithms.multi_objective import (
                economic_efficiency_objective,
                grid_stability_objective,
                environmental_impact_objective
            )
            kwargs.setdefault('objectives', [
                economic_efficiency_objective(),
                grid_stability_objective(),
                environmental_impact_objective()
            ])
            
        elif algorithm_name == 'gnfrl':
            # Graph neural RL needs graph topology
            from ..algorithms.graph_neural import PowerSystemGraph
            kwargs.setdefault('graph_topology', 
                             PowerSystemGraph.create_ieee_bus_graph(13))
            kwargs.setdefault('node_feature_dim', 8)
            
        elif algorithm_name == 'continual_frl':
            # Continual learning needs method specification
            kwargs.setdefault('continual_method', 'ewc')
            
        return algorithm_class(state_dim, action_dim, **kwargs)
    
    @classmethod
    def create_client(
        self,
        algorithm_name: str,
        client_id: str,
        algorithm_instance,
        grid_data: List[Dict[str, Any]],
        **kwargs
    ):
        """Create federated client instance."""
        if algorithm_name not in self.CLIENT_REGISTRY:
            # Use generic federated client
            from ..federated.core import GridUtilityClient
            return GridUtilityClient(client_id, algorithm_instance, grid_data)
        
        client_class = self.CLIENT_REGISTRY[algorithm_name]
        return client_class(client_id, algorithm_instance, grid_data, **kwargs)


class ResultsAggregator:
    """Aggregate and analyze experimental results."""
    
    def __init__(self):
        self.statistical_analyzer = StatisticalAnalyzer()
        self.logger = logging.getLogger(__name__)
    
    def aggregate_results(
        self,
        results: List[ExperimentResult],
        group_by: List[str] = None
    ) -> pd.DataFrame:
        """Aggregate experimental results into DataFrame."""
        if group_by is None:
            group_by = ['algorithm_name', 'test_case', 'scenario']
        
        # Convert results to records
        records = []
        for result in results:
            record = {
                'algorithm': result.algorithm_name,
                'test_case': result.test_case,
                'scenario': result.scenario,
                'seed': result.seed,
                'execution_time': result.execution_time,
                'memory_usage': result.memory_usage,
                'convergence_step': result.convergence_step
            }
            
            # Add performance metrics
            record.update({f'perf_{k}': v for k, v in result.performance_metrics.items()})
            
            # Add other metric categories
            record.update({f'safety_{k}': v for k, v in result.safety_metrics.items()})
            record.update({f'economic_{k}': v for k, v in result.economic_metrics.items()})
            record.update({f'env_{k}': v for k, v in result.environmental_metrics.items()})
            
            records.append(record)
        
        return pd.DataFrame(records)
    
    def compute_summary_statistics(
        self,
        df: pd.DataFrame,
        metrics: List[str] = None
    ) -> pd.DataFrame:
        """Compute summary statistics for each algorithm/test case combination."""
        if metrics is None:
            # Auto-detect numeric metrics
            metrics = [col for col in df.columns if df[col].dtype in ['float64', 'int64']]
            # Filter out ID columns
            metrics = [m for m in metrics if not m.endswith('_id') and m != 'seed']
        
        # Group by algorithm and test case
        grouped = df.groupby(['algorithm', 'test_case'])
        
        summary_stats = []
        
        for (algorithm, test_case), group in grouped:
            stats_row = {
                'algorithm': algorithm,
                'test_case': test_case,
                'n_experiments': len(group),
                'n_scenarios': group['scenario'].nunique()
            }
            
            for metric in metrics:
                if metric in group.columns:
                    values = group[metric].dropna()
                    if len(values) > 0:
                        stats_row.update({
                            f'{metric}_mean': values.mean(),
                            f'{metric}_std': values.std(),
                            f'{metric}_median': values.median(),
                            f'{metric}_min': values.min(),
                            f'{metric}_max': values.max(),
                            f'{metric}_q25': values.quantile(0.25),
                            f'{metric}_q75': values.quantile(0.75)
                        })
            
            summary_stats.append(stats_row)
        
        return pd.DataFrame(summary_stats)
    
    def statistical_comparison(
        self,
        results: List[ExperimentResult],
        metric: str = 'perf_mean_return',
        baseline_algorithm: str = None
    ) -> Dict[str, Any]:
        """Perform statistical comparison between algorithms."""
        
        # Group results by algorithm
        algorithm_groups = {}
        for result in results:
            algo = result.algorithm_name
            if algo not in algorithm_groups:
                algorithm_groups[algo] = []
            
            # Extract metric value
            if 'perf_' in metric:
                metric_key = metric.replace('perf_', '')
                value = result.performance_metrics.get(metric_key, np.nan)
            elif 'safety_' in metric:
                metric_key = metric.replace('safety_', '')
                value = result.safety_metrics.get(metric_key, np.nan)
            elif 'economic_' in metric:
                metric_key = metric.replace('economic_', '')
                value = result.economic_metrics.get(metric_key, np.nan)
            else:
                value = getattr(result, metric, np.nan)
            
            if not np.isnan(value):
                algorithm_groups[algo].append(value)
        
        # Convert to numpy arrays
        algorithm_arrays = {name: np.array(values) 
                           for name, values in algorithm_groups.items() 
                           if len(values) > 0}
        
        if len(algorithm_arrays) < 2:
            return {'error': 'Need at least 2 algorithms with data for comparison'}
        
        # Perform statistical analysis
        if len(algorithm_arrays) == 2:
            algorithms = list(algorithm_arrays.keys())
            comparison = self.statistical_analyzer.compare_two_groups(
                algorithm_arrays[algorithms[0]],
                algorithm_arrays[algorithms[1]], 
                algorithms[0],
                algorithms[1]
            )
        else:
            comparison = self.statistical_analyzer.compare_multiple_groups(algorithm_arrays)
        
        # Add ranking
        rankings = {}
        for name, values in algorithm_arrays.items():
            rankings[name] = np.mean(values)
        
        sorted_rankings = sorted(rankings.items(), key=lambda x: x[1], reverse=True)
        comparison['rankings'] = sorted_rankings
        
        return comparison
    
    def generate_research_summary(
        self,
        results: List[ExperimentResult],
        config: ResearchConfig
    ) -> Dict[str, Any]:
        """Generate comprehensive research summary."""
        df = self.aggregate_results(results)
        summary_stats = self.compute_summary_statistics(df)
        
        # Overall statistics
        total_experiments = len(results)
        algorithms_tested = len(df['algorithm'].unique())
        test_cases_used = len(df['test_case'].unique())
        scenarios_tested = len(df['scenario'].unique())
        
        # Performance analysis
        main_metric_comparison = self.statistical_comparison(
            results, 'perf_mean_return'
        )
        
        # Safety analysis
        safety_comparison = self.statistical_comparison(
            results, 'safety_safety_score'
        )
        
        # Efficiency analysis (execution time)
        efficiency_analysis = df.groupby('algorithm').agg({
            'execution_time': ['mean', 'std', 'min', 'max'],
            'memory_usage': ['mean', 'std', 'min', 'max']
        }).round(4)
        
        # Research questions analysis
        research_insights = self._analyze_research_questions(
            results, config.research_questions, df
        )
        
        return {
            'experiment_overview': {
                'total_experiments': total_experiments,
                'algorithms_tested': algorithms_tested,
                'test_cases_used': test_cases_used,
                'scenarios_tested': scenarios_tested,
                'total_runtime_hours': df['execution_time'].sum() / 3600,
                'avg_experiments_per_algorithm': total_experiments / algorithms_tested
            },
            'summary_statistics': summary_stats,
            'performance_comparison': main_metric_comparison,
            'safety_analysis': safety_comparison,
            'computational_efficiency': efficiency_analysis,
            'research_insights': research_insights,
            'data_quality': {
                'convergence_rate': (df['convergence_step'].notna().sum() / len(df)) * 100,
                'successful_experiments': (df['perf_success_rate'] > 0.8).sum() if 'perf_success_rate' in df else 0,
                'failed_experiments': df['perf_success_rate'].isna().sum() if 'perf_success_rate' in df else 0
            }
        }
    
    def _analyze_research_questions(
        self,
        results: List[ExperimentResult],
        research_questions: List[str],
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Analyze results in context of research questions."""
        insights = {}
        
        # This is a simplified analysis - in practice, you'd have more sophisticated
        # analysis tailored to specific research questions
        
        for i, question in enumerate(research_questions, 1):
            question_key = f"RQ{i}"
            
            # Example analyses based on common research question patterns
            if "federated" in question.lower():
                # Analyze federated learning performance
                fed_algos = [r for r in results if any(x in r.algorithm_name.lower() 
                            for x in ['fed', 'pifrl', 'mofrl', 'gnfrl'])]
                central_algos = [r for r in results if r not in fed_algos]
                
                if fed_algos and central_algos:
                    fed_performance = np.mean([r.performance_metrics.get('mean_return', 0) 
                                             for r in fed_algos])
                    central_performance = np.mean([r.performance_metrics.get('mean_return', 0) 
                                                 for r in central_algos])
                    
                    insights[question_key] = {
                        'question': question,
                        'finding': f"Federated algorithms achieved {fed_performance:.3f} vs centralized {central_performance:.3f}",
                        'federated_advantage': fed_performance > central_performance
                    }
            
            elif "safety" in question.lower():
                # Analyze safety performance
                safety_scores = df.groupby('algorithm')['safety_safety_score'].mean()
                best_safety = safety_scores.idxmax()
                
                insights[question_key] = {
                    'question': question,
                    'finding': f"Best safety performance: {best_safety} ({safety_scores[best_safety]:.3f})",
                    'safety_ranking': safety_scores.sort_values(ascending=False).to_dict()
                }
            
            elif "scalability" in question.lower() or "scale" in question.lower():
                # Analyze computational efficiency
                efficiency_ranking = df.groupby('algorithm')['execution_time'].mean().sort_values()
                most_scalable = efficiency_ranking.index[0]
                
                insights[question_key] = {
                    'question': question,
                    'finding': f"Most scalable algorithm: {most_scalable} ({efficiency_ranking[most_scalable]:.2f}s avg)",
                    'scalability_ranking': efficiency_ranking.to_dict()
                }
            
            else:
                # Generic analysis
                insights[question_key] = {
                    'question': question,
                    'finding': "Analysis framework ready - add specific analysis logic",
                    'data_available': True
                }
        
        return insights


class ExperimentPipeline:
    """Pipeline for running complete research experiments."""
    
    def __init__(
        self,
        config: ResearchConfig,
        output_dir: Optional[Path] = None
    ):
        self.config = config
        self.output_dir = Path(output_dir or config.output_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.algorithm_factory = AlgorithmFactory()
        self.results_aggregator = ResultsAggregator()
        self.benchmark_suite = BenchmarkSuite(
            output_dir=str(self.output_dir / "benchmarks"),
            parallel_execution=config.parallel_execution,
            n_jobs=config.max_workers
        )
        
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
        # Initialize experiment tracking
        self.experiment_log = []
        self.start_time = None
        
    def _setup_logging(self):
        """Setup logging for experiment pipeline."""
        log_file = self.output_dir / "experiment.log"
        
        # Create file handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        
        if not self.logger.handlers:
            self.logger.addHandler(fh)
            self.logger.setLevel(logging.INFO)
    
    def setup_experiments(self) -> List[ResearchExperiment]:
        """Setup all experiments based on configuration."""
        experiments = []
        
        for algorithm in self.config.algorithms:
            for test_case in self.config.test_cases:
                for scenario in self.config.scenarios:
                    experiment = ResearchExperiment(
                        name=f"{algorithm}_{test_case}_{scenario}",
                        algorithm=algorithm,
                        test_case=test_case,
                        scenario=scenario
                    )
                    experiments.append(experiment)
        
        self.logger.info(f"Setup {len(experiments)} experiments")
        return experiments
    
    def run_single_experiment(
        self,
        experiment: ResearchExperiment
    ) -> List[ExperimentResult]:
        """Run a single research experiment."""
        self.logger.info(f"Running experiment: {experiment.name}")
        
        try:
            # Create algorithm factory function
            def algorithm_factory():
                return self.algorithm_factory.create_algorithm(
                    experiment.algorithm,
                    state_dim=50,  # Default - would be determined by test case
                    action_dim=10,  # Default - would be determined by test case
                    **experiment.config_overrides
                )
            
            # Add to benchmark suite
            self.benchmark_suite.add_algorithm(
                algorithm_factory=algorithm_factory,
                algorithm_name=experiment.algorithm,
                test_cases=[experiment.test_case],
                scenarios=[experiment.scenario],
                num_seeds=self.config.num_seeds
            )
            
            # Run benchmark
            results = self.benchmark_suite.run_benchmarks(save_intermediate=True)
            
            self.logger.info(f"Completed experiment: {experiment.name} ({len(results)} results)")
            return results
            
        except Exception as e:
            self.logger.error(f"Experiment failed: {experiment.name} - {str(e)}")
            return []
    
    def run_all_experiments(self) -> List[ExperimentResult]:
        """Run all experiments in the pipeline."""
        self.start_time = time.time()
        self.logger.info(f"Starting research pipeline: {self.config.experiment_name}")
        
        # Setup experiments
        experiments = self.setup_experiments()
        
        # Run experiments
        all_results = []
        
        if self.config.parallel_execution and len(experiments) > 1:
            # Parallel execution
            max_workers = self.config.max_workers if self.config.max_workers > 0 else None
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all experiments
                future_to_experiment = {
                    executor.submit(self.run_single_experiment, exp): exp 
                    for exp in experiments
                }
                
                # Collect results
                for future in as_completed(future_to_experiment):
                    experiment = future_to_experiment[future]
                    try:
                        results = future.result(timeout=self.config.timeout_minutes * 60)
                        all_results.extend(results)
                        
                        self.experiment_log.append({
                            'experiment': experiment.name,
                            'status': 'completed',
                            'results_count': len(results),
                            'timestamp': datetime.now()
                        })
                        
                    except Exception as e:
                        self.logger.error(f"Experiment {experiment.name} failed: {e}")
                        self.experiment_log.append({
                            'experiment': experiment.name,
                            'status': 'failed',
                            'error': str(e),
                            'timestamp': datetime.now()
                        })
        else:
            # Sequential execution
            for experiment in experiments:
                results = self.run_single_experiment(experiment)
                all_results.extend(results)
                
                self.experiment_log.append({
                    'experiment': experiment.name,
                    'status': 'completed' if results else 'failed',
                    'results_count': len(results),
                    'timestamp': datetime.now()
                })
        
        # Save results and generate analysis
        self._save_final_results(all_results)
        
        total_time = time.time() - self.start_time
        self.logger.info(
            f"Research pipeline completed: {len(all_results)} total results "
            f"in {total_time/3600:.2f} hours"
        )
        
        return all_results
    
    def _save_final_results(self, results: List[ExperimentResult]):
        """Save final results and generate analysis."""
        # Save raw results
        if self.config.save_raw_data:
            with open(self.output_dir / "raw_results.pkl", 'wb') as f:
                pickle.dump(results, f)
        
        # Generate research summary
        research_summary = self.results_aggregator.generate_research_summary(
            results, self.config
        )
        
        # Save summary
        with open(self.output_dir / "research_summary.json", 'w') as f:
            json.dump(research_summary, f, indent=2, default=str)
        
        # Save experiment log
        with open(self.output_dir / "experiment_log.json", 'w') as f:
            json.dump(self.experiment_log, f, indent=2, default=str)
        
        # Save configuration
        self.config.save(self.output_dir / "experiment_config.yaml")
        
        self.logger.info(f"Results saved to {self.output_dir}")


class ExperimentManager:
    """High-level manager for research experiments."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def create_research_config(
        self,
        experiment_name: str,
        description: str,
        research_questions: List[str],
        algorithms: List[str] = None,
        **kwargs
    ) -> ResearchConfig:
        """Create research configuration with defaults."""
        
        if algorithms is None:
            algorithms = ['pifrl', 'mofrl', 'uafrl', 'cql', 'iql']
        
        config = ResearchConfig(
            experiment_name=experiment_name,
            description=description,
            authors=kwargs.get('authors', ['Research Team']),
            institution=kwargs.get('institution', 'Grid-Fed-RL Research Lab'),
            contact_email=kwargs.get('contact_email', 'research@gridfedrl.org'),
            research_questions=research_questions,
            hypotheses=kwargs.get('hypotheses', []),
            objectives=kwargs.get('objectives', ['Compare algorithm performance', 
                                               'Analyze safety properties',
                                               'Evaluate computational efficiency']),
            algorithms=algorithms,
            test_cases=kwargs.get('test_cases', ['ieee13_basic', 'ieee34_medium']),
            scenarios=kwargs.get('scenarios', ['sunny_day', 'windy_day', 'cloudy_variable']),
            **{k: v for k, v in kwargs.items() if k not in [
                'authors', 'institution', 'contact_email', 'hypotheses', 
                'objectives', 'test_cases', 'scenarios'
            ]}
        )
        
        return config
    
    def run_research_study(
        self,
        config: ResearchConfig,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run complete research study."""
        
        # Create output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"research_results/{config.experiment_name}_{timestamp}"
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Starting research study: {config.experiment_name}")
        self.logger.info(f"Output directory: {output_path}")
        
        # Create and run experiment pipeline
        pipeline = ExperimentPipeline(config, output_path)
        results = pipeline.run_all_experiments()
        
        # Generate final research package
        research_package = self._generate_research_package(
            config, results, output_path
        )
        
        self.logger.info(f"Research study completed: {len(results)} total results")
        return research_package
    
    def _generate_research_package(
        self,
        config: ResearchConfig,
        results: List[ExperimentResult],
        output_dir: Path
    ) -> Dict[str, Any]:
        """Generate comprehensive research package."""
        
        # Generate analysis
        aggregator = ResultsAggregator()
        research_summary = aggregator.generate_research_summary(results, config)
        
        # Create research package
        package = {
            'config': config.to_dict(),
            'summary': research_summary,
            'total_results': len(results),
            'output_directory': str(output_dir),
            'generated_at': datetime.now().isoformat(),
            'files_created': []
        }
        
        # List generated files
        for file_path in output_dir.rglob('*'):
            if file_path.is_file():
                package['files_created'].append(str(file_path.relative_to(output_dir)))
        
        # Save research package
        with open(output_dir / "research_package.json", 'w') as f:
            json.dump(package, f, indent=2, default=str)
        
        return package
    
    def load_research_results(
        self,
        output_dir: str
    ) -> Tuple[ResearchConfig, List[ExperimentResult], Dict[str, Any]]:
        """Load previously saved research results."""
        output_path = Path(output_dir)
        
        # Load configuration
        config_file = output_path / "experiment_config.yaml"
        if config_file.exists():
            config = ResearchConfig.load(config_file)
        else:
            config = None
        
        # Load results
        results_file = output_path / "raw_results.pkl"
        if results_file.exists():
            with open(results_file, 'rb') as f:
                results = pickle.load(f)
        else:
            results = []
        
        # Load summary
        summary_file = output_path / "research_summary.json"
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                summary = json.load(f)
        else:
            summary = {}
        
        return config, results, summary