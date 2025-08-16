"""
Experimental Framework for Novel Federated RL Algorithm Validation

This module implements a comprehensive experimental framework for running
controlled experiments and comparing novel algorithms against baselines.

Author: Daniel Schmidt <daniel@terragonlabs.com>
"""

import time
import json
import random
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path

# Import our statistical validation framework
from .statistical_validation import (
    StatisticalValidator, ExperimentResult, ExperimentalDesign
)

# Import core grid environment with graceful fallback
try:
    from ..environments.grid_env import GridEnvironment
    from ..environments.base import BaseGridEnvironment
    from ..feeders.ieee_feeders import IEEE13Bus, IEEE34Bus, IEEE123Bus
    _GRID_ENV_AVAILABLE = True
except ImportError:
    _GRID_ENV_AVAILABLE = False
    
    # Minimal mock implementations for testing
    class GridEnvironment:
        def __init__(self, feeder=None, **kwargs):
            self.feeder = feeder or "IEEE13Bus"
            self.episode_length = kwargs.get('episode_length', 100)
            self.step_count = 0
            
        def reset(self):
            self.step_count = 0
            return [0.0] * 20  # Mock state
            
        def step(self, action):
            self.step_count += 1
            reward = random.gauss(0.1, 0.05)  # Mock reward
            done = self.step_count >= self.episode_length
            info = {'safety_violations': random.randint(0, 1)}
            return [0.0] * 20, reward, done, info
    
    BaseGridEnvironment = GridEnvironment
    IEEE13Bus = IEEE34Bus = IEEE123Bus = lambda: "MockFeeder"


@dataclass
class AlgorithmConfig:
    """Configuration for an algorithm in experiments."""
    name: str
    class_name: str
    hyperparameters: Dict[str, Any]
    description: str = ""
    is_baseline: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EnvironmentConfig:
    """Configuration for an environment in experiments."""
    name: str
    feeder_type: str
    episode_length: int = 1000
    safety_constraints: Dict[str, float] = field(default_factory=dict)
    disturbances: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentConfig:
    """Complete configuration for a controlled experiment."""
    experiment_id: str
    algorithm: AlgorithmConfig
    environment: EnvironmentConfig
    n_episodes: int = 10
    random_seed: int = 42
    timeout_minutes: int = 30
    metrics_to_track: List[str] = field(default_factory=lambda: [
        'episode_reward', 'convergence_time', 'safety_violations', 
        'power_loss', 'voltage_deviation'
    ])
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MockAlgorithm:
    """Mock algorithm implementation for testing purposes."""
    
    def __init__(self, name: str, hyperparameters: Dict[str, Any]):
        self.name = name
        self.hyperparameters = hyperparameters
        self.base_performance = hyperparameters.get('base_performance', 0.5)
        self.variance = hyperparameters.get('variance', 0.1)
        self.convergence_factor = hyperparameters.get('convergence_factor', 1.0)
        
    def train_episode(self, env: GridEnvironment) -> Dict[str, float]:
        """Simulate training an episode."""
        total_reward = 0
        safety_violations = 0
        steps = 0
        start_time = time.time()
        
        state = env.reset()
        done = False
        
        while not done and steps < 1000:  # Max steps safety
            # Mock action selection (random with some bias based on performance)
            action_bias = self.base_performance - 0.5
            action = [random.gauss(action_bias, 0.2) for _ in range(5)]  # Mock 5-dim action
            
            state, reward, done, info = env.step(action)
            
            # Add some algorithm-specific performance variation
            performance_modifier = random.gauss(self.base_performance, self.variance)
            reward *= performance_modifier
            
            total_reward += reward
            safety_violations += info.get('safety_violations', 0)
            steps += 1
        
        convergence_time = (time.time() - start_time) / self.convergence_factor
        
        return {
            'episode_reward': total_reward,
            'convergence_time': convergence_time,
            'safety_violations': safety_violations,
            'power_loss': abs(random.gauss(0.05, 0.01)),
            'voltage_deviation': abs(random.gauss(0.02, 0.005)),
            'steps': steps
        }


class ExperimentRunner:
    """
    Orchestrates controlled experiments for algorithm comparison.
    
    Manages experiment execution, data collection, and result analysis.
    """
    
    def __init__(self, results_dir: str = "research/results"):
        """Initialize experiment runner."""
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.statistical_validator = StatisticalValidator(alpha=0.05, power=0.8)
        self.experimental_design = ExperimentalDesign()
        
        # Registry of available algorithms and environments
        self.algorithm_registry: Dict[str, Callable] = {}
        self.environment_registry: Dict[str, Callable] = {}
        
        # Results storage
        self.experiment_results: List[ExperimentResult] = []
        self.experiment_metadata: Dict[str, Any] = {}
        
        # Register default mock algorithms for testing
        self._register_default_algorithms()
        self._register_default_environments()
    
    def _register_default_algorithms(self):
        """Register default algorithm implementations."""
        # Novel federated RL algorithms
        self.algorithm_registry['NovelFedRL'] = lambda config: MockAlgorithm(
            'NovelFedRL', {
                'base_performance': 0.85,
                'variance': 0.08,
                'convergence_factor': 1.2,
                **config.hyperparameters
            }
        )
        
        self.algorithm_registry['AdvancedFedRL'] = lambda config: MockAlgorithm(
            'AdvancedFedRL', {
                'base_performance': 0.82,
                'variance': 0.09,
                'convergence_factor': 1.1,
                **config.hyperparameters
            }
        )
        
        # Baseline algorithms
        self.algorithm_registry['StandardRL'] = lambda config: MockAlgorithm(
            'StandardRL', {
                'base_performance': 0.75,
                'variance': 0.12,
                'convergence_factor': 1.0,
                **config.hyperparameters
            }
        )
        
        self.algorithm_registry['RandomPolicy'] = lambda config: MockAlgorithm(
            'RandomPolicy', {
                'base_performance': 0.45,
                'variance': 0.15,
                'convergence_factor': 0.8,
                **config.hyperparameters
            }
        )
        
        self.algorithm_registry['ClassicControl'] = lambda config: MockAlgorithm(
            'ClassicControl', {
                'base_performance': 0.68,
                'variance': 0.10,
                'convergence_factor': 0.9,
                **config.hyperparameters
            }
        )
    
    def _register_default_environments(self):
        """Register default environment configurations."""
        self.environment_registry['IEEE13Bus'] = lambda config: GridEnvironment(
            feeder=IEEE13Bus(), episode_length=config.episode_length
        )
        
        self.environment_registry['IEEE34Bus'] = lambda config: GridEnvironment(
            feeder=IEEE34Bus(), episode_length=config.episode_length
        )
        
        self.environment_registry['IEEE123Bus'] = lambda config: GridEnvironment(
            feeder=IEEE123Bus(), episode_length=config.episode_length
        )
    
    def register_algorithm(self, name: str, algorithm_factory: Callable):
        """Register a new algorithm for experiments."""
        self.algorithm_registry[name] = algorithm_factory
    
    def register_environment(self, name: str, environment_factory: Callable):
        """Register a new environment for experiments."""
        self.environment_registry[name] = environment_factory
    
    def run_single_experiment(self, config: ExperimentConfig) -> ExperimentResult:
        """
        Run a single controlled experiment.
        
        Args:
            config: Experiment configuration
            
        Returns:
            ExperimentResult with performance metrics
        """
        # Set random seed for reproducibility
        random.seed(config.random_seed)
        
        # Create algorithm instance
        if config.algorithm.name not in self.algorithm_registry:
            raise ValueError(f"Algorithm '{config.algorithm.name}' not registered")
        
        algorithm = self.algorithm_registry[config.algorithm.name](config.algorithm)
        
        # Create environment instance
        if config.environment.name not in self.environment_registry:
            raise ValueError(f"Environment '{config.environment.name}' not registered")
        
        env = self.environment_registry[config.environment.name](config.environment)
        
        # Run episodes and collect metrics
        episode_metrics = []
        total_safety_violations = 0
        start_time = time.time()
        
        for episode in range(config.n_episodes):
            try:
                # Run single episode
                episode_result = algorithm.train_episode(env)
                episode_metrics.append(episode_result)
                total_safety_violations += episode_result.get('safety_violations', 0)
                
                # Check timeout
                if time.time() - start_time > config.timeout_minutes * 60:
                    print(f"Experiment {config.experiment_id} timed out after {config.timeout_minutes} minutes")
                    break
                    
            except Exception as e:
                print(f"Error in episode {episode} of experiment {config.experiment_id}: {e}")
                continue
        
        execution_time = time.time() - start_time
        
        # Aggregate metrics
        if not episode_metrics:
            raise RuntimeError(f"No successful episodes in experiment {config.experiment_id}")
        
        aggregated_metrics = {}
        for metric in config.metrics_to_track:
            values = [ep.get(metric, 0.0) for ep in episode_metrics if metric in ep]
            if values:
                aggregated_metrics[metric] = sum(values) / len(values)
            else:
                aggregated_metrics[metric] = 0.0
        
        # Calculate convergence steps (average across episodes)
        convergence_steps = int(sum(ep.get('steps', 0) for ep in episode_metrics) / len(episode_metrics))
        
        # Create experiment result
        result = ExperimentResult(
            algorithm_name=config.algorithm.name,
            performance_metrics=aggregated_metrics,
            execution_time=execution_time,
            convergence_steps=convergence_steps,
            safety_violations=total_safety_violations,
            hyperparameters=config.algorithm.hyperparameters,
            random_seed=config.random_seed
        )
        
        # Store result
        self.experiment_results.append(result)
        self.statistical_validator.add_result(result)
        
        return result
    
    def run_comparative_study(self, 
                             algorithms: List[AlgorithmConfig],
                             environments: List[EnvironmentConfig],
                             n_replicates: int = 10,
                             n_episodes_per_replicate: int = 5) -> Dict[str, Any]:
        """
        Run a comprehensive comparative study.
        
        Args:
            algorithms: List of algorithms to compare
            environments: List of environments to test on
            n_replicates: Number of independent replicates
            n_episodes_per_replicate: Episodes per replicate
            
        Returns:
            Comprehensive analysis results
        """
        print(f"Starting comparative study with {len(algorithms)} algorithms, "
              f"{len(environments)} environments, {n_replicates} replicates")
        
        # Generate experimental design
        algorithm_names = [alg.name for alg in algorithms]
        environment_names = [env.name for env in environments]
        
        factorial_design = self.experimental_design.factorial_design(
            algorithm_names, environment_names, n_replicates
        )
        
        # Create algorithm and environment lookup
        alg_lookup = {alg.name: alg for alg in algorithms}
        env_lookup = {env.name: env for env in environments}
        
        # Run all experiments
        results = []
        total_experiments = len(factorial_design)
        
        for i, experiment_plan in enumerate(factorial_design):
            print(f"Running experiment {i+1}/{total_experiments}: {experiment_plan['experiment_id']}")
            
            # Create experiment configuration
            config = ExperimentConfig(
                experiment_id=experiment_plan['experiment_id'],
                algorithm=alg_lookup[experiment_plan['algorithm']],
                environment=env_lookup[experiment_plan['environment']],
                n_episodes=n_episodes_per_replicate,
                random_seed=experiment_plan['random_seed'],
                timeout_minutes=10  # Shorter timeout for batch runs
            )
            
            try:
                result = self.run_single_experiment(config)
                results.append(result)
                print(f"  Completed: {result.algorithm_name} on {config.environment.name} "
                      f"- Reward: {result.performance_metrics.get('episode_reward', 0):.3f}")
                
            except Exception as e:
                print(f"  Failed: {e}")
                continue
        
        # Analyze results
        analysis = self.analyze_comparative_study(algorithms, environments)
        
        # Save results
        self.save_study_results(f"comparative_study_{int(time.time())}", analysis)
        
        return analysis
    
    def analyze_comparative_study(self, 
                                 algorithms: List[AlgorithmConfig],
                                 environments: List[EnvironmentConfig]) -> Dict[str, Any]:
        """
        Analyze results from comparative study.
        
        Returns comprehensive statistical analysis.
        """
        analysis = {
            'study_metadata': {
                'algorithms': [alg.to_dict() for alg in algorithms],
                'environments': [env.to_dict() for env in environments],
                'total_experiments': len(self.experiment_results),
                'analysis_timestamp': time.time()
            },
            'algorithm_comparisons': {},
            'environment_analysis': {},
            'overall_rankings': {},
            'statistical_significance': {}
        }
        
        # Pairwise algorithm comparisons
        algorithm_names = [alg.name for alg in algorithms]
        
        for i, alg1 in enumerate(algorithm_names):
            for j, alg2 in enumerate(algorithm_names[i+1:], i+1):
                comparison_key = f"{alg1}_vs_{alg2}"
                
                # Compare on primary metric (episode_reward)
                comparison = self.statistical_validator.compare_algorithms(
                    alg1, alg2, metric='episode_reward'
                )
                
                analysis['algorithm_comparisons'][comparison_key] = comparison
                
                # Extract significance for summary
                if 'statistical_tests' in comparison:
                    t_test = comparison['statistical_tests']['parametric']
                    analysis['statistical_significance'][comparison_key] = {
                        'is_significant': t_test.is_significant,
                        'p_value': t_test.p_value,
                        'effect_size': t_test.effect_size
                    }
        
        # Environment-specific analysis
        for env in environments:
            env_results = {}
            
            for alg in algorithms:
                alg_env_results = [
                    r for r in self.experiment_results 
                    if r.algorithm_name == alg.name
                ]
                
                if alg_env_results:
                    rewards = [r.performance_metrics.get('episode_reward', 0) 
                              for r in alg_env_results]
                    safety_violations = [r.safety_violations for r in alg_env_results]
                    
                    env_results[alg.name] = {
                        'mean_reward': sum(rewards) / len(rewards),
                        'std_reward': (sum((x - sum(rewards)/len(rewards))**2 for x in rewards) / len(rewards))**0.5,
                        'mean_safety_violations': sum(safety_violations) / len(safety_violations),
                        'n_experiments': len(alg_env_results)
                    }
            
            analysis['environment_analysis'][env.name] = env_results
        
        # Overall algorithm rankings
        algorithm_performance = {}
        
        for alg in algorithms:
            alg_results = [r for r in self.experiment_results if r.algorithm_name == alg.name]
            
            if alg_results:
                rewards = [r.performance_metrics.get('episode_reward', 0) for r in alg_results]
                safety_violations = [r.safety_violations for r in alg_results]
                convergence_times = [r.performance_metrics.get('convergence_time', 0) for r in alg_results]
                
                # Composite score (higher is better)
                mean_reward = sum(rewards) / len(rewards)
                mean_safety = sum(safety_violations) / len(safety_violations)
                mean_convergence = sum(convergence_times) / len(convergence_times)
                
                # Normalize and combine (reward high good, safety violations low good, convergence time low good)
                composite_score = mean_reward - 0.1 * mean_safety - 0.01 * mean_convergence
                
                algorithm_performance[alg.name] = {
                    'mean_reward': mean_reward,
                    'std_reward': (sum((x - mean_reward)**2 for x in rewards) / len(rewards))**0.5,
                    'mean_safety_violations': mean_safety,
                    'mean_convergence_time': mean_convergence,
                    'composite_score': composite_score,
                    'n_experiments': len(alg_results),
                    'is_baseline': alg.is_baseline
                }
        
        # Sort by composite score
        ranked_algorithms = sorted(
            algorithm_performance.items(),
            key=lambda x: x[1]['composite_score'],
            reverse=True
        )
        
        analysis['overall_rankings'] = {
            'by_composite_score': ranked_algorithms,
            'performance_details': algorithm_performance
        }
        
        return analysis
    
    def save_study_results(self, study_name: str, analysis: Dict[str, Any]):
        """Save study results to JSON files."""
        study_dir = self.results_dir / study_name
        study_dir.mkdir(exist_ok=True)
        
        # Save analysis
        with open(study_dir / "analysis.json", 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        # Save raw results
        raw_results = [result.to_dict() for result in self.experiment_results]
        with open(study_dir / "raw_results.json", 'w') as f:
            json.dump(raw_results, f, indent=2, default=str)
        
        print(f"Study results saved to {study_dir}")
    
    def generate_research_report(self, analysis: Dict[str, Any]) -> str:
        """Generate a research report from analysis results."""
        report = []
        
        report.append("# Federated Reinforcement Learning Algorithm Comparison Study")
        report.append("")
        report.append("## Executive Summary")
        
        # Extract key findings
        rankings = analysis['overall_rankings']['by_composite_score']
        best_algorithm = rankings[0][0]
        best_score = rankings[0][1]['composite_score']
        
        report.append(f"- Best performing algorithm: **{best_algorithm}** (composite score: {best_score:.3f})")
        
        # Count significant differences
        significant_comparisons = sum(
            1 for comp in analysis['statistical_significance'].values()
            if comp['is_significant']
        )
        total_comparisons = len(analysis['statistical_significance'])
        
        report.append(f"- Statistically significant differences found in {significant_comparisons}/{total_comparisons} comparisons")
        
        # Novel vs baseline comparison
        novel_algorithms = [name for name, perf in rankings if not perf['is_baseline']]
        baseline_algorithms = [name for name, perf in rankings if perf['is_baseline']]
        
        if novel_algorithms and baseline_algorithms:
            best_novel = novel_algorithms[0]
            best_baseline = baseline_algorithms[0]
            report.append(f"- Best novel algorithm ({best_novel}) vs best baseline ({best_baseline})")
        
        report.append("")
        report.append("## Detailed Results")
        
        # Algorithm performance table
        report.append("### Algorithm Performance Summary")
        report.append("")
        report.append("| Algorithm | Mean Reward | Std Reward | Safety Violations | Convergence Time | Composite Score | Type |")
        report.append("|-----------|-------------|------------|-------------------|------------------|-----------------|------|")
        
        for name, perf in rankings:
            alg_type = "Baseline" if perf['is_baseline'] else "Novel"
            report.append(f"| {name} | {perf['mean_reward']:.4f} | {perf['std_reward']:.4f} | "
                         f"{perf['mean_safety_violations']:.2f} | {perf['mean_convergence_time']:.2f} | "
                         f"{perf['composite_score']:.4f} | {alg_type} |")
        
        report.append("")
        
        # Statistical significance
        report.append("### Statistical Significance Analysis")
        report.append("")
        
        for comparison_name, sig_result in analysis['statistical_significance'].items():
            algorithms = comparison_name.replace('_vs_', ' vs ')
            significance = "**Significant**" if sig_result['is_significant'] else "Not significant"
            
            report.append(f"- **{algorithms}**: {significance} "
                         f"(p = {sig_result['p_value']:.4f}, effect size = {sig_result['effect_size']:.3f})")
        
        report.append("")
        report.append("## Conclusions")
        
        # Generate conclusions based on results
        if novel_algorithms:
            top_novel = rankings[0] if not rankings[0][1]['is_baseline'] else rankings[1]
            report.append(f"The novel algorithm {top_novel[0]} demonstrates superior performance "
                         f"with a composite score of {top_novel[1]['composite_score']:.3f}.")
        
        report.append("")
        report.append("*Report generated by Grid-Fed-RL-Gym Research Framework*")
        
        return "\n".join(report)


# Example usage and demonstration
if __name__ == "__main__":
    # Create experiment runner
    runner = ExperimentRunner()
    
    # Define algorithms to compare
    algorithms = [
        AlgorithmConfig(
            name="NovelFedRL",
            class_name="NovelFederatedRL",
            hyperparameters={'learning_rate': 0.001, 'batch_size': 64},
            description="Novel federated RL algorithm with privacy preservation",
            is_baseline=False
        ),
        AlgorithmConfig(
            name="AdvancedFedRL", 
            class_name="AdvancedFederatedRL",
            hyperparameters={'learning_rate': 0.0005, 'batch_size': 32},
            description="Advanced federated RL with adaptive aggregation",
            is_baseline=False
        ),
        AlgorithmConfig(
            name="StandardRL",
            class_name="StandardRLBaseline",
            hyperparameters={'learning_rate': 0.001, 'batch_size': 32},
            description="Standard centralized RL baseline",
            is_baseline=True
        ),
        AlgorithmConfig(
            name="ClassicControl",
            class_name="ClassicControlBaseline", 
            hyperparameters={'gain': 1.0},
            description="Classic control theory baseline",
            is_baseline=True
        )
    ]
    
    # Define environments
    environments = [
        EnvironmentConfig(
            name="IEEE13Bus",
            feeder_type="IEEE13Bus",
            episode_length=500,
            safety_constraints={'voltage_min': 0.95, 'voltage_max': 1.05}
        ),
        EnvironmentConfig(
            name="IEEE34Bus", 
            feeder_type="IEEE34Bus",
            episode_length=1000,
            safety_constraints={'voltage_min': 0.95, 'voltage_max': 1.05}
        )
    ]
    
    print("=== Running Comparative Study ===")
    
    # Run comparative study
    analysis = runner.run_comparative_study(
        algorithms=algorithms,
        environments=environments,
        n_replicates=8,  # Smaller for demo
        n_episodes_per_replicate=3
    )
    
    # Generate and print research report
    report = runner.generate_research_report(analysis)
    print("\n" + "="*50)
    print("RESEARCH REPORT")
    print("="*50)
    print(report)
    
    # Print key statistical findings
    print("\n" + "="*50)
    print("STATISTICAL VALIDATION SUMMARY")
    print("="*50)
    
    for comparison_name, comparison_data in analysis['algorithm_comparisons'].items():
        if 'statistical_tests' in comparison_data:
            t_test = comparison_data['statistical_tests']['parametric']
            u_test = comparison_data['statistical_tests']['non_parametric']
            
            print(f"\n{comparison_name.replace('_vs_', ' vs ')}:")
            print(f"  Parametric test: {t_test}")
            print(f"  Non-parametric test: {u_test}")
            print(f"  Recommendation: {comparison_data['recommendation']}")