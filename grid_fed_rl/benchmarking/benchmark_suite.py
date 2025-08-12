"""Comprehensive benchmark suite for federated RL algorithms in power systems."""

from typing import Any, Dict, List, Optional, Tuple, Callable, Union
import numpy as np
import pandas as pd
import pickle
import json
import time
import logging
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib

from ..algorithms.base import BaseAlgorithm
from ..federated.core import FederatedOfflineRL, FedLearningConfig
from ..environments.grid_env import GridEnvironment
from ..feeders.ieee_feeders import IEEE13Bus, IEEE34Bus, IEEE123Bus


@dataclass
class TestCase:
    """Definition of a test case for benchmarking."""
    name: str
    description: str
    feeder_class: type
    num_agents: int = 5
    episode_length: int = 24 * 60  # 24 hours in minutes
    renewable_penetration: float = 0.3
    load_variability: float = 0.2
    difficulty_level: str = "medium"  # easy, medium, hard
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Scenario:
    """Renewable generation and load scenario."""
    name: str
    description: str
    renewable_profile: np.ndarray
    load_profile: np.ndarray
    weather_data: Optional[np.ndarray] = None
    disturbance_events: Optional[List[Dict]] = None
    duration_hours: int = 24


@dataclass
class ExperimentResult:
    """Results from a benchmark experiment."""
    algorithm_name: str
    test_case: str
    scenario: str
    seed: int
    performance_metrics: Dict[str, float]
    learning_metrics: Dict[str, Any]
    safety_metrics: Dict[str, float]
    economic_metrics: Dict[str, float]
    environmental_metrics: Dict[str, float]
    federated_metrics: Dict[str, Any]
    execution_time: float
    memory_usage: float
    convergence_step: Optional[int] = None
    final_policy_performance: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# Standard test cases
IEEE_TEST_CASES = {
    "ieee13_basic": TestCase(
        name="IEEE 13-Bus Basic",
        description="Basic IEEE 13-bus test feeder with standard loads",
        feeder_class=IEEE13Bus,
        num_agents=3,
        episode_length=1440,
        renewable_penetration=0.2,
        load_variability=0.1,
        difficulty_level="easy"
    ),
    
    "ieee13_high_renewable": TestCase(
        name="IEEE 13-Bus High Renewable",
        description="IEEE 13-bus with high renewable penetration and variability",
        feeder_class=IEEE13Bus,
        num_agents=3,
        episode_length=1440,
        renewable_penetration=0.6,
        load_variability=0.3,
        difficulty_level="hard",
        parameters={"renewable_uncertainty": 0.4, "weather_variation": True}
    ),
    
    "ieee34_medium": TestCase(
        name="IEEE 34-Bus Medium",
        description="IEEE 34-bus system with moderate complexity",
        feeder_class=IEEE34Bus,
        num_agents=5,
        episode_length=2880,  # 48 hours
        renewable_penetration=0.35,
        load_variability=0.25,
        difficulty_level="medium",
        parameters={"dynamic_pricing": True, "demand_response": True}
    ),
    
    "ieee123_complex": TestCase(
        name="IEEE 123-Bus Complex",
        description="Large IEEE 123-bus system with high complexity",
        feeder_class=IEEE123Bus,
        num_agents=10,
        episode_length=4320,  # 72 hours
        renewable_penetration=0.5,
        load_variability=0.3,
        difficulty_level="hard",
        parameters={
            "multi_objective": True,
            "uncertainty_modeling": True,
            "contingency_analysis": True
        }
    )
}

RENEWABLE_SCENARIOS = {
    "sunny_day": Scenario(
        name="Sunny Day",
        description="High solar generation with low variability",
        renewable_profile=np.concatenate([
            np.zeros(6),  # Night
            np.linspace(0, 0.8, 6),  # Morning ramp
            np.full(6, 0.8) + np.random.normal(0, 0.05, 6),  # Peak solar
            np.linspace(0.8, 0, 6),  # Evening ramp
            np.zeros(6)  # Night
        ]) * 1000,  # MW scale
        load_profile=np.concatenate([
            np.full(6, 0.6),  # Night base load
            np.linspace(0.6, 1.0, 6),  # Morning ramp
            np.full(6, 1.0),  # Day peak
            np.linspace(1.0, 0.8, 6),  # Evening
            np.full(6, 0.8)  # Evening/night
        ]) * 1200,  # MW scale
        duration_hours=24
    ),
    
    "windy_day": Scenario(
        name="Windy Day",
        description="High wind generation with moderate variability",
        renewable_profile=np.random.normal(0.6, 0.15, 24) * 800,  # Variable wind
        load_profile=np.concatenate([
            np.full(6, 0.5),
            np.linspace(0.5, 0.9, 6),
            np.full(6, 0.9),
            np.linspace(0.9, 0.6, 6)
        ]) * 1000,
        duration_hours=24
    ),
    
    "cloudy_variable": Scenario(
        name="Cloudy Variable",
        description="Variable cloud cover causing solar intermittency",
        renewable_profile=np.random.beta(2, 5, 24) * 600,  # Intermittent solar
        load_profile=np.full(24, 0.75) + np.random.normal(0, 0.1, 24) * 800,
        duration_hours=24,
        disturbance_events=[
            {"time": 10, "type": "cloud_cover", "magnitude": 0.8, "duration": 2},
            {"time": 15, "type": "wind_gust", "magnitude": 1.5, "duration": 1}
        ]
    ),
    
    "extreme_weather": Scenario(
        name="Extreme Weather",
        description="Extreme weather conditions with equipment failures",
        renewable_profile=np.clip(np.random.normal(0.3, 0.3, 24), 0, 1) * 400,
        load_profile=np.full(24, 1.2) + np.random.normal(0, 0.2, 24) * 1500,
        duration_hours=24,
        disturbance_events=[
            {"time": 8, "type": "line_outage", "component": "line_5", "duration": 4},
            {"time": 16, "type": "generator_failure", "component": "gen_2", "duration": 6}
        ]
    )
}

LOAD_PROFILES = {
    "residential": np.array([0.4, 0.3, 0.3, 0.3, 0.4, 0.5, 0.7, 0.8, 0.6, 0.5, 0.5, 0.6,
                            0.6, 0.6, 0.6, 0.7, 0.8, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]),
    
    "commercial": np.array([0.3, 0.2, 0.2, 0.3, 0.4, 0.6, 0.8, 0.9, 1.0, 1.0, 0.9, 0.9,
                           0.9, 0.9, 0.9, 0.8, 0.7, 0.5, 0.4, 0.4, 0.3, 0.3, 0.3, 0.3]),
    
    "industrial": np.array([0.8, 0.8, 0.8, 0.8, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 0.8, 0.7, 0.7, 0.7, 0.8, 0.8]),
    
    "mixed": np.array([0.5, 0.4, 0.4, 0.4, 0.5, 0.6, 0.8, 0.9, 0.8, 0.8, 0.7, 0.8,
                      0.8, 0.8, 0.8, 0.8, 0.8, 0.7, 0.6, 0.6, 0.5, 0.5, 0.5, 0.5])
}


class BenchmarkExperiment:
    """Individual benchmark experiment runner."""
    
    def __init__(
        self,
        algorithm_factory: Callable[[], BaseAlgorithm],
        algorithm_name: str,
        test_case: TestCase,
        scenario: Scenario,
        num_seeds: int = 5,
        max_training_steps: int = 10000,
        evaluation_episodes: int = 50,
        save_detailed_logs: bool = False,
        **experiment_kwargs
    ):
        self.algorithm_factory = algorithm_factory
        self.algorithm_name = algorithm_name
        self.test_case = test_case
        self.scenario = scenario
        self.num_seeds = num_seeds
        self.max_training_steps = max_training_steps
        self.evaluation_episodes = evaluation_episodes
        self.save_detailed_logs = save_detailed_logs
        self.experiment_kwargs = experiment_kwargs
        
        self.logger = logging.getLogger(__name__)
        
    def run_single_seed(self, seed: int) -> ExperimentResult:
        """Run experiment for a single random seed."""
        start_time = time.time()
        
        # Set random seeds for reproducibility
        np.random.seed(seed)
        if hasattr(np, 'random') and hasattr(np.random, 'default_rng'):
            rng = np.random.default_rng(seed)
        
        try:
            # Create environment
            feeder = self.test_case.feeder_class()
            env = GridEnvironment(
                feeder=feeder,
                episode_length=self.test_case.episode_length,
                renewable_sources=['solar', 'wind'] if self.test_case.renewable_penetration > 0 else [],
                **self.test_case.parameters
            )
            
            # Create algorithm
            algorithm = self.algorithm_factory()
            
            # Setup federated learning if applicable
            if hasattr(algorithm, 'num_clients'):
                fed_config = FedLearningConfig(
                    num_clients=self.test_case.num_agents,
                    rounds=self.max_training_steps // 100,
                    local_epochs=5
                )
                
                federated_learner = FederatedOfflineRL(
                    algorithm_class=type(algorithm),
                    config=fed_config
                )
            else:
                federated_learner = None
            
            # Training phase
            training_metrics = self._run_training(
                algorithm, env, federated_learner, seed
            )
            
            # Evaluation phase
            evaluation_metrics = self._run_evaluation(
                algorithm, env, self.evaluation_episodes, seed
            )
            
            # Collect comprehensive metrics
            result = self._collect_experiment_result(
                algorithm, training_metrics, evaluation_metrics, 
                start_time, seed
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Experiment failed for seed {seed}: {str(e)}")
            # Return failed result
            return ExperimentResult(
                algorithm_name=self.algorithm_name,
                test_case=self.test_case.name,
                scenario=self.scenario.name,
                seed=seed,
                performance_metrics={"success": False, "error": str(e)},
                learning_metrics={},
                safety_metrics={},
                economic_metrics={},
                environmental_metrics={},
                federated_metrics={},
                execution_time=time.time() - start_time,
                memory_usage=0.0
            )
    
    def _run_training(self, algorithm, env, federated_learner, seed):
        """Run training phase of the experiment."""
        training_metrics = {
            "convergence_step": None,
            "final_loss": float('inf'),
            "training_rewards": [],
            "learning_curves": {},
            "federated_rounds": 0
        }
        
        if federated_learner is not None:
            # Federated training
            try:
                # Create dummy datasets for federated clients
                datasets = []
                for client_idx in range(self.test_case.num_agents):
                    # Generate synthetic data based on scenario
                    dataset = self._generate_synthetic_dataset(env, 1000, seed + client_idx)
                    datasets.append(dataset)
                
                # Run federated training
                global_params = federated_learner.train(datasets)
                
                fed_metrics = federated_learner.get_training_metrics()
                training_metrics.update({
                    "federated_rounds": fed_metrics.get("total_rounds", 0),
                    "final_loss": fed_metrics.get("final_loss", float('inf')),
                    "convergence_step": fed_metrics.get("convergence_round", None)
                })
                
            except Exception as e:
                self.logger.warning(f"Federated training failed: {e}")
                training_metrics["error"] = str(e)
        else:
            # Standard single-agent training
            try:
                dataset = self._generate_synthetic_dataset(env, self.max_training_steps, seed)
                
                if hasattr(algorithm, 'train_offline'):
                    metrics_history = algorithm.train_offline(
                        dataset, num_epochs=100, batch_size=64
                    )
                    
                    if metrics_history:
                        losses = [m.loss for m in metrics_history if hasattr(m, 'loss')]
                        training_metrics.update({
                            "final_loss": losses[-1] if losses else float('inf'),
                            "learning_curves": {"loss": losses}
                        })
                        
                        # Find convergence point
                        if len(losses) > 10:
                            for i in range(10, len(losses)):
                                if np.std(losses[i-10:i]) < 0.01:
                                    training_metrics["convergence_step"] = i
                                    break
                                    
            except Exception as e:
                self.logger.warning(f"Training failed: {e}")
                training_metrics["error"] = str(e)
        
        return training_metrics
    
    def _run_evaluation(self, algorithm, env, num_episodes, seed):
        """Run evaluation phase of the experiment."""
        evaluation_metrics = {
            "episode_returns": [],
            "safety_violations": 0,
            "economic_costs": [],
            "environmental_impact": [],
            "success_rate": 0.0,
            "stability_metrics": {}
        }
        
        try:
            successful_episodes = 0
            
            for episode in range(num_episodes):
                # Reset environment
                obs, _ = env.reset(seed=seed + episode * 1000)
                episode_return = 0.0
                episode_violations = 0
                episode_cost = 0.0
                
                for step in range(self.test_case.episode_length):
                    # Get action from algorithm
                    if hasattr(algorithm, 'select_action'):
                        action = algorithm.select_action(obs, eval_mode=True)
                    else:
                        action = env.action_space.sample()  # Random fallback
                    
                    # Environment step
                    next_obs, reward, done, truncated, info = env.step(action)
                    
                    episode_return += reward
                    
                    # Track safety violations
                    if info.get('safety_violation', False):
                        episode_violations += 1
                    
                    # Track economic cost
                    if 'cost' in info:
                        episode_cost += info['cost']
                    
                    obs = next_obs
                    
                    if done or truncated:
                        break
                
                # Record episode metrics
                evaluation_metrics["episode_returns"].append(episode_return)
                evaluation_metrics["safety_violations"] += episode_violations
                evaluation_metrics["economic_costs"].append(episode_cost)
                
                # Environmental impact (simplified)
                renewable_utilization = getattr(env, 'renewable_utilization', 0.5)
                evaluation_metrics["environmental_impact"].append(renewable_utilization)
                
                if episode_violations == 0 and episode_return > -1000:  # Success criteria
                    successful_episodes += 1
            
            # Calculate summary statistics
            evaluation_metrics["success_rate"] = successful_episodes / num_episodes
            evaluation_metrics["mean_return"] = np.mean(evaluation_metrics["episode_returns"])
            evaluation_metrics["std_return"] = np.std(evaluation_metrics["episode_returns"])
            evaluation_metrics["mean_cost"] = np.mean(evaluation_metrics["economic_costs"])
            evaluation_metrics["mean_renewable"] = np.mean(evaluation_metrics["environmental_impact"])
            
        except Exception as e:
            self.logger.warning(f"Evaluation failed: {e}")
            evaluation_metrics["error"] = str(e)
        
        return evaluation_metrics
    
    def _generate_synthetic_dataset(self, env, num_samples, seed):
        """Generate synthetic dataset for training."""
        from ..algorithms.base import collect_random_data, GridDataset
        
        # Collect random data from environment
        data = collect_random_data(env, num_samples)
        
        # Create dataset
        dataset = GridDataset(
            observations=data["observations"],
            actions=data["actions"],
            rewards=data["rewards"],
            next_observations=data["next_observations"],
            terminals=data["terminals"]
        )
        
        return dataset
    
    def _collect_experiment_result(
        self, 
        algorithm, 
        training_metrics, 
        evaluation_metrics, 
        start_time, 
        seed
    ) -> ExperimentResult:
        """Collect comprehensive experiment result."""
        import psutil
        import os
        
        execution_time = time.time() - start_time
        
        # Memory usage
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        
        # Performance metrics
        performance_metrics = {
            "mean_return": evaluation_metrics.get("mean_return", 0.0),
            "std_return": evaluation_metrics.get("std_return", 0.0),
            "success_rate": evaluation_metrics.get("success_rate", 0.0),
            "final_loss": training_metrics.get("final_loss", float('inf'))
        }
        
        # Learning metrics
        learning_metrics = {
            "convergence_step": training_metrics.get("convergence_step"),
            "training_time": execution_time,
            "learning_curves": training_metrics.get("learning_curves", {})
        }
        
        # Safety metrics
        safety_metrics = {
            "total_violations": evaluation_metrics.get("safety_violations", 0),
            "violation_rate": evaluation_metrics.get("safety_violations", 0) / 
                            max(1, len(evaluation_metrics.get("episode_returns", [1]))),
            "safety_score": 1.0 - min(1.0, evaluation_metrics.get("safety_violations", 0) / 100)
        }
        
        # Economic metrics
        economic_metrics = {
            "mean_cost": evaluation_metrics.get("mean_cost", 0.0),
            "cost_efficiency": 1.0 / max(1.0, evaluation_metrics.get("mean_cost", 1.0)),
            "cost_stability": 1.0 / max(1.0, np.std(evaluation_metrics.get("economic_costs", [1.0])))
        }
        
        # Environmental metrics
        environmental_metrics = {
            "renewable_utilization": evaluation_metrics.get("mean_renewable", 0.0),
            "carbon_reduction": evaluation_metrics.get("mean_renewable", 0.0) * 0.8,  # Simplified
            "sustainability_score": evaluation_metrics.get("mean_renewable", 0.0)
        }
        
        # Federated metrics
        federated_metrics = {}
        if hasattr(algorithm, 'get_federated_metrics'):
            federated_metrics = algorithm.get_federated_metrics()
        else:
            federated_metrics = {
                "communication_rounds": training_metrics.get("federated_rounds", 0),
                "aggregation_method": "fedavg",  # Default
                "privacy_preserved": True
            }
        
        return ExperimentResult(
            algorithm_name=self.algorithm_name,
            test_case=self.test_case.name,
            scenario=self.scenario.name,
            seed=seed,
            performance_metrics=performance_metrics,
            learning_metrics=learning_metrics,
            safety_metrics=safety_metrics,
            economic_metrics=economic_metrics,
            environmental_metrics=environmental_metrics,
            federated_metrics=federated_metrics,
            execution_time=execution_time,
            memory_usage=memory_usage,
            convergence_step=training_metrics.get("convergence_step"),
            final_policy_performance=evaluation_metrics.get("mean_return", 0.0)
        )
    
    def run(self, parallel: bool = True, n_jobs: int = -1) -> List[ExperimentResult]:
        """Run the complete experiment across all seeds."""
        self.logger.info(
            f"Running benchmark: {self.algorithm_name} on {self.test_case.name} "
            f"with {self.scenario.name} scenario ({self.num_seeds} seeds)"
        )
        
        if parallel and n_jobs != 1:
            # Parallel execution
            if n_jobs == -1:
                n_jobs = min(mp.cpu_count(), self.num_seeds)
            
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                futures = [
                    executor.submit(self.run_single_seed, seed) 
                    for seed in range(self.num_seeds)
                ]
                
                results = []
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        self.logger.error(f"Parallel execution failed: {e}")
                        
        else:
            # Sequential execution
            results = []
            for seed in range(self.num_seeds):
                result = self.run_single_seed(seed)
                results.append(result)
                
                # Log progress
                if (seed + 1) % max(1, self.num_seeds // 5) == 0:
                    self.logger.info(f"Completed {seed + 1}/{self.num_seeds} seeds")
        
        self.logger.info(f"Benchmark completed: {len(results)} results collected")
        return results


class BenchmarkSuite:
    """Comprehensive benchmark suite for federated RL algorithms."""
    
    def __init__(
        self,
        output_dir: str = "benchmark_results",
        save_raw_data: bool = True,
        parallel_execution: bool = True,
        n_jobs: int = -1
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_raw_data = save_raw_data
        self.parallel_execution = parallel_execution
        self.n_jobs = n_jobs
        
        self.experiments = []
        self.results = []
        
        self.logger = logging.getLogger(__name__)
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging for benchmark suite."""
        log_file = self.output_dir / "benchmark.log"
        
        # Create file handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        
        # Create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        # Add handlers to logger
        if not self.logger.handlers:
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)
            self.logger.setLevel(logging.INFO)
    
    def add_algorithm(
        self,
        algorithm_factory: Callable[[], BaseAlgorithm],
        algorithm_name: str,
        test_cases: Optional[List[str]] = None,
        scenarios: Optional[List[str]] = None,
        num_seeds: int = 5,
        **experiment_kwargs
    ):
        """Add algorithm to benchmark suite."""
        
        # Use default test cases if none specified
        if test_cases is None:
            test_cases = list(IEEE_TEST_CASES.keys())
        
        # Use default scenarios if none specified
        if scenarios is None:
            scenarios = list(RENEWABLE_SCENARIOS.keys())
        
        # Create experiments for all combinations
        for test_case_name in test_cases:
            if test_case_name not in IEEE_TEST_CASES:
                self.logger.warning(f"Unknown test case: {test_case_name}")
                continue
                
            test_case = IEEE_TEST_CASES[test_case_name]
            
            for scenario_name in scenarios:
                if scenario_name not in RENEWABLE_SCENARIOS:
                    self.logger.warning(f"Unknown scenario: {scenario_name}")
                    continue
                    
                scenario = RENEWABLE_SCENARIOS[scenario_name]
                
                experiment = BenchmarkExperiment(
                    algorithm_factory=algorithm_factory,
                    algorithm_name=algorithm_name,
                    test_case=test_case,
                    scenario=scenario,
                    num_seeds=num_seeds,
                    **experiment_kwargs
                )
                
                self.experiments.append(experiment)
        
        self.logger.info(
            f"Added {len(self.experiments)} experiments for algorithm: {algorithm_name}"
        )
    
    def run_benchmarks(self, save_intermediate: bool = True) -> List[ExperimentResult]:
        """Run all benchmark experiments."""
        self.logger.info(f"Starting benchmark suite with {len(self.experiments)} experiments")
        
        all_results = []
        
        for i, experiment in enumerate(self.experiments):
            self.logger.info(
                f"Running experiment {i+1}/{len(self.experiments)}: "
                f"{experiment.algorithm_name} on {experiment.test_case.name}"
            )
            
            try:
                # Run experiment
                results = experiment.run(
                    parallel=self.parallel_execution,
                    n_jobs=self.n_jobs
                )
                
                all_results.extend(results)
                
                # Save intermediate results
                if save_intermediate:
                    self._save_intermediate_results(experiment, results, i)
                    
            except Exception as e:
                self.logger.error(f"Experiment {i+1} failed: {str(e)}")
                continue
        
        self.results = all_results
        
        # Save final results
        self._save_final_results()
        
        self.logger.info(f"Benchmark suite completed: {len(all_results)} total results")
        return all_results
    
    def _save_intermediate_results(self, experiment, results, experiment_idx):
        """Save intermediate results for single experiment."""
        if not self.save_raw_data:
            return
        
        filename = (
            f"{experiment.algorithm_name}_{experiment.test_case.name}_"
            f"{experiment.scenario.name}_{experiment_idx}.pkl"
        )
        filepath = self.output_dir / "intermediate" / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'experiment': experiment,
                'results': results,
                'timestamp': time.time()
            }, f)
    
    def _save_final_results(self):
        """Save final aggregated results."""
        if not self.results:
            return
        
        # Save raw results
        if self.save_raw_data:
            with open(self.output_dir / "all_results.pkl", 'wb') as f:
                pickle.dump(self.results, f)
        
        # Create summary DataFrame
        summary_data = []
        for result in self.results:
            summary_data.append({
                'algorithm': result.algorithm_name,
                'test_case': result.test_case,
                'scenario': result.scenario,
                'seed': result.seed,
                'mean_return': result.performance_metrics.get('mean_return', 0.0),
                'std_return': result.performance_metrics.get('std_return', 0.0),
                'success_rate': result.performance_metrics.get('success_rate', 0.0),
                'safety_score': result.safety_metrics.get('safety_score', 0.0),
                'cost_efficiency': result.economic_metrics.get('cost_efficiency', 0.0),
                'renewable_utilization': result.environmental_metrics.get('renewable_utilization', 0.0),
                'execution_time': result.execution_time,
                'memory_usage': result.memory_usage,
                'convergence_step': result.convergence_step
            })
        
        df = pd.DataFrame(summary_data)
        df.to_csv(self.output_dir / "benchmark_summary.csv", index=False)
        
        # Save experiment metadata
        metadata = {
            'num_experiments': len(self.experiments),
            'num_results': len(self.results),
            'algorithms': list(set(r.algorithm_name for r in self.results)),
            'test_cases': list(set(r.test_case for r in self.results)),
            'scenarios': list(set(r.scenario for r in self.results)),
            'timestamp': time.time(),
            'output_directory': str(self.output_dir)
        }
        
        with open(self.output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Results saved to {self.output_dir}")
    
    def load_results(self, results_dir: Optional[str] = None) -> List[ExperimentResult]:
        """Load previously saved results."""
        if results_dir is None:
            results_dir = self.output_dir
        else:
            results_dir = Path(results_dir)
        
        results_file = results_dir / "all_results.pkl"
        
        if not results_file.exists():
            self.logger.warning(f"No results file found at {results_file}")
            return []
        
        with open(results_file, 'rb') as f:
            self.results = pickle.load(f)
        
        self.logger.info(f"Loaded {len(self.results)} results from {results_file}")
        return self.results
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics across all experiments."""
        if not self.results:
            return {}
        
        df = pd.DataFrame([{
            'algorithm': r.algorithm_name,
            'test_case': r.test_case,
            'scenario': r.scenario,
            'mean_return': r.performance_metrics.get('mean_return', 0.0),
            'success_rate': r.performance_metrics.get('success_rate', 0.0),
            'safety_score': r.safety_metrics.get('safety_score', 0.0),
            'execution_time': r.execution_time
        } for r in self.results])
        
        summary = {
            'total_experiments': len(self.results),
            'algorithms': df['algorithm'].unique().tolist(),
            'test_cases': df['test_case'].unique().tolist(),
            'scenarios': df['scenario'].unique().tolist(),
            
            # Performance statistics
            'performance_by_algorithm': df.groupby('algorithm').agg({
                'mean_return': ['mean', 'std', 'min', 'max'],
                'success_rate': ['mean', 'std'],
                'safety_score': ['mean', 'std'],
                'execution_time': ['mean', 'std']
            }).round(4).to_dict(),
            
            # Best performing algorithms
            'best_overall': df.groupby('algorithm')['mean_return'].mean().idxmax(),
            'most_reliable': df.groupby('algorithm')['success_rate'].mean().idxmax(),
            'safest': df.groupby('algorithm')['safety_score'].mean().idxmax(),
            'fastest': df.groupby('algorithm')['execution_time'].mean().idxmin(),
        }
        
        return summary