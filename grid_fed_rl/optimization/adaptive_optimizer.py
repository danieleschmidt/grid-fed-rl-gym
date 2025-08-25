"""Adaptive optimization engine for grid simulation parameters."""

import time
import json
import logging
from typing import Any, Dict, List, Optional, Callable, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from collections import deque
import statistics


class OptimizationStrategy(Enum):
    """Optimization strategies."""
    GREEDY = "greedy"
    GRADIENT_DESCENT = "gradient_descent"  
    GENETIC_ALGORITHM = "genetic_algorithm"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    ADAPTIVE = "adaptive"


class MetricTrend(Enum):
    """Metric trend directions."""
    IMPROVING = "improving"
    DEGRADING = "degrading"
    STABLE = "stable"
    UNKNOWN = "unknown"


@dataclass
class OptimizationResult:
    """Result of optimization operation."""
    parameter_set: Dict[str, Any]
    performance_score: float
    execution_time: float
    iteration: int
    improvement_ratio: float
    metadata: Dict[str, Any]


@dataclass
class PerformanceMetric:
    """Performance metric tracking."""
    name: str
    value: float
    timestamp: float
    context: Dict[str, Any]


class AdaptiveOptimizer:
    """Intelligent parameter optimization with adaptive strategies."""
    
    def __init__(
        self,
        objective_function: Callable[[Dict[str, Any]], float],
        parameter_bounds: Dict[str, Tuple[Union[int, float], Union[int, float]]],
        maximize: bool = True,
        history_size: int = 1000,
        adaptation_interval: int = 50
    ):
        self.objective_function = objective_function
        self.parameter_bounds = parameter_bounds
        self.maximize = maximize
        self.history_size = history_size
        self.adaptation_interval = adaptation_interval
        
        self.logger = logging.getLogger(__name__)
        self.lock = threading.RLock()
        
        # Optimization history
        self.optimization_history: deque = deque(maxlen=history_size)
        self.performance_metrics: Dict[str, deque] = {}
        self.best_result: Optional[OptimizationResult] = None
        
        # Adaptive strategy management
        self.current_strategy = OptimizationStrategy.ADAPTIVE
        self.strategy_performance: Dict[OptimizationStrategy, List[float]] = {}
        self.iteration_count = 0
        
        # Parameter adaptation
        self.parameter_sensitivity: Dict[str, float] = {}
        self.parameter_correlation: Dict[Tuple[str, str], float] = {}
        
        # Initialize parameter bounds validation
        self._validate_bounds()
    
    def _validate_bounds(self):
        """Validate parameter bounds."""
        for param, (lower, upper) in self.parameter_bounds.items():
            if not isinstance(lower, (int, float)) or not isinstance(upper, (int, float)):
                raise ValueError(f"Parameter bounds for '{param}' must be numeric")
            if lower >= upper:
                raise ValueError(f"Parameter bounds for '{param}': lower >= upper ({lower} >= {upper})")
    
    def _generate_random_parameters(self) -> Dict[str, Any]:
        """Generate random parameters within bounds."""
        import random
        
        parameters = {}
        for param, (lower, upper) in self.parameter_bounds.items():
            if isinstance(lower, int) and isinstance(upper, int):
                parameters[param] = random.randint(lower, upper)
            else:
                parameters[param] = random.uniform(float(lower), float(upper))
        
        return parameters
    
    def _mutate_parameters(self, parameters: Dict[str, Any], mutation_rate: float = 0.1) -> Dict[str, Any]:
        """Mutate parameters for genetic algorithm."""
        import random
        
        mutated = parameters.copy()
        
        for param, value in parameters.items():
            if random.random() < mutation_rate:
                lower, upper = self.parameter_bounds[param]
                
                # Adaptive mutation based on sensitivity
                sensitivity = self.parameter_sensitivity.get(param, 1.0)
                mutation_strength = mutation_rate * sensitivity
                
                if isinstance(lower, int) and isinstance(upper, int):
                    # Integer parameters
                    range_size = upper - lower
                    mutation_delta = int(range_size * mutation_strength * (random.random() - 0.5))
                    mutated[param] = max(lower, min(upper, value + mutation_delta))
                else:
                    # Float parameters
                    range_size = upper - lower
                    mutation_delta = range_size * mutation_strength * (random.random() - 0.5)
                    mutated[param] = max(lower, min(upper, value + mutation_delta))
        
        return mutated
    
    def _crossover_parameters(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Crossover operation for genetic algorithm."""
        import random
        
        child1 = {}
        child2 = {}
        
        for param in parent1:
            if random.random() < 0.5:
                child1[param] = parent1[param]
                child2[param] = parent2[param]
            else:
                child1[param] = parent2[param] 
                child2[param] = parent1[param]
        
        return child1, child2
    
    def _gradient_step(self, parameters: Dict[str, Any], step_size: float = 0.01) -> Dict[str, Any]:
        """Perform gradient descent step."""
        
        # Estimate gradients using finite differences
        gradients = {}
        base_score = self.objective_function(parameters)
        
        for param, value in parameters.items():
            lower, upper = self.parameter_bounds[param]
            
            # Small perturbation
            delta = (upper - lower) * 0.001
            
            # Forward difference
            perturbed_params = parameters.copy()
            perturbed_params[param] = min(upper, value + delta)
            forward_score = self.objective_function(perturbed_params)
            
            # Calculate gradient
            gradient = (forward_score - base_score) / delta
            gradients[param] = gradient
        
        # Update parameters
        updated_params = {}
        for param, value in parameters.items():
            gradient = gradients[param]
            lower, upper = self.parameter_bounds[param]
            
            # Adaptive step size based on sensitivity
            sensitivity = self.parameter_sensitivity.get(param, 1.0)
            adaptive_step = step_size / (sensitivity + 1e-6)
            
            if self.maximize:
                new_value = value + adaptive_step * gradient
            else:
                new_value = value - adaptive_step * gradient
            
            # Ensure bounds
            updated_params[param] = max(lower, min(upper, new_value))
        
        return updated_params
    
    def _bayesian_optimization_step(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Simple Bayesian optimization using exploitation/exploration."""
        
        if len(self.optimization_history) < 5:
            # Not enough data, use random exploration
            return self._generate_random_parameters()
        
        # Use recent history to build simple model
        recent_results = list(self.optimization_history)[-20:]
        
        # Find best performing parameters
        if self.maximize:
            best_recent = max(recent_results, key=lambda x: x.performance_score)
        else:
            best_recent = min(recent_results, key=lambda x: x.performance_score)
        
        # Exploitation: small perturbation around best
        if len(recent_results) % 3 == 0:  # Exploration every 3rd iteration
            # Exploration: larger random perturbation
            perturbation_strength = 0.3
        else:
            # Exploitation: small perturbation
            perturbation_strength = 0.05
        
        import random
        optimized_params = {}
        
        for param, best_value in best_recent.parameter_set.items():
            lower, upper = self.parameter_bounds[param]
            range_size = upper - lower
            
            perturbation = range_size * perturbation_strength * (random.random() - 0.5)
            new_value = best_value + perturbation
            
            # Ensure bounds
            optimized_params[param] = max(lower, min(upper, new_value))
        
        return optimized_params
    
    def _analyze_parameter_sensitivity(self):
        """Analyze parameter sensitivity from optimization history."""
        
        if len(self.optimization_history) < 10:
            return
        
        recent_results = list(self.optimization_history)[-50:]  # Use recent results
        
        for param in self.parameter_bounds:
            # Calculate correlation between parameter value and performance
            param_values = [r.parameter_set[param] for r in recent_results]
            scores = [r.performance_score for r in recent_results]
            
            if len(set(param_values)) > 1:  # Need variation
                try:
                    correlation = self._calculate_correlation(param_values, scores)
                    self.parameter_sensitivity[param] = abs(correlation)
                except:
                    self.parameter_sensitivity[param] = 1.0
            else:
                self.parameter_sensitivity[param] = 1.0
    
    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient."""
        
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi * xi for xi in x)
        sum_y2 = sum(yi * yi for yi in y)
        
        numerator = n * sum_xy - sum_x * sum_y
        denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)) ** 0.5
        
        if abs(denominator) < 1e-10:
            return 0.0
        
        return numerator / denominator
    
    def _adapt_strategy(self):
        """Adapt optimization strategy based on performance."""
        
        if self.iteration_count % self.adaptation_interval != 0:
            return
        
        if len(self.optimization_history) < self.adaptation_interval:
            return
        
        # Analyze recent performance trends
        recent_results = list(self.optimization_history)[-self.adaptation_interval:]
        scores = [r.performance_score for r in recent_results]
        
        # Calculate improvement trend
        if len(scores) >= 3:
            recent_trend = statistics.mean(scores[-5:]) - statistics.mean(scores[:5]) if len(scores) >= 10 else 0
            
            # Adapt strategy based on trend
            if abs(recent_trend) < 0.01:  # Stagnation
                # Try different strategy
                strategies = [s for s in OptimizationStrategy if s != self.current_strategy and s != OptimizationStrategy.ADAPTIVE]
                if strategies:
                    import random
                    self.current_strategy = random.choice(strategies)
                    self.logger.info(f"Adapted to strategy: {self.current_strategy.value}")
            
            # Update parameter sensitivity
            self._analyze_parameter_sensitivity()
    
    def optimize_step(
        self,
        strategy: Optional[OptimizationStrategy] = None,
        population_size: int = 10
    ) -> OptimizationResult:
        """Perform single optimization step."""
        
        with self.lock:
            start_time = time.time()
            
            # Use current strategy if not specified
            if strategy is None:
                strategy = self.current_strategy
            
            # Generate candidate parameters
            if strategy == OptimizationStrategy.GREEDY:
                if self.best_result:
                    # Greedy local search around best result
                    candidates = [self._mutate_parameters(self.best_result.parameter_set, 0.05)]
                else:
                    candidates = [self._generate_random_parameters()]
                    
            elif strategy == OptimizationStrategy.GRADIENT_DESCENT:
                if self.best_result:
                    candidates = [self._gradient_step(self.best_result.parameter_set)]
                else:
                    candidates = [self._generate_random_parameters()]
                    
            elif strategy == OptimizationStrategy.GENETIC_ALGORITHM:
                if len(self.optimization_history) >= 2:
                    # Select parents
                    recent = list(self.optimization_history)[-population_size:]
                    if self.maximize:
                        parents = sorted(recent, key=lambda x: x.performance_score, reverse=True)[:2]
                    else:
                        parents = sorted(recent, key=lambda x: x.performance_score)[:2]
                    
                    # Generate offspring
                    child1, child2 = self._crossover_parameters(
                        parents[0].parameter_set,
                        parents[1].parameter_set
                    )
                    
                    candidates = [
                        self._mutate_parameters(child1),
                        self._mutate_parameters(child2)
                    ]
                else:
                    candidates = [self._generate_random_parameters() for _ in range(2)]
                    
            elif strategy == OptimizationStrategy.BAYESIAN_OPTIMIZATION:
                candidates = [self._bayesian_optimization_step({})]
                
            else:  # ADAPTIVE
                # Mix different strategies
                strategies = [OptimizationStrategy.GREEDY, OptimizationStrategy.BAYESIAN_OPTIMIZATION]
                candidates = []
                for s in strategies:
                    try:
                        result = self.optimize_step(s, population_size)
                        candidates.append(result.parameter_set)
                        break
                    except:
                        continue
                
                if not candidates:
                    candidates = [self._generate_random_parameters()]
            
            # Evaluate candidates
            best_candidate = None
            best_score = float('-inf') if self.maximize else float('inf')
            
            for candidate in candidates:
                try:
                    score = self.objective_function(candidate)
                    
                    if (self.maximize and score > best_score) or (not self.maximize and score < best_score):
                        best_score = score
                        best_candidate = candidate
                        
                except Exception as e:
                    self.logger.warning(f"Objective function evaluation failed: {e}")
                    continue
            
            if best_candidate is None:
                # Fallback to random
                best_candidate = self._generate_random_parameters()
                best_score = self.objective_function(best_candidate)
            
            # Calculate improvement
            if self.best_result:
                if self.maximize:
                    improvement = (best_score - self.best_result.performance_score) / abs(self.best_result.performance_score + 1e-6)
                else:
                    improvement = (self.best_result.performance_score - best_score) / abs(self.best_result.performance_score + 1e-6)
            else:
                improvement = 0.0
            
            # Create result
            execution_time = time.time() - start_time
            self.iteration_count += 1
            
            result = OptimizationResult(
                parameter_set=best_candidate,
                performance_score=best_score,
                execution_time=execution_time,
                iteration=self.iteration_count,
                improvement_ratio=improvement,
                metadata={
                    "strategy": strategy.value,
                    "candidates_evaluated": len(candidates),
                    "timestamp": time.time()
                }
            )
            
            # Update history and best result
            self.optimization_history.append(result)
            
            if self.best_result is None or (
                (self.maximize and best_score > self.best_result.performance_score) or
                (not self.maximize and best_score < self.best_result.performance_score)
            ):
                self.best_result = result
            
            # Adaptive strategy adjustment
            self._adapt_strategy()
            
            return result
    
    def optimize(
        self,
        max_iterations: int = 100,
        target_score: Optional[float] = None,
        patience: int = 20,
        strategy: Optional[OptimizationStrategy] = None
    ) -> OptimizationResult:
        """Run full optimization process."""
        
        self.logger.info(f"Starting optimization: {max_iterations} max iterations")
        
        best_result = None
        no_improvement_count = 0
        
        for iteration in range(max_iterations):
            try:
                result = self.optimize_step(strategy)
                
                # Check for improvement
                if best_result is None or (
                    (self.maximize and result.performance_score > best_result.performance_score) or
                    (not self.maximize and result.performance_score < best_result.performance_score)
                ):
                    best_result = result
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                
                # Log progress
                if iteration % 10 == 0:
                    self.logger.info(
                        f"Iteration {iteration}: score={result.performance_score:.4f}, "
                        f"best={best_result.performance_score:.4f}, "
                        f"strategy={result.metadata['strategy']}"
                    )
                
                # Check termination conditions
                if target_score is not None:
                    if (self.maximize and result.performance_score >= target_score) or \
                       (not self.maximize and result.performance_score <= target_score):
                        self.logger.info(f"Target score reached: {result.performance_score}")
                        break
                
                if no_improvement_count >= patience:
                    self.logger.info(f"Early stopping: no improvement for {patience} iterations")
                    break
                    
            except Exception as e:
                self.logger.error(f"Optimization step failed: {e}")
                continue
        
        return best_result or self.best_result
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        
        if not self.optimization_history:
            return {"message": "No optimization history available"}
        
        scores = [r.performance_score for r in self.optimization_history]
        execution_times = [r.execution_time for r in self.optimization_history]
        
        return {
            "total_iterations": len(self.optimization_history),
            "best_score": max(scores) if self.maximize else min(scores),
            "current_score": scores[-1] if scores else 0,
            "score_improvement": scores[-1] - scores[0] if len(scores) >= 2 else 0,
            "avg_execution_time": statistics.mean(execution_times),
            "total_time": sum(execution_times),
            "parameter_sensitivity": dict(self.parameter_sensitivity),
            "strategy_usage": {
                strategy.value: len([r for r in self.optimization_history 
                                   if r.metadata.get('strategy') == strategy.value])
                for strategy in OptimizationStrategy
            }
        }


def optimize_function(
    objective_function: Callable[[Dict[str, Any]], float],
    parameter_bounds: Dict[str, Tuple[Union[int, float], Union[int, float]]],
    max_iterations: int = 50,
    maximize: bool = True
) -> Tuple[Dict[str, Any], float]:
    """Simple optimization function."""
    
    optimizer = AdaptiveOptimizer(
        objective_function=objective_function,
        parameter_bounds=parameter_bounds,
        maximize=maximize
    )
    
    result = optimizer.optimize(max_iterations=max_iterations)
    
    return result.parameter_set, result.performance_score


if __name__ == "__main__":
    # Test optimization
    def test_objective(params: Dict[str, Any]) -> float:
        """Test objective function: maximize -(x-2)^2 - (y-3)^2."""
        x = params.get('x', 0)
        y = params.get('y', 0)
        return -(x - 2)**2 - (y - 3)**2
    
    bounds = {
        'x': (-10.0, 10.0),
        'y': (-10.0, 10.0)
    }
    
    optimizer = AdaptiveOptimizer(
        objective_function=test_objective,
        parameter_bounds=bounds,
        maximize=True
    )
    
    # Run optimization
    result = optimizer.optimize(max_iterations=100)
    
    print(f"Optimization completed:")
    print(f"Best parameters: {result.parameter_set}")
    print(f"Best score: {result.performance_score:.4f}")
    print(f"Iterations: {result.iteration}")
    
    # Print statistics
    stats = optimizer.get_optimization_stats()
    print(f"\nOptimization stats:")
    print(json.dumps(stats, indent=2))