"""
Statistical Validation Framework for Novel Federated RL Algorithms

This module provides comprehensive statistical validation and significance testing
for novel algorithms in the Grid-Fed-RL-Gym framework.

Author: Daniel Schmidt <daniel@terragonlabs.com>
"""

import time
import math
import random
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict

# Minimal dependencies approach
try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False
    # Use built-in math for statistical operations
    import math


@dataclass
class ExperimentResult:
    """Container for experimental results with metadata."""
    algorithm_name: str
    performance_metrics: Dict[str, float]
    execution_time: float
    convergence_steps: int
    safety_violations: int
    hyperparameters: Dict[str, Any]
    random_seed: int
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'algorithm_name': self.algorithm_name,
            'performance_metrics': self.performance_metrics,
            'execution_time': self.execution_time,
            'convergence_steps': self.convergence_steps,
            'safety_violations': self.safety_violations,
            'hyperparameters': self.hyperparameters,
            'random_seed': self.random_seed,
            'timestamp': self.timestamp
        }


@dataclass
class StatisticalTest:
    """Results of a statistical significance test."""
    test_name: str
    statistic: float
    p_value: float
    critical_value: float
    confidence_level: float
    is_significant: bool
    effect_size: float
    interpretation: str
    
    def __str__(self) -> str:
        significance = "significant" if self.is_significant else "not significant"
        return (f"{self.test_name}: statistic={self.statistic:.4f}, "
                f"p={self.p_value:.4f} ({significance}), "
                f"effect_size={self.effect_size:.4f}")


class StatisticalValidator:
    """
    Comprehensive statistical validation framework for novel algorithms.
    
    Provides multiple statistical tests, confidence intervals, and effect size
    calculations for rigorous algorithm comparison.
    """
    
    def __init__(self, alpha: float = 0.05, power: float = 0.8):
        """
        Initialize statistical validator.
        
        Args:
            alpha: Significance level (Type I error rate)
            power: Statistical power (1 - Type II error rate)
        """
        self.alpha = alpha
        self.power = power
        self.results_cache: Dict[str, List[ExperimentResult]] = defaultdict(list)
        
    def add_result(self, result: ExperimentResult) -> None:
        """Add experimental result to the cache."""
        self.results_cache[result.algorithm_name].append(result)
    
    def calculate_mean(self, values: List[float]) -> float:
        """Calculate arithmetic mean."""
        if not values:
            return 0.0
        return sum(values) / len(values)
    
    def calculate_variance(self, values: List[float]) -> float:
        """Calculate sample variance."""
        if len(values) < 2:
            return 0.0
        
        mean = self.calculate_mean(values)
        squared_diffs = [(x - mean) ** 2 for x in values]
        return sum(squared_diffs) / (len(values) - 1)
    
    def calculate_std(self, values: List[float]) -> float:
        """Calculate sample standard deviation."""
        return math.sqrt(self.calculate_variance(values))
    
    def calculate_sem(self, values: List[float]) -> float:
        """Calculate standard error of the mean."""
        if len(values) < 2:
            return float('inf')
        return self.calculate_std(values) / math.sqrt(len(values))
    
    def welch_t_test(self, group1: List[float], group2: List[float]) -> StatisticalTest:
        """
        Perform Welch's t-test for unequal variances.
        
        More robust than Student's t-test when variances are unequal.
        """
        n1, n2 = len(group1), len(group2)
        
        if n1 < 2 or n2 < 2:
            return StatisticalTest(
                test_name="Welch's t-test",
                statistic=0.0,
                p_value=1.0,
                critical_value=float('inf'),
                confidence_level=1-self.alpha,
                is_significant=False,
                effect_size=0.0,
                interpretation="Insufficient data for testing"
            )
        
        mean1 = self.calculate_mean(group1)
        mean2 = self.calculate_mean(group2)
        var1 = self.calculate_variance(group1)
        var2 = self.calculate_variance(group2)
        
        # Welch's t-statistic
        pooled_se = math.sqrt(var1/n1 + var2/n2)
        if pooled_se == 0:
            t_stat = float('inf') if mean1 != mean2 else 0.0
        else:
            t_stat = (mean1 - mean2) / pooled_se
        
        # Welch-Satterthwaite degrees of freedom
        if var1 == 0 and var2 == 0:
            df = n1 + n2 - 2
        else:
            numerator = (var1/n1 + var2/n2) ** 2
            denominator = (var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1)
            df = numerator / denominator if denominator > 0 else n1 + n2 - 2
        
        # Approximate p-value using t-distribution
        # For simplicity, using normal approximation for large df
        if df > 30:
            p_value = 2 * (1 - self._normal_cdf(abs(t_stat)))
            critical_value = self._normal_inverse_cdf(1 - self.alpha/2)
        else:
            # Simple approximation for small df
            p_value = 2 * (1 - self._t_cdf(abs(t_stat), df))
            critical_value = self._t_critical(self.alpha/2, df)
        
        # Cohen's d effect size
        pooled_std = math.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2)) if n1+n2 > 2 else 1.0
        cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0.0
        
        is_significant = abs(t_stat) > critical_value
        
        # Interpretation
        if abs(cohens_d) < 0.2:
            effect_interpretation = "negligible"
        elif abs(cohens_d) < 0.5:
            effect_interpretation = "small"
        elif abs(cohens_d) < 0.8:
            effect_interpretation = "medium"
        else:
            effect_interpretation = "large"
        
        interpretation = f"Effect size is {effect_interpretation} (Cohen's d = {cohens_d:.3f})"
        
        return StatisticalTest(
            test_name="Welch's t-test",
            statistic=t_stat,
            p_value=p_value,
            critical_value=critical_value,
            confidence_level=1-self.alpha,
            is_significant=is_significant,
            effect_size=cohens_d,
            interpretation=interpretation
        )
    
    def _normal_cdf(self, x: float) -> float:
        """Approximate normal cumulative distribution function."""
        # Abramowitz and Stegun approximation
        if x < 0:
            return 1 - self._normal_cdf(-x)
        
        # Constants for approximation
        a1, a2, a3, a4, a5 = 0.31938153, -0.356563782, 1.781477937, -1.821255978, 1.330274429
        k = 1 / (1 + 0.2316419 * x)
        w = k * (a1 + k * (a2 + k * (a3 + k * (a4 + k * a5))))
        return 1 - math.exp(-x*x/2) / math.sqrt(2*math.pi) * w
    
    def _normal_inverse_cdf(self, p: float) -> float:
        """Approximate inverse normal CDF (quantile function)."""
        if p <= 0 or p >= 1:
            return float('inf') if p >= 1 else float('-inf')
        
        # Beasley-Springer-Moro algorithm approximation
        a0, a1, a2, a3 = -3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02
        b1, b2, b3, b4 = -5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01
        c0, c1, c2, c3 = -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00
        d1, d2 = 7.784695709041462e-03, 3.224671290700398e-01
        
        if p > 0.5:
            p = 1 - p
            sign = -1
        else:
            sign = 1
        
        if p >= 1e-20:
            y = math.sqrt(-2 * math.log(p))
            x = (((c3*y + c2)*y + c1)*y + c0) / ((d2*y + d1)*y + 1)
        else:
            x = 9.0  # Large value for extreme tail
        
        return sign * x
    
    def _t_cdf(self, t: float, df: float) -> float:
        """Approximate t-distribution CDF."""
        if df > 30:
            return self._normal_cdf(t)
        
        # Simple approximation for t-distribution
        # For small df, use normal approximation with adjustment
        adjustment = 1 + (t*t) / (4*df)
        return self._normal_cdf(t / math.sqrt(adjustment))
    
    def _t_critical(self, alpha: float, df: float) -> float:
        """Approximate critical value for t-distribution."""
        if df > 30:
            return self._normal_inverse_cdf(1 - alpha)
        
        # Simple lookup table for common critical values
        critical_values = {
            1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
            10: 2.228, 15: 2.131, 20: 2.086, 25: 2.060, 30: 2.042
        }
        
        # Find closest df in lookup table
        closest_df = min(critical_values.keys(), key=lambda x: abs(x - df))
        base_critical = critical_values[closest_df]
        
        # Adjust for alpha (this is very approximate)
        if alpha != 0.025:  # Default is for alpha=0.05 (two-tailed)
            adjustment = math.log(0.025 / alpha) * 0.1
            base_critical += adjustment
        
        return base_critical
    
    def mann_whitney_u_test(self, group1: List[float], group2: List[float]) -> StatisticalTest:
        """
        Perform Mann-Whitney U test (non-parametric alternative to t-test).
        
        More robust when normality assumptions are violated.
        """
        n1, n2 = len(group1), len(group2)
        
        if n1 == 0 or n2 == 0:
            return StatisticalTest(
                test_name="Mann-Whitney U test",
                statistic=0.0,
                p_value=1.0,
                critical_value=0.0,
                confidence_level=1-self.alpha,
                is_significant=False,
                effect_size=0.0,
                interpretation="No data available"
            )
        
        # Combine and rank all values
        combined = [(val, 1) for val in group1] + [(val, 2) for val in group2]
        combined.sort(key=lambda x: x[0])
        
        # Assign ranks (handle ties by averaging)
        ranks = []
        i = 0
        while i < len(combined):
            j = i
            while j < len(combined) and combined[j][0] == combined[i][0]:
                j += 1
            # Average rank for tied values
            avg_rank = (i + j + 1) / 2
            for k in range(i, j):
                ranks.append((avg_rank, combined[k][1]))
            i = j
        
        # Calculate rank sums
        r1 = sum(rank for rank, group in ranks if group == 1)
        r2 = sum(rank for rank, group in ranks if group == 2)
        
        # Calculate U statistics
        u1 = r1 - n1 * (n1 + 1) / 2
        u2 = r2 - n2 * (n2 + 1) / 2
        u_stat = min(u1, u2)
        
        # Expected value and variance for normal approximation
        mu_u = n1 * n2 / 2
        sigma_u = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
        
        # Z-score for normal approximation
        if sigma_u > 0:
            z_score = (u_stat - mu_u) / sigma_u
        else:
            z_score = 0.0
        
        # Two-tailed p-value
        p_value = 2 * (1 - self._normal_cdf(abs(z_score)))
        
        # Critical value
        critical_value = self._normal_inverse_cdf(1 - self.alpha/2)
        
        # Effect size (r = Z / sqrt(N))
        effect_size = abs(z_score) / math.sqrt(n1 + n2) if n1 + n2 > 0 else 0.0
        
        is_significant = abs(z_score) > critical_value
        
        # Interpretation
        if effect_size < 0.1:
            effect_interpretation = "negligible"
        elif effect_size < 0.3:
            effect_interpretation = "small"
        elif effect_size < 0.5:
            effect_interpretation = "medium"
        else:
            effect_interpretation = "large"
        
        interpretation = f"Effect size is {effect_interpretation} (r = {effect_size:.3f})"
        
        return StatisticalTest(
            test_name="Mann-Whitney U test",
            statistic=u_stat,
            p_value=p_value,
            critical_value=critical_value,
            confidence_level=1-self.alpha,
            is_significant=is_significant,
            effect_size=effect_size,
            interpretation=interpretation
        )
    
    def confidence_interval(self, values: List[float], confidence_level: float = None) -> Tuple[float, float, float]:
        """
        Calculate confidence interval for the mean.
        
        Returns:
            Tuple of (lower_bound, mean, upper_bound)
        """
        if confidence_level is None:
            confidence_level = 1 - self.alpha
        
        if len(values) < 2:
            mean_val = self.calculate_mean(values)
            return (mean_val, mean_val, mean_val)
        
        mean_val = self.calculate_mean(values)
        sem = self.calculate_sem(values)
        
        # Use t-distribution for small samples, normal for large
        if len(values) > 30:
            critical_value = self._normal_inverse_cdf(1 - (1-confidence_level)/2)
        else:
            critical_value = self._t_critical((1-confidence_level)/2, len(values)-1)
        
        margin_error = critical_value * sem
        
        return (mean_val - margin_error, mean_val, mean_val + margin_error)
    
    def power_analysis(self, effect_size: float, sample_size: int) -> float:
        """
        Calculate statistical power for given effect size and sample size.
        
        Returns the probability of detecting an effect if it exists.
        """
        if sample_size <= 1:
            return 0.0
        
        # Simplified power calculation for t-test
        # Power = P(reject H0 | H1 is true)
        
        # Critical value under null hypothesis
        critical_value = self._normal_inverse_cdf(1 - self.alpha/2)
        
        # Non-centrality parameter
        ncp = effect_size * math.sqrt(sample_size / 2)
        
        # Power is probability that test statistic exceeds critical value
        # under alternative hypothesis
        power = 1 - self._normal_cdf(critical_value - ncp)
        power += self._normal_cdf(-critical_value - ncp)  # Two-tailed test
        
        return min(1.0, max(0.0, power))
    
    def sample_size_calculation(self, effect_size: float, power: float = None) -> int:
        """
        Calculate required sample size for desired power and effect size.
        """
        if power is None:
            power = self.power
        
        if effect_size <= 0 or power <= 0 or power >= 1:
            return float('inf')
        
        # Simplified calculation for t-test
        # Based on Cohen's power analysis
        
        z_alpha = self._normal_inverse_cdf(1 - self.alpha/2)
        z_beta = self._normal_inverse_cdf(power)
        
        # Sample size formula for two-sample t-test
        n_per_group = 2 * ((z_alpha + z_beta) / effect_size) ** 2
        
        return max(2, int(math.ceil(n_per_group)))
    
    def compare_algorithms(self, algorithm1: str, algorithm2: str, 
                          metric: str = 'reward') -> Dict[str, Any]:
        """
        Comprehensive comparison between two algorithms.
        
        Returns detailed statistical analysis including multiple tests
        and effect sizes.
        """
        if algorithm1 not in self.results_cache or algorithm2 not in self.results_cache:
            return {
                'error': f'Insufficient data for algorithms: {algorithm1}, {algorithm2}',
                'available_algorithms': list(self.results_cache.keys())
            }
        
        # Extract metric values
        values1 = [r.performance_metrics.get(metric, 0.0) 
                  for r in self.results_cache[algorithm1]]
        values2 = [r.performance_metrics.get(metric, 0.0) 
                  for r in self.results_cache[algorithm2]]
        
        if not values1 or not values2:
            return {
                'error': f'No {metric} data available for comparison',
                'algorithm1_samples': len(values1),
                'algorithm2_samples': len(values2)
            }
        
        # Descriptive statistics
        stats1 = self._descriptive_stats(values1, algorithm1)
        stats2 = self._descriptive_stats(values2, algorithm2)
        
        # Statistical tests
        t_test = self.welch_t_test(values1, values2)
        u_test = self.mann_whitney_u_test(values1, values2)
        
        # Confidence intervals
        ci1 = self.confidence_interval(values1)
        ci2 = self.confidence_interval(values2)
        
        # Power analysis
        observed_power = self.power_analysis(abs(t_test.effect_size), min(len(values1), len(values2)))
        recommended_n = self.sample_size_calculation(abs(t_test.effect_size))
        
        return {
            'comparison': f'{algorithm1} vs {algorithm2}',
            'metric': metric,
            'descriptive_stats': {
                algorithm1: stats1,
                algorithm2: stats2
            },
            'statistical_tests': {
                'parametric': t_test,
                'non_parametric': u_test
            },
            'confidence_intervals': {
                algorithm1: {
                    'lower': ci1[0],
                    'mean': ci1[1], 
                    'upper': ci1[2]
                },
                algorithm2: {
                    'lower': ci2[0],
                    'mean': ci2[1],
                    'upper': ci2[2]
                }
            },
            'power_analysis': {
                'observed_power': observed_power,
                'recommended_sample_size': recommended_n,
                'current_sample_sizes': [len(values1), len(values2)]
            },
            'recommendation': self._generate_recommendation(t_test, u_test, observed_power, recommended_n, min(len(values1), len(values2)))
        }
    
    def _descriptive_stats(self, values: List[float], algorithm_name: str) -> Dict[str, float]:
        """Calculate descriptive statistics for a set of values."""
        if not values:
            return {}
        
        sorted_values = sorted(values)
        n = len(values)
        
        # Percentiles
        def percentile(p):
            k = (n - 1) * p / 100
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                return sorted_values[int(k)]
            else:
                d0 = sorted_values[int(f)] * (c - k)
                d1 = sorted_values[int(c)] * (k - f)
                return d0 + d1
        
        return {
            'count': n,
            'mean': self.calculate_mean(values),
            'std': self.calculate_std(values),
            'min': min(values),
            'max': max(values),
            'median': percentile(50),
            'q25': percentile(25),
            'q75': percentile(75),
            'iqr': percentile(75) - percentile(25),
            'skewness': self._calculate_skewness(values),
            'kurtosis': self._calculate_kurtosis(values)
        }
    
    def _calculate_skewness(self, values: List[float]) -> float:
        """Calculate sample skewness."""
        if len(values) < 3:
            return 0.0
        
        mean = self.calculate_mean(values)
        std = self.calculate_std(values)
        
        if std == 0:
            return 0.0
        
        n = len(values)
        skew_sum = sum(((x - mean) / std) ** 3 for x in values)
        return (n / ((n - 1) * (n - 2))) * skew_sum
    
    def _calculate_kurtosis(self, values: List[float]) -> float:
        """Calculate sample kurtosis (excess kurtosis)."""
        if len(values) < 4:
            return 0.0
        
        mean = self.calculate_mean(values)
        std = self.calculate_std(values)
        
        if std == 0:
            return 0.0
        
        n = len(values)
        kurt_sum = sum(((x - mean) / std) ** 4 for x in values)
        
        # Excess kurtosis (normal distribution has kurtosis = 0)
        kurt = (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))) * kurt_sum
        kurt -= 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
        
        return kurt
    
    def _generate_recommendation(self, t_test: StatisticalTest, u_test: StatisticalTest, 
                                observed_power: float, recommended_n: int, current_n: int) -> str:
        """Generate interpretation and recommendations."""
        recommendations = []
        
        # Significance interpretation
        if t_test.is_significant and u_test.is_significant:
            recommendations.append("Both parametric and non-parametric tests show significant difference.")
        elif t_test.is_significant or u_test.is_significant:
            recommendations.append("Results are mixed - one test shows significance while the other doesn't.")
        else:
            recommendations.append("No significant difference detected by either test.")
        
        # Effect size interpretation
        if abs(t_test.effect_size) < 0.2:
            recommendations.append("Effect size is negligible - practical significance is questionable.")
        elif abs(t_test.effect_size) > 0.8:
            recommendations.append("Large effect size detected - difference is practically significant.")
        
        # Power analysis
        if observed_power < 0.8:
            recommendations.append(f"Statistical power is low ({observed_power:.2f}). Consider collecting more data.")
            recommendations.append(f"Recommended sample size: {recommended_n} per group (current: {current_n})")
        
        # Test agreement
        if abs(t_test.p_value - u_test.p_value) > 0.1:
            recommendations.append("Large discrepancy between parametric and non-parametric tests suggests non-normal data.")
        
        return " ".join(recommendations)


class ExperimentalDesign:
    """
    Experimental design framework for controlled algorithm comparison.
    
    Implements various experimental designs to ensure valid statistical inference.
    """
    
    def __init__(self, random_seed: int = 42):
        """Initialize experimental design framework."""
        self.random_seed = random_seed
        self.experiments: List[Dict[str, Any]] = []
        
    def generate_random_seeds(self, n_replicates: int) -> List[int]:
        """Generate reproducible random seeds for experiments."""
        random.seed(self.random_seed)
        return [random.randint(1, 1000000) for _ in range(n_replicates)]
    
    def factorial_design(self, algorithms: List[str], environments: List[str], 
                        n_replicates: int = 10) -> List[Dict[str, Any]]:
        """
        Generate factorial experimental design.
        
        Creates all combinations of algorithms and environments with replicates.
        """
        experiments = []
        seeds = self.generate_random_seeds(len(algorithms) * len(environments) * n_replicates)
        seed_idx = 0
        
        for algorithm in algorithms:
            for environment in environments:
                for replicate in range(n_replicates):
                    experiments.append({
                        'experiment_id': f'{algorithm}_{environment}_rep{replicate}',
                        'algorithm': algorithm,
                        'environment': environment,
                        'replicate': replicate,
                        'random_seed': seeds[seed_idx],
                        'status': 'pending'
                    })
                    seed_idx += 1
        
        # Randomize execution order
        random.seed(self.random_seed)
        random.shuffle(experiments)
        
        self.experiments = experiments
        return experiments
    
    def blocked_design(self, algorithms: List[str], blocks: List[Dict[str, Any]], 
                      n_replicates: int = 5) -> List[Dict[str, Any]]:
        """
        Generate randomized block design.
        
        Controls for known sources of variation by grouping similar conditions.
        """
        experiments = []
        seeds = self.generate_random_seeds(len(algorithms) * len(blocks) * n_replicates)
        seed_idx = 0
        
        for block_id, block_params in enumerate(blocks):
            # Randomize algorithm order within each block
            random.seed(self.random_seed + block_id)
            block_algorithms = algorithms.copy()
            
            for replicate in range(n_replicates):
                random.shuffle(block_algorithms)
                
                for algorithm in block_algorithms:
                    experiments.append({
                        'experiment_id': f'block{block_id}_{algorithm}_rep{replicate}',
                        'algorithm': algorithm,
                        'block_id': block_id,
                        'block_params': block_params,
                        'replicate': replicate,
                        'random_seed': seeds[seed_idx],
                        'status': 'pending'
                    })
                    seed_idx += 1
        
        self.experiments = experiments
        return experiments
    
    def latin_square_design(self, algorithms: List[str], conditions: List[str]) -> List[Dict[str, Any]]:
        """
        Generate Latin square design for controlling two sources of variation.
        
        Each algorithm appears exactly once in each row and column.
        """
        n = len(algorithms)
        if len(conditions) != n:
            raise ValueError("Latin square requires equal number of algorithms and conditions")
        
        # Generate Latin square
        latin_square = []
        for i in range(n):
            row = []
            for j in range(n):
                row.append(algorithms[(i + j) % n])
            latin_square.append(row)
        
        experiments = []
        seeds = self.generate_random_seeds(n * n)
        seed_idx = 0
        
        for row_idx, condition in enumerate(conditions):
            for col_idx in range(n):
                algorithm = latin_square[row_idx][col_idx]
                experiments.append({
                    'experiment_id': f'ls_r{row_idx}_c{col_idx}_{algorithm}',
                    'algorithm': algorithm,
                    'condition': condition,
                    'row': row_idx,
                    'column': col_idx,
                    'random_seed': seeds[seed_idx],
                    'status': 'pending'
                })
                seed_idx += 1
        
        self.experiments = experiments
        return experiments
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get summary of current experimental design."""
        if not self.experiments:
            return {'message': 'No experiments designed yet'}
        
        algorithms = set(exp['algorithm'] for exp in self.experiments)
        total_experiments = len(self.experiments)
        
        # Count by status
        status_counts = {}
        for exp in self.experiments:
            status = exp['status']
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            'total_experiments': total_experiments,
            'unique_algorithms': len(algorithms),
            'algorithms': sorted(list(algorithms)),
            'status_breakdown': status_counts,
            'design_seed': self.random_seed
        }


# Example usage and testing functionality
if __name__ == "__main__":
    # Create statistical validator
    validator = StatisticalValidator(alpha=0.05, power=0.8)
    
    # Simulate some experimental results
    def simulate_algorithm_results(base_performance: float, variance: float, n_runs: int = 20):
        """Simulate algorithm performance results."""
        results = []
        for i in range(n_runs):
            # Add some noise to base performance
            noise = random.gauss(0, variance)
            performance = base_performance + noise
            
            result = ExperimentResult(
                algorithm_name=f"Algorithm_{base_performance}",
                performance_metrics={
                    'reward': performance,
                    'convergence_time': abs(random.gauss(100, 20)),
                    'safety_score': min(1.0, max(0.0, random.gauss(0.9, 0.1)))
                },
                execution_time=abs(random.gauss(120, 30)),
                convergence_steps=int(abs(random.gauss(1000, 200))),
                safety_violations=random.randint(0, 5),
                hyperparameters={'lr': 0.001, 'batch_size': 32},
                random_seed=random.randint(1, 1000000)
            )
            results.append(result)
        return results
    
    # Simulate results for two algorithms
    random.seed(42)
    
    # Algorithm A: Better performance
    results_a = simulate_algorithm_results(base_performance=0.85, variance=0.1, n_runs=25)
    for result in results_a:
        result.algorithm_name = "NovelFedRL"
        validator.add_result(result)
    
    # Algorithm B: Baseline performance  
    results_b = simulate_algorithm_results(base_performance=0.75, variance=0.12, n_runs=25)
    for result in results_b:
        result.algorithm_name = "StandardRL"
        validator.add_result(result)
    
    # Compare algorithms
    comparison = validator.compare_algorithms("NovelFedRL", "StandardRL", metric="reward")
    
    print("=== Statistical Validation Results ===")
    print(f"Comparison: {comparison['comparison']}")
    print(f"Metric: {comparison['metric']}")
    print()
    
    print("Descriptive Statistics:")
    for alg, stats in comparison['descriptive_stats'].items():
        print(f"  {alg}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, n={stats['count']}")
    print()
    
    print("Statistical Tests:")
    print(f"  {comparison['statistical_tests']['parametric']}")
    print(f"  {comparison['statistical_tests']['non_parametric']}")
    print()
    
    print("Power Analysis:")
    power_info = comparison['power_analysis']
    print(f"  Observed power: {power_info['observed_power']:.3f}")
    print(f"  Recommended sample size: {power_info['recommended_sample_size']}")
    print(f"  Current sample sizes: {power_info['current_sample_sizes']}")
    print()
    
    print("Recommendation:")
    print(f"  {comparison['recommendation']}")
    print()
    
    # Demonstrate experimental design
    design = ExperimentalDesign(random_seed=42)
    
    algorithms = ["NovelFedRL", "StandardRL", "BaselineRL"]
    environments = ["IEEE13Bus", "IEEE34Bus", "IEEE123Bus"]
    
    factorial_experiments = design.factorial_design(algorithms, environments, n_replicates=5)
    summary = design.get_experiment_summary()
    
    print("=== Experimental Design ===")
    print(f"Total experiments planned: {summary['total_experiments']}")
    print(f"Algorithms: {summary['algorithms']}")
    print(f"First 5 experiments:")
    for i, exp in enumerate(factorial_experiments[:5]):
        print(f"  {i+1}: {exp['experiment_id']} (seed: {exp['random_seed']})")