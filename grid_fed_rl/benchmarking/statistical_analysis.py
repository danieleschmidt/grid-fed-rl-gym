"""Statistical analysis tools for benchmarking federated RL algorithms."""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import (
    ttest_ind, mannwhitneyu, wilcoxon, kruskal, friedmanchisquare,
    normaltest, levene, bartlett, shapiro
)
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.contingency_tables import mcnemar
import warnings
from dataclasses import dataclass
from enum import Enum
import logging

from .benchmark_suite import ExperimentResult


class TestType(Enum):
    """Types of statistical tests."""
    PARAMETRIC = "parametric"
    NON_PARAMETRIC = "non_parametric"
    PERMUTATION = "permutation"
    BOOTSTRAP = "bootstrap"


@dataclass
class SignificanceTest:
    """Result of a statistical significance test."""
    test_name: str
    statistic: float
    p_value: float
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    interpretation: Optional[str] = None
    assumptions_met: Optional[Dict[str, bool]] = None
    power: Optional[float] = None
    
    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if result is statistically significant."""
        return self.p_value < alpha
    
    def get_interpretation(self, alpha: float = 0.05) -> str:
        """Get human-readable interpretation."""
        if self.interpretation:
            return self.interpretation
        
        if self.is_significant(alpha):
            significance = "statistically significant"
        else:
            significance = "not statistically significant"
        
        effect_desc = ""
        if self.effect_size is not None:
            if abs(self.effect_size) < 0.2:
                effect_desc = " (small effect)"
            elif abs(self.effect_size) < 0.5:
                effect_desc = " (medium effect)"
            elif abs(self.effect_size) < 0.8:
                effect_desc = " (large effect)"
            else:
                effect_desc = " (very large effect)"
        
        return f"Result is {significance} (p={self.p_value:.4f}){effect_desc}"


@dataclass
class EffectSize:
    """Effect size calculation result."""
    name: str
    value: float
    interpretation: str
    confidence_interval: Optional[Tuple[float, float]] = None


class EffectSizeCalculator:
    """Calculate various effect size measures."""
    
    @staticmethod
    def cohens_d(group1: np.ndarray, group2: np.ndarray) -> EffectSize:
        """Calculate Cohen's d effect size."""
        n1, n2 = len(group1), len(group2)
        pooled_std = np.sqrt(((n1 - 1) * np.var(group1, ddof=1) + 
                             (n2 - 1) * np.var(group2, ddof=1)) / (n1 + n2 - 2))
        
        d = (np.mean(group1) - np.mean(group2)) / pooled_std
        
        # Interpretation
        if abs(d) < 0.2:
            interpretation = "negligible"
        elif abs(d) < 0.5:
            interpretation = "small"
        elif abs(d) < 0.8:
            interpretation = "medium"
        else:
            interpretation = "large"
        
        return EffectSize(
            name="Cohen's d",
            value=d,
            interpretation=interpretation
        )
    
    @staticmethod
    def hedges_g(group1: np.ndarray, group2: np.ndarray) -> EffectSize:
        """Calculate Hedges' g effect size (bias-corrected Cohen's d)."""
        cohens_d = EffectSizeCalculator.cohens_d(group1, group2)
        n1, n2 = len(group1), len(group2)
        
        # Bias correction factor
        j = 1 - (3 / (4 * (n1 + n2 - 2) - 1))
        g = cohens_d.value * j
        
        return EffectSize(
            name="Hedges' g",
            value=g,
            interpretation=cohens_d.interpretation
        )
    
    @staticmethod
    def glass_delta(group1: np.ndarray, group2: np.ndarray) -> EffectSize:
        """Calculate Glass's delta effect size."""
        delta = (np.mean(group1) - np.mean(group2)) / np.std(group2, ddof=1)
        
        if abs(delta) < 0.2:
            interpretation = "small"
        elif abs(delta) < 0.5:
            interpretation = "medium"
        else:
            interpretation = "large"
        
        return EffectSize(
            name="Glass's delta",
            value=delta,
            interpretation=interpretation
        )
    
    @staticmethod
    def cliff_delta(group1: np.ndarray, group2: np.ndarray) -> EffectSize:
        """Calculate Cliff's delta (non-parametric effect size)."""
        n1, n2 = len(group1), len(group2)
        
        # Count dominance
        dominance = 0
        for x1 in group1:
            for x2 in group2:
                if x1 > x2:
                    dominance += 1
                elif x1 < x2:
                    dominance -= 1
        
        delta = dominance / (n1 * n2)
        
        # Interpretation for Cliff's delta
        if abs(delta) < 0.11:
            interpretation = "negligible"
        elif abs(delta) < 0.28:
            interpretation = "small"
        elif abs(delta) < 0.43:
            interpretation = "medium"
        else:
            interpretation = "large"
        
        return EffectSize(
            name="Cliff's delta",
            value=delta,
            interpretation=interpretation
        )
    
    @staticmethod
    def vargha_delaney_a(group1: np.ndarray, group2: np.ndarray) -> EffectSize:
        """Calculate Vargha-Delaney A measure."""
        n1, n2 = len(group1), len(group2)
        
        # Count wins and ties
        wins = 0
        ties = 0
        
        for x1 in group1:
            for x2 in group2:
                if x1 > x2:
                    wins += 1
                elif x1 == x2:
                    ties += 1
        
        a = (wins + 0.5 * ties) / (n1 * n2)
        
        # Interpretation
        if a < 0.44:
            interpretation = "small (favors group 2)"
        elif a < 0.56:
            interpretation = "negligible"
        elif a < 0.64:
            interpretation = "small (favors group 1)"
        elif a < 0.71:
            interpretation = "medium (favors group 1)"
        else:
            interpretation = "large (favors group 1)"
        
        return EffectSize(
            name="Vargha-Delaney A",
            value=a,
            interpretation=interpretation
        )


class BootstrapAnalysis:
    """Bootstrap-based statistical analysis."""
    
    def __init__(self, n_bootstrap: int = 10000, confidence_level: float = 0.95):
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.logger = logging.getLogger(__name__)
    
    def bootstrap_mean_difference(
        self, 
        group1: np.ndarray, 
        group2: np.ndarray
    ) -> Dict[str, Any]:
        """Bootstrap test for difference in means."""
        observed_diff = np.mean(group1) - np.mean(group2)
        
        # Combined sample for null hypothesis
        combined = np.concatenate([group1, group2])
        n1, n2 = len(group1), len(group2)
        
        # Bootstrap resampling under null hypothesis
        bootstrap_diffs = []
        
        for _ in range(self.n_bootstrap):
            # Resample without replacement
            resampled = np.random.choice(combined, size=len(combined), replace=False)
            boot_group1 = resampled[:n1]
            boot_group2 = resampled[n1:]
            
            boot_diff = np.mean(boot_group1) - np.mean(boot_group2)
            bootstrap_diffs.append(boot_diff)
        
        bootstrap_diffs = np.array(bootstrap_diffs)
        
        # Calculate p-value (two-tailed)
        p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff))
        
        # Confidence interval
        alpha = 1 - self.confidence_level
        ci_lower = np.percentile(bootstrap_diffs, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_diffs, 100 * (1 - alpha / 2))
        
        return {
            'observed_difference': observed_diff,
            'p_value': p_value,
            'confidence_interval': (ci_lower, ci_upper),
            'bootstrap_distribution': bootstrap_diffs,
            'null_hypothesis': 'No difference in means'
        }
    
    def bootstrap_confidence_interval(
        self,
        data: np.ndarray,
        statistic_func: callable = np.mean
    ) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval for any statistic."""
        bootstrap_stats = []
        
        for _ in range(self.n_bootstrap):
            bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_stat = statistic_func(bootstrap_sample)
            bootstrap_stats.append(bootstrap_stat)
        
        bootstrap_stats = np.array(bootstrap_stats)
        
        alpha = 1 - self.confidence_level
        ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
        
        return ci_lower, ci_upper
    
    def permutation_test(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        statistic_func: callable = None
    ) -> Dict[str, Any]:
        """Permutation test for comparing two groups."""
        if statistic_func is None:
            statistic_func = lambda x, y: np.mean(x) - np.mean(y)
        
        observed_stat = statistic_func(group1, group2)
        
        # Combined data
        combined = np.concatenate([group1, group2])
        n1 = len(group1)
        
        # Permutation resampling
        permutation_stats = []
        
        for _ in range(self.n_bootstrap):
            permuted = np.random.permutation(combined)
            perm_group1 = permuted[:n1]
            perm_group2 = permuted[n1:]
            
            perm_stat = statistic_func(perm_group1, perm_group2)
            permutation_stats.append(perm_stat)
        
        permutation_stats = np.array(permutation_stats)
        
        # Two-tailed p-value
        p_value = np.mean(np.abs(permutation_stats) >= np.abs(observed_stat))
        
        return {
            'observed_statistic': observed_stat,
            'p_value': p_value,
            'permutation_distribution': permutation_stats
        }


class MultipleComparison:
    """Handle multiple comparison corrections."""
    
    @staticmethod
    def correct_p_values(
        p_values: List[float],
        method: str = "holm",
        alpha: float = 0.05
    ) -> Tuple[List[bool], List[float], float, float]:
        """Apply multiple comparison correction."""
        methods = {
            'bonferroni': 'bonferroni',
            'holm': 'holm',
            'hochberg': 'hochberg', 
            'hommel': 'hommel',
            'bh': 'fdr_bh',  # Benjamini-Hochberg
            'by': 'fdr_by',  # Benjamini-Yekutieli
            'fdr': 'fdr_bh'
        }
        
        if method not in methods:
            raise ValueError(f"Unknown method: {method}. Available: {list(methods.keys())}")
        
        rejected, p_corrected, alpha_sidak, alpha_bonf = multipletests(
            p_values, alpha=alpha, method=methods[method]
        )
        
        return rejected.tolist(), p_corrected.tolist(), alpha_sidak, alpha_bonf
    
    @staticmethod
    def pairwise_comparisons(
        groups: Dict[str, np.ndarray],
        test_func: callable = None,
        correction_method: str = "holm"
    ) -> pd.DataFrame:
        """Perform pairwise comparisons between all groups."""
        if test_func is None:
            test_func = lambda x, y: ttest_ind(x, y, equal_var=False)
        
        group_names = list(groups.keys())
        results = []
        
        # Perform all pairwise tests
        for i in range(len(group_names)):
            for j in range(i + 1, len(group_names)):
                name1, name2 = group_names[i], group_names[j]
                group1, group2 = groups[name1], groups[name2]
                
                try:
                    stat, p_val = test_func(group1, group2)
                    
                    # Calculate effect size
                    effect_size = EffectSizeCalculator.cohens_d(group1, group2)
                    
                    results.append({
                        'group1': name1,
                        'group2': name2,
                        'statistic': stat,
                        'p_value': p_val,
                        'effect_size': effect_size.value,
                        'effect_interpretation': effect_size.interpretation
                    })
                    
                except Exception as e:
                    results.append({
                        'group1': name1,
                        'group2': name2,
                        'statistic': np.nan,
                        'p_value': np.nan,
                        'effect_size': np.nan,
                        'effect_interpretation': f'Error: {str(e)}'
                    })
        
        df = pd.DataFrame(results)
        
        if len(df) == 0:
            return df
        
        # Apply multiple comparison correction
        p_values = df['p_value'].dropna().values
        if len(p_values) > 0:
            rejected, p_corrected, _, _ = MultipleComparison.correct_p_values(
                p_values, method=correction_method
            )
            
            df.loc[df['p_value'].notna(), 'p_corrected'] = p_corrected
            df.loc[df['p_value'].notna(), 'significant'] = rejected
        
        return df


class StatisticalAnalyzer:
    """Main class for statistical analysis of benchmark results."""
    
    def __init__(
        self,
        alpha: float = 0.05,
        power_threshold: float = 0.8,
        bootstrap_samples: int = 10000
    ):
        self.alpha = alpha
        self.power_threshold = power_threshold
        self.bootstrap_samples = bootstrap_samples
        
        self.effect_calculator = EffectSizeCalculator()
        self.bootstrap_analyzer = BootstrapAnalysis(
            n_bootstrap=bootstrap_samples,
            confidence_level=1 - alpha
        )
        
        self.logger = logging.getLogger(__name__)
    
    def check_assumptions(self, data: np.ndarray) -> Dict[str, bool]:
        """Check statistical test assumptions."""
        assumptions = {}
        
        # Normality test (Shapiro-Wilk for n <= 5000, otherwise Anderson-Darling)
        if len(data) <= 5000:
            try:
                _, p_normality = shapiro(data)
                assumptions['normality'] = p_normality > self.alpha
            except:
                assumptions['normality'] = False
        else:
            try:
                _, p_normality = normaltest(data)
                assumptions['normality'] = p_normality > self.alpha
            except:
                assumptions['normality'] = False
        
        return assumptions
    
    def check_equal_variances(self, *groups) -> Dict[str, bool]:
        """Check equal variance assumption across groups."""
        results = {}
        
        if len(groups) < 2:
            return results
        
        # Levene's test (robust to non-normality)
        try:
            _, p_levene = levene(*groups)
            results['equal_variances_levene'] = p_levene > self.alpha
        except:
            results['equal_variances_levene'] = False
        
        # Bartlett's test (assumes normality)
        try:
            _, p_bartlett = bartlett(*groups)
            results['equal_variances_bartlett'] = p_bartlett > self.alpha
        except:
            results['equal_variances_bartlett'] = False
        
        return results
    
    def compare_two_groups(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        group1_name: str = "Group 1",
        group2_name: str = "Group 2",
        test_type: Optional[TestType] = None
    ) -> Dict[str, Any]:
        """Comprehensive comparison of two groups."""
        
        results = {
            'group1_name': group1_name,
            'group2_name': group2_name,
            'group1_stats': {
                'n': len(group1),
                'mean': np.mean(group1),
                'std': np.std(group1, ddof=1),
                'median': np.median(group1),
                'q25': np.percentile(group1, 25),
                'q75': np.percentile(group1, 75)
            },
            'group2_stats': {
                'n': len(group2),
                'mean': np.mean(group2),
                'std': np.std(group2, ddof=1),
                'median': np.median(group2),
                'q25': np.percentile(group2, 25),
                'q75': np.percentile(group2, 75)
            }
        }
        
        # Check assumptions
        assumptions_g1 = self.check_assumptions(group1)
        assumptions_g2 = self.check_assumptions(group2)
        variance_assumptions = self.check_equal_variances(group1, group2)
        
        results['assumptions'] = {
            'group1_normal': assumptions_g1.get('normality', False),
            'group2_normal': assumptions_g2.get('normality', False),
            'equal_variances': variance_assumptions.get('equal_variances_levene', False)
        }
        
        # Determine appropriate test
        if test_type is None:
            both_normal = (assumptions_g1.get('normality', False) and 
                          assumptions_g2.get('normality', False))
            equal_vars = variance_assumptions.get('equal_variances_levene', False)
            
            if both_normal:
                test_type = TestType.PARAMETRIC
            else:
                test_type = TestType.NON_PARAMETRIC
        
        # Perform statistical tests
        results['tests'] = {}
        
        # Parametric tests
        if test_type == TestType.PARAMETRIC or test_type == TestType.BOOTSTRAP:
            # Independent t-test
            equal_var = results['assumptions']['equal_variances']
            try:
                t_stat, t_p = ttest_ind(group1, group2, equal_var=equal_var)
                results['tests']['t_test'] = SignificanceTest(
                    test_name="Independent t-test",
                    statistic=t_stat,
                    p_value=t_p,
                    assumptions_met=results['assumptions']
                )
            except Exception as e:
                self.logger.warning(f"T-test failed: {e}")
        
        # Non-parametric tests
        if test_type == TestType.NON_PARAMETRIC or test_type == TestType.BOOTSTRAP:
            # Mann-Whitney U test
            try:
                u_stat, u_p = mannwhitneyu(group1, group2, alternative='two-sided')
                results['tests']['mann_whitney'] = SignificanceTest(
                    test_name="Mann-Whitney U test",
                    statistic=u_stat,
                    p_value=u_p
                )
            except Exception as e:
                self.logger.warning(f"Mann-Whitney test failed: {e}")
        
        # Bootstrap tests
        if test_type == TestType.BOOTSTRAP:
            try:
                bootstrap_result = self.bootstrap_analyzer.bootstrap_mean_difference(group1, group2)
                results['tests']['bootstrap'] = SignificanceTest(
                    test_name="Bootstrap test",
                    statistic=bootstrap_result['observed_difference'],
                    p_value=bootstrap_result['p_value'],
                    confidence_interval=bootstrap_result['confidence_interval']
                )
            except Exception as e:
                self.logger.warning(f"Bootstrap test failed: {e}")
        
        # Permutation test
        try:
            perm_result = self.bootstrap_analyzer.permutation_test(group1, group2)
            results['tests']['permutation'] = SignificanceTest(
                test_name="Permutation test",
                statistic=perm_result['observed_statistic'],
                p_value=perm_result['p_value']
            )
        except Exception as e:
            self.logger.warning(f"Permutation test failed: {e}")
        
        # Effect sizes
        results['effect_sizes'] = {}
        try:
            results['effect_sizes']['cohens_d'] = self.effect_calculator.cohens_d(group1, group2)
            results['effect_sizes']['hedges_g'] = self.effect_calculator.hedges_g(group1, group2)
            results['effect_sizes']['cliff_delta'] = self.effect_calculator.cliff_delta(group1, group2)
            results['effect_sizes']['vargha_delaney_a'] = self.effect_calculator.vargha_delaney_a(group1, group2)
        except Exception as e:
            self.logger.warning(f"Effect size calculation failed: {e}")
        
        return results
    
    def compare_multiple_groups(
        self,
        groups: Dict[str, np.ndarray],
        test_type: Optional[TestType] = None
    ) -> Dict[str, Any]:
        """Compare multiple groups with appropriate statistical tests."""
        
        if len(groups) < 2:
            raise ValueError("Need at least 2 groups for comparison")
        
        group_arrays = list(groups.values())
        group_names = list(groups.keys())
        
        results = {
            'group_names': group_names,
            'group_stats': {}
        }
        
        # Descriptive statistics for each group
        for name, data in groups.items():
            results['group_stats'][name] = {
                'n': len(data),
                'mean': np.mean(data),
                'std': np.std(data, ddof=1),
                'median': np.median(data),
                'q25': np.percentile(data, 25),
                'q75': np.percentile(data, 75)
            }
        
        # Check assumptions
        assumptions = {}
        for name, data in groups.items():
            assumptions[name] = self.check_assumptions(data)
        
        variance_check = self.check_equal_variances(*group_arrays)
        
        results['assumptions'] = {
            'individual_normality': {name: assump.get('normality', False) 
                                   for name, assump in assumptions.items()},
            'all_normal': all(assump.get('normality', False) for assump in assumptions.values()),
            'equal_variances': variance_check.get('equal_variances_levene', False)
        }
        
        # Omnibus tests
        results['omnibus_tests'] = {}
        
        # ANOVA (parametric)
        if results['assumptions']['all_normal'] and results['assumptions']['equal_variances']:
            try:
                f_stat, f_p = stats.f_oneway(*group_arrays)
                results['omnibus_tests']['anova'] = SignificanceTest(
                    test_name="One-way ANOVA",
                    statistic=f_stat,
                    p_value=f_p,
                    assumptions_met=results['assumptions']
                )
            except Exception as e:
                self.logger.warning(f"ANOVA failed: {e}")
        
        # Kruskal-Wallis test (non-parametric)
        try:
            h_stat, h_p = kruskal(*group_arrays)
            results['omnibus_tests']['kruskal_wallis'] = SignificanceTest(
                test_name="Kruskal-Wallis test",
                statistic=h_stat,
                p_value=h_p
            )
        except Exception as e:
            self.logger.warning(f"Kruskal-Wallis test failed: {e}")
        
        # Pairwise comparisons
        try:
            pairwise_df = MultipleComparison.pairwise_comparisons(groups)
            results['pairwise_comparisons'] = pairwise_df
        except Exception as e:
            self.logger.warning(f"Pairwise comparisons failed: {e}")
            results['pairwise_comparisons'] = pd.DataFrame()
        
        return results
    
    def analyze_benchmark_results(
        self,
        results: List[ExperimentResult],
        metric: str = 'mean_return',
        group_by: str = 'algorithm_name'
    ) -> Dict[str, Any]:
        """Analyze benchmark results for a specific metric."""
        
        # Extract data
        data_dict = {}
        for result in results:
            group_key = getattr(result, group_by)
            
            if metric in result.performance_metrics:
                value = result.performance_metrics[metric]
            elif hasattr(result, metric):
                value = getattr(result, metric)
            else:
                continue
            
            if group_key not in data_dict:
                data_dict[group_key] = []
            data_dict[group_key].append(value)
        
        # Convert to numpy arrays
        groups = {name: np.array(values) for name, values in data_dict.items()}
        
        if len(groups) == 0:
            return {"error": f"No data found for metric: {metric}"}
        
        if len(groups) == 1:
            return {"error": "Need at least 2 groups for statistical comparison"}
        
        # Perform analysis
        if len(groups) == 2:
            group_names = list(groups.keys())
            analysis_result = self.compare_two_groups(
                groups[group_names[0]], 
                groups[group_names[1]],
                group_names[0], 
                group_names[1]
            )
        else:
            analysis_result = self.compare_multiple_groups(groups)
        
        analysis_result['metric'] = metric
        analysis_result['group_by'] = group_by
        analysis_result['total_results'] = len(results)
        
        return analysis_result
    
    def power_analysis(
        self,
        effect_size: float,
        n1: int,
        n2: int = None,
        alpha: float = None,
        test_type: str = "t_test"
    ) -> Dict[str, float]:
        """Perform power analysis for given parameters."""
        if alpha is None:
            alpha = self.alpha
        
        if n2 is None:
            n2 = n1
        
        # This is a simplified power analysis
        # For more comprehensive analysis, consider using statsmodels.stats.power
        
        if test_type == "t_test":
            # Effect size for t-test (Cohen's d)
            pooled_n = (n1 + n2) / 2
            ncp = effect_size * np.sqrt(pooled_n / 2)  # Non-centrality parameter
            
            # Critical t-value
            df = n1 + n2 - 2
            t_critical = stats.t.ppf(1 - alpha/2, df)
            
            # Power calculation (simplified)
            power = 1 - stats.t.cdf(t_critical, df, ncp) + stats.t.cdf(-t_critical, df, ncp)
        else:
            # Simplified power for other tests
            power = 0.8  # Placeholder
        
        return {
            'power': power,
            'effect_size': effect_size,
            'sample_size_1': n1,
            'sample_size_2': n2,
            'alpha': alpha,
            'adequate_power': power >= self.power_threshold
        }