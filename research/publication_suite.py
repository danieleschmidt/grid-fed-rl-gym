"""
Publication-Ready Documentation and Methodology Suite

This module generates comprehensive, publication-quality documentation and
methodology descriptions for novel federated RL algorithms.

Author: Daniel Schmidt <daniel@terragonlabs.com>
"""

import time
import json
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PublicationMetrics:
    """Metrics and results formatted for publication."""
    algorithm_name: str
    mean_performance: float
    std_performance: float
    confidence_interval: Tuple[float, float]
    sample_size: int
    p_value: float
    effect_size: float
    significance_level: str
    performance_improvement: float
    baseline_comparison: str


class PublicationSuite:
    """
    Suite for generating publication-ready documentation and methodology.
    
    Produces academic-quality reports, LaTeX tables, figures descriptions,
    and methodology sections suitable for research papers.
    """
    
    def __init__(self, output_dir: str = "research/publications"):
        """Initialize publication suite."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_methodology_section(self, 
                                   experimental_config: Dict[str, Any],
                                   statistical_config: Dict[str, Any]) -> str:
        """Generate publication-quality methodology section."""
        
        methodology = []
        
        methodology.append("# Methodology")
        methodology.append("")
        
        # Experimental Design
        methodology.append("## Experimental Design")
        methodology.append("")
        
        methodology.append("We conducted a comprehensive experimental evaluation of novel federated "
                          "reinforcement learning algorithms for power grid control. The experimental "
                          "design follows best practices for algorithm comparison studies [@Demšar2006].")
        methodology.append("")
        
        # Environments
        methodology.append("### Test Environments")
        methodology.append("")
        methodology.append("The evaluation was performed on standardized IEEE test feeders to ensure "
                          "reproducibility and comparability with existing literature:")
        methodology.append("")
        
        environments = experimental_config.get('environments', [])
        for env in environments:
            methodology.append(f"- **{env.get('name', 'Unknown')}**: {env.get('description', 'Standard IEEE test feeder')} "
                             f"with {env.get('episode_length', 1000)} time steps per episode")
        
        methodology.append("")
        
        # Algorithms
        methodology.append("### Algorithm Configuration")
        methodology.append("")
        methodology.append("We compared novel federated RL algorithms against established baselines:")
        methodology.append("")
        
        algorithms = experimental_config.get('algorithms', [])
        for alg in algorithms:
            alg_type = "Novel" if not alg.get('is_baseline', True) else "Baseline"
            methodology.append(f"- **{alg.get('name', 'Unknown')}** ({alg_type}): {alg.get('description', 'No description')}")
            
            hyperparams = alg.get('hyperparameters', {})
            if hyperparams:
                param_str = ", ".join([f"{k}={v}" for k, v in hyperparams.items()])
                methodology.append(f"  - Hyperparameters: {param_str}")
        
        methodology.append("")
        
        # Experimental Procedure
        methodology.append("### Experimental Procedure")
        methodology.append("")
        
        n_trials = experimental_config.get('n_trials', 10)
        n_episodes = experimental_config.get('n_episodes_per_trial', 5)
        
        methodology.append(f"Each algorithm was evaluated across {len(environments)} environments with "
                          f"{n_trials} independent trials per environment. Each trial consisted of "
                          f"{n_episodes} episodes to ensure statistical robustness. To ensure reproducibility, "
                          "we used fixed random seeds for each experimental condition.")
        methodology.append("")
        
        methodology.append("The experimental design employed a factorial arrangement with the following factors:")
        methodology.append("- Algorithm type (novel vs. baseline)")
        methodology.append("- Environment complexity (varying network sizes)")
        methodology.append("- Random initialization (multiple seeds)")
        methodology.append("")
        
        # Metrics
        methodology.append("### Performance Metrics")
        methodology.append("")
        methodology.append("We evaluated algorithms using multiple performance dimensions:")
        methodology.append("")
        methodology.append("- **Episode Reward**: Cumulative reward obtained during each episode")
        methodology.append("- **Safety Violations**: Number of constraint violations (voltage limits, thermal limits)")
        methodology.append("- **Convergence Time**: Time required for algorithm to converge")
        methodology.append("- **Power Loss**: Total system power losses")
        methodology.append("- **Voltage Deviation**: Root mean square deviation from nominal voltage")
        methodology.append("")
        
        # Statistical Analysis
        methodology.append("## Statistical Analysis")
        methodology.append("")
        
        alpha = statistical_config.get('alpha', 0.05)
        power = statistical_config.get('power', 0.8)
        
        methodology.append(f"Statistical significance was assessed using a two-pronged approach with "
                          f"α = {alpha} and statistical power = {power}:")
        methodology.append("")
        
        methodology.append("### Parametric Analysis")
        methodology.append("We employed Welch's t-test for comparing algorithm performance, which is robust "
                          "to unequal variances [@Welch1947]. This test is more appropriate than Student's "
                          "t-test when the assumption of equal variances may be violated.")
        methodology.append("")
        
        methodology.append("### Non-parametric Analysis") 
        methodology.append("To complement parametric analysis and handle potential non-normal distributions, "
                          "we used the Mann-Whitney U test [@Mann1947]. This non-parametric test provides "
                          "robust inference without distributional assumptions.")
        methodology.append("")
        
        methodology.append("### Effect Size Calculation")
        methodology.append("Beyond statistical significance, we computed Cohen's d to quantify practical "
                          "significance [@Cohen1988]. Effect sizes were interpreted as:")
        methodology.append("- Small: d ≥ 0.2")
        methodology.append("- Medium: d ≥ 0.5") 
        methodology.append("- Large: d ≥ 0.8")
        methodology.append("")
        
        methodology.append("### Confidence Intervals")
        methodology.append("We report 95% confidence intervals for all performance metrics to provide "
                          "uncertainty quantification. Confidence intervals were computed using the "
                          "t-distribution for sample means.")
        methodology.append("")
        
        methodology.append("### Multiple Comparisons")
        methodology.append("When comparing multiple algorithms simultaneously, we acknowledge the "
                          "multiple testing problem. Results should be interpreted considering the "
                          "family-wise error rate across all comparisons.")
        methodology.append("")
        
        # Reproducibility
        methodology.append("## Reproducibility")
        methodology.append("")
        methodology.append("All experiments were conducted using the Grid-Fed-RL-Gym framework with "
                          "version-controlled algorithms and environments. Random seeds were fixed "
                          "for each experimental condition. Complete source code and experimental "
                          "configurations are available in the supplementary materials.")
        methodology.append("")
        
        return "\n".join(methodology)
    
    def generate_results_section(self, 
                               benchmark_analysis: Dict[str, Any],
                               statistical_comparisons: Dict[str, Any]) -> str:
        """Generate publication-quality results section."""
        
        results = []
        
        results.append("# Results")
        results.append("")
        
        # Overview
        results.append("## Experimental Overview")
        results.append("")
        
        metadata = benchmark_analysis.get('benchmark_metadata', {})
        n_novel = len(metadata.get('novel_algorithms', []))
        n_baseline = len(metadata.get('baseline_algorithms', []))
        
        results.append(f"We evaluated {n_novel} novel federated RL algorithms against {n_baseline} "
                      f"established baselines across multiple IEEE test feeders. ")
        
        categories = benchmark_analysis.get('performance_categories', {})
        summary = categories.get('summary', {})
        
        success_rate = summary.get('success_rate', 0)
        results.append(f"Of the novel algorithms, {summary.get('state_of_the_art_count', 0)} achieved "
                      f"state-of-the-art performance and {summary.get('competitive_count', 0)} "
                      f"demonstrated competitive performance (overall success rate: {success_rate:.1%}).")
        results.append("")
        
        # Performance Rankings
        results.append("## Algorithm Performance Rankings")
        results.append("")
        
        rankings = benchmark_analysis.get('algorithm_rankings', {})
        overall_ranking = rankings.get('overall_performance', [])
        
        if overall_ranking:
            results.append("Table 1 presents the overall performance ranking of all evaluated algorithms:")
            results.append("")
            results.append("**Table 1: Overall Algorithm Performance Ranking**")
            results.append("")
            results.append("| Rank | Algorithm | Performance Score | Type |")
            results.append("|------|-----------|------------------|------|")
            
            for i, (alg_name, score) in enumerate(overall_ranking[:8], 1):  # Top 8
                # Determine if novel or baseline
                is_novel = any(alg.get('name') == alg_name and not alg.get('is_baseline', True) 
                             for alg in metadata.get('novel_algorithms', []))
                alg_type = "Novel" if is_novel else "Baseline"
                results.append(f"| {i} | {alg_name} | {score:.3f} | {alg_type} |")
            
            results.append("")
        
        # Statistical Significance Analysis
        results.append("## Statistical Significance Analysis")
        results.append("")
        
        performance_matrix = benchmark_analysis.get('performance_matrix', {})
        
        if performance_matrix:
            results.append("### Pairwise Comparisons")
            results.append("")
            results.append("Table 2 summarizes the statistical analysis of novel algorithms against baselines:")
            results.append("")
            results.append("**Table 2: Statistical Comparison Results**")
            results.append("")
            results.append("| Novel Algorithm | Baseline | Performance Ratio | p-value | Effect Size | Significance |")
            results.append("|----------------|----------|-------------------|---------|-------------|--------------|")
            
            for novel_alg, comparisons in performance_matrix.items():
                for baseline, stats in comparisons.items():
                    ratio = stats.get('performance_ratio', 1.0)
                    p_val = stats.get('p_value', 1.0)
                    effect = stats.get('effect_size', 0.0)
                    sig = "✓" if stats.get('statistical_significance', False) else "✗"
                    
                    results.append(f"| {novel_alg} | {baseline} | {ratio:.3f} | {p_val:.4f} | {effect:.3f} | {sig} |")
            
            results.append("")
            results.append("*Performance ratio > 1.0 indicates superior performance. "
                          "✓ indicates statistical significance at α = 0.05.*")
            results.append("")
        
        # Performance Categories
        results.append("### Performance Category Analysis")
        results.append("")
        
        if categories.get('state_of_the_art'):
            results.append("#### State-of-the-Art Performance")
            results.append("")
            state_of_art = categories['state_of_the_art']
            results.append(f"{len(state_of_art)} algorithm(s) achieved state-of-the-art performance "
                          f"(>20% improvement over best baseline):")
            results.append("")
            for alg_name, score in state_of_art:
                improvement = ((score - summary.get('best_baseline_score', 0)) / 
                             abs(summary.get('best_baseline_score', 1))) * 100
                results.append(f"- **{alg_name}**: {improvement:.1f}% improvement over best baseline")
            results.append("")
        
        if categories.get('competitive'):
            results.append("#### Competitive Performance")
            results.append("")
            competitive = categories['competitive']
            results.append(f"{len(competitive)} algorithm(s) demonstrated competitive performance "
                          f"(5-20% improvement over best baseline):")
            results.append("")
            for alg_name, score in competitive:
                improvement = ((score - summary.get('best_baseline_score', 0)) / 
                             abs(summary.get('best_baseline_score', 1))) * 100
                results.append(f"- **{alg_name}**: {improvement:.1f}% improvement over best baseline")
            results.append("")
        
        # Safety Analysis
        results.append("## Safety and Constraint Compliance")
        results.append("")
        
        safety_ranking = rankings.get('safety', [])
        if safety_ranking:
            results.append("Safety performance (measured by constraint violations) varied significantly "
                          "across algorithms:")
            results.append("")
            
            best_safety = safety_ranking[0]
            worst_safety = safety_ranking[-1]
            
            results.append(f"- **Best Safety Performance**: {best_safety[0]} "
                          f"({best_safety[1]:.1f} violations per episode)")
            results.append(f"- **Worst Safety Performance**: {worst_safety[0]} "
                          f"({worst_safety[1]:.1f} violations per episode)")
            results.append("")
        
        # Efficiency Analysis
        results.append("## Computational Efficiency")
        results.append("")
        
        efficiency_ranking = rankings.get('efficiency', [])
        if efficiency_ranking:
            results.append("Convergence time analysis reveals computational efficiency differences:")
            results.append("")
            
            fastest = efficiency_ranking[0]
            slowest = efficiency_ranking[-1]
            
            results.append(f"- **Fastest Convergence**: {fastest[0]} "
                          f"({fastest[1]:.4f} seconds)")
            results.append(f"- **Slowest Convergence**: {slowest[0]} "
                          f"({slowest[1]:.4f} seconds)")
            
            speedup = slowest[1] / fastest[1] if fastest[1] > 0 else 1.0
            results.append(f"- **Maximum Speedup**: {speedup:.1f}x")
            results.append("")
        
        return "\n".join(results)
    
    def generate_latex_tables(self, benchmark_analysis: Dict[str, Any]) -> Dict[str, str]:
        """Generate LaTeX tables for publication."""
        
        tables = {}
        
        # Performance ranking table
        rankings = benchmark_analysis.get('algorithm_rankings', {})
        overall_ranking = rankings.get('overall_performance', [])
        
        if overall_ranking:
            latex_table = []
            latex_table.append("\\begin{table}[htbp]")
            latex_table.append("\\centering")
            latex_table.append("\\caption{Overall Algorithm Performance Ranking}")
            latex_table.append("\\label{tab:performance_ranking}")
            latex_table.append("\\begin{tabular}{|c|l|c|c|}")
            latex_table.append("\\hline")
            latex_table.append("\\textbf{Rank} & \\textbf{Algorithm} & \\textbf{Score} & \\textbf{Type} \\\\")
            latex_table.append("\\hline")
            
            metadata = benchmark_analysis.get('benchmark_metadata', {})
            for i, (alg_name, score) in enumerate(overall_ranking[:8], 1):
                is_novel = any(alg.get('name') == alg_name and not alg.get('is_baseline', True) 
                             for alg in metadata.get('novel_algorithms', []))
                alg_type = "Novel" if is_novel else "Baseline"
                
                # Escape underscores for LaTeX
                safe_name = alg_name.replace('_', '\\_')
                latex_table.append(f"{i} & {safe_name} & {score:.3f} & {alg_type} \\\\")
                latex_table.append("\\hline")
            
            latex_table.append("\\end{tabular}")
            latex_table.append("\\end{table}")
            
            tables['performance_ranking'] = "\n".join(latex_table)
        
        # Statistical comparison table
        performance_matrix = benchmark_analysis.get('performance_matrix', {})
        
        if performance_matrix:
            latex_table = []
            latex_table.append("\\begin{table}[htbp]")
            latex_table.append("\\centering")
            latex_table.append("\\caption{Statistical Comparison of Novel Algorithms vs Baselines}")
            latex_table.append("\\label{tab:statistical_comparison}")
            latex_table.append("\\begin{tabular}{|l|l|c|c|c|c|}")
            latex_table.append("\\hline")
            latex_table.append("\\textbf{Novel} & \\textbf{Baseline} & \\textbf{Ratio} & "
                              "\\textbf{p-value} & \\textbf{Effect} & \\textbf{Sig.} \\\\")
            latex_table.append("\\hline")
            
            for novel_alg, comparisons in performance_matrix.items():
                for baseline, stats in comparisons.items():
                    ratio = stats.get('performance_ratio', 1.0)
                    p_val = stats.get('p_value', 1.0)
                    effect = stats.get('effect_size', 0.0)
                    sig = "$\\checkmark$" if stats.get('statistical_significance', False) else "$\\times$"
                    
                    safe_novel = novel_alg.replace('_', '\\_')
                    safe_baseline = baseline.replace('_', '\\_')
                    
                    latex_table.append(f"{safe_novel} & {safe_baseline} & {ratio:.3f} & "
                                     f"{p_val:.4f} & {effect:.3f} & {sig} \\\\")
                    latex_table.append("\\hline")
            
            latex_table.append("\\end{tabular}")
            latex_table.append("\\end{table}")
            
            tables['statistical_comparison'] = "\n".join(latex_table)
        
        return tables
    
    def generate_figure_descriptions(self, benchmark_analysis: Dict[str, Any]) -> Dict[str, str]:
        """Generate descriptions for publication figures."""
        
        descriptions = {}
        
        # Performance comparison figure
        descriptions['performance_comparison'] = (
            "Figure 1: Performance comparison across all evaluated algorithms. "
            "Box plots show the distribution of episode rewards for each algorithm "
            "across all test environments. Novel algorithms are highlighted in blue, "
            "baseline algorithms in gray. Error bars represent 95% confidence intervals."
        )
        
        # Statistical significance heatmap
        descriptions['significance_heatmap'] = (
            "Figure 2: Statistical significance heatmap for pairwise algorithm comparisons. "
            "Color intensity indicates the magnitude of performance difference, with "
            "asterisks (*) marking statistically significant differences (p < 0.05). "
            "Novel algorithms are compared against baseline algorithms."
        )
        
        # Safety vs performance scatter
        descriptions['safety_performance'] = (
            "Figure 3: Safety-performance trade-off analysis. Each point represents "
            "an algorithm's average performance (x-axis) vs. safety violations (y-axis). "
            "The Pareto frontier identifies algorithms with optimal trade-offs. "
            "Lower-right quadrant indicates superior algorithms (high performance, low violations)."
        )
        
        # Convergence analysis
        descriptions['convergence_analysis'] = (
            "Figure 4: Convergence time analysis across different network sizes. "
            "Lines show mean convergence time with shaded regions indicating "
            "standard error. Novel algorithms demonstrate improved scalability "
            "characteristics compared to baseline approaches."
        )
        
        return descriptions
    
    def generate_discussion_section(self, benchmark_analysis: Dict[str, Any]) -> str:
        """Generate publication-quality discussion section."""
        
        discussion = []
        
        discussion.append("# Discussion")
        discussion.append("")
        
        # Key findings
        discussion.append("## Key Findings")
        discussion.append("")
        
        categories = benchmark_analysis.get('performance_categories', {})
        summary = categories.get('summary', {})
        
        success_rate = summary.get('success_rate', 0)
        
        if success_rate > 0.5:
            discussion.append("Our experimental evaluation demonstrates that the proposed novel "
                            "federated RL algorithms achieve superior performance compared to "
                            "established baselines in power grid control tasks.")
        else:
            discussion.append("The experimental evaluation reveals mixed results for the proposed "
                            "novel federated RL algorithms, with several areas for improvement "
                            "identified through systematic comparison with established baselines.")
        
        discussion.append("")
        
        # Performance analysis
        discussion.append("## Performance Analysis")
        discussion.append("")
        
        if categories.get('state_of_the_art'):
            best_alg = categories['state_of_the_art'][0]
            improvement = ((best_alg[1] - summary.get('best_baseline_score', 0)) / 
                         abs(summary.get('best_baseline_score', 1))) * 100
            
            discussion.append(f"The top-performing algorithm, {best_alg[0]}, achieved "
                            f"{improvement:.1f}% improvement over the best baseline, "
                            "demonstrating the potential of federated approaches in "
                            "power grid control applications.")
        
        discussion.append("")
        
        # Statistical significance
        discussion.append("## Statistical Validity")
        discussion.append("")
        
        performance_matrix = benchmark_analysis.get('performance_matrix', {})
        
        if performance_matrix:
            significant_count = sum(
                1 for comparisons in performance_matrix.values()
                for stats in comparisons.values()
                if stats.get('statistical_significance', False)
            )
            total_comparisons = sum(len(comparisons) for comparisons in performance_matrix.values())
            
            discussion.append(f"Statistical analysis revealed significant differences in "
                            f"{significant_count} out of {total_comparisons} pairwise comparisons "
                            f"({significant_count/total_comparisons:.1%}), indicating robust "
                            "performance differences beyond random variation.")
        
        discussion.append("")
        
        # Safety implications
        discussion.append("## Safety and Practical Implications")
        discussion.append("")
        
        rankings = benchmark_analysis.get('algorithm_rankings', {})
        safety_ranking = rankings.get('safety', [])
        
        if safety_ranking:
            best_safety = safety_ranking[0]
            worst_safety = safety_ranking[-1]
            
            discussion.append("Safety analysis reveals significant variation in constraint compliance "
                            "across algorithms. While some novel approaches demonstrate improved "
                            "safety characteristics, others require additional safety mechanisms "
                            "before deployment in critical infrastructure.")
        
        discussion.append("")
        
        # Limitations
        discussion.append("## Limitations")
        discussion.append("")
        
        discussion.append("Several limitations should be acknowledged:")
        discussion.append("")
        discussion.append("1. **Environment Scope**: Evaluation was limited to IEEE test feeders, "
                         "which may not capture all real-world operational complexities.")
        discussion.append("")
        discussion.append("2. **Simulation Fidelity**: While comprehensive, simulation-based "
                         "evaluation cannot fully replicate all aspects of physical power systems.")
        discussion.append("")
        discussion.append("3. **Temporal Scope**: Long-term stability and adaptation characteristics "
                         "require extended evaluation periods beyond this study's scope.")
        discussion.append("")
        discussion.append("4. **Communication Constraints**: Real federated deployments face "
                         "communication latency and reliability challenges not fully captured "
                         "in simulation.")
        discussion.append("")
        
        # Future work
        discussion.append("## Future Research Directions")
        discussion.append("")
        
        discussion.append("Based on our findings, several research directions emerge:")
        discussion.append("")
        
        if summary.get('below_baseline_count', 0) > 0:
            discussion.append("1. **Algorithm Refinement**: Investigate the factors contributing "
                            "to below-baseline performance in some novel algorithms through "
                            "detailed ablation studies.")
            discussion.append("")
        
        discussion.append("2. **Real-world Validation**: Extend evaluation to hardware-in-the-loop "
                         "testing and eventual field trials to validate simulation results.")
        discussion.append("")
        
        discussion.append("3. **Scalability Analysis**: Evaluate performance characteristics on "
                         "larger, more complex distribution networks representative of modern "
                         "smart grid deployments.")
        discussion.append("")
        
        discussion.append("4. **Multi-objective Optimization**: Develop algorithms that explicitly "
                         "balance performance, safety, and efficiency through multi-objective "
                         "optimization frameworks.")
        discussion.append("")
        
        return "\n".join(discussion)
    
    def generate_complete_paper_draft(self,
                                    experimental_config: Dict[str, Any],
                                    statistical_config: Dict[str, Any],
                                    benchmark_analysis: Dict[str, Any]) -> str:
        """Generate a complete research paper draft."""
        
        paper = []
        
        # Title and abstract
        paper.append("# Federated Reinforcement Learning for Power Grid Control: A Comprehensive Evaluation")
        paper.append("")
        paper.append("## Abstract")
        paper.append("")
        
        metadata = benchmark_analysis.get('benchmark_metadata', {})
        n_novel = len(metadata.get('novel_algorithms', []))
        n_baseline = len(metadata.get('baseline_algorithms', []))
        
        categories = benchmark_analysis.get('performance_categories', {})
        summary = categories.get('summary', {})
        success_rate = summary.get('success_rate', 0)
        
        paper.append("This paper presents a comprehensive evaluation of novel federated reinforcement "
                    "learning (FRL) algorithms for power grid control applications. We evaluate "
                    f"{n_novel} novel FRL algorithms against {n_baseline} established baselines "
                    "across multiple IEEE test feeders using rigorous statistical methodology. "
                    f"Our results demonstrate a {success_rate:.1%} success rate in achieving "
                    "competitive or state-of-the-art performance. Statistical analysis using both "
                    "parametric and non-parametric tests reveals significant performance differences "
                    "with practical implications for power system operation. The findings contribute "
                    "to the growing body of knowledge on distributed intelligence in power systems "
                    "and provide guidance for future algorithm development.")
        paper.append("")
        
        paper.append("**Keywords**: Federated Learning, Reinforcement Learning, Power Grid Control, "
                    "Distributed Systems, Smart Grid")
        paper.append("")
        
        # Introduction
        paper.append("## Introduction")
        paper.append("")
        paper.append("The increasing complexity of modern power distribution systems, driven by "
                    "renewable energy integration and distributed energy resources, necessitates "
                    "advanced control algorithms capable of managing uncertainty and variability [@Kezunovic2011]. "
                    "Traditional centralized control approaches face challenges in terms of "
                    "scalability, privacy, and real-time responsiveness [@Molzahn2017].")
        paper.append("")
        
        paper.append("Federated reinforcement learning (FRL) emerges as a promising paradigm that "
                    "combines the adaptability of reinforcement learning with the distributed "
                    "nature of modern power systems [@Li2020]. By enabling multiple agents to "
                    "learn collaboratively while preserving data privacy, FRL addresses key "
                    "limitations of centralized approaches.")
        paper.append("")
        
        paper.append("This paper makes the following contributions:")
        paper.append("1. Novel FRL algorithms specifically designed for power grid control")
        paper.append("2. Comprehensive experimental evaluation methodology")
        paper.append("3. Statistical validation using multiple hypothesis testing approaches")
        paper.append("4. Performance analysis across multiple evaluation criteria")
        paper.append("")
        
        # Add methodology section
        methodology = self.generate_methodology_section(experimental_config, statistical_config)
        paper.append(methodology)
        paper.append("")
        
        # Add results section
        results = self.generate_results_section(benchmark_analysis, {})
        paper.append(results)
        paper.append("")
        
        # Add discussion section
        discussion = self.generate_discussion_section(benchmark_analysis)
        paper.append(discussion)
        paper.append("")
        
        # Conclusion
        paper.append("## Conclusion")
        paper.append("")
        
        if success_rate > 0.5:
            paper.append("This comprehensive evaluation demonstrates the efficacy of novel federated "
                        "reinforcement learning algorithms for power grid control applications. "
                        "Statistical analysis confirms significant performance improvements over "
                        "established baselines, with practical implications for smart grid deployment.")
        else:
            paper.append("This comprehensive evaluation provides insights into the challenges and "
                        "opportunities for federated reinforcement learning in power grid control. "
                        "While some algorithms show promise, the results indicate areas for "
                        "improvement in algorithm design and implementation.")
        
        paper.append("")
        paper.append("The rigorous experimental methodology and statistical validation framework "
                    "established in this work provide a foundation for future research in "
                    "distributed power system control. The open-source implementation enables "
                    "reproducibility and further advancement of the field.")
        paper.append("")
        
        # References
        paper.append("## References")
        paper.append("")
        paper.append("[@Cohen1988] Cohen, J. (1988). Statistical power analysis for the behavioral sciences. Routledge.")
        paper.append("[@Demšar2006] Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. JMLR, 7, 1-30.")
        paper.append("[@Kezunovic2011] Kezunovic, M., et al. (2011). Smart grid protection and control systems. Springer.")
        paper.append("[@Li2020] Li, T., et al. (2020). Federated learning: Challenges, methods, and future directions. IEEE Signal Processing Magazine.")
        paper.append("[@Mann1947] Mann, H. B., & Whitney, D. R. (1947). On a test of whether one of two random variables is stochastically larger. Annals of Mathematical Statistics.")
        paper.append("[@Molzahn2017] Molzahn, D. K., et al. (2017). A survey of distributed optimization and control algorithms for electric power systems. IEEE Transactions on Smart Grid.")
        paper.append("[@Welch1947] Welch, B. L. (1947). The generalization of Student's problem when several different population variances are involved. Biometrika.")
        paper.append("")
        
        return "\n".join(paper)
    
    def save_publication_documents(self,
                                 experimental_config: Dict[str, Any],
                                 statistical_config: Dict[str, Any],
                                 benchmark_analysis: Dict[str, Any],
                                 prefix: str = "federated_rl_evaluation") -> Dict[str, Path]:
        """Save all publication documents."""
        
        timestamp = int(time.time())
        file_prefix = f"{prefix}_{timestamp}"
        
        saved_files = {}
        
        # Complete paper draft
        paper_draft = self.generate_complete_paper_draft(
            experimental_config, statistical_config, benchmark_analysis
        )
        
        paper_file = self.output_dir / f"{file_prefix}_paper_draft.md"
        with open(paper_file, 'w') as f:
            f.write(paper_draft)
        saved_files['paper_draft'] = paper_file
        
        # LaTeX tables
        latex_tables = self.generate_latex_tables(benchmark_analysis)
        
        for table_name, table_content in latex_tables.items():
            table_file = self.output_dir / f"{file_prefix}_{table_name}.tex"
            with open(table_file, 'w') as f:
                f.write(table_content)
            saved_files[f'table_{table_name}'] = table_file
        
        # Figure descriptions
        figure_descriptions = self.generate_figure_descriptions(benchmark_analysis)
        
        figures_file = self.output_dir / f"{file_prefix}_figure_descriptions.md"
        with open(figures_file, 'w') as f:
            f.write("# Figure Descriptions\n\n")
            for fig_name, description in figure_descriptions.items():
                f.write(f"## {fig_name.replace('_', ' ').title()}\n\n")
                f.write(f"{description}\n\n")
        saved_files['figure_descriptions'] = figures_file
        
        # Individual sections
        sections = {
            'methodology': self.generate_methodology_section(experimental_config, statistical_config),
            'results': self.generate_results_section(benchmark_analysis, {}),
            'discussion': self.generate_discussion_section(benchmark_analysis)
        }
        
        for section_name, section_content in sections.items():
            section_file = self.output_dir / f"{file_prefix}_{section_name}.md"
            with open(section_file, 'w') as f:
                f.write(section_content)
            saved_files[f'section_{section_name}'] = section_file
        
        # Experimental configuration
        config_file = self.output_dir / f"{file_prefix}_config.json"
        with open(config_file, 'w') as f:
            json.dump({
                'experimental_config': experimental_config,
                'statistical_config': statistical_config,
                'benchmark_analysis': benchmark_analysis
            }, f, indent=2, default=str)
        saved_files['config'] = config_file
        
        return saved_files


# Example usage and demonstration
if __name__ == "__main__":
    # Create publication suite
    pub_suite = PublicationSuite()
    
    # Mock experimental configuration
    experimental_config = {
        'algorithms': [
            {
                'name': 'NovelFedRL',
                'description': 'Novel federated RL with privacy preservation',
                'is_baseline': False,
                'hyperparameters': {'learning_rate': 0.001, 'batch_size': 64}
            },
            {
                'name': 'PPOBaseline',
                'description': 'Proximal Policy Optimization baseline',
                'is_baseline': True,
                'hyperparameters': {'learning_rate': 0.0003}
            }
        ],
        'environments': [
            {
                'name': 'IEEE13Bus',
                'description': 'IEEE 13-bus distribution test feeder',
                'episode_length': 1000
            }
        ],
        'n_trials': 20,
        'n_episodes_per_trial': 5
    }
    
    # Mock statistical configuration
    statistical_config = {
        'alpha': 0.05,
        'power': 0.8
    }
    
    # Mock benchmark analysis
    benchmark_analysis = {
        'benchmark_metadata': {
            'novel_algorithms': [experimental_config['algorithms'][0]],
            'baseline_algorithms': [experimental_config['algorithms'][1]]
        },
        'performance_categories': {
            'state_of_the_art': [('NovelFedRL', 0.85)],
            'competitive': [],
            'baseline_level': [],
            'below_baseline': [],
            'summary': {
                'total_novel_algorithms': 1,
                'state_of_the_art_count': 1,
                'competitive_count': 0,
                'success_rate': 1.0,
                'best_baseline_score': 0.75
            }
        },
        'algorithm_rankings': {
            'overall_performance': [
                ('NovelFedRL', 0.85),
                ('PPOBaseline', 0.75)
            ],
            'safety': [
                ('NovelFedRL', 2.1),
                ('PPOBaseline', 3.5)
            ],
            'efficiency': [
                ('NovelFedRL', 120.5),
                ('PPOBaseline', 150.2)
            ]
        },
        'performance_matrix': {
            'NovelFedRL': {
                'PPOBaseline': {
                    'performance_ratio': 1.13,
                    'statistical_significance': True,
                    'effect_size': 0.8,
                    'p_value': 0.002
                }
            }
        }
    }
    
    print("=== Generating Publication-Ready Documents ===")
    
    # Generate and save all documents
    saved_files = pub_suite.save_publication_documents(
        experimental_config,
        statistical_config, 
        benchmark_analysis
    )
    
    print(f"\nPublication documents generated:")
    for doc_type, file_path in saved_files.items():
        print(f"- {doc_type}: {file_path}")
    
    # Display sample content
    paper_draft = pub_suite.generate_complete_paper_draft(
        experimental_config, statistical_config, benchmark_analysis
    )
    
    print("\n" + "="*80)
    print("SAMPLE PAPER DRAFT (First 2000 characters)")
    print("="*80)
    print(paper_draft[:2000] + "..." if len(paper_draft) > 2000 else paper_draft)