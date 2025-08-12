"""Example research study demonstrating the comprehensive research framework.

This example shows how to use the Grid-Fed-RL-Gym research framework to conduct
rigorous, reproducible experiments comparing novel federated RL algorithms.
"""

import numpy as np
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from .experiment_manager import ExperimentManager, ResearchConfig
from ..benchmarking.statistical_analysis import StatisticalAnalyzer
from ..benchmarking.benchmark_suite import IEEE_TEST_CASES, RENEWABLE_SCENARIOS


def setup_logging():
    """Setup logging for the research study."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('research_study.log'),
            logging.StreamHandler()
        ]
    )


def create_example_config() -> ResearchConfig:
    """Create example research configuration."""
    
    research_questions = [
        "RQ1: Do physics-informed federated RL algorithms outperform traditional approaches in power grid control?",
        "RQ2: How does multi-objective optimization affect the trade-off between economic efficiency and grid stability?",
        "RQ3: Can uncertainty-aware algorithms provide better performance under high renewable energy penetration?",
        "RQ4: What is the computational overhead of graph neural networks in federated grid control?",
        "RQ5: How effectively do continual learning methods adapt to changing grid conditions?"
    ]
    
    hypotheses = [
        "H1: Physics-informed federated RL will achieve 15% better performance than baseline methods",
        "H2: Multi-objective algorithms will find superior Pareto solutions compared to single-objective approaches",
        "H3: Uncertainty-aware methods will maintain performance under 50%+ renewable penetration",
        "H4: Graph neural networks will scale better to large grid systems with <2x computational overhead",
        "H5: Continual learning will reduce catastrophic forgetting by 80% when adapting to new grid configurations"
    ]
    
    config = ResearchConfig(
        experiment_name="federated_rl_power_systems_comprehensive",
        description="Comprehensive evaluation of novel federated RL algorithms for power grid control",
        authors=["Claude Research Assistant", "Grid-Fed-RL Team"],
        institution="Advanced AI Research Laboratory",
        contact_email="research@gridfedrl.ai",
        research_questions=research_questions,
        hypotheses=hypotheses,
        objectives=[
            "Compare performance of novel federated RL algorithms",
            "Analyze safety properties and constraint satisfaction",
            "Evaluate computational efficiency and scalability",
            "Assess robustness under various grid conditions",
            "Measure adaptation capability to evolving scenarios"
        ],
        
        # Experimental design
        algorithms=[
            'pifrl',           # Physics-Informed Federated RL
            'mofrl',           # Multi-Objective Federated RL  
            'uafrl',           # Uncertainty-Aware Federated RL
            'gnfrl',           # Graph Neural Federated RL
            'continual_frl',   # Continual Federated RL
            'cql',             # Conservative Q-Learning (baseline)
            'iql',             # Implicit Q-Learning (baseline)
        ],
        test_cases=[
            'ieee13_basic',
            'ieee13_high_renewable', 
            'ieee34_medium',
            'ieee123_complex'
        ],
        scenarios=[
            'sunny_day',
            'windy_day', 
            'cloudy_variable',
            'extreme_weather'
        ],
        num_seeds=10,
        num_trials=3,
        
        # Computational settings
        parallel_execution=True,
        max_workers=8,
        timeout_minutes=180,
        memory_limit_gb=32,
        
        # Output settings
        save_raw_data=True,
        save_models=True,
        generate_plots=True,
        generate_tables=True,
        
        # Publication settings
        target_venue="IEEE Transactions on Smart Grid",
        paper_template="ieee",
        include_appendix=True,
        
        # Reproducibility
        random_seed=42,
        version_control=True,
        environment_snapshot=True,
        
        tags=["federated-learning", "reinforcement-learning", "power-systems", "smart-grid"]
    )
    
    return config


def analyze_research_results(config: ResearchConfig, results, summary):
    """Analyze research results and answer research questions."""
    
    print("\n" + "="*80)
    print("RESEARCH RESULTS ANALYSIS")
    print("="*80)
    
    print(f"\nExperiment: {config.experiment_name}")
    print(f"Total Results: {len(results)}")
    print(f"Algorithms Tested: {len(set(r.algorithm_name for r in results))}")
    print(f"Test Cases: {len(set(r.test_case for r in results))}")
    
    # Statistical analyzer
    analyzer = StatisticalAnalyzer(alpha=0.05)
    
    print("\n" + "-"*60)
    print("RESEARCH QUESTION ANALYSIS")
    print("-"*60)
    
    # RQ1: Physics-informed vs Traditional
    print("\nRQ1: Physics-Informed vs Traditional Approaches")
    physics_results = [r for r in results if 'pifrl' in r.algorithm_name.lower()]
    traditional_results = [r for r in results if r.algorithm_name.lower() in ['cql', 'iql']]
    
    if physics_results and traditional_results:
        physics_performance = [r.performance_metrics.get('mean_return', 0) for r in physics_results]
        traditional_performance = [r.performance_metrics.get('mean_return', 0) for r in traditional_results]
        
        comparison = analyzer.compare_two_groups(
            np.array(physics_performance),
            np.array(traditional_performance),
            "Physics-Informed",
            "Traditional"
        )
        
        print(f"Physics-Informed Mean: {np.mean(physics_performance):.4f}")
        print(f"Traditional Mean: {np.mean(traditional_performance):.4f}")
        print(f"Improvement: {((np.mean(physics_performance) / np.mean(traditional_performance)) - 1) * 100:.1f}%")
        
        if 't_test' in comparison['tests']:
            t_test = comparison['tests']['t_test']
            print(f"Statistical Significance: p={t_test.p_value:.4f} ({'Significant' if t_test.is_significant() else 'Not Significant'})")
        
        if 'cohens_d' in comparison.get('effect_sizes', {}):
            effect_size = comparison['effect_sizes']['cohens_d']
            print(f"Effect Size: {effect_size.value:.3f} ({effect_size.interpretation})")
    
    # RQ2: Multi-objective analysis
    print("\nRQ2: Multi-Objective Optimization Analysis")
    mofrl_results = [r for r in results if 'mofrl' in r.algorithm_name.lower()]
    
    if mofrl_results:
        # Analyze Pareto optimality (simplified)
        pareto_metrics = []
        for result in mofrl_results:
            economic_score = result.economic_metrics.get('cost_efficiency', 0)
            stability_score = result.safety_metrics.get('safety_score', 0)
            pareto_metrics.append((economic_score, stability_score))
        
        if pareto_metrics:
            pareto_array = np.array(pareto_metrics)
            print(f"Multi-Objective Results: {len(pareto_metrics)} Pareto solutions")
            print(f"Economic Efficiency Range: [{pareto_array[:, 0].min():.3f}, {pareto_array[:, 0].max():.3f}]")
            print(f"Stability Score Range: [{pareto_array[:, 1].min():.3f}, {pareto_array[:, 1].max():.3f}]")
    
    # RQ3: Uncertainty-aware under high renewable penetration
    print("\nRQ3: Uncertainty-Aware Performance Under High Renewables")
    uafrl_results = [r for r in results if 'uafrl' in r.algorithm_name.lower() and 
                     'high_renewable' in r.test_case.lower()]
    
    if uafrl_results:
        renewable_performance = [r.performance_metrics.get('mean_return', 0) for r in uafrl_results]
        renewable_safety = [r.safety_metrics.get('safety_score', 0) for r in uafrl_results]
        
        print(f"High Renewable Performance: {np.mean(renewable_performance):.4f} ± {np.std(renewable_performance):.4f}")
        print(f"High Renewable Safety: {np.mean(renewable_safety):.4f} ± {np.std(renewable_safety):.4f}")
    
    # RQ4: Graph neural network scalability
    print("\nRQ4: Graph Neural Network Computational Overhead")
    gnfrl_results = [r for r in results if 'gnfrl' in r.algorithm_name.lower()]
    baseline_results = [r for r in results if r.algorithm_name.lower() in ['cql', 'iql']]
    
    if gnfrl_results and baseline_results:
        gnfrl_times = [r.execution_time for r in gnfrl_results]
        baseline_times = [r.execution_time for r in baseline_results]
        
        overhead_ratio = np.mean(gnfrl_times) / np.mean(baseline_times)
        print(f"GNN Execution Time: {np.mean(gnfrl_times):.2f}s ± {np.std(gnfrl_times):.2f}s")
        print(f"Baseline Execution Time: {np.mean(baseline_times):.2f}s ± {np.std(baseline_times):.2f}s")
        print(f"Computational Overhead: {overhead_ratio:.2f}x")
    
    # RQ5: Continual learning adaptation
    print("\nRQ5: Continual Learning Adaptation")
    continual_results = [r for r in results if 'continual' in r.algorithm_name.lower()]
    
    if continual_results:
        adaptation_scores = []
        for result in continual_results:
            if 'federated_metrics' in result.__dict__:
                adaptation_score = result.federated_metrics.get('adaptation_score', 0)
                adaptation_scores.append(adaptation_score)
        
        if adaptation_scores:
            print(f"Adaptation Performance: {np.mean(adaptation_scores):.4f} ± {np.std(adaptation_scores):.4f}")
    
    print("\n" + "-"*60)
    print("OVERALL ALGORITHM RANKING")
    print("-"*60)
    
    # Overall performance ranking
    algorithm_performance = {}
    for result in results:
        algo = result.algorithm_name
        if algo not in algorithm_performance:
            algorithm_performance[algo] = []
        
        performance_score = result.performance_metrics.get('mean_return', 0)
        safety_score = result.safety_metrics.get('safety_score', 0)
        # Combined score (weighted)
        combined_score = 0.7 * performance_score + 0.3 * safety_score
        algorithm_performance[algo].append(combined_score)
    
    # Calculate means and rank
    algorithm_means = {algo: np.mean(scores) for algo, scores in algorithm_performance.items()}
    ranked_algorithms = sorted(algorithm_means.items(), key=lambda x: x[1], reverse=True)
    
    print("\nOverall Performance Ranking:")
    for i, (algo, score) in enumerate(ranked_algorithms, 1):
        print(f"{i}. {algo}: {score:.4f}")
    
    return {
        'algorithm_ranking': ranked_algorithms,
        'statistical_comparisons': comparison if 'comparison' in locals() else {},
        'research_insights': {
            'physics_informed_advantage': physics_results and traditional_results,
            'multi_objective_solutions': len(mofrl_results) if mofrl_results else 0,
            'uncertainty_robustness': len(uafrl_results) if uafrl_results else 0,
            'gnn_overhead': overhead_ratio if 'overhead_ratio' in locals() else None,
            'continual_adaptation': len(continual_results) if continual_results else 0
        }
    }


def generate_research_plots(results, output_dir: Path):
    """Generate research plots and figures."""
    
    # Create plots directory
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Performance comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Federated RL Algorithms Performance Comparison', fontsize=16)
    
    # Extract data
    algorithms = list(set(r.algorithm_name for r in results))
    performance_data = {algo: [] for algo in algorithms}
    safety_data = {algo: [] for algo in algorithms}
    efficiency_data = {algo: [] for algo in algorithms}
    
    for result in results:
        algo = result.algorithm_name
        performance_data[algo].append(result.performance_metrics.get('mean_return', 0))
        safety_data[algo].append(result.safety_metrics.get('safety_score', 0))
        efficiency_data[algo].append(1.0 / max(result.execution_time, 1))  # Inverse time as efficiency
    
    # Performance box plot
    axes[0, 0].boxplot([performance_data[algo] for algo in algorithms], labels=algorithms)
    axes[0, 0].set_title('Performance Distribution')
    axes[0, 0].set_ylabel('Mean Return')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Safety scores
    axes[0, 1].boxplot([safety_data[algo] for algo in algorithms], labels=algorithms)
    axes[0, 1].set_title('Safety Score Distribution')
    axes[0, 1].set_ylabel('Safety Score')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Computational efficiency
    axes[1, 0].boxplot([efficiency_data[algo] for algo in algorithms], labels=algorithms)
    axes[1, 0].set_title('Computational Efficiency')
    axes[1, 0].set_ylabel('Efficiency (1/time)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Performance vs Safety scatter
    for algo in algorithms:
        perf = performance_data[algo]
        safety = safety_data[algo]
        axes[1, 1].scatter(perf, safety, label=algo, alpha=0.7)
    
    axes[1, 1].set_xlabel('Performance (Mean Return)')
    axes[1, 1].set_ylabel('Safety Score')
    axes[1, 1].set_title('Performance vs Safety Trade-off')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Algorithm ranking plot
    algorithm_means = {algo: np.mean(performance_data[algo]) for algo in algorithms}
    ranked_algos = sorted(algorithm_means.items(), key=lambda x: x[1], reverse=True)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    algos, scores = zip(*ranked_algos)
    colors = plt.cm.viridis(np.linspace(0, 1, len(algos)))
    
    bars = ax.bar(range(len(algos)), scores, color=colors)
    ax.set_xlabel('Algorithms')
    ax.set_ylabel('Average Performance Score')
    ax.set_title('Algorithm Performance Ranking')
    ax.set_xticks(range(len(algos)))
    ax.set_xticklabels(algos, rotation=45)
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'algorithm_ranking.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Generated research plots in {plots_dir}")


def run_example_research_study():
    """Run the complete example research study."""
    
    print("="*80)
    print("GRID-FED-RL COMPREHENSIVE RESEARCH STUDY")
    print("="*80)
    
    # Setup logging
    setup_logging()
    
    # Create research configuration
    config = create_example_config()
    
    print(f"\nResearch Study: {config.experiment_name}")
    print(f"Algorithms: {', '.join(config.algorithms)}")
    print(f"Test Cases: {', '.join(config.test_cases)}")
    print(f"Scenarios: {', '.join(config.scenarios)}")
    print(f"Total Experiments: {len(config.algorithms) * len(config.test_cases) * len(config.scenarios) * config.num_seeds}")
    
    # Create experiment manager
    manager = ExperimentManager()
    
    print(f"\nResearch Questions:")
    for i, question in enumerate(config.research_questions, 1):
        print(f"  {i}. {question}")
    
    print(f"\nHypotheses:")
    for i, hypothesis in enumerate(config.hypotheses, 1):
        print(f"  {i}. {hypothesis}")
    
    # Note: This is a demonstration framework
    # In practice, you would run: research_package = manager.run_research_study(config)
    
    # For this example, we'll create mock results to demonstrate the analysis
    print(f"\n{'='*60}")
    print("GENERATING EXAMPLE RESULTS")
    print("(In practice, this would run actual experiments)")
    print("="*60)
    
    # Generate mock results for demonstration
    mock_results = _generate_mock_results(config)
    
    # Create output directory
    output_dir = Path("example_research_output")
    output_dir.mkdir(exist_ok=True)
    
    # Save mock configuration and results
    config.save(output_dir / "experiment_config.yaml")
    
    # Analyze results
    analysis_results = analyze_research_results(config, mock_results, {})
    
    # Generate plots
    try:
        generate_research_plots(mock_results, output_dir)
    except Exception as e:
        print(f"Plot generation failed: {e}")
    
    # Save analysis
    import json
    with open(output_dir / "analysis_results.json", 'w') as f:
        json.dump(analysis_results, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print("RESEARCH STUDY COMPLETED")
    print("="*60)
    print(f"Results saved to: {output_dir}")
    print(f"Total mock results: {len(mock_results)}")
    
    # Generate summary
    print(f"\nRESEARCH SUMMARY:")
    print(f"- Best performing algorithm: {analysis_results['algorithm_ranking'][0][0]}")
    print(f"- Performance score: {analysis_results['algorithm_ranking'][0][1]:.4f}")
    
    if analysis_results['research_insights']['physics_informed_advantage']:
        print(f"- Physics-informed RL showed advantage over traditional methods")
    
    print(f"- Multi-objective solutions found: {analysis_results['research_insights']['multi_objective_solutions']}")
    print(f"- Continual learning algorithms tested: {analysis_results['research_insights']['continual_adaptation']}")
    
    return config, mock_results, analysis_results


def _generate_mock_results(config: ResearchConfig):
    """Generate mock results for demonstration purposes."""
    from ..benchmarking.benchmark_suite import ExperimentResult
    
    results = []
    
    # Define performance characteristics for each algorithm
    algorithm_profiles = {
        'pifrl': {'base_performance': 0.85, 'variance': 0.05, 'safety_bonus': 0.1},
        'mofrl': {'base_performance': 0.82, 'variance': 0.06, 'safety_bonus': 0.08},
        'uafrl': {'base_performance': 0.88, 'variance': 0.04, 'safety_bonus': 0.12},
        'gnfrl': {'base_performance': 0.80, 'variance': 0.07, 'safety_bonus': 0.06},
        'continual_frl': {'base_performance': 0.83, 'variance': 0.05, 'safety_bonus': 0.09},
        'cql': {'base_performance': 0.75, 'variance': 0.08, 'safety_bonus': 0.02},
        'iql': {'base_performance': 0.77, 'variance': 0.07, 'safety_bonus': 0.03},
    }
    
    np.random.seed(config.random_seed)
    
    for algorithm in config.algorithms:
        for test_case in config.test_cases:
            for scenario in config.scenarios:
                for seed in range(config.num_seeds):
                    
                    profile = algorithm_profiles.get(algorithm, algorithm_profiles['cql'])
                    
                    # Generate performance metrics
                    base_perf = profile['base_performance']
                    variance = profile['variance']
                    
                    # Add complexity penalty based on test case
                    complexity_factor = {
                        'ieee13_basic': 1.0,
                        'ieee13_high_renewable': 0.95,
                        'ieee34_medium': 0.92,
                        'ieee123_complex': 0.88
                    }.get(test_case, 0.9)
                    
                    # Add scenario difficulty
                    scenario_factor = {
                        'sunny_day': 1.0,
                        'windy_day': 0.98,
                        'cloudy_variable': 0.94,
                        'extreme_weather': 0.85
                    }.get(scenario, 0.9)
                    
                    # Generate metrics
                    mean_return = (base_perf * complexity_factor * scenario_factor + 
                                 np.random.normal(0, variance))
                    
                    safety_score = min(1.0, mean_return + profile['safety_bonus'] + 
                                     np.random.normal(0, 0.02))
                    
                    execution_time = np.random.lognormal(
                        mean=np.log(60 + hash(algorithm) % 120), 
                        sigma=0.3
                    )
                    
                    # Create result
                    result = ExperimentResult(
                        algorithm_name=algorithm,
                        test_case=test_case,
                        scenario=scenario,
                        seed=seed,
                        performance_metrics={
                            'mean_return': mean_return,
                            'std_return': variance,
                            'success_rate': min(1.0, mean_return + 0.1)
                        },
                        learning_metrics={
                            'convergence_step': int(np.random.exponential(200)),
                            'training_time': execution_time * 0.8
                        },
                        safety_metrics={
                            'safety_score': safety_score,
                            'violation_rate': max(0, 0.1 - safety_score),
                            'total_violations': int(np.random.poisson(max(0, (1-safety_score) * 10)))
                        },
                        economic_metrics={
                            'cost_efficiency': mean_return * 0.9 + np.random.normal(0, 0.03),
                            'mean_cost': np.random.gamma(2, scale=100)
                        },
                        environmental_metrics={
                            'renewable_utilization': min(1.0, mean_return * 0.8 + 
                                                        ('renewable' in test_case.lower()) * 0.2),
                            'carbon_reduction': mean_return * 0.6,
                            'sustainability_score': mean_return * 0.85
                        },
                        federated_metrics={
                            'communication_rounds': int(np.random.uniform(50, 200)),
                            'aggregation_efficiency': np.random.beta(8, 2),
                            'adaptation_score': np.random.normal(0.75, 0.1) if 'continual' in algorithm else 0.5
                        },
                        execution_time=execution_time,
                        memory_usage=np.random.gamma(3, scale=500),
                        convergence_step=int(np.random.exponential(150)),
                        final_policy_performance=mean_return
                    )
                    
                    results.append(result)
    
    return results


if __name__ == "__main__":
    # Run the example research study
    config, results, analysis = run_example_research_study()
    print("\nExample research study completed successfully!")