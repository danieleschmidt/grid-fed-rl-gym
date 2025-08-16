# Methodology

## Experimental Design

We conducted a comprehensive experimental evaluation of novel federated reinforcement learning algorithms for power grid control. The experimental design follows best practices for algorithm comparison studies [@Demšar2006].

### Test Environments

The evaluation was performed on standardized IEEE test feeders to ensure reproducibility and comparability with existing literature:

- **IEEE13Bus**: IEEE 13-bus distribution test feeder with 1000 time steps per episode

### Algorithm Configuration

We compared novel federated RL algorithms against established baselines:

- **NovelFedRL** (Novel): Novel federated RL with privacy preservation
  - Hyperparameters: learning_rate=0.001, batch_size=64
- **PPOBaseline** (Baseline): Proximal Policy Optimization baseline
  - Hyperparameters: learning_rate=0.0003

### Experimental Procedure

Each algorithm was evaluated across 1 environments with 20 independent trials per environment. Each trial consisted of 5 episodes to ensure statistical robustness. To ensure reproducibility, we used fixed random seeds for each experimental condition.

The experimental design employed a factorial arrangement with the following factors:
- Algorithm type (novel vs. baseline)
- Environment complexity (varying network sizes)
- Random initialization (multiple seeds)

### Performance Metrics

We evaluated algorithms using multiple performance dimensions:

- **Episode Reward**: Cumulative reward obtained during each episode
- **Safety Violations**: Number of constraint violations (voltage limits, thermal limits)
- **Convergence Time**: Time required for algorithm to converge
- **Power Loss**: Total system power losses
- **Voltage Deviation**: Root mean square deviation from nominal voltage

## Statistical Analysis

Statistical significance was assessed using a two-pronged approach with α = 0.05 and statistical power = 0.8:

### Parametric Analysis
We employed Welch's t-test for comparing algorithm performance, which is robust to unequal variances [@Welch1947]. This test is more appropriate than Student's t-test when the assumption of equal variances may be violated.

### Non-parametric Analysis
To complement parametric analysis and handle potential non-normal distributions, we used the Mann-Whitney U test [@Mann1947]. This non-parametric test provides robust inference without distributional assumptions.

### Effect Size Calculation
Beyond statistical significance, we computed Cohen's d to quantify practical significance [@Cohen1988]. Effect sizes were interpreted as:
- Small: d ≥ 0.2
- Medium: d ≥ 0.5
- Large: d ≥ 0.8

### Confidence Intervals
We report 95% confidence intervals for all performance metrics to provide uncertainty quantification. Confidence intervals were computed using the t-distribution for sample means.

### Multiple Comparisons
When comparing multiple algorithms simultaneously, we acknowledge the multiple testing problem. Results should be interpreted considering the family-wise error rate across all comparisons.

## Reproducibility

All experiments were conducted using the Grid-Fed-RL-Gym framework with version-controlled algorithms and environments. Random seeds were fixed for each experimental condition. Complete source code and experimental configurations are available in the supplementary materials.
