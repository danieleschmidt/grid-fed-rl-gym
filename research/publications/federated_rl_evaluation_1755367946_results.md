# Results

## Experimental Overview

We evaluated 1 novel federated RL algorithms against 1 established baselines across multiple IEEE test feeders. 
Of the novel algorithms, 1 achieved state-of-the-art performance and 0 demonstrated competitive performance (overall success rate: 100.0%).

## Algorithm Performance Rankings

Table 1 presents the overall performance ranking of all evaluated algorithms:

**Table 1: Overall Algorithm Performance Ranking**

| Rank | Algorithm | Performance Score | Type |
|------|-----------|------------------|------|
| 1 | NovelFedRL | 0.850 | Novel |
| 2 | PPOBaseline | 0.750 | Baseline |

## Statistical Significance Analysis

### Pairwise Comparisons

Table 2 summarizes the statistical analysis of novel algorithms against baselines:

**Table 2: Statistical Comparison Results**

| Novel Algorithm | Baseline | Performance Ratio | p-value | Effect Size | Significance |
|----------------|----------|-------------------|---------|-------------|--------------|
| NovelFedRL | PPOBaseline | 1.130 | 0.0020 | 0.800 | ✓ |

*Performance ratio > 1.0 indicates superior performance. ✓ indicates statistical significance at α = 0.05.*

### Performance Category Analysis

#### State-of-the-Art Performance

1 algorithm(s) achieved state-of-the-art performance (>20% improvement over best baseline):

- **NovelFedRL**: 13.3% improvement over best baseline

## Safety and Constraint Compliance

Safety performance (measured by constraint violations) varied significantly across algorithms:

- **Best Safety Performance**: NovelFedRL (2.1 violations per episode)
- **Worst Safety Performance**: PPOBaseline (3.5 violations per episode)

## Computational Efficiency

Convergence time analysis reveals computational efficiency differences:

- **Fastest Convergence**: NovelFedRL (120.5000 seconds)
- **Slowest Convergence**: PPOBaseline (150.2000 seconds)
- **Maximum Speedup**: 1.2x
