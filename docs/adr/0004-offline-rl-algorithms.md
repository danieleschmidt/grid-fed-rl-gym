# ADR-0004: Offline RL Algorithm Selection

## Status
Accepted

## Context
Power grid control requires learning from historical operational data without risky online exploration. Utility companies have extensive historical datasets but cannot afford trial-and-error learning on live systems. Offline RL enables learning effective policies from logged data while avoiding potentially dangerous exploration.

## Decision
Prioritize the following offline RL algorithms based on power system requirements:

### Primary Algorithms:
1. **Conservative Q-Learning (CQL)**
   - Conservative policy evaluation to prevent overestimation
   - Excellent performance on suboptimal datasets
   - Strong theoretical guarantees for offline learning

2. **Implicit Q-Learning (IQL)**
   - Avoids explicit policy constraints through expectile regression
   - Robust to distribution shift between data and deployment
   - Suitable for multi-modal action distributions

3. **Advantage-Weighted Regression (AWR)**
   - Simple and stable offline policy learning
   - Natural handling of continuous action spaces
   - Good baseline performance across domains

### Secondary Algorithms:
4. **Batch Constrained Q-Learning (BCQ)**
   - Explicit constraint on actions to match data distribution
   - Pioneering work in offline RL with proven stability

5. **One-Step RL (OSRL)**
   - Single-step policy improvement for safety
   - Minimal deviation from behavior policy

### Algorithm Selection Criteria:
- **Safety**: Conservative policy updates minimize risk
- **Data Efficiency**: Effective learning from limited datasets
- **Distribution Shift**: Robustness to deployment environment differences
- **Continuous Control**: Support for continuous action spaces
- **Theoretical Grounding**: Formal guarantees for safety-critical applications

### Implementation Priorities:
1. CQL as primary algorithm with safety constraints
2. IQL for handling multi-modal policies (e.g., emergency vs. normal operations)
3. AWR for simple baseline comparisons
4. Federated variants of all algorithms for distributed training

## Consequences

### Positive:
- Enables safe learning from historical power system data
- Reduces risk of equipment damage or outages during training
- Leverages decades of operational data from utilities
- Provides multiple algorithm options for different scenarios

### Negative:
- Performance limited by quality of historical data
- May be conservative compared to optimal online policies
- Requires careful hyperparameter tuning for each algorithm

### Implementation Requirements:
- Comprehensive evaluation on power system benchmarks
- Safety validation through constraint satisfaction
- Comparison studies between algorithms on different data types
- Integration with federated learning for multi-utility scenarios