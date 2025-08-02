# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records (ADRs) for the Grid-Fed-RL-Gym project.

## What is an ADR?

An Architecture Decision Record (ADR) is a document that captures an important architectural decision made along with its context and consequences.

## ADR Format

We use the following template for ADRs:

```
# ADR-XXXX: [Decision Title]

## Status
[Proposed | Accepted | Deprecated | Superseded]

## Context
What is the issue that we're seeing that is motivating this decision or change?

## Decision
What is the change that we're proposing and/or doing?

## Consequences
What becomes easier or more difficult to do because of this change?
```

## Index

- [ADR-0001: Project Architecture](0001-project-architecture.md)
- [ADR-0002: Federated Learning Framework](0002-federated-learning-framework.md)
- [ADR-0003: Safety Constraints Implementation](0003-safety-constraints.md)
- [ADR-0004: Offline RL Algorithm Selection](0004-offline-rl-algorithms.md)

## Creating New ADRs

1. Create a new file with format `XXXX-short-title.md`
2. Use the next available number
3. Follow the template structure
4. Submit for review via pull request