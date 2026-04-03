# Benchmark Protocol

## Purpose

Define a fair comparison protocol for three model levels on the same benchmark task.

## Core Comparison

We will compare:

1. A plain neural baseline
2. A concept-based baseline
3. A neuro-symbolic model

## Evaluation Views

- Outside-view performance:
  - Task accuracy
  - Other final-task metrics if needed
- Inside-view contribution:
  - Concept quality
  - Semantic consistency
  - Shortcut resistance
  - Other symbolic-layer diagnostics that are feasible and interpretable

## Control Variables

- Same task and task definition
- Same train/validation/test split
- Same supervision access where possible
- Similar parameter budget where possible
- Similar training budget where possible
- Comparable runtime reporting

## Initial Task Scope

- Primary task: `MNLogic`
- Optional later task after stabilization: `Kand-Logic`

## Implementation Notes

- `rsbench` will be used as the benchmark framework
- The plain baseline will be implemented in PyTorch
- The concept baseline will follow a simple CBM-style design
- The neuro-symbolic model will use `LTNtorch`

## TODO

- TODO: Phase 3 should document how `rsbench`, `LTNtorch`, and CBM resources are integrated
- TODO: Phase 4 should turn the MNLogic task into a concrete task specification
- TODO: Phase 8 should link each metric to a reproducible script or module
