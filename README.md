# Toward Fair Benchmarking of Neuro-Symbolic AI

This repository contains a bachelor thesis project focused on fair and reproducible benchmarking for neuro-symbolic AI.

The benchmark goal is to compare three model levels on the same task under controlled conditions:

1. Plain neural baseline
2. Concept-based baseline
3. Neuro-symbolic model

The project is intentionally focused on benchmarking methodology rather than proposing a new architecture.

## Current Status

Phase 1 is complete: the repository scaffold and internal documentation placeholders are in place.

## Planned Scope

- Benchmark framework: `rsbench`
- Primary task: `MNLogic`
- Optional later task: `Kand-Logic`
- Plain neural baseline: PyTorch
- Concept-based baseline: CBM-style model
- Neuro-symbolic model: LTN-based model with `LTNtorch`

## Repository Structure

- `docs/`: benchmark protocol, fairness notes, experiment planning, and thesis notes
- `src/`: thesis code for data loading, models, training, evaluation, and utilities
- `configs/`: experiment and model configuration files
- `results/`: checkpoints, metrics, figures, tables, logs, and saved predictions
- `notebooks/`: exploratory and inspection notebooks
- `external/`: third-party repositories or external code snapshots kept separate from thesis code

## Notes

- Environment setup is intentionally deferred to Phase 2.
- External repository placement and integration decisions are intentionally deferred to Phase 3.
- Many files contain `TODO` markers to keep the next implementation steps explicit.
