# External Integration Plan

This document defines how external repositories should be used in the thesis project without making the codebase messy.

## Goal

Use external resources for benchmarking context and library support while keeping thesis-owned code in this repository simple, auditable, and reproducible.

## Decision Summary

| Resource | Primary Role In Thesis | Default Handling | Planned Location | Runtime Dependency Policy |
| --- | --- | --- | --- | --- |
| `rsbench-code` | Upstream benchmark and task reference | Keep as external codebase | `external/rsbench-code/` | Reference its task and dataset structure, but keep training and evaluation logic in `src/` |
| `LTNtorch` | Symbolic reasoning library for the neuro-symbolic model | Install as Python dependency when needed | Not required as a local repo for normal use; optional reference clone at `external/LTNtorch-reference/` | Use as a library import in our own model code |
| `ConceptBottleneck` | Conceptual reference for CBM design and interventions | Reference only | Optional reference clone at `external/ConceptBottleneck-reference/` | Do not depend on it at runtime |

## Planned External Layout

The intended layout is:

```text
external/
  README.md
  rsbench-code/                    # main external benchmark reference
  LTNtorch-reference/              # optional read-only reference clone
  ConceptBottleneck-reference/     # optional read-only reference clone
```

Important:

- Only `rsbench-code/` is expected to matter early in the thesis workflow.
- The two `*-reference/` folders are optional and should stay read-only.
- Thesis-owned scripts, wrappers, configs, and experiments must stay in `src/`, `configs/`, `docs/`, and `notebooks/`.

## Repository-Specific Plan

### 1. `rsbench-code`

#### Purpose

`rsbench-code` is the upstream benchmark reference. It contains:

- task implementations
- dataset utilities
- baseline model implementations
- evaluation utilities

For this thesis, its main value is not to become our main codebase. Its value is to define the benchmark task and give us a concrete upstream reference for MNLogic and later Kand-Logic.

Important observation from inspection:

- `rsseval/rss/datasets/xor.py` defines a dataset class named `MNLOGIC`
- its dataset `NAME` is `"xor"`
- it reads data from `data/mnlogic`

This suggests that the thesis task name `MNLogic` likely maps to the upstream `xor` dataset entrypoint, even though the storage path is `mnlogic`.

#### Planned Placement

Store it inside this repository at:

```text
external/rsbench-code/
```

#### How Our Code Will Interact With It

Our project should interact with `rsbench-code` in a narrow way:

- inspect dataset and task definitions from `rsseval`
- reuse task semantics and split conventions where possible
- write thesis-owned adapters in `src/neurosymbolic_benchmark/data/`
- write thesis-owned models in `src/neurosymbolic_benchmark/models/`
- write thesis-owned train and evaluation scripts in `src/neurosymbolic_benchmark/training/` and `src/neurosymbolic_benchmark/evaluation/`

We should avoid making our project depend on the upstream `main.py` training entrypoint as the center of our workflow.

Reason:

- the upstream codebase has a large, research-style experiment stack
- it mixes task code, model code, logging, tuning, and benchmark utilities
- using it directly would make controlled thesis comparisons harder to audit

#### Practical Strategy

Use `rsbench-code` as an upstream task source, not as the thesis runtime owner.

This means:

- task loading may be wrapped or lightly adapted
- experiment orchestration should remain ours
- result storage and naming should remain ours
- fairness controls should remain ours

#### Risk

There is a macOS-specific filesystem issue in the upstream repository:

- `rssgen/boia_config/shapes/StopSign.blend`
- `rssgen/boia_config/shapes/stopsign.blend`

These collide on a case-insensitive filesystem, which is common on macOS. A plain clone can succeed with warnings but produce an incomplete working tree for that part of the repository.

Implication:

- This is acceptable for our thesis if we only need `rsseval`
- It is a risk if we later need the full `rssgen` generator stack on this machine

Recommended handling:

- prefer a sparse checkout or a task-focused checkout when we integrate it
- prioritize `rsseval/` first

### 2. `LTNtorch`

#### Purpose

`LTNtorch` is a focused PyTorch library for Logic Tensor Networks. Unlike `rsbench-code`, it is a reusable library rather than a large experiment repository.

#### Planned Placement

Normal use:

- install it as a Python dependency when the neuro-symbolic phase starts

Optional local reference clone:

```text
external/LTNtorch-reference/
```

#### How Our Code Will Interact With It

Our neuro-symbolic model should import `LTNtorch` as a library from our own code.

Planned location of our code:

- `src/neurosymbolic_benchmark/models/`

What remains ours:

- the model wrapper around the task
- the rule definitions we choose
- the training loop
- metric logging
- checkpointing
- result export

What remains external:

- the LTN primitives
- fuzzy logic operators
- quantifier/connective implementations

This keeps the symbolic layer auditable while avoiding a fork of the library.

#### Risk

- `LTNtorch` is easy to install in principle, but version compatibility still needs validation with the chosen PyTorch stack
- the PyPI metadata is older-looking than our current environment, so we should verify it before Phase 7 instead of assuming full compatibility

### 3. `ConceptBottleneck`

#### Purpose

The original `ConceptBottleneck` repository is useful as a conceptual reference for:

- concept bottleneck architecture shape
- intervention logic
- saving concept predictions separately from task predictions

It is not a good base runtime dependency for this thesis.

#### Planned Placement

Do not depend on it by default.

Optional local reference clone:

```text
external/ConceptBottleneck-reference/
```

#### How Our Code Will Interact With It

Our project should use it only as a design reference.

The thesis CBM-style baseline should be implemented in-house in:

```text
src/neurosymbolic_benchmark/models/
```

This implementation should be:

- simpler than the original repo
- directly compatible with MNLogic
- easier to compare fairly against the neural and LTN models

#### Risk

- the original repository uses an old stack centered around `torch==1.1.0`
- it is tied to specific datasets such as CUB and OAI
- adopting it directly would create unnecessary maintenance problems and make the thesis environment less stable

## Ownership Boundary

To keep the codebase clean, use this rule:

- external repositories define upstream assets or libraries
- thesis code defines the actual benchmark pipeline used in experiments

In practice:

- do not edit external repositories unless absolutely necessary
- do not place thesis experiments inside external repositories
- do not let external scripts define our final result format
- do not let external configs become the source of truth for fairness controls

## Planned Wrapper Strategy

Our own repository should provide a thin internal layer around external resources.

Planned ownership:

- `src/neurosymbolic_benchmark/data/`
  - task adapters
  - dataset inspection helpers
  - conversion utilities if upstream formats are awkward
- `src/neurosymbolic_benchmark/models/`
  - plain neural baseline
  - in-house CBM-style baseline
  - LTN-based model that imports `LTNtorch`
- `src/neurosymbolic_benchmark/training/`
  - shared training logic
  - seed handling
  - checkpointing
  - logging
- `src/neurosymbolic_benchmark/evaluation/`
  - outside-view metrics
  - inside-view metrics
  - control metrics

## Recommended Acquisition Strategy

When we move from planning to actual integration:

1. bring in `rsbench-code` first, because it defines the benchmark task
2. inspect only the needed task path before writing adapters
3. postpone `LTNtorch` installation until the neuro-symbolic model phase
4. keep `ConceptBottleneck` optional unless we need to inspect a specific intervention detail

## Open Questions

- Confirm that the thesis task should use the upstream `MNLOGIC` dataset implementation in `rsseval/rss/datasets/xor.py`, exposed under the dataset name `xor`
- Decide whether `rsbench-code` should be brought in as a normal clone or a sparse checkout focused on `rsseval/`
- Validate the final `LTNtorch` installation path against the chosen PyTorch version before Phase 7

## TODO

- TODO: Phase 4 should inspect `rsseval/rss/datasets/xor.py` and `datasets/utils/xor_creation.py` first
- TODO: Phase 4 should record the task file(s) that define the primary benchmark
- TODO: Phase 6 should record the exact concept output format used by the in-house CBM baseline
- TODO: Phase 7 should record the final `LTNtorch` version decision
