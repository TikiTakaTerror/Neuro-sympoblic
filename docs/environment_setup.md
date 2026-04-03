# Environment Setup

This document defines the Phase 2 environment for the thesis project.

## Recommended Python Version

Use `Python 3.10`.

Why this choice:

- It matches the current local machine setup
- It avoids the global `pip3` and `python3` mismatch by using a local virtual environment
- It is a practical middle ground for the core thesis stack

## Create the Environment

From the repository root:

```bash
cd /Users/abdullahsaeed/Neuro-sympoblic
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

## Core Dependencies

The core Phase 2 environment includes:

- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `jupyter`
- `torch`
- `torchvision`

`torchvision` is included because the benchmark tasks are image-based and later phases will likely need standard PyTorch vision utilities.

## Sanity Check

Run:

```bash
source .venv/bin/activate
python scripts/check_environment.py
```

This should print the installed versions of the main libraries and the available PyTorch backend information.

## Notes On External Dependencies

### `rsbench`

`rsbench-code` is not installed as part of the base Phase 2 environment.

Reason:

- The official codebase is organized as multiple components (`rsscount`, `rsseval`, `rssgen`)
- The `rsseval` README is Linux and conda oriented and documents its own larger dependency stack
- The same repository also contains version signals that do not point to one single clean base environment

Planned handling:

- Keep `rsbench` external and document the integration boundary in Phase 3
- Wrap only the pieces needed for the thesis benchmark instead of adopting the full repository environment

### `LTNtorch`

`LTNtorch` is not installed yet.

Reason:

- It is only needed starting from the neuro-symbolic model phase
- `rsbench-code/rsseval` pins `LTNtorch==1.0.1`
- The current `LTNtorch` PyPI project exists, but its published Python classifiers are older than our chosen base Python

Planned handling:

- Validate a compatible installation path before Phase 7
- Keep the core environment small until the symbolic layer is implemented

### `ConceptBottleneck`

The original Concept Bottleneck Models repository is not used as a direct environment dependency.

Reason:

- The official repository uses a much older software stack
- Reusing that environment directly would make this thesis project harder to maintain

Planned handling:

- Implement a lightweight in-house CBM-style baseline later
- Use the original repository mainly as a conceptual reference, not as the runtime base

## TODO

- TODO: Phase 7 should add the final `LTNtorch` installation note once validated
