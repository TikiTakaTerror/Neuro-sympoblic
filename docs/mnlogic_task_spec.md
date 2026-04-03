# MNLogic Task Spec

This document records the Phase 4 inspection of the primary thesis task in `rsbench`.

## Upstream Files Inspected

- `external/rsbench-code/rsseval/rss/datasets/xor.py`
- `external/rsbench-code/rsseval/rss/datasets/utils/xor_creation.py`
- `external/rsbench-code/rsseval/rss/models/xornn.py`
- `external/rsbench-code/rsseval/rss/models/xorcbm.py`
- `external/rsbench-code/rsseval/rss/models/xordpl.py`
- `external/rsbench-code/rsseval/rss/models/utils/utils_problog.py`
- `external/rsbench-code/rsseval/rss/utils/metrics.py`
- `external/rsbench-code/rssgen/examples_config/xor.yml`
- `external/rsbench-code/rssgen/rssgen/generators/xor_generator.py`
- `external/rsbench-code/rssgen/rssgen/parsers/xor_parser.py`

## Naming Mapping

There is an upstream naming mismatch that should stay visible:

- the dataset class is named `MNLOGIC`
- the dataset entrypoint name is `xor`
- the data path used by `rsseval` is `data/mnlogic`

For this thesis, the safest reading is:

- "MNLogic" is the conceptual task name
- `xor` is the upstream CLI dataset name inside `rsseval`

## Input Format

The raw dataset loader expects:

- one `.png` image per sample
- one matching `.joblib` metadata file per sample
- paired by numeric filename inside each split folder

The loader returns:

- `image`
- `label`
- `concepts`

The image is converted to grayscale and then passed through `transforms.ToTensor()`.

Inferred from source:

- if the task uses MNIST-based generation, each sample is a horizontal strip of binary digit images
- the `rsseval` XOR models assume `n_images=4`
- with standard `28x28` digit tiles, this implies an expected image tensor shape of about `1 x 28 x 112`

This image shape is an inference from the generator and model assumptions, not from bundled raw task files, because the actual full dataset is not shipped inside the repository.

## Label Format

The metadata loader reads:

- `data["label"]`

The label is stored in the `.joblib` file and loaded as a binary target.

The upstream generator writes the label as a boolean value. The evaluation code later treats labels as binary class targets (`0` or `1`).

For thesis purposes, the task label should be treated as:

- binary scalar
- semantic meaning: false or true for the logic rule

## Concept Format

The metadata loader reads:

- `data["meta"]["concepts"]`

For the `rsseval` XOR path:

- concepts are binary
- there are 4 concept positions expected by the evaluation models
- concept vector format is effectively `[c1, c2, c3, c4]`

Each concept corresponds to one sub-image position in the horizontal strip.

## Split Format

The task uses four split folders:

- `train/`
- `val/`
- `test/`
- `ood/`

Inside each split:

- sample image files are stored as `N.png`
- metadata files are stored as `N.joblib`

The generator base class creates these split directories and writes one image plus one metadata file per sample.

The `ood/` split is treated as out-of-distribution.

## Rule / Constraint Structure

There are two relevant upstream views of the task.

### 1. Generic generator view

The `rssgen` XOR generator is configurable:

- it supports a user-specified logic expression
- it can also auto-generate an `Xor(...)` rule
- it supports configurable `n_digits`

The example config in `rssgen/examples_config/xor.yml` uses:

- `n_digits: 3`
- a custom logic expression `Or(And(a, b), Not(c))`

### 2. `rsseval` benchmark view

The `rsseval` XOR models do not use the generic example directly.

In `xordpl.py`, the logic matrix is hard-coded as:

- `create_xor(n_digits=2, sequence_len=4)`

That utility marks label `1` when:

- the sum of the four binary concepts is even

So the effective `rsseval` task behavior is closer to a 4-bit parity-style binary classification task than a generic configurable XOR expression.

## Important Upstream Inconsistency

There is a real inconsistency between:

- the flexible `rssgen` XOR generator interface
- the fixed 4-bit binary logic assumed by `rsseval`

This matters for the thesis because the final benchmark should commit to one concrete task definition.

Current best interpretation:

- for the thesis benchmark pipeline, the primary inspectable `rsseval` task is the 4-bit `MNLOGIC` / `xor` path
- Phase 4 should treat that as the main operational task definition unless later evidence shows a different upstream dataset is intended

## Existing Evaluation Hooks

The upstream code already includes:

- `xornn.py` for the plain neural baseline
- `xorcbm.py` for a concept bottleneck style model
- `xordpl.py` for the logic-based model
- `XOR_eval_tloss_cacc_acc()` in `utils/metrics.py`
- `evaluate_metrics()` dispatch support for dataset name `xor`
- TCAV wiring that imports `datasets.xor.MNLOGIC`

This means the upstream benchmark already recognizes the task as a first-class evaluation path.

## Example Input / Output Pair

Using the `rsseval` parity-style interpretation:

- concepts `[0, 0, 0, 0]` produce label `1`
- concepts `[0, 0, 0, 1]` produce label `0`

Reason:

- the hard-coded rule matrix labels worlds with an even number of ones as positive

## What Is Confirmed In Phase 4

- the upstream MNLogic task path has been identified
- the sample file layout is clear
- the split structure is clear
- the binary concept and binary label format are clear
- the existing model and metric hooks are identifiable

## What Is Still Not Fully Confirmed

- whether the final thesis should follow the generic `rssgen` logic config or the fixed `rsseval` parity interpretation
- whether the full official `data/mnlogic` dataset files will exactly match the inferred `4 x binary concept` expectation

## Phase 4 Verification Script

Use:

- `scripts/inspect_mnlogic.py`

This script can:

- inspect the upstream source path
- create a tiny demo dataset with the same file layout
- load the demo dataset through the upstream `XORDataset` reader
- print example shapes and sample structures
