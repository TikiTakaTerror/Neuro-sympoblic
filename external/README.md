# External Code

This folder is reserved for third-party repositories or code snapshots that should stay separate from thesis-owned code.

## Planned Layout

```text
external/
  rsbench-code/
  LTNtorch-reference/
  ConceptBottleneck-reference/
```

## Usage Policy

- `rsbench-code/` is the main upstream benchmark reference and is the only external repo expected to matter early.
- `LTNtorch-reference/` is optional and should only be used as a read-only reference clone if needed.
- `ConceptBottleneck-reference/` is optional and should only be used as a read-only reference clone if needed.
- Thesis-owned wrappers, train scripts, evaluation code, and configs must stay outside `external/`.

## Important Rule

Keep adapters and benchmark logic in `src/`, not inside external repositories.

## See Also

- `docs/external_integration_plan.md`
