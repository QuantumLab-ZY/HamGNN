# Task 3 Report

## Status

Implemented the owned Lightning 2 import migration for graph data and the Uni
predictor. The required import regression test was added.

Task 3 implementation commit: `a844553 refactor: use Lightning 2 imports in owned paths`

## Changes

- `hamgnn/data/graph_data.py` now imports `lightning.pytorch as pl`, so
  `graph_data_module` subclasses `lightning.pytorch.LightningDataModule`.
- `Uni-HamGNN/Uni-HamiltonianPredictor.py` now imports `Model` from
  `hamgnn.models.Model`, avoiding the CLI module as an indirect loading path.
- Removed the duplicate `yaml` import in the touched Uni import block.
- Added `tests/test_lightning2_imports.py` with source-scan and DataModule base
  checks. The source scan excludes `hamgnn/toolbox/nequip` as required.

## Verification

- `git diff --check 3e2e3f7..HEAD`: passed after restoring the parent file's
  mixed LF/CRLF convention.
- `git diff --name-only 3e2e3f7..HEAD`: contains only
  `hamgnn/data/graph_data.py`, `Uni-HamGNN/Uni-HamiltonianPredictor.py`, and
  `tests/test_lightning2_imports.py`; no NequIP path is included.
- Source stale-import scan: passed (`stale: []`), excluding
  `hamgnn/toolbox/nequip` as required.
- `python -m py_compile hamgnn/data/graph_data.py
  Uni-HamGNN/Uni-HamiltonianPredictor.py tests/test_lightning2_imports.py`:
  passed.
- `pytest tests/test_lightning2_imports.py tests/test_graph_data_cache.py -q`:
  blocked because the active Python environment has no `pytest` module.
- Import preflight: the active environment also lacks `lightning` and
  `torch_geometric`, so runtime collection could not be attempted.

## Concerns

The focused tests still need to run in an environment containing the project
runtime dependencies and pytest. No source-level stale `pytorch_lightning`
imports remain in the owned paths covered by the test. The Task 3 diff for
`hamgnn/data/graph_data.py` is now only the intended import replacement; its
unrelated CRLF normalization has been removed.
