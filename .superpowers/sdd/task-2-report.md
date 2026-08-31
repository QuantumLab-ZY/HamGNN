# Task 2 Report: Lightning 2 Model Lifecycle and Artifacts

## Scope

Implemented Task 2 from `task-2-brief.md` on the current branch. The change is
limited to the model lifecycle boundary, its focused tests, the checkpoint
filename configuration, and this report.

Changed files:

- `hamgnn/models/Model.py`
- `hamgnn/main.py`
- `tests/test_lightning2_model.py`
- `.superpowers/sdd/task-2-report.md`

The protected `hamgnn/toolbox/nequip` subtree was not modified.

## Implementation

- Switched `Model` to `import lightning.pytorch as pl`.
- Added per-instance `validation_step_outputs` and `test_step_outputs` buffers.
- Appended the returned output dictionary in both `validation_step` and
  `test_step` before returning it.
- Replaced the removed Lightning 1.x `validation_epoch_end` and
  `test_epoch_end` hooks with Lightning 2-compatible epoch-start and
  epoch-end hooks.
- Cleared buffers at epoch start and in `finally` blocks at epoch end.
- Preserved distributed gathering before rank-zero filtering.
- Preserved sanity-check suppression for validation plots.
- Preserved validation/test metric names, distributed synchronization, plot
  tags, global-step handling, prediction/target filenames, and EPC output
  filename.
- Added the filename-safe `validation_total_loss` logging alias while retaining
  `validation/total_loss`.
- Updated the checkpoint callback filename to
  `{epoch}-{validation_total_loss:.6f}` while retaining its
  `validation/total_loss` monitor.
- Kept the existing ReduceLROnPlateau constructor values and returned the
  Lightning 2 scheduler metadata structure with `monitor`, `interval`,
  `frequency`, and `strict`.

## Tests Added

`tests/test_lightning2_model.py` uses deterministic scalar representation,
output, loss, and data-module doubles. It covers:

- Validation sanity-check suppression and buffer cleanup.
- Epoch-start buffer reset.
- Gather-before-rank-zero filtering for validation and test.
- Buffer cleanup when epoch processing raises.
- Test prediction, target, EPC artifact creation, and plot invocation.
- ReduceLROnPlateau scheduler metadata and factor.
- A minimal CPU `fit`, `validate`, and `test` lifecycle smoke test.

## Verification

Command:

```text
python -m py_compile hamgnn/models/Model.py hamgnn/main.py tests/test_lightning2_model.py
```

Output: no output; passed.

Command:

```text
pytest tests/test_lightning2_model.py -q
```

Output:

```text
/bin/bash: pytest: command not found
```

Command:

```text
pytest tests/test_lightning2_model.py tests/test_graph_data_cache.py -q
```

Output:

```text
/bin/bash: pytest: command not found
```

Command:

```text
git diff --check
```

Output: reports carriage-return characters on changed lines in the existing
CRLF-formatted `hamgnn/main.py` and `hamgnn/models/Model.py`. The files retain
their repository CRLF convention; no visible trailing-space content remains.

The environment does not provide the `pytest` executable. Runtime test
dependencies such as Lightning and PyTorch are also not available for a full
test run in this workspace, so the lifecycle and smoke tests could not be
executed here.

## Self-review

The implementation removes the unsupported lifecycle hook names from `Model`
and uses instance buffers, avoiding cross-epoch or shared mutable output state.
Gathering occurs before rank-zero checks on both epoch-end paths, and all early
returns still pass through buffer cleanup. Test artifact handling only writes
processed EPC values when at least one gathered output provides them, avoiding
an empty `np.concatenate` call.

No changes were made to `hamgnn/toolbox/nequip`. The unrelated legacy import in
`hamgnn/data/graph_data.py` remains outside this task's requested model
lifecycle scope.
