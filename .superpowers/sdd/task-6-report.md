# Task 6: Final Migration Verification Report

Date: 2026-08-31
Environment: `HamGNN-oeq-pl2.x`

## Corrections

- Updated the DDP strategy test to verify Lightning 2.5's supported internal
  `_ddp_kwargs["static_graph"]` value. `DDPStrategy` no longer exposes a
  public `static_graph` attribute, while the production constructor still
  receives `static_graph=True`.
- Updated validation and test epoch-end hooks to gather a shallow snapshot of
  the local output buffer. This preserves gathered results when the required
  `finally` cleanup clears the live buffer, including non-distributed tests
  where gather returns its input list.
- Corrected the test artifact fixture to use the prediction and target keys
  declared by its `ScalarLoss` contract.
- Kept the import scan broad and split intentional legacy literals in the
  migration tests so the scan does not report its own assertion source.

## Verification

Initial focused command, before correction:

```text
5 failed, 56 passed
```

The failures were the five Task 6 issues: missing `DDPStrategy.static_graph`,
three lifecycle/artifact failures caused by buffer aliasing, and the stale
source scan matching the intentional docs-test literal.

Focused migration suite after correction:

```text
conda run -n HamGNN-oeq-pl2.x python -m pytest tests/test_lightning2_main.py tests/test_lightning2_model.py tests/test_lightning2_imports.py tests/test_lightning2_docs.py tests/test_graph_data_cache.py -q
61 passed, 25 warnings in 83.40s
```

Full repository suite:

```text
conda run -n HamGNN-oeq-pl2.x python -m pytest -q
61 passed, 25 warnings in 90.81s
```

Stale-reference scan:

- The requested `rg` command could not run because `rg` is not installed in
  the environment.
- Equivalent repository searches found no stale references under `hamgnn`
  or `Uni-HamGNN`. The only test matches were the intentionally retained
  historical-example assertions in `test_lightning2_docs.py` and the scan's
  own pattern string in `test_lightning2_imports.py`; both files are explicitly
  excluded by the requested scan.

Protected NequIP scope check:

```text
git diff --name-only HEAD~5..HEAD -- hamgnn/toolbox/nequip
```

Result: no output.

Diff/status checks:

- `git diff --check` is run with `cr-at-eol` handling because the repository's
  existing `Model.py` uses CRLF line endings.
- No whitespace errors were found under that repository-compatible check.
- The semantic diff contains only the four correction files plus this report.
- Pre-existing untracked `.superpowers` review/brief artifacts were left
  untouched.

## Self-review

The production behavior remains gather-before-rank-zero filtering, and cleanup
still runs on success, rank-zero skips, and exceptions. The snapshot only
prevents cleanup from mutating the gathered non-distributed return value. The
strategy test checks the actual Lightning 2.5 constructor state rather than
weakening the production setting. No NequIP files or unrelated documentation
were changed.

## Commit

Commit required because final verification identified and corrected five
failures. The fix commit is recorded after this report is staged.

## Concerns

- Test execution emits 25 existing warnings related to matplotlib parsing,
  SLURM detection, single-worker dataloaders, logging intervals, and direct
  checkpoint restoration. They do not fail the suite.
- The exact stale scan was unavailable due to the missing `rg` executable;
  equivalent scoped searches and the migration tests passed.

## Final Fix Wave

- Synchronized `docs/source/user_guide/parameters.rst` with the runtime
  `config_default`: `setup.checkpoint_path` defaults to `'./'`, rather than
  `null`. The V2 example continues to set `null` explicitly.
- Added a focused consistency assertion that the documented value and runtime
  default remain aligned.
- Commands and results:

```text
conda run -n HamGNN-oeq-pl2.x python -m pytest tests/test_lightning2_main.py tests/test_lightning2_model.py tests/test_lightning2_imports.py tests/test_lightning2_docs.py tests/test_graph_data_cache.py -q
62 passed, 25 warnings in 69.65s

conda run -n HamGNN-oeq-pl2.x python -m pytest -q
62 passed, 25 warnings in 77.33s

rg -n "pytorch_lightning|resume_from_checkpoint|progress_bar_refresh_rate|['\"]gpus['\"]|examples/V1\.0" hamgnn Uni-HamGNN tests README.md docs/source HamGNN.yaml docs/environment.yml setup.py -g '!hamgnn/toolbox/nequip/**' -g '!tests/test_lightning2_imports.py' -g '!tests/test_lightning2_docs.py'
/bin/bash: rg: command not found

Equivalent scan using git grep with the same pattern and scope
No matches.

conda run -n base sphinx-build -W -b html docs/source /tmp/hamgnn-sphinx-html
Failed: sphinx-build: command not found

conda run -n HamGNN sphinx-build -W -b html docs/source /tmp/hamgnn-sphinx-html
Failed: sphinx-build: command not found

conda run -n HamGNN-oeq sphinx-build -W -b html docs/source /tmp/hamgnn-sphinx-html
Failed: sphinx-build: command not found

conda run -n HamGNN-oeq-pl2.x sphinx-build -W -b html docs/source /tmp/hamgnn-sphinx-html
Failed: sphinx-build: command not found

conda run -n openequivariance-py311 sphinx-build -W -b html docs/source /tmp/hamgnn-sphinx-html
Failed: sphinx-build: command not found

conda run -n HamGNN-oeq-pl2.x python -m pytest tests/test_lightning2_docs.py -q
Initial post-edit attempt: 1 failed, 8 passed. The failure was the new
test's overly specific RST indentation assertion; it was corrected.

conda run -n HamGNN-oeq-pl2.x python -m pytest tests/test_lightning2_docs.py -q
9 passed, 0 warnings in 0.34s
```
