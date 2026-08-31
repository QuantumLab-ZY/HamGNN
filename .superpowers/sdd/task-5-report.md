# Task 5 Report

## Scope

Implemented the Lightning 2 dependency, metadata, example, documentation,
intersphinx, historical-example removal, and consistency-test requirements
from `task-5-brief.md`.

Preserved `source_v1` handling in `docs/source/conf.py`. No files under
`docs/_build/` or `hamgnn/toolbox/nequip/` were changed.

## Changes

- Replaced both environment declarations with `lightning>=2.5,<2.6`.
- Added `lightning>=2.5,<2.6` to `setup.py` and raised `python_requires` to `>=3.9`.
- Updated the V2.x configuration and removed the obsolete
  `progress_bar_refresh_rat` setting from the example and user parameter docs.
- Updated README and user-guide documentation for Lightning 2.5,
  `lightning.pytorch`, accelerator values, `num_gpus`, precision, and exact
  resume semantics.
- Changed the Sphinx intersphinx key to `lightning.pytorch`.
- Deleted `examples/V1.0/` and added `tests/test_lightning2_docs.py`.

## Verification

### Prescribed consistency test

Command:

```text
pytest tests/test_lightning2_docs.py -q
```

Output:

```text
/bin/bash: pytest: command not found
```

Fallback command:

```text
python -m pytest tests/test_lightning2_docs.py -q
```

Output:

```text
/data/home/zhongyang/miniconda3/bin/python: No module named pytest
```

The five original test functions were executed directly through Python after import.
All passed:

```text
PASS test_dependency_files_declare_lightning_25
PASS test_historical_examples_are_removed
PASS test_setup_metadata_requires_supported_python_and_lightning
PASS test_sphinx_intersphinx_uses_lightning_namespace
PASS test_user_docs_describe_lightning2_resume_and_keep_num_gpus
```

### Import checks

The source-text import check passed:

```text
PASS test_hamgnn_owned_sources_use_lightning_pytorch
```

The runtime graph-data-module check could not run because the environment does
not contain Lightning:

```text
BLOCKED test_graph_data_module_uses_lightning2_base: ModuleNotFoundError: No module named 'lightning'
```

### Static checks

Passed:

```text
PASS syntax and scope assertions
PASS static scope and syntax checks
```

These checks included YAML parsing, setup/test Python compilation,
`git diff --check`, exact dependency and metadata assertions, absence of the
V1.0 directory, retention of `source_v1`, and rejection of changed generated
docs or NequIP paths.

The requested `rg` scan could not run because `rg` is not installed:

```text
/bin/bash: rg: command not found
```

### Sphinx build

Command requested:

```text
sphinx-build -W -b html docs/source /tmp/hamgnn-sphinx-html
```

The build was not run because Sphinx is unavailable. The preflight check
reported:

```text
sphinx unavailable
```

No output was written to `docs/_build/`.

## Concerns

- Full pytest execution, the Lightning runtime import assertion, the Sphinx
  build, and the requested `rg` scan remain unverified until their tools and
  dependencies are installed.
- Existing historical checkpoint metadata containing the old package name was
  not changed; the task scope concerns current dependency declarations and
  user-facing documentation.

## Review Fix Verification

Addressed all actionable Task 5 review findings:

- Synchronized `docs/environment.yml` with the documented PyTorch 2.5.0 and
  PyG 2.6.1 stack, including matching PyG wheel links and the pinned
  `torch-geometric==2.6.1` dependency.
- Strengthened `tests/test_lightning2_docs.py` to assert the exact resume call,
  the non-empty checkpoint requirement, CPU/GPU `num_gpus` mapping, supported
  accelerator values, precision 32/64 behavior, and the parsed V2.x setup
  contract directly.
- Distinguished intentional legacy checkpoint metadata in
  `tests/test_lightning2_model.py` from stale current dependency, import, and
  documentation references.
- Corrected the direct consistency-test count from six to eight functions.

Verification performed:

```text
PASS direct execution of all 8 tests in tests/test_lightning2_docs.py
PASS python -m py_compile setup.py tests/test_lightning2_docs.py
PASS git diff --check
BLOCKED pytest tests/test_lightning2_docs.py -q: pytest command not found
BLOCKED python -m pytest tests/test_lightning2_docs.py -q: No module named pytest
BLOCKED sphinx-build -W -b html docs/source /tmp/hamgnn-sphinx-html: sphinx-build command not found
```
