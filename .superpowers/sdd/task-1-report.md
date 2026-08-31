# Task 1 Report: Lightning 2 Trainer and Resume Boundary

## Scope

Implemented Task 1 from `task-1-brief.md` on the current branch at migration base
`e4e3fb7b318c9e0417a66e9f4d5a96c951221726`.

Changed files:

- `hamgnn/main.py`
- `tests/test_lightning2_main.py`
- `.superpowers/sdd/task-1-report.md`

The protected `hamgnn/toolbox/nequip` subtree was not modified.

## Implementation

- Replaced the compatibility Lightning import block with Lightning 2 imports:
  `lightning.pytorch`, `TensorBoardLogger`, and `DDPStrategy`.
- Added Lightning 2 trainer device configuration using `accelerator` and
  `devices`.
- Normalized absent, zero, string-zero, and empty GPU configurations to CPU
  execution.
- Validated explicit accelerator values and invalid GPU/accelerator combinations.
- Configured static-graph `DDPStrategy` for inferred or explicitly requested
  multi-GPU DDP.
- Removed Lightning 1 trainer arguments `gpus`, `resume_from_checkpoint`, and
  `progress_bar_refresh_rate`.
- Added `_resume_checkpoint_path(config)` to validate and trim resume paths.
- Added the optional `checkpoint_path` argument to `train_model` and passed it to
  `trainer.fit(..., ckpt_path=checkpoint_path)`.
- Kept `load_from_checkpoint` behavior separate from resume behavior.

## Tests

Added `tests/test_lightning2_main.py` covering:

- GPU normalization and counting.
- CPU/GPU/device-list accelerator resolution.
- Static-graph DDP selection.
- Invalid accelerator combinations.
- Resume checkpoint path validation and trimming.
- `ckpt_path` forwarding for resumed and new fits.

Verification performed:

- `python -m py_compile hamgnn/main.py tests/test_lightning2_main.py`: passed.
- `pytest tests/test_lightning2_main.py -q`: blocked because `pytest` is not
  installed in the environment.
- `pytest -q`: blocked because `pytest` is not installed in the environment.

The environment also does not currently provide `lightning`, `torch`,
`torch_geometric`, or `e3nn`, so the test suite could not be installed or run
from this workspace without changing dependencies.

## Self-review

The implementation is limited to the requested trainer and resume orchestration
boundary. No changes were made to model loading semantics, configuration
defaults, or the protected NequIP subtree. The existing `main.py` uses CRLF line
endings; preserving that format causes `git diff --check` to identify CR bytes
on changed lines as trailing whitespace, although no new visible trailing-space
content was introduced.

## Fixes After Task 1 Review

Changed files:

- `hamgnn/main.py`
- `tests/test_lightning2_main.py`
- `.superpowers/sdd/task-1-report.md`

Fixes applied:

- Imported `train_model` in `tests/test_lightning2_main.py`.
- Made resumed fits with a missing, non-string, or blank checkpoint path raise
  `ValueError`.
- Enforced the documented `num_gpus` contract so invalid values raise
  `ValueError` instead of falling through to CPU execution.
- Restricted trainer and model precision handling to supported `32` and `64`
  values, with focused boundary tests.
- The protected `hamgnn/toolbox/nequip` subtree remains unchanged.

Verification:

- Command: `python -m py_compile hamgnn/main.py tests/test_lightning2_main.py`
- Output: passed (no output).
- Command: `pytest tests/test_lightning2_main.py -q`
- Output: blocked: `/bin/bash: pytest: command not found`.

## Remaining Task 1 Review Fix

- Made `_resume_checkpoint_path` use optional lookup for `setup.resume` and
  `setup.checkpoint_path`.
- Added regression coverage for a missing checkpoint path while resuming and a
  missing `setup.resume` defaulting to false.

Verification:

- Command: `python -m py_compile hamgnn/main.py tests/test_lightning2_main.py`
- Output: passed (no output).
- Command: `pytest tests/test_lightning2_main.py -q`
- Output: blocked: `/bin/bash: pytest: command not found`.

## Final Task 1 Review Fix

- Made `load_or_create_model` use the optional `setup.resume` lookup as well,
  keeping resume handling consistent across model loading and fit orchestration.
- Added a regression test covering checkpoint loading when `setup.resume` is
  omitted.

Verification:

- Command: `python -m py_compile hamgnn/main.py tests/test_lightning2_main.py`
- Output: passed (no output).
- Command: `pytest tests/test_lightning2_main.py -q`
- Output: blocked: `/bin/bash: pytest: command not found`.
