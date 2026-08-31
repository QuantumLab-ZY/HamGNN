# Task 4 Report

## Status

Implemented runtime-generated Lightning 1.5 checkpoint compatibility coverage.
The focused continuation test passes on Lightning 2.5.0 with PyTorch 2.5.0.

## Changes

- Added `test_lightning15_checkpoint_restores_and_continues_cpu_fit` to
  `tests/test_lightning2_model.py`.
- The test creates a checkpoint under `tmp_path` with model weights,
  AdamW state, ReduceLROnPlateau state, epoch, global step, hyperparameters,
  and the Lightning 1.5.10 version marker. No binary fixture is committed.
- The test verifies restored model weight, optimizer step, scheduler
  `last_epoch` and `best`, epoch/global-step values at `on_train_start`, and
  that both counters advance after continuation through `train_model`.
- Added a narrowly scoped `Model.on_load_checkpoint` adjustment that carries
  legacy top-level `epoch` and `global_step` into migrated loop progress.
  It is gated by Lightning's legacy-version marker and does not affect native
  Lightning 2 checkpoints.

## Verification

Focused command:

```text
PYTHONPATH=/data/home/zhongyang/Github/HamGNN conda run -n HamGNN-oeq-pl2.x python -m pytest tests/test_lightning2_model.py::test_lightning15_checkpoint_restores_and_continues_cpu_fit -q
```

Result: `1 passed, 20 warnings`.

Full requested migration command:

```text
PYTHONPATH=/data/home/zhongyang/Github/HamGNN conda run -n HamGNN-oeq-pl2.x python -m pytest tests/test_lightning2_main.py tests/test_lightning2_model.py tests/test_lightning2_imports.py tests/test_graph_data_cache.py -q
```

Result: `49 passed, 4 failed, 25 warnings`.

The four failures are unrelated to Task 4:

- `test_setup_trainer_uses_static_graph_ddp_for_multiple_inferred_gpus`:
  Lightning 2.5 `DDPStrategy` has no `static_graph` attribute.
- `test_validation_gathers_before_rank_zero_filter`.
- `test_test_epoch_end_writes_expected_outputs_and_epc`.
- `test_test_epoch_end_gathers_before_rank_zero_filter`.

The last three failures are existing mocked validation/test hook assumptions
that do not hold with the installed Lightning 2.5 runtime. No unrelated tests
or NequIP files were modified.

Static checks passed:

```text
python -m compileall -q hamgnn/models/Model.py hamgnn/main.py tests/test_lightning2_model.py tests/test_lightning2_main.py
```

## Self-review

The checkpoint remains generated during the test and is not stored in the
repository. The fixture uses AdamW, a single ReduceLROnPlateau state dict,
`epoch=1`, `global_step=1`, optimizer step 1, scheduler `last_epoch=1`, and
finite scheduler best metric `0.25`. The production edit is limited to the
legacy restore hook; `hamgnn/main.py` required no change because it already
passes `ckpt_path` to `Trainer.fit`.
