from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import hamgnn.main as main
from hamgnn.main import (
    _count_requested_gpus,
    _normalize_num_gpus,
    setup_trainer,
    train_model,
)


@pytest.mark.parametrize(
    ('value', 'normalized', 'count'),
    [
        (None, None, 0),
        (0, None, 0),
        ('0', None, 0),
        ([], None, 0),
        (2, 2, 2),
        ([1, 3], [1, 3], 2),
        ((1, 3), (1, 3), 2),
    ],
)
def test_num_gpus_normalization(value, normalized, count):
    assert _normalize_num_gpus(value) == normalized
    assert _count_requested_gpus(normalized) == count


@pytest.mark.parametrize('value', [-1, 1.5, '1', {}, True, False])
def test_invalid_num_gpus_values_raise(value):
    with pytest.raises(ValueError, match='setup.num_gpus'):
        _normalize_num_gpus(value)


def _config(tmp_path, num_gpus=None, accelerator=None, resume=False, checkpoint_path=None):
    return SimpleNamespace(
        setup=SimpleNamespace(
            num_gpus=num_gpus,
            accelerator=accelerator,
            resume=resume,
            checkpoint_path=checkpoint_path,
            precision=32,
        ),
        optim_params=SimpleNamespace(
            gradient_clip_val=0.0,
            max_epochs=1,
            min_epochs=0,
        ),
        profiler_params=SimpleNamespace(train_dir=str(tmp_path)),
    )


@pytest.mark.parametrize(
    ('num_gpus', 'accelerator', 'expected'),
    [
        (None, None, ('cpu', 1)),
        (0, None, ('cpu', 1)),
        ('0', None, ('cpu', 1)),
        ([], None, ('cpu', 1)),
        (2, None, ('gpu', 2)),
        ([1, 3], None, ('gpu', [1, 3])),
        (4, 'cpu', ('cpu', 1)),
        (2, 'gpu', ('gpu', 2)),
    ],
)
def test_setup_trainer_resolves_lightning2_devices(
    tmp_path, monkeypatch, num_gpus, accelerator, expected
):
    captured = {}

    class TrainerDouble:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(main.pl, 'Trainer', TrainerDouble)
    setup_trainer(_config(tmp_path, num_gpus, accelerator), callbacks=[])
    assert captured['accelerator'] == expected[0]
    assert captured['devices'] == expected[1]


def test_setup_trainer_uses_static_graph_ddp_for_multiple_inferred_gpus(
    tmp_path, monkeypatch
):
    captured = {}

    class TrainerDouble:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(main.pl, 'Trainer', TrainerDouble)
    setup_trainer(_config(tmp_path, 2), callbacks=[])
    assert captured['strategy'].__class__.__name__ == 'DDPStrategy'
    assert captured['strategy'].static_graph is True


@pytest.mark.parametrize('accelerator', ['tpu', 'ddp'])
def test_invalid_accelerator_combinations_raise(tmp_path, accelerator):
    if accelerator == 'ddp':
        with pytest.raises(ValueError, match='more than one'):
            setup_trainer(_config(tmp_path, 1, accelerator), callbacks=[])
    else:
        with pytest.raises(ValueError, match='accelerator'):
            setup_trainer(_config(tmp_path, 0, accelerator), callbacks=[])


def test_train_model_passes_checkpoint_path_to_fit():
    trainer = Mock()
    trainer.test.return_value = [{'test/total_loss': 0.5}]
    model = object()
    data_module = object()

    assert train_model(trainer, model, data_module, ' /tmp/resume.ckpt ') == [
        {'test/total_loss': 0.5}
    ]
    trainer.fit.assert_called_once_with(
        model, data_module, ckpt_path=' /tmp/resume.ckpt '
    )


def test_train_model_passes_none_for_new_fit():
    trainer = Mock()
    trainer.test.return_value = []
    model = object()
    data_module = object()

    train_model(trainer, model, data_module, None)
    trainer.fit.assert_called_once_with(model, data_module, ckpt_path=None)


def test_resume_requires_non_empty_checkpoint_path():
    config = SimpleNamespace(setup=SimpleNamespace(resume=True, checkpoint_path='  '))

    with pytest.raises(ValueError, match='checkpoint path'):
        main._resume_checkpoint_path(config)


@pytest.mark.parametrize('checkpoint_path', [None, ''])
def test_resume_requires_checkpoint_path_value(checkpoint_path):
    config = SimpleNamespace(
        setup=SimpleNamespace(resume=True, checkpoint_path=checkpoint_path)
    )

    with pytest.raises(ValueError, match='checkpoint path'):
        main._resume_checkpoint_path(config)


def test_resume_requires_checkpoint_path_attribute():
    config = SimpleNamespace(setup=SimpleNamespace(resume=True))

    with pytest.raises(ValueError, match='checkpoint path'):
        main._resume_checkpoint_path(config)


@pytest.mark.parametrize('precision', [16, '32', 128])
def test_setup_trainer_rejects_unsupported_precision(tmp_path, precision):
    config = _config(tmp_path)
    config.setup.precision = precision

    with pytest.raises(ValueError, match='precision'):
        setup_trainer(config, callbacks=[])


def test_resume_checkpoint_path_is_trimmed():
    config = SimpleNamespace(
        setup=SimpleNamespace(resume=True, checkpoint_path='  /tmp/resume.ckpt  ')
    )

    assert main._resume_checkpoint_path(config) == '/tmp/resume.ckpt'


def test_resume_checkpoint_path_is_none_for_new_fit():
    config = SimpleNamespace(setup=SimpleNamespace(resume=False, checkpoint_path=''))

    assert main._resume_checkpoint_path(config) is None


def test_resume_defaults_to_false_when_missing():
    config = SimpleNamespace(setup=SimpleNamespace())

    assert main._resume_checkpoint_path(config) is None
