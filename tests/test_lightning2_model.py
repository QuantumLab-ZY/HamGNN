from unittest.mock import Mock

import lightning.pytorch as pl
import numpy as np
import pytest
import torch
from torch import nn

from hamgnn.models.Model import Model


class ScalarRepresentation(nn.Module):
    def forward(self, batch):
        return batch["x"]


class ScalarOutput(nn.Module):
    derivative = False

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, batch, representation):
        return {"prediction": representation * self.weight}


class ScalarLoss:
    name = "mae"

    def __call__(self, prediction, target):
        return torch.mean(torch.abs(prediction - target))


def make_model():
    return Model(
        representation=ScalarRepresentation(),
        output=ScalarOutput(),
        losses=[{
            "metric": ScalarLoss(),
            "prediction": "prediction",
            "target": "target",
            "loss_weight": 1.0,
        }],
        validation_metrics=[],
        lr=0.01,
        lr_decay=0.5,
        lr_patience=1,
    )


@pytest.fixture
def model():
    return make_model()


@pytest.fixture
def trainer(tmp_path):
    logger = Mock(log_dir=str(tmp_path), experiment=Mock())
    return Mock(sanity_checking=False, is_global_zero=True, logger=logger)


class TinyDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.samples = [{"x": torch.tensor([1.0]), "target": torch.tensor([0.0])}]

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.samples, batch_size=1)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.samples, batch_size=1)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.samples, batch_size=1)


def test_validation_epoch_end_uses_buffer_and_skips_sanity_check(model, trainer, monkeypatch):
    model.trainer = trainer
    model.validation_step_outputs = [{"pred": {"hamiltonian": np.array([1])}, "target": {"hamiltonian": np.array([1])}}]
    trainer.sanity_checking = True
    plot = Mock()
    monkeypatch.setattr(model, "_plot_prediction_vs_target", plot)

    model.on_validation_epoch_end()

    plot.assert_not_called()
    assert model.validation_step_outputs == []


def test_epoch_start_hooks_clear_buffers(model):
    model.validation_step_outputs.append({})
    model.test_step_outputs.append({})

    model.on_validation_epoch_start()
    model.on_test_epoch_start()

    assert model.validation_step_outputs == []
    assert model.test_step_outputs == []


def test_validation_gathers_before_rank_zero_filter(model, trainer, monkeypatch):
    model.trainer = trainer
    trainer.is_global_zero = False
    local_output = {"pred": {}, "target": {}}
    remote_output = {"pred": {}, "target": {}}
    model.validation_step_outputs = [local_output]
    gather = Mock(return_value=[local_output, remote_output])
    monkeypatch.setattr(model, "_gather_step_outputs", gather)
    plot = Mock()
    monkeypatch.setattr(model, "_plot_prediction_vs_target", plot)

    model.on_validation_epoch_end()

    gather.assert_called_once_with([local_output])
    plot.assert_not_called()
    assert model.validation_step_outputs == []


def test_epoch_end_clears_buffers_when_processing_raises(model, trainer, monkeypatch):
    model.trainer = trainer
    model.validation_step_outputs = [{}]
    monkeypatch.setattr(model, "_gather_step_outputs", Mock(side_effect=RuntimeError("boom")))

    with pytest.raises(RuntimeError, match="boom"):
        model.on_validation_epoch_end()

    assert model.validation_step_outputs == []


def test_test_epoch_end_writes_expected_outputs_and_epc(model, trainer, tmp_path, monkeypatch):
    model.trainer = trainer
    trainer.logger.log_dir = str(tmp_path)
    model.test_step_outputs = [{
        "pred": {"hamiltonian": np.array([1.0])},
        "target": {"hamiltonian": np.array([2.0])},
        "processed_values": {"epc_mat": np.array([3.0])},
    }]
    model.post_processing = object()
    plot = Mock()
    monkeypatch.setattr(model, "_plot_prediction_vs_target", plot)
    gathered = list(model.test_step_outputs)
    monkeypatch.setattr(model, "_gather_step_outputs", Mock(return_value=gathered))

    model.on_test_epoch_end()

    assert (tmp_path / "prediction_hamiltonian.npy").exists()
    assert (tmp_path / "target_hamiltonian.npy").exists()
    assert (tmp_path / "processed_values_epc_mat.npy").exists()
    plot.assert_called_once_with(gathered, mode="test")
    assert model.test_step_outputs == []


def test_test_epoch_end_gathers_before_rank_zero_filter(model, trainer, monkeypatch):
    model.trainer = trainer
    trainer.is_global_zero = False
    output = {"pred": {}, "target": {}, "processed_values": None}
    model.test_step_outputs = [output]
    gather = Mock(return_value=[output])
    monkeypatch.setattr(model, "_gather_step_outputs", gather)

    model.on_test_epoch_end()

    gather.assert_called_once_with([output])
    assert model.test_step_outputs == []


def test_test_epoch_end_supports_outputs_without_processed_values(model, trainer, tmp_path, monkeypatch):
    model.trainer = trainer
    trainer.logger.log_dir = str(tmp_path)
    output = {"pred": {}, "target": {}}
    model.test_step_outputs = [output]
    monkeypatch.setattr(model, "_gather_step_outputs", Mock(return_value=[output]))
    monkeypatch.setattr(model, "_save_predictions_and_targets", Mock())
    monkeypatch.setattr(model, "_plot_prediction_vs_target", Mock())
    model.post_processing = object()

    model.on_test_epoch_end()

    assert model.test_step_outputs == []


def test_test_epoch_end_clears_buffer_when_processing_raises(model, trainer, monkeypatch):
    model.trainer = trainer
    model.test_step_outputs = [{"pred": {}, "target": {}}]
    monkeypatch.setattr(model, "_gather_step_outputs", Mock(side_effect=RuntimeError("boom")))

    with pytest.raises(RuntimeError, match="boom"):
        model.on_test_epoch_end()

    assert model.test_step_outputs == []


def test_configure_optimizers_preserves_plateau_scheduler(model):
    optimizers, schedulers = model.configure_optimizers()
    scheduler = schedulers[0]

    assert scheduler["monitor"] == "training/total_loss"
    assert scheduler["interval"] == "epoch"
    assert scheduler["frequency"] == 1
    assert scheduler["strict"] is True
    assert scheduler["scheduler"].factor == model.lr_decay


def test_minimal_cpu_fit_validate_test_executes(tmp_path):
    model = make_model()
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        logger=pl.loggers.TensorBoardLogger(save_dir=str(tmp_path), name="smoke"),
        enable_checkpointing=False,
        default_root_dir=str(tmp_path),
    )
    data = TinyDataModule()

    trainer.fit(model, datamodule=data)
    trainer.validate(model, datamodule=data)
    trainer.test(model, datamodule=data)
