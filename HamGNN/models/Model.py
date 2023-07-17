"""
/*
 * @Author: Yang Zhong 
 * @Date: 2021-10-09 13:46:53 
 * @Last Modified by: Yang Zhong
 * @Last Modified time: 2021-10-29 21:09:02
 */
"""
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as opt
from typing import List, Dict, Union
from torch.nn import functional as F
from .utils import scatter_plot
import numpy as np
import os
import pandas as pd


class Model(pl.LightningModule):
    def __init__(
            self,
            representation: nn.Module,
            output: nn.Module,
            losses: List[Dict],
            validation_metrics: List[Dict],
            lr: float = 1e-3,
            lr_decay: float = 0.1,
            lr_patience: int = 100,
            lr_monitor="training/total_loss",
            epsilon: float = 1e-8,
            beta1: float = 0.99,
            beta2: float = 0.999,
            amsgrad: bool = True,
            max_points_to_scatter: int = 100000,
            post_processing: callable = None
            ):
        super().__init__()

        self.representation = representation
        self.output_module = output

        self.losses = losses
        self.metrics = validation_metrics

        self.lr = lr
        self.lr_decay = lr_decay
        self.lr_patience = lr_patience
        self.lr_monitor = lr_monitor
        
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        self.amsgrad = amsgrad
        
        self.max_points_to_scatter = max_points_to_scatter
        # post_processing is used to calculate some physical quantities that rely on gradient backpropagation
        self.post_processing = post_processing

        #self.save_hyperparameters()

        # For gradients
        self.requires_dr = self.output_module.derivative

    def calculate_loss(self, batch, result, mode):
        loss = torch.tensor(0.0, device=self.device)
        for loss_dict in self.losses:
            loss_fn = loss_dict["metric"]

            if "target" in loss_dict.keys():
                pred = result[loss_dict["prediction"]]
                target = batch[loss_dict["target"]]

                loss_i = loss_fn(pred, target)
            else:
                loss_i = loss_fn(result[loss_dict["prediction"]])
            loss += loss_dict["loss_weight"] * loss_i

            if hasattr(loss_fn, "name"):
                lossname = loss_fn.name
            else:
                lossname = type(loss_fn).__name__.split(".")[-1]

            self.log(
                mode
                + "/"
                + lossname
                + "_"
                + loss_dict["prediction"],
                loss_i,
                on_step=False,
                on_epoch=True,
            )

        return loss

    def training_step(self, data, batch_idx):
        self._enable_grads(data)
        pred = self(data)
        loss = self.calculate_loss(data, pred, 'training')
        self.log("training/total_loss", loss, on_step=False, on_epoch=True)
        # self.check_param()
        return loss

    def validation_step(self, data, batch_idx):
        if self.requires_dr:
            torch.set_grad_enabled(True)
        else:
            torch.set_grad_enabled(False)
        self._enable_grads(data)
        pred = self(data)
        val_loss = self.calculate_loss(
            data, pred, 'validation').detach().item()
        self.log("validation/total_loss", val_loss,
                 on_step=False, on_epoch=True)
        self.log_metrics(data, pred, 'validation')
        outputs_pred, outputs_target = {}, {}
        for loss_dict in self.losses:
            outputs_pred[loss_dict["prediction"]] = pred[loss_dict["prediction"]].detach().cpu().numpy()  
            outputs_target[loss_dict["target"]] = data[loss_dict["target"]].detach().cpu().numpy()      
        return {'pred': outputs_pred, 'target': outputs_target}

    def validation_epoch_end(self, validation_step_outputs):
        for loss_dict in self.losses:
            if "target" in loss_dict.keys():
                pred = np.concatenate([out['pred'][loss_dict["prediction"]]
                                 for out in validation_step_outputs])
                target = np.concatenate([out['target'][loss_dict["target"]]
                                    for out in validation_step_outputs])
                if (pred.dtype == np.complex64) and (target.dtype == np.complex64):
                    lossname = type(loss_dict['metric']).__name__.split(".")[-1]
                    if lossname.lower() == 'abs_mae':
                        pred = np.absolute(pred)
                        target = np.absolute(target)
                    else:
                        pred = np.concatenate([pred.real, pred.imag], axis=-1)
                        target = np.concatenate([target.real, target.imag], axis=-1)
                # Control the number of scatter points to plot
                if pred.size > self.max_points_to_scatter:
                    random_state = np.random.RandomState(seed=42)
                    perm = list(random_state.permutation(np.arange(pred.size)))
                    pred = pred.reshape(-1)[perm[:self.max_points_to_scatter]]
                    target = target.reshape(-1)[perm[:self.max_points_to_scatter]]
                figure = scatter_plot(pred.reshape(-1), target.reshape(-1))
                figname = 'PredVSTarget_' + loss_dict['prediction']
                self.logger.experiment.add_figure(
                    'validation/'+figname, figure, global_step=self.global_step)
            else:
                pass

    def test_step(self, data, batch_idx):
        if self.requires_dr:
            torch.set_grad_enabled(True)
        else:
            torch.set_grad_enabled(False)
        self._enable_grads(data)
        
        if self.post_processing is not None:
            pred = self.post_processing(data)
            if type(self.post_processing).__name__.split(".")[-1].lower() == 'epc_output':
                proessed_values = {'epc_mat': pred['epc_mat'].detach().cpu().numpy()}
            else:
                raise NotImplementedError
        else:
            pred = self(data)
            proessed_values = None
            
        loss = self.calculate_loss(data, pred, 'test').detach().item()
        self.log("test/total_loss", loss, on_step=False, on_epoch=True)
        self.log_metrics(data, pred, "test") 
        outputs_pred, outputs_target = {}, {}
        for loss_dict in self.losses:
            outputs_pred[loss_dict["prediction"]] = pred[loss_dict["prediction"]].detach().cpu().numpy()  
            outputs_target[loss_dict["target"]] = data[loss_dict["target"]].detach().cpu().numpy()      
        return {'pred': outputs_pred, 'target': outputs_target, 'processed_values': proessed_values}

    def test_epoch_end(self, test_step_outputs):
        for loss_dict in self.losses:
            if "target" in loss_dict.keys():
                pred = np.concatenate([out['pred'][loss_dict["prediction"]]
                                 for out in test_step_outputs])
                target = np.concatenate([out['target'][loss_dict["target"]]
                                    for out in test_step_outputs])
                
                if not os.path.exists(self.trainer.logger.log_dir):
                    os.makedirs(self.trainer.logger.log_dir)
                    
                np.save(os.path.join(
                    self.trainer.logger.log_dir, 'prediction_'+loss_dict["prediction"]+'.npy'), pred)
                np.save(os.path.join(self.trainer.logger.log_dir,
                        'target_'+loss_dict["target"]+'.npy'), target)
                
                # plot
                if (pred.dtype == np.complex64) and (target.dtype == np.complex64):
                    lossname = type(loss_dict['metric']).__name__.split(".")[-1]
                    if lossname.lower() == 'abs_mae':
                        pred = np.absolute(pred)
                        target = np.absolute(target)
                    else:
                        pred = np.concatenate([pred.real, pred.imag], axis=-1)
                        target = np.concatenate([target.real, target.imag], axis=-1)
                
                # Control the number of scatter points to plot
                if pred.size > self.max_points_to_scatter:
                    random_state = np.random.RandomState(seed=42)
                    perm = list(random_state.permutation(np.arange(pred.size)))
                    pred = pred.reshape(-1)[perm[:self.max_points_to_scatter]]
                    target = target.reshape(-1)[perm[:self.max_points_to_scatter]]
                    
                figure = scatter_plot(pred.reshape(-1), target.reshape(-1))
                figname = 'PredVSTarget_' + loss_dict['prediction']
                self.logger.experiment.add_figure(
                    'test/'+figname, figure, global_step=self.global_step)
            else:
                pass
        
        if self.post_processing is not None:
            if type(self.post_processing).__name__.split(".")[-1].lower() == 'epc_output':
                processed_values = np.concatenate([out['processed_values']["epc_mat"]
                                        for out in test_step_outputs])
                np.save(os.path.join(
                    self.trainer.logger.log_dir, 'processed_values_'+'epc_mat'+'.npy'), processed_values)
            
    def forward(self, data):
        torch.set_grad_enabled(True)
        self._enable_grads(data)
        representation = self.representation(data)
        pred = self.output_module(data, representation)
        return pred

    def log_metrics(self, batch, result, mode):
        for metric_dict in self.metrics:
            loss_fn = metric_dict["metric"]

            if "target" in metric_dict.keys():
                pred = result[metric_dict["prediction"]]

                target = batch[metric_dict["target"]]

                loss_i = loss_fn(
                    pred, target
                ).detach().item()
            else:
                loss_i = loss_fn(
                    result[metric_dict["prediction"]]).detach().item()

            if hasattr(loss_fn, "name"):
                lossname = loss_fn.name
            else:
                lossname = type(loss_fn).__name__.split(".")[-1]

            self.log(
                mode
                + "/"
                + lossname
                + "_"
                + metric_dict["prediction"],
                loss_i,
                on_step=False,
                on_epoch=True,
            )

    def configure_optimizers(
            self,
    ):
        optimizer = opt.AdamW(self.parameters(), lr=self.lr, eps=self.epsilon, betas=(self.beta1, self.beta2), weight_decay=0.0, amsgrad=True)
        scheduler = {
            "scheduler": opt.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=self.lr_decay,
                patience=self.lr_patience,
                threshold=1e-6,
                cooldown=self.lr_patience // 2,
                min_lr=1e-6,
            ),
            "monitor": self.lr_monitor,
            "interval": "epoch",
            "frequency": 1,
            "strict": True,
        }
        return [optimizer], [scheduler]

    def _enable_grads(self, data):
        if self.requires_dr:
            data.pos.requires_grad_()

    def check_param(self):
        for name, parms in self.named_parameters():
            print('-->name:', name, '-->grad_requirs:', parms.requires_grad,
                  '-->grad_value:', parms.grad)
