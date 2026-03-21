"""
/*
 * @Author: Yang Zhong 
 * @Date: 2021-10-09 13:46:53 
 * @Last Modified by: Yang Zhong
 * @Last Modified time: 2021-10-29 21:09:02
 */
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import pytorch_lightning as pl
from typing import List, Dict, Union, Callable, Optional, Any

from ..utils.visualization import scatter_plot


class Model(pl.LightningModule):
    """
    A PyTorch Lightning module for scientific machine learning models.
    
    This class implements a modular architecture with representation and output components,
    handles training, validation, and testing with customizable losses and metrics, and
    supports gradient-based computations.
    
    Parameters
    ----------
    representation : nn.Module
        Neural network module that computes feature representations from input data
    output : nn.Module
        Neural network module that transforms representations into predictions
    losses : List[Dict]
        List of dictionaries defining loss functions, their targets, predictions, and weights
    validation_metrics : List[Dict]
        List of dictionaries defining metrics to track during validation
    lr : float, default=1e-3
        Initial learning rate for optimizer
    lr_decay : float, default=0.1
        Factor by which learning rate is reduced on plateau
    lr_patience : int, default=100
        Number of epochs with no improvement after which learning rate is reduced
    lr_monitor : str, default="training/total_loss"
        Metric to monitor for learning rate scheduling
    epsilon : float, default=1e-8
        Small constant for numerical stability in optimizer
    beta1 : float, default=0.99
        Exponential decay rate for first moment estimates in Adam optimizer
    beta2 : float, default=0.999
        Exponential decay rate for second moment estimates in Adam optimizer
    amsgrad : bool, default=True
        Whether to use AMSGrad variant of Adam optimizer
    max_points_to_scatter : int, default=100000
        Maximum number of points to include in scatter plots
    post_processing : callable, optional
        Function for calculating additional physical quantities that may require gradient backpropagation
    """

    def __init__(
            self,
            representation: nn.Module,
            output: nn.Module,
            losses: List[Dict],
            validation_metrics: List[Dict],
            lr: float = 1e-3,
            lr_decay: float = 0.1,
            lr_patience: int = 100,
            lr_monitor: str = "training/total_loss",
            epsilon: float = 1e-8,
            beta1: float = 0.99,
            beta2: float = 0.999,
            amsgrad: bool = True,
            max_points_to_scatter: int = 100000,
            post_processing: Optional[Callable] = None
    ):
        super().__init__()
        self.representation = representation
        self.output_module = output
        self.losses = losses
        self.metrics = validation_metrics
        
        # Optimizer parameters
        self.lr = lr
        self.lr_decay = lr_decay
        self.lr_patience = lr_patience
        self.lr_monitor = lr_monitor
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        self.amsgrad = amsgrad
        
        # Visualization parameters
        self.max_points_to_scatter = max_points_to_scatter
        
        # Post-processing for gradient-dependent physical quantities
        self.post_processing = post_processing
        
        # Track if derivatives are required
        self.requires_derivatives = self.output_module.derivative

    def _use_sync_dist(self) -> bool:
        """Return whether distributed metric synchronization is active."""
        return dist.is_available() and dist.is_initialized()

    def _is_global_zero(self) -> bool:
        """Return whether the current process is the global rank zero process."""
        return getattr(self.trainer, 'is_global_zero', True)

    def _gather_step_outputs(self, step_outputs: List[Dict]) -> List[Dict]:
        """Gather validation/test outputs from all distributed ranks."""
        if not self._use_sync_dist():
            return step_outputs

        gathered_outputs = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered_outputs, step_outputs)

        merged_outputs = []
        for rank_outputs in gathered_outputs:
            if rank_outputs is not None:
                merged_outputs.extend(rank_outputs)

        return merged_outputs

    def calculate_loss(self, batch: Dict[str, torch.Tensor], 
                       predictions: Dict[str, torch.Tensor], 
                       mode: str) -> torch.Tensor:
        """
        Calculate the total loss by summing weighted individual loss components.
        
        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Dictionary containing input data and target values
        predictions : Dict[str, torch.Tensor]
            Dictionary containing model predictions
        mode : str
            Current mode ('training', 'validation', or 'test')
            
        Returns
        -------
        torch.Tensor
            Total weighted loss
        """
        total_loss = torch.tensor(0.0, device=self.device)
        
        for loss_dict in self.losses:
            loss_fn = loss_dict["metric"]
            
            if "target" in loss_dict:
                prediction = predictions[loss_dict["prediction"]]
                target = batch[loss_dict["target"]]
                component_loss = loss_fn(prediction, target)
                
                # Apply sparsity correction if available and applicable
                if ('sparsity_ratio' in predictions and 
                    loss_dict["prediction"].lower() in ['hamiltonian', 'hamiltonian_real', 'hamiltonian_imag']):
                    sparsity_ratio = predictions['sparsity_ratio']
                    component_loss = component_loss * sparsity_ratio
            else:
                component_loss = loss_fn(predictions[loss_dict["prediction"]])
                
            # Weight and add the loss component
            total_loss += loss_dict["loss_weight"] * component_loss
            
            # Log the individual loss component
            loss_name = getattr(loss_fn, "name", type(loss_fn).__name__.split(".")[-1])
            self.log(
                f"{mode}/{loss_name}_{loss_dict['prediction']}",
                component_loss,
                on_step=False,
                on_epoch=True,
                sync_dist=self._use_sync_dist(),
            )
            
        return total_loss

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Perform a single training step with OOM protection.
        """
        try:
            self._enable_position_gradients(batch)
            predictions = self(batch)
            loss = self.calculate_loss(batch, predictions, 'training')
            self.log(
                "training/total_loss",
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                sync_dist=self._use_sync_dist(),
            )
            return loss
        except RuntimeError as e:
            if 'out of memory' in str(e):
                # Skip this batch — structure too large for GPU memory
                torch.cuda.empty_cache()
                n_atoms = len(batch.z) if hasattr(batch, 'z') else '?'
                n_edges = batch.edge_index.shape[1] if hasattr(batch, 'edge_index') else '?'
                print(f"[OOM] Skipping batch {batch_idx} (atoms={n_atoms}, edges={n_edges})")
                return None
            raise

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict:
        """
        Perform a single validation step.
        
        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Dictionary containing validation data
        batch_idx : int
            Index of the current batch
            
        Returns
        -------
        Dict
            Dictionary containing predictions and targets for logging
        """
        # Enable gradients if required for derivatives
        torch.set_grad_enabled(self.requires_derivatives)
        
        self._enable_position_gradients(batch)
        predictions = self(batch)
        
        val_loss = self.calculate_loss(batch, predictions, 'validation')
        self.log(
            "validation/total_loss",
            val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self._use_sync_dist(),
        )
        self.log_metrics(batch, predictions, 'validation')
        
        # Collect outputs for epoch-end processing
        outputs_pred, outputs_target = {}, {}
        for loss_dict in self.losses:
            if "target" in loss_dict:
                outputs_pred[loss_dict["prediction"]] = predictions[loss_dict["prediction"]].detach().cpu().numpy()  
                outputs_target[loss_dict["target"]] = batch[loss_dict["target"]].detach().cpu().numpy()
                
        return {'pred': outputs_pred, 'target': outputs_target}

    def on_validation_epoch_end(self) -> None:
        """
        Process and log validation results at the end of an epoch.
        PL 2.x: renamed from validation_epoch_end, no longer receives outputs as argument.
        """
        pass

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict:
        """
        Perform a single test step.
        
        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Dictionary containing test data
        batch_idx : int
            Index of the current batch
            
        Returns
        -------
        Dict
            Dictionary containing predictions, targets, and processed values
        """
        # Enable gradients if required for derivatives
        torch.set_grad_enabled(self.requires_derivatives)
        
        self._enable_position_gradients(batch)
        
        processed_values = None
        if self.post_processing is not None:
            predictions = self.post_processing(batch)
            post_processing_name = type(self.post_processing).__name__.split(".")[-1].lower()
            
            if post_processing_name == 'epc_output':
                processed_values = {'epc_mat': predictions['epc_mat'].detach().cpu().numpy()}
            else:
                raise NotImplementedError(f"Post-processing type {post_processing_name} not implemented")
        else:
            predictions = self(batch)
        
        test_loss = self.calculate_loss(batch, predictions, 'test')
        self.log(
            "test/total_loss",
            test_loss,
            on_step=False,
            on_epoch=True,
            sync_dist=self._use_sync_dist(),
        )
        self.log_metrics(batch, predictions, "test")
        
        # Collect outputs for epoch-end processing
        outputs_pred, outputs_target = {}, {}
        for loss_dict in self.losses:
            if "target" in loss_dict:
                outputs_pred[loss_dict["prediction"]] = predictions[loss_dict["prediction"]].detach().cpu().numpy()  
                outputs_target[loss_dict["target"]] = batch[loss_dict["target"]].detach().cpu().numpy()
                
        return {
            'pred': outputs_pred, 
            'target': outputs_target, 
            'processed_values': processed_values
        }

    def on_test_epoch_end(self) -> None:
        """
        Process and log test results at the end of testing.
        PL 2.x: renamed from test_epoch_end, no longer receives outputs as argument.
        """
        pass

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Dictionary containing input data
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing model predictions
        """
        self._enable_position_gradients(batch)
        representation = self.representation(batch)
        predictions = self.output_module(batch, representation)
        return predictions

    def log_metrics(self, batch: Dict[str, torch.Tensor], 
                   predictions: Dict[str, torch.Tensor], 
                   mode: str) -> None:
        """
        Log evaluation metrics for the current batch.
        
        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Dictionary containing input data and target values
        predictions : Dict[str, torch.Tensor]
            Dictionary containing model predictions
        mode : str
            Current mode ('validation' or 'test')
        """
        for metric_dict in self.metrics:
            metric_fn = metric_dict["metric"]
            
            if "target" in metric_dict:
                prediction = predictions[metric_dict["prediction"]]
                target = batch[metric_dict["target"]]
                metric_value = metric_fn(prediction, target).detach().item()
            else:
                metric_value = metric_fn(predictions[metric_dict["prediction"]]).detach().item()
                
            # Get metric name
            metric_name = getattr(metric_fn, "name", type(metric_fn).__name__.split(".")[-1])
            
            # Log the metric
            self.log(
                f"{mode}/{metric_name}_{metric_dict['prediction']}",
                metric_value,
                on_step=False,
                on_epoch=True,
                sync_dist=self._use_sync_dist(),
            )

    def configure_optimizers(self) -> Dict:
        """
        Configure optimizers and learning rate schedulers.
        
        Returns
        -------
        Dict
            Configuration dictionary for PyTorch Lightning
        """
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.lr,
            eps=self.epsilon,
            betas=(self.beta1, self.beta2),
            weight_decay=0.0,
            amsgrad=self.amsgrad
        )
        
        scheduler = {
            "scheduler": optim.lr_scheduler.ReduceLROnPlateau(
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

    def _enable_position_gradients(self, batch: Dict[str, torch.Tensor]) -> None:
        """
        Enable gradients for position vectors if derivatives are required.
        
        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            Dictionary containing input data including position vectors
        """
        if self.requires_derivatives and hasattr(batch, 'pos'):
            batch.pos.requires_grad_()

    def _prepare_data_for_scatter_plot(self, pred: np.ndarray, target: np.ndarray) -> tuple:
        """
        Prepare complex data for scatter plotting and handle subsampling.
        
        Parameters
        ----------
        pred : np.ndarray
            Array of prediction values
        target : np.ndarray
            Array of target values
            
        Returns
        -------
        tuple
            Processed prediction and target arrays ready for scatter plotting
        """
        # Handle complex data
        if (pred.dtype == np.complex64) and (target.dtype == np.complex64):
            # Check if we need absolute values or real/imag components
            for loss_dict in self.losses:
                if hasattr(loss_dict.get('metric', None), 'name'):
                    lossname = loss_dict['metric'].name
                elif loss_dict.get('metric', None) is not None:
                    lossname = type(loss_dict['metric']).__name__.split(".")[-1]
                else:
                    lossname = ""
                    
                if lossname.lower() == 'abs_mae':
                    pred = np.absolute(pred)
                    target = np.absolute(target)
                    break
            else:
                # Default handling for complex numbers
                pred = np.concatenate([pred.real, pred.imag], axis=-1)
                target = np.concatenate([target.real, target.imag], axis=-1)
        
        # Subsample if too many points
        if pred.size > self.max_points_to_scatter:
            random_state = np.random.RandomState(seed=42)
            perm = random_state.permutation(np.arange(pred.size))
            pred = pred.reshape(-1)[perm[:self.max_points_to_scatter]]
            target = target.reshape(-1)[perm[:self.max_points_to_scatter]]
            
        return pred.reshape(-1), target.reshape(-1)

    def _plot_prediction_vs_target(self, step_outputs: List[Dict], mode: str) -> None:
        """
        Create and log scatter plots comparing predictions to targets.
        
        Parameters
        ----------
        step_outputs : List[Dict]
            List of outputs from validation or test steps
        mode : str
            Current mode ('validation' or 'test')
        """
        for loss_dict in self.losses:
            if "target" in loss_dict:
                pred_key = loss_dict["prediction"]
                target_key = loss_dict["target"]
                
                # Skip if this prediction or target isn't in the outputs
                if not all(pred_key in out['pred'] and target_key in out['target'] for out in step_outputs):
                    continue
                
                # Concatenate predictions and targets from all batches
                pred = np.concatenate([out['pred'][pred_key] for out in step_outputs])
                target = np.concatenate([out['target'][target_key] for out in step_outputs])
                
                # Prepare data for plotting
                plot_pred, plot_target = self._prepare_data_for_scatter_plot(pred, target)
                
                # Create and log the scatter plot
                figure = scatter_plot(plot_pred, plot_target)
                figname = f'PredVSTarget_{pred_key}'
                self.logger.experiment.add_figure(
                    f'{mode}/{figname}', figure, global_step=self.global_step
                )

    def _save_predictions_and_targets(self, test_outputs: List[Dict], log_dir: str) -> None:
        """
        Save prediction and target arrays to disk.
        
        Parameters
        ----------
        test_outputs : List[Dict]
            List of outputs from test steps
        log_dir : str
            Directory to save the arrays
        """
        for loss_dict in self.losses:
            if "target" in loss_dict:
                pred_key = loss_dict["prediction"]
                target_key = loss_dict["target"]
                
                # Skip if this prediction or target isn't in the outputs
                if not all(pred_key in out['pred'] and target_key in out['target'] for out in test_outputs):
                    continue
                
                # Concatenate predictions and targets from all batches
                pred = np.concatenate([out['pred'][pred_key] for out in test_outputs])
                target = np.concatenate([out['target'][target_key] for out in test_outputs])
                
                # Save to disk
                np.save(os.path.join(log_dir, f'prediction_{pred_key}.npy'), pred)
                np.save(os.path.join(log_dir, f'target_{target_key}.npy'), target)
