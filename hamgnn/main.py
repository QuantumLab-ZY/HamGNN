"""
/*
 * @Author: Yang Zhong 
 * @Date: 2021-10-12 23:42:11 
 * @Last Modified by: Yang Zhong
 * @Last Modified time: 2021-11-07 19:15:27
 */
 """
import os
import socket
import argparse
from datetime import datetime
import warnings
import yaml

import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import pprint

from .data.graph_data import graph_data_module
from .config.config_parsing import load_config
from .models.Model import Model
from .version import get_version, get_full_version_info, soft_logo
from .models.hamgnn_transformer import HamGNNTransformer 
from .models.hamgnn_conv import HamGNNConvE3
from .models.hamgnn_output import HamGNNPlusPlusOut
from .utils.hparam import get_hparam_dict


def initialize_output_parameters(output_params):
    """
    Initialize default values for output parameters if they don't already exist.
    
    Parameters
    ----------
    output_params : object
        Object to store output configuration parameters. Can be a SimpleNamespace,
        custom class instance, or any object supporting attribute access.
    
    Returns
    -------
    object
        The same output_params object with default values set where needed.
    """
    # Define default parameter values
    default_params = {
        'add_H_nonsoc': False,          # Add non-spin-orbit coupling Hamiltonian
        'get_nonzero_mask_tensor': False, # Generate mask for non-zero elements
        'zero_point_shift': True,       # Apply zero-point energy shift
        'soc_basis': 'so3',           # Spin-orbit coupling basis
    }
    
    # Set default values for parameters not already defined
    for param_name, default_value in default_params.items():
        if not hasattr(output_params, param_name):
            setattr(output_params, param_name, default_value)
    
    return output_params


def prepare_dataset(config):
    """
    Prepare the graph dataset for training, validation, and testing.
    
    This function initializes the data module with appropriate splitting ratios,
    batch size, and other parameters from the configuration.
    
    Parameters
    ----------
    config : object
        Configuration object containing dataset parameters.
        Expected attributes:
        - dataset_params: Contains dataset-specific configuration
        - setup: Contains training setup configuration
    
    Returns
    -------
    graph_data_module
        Initialized dataset module ready for training/evaluation.
    """
    # Extract dataset parameters from config
    train_ratio = config.dataset_params.train_ratio
    val_ratio = config.dataset_params.val_ratio
    test_ratio = config.dataset_params.test_ratio
    batch_size = config.dataset_params.batch_size
    split_file = config.dataset_params.split_file
    graph_data_path = config.dataset_params.graph_data_path
    
    # Check if graph_data_path is a file, if not append 'graph_data.npz'
    if not os.path.isfile(graph_data_path) and not graph_data_path.lower().endswith(".lmdb"):
        graph_data_path = os.path.join(graph_data_path, 'graph_data.npz')
    
    # Get optional parameters with defaults
    num_workers = getattr(config.dataset_params, 'num_workers', 4)
    preload = getattr(config.dataset_params, 'preload', 0)
    data_format = getattr(config.dataset_params, 'data_format', 'auto')
    is_test_mode = (config.setup.stage == 'test')
    
    # Initialize the graph dataset module
    graph_dataset = graph_data_module(
        dataset=graph_data_path, 
        train_ratio=train_ratio, 
        val_ratio=val_ratio, 
        test_ratio=test_ratio, 
        batch_size=batch_size, 
        split_file=split_file,
        num_workers=num_workers,
        preload=preload,
        test_mode=is_test_mode,
        data_format=data_format
    )
    
    return graph_dataset


def build_hamgnn_model(config):
    """
    Build the HamGNN model components based on configuration.
    
    This function creates the graph neural network representation and output
    modules according to the specified configuration.
    
    Parameters
    ----------
    config : object
        Configuration object containing model parameters.
        Expected attributes:
        - representation_nets: Neural network representation configuration
        - output_nets: Output module configuration
        - setup: General setup parameters
    
    Returns
    -------
    tuple
        A tuple containing:
        - graph_representation: The GNN representation module
        - output_module: The output module for Hamiltonian prediction
        - post_processing_utility: Post-processing utility (None for Hamiltonian predictions)
    
    Raises
    ------
    SystemExit
        If an unsupported network type or property is specified.
    """
    print("Building model")
    
    # Set radius type for Hamiltonian calculation
    config.representation_nets.HamGNN_pre.radius_type = config.output_nets.HamGNN_out.ham_type.lower()
    
    # Build the graph neural network based on configuration
    gnn_net_type = config.setup.GNN_Net.lower()
    if gnn_net_type in ['hamgnnconv', 'hamgnnpre', 'hamgnn_pre']:
        # Set default parameter if missing
        if 'use_corr_prod' not in config.representation_nets.HamGNN_pre:
            config.representation_nets.HamGNN_pre.use_corr_prod = True
        graph_representation = HamGNNConvE3(config.representation_nets)
    elif gnn_net_type == 'hamgnntransformer':
        graph_representation = HamGNNTransformer(config.representation_nets)
    else:
        print(f"The network: {config.setup.GNN_Net} is not yet supported!")
        raise SystemExit(1)
    
    # Configure output module based on property type
    property_type = config.setup.property.lower()
    if property_type == 'hamiltonian':
        output_params = config.output_nets.HamGNN_out
        
        # Initialize default parameters if not provided
        output_params = initialize_output_parameters(output_params)
        
        # Create output module for Hamiltonian prediction
        output_module = HamGNNPlusPlusOut(
            irreps_in_node=graph_representation.irreps_node_features, 
            irreps_in_edge=graph_representation.irreps_node_features, 
            nao_max=output_params.nao_max, 
            ham_type=output_params.ham_type,
            ham_only=output_params.ham_only, 
            symmetrize=output_params.symmetrize,
            calculate_band_energy=output_params.calculate_band_energy,
            num_k=output_params.num_k,
            k_path=output_params.k_path,
            band_num_control=output_params.band_num_control, 
            soc_switch=output_params.soc_switch, 
            soc_basis=output_params.soc_basis,
            nonlinearity_type=output_params.nonlinearity_type, 
            add_H0=output_params.add_H0, 
            spin_constrained=output_params.spin_constrained, 
            collinear_spin=output_params.collinear_spin, 
            minMagneticMoment=output_params.minMagneticMoment, 
            add_H_nonsoc=output_params.add_H_nonsoc,
            get_nonzero_mask_tensor=output_params.get_nonzero_mask_tensor, 
            zero_point_shift=output_params.zero_point_shift,
        )
    else:
        print(f'Property type "{property_type}" is not supported!')
        raise SystemExit(1)
    
    # No post-processing utility needed for Hamiltonian predictions
    post_processing_utility = None
    
    return graph_representation, output_module, post_processing_utility


def setup_trainer(config, callbacks):
    """
    Set up PyTorch Lightning trainer based on configuration.
    
    Parameters
    ----------
    config : object
        Configuration object containing trainer parameters.
        Expected attributes:
        - setup: Training setup configuration including hardware settings
        - optim_params: Optimization parameters including epochs and gradient clipping
        - profiler_params: Configuration for logging and checkpoints
    
    callbacks : list
        List of PyTorch Lightning callbacks for the trainer
    
    Returns
    -------
    tuple
        A tuple containing:
        - trainer: Configured PyTorch Lightning trainer
        - tb_logger: TensorBoard logger for the trainer
    """
    # Set up TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir=config.profiler_params.train_dir, 
        name="", 
        default_hp_metric=False
    )
    
    # Configure trainer with parameters from config
    trainer_params = {
        'gpus': config.setup.num_gpus, 
        'precision': config.setup.precision,
        'callbacks': callbacks,
        'progress_bar_refresh_rate': 1,
        'logger': tb_logger,
        'gradient_clip_val': config.optim_params.gradient_clip_val,
        'max_epochs': config.optim_params.max_epochs,
        'default_root_dir': config.profiler_params.train_dir,
        'min_epochs': config.optim_params.min_epochs,
    }
    
    # Add checkpoint path if resuming training
    if config.setup.resume and config.setup.checkpoint_path:
        trainer_params['resume_from_checkpoint'] = config.setup.checkpoint_path
    
    # Create the trainer with the configured parameters
    trainer = pl.Trainer(**trainer_params)
    
    return trainer, tb_logger


def load_or_create_model(config, graph_representation, output_module, post_processing_utility, losses, metrics):
    """
    Load an existing model from checkpoint or create a new one.
    
    Parameters
    ----------
    config : object
        Configuration object containing model parameters
    graph_representation : nn.Module
        Graph neural network for representation learning
    output_module : nn.Module
        Output module for predicting Hamiltonian matrices
    post_processing_utility : object or None
        Post-processing utility
    losses : dict
        Dictionary of loss functions
    metrics : dict
        Dictionary of evaluation metrics
    
    Returns
    -------
    Model
        Loaded or newly created model
    """
    # Set model parameters from config
    model_params = {
        'representation': graph_representation,
        'output': output_module,
        'post_processing': post_processing_utility,
        'losses': losses,
        'validation_metrics': metrics,
        'lr': config.optim_params.lr,
        'lr_decay': config.optim_params.lr_decay,
        'lr_patience': config.optim_params.lr_patience
    }
    
    # Load from checkpoint or create new model
    is_load_checkpoint = config.setup.load_from_checkpoint and not config.setup.resume
    if is_load_checkpoint:
        model = Model.load_from_checkpoint(
            checkpoint_path=config.setup.checkpoint_path,
            **model_params
        )
    else:
        model = Model(**model_params)
    
    # Print model size if master process
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params_count = sum([np.prod(p.size()) for p in model_parameters])
    print(f"The model you built has {params_count:,} parameters.")
    
    return model


def train_model(trainer, model, data_module):
    """
    Train the model using the configured trainer.
    
    Parameters
    ----------
    trainer : pl.Trainer
        PyTorch Lightning trainer
    model : Model
        Model to train
    data_module : graph_data_module
        Data module containing training data
    
    Returns
    -------
    list
        Test results after training
    """
    print("Starting training...")
    
    # Train the model
    trainer.fit(model, data_module)
    
    print("Training completed.")
    print("Starting evaluation...")
    
    # Evaluate the model
    test_results = trainer.test(model, data_module)
    
    print("Evaluation completed.")
    
    return test_results


def test_model(trainer, model, data_module):
    """
    Test the model using the configured trainer.
    
    Parameters
    ----------
    trainer : pl.Trainer
        PyTorch Lightning trainer
    model : Model
        Model to test
    data_module : graph_data_module
        Data module containing test data
    """
    print("Starting model testing...")
    
    # Test the model
    trainer.test(model=model, datamodule=data_module)
    
    print("Testing completed.")


def train_and_evaluate(config):
    """
    Train and evaluate the HamGNN model based on configuration.
    
    This function orchestrates the entire training and evaluation process,
    including data preparation, model building, training, and testing.
    
    Parameters
    ----------
    config : object
        Configuration object containing all parameters for training and evaluation.
        Expected attributes:
        - setup: Training setup configuration
        - dataset_params: Dataset parameters
        - optim_params: Optimization parameters
        - losses_metrics: Loss functions and metrics
        - profiler_params: Profiling and logging parameters
    """
    # Prepare dataset
    data_module = prepare_dataset(config)
    
    # Build model components
    graph_representation, output_module, post_processing_utility = build_hamgnn_model(config)
    
    # Set precision (data type)
    dtype = torch.float32 if config.setup.precision == 32 else torch.float64
    torch.set_default_dtype(dtype)
    
    # Convert model components to correct precision
    graph_representation.to(dtype)
    output_module.to(dtype)
    
    # Get losses and metrics from config
    losses = config.losses_metrics.losses
    metrics = config.losses_metrics.metrics
    
    # Setup callbacks
    callbacks = [
        pl.callbacks.LearningRateMonitor(),
        pl.callbacks.EarlyStopping(
            monitor="training/total_loss",
            patience=config.optim_params.stop_patience, 
            min_delta=1e-6,
        ),
        pl.callbacks.ModelCheckpoint(
            filename="{epoch}-{val_loss:.6f}",
            save_top_k=1,
            verbose=False,
            monitor='validation/total_loss',
            mode='min',
        )
    ]
    
    # Training stage
    if config.setup.stage == 'fit':
        # Load or create model
        model = load_or_create_model(
            config, graph_representation, output_module, 
            post_processing_utility, losses, metrics
        )
        
        # Setup trainer
        trainer, tb_logger = setup_trainer(config, callbacks)
        
        # Log version info to TensorBoard
        version_info = get_full_version_info()
        for key, value in version_info.items():
            tb_logger.experiment.add_text(f"version/{key}", str(value), global_step=0)
        
        # Train and evaluate model
        test_results = train_model(trainer, model, data_module)
        
        # Log hyperparameters in tensorboard
        hparam_dict = get_hparam_dict(config)
        hparam_dict['version'] = get_version()  # Add version to hyperparameters
        metric_dict = {}
        for result_dict in test_results:
            metric_dict.update(result_dict)
        tb_logger.experiment.add_hparams(hparam_dict, metric_dict)
    
    # Testing/prediction stage
    elif config.setup.stage == 'test':
        # Load model from checkpoint
        model = Model.load_from_checkpoint(
            checkpoint_path=config.setup.checkpoint_path,
            representation=graph_representation,
            output=output_module,
            post_processing=post_processing_utility,
            losses=losses,
            validation_metrics=metrics,
            lr=config.optim_params.lr,
            lr_decay=config.optim_params.lr_decay,
            lr_patience=config.optim_params.lr_patience
        )
        
        # Setup trainer for testing (with minimal callbacks)
        trainer, _ = setup_trainer(config, callbacks=None)
        
        # Test model
        test_model(trainer, model, data_module)


def HamGNN():
    #torch.autograd.set_detect_anomaly(True)
    pl.utilities.seed.seed_everything(666)
    
    # Print version info on master process
    print(soft_logo)
    version_info = get_full_version_info()
    print(f"Build timestamp: {version_info['timestamp']}")
    if version_info['is_dirty']:
        print("WARNING: This version was built with uncommitted changes")
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Hamiltonian Graph Neural Network')
    parser.add_argument('--config', default='config.yaml', type=str, metavar='N')
    args = parser.parse_args()

    # Read configuration file
    config = load_config(config_file_path=args.config)
    hostname = socket.getfqdn(socket.gethostname())
    config.setup.hostname = hostname
    
    # Print configuration information on master process
    pprint.pprint(config)
    
    # Ignore warnings if specified  
    if config.setup.ignore_warnings:
        warnings.filterwarnings('ignore')
    
    train_and_evaluate(config)

if __name__ == '__main__':
    HamGNN()

