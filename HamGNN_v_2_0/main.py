"""
/*
 * @Author: Yang Zhong 
 * @Date: 2021-10-12 23:42:11 
 * @Last Modified by: Yang Zhong
 * @Last Modified time: 2021-11-07 19:15:27
 */
 """
import torch
import torch.nn as nn
import numpy as np
import os
import e3nn
from e3nn import o3
from .GraphData.graph_data import graph_data_module
from .input.config_parsing import read_config
from .models.outputs import (Born, Born_node_vec, scalar, trivial_scalar, Force, 
                            Force_node_vec, crystal_tensor, piezoelectric, total_energy_and_atomic_forces, EPC_output)
import pytorch_lightning as pl
from .models.Model import Model
from .models.version import soft_logo
from pytorch_lightning.loggers import TensorBoardLogger
from .models.HamGNN.net import HamGNNTransformer, HamGNNConvE3, HamGNNPlusPlusOut
from torch.nn import functional as F
import pprint
import warnings
import sys
import socket
from .models.utils import get_hparam_dict
import argparse


def prepare_data(config):
    train_ratio = config.dataset_params.train_ratio
    val_ratio = config.dataset_params.val_ratio
    test_ratio = config.dataset_params.test_ratio
    batch_size = config.dataset_params.batch_size
    split_file = config.dataset_params.split_file
    graph_data_path = config.dataset_params.graph_data_path
    if not os.path.isfile(graph_data_path):
        if not os.path.exists(graph_data_path):
            os.mkdir(graph_data_path)
        graph_data_path = os.path.join(graph_data_path, 'graph_data.npz')
    if os.path.exists(graph_data_path):
        print(f"Loading graph data from {graph_data_path}!")
    else:
        print(f'The graph_data.npz file was not found in {graph_data_path}!')

    graph_data = np.load(graph_data_path, allow_pickle=True)
    graph_data = graph_data['graph'].item()
    graph_dataset = list(graph_data.values())

    graph_dataset = graph_data_module(graph_dataset, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio, 
                                        batch_size=batch_size, split_file=split_file)
    graph_dataset.setup(stage=config.setup.stage)

    return graph_dataset

def build_model(config):
    print("Building model")
    config.representation_nets.HamGNN_pre.radius_type = config.output_nets.HamGNN_out.ham_type.lower()
    if config.setup.GNN_Net.lower() in ['hamgnnconv', 'hamgnnpre', 'hamgnn_pre']:
        Gnn_net = HamGNNConvE3(config.representation_nets)
    elif config.setup.GNN_Net.lower() == 'hamgnntransformer':
        Gnn_net = HamGNNTransformer(config.representation_nets)
    else:
        print(f"The network: {config.setup.GNN_Net} is not yet supported!")
        quit()

    # second order tensor
    if config.setup.property.lower() in ['born', 'dielectric']:
        if config.setup.GNN_Net.lower() == 'cgcnn_edge':
            output_module = crystal_tensor(l_pred_atomwise_tensor=config.setup.l_pred_atomwise_tensor, include_triplet=Gnn_net.export_triplet, num_node_features=Gnn_net.atom_fea_len, num_edge_features=Gnn_net.nbr_fea_len, 
                                num_triplet_features=Gnn_net.triplet_feature_len, activation=Gnn_net.activation, use_bath_norm=True, bias=True, n_h=3, l_minus_mean=config.setup.l_minus_mean)
        elif config.setup.GNN_Net.lower() == 'edge_gnn':
            output_module = crystal_tensor(l_pred_atomwise_tensor=config.setup.l_pred_atomwise_tensor, include_triplet=False, num_node_features=Gnn_net.num_node_pooling_features, num_edge_features=Gnn_net.num_edge_pooling_features, num_triplet_features=Gnn_net.in_features_three_body,
                                           activation=nn.Softplus(), use_bath_norm=True, bias=True, n_h=3, l_minus_mean=config.setup.l_minus_mean)
        elif config.setup.GNN_Net.lower() == 'painn':
            #output_module = Born_node_vec(num_node_features=Gnn_net.num_scaler_out, activation=Gnn_net.activation, use_bath_norm=Gnn_net.use_batch_norm, bias=Gnn_net.lnode_bias,n_h=3)
            output_module = crystal_tensor(l_pred_atomwise_tensor=config.setup.l_pred_atomwise_tensor, include_triplet=Gnn_net.luse_triplet, num_node_features=Gnn_net.num_node_features, num_edge_features=Gnn_net.n_edge_features, 
                                            num_triplet_features=Gnn_net.triplet_feature_len, activation=Gnn_net.activation, l_minus_mean=config.setup.l_minus_mean)
        elif config.setup.GNN_Net.lower() == 'cgcnn_triplet':
            output_module = crystal_tensor(l_pred_atomwise_tensor=config.setup.l_pred_atomwise_tensor, include_triplet=True, num_node_features=Gnn_net.atom_fea_len, num_edge_features=Gnn_net.nbr_fea_len, 
                                           num_triplet_features=Gnn_net.triplet_feature_len, activation=Gnn_net.activation, use_bath_norm=True, bias=True, n_h=3, l_minus_mean=config.setup.l_minus_mean)
        elif config.setup.GNN_Net.lower() == 'dimenet_triplet':
            output_module = crystal_tensor(l_pred_atomwise_tensor=config.setup.l_pred_atomwise_tensor, include_triplet=Gnn_net.export_triplet, num_node_features=Gnn_net.num_node_features, num_edge_features=Gnn_net.hidden_channels, 
                                           num_triplet_features=Gnn_net.num_triplet_features, activation=Gnn_net.act, use_bath_norm=True, bias=True, n_h=3, cutoff_triplet=config.representation_nets.dimenet_triplet.cutoff_triplet, l_minus_mean=config.setup.l_minus_mean)
        else:
            quit()

    #Force
    elif config.setup.property.lower() == 'force':
        if config.setup.GNN_Net.lower() == 'dimenet_triplet':
            output_module = Force(num_edge_features=Gnn_net.hidden_channels, activation=Gnn_net.act, use_bath_norm=True, bias=True, n_h=3)
        else:
            quit()

    #piezoelectric
    elif config.setup.property.lower() == 'piezoelectric':
        if config.setup.GNN_Net.lower() == 'dimenet_triplet':
            output_module = piezoelectric(include_triplet=Gnn_net.export_triplet, num_node_features=Gnn_net.num_node_features, num_edge_features=Gnn_net.hidden_channels,
                                          num_triplet_features=Gnn_net.num_triplet_features, activation=Gnn_net.act, use_bath_norm=True, bias=True, n_h=3, cutoff_triplet=config.representation_nets.dimenet_triplet.cutoff_triplet)
        else:
            quit()
            
    # scalar_per_atom
    elif config.setup.property.lower() == 'scalar_per_atom':
        if config.setup.GNN_Net.lower() == 'dimnet':
            output_module = trivial_scalar('mean')
        elif config.setup.GNN_Net.lower() == 'edge_gnn':
            output_module = scalar('mean', False, num_node_features=Gnn_net.num_node_features, n_h=2)
        elif config.setup.GNN_Net.lower() == 'schnet':
            output_module = trivial_scalar('mean')
        elif config.setup.GNN_Net.lower() == 'cgcnn':
            output_module = scalar('mean', Gnn_net.classification, num_node_features=Gnn_net.atom_fea_len, n_h=config.representation_nets.cgcnn.n_h)
        elif config.setup.GNN_Net.lower() == 'cgcnn_edge':
            output_module = scalar('mean', Gnn_net.classification,
                                   num_node_features=Gnn_net.atom_fea_len, n_h=config.representation_nets.cgcnn_edge.n_h)
        elif config.setup.GNN_Net.lower() == 'cgcnn_triplet':
            output_module = scalar('mean', Gnn_net.classification, num_node_features=Gnn_net.atom_fea_len,
                                   n_h=config.representation_nets.cgcnn_triplet.n_h)
        elif config.setup.GNN_Net.lower() == 'painn':
            output_module = trivial_scalar('mean')
        elif config.setup.GNN_Net.lower() == 'dimenet_triplet':
            output_module = scalar('mean', False, num_node_features=Gnn_net.num_node_features, n_h=3, activation=Gnn_net.act)
        else:
            quit()
    
    # scalar_max
    elif config.setup.property.lower() == 'scalar_max':
        if config.setup.GNN_Net.lower() == 'dimnet':
            output_module = trivial_scalar('max')
        elif config.setup.GNN_Net.lower() == 'edge_gnn':
            output_module = scalar(
                'max', False, num_node_features=Gnn_net.num_node_features, n_h=2)
        elif config.setup.GNN_Net.lower() == 'schnet':
            output_module = trivial_scalar('max')
        elif config.setup.GNN_Net.lower() == 'cgcnn':
            output_module = scalar('max', Gnn_net.classification,
                                   num_node_features=Gnn_net.atom_fea_len, n_h=config.representation_nets.cgcnn.n_h)
        elif config.setup.GNN_Net.lower() == 'cgcnn_edge':
            output_module = scalar('max', Gnn_net.classification,
                                   num_node_features=Gnn_net.atom_fea_len, n_h=config.representation_nets.cgcnn_edge.n_h)
        elif config.setup.GNN_Net.lower() == 'cgcnn_triplet':
            output_module = scalar('max', Gnn_net.classification,
                                   num_node_features=Gnn_net.atom_fea_len, n_h=config.representation_nets.cgcnn_triplet.n_h)
        elif config.setup.GNN_Net.lower() == 'painn':
            output_module = trivial_scalar('max')
        elif config.setup.GNN_Net.lower() == 'dimenet_triplet':
            output_module = scalar(
                'max', False, num_node_features=Gnn_net.num_node_features, n_h=3, activation=Gnn_net.act)
        else:
            quit()
    
    # scalar
    elif config.setup.property.lower() == 'scalar':
        if config.setup.GNN_Net.lower() == 'dimnet':
            output_module = trivial_scalar('sum')
        elif config.setup.GNN_Net.lower() == 'edge_gnn':
            output_module = trivial_scalar('sum')
        elif config.setup.GNN_Net.lower() == 'schnet':
            output_module = trivial_scalar('sum')
        elif config.setup.GNN_Net.lower() == 'cgcnn':
            output_module = scalar('sum', Gnn_net.classification, num_node_features=Gnn_net.atom_fea_len, n_h=2)
        elif config.setup.GNN_Net.lower() == 'cgcnn_edge':
            output_module = scalar('sum', Gnn_net.classification, num_node_features=Gnn_net.atom_fea_len,
                                   n_h=config.representation_nets.cgcnn_edge.n_h)
        elif config.setup.GNN_Net.lower() == 'cgcnn_triplet':
            output_module = scalar('sum', Gnn_net.classification, num_node_features=Gnn_net.atom_fea_len,
                                   n_h=config.representation_nets.cgcnn_triplet.n_h)
        elif config.setup.GNN_Net.lower() == 'painn':
            output_module = trivial_scalar('sum')
        elif config.setup.GNN_Net.lower() == 'dimenet_triplet':
            output_module = scalar('sum', False, num_node_features=Gnn_net.num_node_features, n_h=3, activation=Gnn_net.act)
        else:
            quit()
        
    # Hamiltonian
    elif config.setup.property.lower() == 'hamiltonian':
        output_params = config.output_nets.HamGNN_out
        output_module = HamGNNPlusPlusOut(irreps_in_node = Gnn_net.irreps_node_features, irreps_in_edge = Gnn_net.irreps_node_features, nao_max= output_params.nao_max, ham_type= output_params.ham_type,
                                         ham_only= output_params.ham_only, symmetrize=output_params.symmetrize,calculate_band_energy=output_params.calculate_band_energy,num_k=output_params.num_k,k_path=output_params.k_path,
                                         band_num_control=output_params.band_num_control, soc_switch=output_params.soc_switch, nonlinearity_type = output_params.nonlinearity_type, add_H0=output_params.add_H0, 
                                         spin_constrained=output_params.spin_constrained, collinear_spin=output_params.collinear_spin, minMagneticMoment=output_params.minMagneticMoment)

    else:
        print('Evaluation of this property is not supported!')
        quit()
    
    # Initialize post_utility
    post_utility = None
    
    return Gnn_net, output_module, post_utility

def train_and_eval(config):
    data = prepare_data(config)

    graph_representation, output_module, post_utility = build_model(config)
    graph_representation.to(torch.float32)
    output_module.to(torch.float32)

    # define metrics
    losses = config.losses_metrics.losses
    metrics = config.losses_metrics.metrics
    
    # Training
    if config.setup.stage == 'fit':
        # laod network weights
        if config.setup.load_from_checkpoint and not config.setup.resume:
            model = Model.load_from_checkpoint(checkpoint_path=config.setup.checkpoint_path,
            representation=graph_representation,
            output=output_module,
            post_processing=post_utility,
            losses=losses,
            validation_metrics=metrics,
            lr=config.optim_params.lr,
            lr_decay=config.optim_params.lr_decay,
            lr_patience=config.optim_params.lr_patience
            )   
        else:            
            model = Model(
            representation=graph_representation,
            output=output_module,
            post_processing=post_utility,
            losses=losses,
            validation_metrics=metrics,
            lr=config.optim_params.lr,
            lr_decay=config.optim_params.lr_decay,
            lr_patience=config.optim_params.lr_patience,
            )

        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("The model you built has %d parameters." % params)

        callbacks = [
            pl.callbacks.LearningRateMonitor(),
            pl.callbacks.EarlyStopping(
                monitor="training/total_loss",
                patience=config.optim_params.stop_patience, min_delta=1e-6,
            ),
            pl.callbacks.ModelCheckpoint(
                filename="{epoch}-{val_loss:.6f}",
                save_top_k=1,
                verbose=False,
                monitor='validation/total_loss',
                mode='min',
            )
        ]

        tb_logger = TensorBoardLogger(
            save_dir=config.profiler_params.train_dir, name="", default_hp_metric=False)    

        trainer = pl.Trainer(
            gpus=config.setup.num_gpus,
            precision=config.setup.precision,
            callbacks=callbacks,
            progress_bar_refresh_rate=1,
            logger=tb_logger,
            gradient_clip_val = config.optim_params.gradient_clip_val,
            max_epochs=config.optim_params.max_epochs,
            default_root_dir=config.profiler_params.train_dir,
            min_epochs=config.optim_params.min_epochs,
            resume_from_checkpoint = config.setup.checkpoint_path if config.setup.resume else None
        )

        print("Start training.")
        trainer.fit(model, data)
        print("Training done.")

        # Eval
        print("Start eval.")
        results = trainer.test(model, data.test_dataloader())
        # log hyper-parameters in tensorboard.
        hparam_dict = get_hparam_dict(config)
        metric_dict = dict() 
        for result_dict in results:
            metric_dict.update(result_dict)
        trainer.logger.experiment.add_hparams(hparam_dict, metric_dict)
        print("Eval done.")
    
    # Prediction
    if config.setup.stage == 'test': 
        model = Model.load_from_checkpoint(checkpoint_path=config.setup.checkpoint_path,
            representation=graph_representation,
            output=output_module,
            post_processing=post_utility,
            losses=losses,
            validation_metrics=metrics,
            lr=config.optim_params.lr,
            lr_decay=config.optim_params.lr_decay,
            lr_patience=config.optim_params.lr_patience
            ) 
        tb_logger = TensorBoardLogger(
            save_dir=config.profiler_params.train_dir, name="", default_hp_metric=False)

        trainer = pl.Trainer(gpus=config.setup.num_gpus, precision=config.setup.precision, logger=tb_logger)
        trainer.test(model=model, datamodule=data)

def HamGNN():
    #torch.autograd.set_detect_anomaly(True)
    pl.utilities.seed.seed_everything(666)
    print(soft_logo)
    parser = argparse.ArgumentParser(description='Deep Hamiltonian')
    parser.add_argument('--config', default='config.yaml', type=str, metavar='N')
    args = parser.parse_args()

    configure = read_config(config_file_name=args.config)
    hostname = socket.getfqdn(socket.gethostname())
    configure.setup.hostname = hostname
    pprint.pprint(configure)
    if configure.setup.ignore_warnings:
        warnings.filterwarnings('ignore')
    
    train_and_eval(configure)

if __name__ == '__main__':
    HamGNN()
