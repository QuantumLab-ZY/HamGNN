"""
/*
 * @Author: Yang Zhong 
 * @Date: 2021-10-29 15:03:46 
 * @Last Modified by: Yang Zhong
 * @Last Modified time: 2021-10-29 16:45:04
 */
"""
import yaml
from easydict import EasyDict
from ..models.utils import get_activation, parse_metric_func
import pprint

"""
Default configuration parameters.
"""
config_default = dict()

"""The parameters for setup"""
config_default_setup = dict()
config_default_setup['GNN_Net'] = 'Edge_GNN'
config_default_setup['property'] = 'scalar_per_atom'
config_default_setup['num_gpus'] = [1]
config_default_setup['accelerator'] = None # 'dp' 'ddp' 'ddp_cpu'
config_default_setup['precision'] = 32
config_default_setup['stage'] = 'fit'
config_default_setup['resume'] = False
config_default_setup['load_from_checkpoint'] = False
config_default_setup['checkpoint_path'] = './'
config_default_setup['ignore_warnings'] = False
config_default_setup['l_minus_mean'] = False
config_default['setup'] = config_default_setup

"""The parameters for dataset."""
config_default_dataset = dict()
config_default_dataset['database_type'] = 'db' # 'db' or 'csv'
config_default_dataset['train_ratio'] = 0.6
config_default_dataset['val_ratio'] = 0.2
config_default_dataset['test_ratio'] = 0.2
config_default_dataset['batch_size'] = 200
config_default_dataset['split_file'] = None
config_default_dataset['radius'] = 6.0
config_default_dataset['max_num_nbr'] = 32
config_default_dataset['graph_data_path'] = './graph_data'

config_default_db_params = dict()
config_default_db_params['db_path'] = './'
config_default_db_params['property_list'] = ['energy','hamiltonian']
config_default_dataset['db_params'] = config_default_db_params

config_default_csv_params = dict()
config_default_csv_params['crystal_path'] = 'crystals'
config_default_csv_params['file_type'] = 'poscar'
config_default_csv_params['id_prop_path'] = './'
config_default_csv_params['rank_tensor'] = 0
config_default_csv_params['l_pred_atomwise_tensor'] = True
config_default_csv_params['l_pred_crystal_tensor'] = False
config_default_dataset['csv_params'] = config_default_csv_params

config_default['dataset_params'] = config_default_dataset

"""The parameters for optimizer."""
config_default_optimizer = dict()
config_default_optimizer['lr'] = 0.01
config_default_optimizer['lr_decay'] = 0.5
config_default_optimizer['lr_patience'] = 5
config_default_optimizer['gradient_clip_val'] = 0.0
config_default_optimizer['stop_patience'] = 30
config_default_optimizer['min_epochs'] = 100
config_default_optimizer['max_epochs'] = 500
config_default['optim_params'] = config_default_optimizer

"""The parameters for losses_metrics."""
config_default_metric = dict()
config_default_metric['losses'] = [{'metric': 'mse', 'prediction': 'energy',  'target': 'energy', 'loss_weight': 1.0}, {
    'metric': 'cosine_similarity', 'prediction': 'energy',  'target': 'energy', 'loss_weight': 0.0}]
config_default_metric['metrics'] = [{'metric': 'mae', 'prediction': 'energy',  'target': 'energy'}, {
    'metric': 'cosine_similarity', 'prediction': 'energy',  'target': 'energy'}]
config_default['losses_metrics'] = config_default_metric

"""The parameters for profiler"""
config_default_profiler = dict()
config_default_profiler['train_dir'] = 'train_data'
config_default_profiler['progress_bar_refresh_rat'] = 1
config_default['profiler_params'] = config_default_profiler

"""The parameters for representation_nets"""
config_default_representation_nets = dict()

config_default['representation_nets'] = config_default_representation_nets

"""The parameters for output_nets"""
config_default_output_nets = dict()
config_default_output_nets['output_module'] = 'HamGNN_out'

"""The parameters for HamGNN_out"""
config_default_HamGNN_out = dict()
config_default_HamGNN_out['nao_max'] = 14
config_default_HamGNN_out['return_forces'] = False
config_default_HamGNN_out['create_graph'] = False
config_default_HamGNN_out['ham_type'] = 'openmx'
config_default_HamGNN_out['ham_only'] = True
config_default_HamGNN_out['irreps_in_node'] = ''
config_default_HamGNN_out['irreps_in_edge'] = ''
config_default_HamGNN_out['irreps_in_triplet'] = ''
config_default_HamGNN_out['include_triplet'] = False
config_default_HamGNN_out['symmetrize'] = True
config_default_HamGNN_out['calculate_band_energy'] = False
config_default_HamGNN_out['num_k'] = 5
config_default_HamGNN_out['soc_switch'] = False 
config_default_HamGNN_out['nonlinearity_type'] = 'gate'
config_default_HamGNN_out['band_num_control'] = 6
config_default_HamGNN_out['k_path'] = None
config_default_HamGNN_out['spin_constrained'] = False
config_default_HamGNN_out['collinear_spin'] = False
config_default_HamGNN_out['minMagneticMoment'] = 0.5
config_default_output_nets['HamGNN_out'] = config_default_HamGNN_out

config_default['output_nets'] = config_default_output_nets


def read_config(config_file_name: str = 'config_default.yaml', config_default=config_default):
    with open(config_file_name, encoding='utf-8') as rstream:
        data = yaml.load(rstream, yaml.SafeLoader)
    for key in data.keys():
        config_default[key].update(data[key])
    config = EasyDict(config_default)
    config.losses_metrics.losses = parse_metric_func(config.losses_metrics.losses)
    config.losses_metrics.metrics = parse_metric_func(config.losses_metrics.metrics)
    return config
