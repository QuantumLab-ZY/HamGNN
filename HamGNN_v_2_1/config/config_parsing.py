"""
/*
 * @Author: Yang Zhong 
 * @Date: 2021-10-29 15:03:46 
 * @Last Modified by: Yang Zhong
 * @Last Modified time: 2021-10-29 16:45:04
 */
"""
import yaml
import argparse
import copy
from typing import Dict, Any, Optional, Union
from easydict import EasyDict

from ..utils.losses import parse_metric_func

"""
Default configuration parameters.
"""
config_default = dict()

"""The parameters for setup"""
config_default_setup = dict()
config_default_setup['GNN_Net'] = 'HamGNNpre'
config_default_setup['ignore_warnings'] = True
config_default_setup['checkpoint_path'] = './'
config_default_setup['load_from_checkpoint'] = False
config_default_setup['resume'] = False
config_default_setup['num_gpus'] = 1
config_default_setup['hostname'] = 'host'
config_default_setup['job_id'] = 'time_2025'
config_default_setup['precision'] = 32
config_default_setup['property'] = 'hamiltonian'
config_default_setup['stage'] = 'fit'
config_default['setup'] = config_default_setup

"""The parameters for profiler"""
config_default_profiler = dict()
config_default_profiler['train_dir'] = './'
config_default_profiler['progress_bar_refresh_rat'] = 1
config_default['profiler_params'] = config_default_profiler

"""The parameters for representation_nets"""
config_default_representation_nets = dict()
config_default_representation_nets['HamGNN_pre'] = dict()
config_default_representation_nets['HamGNN_pre']['cutoff'] = 26.0
config_default_representation_nets['HamGNN_pre']['cutoff_func'] = 'cos'
config_default_representation_nets['HamGNN_pre']['radius_type'] = 'openmx'
config_default_representation_nets['HamGNN_pre']['edge_sh_normalization'] = 'component'
config_default_representation_nets['HamGNN_pre']['edge_sh_normalize'] = True
config_default_representation_nets['HamGNN_pre']['irreps_edge_sh'] = '0e + 1o + 2e + 3o + 4e + 5o'
config_default_representation_nets['HamGNN_pre']['irreps_node_features'] = '64x0e+64x0o+32x1o+16x1e+12x2o+25x2e+18x3o+9x3e+4x4o+9x4e+4x5o+4x5e+2x6e'
config_default_representation_nets['HamGNN_pre']['num_layers'] = 3
config_default_representation_nets['HamGNN_pre']['num_radial'] = 64
config_default_representation_nets['HamGNN_pre']['num_types'] = 96
config_default_representation_nets['HamGNN_pre']['rbf_func'] = 'bessel'
config_default_representation_nets['HamGNN_pre']['set_features'] = True
config_default_representation_nets['HamGNN_pre']['radial_MLP'] = [64, 64]
config_default_representation_nets['HamGNN_pre']['use_corr_prod'] = False
config_default_representation_nets['HamGNN_pre']['correlation'] = 2
config_default_representation_nets['HamGNN_pre']['num_hidden_features'] = 16
config_default_representation_nets['HamGNN_pre']['use_kan'] = False
config_default_representation_nets['HamGNN_pre']['radius_scale'] = 1.01
config_default_representation_nets['HamGNN_pre']['build_internal_graph'] = False
config_default['representation_nets'] = config_default_representation_nets

"""The parameters for output_nets"""
config_default_output_nets = dict()
config_default_output_nets['output_module'] = 'HamGNN_out'
config_default_output_nets['HamGNN_out'] = dict()
config_default_output_nets['HamGNN_out']['ham_only'] = True
config_default_output_nets['HamGNN_out']['ham_type'] = 'openmx'
config_default_output_nets['HamGNN_out']['nao_max'] = 26
config_default_output_nets['HamGNN_out']['add_H0'] = True
config_default_output_nets['HamGNN_out']['add_H_nonsoc'] = False
config_default_output_nets['HamGNN_out']['symmetrize'] = True
config_default_output_nets['HamGNN_out']['calculate_band_energy'] = False
config_default_output_nets['HamGNN_out']['num_k'] = 5
config_default_output_nets['HamGNN_out']['band_num_control'] = 8
config_default_output_nets['HamGNN_out']['k_path'] = None
config_default_output_nets['HamGNN_out']['soc_switch'] = False
config_default_output_nets['HamGNN_out']['nonlinearity_type'] = 'gate'
config_default_output_nets['HamGNN_out']['spin_constrained'] = False
config_default_output_nets['HamGNN_out']['collinear_spin'] = False
config_default_output_nets['HamGNN_out']['minMagneticMoment'] = 0.5
config_default_output_nets['HamGNN_out']['zero_point_shift'] = True
config_default_output_nets['HamGNN_out']['get_nonzero_mask_tensor'] = False
config_default['output_nets'] = config_default_output_nets

"""The parameters for optimizer."""
config_default_optimizer = dict()
config_default_optimizer['lr'] = 0.01
config_default_optimizer['lr_decay'] = 0.5
config_default_optimizer['lr_patience'] = 5
config_default_optimizer['gradient_clip_val'] = 0.0
config_default_optimizer['stop_patience'] = 30
config_default_optimizer['min_epochs'] = 100
config_default_optimizer['max_epochs'] = 3000
config_default['optim_params'] = config_default_optimizer

"""The parameters for losses_metrics."""
config_default_metric = dict()
config_default_metric['losses'] = [{'metric': 'mae', 'prediction': 'hamiltonian', 'target': 'hamiltonian', 'loss_weight': 27.211}]
config_default_metric['metrics'] = [{'metric': 'mae', 'prediction': 'hamiltonian', 'target': 'hamiltonian'}]
config_default['losses_metrics'] = config_default_metric

"""The parameters for dataset."""
config_default_dataset = dict()
config_default_dataset['batch_size'] = 1
config_default_dataset['split_file'] = None
config_default_dataset['test_ratio'] = 0.2
config_default_dataset['train_ratio'] = 0.6
config_default_dataset['val_ratio'] = 0.2
config_default_dataset['graph_data_path'] = './'
config_default['dataset_params'] = config_default_dataset


def recursive_update(base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively update a dictionary with values from another dictionary.
    
    Args:
        base_dict: The dictionary to be updated (typically default configuration)
        update_dict: The dictionary containing updates (typically from user YAML file)
        
    Returns:
        Dict[str, Any]: The updated dictionary
    
    Notes:
        - If a key exists in both dictionaries and both values are dictionaries,
          recursively merge these nested dictionaries
        - If a key exists in both dictionaries but values aren't both dictionaries,
          the value from update_dict overrides the value in base_dict
        - If a key exists only in update_dict, it's added to base_dict
    """
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            # If both values are dictionaries, recursively update
            base_dict[key] = recursive_update(base_dict[key], value)
        else:
            # Otherwise, directly update the value
            base_dict[key] = value
    return base_dict


def load_config(config_file_path: Optional[str] = None) -> EasyDict:
    """
    Load configuration from a YAML file and merge it with default configuration.
    
    This function reads a YAML configuration file and recursively merges its contents
    with the default configuration. The config file path can be provided either as a
    function parameter or as a command-line argument.
    
    Args:
        config_file_path: Path to the YAML configuration file. If None, attempts to
                          get the path from command-line arguments. If not provided
                          via command line either, uses 'config_default.yaml'.
        
    Returns:
        EasyDict: An EasyDict object containing the merged configuration.
        
    Raises:
        FileNotFoundError: If the configuration file doesn't exist.
        yaml.YAMLError: If the YAML file has parsing errors.
        UnicodeDecodeError: If there are encoding issues when reading the file.
        Exception: For any other unexpected errors.
    """
    # Check if config file path is provided via command line if not given as parameter
    if not config_file_path:
        parser = argparse.ArgumentParser(description='Load configuration from a YAML file.')
        parser.add_argument('--config', '-c', type=str, default='config_default.yaml',
                           help='Path to the YAML configuration file.')
        args, _ = parser.parse_known_args()
        config_file_path = args.config
    
    # Create a deep copy of config_default to avoid modifying the original
    config_copy = copy.deepcopy(config_default)
    
    # Try to read and merge configuration from the file
    try:
        with open(config_file_path, encoding='utf-8') as config_file:
            user_config = yaml.safe_load(config_file)
        
        # If file is empty or invalid, use an empty dict
        if user_config is None:
            user_config = {}
            
        # Recursively update each configuration section
        for section_key in user_config:
            if (section_key in config_copy and 
                isinstance(config_copy[section_key], dict) and 
                isinstance(user_config[section_key], dict)):
                config_copy[section_key] = recursive_update(config_copy[section_key], user_config[section_key])
            else:
                config_copy[section_key] = user_config[section_key]
                
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_file_path}")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file {config_file_path}: {e}")
    except UnicodeDecodeError as e:
        raise UnicodeDecodeError(f"Encoding error when reading {config_file_path}: {e}")
    except Exception as e:
        raise Exception(f"Unexpected error when reading {config_file_path}: {e}")
    
    # Convert to EasyDict
    config = EasyDict(config_copy)
    
    # Process specific fields if they exist
    if hasattr(config, 'losses_metrics'):
        if hasattr(config.losses_metrics, 'losses'):
            config.losses_metrics.losses = parse_metric_func(config.losses_metrics.losses)
        if hasattr(config.losses_metrics, 'metrics'):
            config.losses_metrics.metrics = parse_metric_func(config.losses_metrics.metrics)
    
    return config

