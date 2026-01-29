import os
import shutil
import numpy as np
import torch
import pytorch_lightning as pl
from easydict import EasyDict
from ase.units import Bohr
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import yaml
import subprocess
import pickle
from typing import Tuple, List, Optional
import yaml
import argparse
from HamGNN_v_2_1.models.hamgnn_conv import HamGNNConvE3 
from HamGNN_v_2_1.models.hamgnn_output import HamGNNPlusPlusOut
from HamGNN_v_2_1.main import Model

def read_config(config_file_name: str = 'config_default.yaml') -> EasyDict:
    """Read and parse a YAML configuration file into an EasyDict object.

    Args:
        config_file_name (str, optional): Path to the YAML configuration file. 
            Defaults to 'config_default.yaml'.

    Returns:
        EasyDict: Configuration parameters as an EasyDict for attribute-style access.
    """
    with open(config_file_name, encoding='utf-8') as rstream:
        config_data = yaml.load(rstream, yaml.SafeLoader)
    return EasyDict(config_data)

def build_hamgnn_components(config: EasyDict) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """Build the HamGNN model components including the graph representation network and output module.

    Args:
        config (EasyDict): Configuration parameters for model construction.

    Returns:
        Tuple[torch.nn.Module, torch.nn.Module]: A tuple containing the graph representation network 
        and the Hamiltonian output module.
    """
    print("Building HamGNN model components")
    # Synchronize configuration parameters between representation and output networks
    config.representation_nets.HamGNN_pre.radius_type = config.output_nets.HamGNN_out.ham_type.lower()
    config.representation_nets.HamGNN_pre.ham_type = config.output_nets.HamGNN_out.ham_type.lower()
    config.representation_nets.HamGNN_pre.nao_max = config.output_nets.HamGNN_out.nao_max
    config.representation_nets.HamGNN_pre.use_corr_prod = False
    # Initialize graph representation network
    graph_representation_network = HamGNNConvE3(config.representation_nets)
    
    # Initialize Hamiltonian output module
    output_params = config.output_nets.HamGNN_out
    hamiltonian_output_module = HamGNNPlusPlusOut(
        irreps_in_node=graph_representation_network.irreps_node_features,
        irreps_in_edge=graph_representation_network.irreps_node_features,
        nao_max=output_params.nao_max,
        ham_type=output_params.ham_type,
        ham_only=output_params.ham_only,
        symmetrize=output_params.symmetrize,
        calculate_band_energy=output_params.calculate_band_energy,
        num_k=output_params.num_k,
        k_path=output_params.k_path,
        band_num_control=output_params.band_num_control,
        soc_switch=output_params.soc_switch,
        nonlinearity_type=output_params.nonlinearity_type,
        add_H0=output_params.add_H0,
        spin_constrained=output_params.spin_constrained,
        collinear_spin=output_params.collinear_spin,
        minMagneticMoment=output_params.minMagneticMoment,
        add_H_nonsoc=True if output_params.soc_switch else False,
        zero_point_shift=False if output_params.soc_switch else True,
        get_nonzero_mask_tensor=True
    )
    return graph_representation_network, hamiltonian_output_module

def save_model_predictor(predictor, model_filepath):
    with open(model_filepath, 'wb') as f:
        pickle.dump(predictor, f)

def load_model_predictor(model_filepath: str, device: Optional[str] = None):
    """
    Loads a model predictor from a pickle file and assigns it to the specified device.

    Args:
        model_filepath (str): Path to the pickle file containing the predictor model.
        device (Optional[str]): The computation device ('cpu' or 'cuda') to assign the model to.
                                 If None, the device assignment is skipped.

    Returns:
        Predictor: The loaded predictor model with device assigned if specified.
    """
    with open(model_filepath, 'rb') as file_handle:
        model_predictor = pickle.load(file_handle)

    if device:
        model_predictor.device = device
        model_predictor.non_soc_model = model_predictor.non_soc_model.to(device)
        if hasattr(model_predictor, 'soc_model') and model_predictor.soc_model:
            model_predictor.soc_model = model_predictor.soc_model.to(device)

    return model_predictor

class HamiltonianPredictor:
    """Manages Hamiltonian prediction using GNN models, handling both non-SOC and SOC calculations."""
    
    @staticmethod
    def _get_most_available_gpu() -> int:
        """Identifies the GPU with the most available memory.
        
        Returns:
            int: Index of the GPU with highest available memory. Returns 0 if detection fails.
        """
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,noununits"],
                capture_output=True, text=True, check=True
            )
            gpu_stats = []
            for line in result.stdout.strip().split('\n'):
                idx, used, total = map(int, line.split(','))
                gpu_stats.append(((total - used) / total, idx))
            return max(gpu_stats)[1] if gpu_stats else 0
        except Exception as e:
            print(f"GPU selection error: {e}")
            return 0

    def __init__(
        self,
        config_nonsoc_path: str,
        config_soc_path: Optional[str] = None,
        soc_switch: bool = False,
        device: str = 'cuda'
    ):
        """Initializes predictor with configurations and device setup.

        Args:
            config_nonsoc_path (str): Path to non-SOC model configuration.
            config_soc_path (Optional[str]): Path to SOC model configuration if applicable.
            soc_switch (bool): Enable SOC calculation if True.
            device (str): Target device ('cpu' or 'cuda').
        """
        self.soc_enabled = soc_switch
        self._validate_config_paths(config_nonsoc_path, config_soc_path)
        
        # Load configurations
        self.config_nonsoc = read_config(config_nonsoc_path)
        self.config_soc = read_config(config_soc_path) if self.soc_enabled else None
        
        # Device configuration
        self.device = self._configure_device(device)
        
        # Model initialization
        self.non_soc_model = self._load_model(self.config_nonsoc).to(self.device).eval()
        self.soc_model = self._load_model(self.config_soc).to(self.device).eval() if self.soc_enabled else None

    def _validate_config_paths(self, *paths: Optional[str]) -> None:
        """Validates existence of required configuration files."""
        for path in filter(None, paths):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Configuration file missing: {path}")

    def _configure_device(self, device: str) -> str:
        """Configures and validates the target device."""
        if device.lower().startswith('cuda'):
            if not torch.cuda.is_available():
                print("CUDA unavailable, falling back to CPU")
                return 'cpu'
            device = f'cuda:{self._get_most_available_gpu()}'
        return device

    @staticmethod
    def _load_model(config: EasyDict) -> Model:
        """Loads a trained model from checkpoint."""
        graph_net, output_module = build_hamgnn_components(config)
        return Model.load_from_checkpoint(
            checkpoint_path=config.setup.checkpoint_path,
            representation=graph_net,
            output=output_module,
            post_processing=None,
            losses=None,
            validation_metrics=None,
            lr=None,
            lr_decay=None,
            lr_patience=None
        )

    def predict_hamiltonians(
        self,
        non_soc_data_path: str,
        soc_data_path: Optional[str] = None,
        calculate_mae: bool = False
    ) -> Tuple[np.ndarray, ...]:
        """Predicts Hamiltonian matrices with optional MAE calculation.

        Returns:
            Tuple containing predicted Hamiltonians and optional MAE metrics.
        """
        non_soc_loader = self._create_data_loader(non_soc_data_path)
        soc_loader = self._create_data_loader(soc_data_path) if self.soc_enabled else None
        
        hamiltonians = []
        real_mae_values = [] if calculate_mae else None
        imag_mae_values = [] if calculate_mae and self.soc_enabled else None
        
        if calculate_mae:
            self.non_soc_model.output_module.zero_point_shift = True
            self.soc_model.output_module.zero_point_shift = True
        else:
            self.non_soc_model.output_module.zero_point_shift = False
        
        data_pairs = zip(non_soc_loader, soc_loader) if self.soc_enabled else non_soc_loader
        for batch in tqdm(data_pairs, total=len(non_soc_loader)):
            batch = self._process_batch(batch)
            with torch.no_grad():
                pred = self._model_forward(batch)
                hamiltonians.append(pred['hamiltonian'].cpu().numpy())
                if calculate_mae:
                    self._calculate_batch_mae(pred, batch, real_mae_values, imag_mae_values)
        
        final_hamiltonian = np.concatenate(hamiltonians)
        if not calculate_mae:
            return (final_hamiltonian,)
        return self._package_mae_results(final_hamiltonian, real_mae_values, imag_mae_values)

    def _create_data_loader(self, data_path: str) -> DataLoader:
        """Creates a DataLoader with optimized settings."""
        return DataLoader(
            dataset=self._load_graph_data(data_path),
            batch_size=1,
            pin_memory=self.device.startswith('cuda')
        )

    @staticmethod
    def _load_graph_data(data_path: str) -> List:
        """Loads graph data from .npz file."""
        data_file = data_path if data_path.endswith('.npz') else os.path.join(data_path, "graph_data.npz")
        return list(np.load(data_file, allow_pickle=True)['graph'].item().values())

    def _process_batch(self, batch) -> torch.Tensor:
        """Transfers batch data to target device."""
        if self.soc_enabled:
            return [data.to(self.device, non_blocking=True) for data in batch]
        return batch.to(self.device, non_blocking=True)

    def _model_forward(self, batch) -> dict:
        """Executes model forward pass with SOC/non-SOC handling."""
        if self.soc_enabled:
            non_soc_batch, soc_batch = batch
            if 'hamiltonian' not in non_soc_batch:
                if self.non_soc_model.output_module.zero_point_shift:
                    non_soc_batch['hamiltonian'] = torch.cat([non_soc_batch['Hon'], non_soc_batch['Hoff']])
                else:
                    non_soc_batch['hamiltonian'] = 0.0
            if 'hamiltonian' not in soc_batch:
                if self.soc_model.output_module.zero_point_shift:
                    soc_batch['hamiltonian'] = torch.cat([soc_batch['Hon'], soc_batch['Hoff'], soc_batch['iHon'], soc_batch['iHoff']])
                else:
                    soc_batch['hamiltonian'] = 0.0
            non_soc_pred = self.non_soc_model(non_soc_batch)
            soc_batch.update({
                'Hon_nonsoc': non_soc_pred['hamiltonian'][:len(soc_batch.z)],
                'Hoff_nonsoc': non_soc_pred['hamiltonian'][len(soc_batch.z):]
            })
            return self.soc_model(soc_batch)
        
        if 'hamiltonian' not in batch:
            if self.non_soc_model.output_module.zero_point_shift:
                batch['hamiltonian'] = torch.cat([batch['Hon'], batch['Hoff']])
            else:
                batch['hamiltonian'] = 0.0

        return self.non_soc_model(batch)

    def _calculate_batch_mae(self, pred, batch, real_mae, imag_mae) -> None:
        """Calculates MAE metrics for a batch."""
        if self.soc_enabled:
            _, soc_batch = batch
            real_diff = torch.abs(pred['hamiltonian_real'] - torch.cat([soc_batch['Hon'], soc_batch['Hoff']]))
            imag_diff = torch.abs(pred['hamiltonian_imag'] - torch.cat([soc_batch['iHon'], soc_batch['iHoff']]))
            real_mae.extend(real_diff.flatten()[pred['mask_real_imag'].flatten().bool()].cpu().numpy())
            imag_mae.extend(imag_diff.flatten()[pred['mask_real_imag'].flatten().bool()].cpu().numpy())
        else:
            diff = torch.abs(pred['hamiltonian'] - torch.cat([batch['Hon'], batch['Hoff']]))
            real_mae.extend(diff.flatten()[pred['mask'].flatten().bool()].cpu().numpy())

    def _package_mae_results(self, hamiltonian, real_mae, imag_mae) -> Tuple:
        """Packages results with MAE metrics."""
        if self.soc_enabled:
            return (hamiltonian, np.mean(real_mae), np.mean(imag_mae))
        return (hamiltonian, np.mean(real_mae))

def predict_and_save_hamiltonian(
    model_pkl_path: str,
    non_soc_data_dir: str,
    soc_data_dir: Optional[str] = None,
    output_dir: str = None,
    device: str = 'cuda',
    calculate_mae: bool = False
) -> None:
    """Predicts and saves Hamiltonian matrices with optional SOC calculation.

    Args:
        model_filepath (str): Path to the pickle file containing the predictor model.
        non_soc_data_dir (Optional[str]): non-SOC graph data directory
        soc_data_dir (Optional[str]): SOC graph data directory
        device (str): Target device
        calculate_mae (bool): Calculate MAE metrics
        output_dir (str): Output directory for results
    """
    
    predictor = load_model_predictor(model_pkl_path)

    if soc_data_dir is None:
        print("Warning: soc_data_dir is None. Forcing soc_enabled to False.")
        predictor.soc_enabled = False

    prediction_args = {
        'non_soc_data_path': non_soc_data_dir,
        'soc_data_path': soc_data_dir,
        'calculate_mae': calculate_mae
    }
    results = predictor.predict_hamiltonians(**prediction_args)
    
    np.save(os.path.join(output_dir, 'hamiltonian.npy'), results[0])

    if calculate_mae:
        mae_info = "\n".join([f"MAE ({desc}): {val:.4e}" 
                            for desc, val in zip(['Real', 'Imag'][:len(results)-1], results[1:])])
        print(mae_info)


if __name__ == "__main__":
    # Parse command line arguments to specify config file path
    parser = argparse.ArgumentParser(description='Read parameters from config file and predict Hamiltonian')
    parser.add_argument('--config', type=str, default='Input.yaml', help='Path to configuration file')
    args = parser.parse_args()
    
    # Read YAML configuration file
    with open(args.config, 'r') as config_file:
        config = yaml.safe_load(config_file)
    
    # Extract parameters from configuration
    model_pkl_path = config.get('model_pkl_path', '')
    non_soc_data_dir = config.get('non_soc_data_dir', '')
    soc_data_dir = config.get('soc_data_dir', '')
    output_dir = config.get('output_dir', './')
    device = config.get('device', 'cpu')
    calculate_mae = config.get('calculate_mae', False)
    
    # Call the function with configured parameters
    predict_and_save_hamiltonian(
        model_pkl_path=model_pkl_path,
        non_soc_data_dir=non_soc_data_dir,
        soc_data_dir=soc_data_dir,
        output_dir=output_dir,
        device=device,
        calculate_mae=calculate_mae

    )
