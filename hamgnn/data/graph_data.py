# Copyright (c) 2021-2026 HamGNN Team
# SPDX-License-Identifier: GPL-3.0-only

"""Graph datasets, LMDB storage, and PyTorch Lightning data interfaces for HamGNN.

Provides :class:`LMDBGraphDataset`, graph collation, train/val/test splits, and
loaders built on PyTorch Geometric :class:`~torch_geometric.data.Data`.
"""

import pytorch_lightning as pl
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from typing import Union, Callable, Optional, List
import numpy as np
from torch.utils.data import random_split, Subset, Dataset
import os
import lmdb
import pickle
import torch
import math
from tqdm import tqdm
from collections import OrderedDict

from e3nn import o3

from ..toolbox.nequip.data import AtomicDataDict
from ..utils.basis_functions import (
    BernsteinRadialBasisFunctions,
    BesselBasis,
    ExponentialBernsteinRadialBasisFunctions,
    ExponentialGaussianRadialBasisFunctions,
    GaussianSmearing,
)
from ..utils.cutoff_functions import CosineCutoff


class StaticEdgeFeatureTransform:
    """Precompute static edge attributes/embeddings for fixed-graph workloads.

    Replicates, once per graph, the edge features that the representation
    network would otherwise recompute on every training step: spherical
    harmonics of the edge directions (``edge_attrs``), the radial basis
    expansion of edge lengths (``edge_embedding``), and ``edge_lengths``.
    These quantities depend only on the fixed graph geometry, so caching
    them is exact.
    """

    def __init__(
        self,
        irreps_edge_sh: str,
        edge_sh_normalization: str,
        edge_sh_normalize: bool,
        rbf_func: str,
        num_radial: int,
        cutoff: float,
    ):
        self.coord_change = torch.LongTensor([1, 2, 0])
        self.sh = o3.SphericalHarmonics(
            o3.Irreps(irreps_edge_sh),
            edge_sh_normalize,
            edge_sh_normalization,
        )
        self.cutoff = CosineCutoff(cutoff)
        self.basis = self._build_radial_basis(
            rbf_func=rbf_func,
            num_radial=num_radial,
            cutoff=cutoff,
        )

    @staticmethod
    def _build_radial_basis(rbf_func: str, num_radial: int, cutoff: float):
        rbf_func = rbf_func.lower()
        if rbf_func == "gaussian":
            return GaussianSmearing(
                start=0.0, stop=cutoff, num_gaussians=num_radial, cutoff_func=None
            )
        if rbf_func == "bessel":
            return BesselBasis(cutoff=cutoff, n_rbf=num_radial, cutoff_func=None)
        if rbf_func == "exp-gaussian":
            return ExponentialGaussianRadialBasisFunctions(num_radial, cutoff)
        if rbf_func == "exp-bernstein":
            return ExponentialBernsteinRadialBasisFunctions(num_radial, cutoff)
        if rbf_func == "bernstein":
            return BernsteinRadialBasisFunctions(num_radial, cutoff)
        raise ValueError(f"Unsupported radial basis function: {rbf_func}")

    def __call__(self, data):
        if (
            AtomicDataDict.EDGE_ATTRS_KEY in data
            and AtomicDataDict.EDGE_EMBEDDING_KEY in data
            and AtomicDataDict.EDGE_LENGTH_KEY in data
        ):
            return data

        j, i = data.edge_index
        nbr_shift = data.nbr_shift
        pos = data.pos

        edge_dir = (pos[i] + nbr_shift) - pos[j]
        edge_length = edge_dir.norm(dim=-1)

        if AtomicDataDict.EDGE_ATTRS_KEY not in data:
            edge_unit = edge_dir / edge_length[:, None]
            data[AtomicDataDict.EDGE_ATTRS_KEY] = self.sh(
                edge_unit[:, self.coord_change]
            )
        if AtomicDataDict.EDGE_EMBEDDING_KEY not in data:
            edge_embedding = self.basis(edge_length) * self.cutoff(edge_length)[:, None]
            data[AtomicDataDict.EDGE_EMBEDDING_KEY] = edge_embedding
        data[AtomicDataDict.EDGE_LENGTH_KEY] = edge_length
        return data


def build_graph_feature_transform(config):
    """Build an optional exact-preserving static edge-feature transform.

    Returns ``None`` (feature disabled) unless
    ``config.dataset_params.enable_static_edge_features`` is true and the
    configuration describes a plain Hamiltonian workload without force
    outputs, i.e. a setting where edge geometry features never change
    during training.
    """
    dataset_params = getattr(config, "dataset_params", None)
    if dataset_params is None or not getattr(
        dataset_params, "enable_static_edge_features", False
    ):
        return None

    gnn_net_type = getattr(config.setup, "GNN_Net", "").lower()
    if gnn_net_type not in {"hamgnnconv", "hamgnnpre", "hamgnn_pre", "hamgnntransformer"}:
        return None

    property_type = str(getattr(config.setup, "property", "")).lower()
    if property_type != "hamiltonian" or "epc" in property_type or "force" in property_type:
        return None

    output_nets = getattr(config, "output_nets", None)
    output_module_name = getattr(output_nets, "output_module", None) if output_nets is not None else None
    output_params = None
    if output_module_name is not None and hasattr(output_nets, output_module_name):
        output_params = getattr(output_nets, output_module_name)
    elif output_nets is not None and hasattr(output_nets, "HamGNN_out"):
        output_params = output_nets.HamGNN_out

    if output_params is not None and (
        getattr(output_params, "return_forces", False)
        or getattr(output_params, "create_graph", False)
    ):
        return None

    ham_pre = config.representation_nets.HamGNN_pre
    return StaticEdgeFeatureTransform(
        irreps_edge_sh=ham_pre.irreps_edge_sh,
        edge_sh_normalization=ham_pre.edge_sh_normalization,
        edge_sh_normalize=ham_pre.edge_sh_normalize,
        rbf_func=ham_pre.rbf_func,
        num_radial=ham_pre.num_radial,
        cutoff=ham_pre.cutoff,
    )


class LMDBGraphDataset(Dataset):
    """
    LMDB graph data loader based on regular Dataset implementation
    """
    def __init__(self, lmdb_path: str, indices: List[int] = None,
                 transform: Callable = None, preload: int = 0, cache_size: int = 0):
        super(LMDBGraphDataset, self).__init__()
        self.lmdb_path = lmdb_path
        self.transform = transform
        self.preload = preload
        self.cache_size = max(int(cache_size), 0)
        self.preloaded_data = {}
        self.runtime_cache = OrderedDict()

        # Open LMDB environment to read metadata
        env = lmdb.open(lmdb_path, readonly=True, lock=False)
        with env.begin() as txn:
            self.total_length = int(txn.get('num_graphs'.encode()).decode())

        # Set indices
        self.indices = indices if indices is not None else list(range(self.total_length))

        # Optionally preload some data to improve performance
        if self.preload > 0:
            n_preload = min(self.preload, len(self.indices))
            indices_to_preload = self.indices[:n_preload]

            with env.begin() as txn:
                for idx in tqdm(indices_to_preload, desc="Preloading data"):
                    data_bytes = txn.get(f'graph_{idx}'.encode())
                    if data_bytes is not None:
                        data = pickle.loads(data_bytes)
                        # Apply the transform once here so preloaded graphs
                        # are served ready-to-use.
                        if self.transform is not None:
                            data = self.transform(data)
                        self.preloaded_data[idx] = data

        env.close()

        # Maintain an environment connection to avoid repeated opening/closing
        self.env = None
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        if isinstance(idx, list):
            return [self[i] for i in idx]

        # Get the actual index
        real_idx = self.indices[idx]

        # Check if already preloaded (transform was applied at preload time)
        preloaded_data = self.preloaded_data.get(real_idx)
        if preloaded_data is not None:
            return preloaded_data

        # Check the LRU runtime cache of transformed graphs
        cached_data = self.runtime_cache.get(real_idx)
        if cached_data is not None:
            self.runtime_cache.move_to_end(real_idx)
            return cached_data

        # Lazy load LMDB environment
        if self.env is None:
            self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False,
                                 readahead=False, meminit=False)

        # Load from LMDB
        with self.env.begin() as txn:
            data_bytes = txn.get(f'graph_{real_idx}'.encode())
            if data_bytes is None:
                raise IndexError(f"Index {real_idx} out of bounds for LMDB dataset")
            data = pickle.loads(data_bytes)

        # Apply transformation
        if self.transform is not None:
            data = self.transform(data)

        # Store the transformed graph in the LRU runtime cache
        if self.cache_size > 0:
            self.runtime_cache[real_idx] = data
            self.runtime_cache.move_to_end(real_idx)
            while len(self.runtime_cache) > self.cache_size:
                self.runtime_cache.popitem(last=False)

        return data
    
    def __del__(self):
        # Ensure environment is closed
        if self.env is not None:
            self.env.close()

class NPZGraphDataset(Dataset):
    """
    NPZ graph data loader based on regular Dataset implementation
    """
    def __init__(self, npz_path: str, indices: List[int] = None,
                 transform: Callable = None, preload: int = 0, cache_size: int = 0):
        super(NPZGraphDataset, self).__init__()
        self.npz_path = npz_path
        self.transform = transform
        self.preload = preload
        self.cache_size = max(int(cache_size), 0)
        self.preloaded_data = {}
        self.runtime_cache = OrderedDict()
        
        # Load NPZ file to get metadata
        try:
            with np.load(npz_path, allow_pickle=True) as data:
                # Handle NPZ files of different architecture
                if 'graph' in data:
                    # Architecture: {'graph': dict_of_graphs}
                    graph_data = data['graph'].item()
                    if isinstance(graph_data, dict):
                        self.data_list = list(graph_data.values())
                    else:
                        self.data_list = graph_data
                    self.total_length = len(self.data_list)
                else:
                    # Try to directly use keys as the list of graph data
                    keys = list(data.keys())
                    self.data_list = [data[key] for key in keys]
                    self.total_length = len(self.data_list)

                print(f"Loaded {self.total_length} graphs from NPZ file")
        except Exception as e:
            raise RuntimeError(f"Failed to load NPZ file: {e}")
        
        # Set indices
        self.indices = indices if indices is not None else list(range(self.total_length))
        
        # Optional preloaded data to improve performance
        if self.preload > 0:
            n_preload = min(self.preload, len(self.indices))
            indices_to_preload = self.indices[:n_preload]

            for idx in tqdm(indices_to_preload, desc="Preloading data"):
                data = self._process_graph(self.data_list[idx])
                # Apply the transform once here so preloaded graphs
                # are served ready-to-use.
                if self.transform is not None:
                    data = self.transform(data)
                self.preloaded_data[idx] = data
    
    def _process_graph(self, data):
        """Process the graph data to ensure consistent format"""
        # If the data is already a torch_geometric.data.Data object, return it directly
        if isinstance(data, Data):
            return data
        
        # If the data is a dictionary, attempt to convert it to a Data object
        if isinstance(data, dict) and 'edge_index' in data:
            # Convert a NumPy array to a PyTorch tensor
            processed_data = {}
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    processed_data[key] = torch.from_numpy(value)
                else:
                    processed_data[key] = value
            
            # Create a Data object
            return Data(**processed_data)
            
        # Return as is when conversion is not possible
        return data
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        if isinstance(idx, list):
            return [self[i] for i in idx]

        # Obtain the actual index
        real_idx = self.indices[idx]

        # Check if it has been preloaded (transform was applied at preload time)
        preloaded_data = self.preloaded_data.get(real_idx)
        if preloaded_data is not None:
            return preloaded_data

        # Check the LRU runtime cache of transformed graphs
        cached_data = self.runtime_cache.get(real_idx)
        if cached_data is not None:
            self.runtime_cache.move_to_end(real_idx)
            return cached_data

        # Obtain and process from the data list
        data = self._process_graph(self.data_list[real_idx])

        # Application of transformation
        if self.transform is not None:
            data = self.transform(data)

        # Store the transformed graph in the LRU runtime cache
        if self.cache_size > 0:
            self.runtime_cache[real_idx] = data
            self.runtime_cache.move_to_end(real_idx)
            while len(self.runtime_cache) > self.cache_size:
                self.runtime_cache.popitem(last=False)

        return data

class graph_data_module(pl.LightningDataModule):
    """
    Data module for efficiently handling large-scale graph data
    """
    def __init__(self, 
                 dataset: Union[list, tuple, np.array, str] = None,
                 train_ratio: float = 0.6,
                 val_ratio: float = 0.2,
                 test_ratio: float = 0.2,
                 batch_size: int = 64,
                 val_batch_size: int = None,
                 test_batch_size: int = None,
                 split_file: str = None,
                 num_workers: int = 4,
                 prefetch_factor: int = 2,
                 cache_size: int = 100,
                 transform: Callable = None,
                 persistent_workers: bool = True,
                 preload: int = 0,
                 test_mode: bool = False,
                 data_format: str = 'auto'):
        super(graph_data_module, self).__init__()
        self.dataset_input = dataset
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.batch_size = batch_size
        self.split_file = split_file
        self.val_batch_size = val_batch_size or batch_size
        self.test_batch_size = test_batch_size or self.val_batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.cache_size = cache_size
        self.transform = transform
        self.persistent_workers = persistent_workers
        self.preload = preload
        self.test_mode = test_mode
        self.data_format = data_format
        
    def prepare_data(self):
        """Prepare data - Verify the accessibility of the validation dataset"""
        if isinstance(self.dataset_input, str):
            # If set to 'auto', determine the data format
            if self.data_format == 'auto':
                if self.dataset_input.endswith('.lmdb'):
                    self.data_format = 'lmdb'
                elif self.dataset_input.endswith('.npz'):
                    self.data_format = 'npz'
                else:
                    raise ValueError(f"The data format cannot be determined from the file extension: {self.dataset_input}")

            # Verify that the file exists and is readable
            if not os.path.exists(self.dataset_input):
                raise FileNotFoundError(f"The data path does not exist: {self.dataset_input}")

            # Verify the LMDB database
            if self.data_format == 'lmdb':
                try:
                    env = lmdb.open(self.dataset_input, readonly=True, lock=False)
                    with env.begin() as txn:
                        num_graphs = int(txn.get('num_graphs'.encode()).decode())
                        print(f"Found {num_graphs} graphs in the LMDB database")
                    env.close()
                except Exception as e:
                    raise RuntimeError(f"Failed to open the LMDB database: {e}")

            # Verify the NPZ file
            elif self.data_format == 'npz':
                try:
                    with np.load(self.dataset_input, allow_pickle=True) as data:
                        if 'graph' in data:
                            graph_data = data['graph'].item()
                            num_graphs = len(graph_data)
                        else:
                            num_graphs = len(data.keys())
                        
                        print(f"Found {num_graphs} graphs in the NPZ file")
                except Exception as e:
                    raise RuntimeError(f"Failed to open NPZ file: {e}")

    def get_dataset_length(self):
        """Get the length of the dataset"""
        if isinstance(self.dataset_input, str):
            if self.data_format == 'lmdb':
                env = lmdb.open(self.dataset_input, readonly=True, lock=False)
                with env.begin() as txn:
                    length = int(txn.get('num_graphs'.encode()).decode())
                env.close()
                return length
            elif self.data_format == 'npz':
                with np.load(self.dataset_input, allow_pickle=True) as data:
                    if 'graph' in data:
                        graph_data = data['graph'].item()
                        return len(graph_data)
                    else:
                        return len(data.keys())
        else:
            return len(self.dataset_input)
            
    def setup(self, stage=None):
        """Set dataset split"""
        # Determine the data format
        if isinstance(self.dataset_input, str):
            if self.data_format == 'auto':
                if self.dataset_input.endswith('.lmdb'):
                    use_lmdb = True
                    use_npz = False
                    self.data_format = 'lmdb'
                elif self.dataset_input.endswith('.npz'):
                    use_lmdb = False
                    use_npz = True
                    self.data_format = 'npz'
                else:
                    raise ValueError(f"The data format cannot be determined from the file extension: {self.dataset_input}")
            elif self.data_format == 'lmdb':
                use_lmdb = True
                use_npz = False
            elif self.data_format == 'npz':
                use_lmdb = False
                use_npz = True
            else:
                raise ValueError(f"Unsupported data format: {self.data_format}")

            dataset_length = self.get_dataset_length()
        else:
            use_lmdb = False
            use_npz = False
            self.dataset = self.dataset_input
            dataset_length = len(self.dataset)

        # If in test mode, use the entire dataset as the test set
        if self.test_mode:
            print("Test mode: Use the entire dataset as the test set without splitting")
            all_indices = list(range(dataset_length))

            if use_lmdb:
                # Test using all indices
                self.test_data = LMDBGraphDataset(
                    self.dataset_input, indices=all_indices, 
                    transform=self.transform, preload=self.preload,
                    cache_size=self.cache_size
                )

                # Create empty training and validation sets (for placeholder purposes)
                self.train_data = LMDBGraphDataset(
                    self.dataset_input, indices=[],
                    transform=self.transform,
                    cache_size=self.cache_size
                )
                self.val_data = LMDBGraphDataset(
                    self.dataset_input, indices=[],
                    transform=self.transform,
                    cache_size=self.cache_size
                )
            elif use_npz:
                # Use the dataset in NPZ format
                self.test_data = NPZGraphDataset(
                    self.dataset_input, indices=all_indices, 
                    transform=self.transform, preload=self.preload,
                    cache_size=self.cache_size
                )

                # Create empty training and validation sets
                self.train_data = NPZGraphDataset(
                    self.dataset_input, indices=[],
                    transform=self.transform,
                    cache_size=self.cache_size
                )
                self.val_data = NPZGraphDataset(
                    self.dataset_input, indices=[],
                    transform=self.transform,
                    cache_size=self.cache_size
                )
            else:
                # Test using the entire dataset
                self.test_data = Subset(self.dataset, indices=all_indices)

                # Create empty training and validation sets
                self.train_data = Subset(self.dataset, indices=[])
                self.val_data = Subset(self.dataset, indices=[])
        # Regular mode (non-test mode)
        else:
            # Load or create dataset splits
            if self.split_file is not None and os.path.exists(self.split_file):
                print(f"Load the split index from {self.split_file}")
                split_data = np.load(self.split_file)
                train_indices = split_data["train_idx"].tolist()
                val_indices = split_data["val_idx"].tolist()
                test_indices = split_data["test_idx"].tolist()
            else:
                print("Create a new dataset split")
                # Create a random index
                random_state = np.random.RandomState(seed=42)
                indices = list(range(dataset_length))
                random_state.shuffle(indices)

                num_train = round(self.train_ratio * dataset_length)
                num_val = round(self.val_ratio * dataset_length)

                train_indices = indices[:num_train]
                val_indices = indices[num_train:num_train+num_val]
                test_indices = indices[num_train+num_val:]

                # Save the split information
                if self.split_file is not None:
                    np.savez(self.split_file, 
                            train_idx=np.array(train_indices), 
                            val_idx=np.array(val_indices), 
                            test_idx=np.array(test_indices))
                    
                    print(f"The split index has been saved to {self.split_file}")

            # Create an appropriate type of dataset
            if stage == 'fit' or stage is None:
                if use_lmdb:
                    # Processing LMDB with LMDBGraphDataset
                    
                    print("Create training and validation datasets from LMDB")
                    self.train_data = LMDBGraphDataset(
                        self.dataset_input, indices=train_indices, 
                        transform=self.transform, preload=self.preload,
                        cache_size=self.cache_size
                    )
                    self.val_data = LMDBGraphDataset(
                        self.dataset_input, indices=val_indices, 
                        transform=self.transform, preload=self.preload,
                        cache_size=self.cache_size
                    )
                elif use_npz:
                    # Processing NPZ files using NPZGraphDataset
                    
                    print("Create training and validation datasets from NPZ")
                    self.train_data = NPZGraphDataset(
                        self.dataset_input, indices=train_indices, 
                        transform=self.transform, preload=self.preload,
                        cache_size=self.cache_size
                    )
                    self.val_data = NPZGraphDataset(
                        self.dataset_input, indices=val_indices, 
                        transform=self.transform, preload=self.preload,
                        cache_size=self.cache_size
                    )
                else:
                    # Using Subset to Process Regular Lists
                    self.train_data = Subset(self.dataset, indices=train_indices)
                    self.val_data = Subset(self.dataset, indices=val_indices)

            if stage == 'test' or stage is None:
                if use_lmdb:
                    print("Create a test dataset from LMDB")
                    self.test_data = LMDBGraphDataset(
                        self.dataset_input, indices=test_indices, 
                        transform=self.transform, preload=self.preload,
                        cache_size=self.cache_size
                    )
                elif use_npz:
                    print("Create a test dataset from NPZ")
                    self.test_data = NPZGraphDataset(
                        self.dataset_input, indices=test_indices, 
                        transform=self.transform, preload=self.preload,
                        cache_size=self.cache_size
                    )
                else:
                    self.test_data = Subset(self.dataset, indices=test_indices)

    def train_dataloader(self):
        loader_kwargs = {
            'batch_size': self.batch_size,
            'pin_memory': True,
            'shuffle': True,
        }
        
        # Add data loading performance related parameters
        loader_kwargs.update({
            'num_workers': self.num_workers,
            'prefetch_factor': self.prefetch_factor if self.num_workers > 0 else None,
            'persistent_workers': self.persistent_workers if self.num_workers > 0 else False
        })
        
        return DataLoader(self.train_data, **loader_kwargs)
    
    def val_dataloader(self):
        loader_kwargs = {
            'batch_size': self.val_batch_size,
            'pin_memory': True,
            'shuffle': False,
        }
        
        # Add data loading performance related parameters
        loader_kwargs.update({
            'num_workers': self.num_workers,
            'prefetch_factor': self.prefetch_factor if self.num_workers > 0 else None,
            'persistent_workers': self.persistent_workers if self.num_workers > 0 else False
        })
        
        return DataLoader(self.val_data, **loader_kwargs)
    
    def test_dataloader(self):
        loader_kwargs = {
            'batch_size': self.test_batch_size,
            'pin_memory': True,
            'shuffle': False,
        }
        
        # Add data loading performance related parameters
        loader_kwargs.update({
            'num_workers': self.num_workers,
            'prefetch_factor': self.prefetch_factor if self.num_workers > 0 else None,
            'persistent_workers': self.persistent_workers if self.num_workers > 0 else False
        })
        
        return DataLoader(self.test_data, **loader_kwargs)
        
    def save_split(self, split_file):
        """
        Save current dataset split to file
        """
        if not hasattr(self, 'train_data') or not hasattr(self, 'val_data') or not hasattr(self, 'test_data'):
            raise RuntimeError("Dataset has not been set up yet. Call setup() first.")
            
        # Handle different types of datasets to get indices
        if isinstance(self.train_data, LMDBGraphDataset):
            train_indices = self.train_data.indices
        elif isinstance(self.train_data, Subset):
            train_indices = self.train_data.indices
        else:
            train_indices = list(range(len(self.train_data)))
            
        if isinstance(self.val_data, LMDBGraphDataset):
            val_indices = self.val_data.indices
        elif isinstance(self.val_data, Subset):
            val_indices = self.val_data.indices
        else:
            val_indices = list(range(len(self.val_data)))
            
        if isinstance(self.test_data, LMDBGraphDataset):
            test_indices = self.test_data.indices
        elif isinstance(self.test_data, Subset):
            test_indices = self.test_data.indices
        else:
            test_indices = list(range(len(self.test_data)))
        
        np.savez(split_file, 
                 train_idx=np.array(train_indices), 
                 val_idx=np.array(val_indices), 
                 test_idx=np.array(test_indices))
        
        print(f"Split indices saved to {split_file}")

