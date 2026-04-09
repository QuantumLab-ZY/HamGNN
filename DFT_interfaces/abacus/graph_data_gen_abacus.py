# Copyright (c) 2021-2026 HamGNN Team
# SPDX-License-Identifier: GPL-3.0-only

"""Batch generation of training graphs and labels from ABACUS SCF / sparse outputs.

Reads STRU and Hamiltonian data, writes LMDB shards with PyG :class:`~torch_geometric.data.Data`
for HamGNN training or inference prep.
"""

import os
import pickle
import shutil
import lmdb
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import multiprocessing
from functools import lru_cache
import argparse
from read_abacus import STRU, ABACUSHS, read_abacus_input, get_neutral_electrons, calculate_doping_charge
from build_graph_from_coordinates import build_graph, compute_graph_difference, find_inverse_edge_index
from utils import *

################################ Input Parameters ##############################
# Maximum number of atomic orbitals (basis set size)
NAO_MAX = 13

# Scaling factor for radius, used for graph construction.
# Suggested scaling factors for different functionals:
# - For HSE: 1.5-2.0, to include long-range interactions. Users should tune this parameter based on their own systems.
# - For PBE: 1.0
RADIUS_SCALE_FACTOR = 1.0

# Flag to skip DFT Hamiltonian (useful for generating graphs for testing)
SKIP_DFT_HAMILTONIAN = False

# Paths for input and output data
# Base directories containing SCF calculations (each dir has OUT.ABACUS/ with
# running_scf.log and sparse CSR files, plus INPUT)
# Default value if not provided via command line
DEFAULT_DATA_DIRS = [f"/data/home/whlu/144-ham_V_Si/scf/{i:04d}" for i in range(1,500)]
DEFAULT_GRAPH_DATA_FOLDER = '../graph/'
DEFAULT_OUTPUT_FORMAT = 'lmdb'
DEFAULT_NUM_PROCESSES = 0
DEFAULT_WORKER_THREADS = 1
DEFAULT_POOL_CHUNKSIZE = 0
DEFAULT_LMDB_COMMIT_INTERVAL = 64
LMDB_OUTPUT_FILENAME = 'graph_data.lmdb'
NPZ_OUTPUT_FILENAME = 'graph_data.npz'
LMDB_INITIAL_MAP_SIZE = 64 * 1024 ** 3
SCF_LOG_FILENAME = "running_scf.log"  # Log filename inside OUT.ABACUS dir

DATA_DIRS = []
GRAPH_DATA_FOLDER = DEFAULT_GRAPH_DATA_FOLDER
OUTPUT_FORMAT = DEFAULT_OUTPUT_FORMAT
NUM_PROCESSES = DEFAULT_NUM_PROCESSES
WORKER_THREADS = DEFAULT_WORKER_THREADS
POOL_CHUNKSIZE = DEFAULT_POOL_CHUNKSIZE
LMDB_COMMIT_INTERVAL = DEFAULT_LMDB_COMMIT_INTERVAL
SCF_OUTPUT_DIRS = []
INPUT_FILE_PATHS = []
_THREADPOOL_LIMITS = None

# Command line argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description='Generate graph data from ABACUS SCF calculations')
    parser.add_argument('--data-dirs', nargs='+', type=str, default=DEFAULT_DATA_DIRS,
                       help='Directories containing SCF calculations (each dir has OUT.ABACUS/ and INPUT)')
    parser.add_argument('--graph-data-folder', type=str, default=DEFAULT_GRAPH_DATA_FOLDER,
                       help='Output folder for graph data. LMDB is the default output format.')
    parser.add_argument('--output-format', choices=('lmdb', 'npz', 'both'), default=DEFAULT_OUTPUT_FORMAT,
                       help='Output format: `lmdb` (default), `npz` for legacy behavior, or `both`.')
    parser.add_argument('--num-processes', type=int, default=DEFAULT_NUM_PROCESSES,
                       help='Number of worker processes. Use 0 (default) to automatically use all available CPU cores.')
    parser.add_argument('--worker-threads', type=int, default=DEFAULT_WORKER_THREADS,
                       help='Max CPU threads used inside each worker process. Default is 1 to avoid oversubscription.')
    parser.add_argument('--chunksize', type=int, default=DEFAULT_POOL_CHUNKSIZE,
                       help='Tasks submitted to each worker per batch. Use 0 (default) to choose automatically.')
    parser.add_argument('--lmdb-commit-interval', type=int, default=DEFAULT_LMDB_COMMIT_INTERVAL,
                       help='Number of graphs buffered before each LMDB write transaction.')
    return parser.parse_args()


def build_runtime_config(parsed_args):
    data_dirs = parsed_args.data_dirs
    scf_output_dirs = [os.path.join(d, 'OUT.ABACUS') for d in data_dirs]
    return {
        'data_dirs': data_dirs,
        'graph_data_folder': parsed_args.graph_data_folder,
        'output_format': parsed_args.output_format,
        'num_processes': parsed_args.num_processes,
        'worker_threads': parsed_args.worker_threads,
        'chunksize': parsed_args.chunksize,
        'lmdb_commit_interval': parsed_args.lmdb_commit_interval,
        'scf_output_dirs': scf_output_dirs,
        'input_file_paths': [os.path.join(d, 'INPUT') for d in scf_output_dirs],
    }


def configure_runtime(runtime_config):
    global DATA_DIRS, GRAPH_DATA_FOLDER, OUTPUT_FORMAT
    global NUM_PROCESSES, WORKER_THREADS, POOL_CHUNKSIZE, LMDB_COMMIT_INTERVAL
    global SCF_OUTPUT_DIRS, INPUT_FILE_PATHS

    DATA_DIRS = runtime_config['data_dirs']
    GRAPH_DATA_FOLDER = runtime_config['graph_data_folder']
    OUTPUT_FORMAT = runtime_config['output_format']
    NUM_PROCESSES = runtime_config['num_processes']
    WORKER_THREADS = runtime_config['worker_threads']
    POOL_CHUNKSIZE = runtime_config['chunksize']
    LMDB_COMMIT_INTERVAL = runtime_config['lmdb_commit_interval']
    SCF_OUTPUT_DIRS = runtime_config['scf_output_dirs']
    INPUT_FILE_PATHS = runtime_config['input_file_paths']


def get_available_cpu_count() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def resolve_num_processes(requested_num_processes: int, total_tasks: int) -> int:
    available_cpu_count = get_available_cpu_count()
    if requested_num_processes <= 0:
        requested_num_processes = available_cpu_count
    return max(1, min(requested_num_processes, available_cpu_count, max(1, total_tasks)))


def resolve_chunksize(total_tasks: int, num_processes: int, requested_chunksize: int) -> int:
    if requested_chunksize > 0:
        return requested_chunksize
    return max(1, total_tasks // (num_processes * 4))


def configure_worker_threads(worker_threads: int) -> None:
    worker_threads = max(1, int(worker_threads))
    try:
        torch.set_num_threads(worker_threads)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    global _THREADPOOL_LIMITS
    try:
        from threadpoolctl import threadpool_limits

        _THREADPOOL_LIMITS = threadpool_limits(limits=worker_threads)
        _THREADPOOL_LIMITS.__enter__()
    except Exception:
        _THREADPOOL_LIMITS = None


def initialize_worker(runtime_config) -> None:
    configure_runtime(runtime_config)
    configure_worker_threads(WORKER_THREADS)


def remove_output_path(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)


class LMDBGraphWriter:
    def __init__(self, lmdb_path: str, map_size: int = LMDB_INITIAL_MAP_SIZE, commit_interval: int = DEFAULT_LMDB_COMMIT_INTERVAL):
        remove_output_path(lmdb_path)
        self.lmdb_path = lmdb_path
        self.map_size = map_size
        self.commit_interval = max(1, int(commit_interval))
        self.count = 0
        self.buffer = []
        self.env = lmdb.open(lmdb_path, map_size=self.map_size, subdir=True, meminit=False)

    def _grow_map_size(self) -> None:
        self.map_size *= 2
        self.env.set_mapsize(self.map_size)
        print(f"LMDB map_size increased to {self.map_size / (1024 ** 3):.1f} GB")

    def _put_items(self, items) -> None:
        while True:
            try:
                with self.env.begin(write=True) as txn:
                    for key, value in items:
                        txn.put(key, value)
                return
            except lmdb.MapFullError:
                self._grow_map_size()

    def flush(self) -> None:
        if not self.buffer:
            return
        self._put_items(self.buffer)
        self.buffer.clear()

    def write_payload(self, payload: bytes) -> None:
        self.buffer.append((f'graph_{self.count}'.encode(), payload))
        self.count += 1
        if len(self.buffer) >= self.commit_interval:
            self.flush()

    def write_graph(self, graph: Data) -> None:
        payload = pickle.dumps(graph, protocol=pickle.HIGHEST_PROTOCOL)
        self.write_payload(payload)

    def finalize(self) -> None:
        self.flush()
        self._put_items([(b'num_graphs', str(self.count).encode())])
        self.env.sync()

    def close(self) -> None:
        if self.env is not None:
            self.flush()
            self.env.close()
            self.env = None

# Derived paths (automatically generated from DATA_DIRS)
SCF_OUTPUT_DIRS = [os.path.join(d, 'OUT.ABACUS') for d in DATA_DIRS]  # Directory containing sparse CSR files
INPUT_FILE_PATHS = [os.path.join(d, 'INPUT') for d in SCF_OUTPUT_DIRS]

# Maximum SCF iterations (to check for convergence)
MAX_SCF_SKIP = 200

# SOC flag (Spin-Orbit Coupling)
SOC_ENABLED = False

# Allowed doping charge range (global graph-level charge)
DOPING_CHARGE_MIN = -8.0
DOPING_CHARGE_MAX = 8.0
################################################################################

# Load basis definitions based on NAO_MAX
if NAO_MAX == 13:
    BASIS_DEF = basis_def_13_abacus
elif NAO_MAX == 15:
    BASIS_DEF = basis_def_15_abacus
elif NAO_MAX == 27:
    BASIS_DEF = basis_def_27_abacus
elif NAO_MAX == 40:
    BASIS_DEF = basis_def_40_abacus
else:
    raise NotImplementedError("Unsupported NAO_MAX value.")

# Cache basis sizes for faster edge tensor expansion
BASIS_NUM = np.zeros((99,), dtype=int)
for k in BASIS_DEF.keys():
    BASIS_NUM[k] = len(BASIS_DEF[k])

@lru_cache(maxsize=None)
def _get_mask_indices(z_src: int, z_tar: int) -> np.ndarray:
    mask = np.zeros((NAO_MAX, NAO_MAX), dtype=bool)
    mask[np.ix_(BASIS_DEF[z_src], BASIS_DEF[z_tar])] = True
    return np.flatnonzero(mask.ravel())

def generate_hamiltonian_and_overlap(graph_h0, graph_h, graph_s, z_indices, basis_definition, nao_max, use_soc=False):
    """
    Generates the Hamiltonian (H), overlap (S), and their spin-orbit coupling (SOC) counterparts for a given system.

    Parameters:
    - graph_h0 (dict): Contains the Hamiltonian terms for the zero-order Hamiltonian, including keys 'Hon' and 'Hoff'.
    - graph_h (dict): Contains the Hamiltonian terms, including keys 'edge_index', 'inv_edge_idx', 'pos', 'Hon', and 'Hoff'.
    - graph_s (dict): Contains the overlap terms, including keys 'Hon' and 'Hoff'.
    - z_indices (list): A list of indices to map basis functions.
    - basis_definition (list): A list of arrays defining the basis functions for the system.
    - nao_max (int): Maximum number of atomic orbitals.
    - use_soc (bool): Flag to indicate whether to include spin-orbit coupling.

    Returns:
    - H (numpy.ndarray): The Hamiltonian matrix.
    - iH (numpy.ndarray or None): The imaginary part of the Hamiltonian if SOC is included, otherwise None.
    - H0 (numpy.ndarray): The zero-order Hamiltonian matrix.
    - iH0 (numpy.ndarray or None): The imaginary part of the zero-order Hamiltonian if SOC is included, otherwise None.
    - S (numpy.ndarray): The overlap matrix.
    """
    try:
        # Extract relevant data from graph_h and graph_s
        edge_index = graph_h['edge_index']
        inv_edge_idx = graph_h['inv_edge_idx']
        pos = graph_h['pos']
        Hon = graph_h['Hon']
        Hoff = graph_h['Hoff']
        Hon0 = graph_h0['Hon']
        Hoff0 = graph_h0['Hoff']
        Son = graph_s['Hon'][0]
        Soff = graph_s['Hoff'][0]

        # Validate edge indices
        if len(inv_edge_idx) != len(edge_index[0]):
            raise ValueError(f"Mismatch in lengths: len(inv_edge_idx) ({len(inv_edge_idx)}) != len(edge_index[0]) ({len(edge_index[0])})")

        # Initialize matrices
        num_sub_matrix = pos.shape[0] + edge_index.shape[1]
        matrix_size = (2 * nao_max if use_soc else nao_max) ** 2
        H = np.zeros((num_sub_matrix, matrix_size), dtype=np.float32)
        H0 = np.zeros((num_sub_matrix, matrix_size), dtype=np.float32)
        iH = np.zeros_like(H, dtype=np.float32) if use_soc else None
        iH0 = np.zeros_like(H0, dtype=np.float32) if use_soc else None
        S = np.zeros((num_sub_matrix, nao_max**2), dtype=np.float32)

        # Fill in on-site terms for Hamiltonian and overlap
        for i, src in enumerate(z_indices):
            if not use_soc:
                mask_idx = _get_mask_indices(int(src), int(src))
                H[i][mask_idx] = Hon[0][i]
                H0[i][mask_idx] = Hon0[0][i]
                S[i][mask_idx] = Son[i]
            else:
                mask = np.zeros((nao_max, nao_max), dtype=int)
                mask[basis_definition[src][:, None], basis_definition[src][None, :]] = 1
                mask = mask.astype(bool)  # Ensure mask is binary
                H[i], iH[i], H0[i], iH0[i] = _fill_soc_terms(
                    H[i], iH[i], H0[i], iH0[i], mask, Hon, Hon0, i
                )
                S[i][mask.flatten()] = Son[i]

        # Fill in off-site terms for Hamiltonian and overlap
        for num, (src, tar) in enumerate(zip(edge_index[0], edge_index[1])):
            if not use_soc:
                mask_idx = _get_mask_indices(int(z_indices[src]), int(z_indices[tar]))
                H[num + len(z_indices)][mask_idx] = Hoff[0][num]
                H0[num + len(z_indices)][mask_idx] = Hoff0[0][num]
                S[num + len(z_indices)][mask_idx] = Soff[num]
            else:
                mask = np.zeros((nao_max, nao_max), dtype=int)
                mask[basis_definition[z_indices[src]][:, None], basis_definition[z_indices[tar]][None, :]] = 1
                mask = mask.astype(bool)
                H[num + len(z_indices)], iH[num + len(z_indices)], H0[num + len(z_indices)], iH0[num + len(z_indices)] = _fill_soc_terms(
                    H[num + len(z_indices)], iH[num + len(z_indices)], H0[num + len(z_indices)], iH0[num + len(z_indices)], mask, Hoff, Hoff0, num
                )
                S[num + len(z_indices)][mask.flatten()] = Soff[num]

        # Return the computed matrices
        if use_soc:
            return H, iH, H0, iH0
        else:
            return H, H0, S

    except Exception as e:
        print(f"Error generating Hamiltonian and overlap matrices: {e}")
        if use_soc:
            return None, None, None, None
        else:
            return None, None, None


def _fill_soc_terms(H, iH, H0, iH0, mask, graph_hon, graph_hon0, index):
    """
    Helper function to fill in Hamiltonian and overlap matrices for spin-orbit coupling (SOC) terms.
    
    Parameters:
    - H, iH, H0, iH0 (numpy.ndarray): The matrices to be populated.
    - mask (numpy.ndarray): A boolean mask to indicate the positions to populate.
    - graph_hon, graph_hon0 (list): Lists of Hamiltonian terms, including spin components.
    - index (int): The index of the current element to access the Hamiltonian terms.

    Returns:
    - Updated matrices (H, iH, H0, iH0).
    """
    tH = np.zeros((2 * H.shape[0], 2 * H.shape[0]), dtype=np.complex64)
    
    # Populate the Hamiltonian matrix with SOC terms
    tH[:H.shape[0], :H.shape[0]][mask] = graph_hon[0][index]  # uu
    tH[:H.shape[0], H.shape[0]:][mask] = graph_hon[1][index]  # ud
    tH[H.shape[0]:, :H.shape[0]][mask] = graph_hon[2][index]  # du
    tH[H.shape[0]:, H.shape[0]:][mask] = graph_hon[3][index]  # dd
    H = tH.real.flatten()
    iH = tH.imag.flatten()

    # Populate the zero-order Hamiltonian with SOC terms
    tH[:H.shape[0], :H.shape[0]][mask] = graph_hon0[0][index]  # uu
    tH[:H.shape[0], H.shape[0]:][mask] = graph_hon0[1][index]  # ud
    tH[H.shape[0]:, :H.shape[0]][mask] = graph_hon0[2][index]  # du
    tH[H.shape[0]:, H.shape[0]:][mask] = graph_hon0[3][index]  # dd
    H0 = tH.real.flatten()
    iH0 = tH.imag.flatten()

    return H, iH, H0, iH0


def generate_expanded_graph_h0(atomic_numbers, lattice, pos, graph_h0, soc_enabled=False, radius_type='abacus', radius_scale=1.5):
    """
    Generates an expanded graph structure by adjusting the edge indices, cell shifts, and tensors.

    Parameters:
    - atomic_numbers (list): A list of atomic numbers.
    - lattice (array-like): The lattice structure of the material.
    - pos (array-like): The positions of the atoms in the material.
    - graph_h0 (dict): The initial graph data, including edge indices, cell shifts, and tensors.
    - soc_enabled (bool): A flag to enable or disable Spin-Orbit Coupling (SOC) computation (default is False).
    - radius_type (str): The type of radius used for graph construction (default is 'abacus').
    - radius_scale (float): The scaling factor applied to the radius (default is 1.5).

    Returns:
    - dict: The updated graph_h0 with expanded edge indices, cell shifts, and tensors.
    """
    
    # Build the graph using the specified radius type, scale, and atomic information
    graph_ref = build_graph(radius_type=radius_type, radius_scale=radius_scale, 
                            atomic_numbers=atomic_numbers, lattice=lattice, positions=pos)

    # Select tensors to expand based on SOC_ENABLED flag
    tensors_to_expand = [graph_h0['Hoff']] + ([graph_h0['iHoff']] if soc_enabled else [])
    
    # Expand the graph by adjusting the edge indices, cell shifts, and tensors
    edge_indices_exp, cell_shifts_exp, nbr_shifts_exp, inv_edge_idx_exp, tensors_expanded = expand_graph(
        lattice=lattice,
        edge_indices_1=graph_ref['edge_index'], 
        cell_shifts_1=graph_ref['cell_shift'], 
        edge_indices_2=graph_h0['edge_index'], 
        cell_shifts_2=graph_h0['cell_shift'], 
        nbr_shifts_2=graph_h0['nbr_shift'], 
        inv_edge_idx_2=graph_h0['inv_edge_idx'],
        atomic_numbers=atomic_numbers,
        tensors_to_expand=tensors_to_expand,
        soc_switch=soc_enabled
    )

    # Update graph_h0 with the expanded data
    graph_h0.update({
        'edge_index': edge_indices_exp,
        'cell_shift': cell_shifts_exp,
        'nbr_shift': nbr_shifts_exp,
        'inv_edge_idx': inv_edge_idx_exp,
    })
    
    # Handle the tensors for SOC or non-SOC cases
    if soc_enabled:
        graph_h0['Hoff'], graph_h0['iHoff'] = tensors_expanded
    else:
        graph_h0['Hoff'] = tensors_expanded[0]

    return graph_h0


def expand_graph(lattice, edge_indices_1, cell_shifts_1, edge_indices_2, cell_shifts_2, nbr_shifts_2, inv_edge_idx_2, atomic_numbers, tensors_to_expand, soc_switch):
    """
    Expands the graph by adding edges, cell shifts, and tensors from the difference between two graphs.

    This function calculates the difference in edges and cell shifts between two graphs, then
    expands the graph by adding the new edges, shifts, and expanding the associated tensors.
    
    Parameters:
    -----------
    lattice : np.ndarray
        A matrix representing the lattice used for periodic boundary conditions (shape: (3, 3)).
        
    edge_indices_1 : np.ndarray
        A 2xN numpy array of edge indices for the first graph (shape: (2, n_edges_1)).
    
    cell_shifts_1 : np.ndarray
        A Nx3 numpy array of cell shifts corresponding to the edges in edge_indices_1 (shape: (n_edges_1, 3)).
    
    edge_indices_2 : np.ndarray
        A 2xM numpy array of edge indices for the second graph (shape: (2, n_edges_2)).
    
    cell_shifts_2 : np.ndarray
        A Mx3 numpy array of cell shifts corresponding to the edges in edge_indices_2 (shape: (n_edges_2, 3)).
    
    inv_edge_idx2 : np.ndarray
        A numpy array containing the inverse edge indices for the second graph (shape: (n_edges_2,)).
    
    tensors_to_expand : list of np.ndarray
        A list of tensors to be expanded, where each tensor has at least two dimensions.
    
    soc_switch : bool
        A flag that enables or disables the SOC (Spin-Orbit Coupling) calculations. If True, SOC is enabled.

    Returns:
    --------
    edge_indices_exp : np.ndarray
        A 2x(N+M) numpy array of the expanded edge indices after combining the two graphs.
    
    cell_shifts_exp : np.ndarray
        A (N+M)x3 numpy array of the expanded cell shifts corresponding to the new edge indices.
    
    inv_edge_idx_exp : np.ndarray
        A numpy array of the expanded inverse edge indices for the graph.
    
    tensors_expanded : list of np.ndarray
        A list of tensors with expanded shapes to accommodate the new edges.
    """
    
    # Compute the difference in edges and cell shifts between the two graphs (new edges to add)
    edge_indices_diff, cell_shifts_diff = compute_graph_difference(edge_indices_1, cell_shifts_1, edge_indices_2, cell_shifts_2)
    
    # Find the inverse edge indices for the new edges
    inv_edge_idx_diff = find_inverse_edge_index(edge_indices_diff, cell_shifts_diff) + len(edge_indices_2[0])
    inv_edge_idx_exp = np.concatenate([inv_edge_idx_2, inv_edge_idx_diff], axis=0)
    
    # Compute the neighbor shifts using lattice matrix for PBC correction
    nbr_shifts_diff = np.einsum('ni, ij -> nj', cell_shifts_diff, lattice)
    nbr_shifts_exp = np.concatenate([nbr_shifts_2, nbr_shifts_diff], axis=0)
    
    # Number of new edges to expand
    num_edges_diff = len(edge_indices_diff[0])
    
    # Concatenate the existing and new edge indices, and the cell shifts
    edge_indices_exp = np.concatenate([edge_indices_2, edge_indices_diff], axis=-1)
    cell_shifts_exp = np.concatenate([cell_shifts_2, cell_shifts_diff], axis=0)  
    
    src_diff, dst_diff = atomic_numbers[edge_indices_diff] 
    num_orbs_edge_diff = BASIS_NUM[src_diff]*BASIS_NUM[dst_diff]
    
    # Expand the tensors by adding the new edges
    tensors_expanded = []
    for tensor in tensors_to_expand:
        # Calculate the new size for each tensor based on the new edge indices
        for iedge in range(num_edges_diff):
            new_tensor_values = np.array(num_orbs_edge_diff[iedge] * [0.0])

            if soc_switch:
                # If SOC is enabled, expand all 4 tensor components
                for i in range(4):
                    tensor[i] += [new_tensor_values]
            else:
                # If SOC is not enabled, expand only the first tensor component
                tensor[0] += [new_tensor_values]
        
        tensors_expanded.append(tensor)
    
    return edge_indices_exp, cell_shifts_exp, nbr_shifts_exp, inv_edge_idx_exp, tensors_expanded


def generate_graph(task: tuple) -> tuple:
    """
    Generates graph data for a given SCF calculation.

    Args:
        task (tuple): (index, scf_path) pair.

    Returns:
        tuple: (index, success_flag, graph_data, serialized_graph) where:
               - index (int): Index of the SCF calculation.
               - success_flag (bool): Indicates if graph generation was successful.
               - graph_data (torch_geometric.data.Data): Graph object with properties.
               - serialized_graph (bytes): Pickled graph payload for LMDB-only output.
    """
    idx, scf_path = task
    # Define paths for the required files
    scf_log_path = os.path.join(scf_path, SCF_LOG_FILENAME)

    # Read energy and SCF iteration data
    if SKIP_DFT_HAMILTONIAN:
        energy = 0.0
        max_scf_iterations = 0
    else:
        try:
            with open(scf_log_path, 'r') as f:
                log_content = f.read().strip()
                energy = float(pattern_eng_abacus.findall(log_content)[0])
                max_scf_iterations = int(pattern_md_abacus.findall(log_content)[-1])
        except Exception as e:
            print(f"Error reading SCF log file: {e}. Skipping...")
            return idx, False, None, None

    # Check SCF convergence
    if max_scf_iterations >= MAX_SCF_SKIP:
        print("Error: SCF did not converge. Skipping...")
        return idx, False, None, None

    # Read crystal structure parameters
    try:
        crystal = STRU(scf_log_path)
        lattice = crystal.cell
        atomic_numbers = crystal.atomic_numbers.astype(int)
        
        # Calculate doping charge from INPUT file
        input_file_path = INPUT_FILE_PATHS[idx]
        input_params = read_abacus_input(input_file_path)
        neutral_electrons = get_neutral_electrons(crystal)
        doping_charge = calculate_doping_charge(input_params, neutral_electrons)
        if not DOPING_CHARGE_MIN <= doping_charge <= DOPING_CHARGE_MAX:
            raise ValueError(
                f"doping_charge {doping_charge} is out of allowed range "
                f"[{DOPING_CHARGE_MIN}, {DOPING_CHARGE_MAX}] for {input_file_path}"
            )
        doping_charge_tensor = torch.tensor([doping_charge], dtype=torch.float32)
        
    except Exception as e:
        print(f"Error reading structure from SCF log or calculating doping charge: {e}. Skipping...")
        return idx, False, None, None

    # Read hopping and overlap parameters
    try:
        # Load sparse Hamiltonian and overlap matrices
        h0_sparse = ABACUSHS(os.path.join(scf_path, 'data-H0R-sparse_SPIN0.csr'))
        if SKIP_DFT_HAMILTONIAN:
            h_sparse = None
        else:
            h_sparse = ABACUSHS(os.path.join(scf_path, 'data-HR-sparse_SPIN0.csr'))
        s_sparse = ABACUSHS(os.path.join(scf_path, 'data-S0R-sparse_SPIN0.csr'))

        # Generate graphs for Hamiltonian and overlap     
        graph_h0 = h0_sparse.getGraph(crystal, graph={}, isH=True, isSOC=SOC_ENABLED)
        graph_h0 = generate_expanded_graph_h0(atomic_numbers, lattice, crystal.positions, graph_h0, soc_enabled=SOC_ENABLED, radius_type='abacus', radius_scale=RADIUS_SCALE_FACTOR)
        if SKIP_DFT_HAMILTONIAN:
            graph_h = graph_h0
        else:
            graph_h = h_sparse.getGraph(crystal, graph=graph_h0, isH=True, calcRcut=True, isSOC=SOC_ENABLED, skip=True)
        graph_s = s_sparse.getGraph(crystal, graph=graph_h, skip=True, isSOC=SOC_ENABLED)

        # Extract graph properties
        pos = graph_h['pos']
        edge_index = graph_h['edge_index']
        inv_edge_idx = graph_h['inv_edge_idx']
        nbr_shift = graph_h['nbr_shift']
        cell_shift = graph_h['cell_shift']

        # Close file handles
        h0_sparse.close()
        if not SKIP_DFT_HAMILTONIAN:
            h_sparse.close()
        s_sparse.close()
    except Exception as e:
        print(f"Error reading Hamiltonian or overlap matrices: {e}. Skipping...")
        return idx, False, None, None

    # Prepare Hamiltonian and overlap matrices
    try:
        if SOC_ENABLED:
            H, iH, H0, iH0 = generate_hamiltonian_and_overlap(graph_h0, graph_h, graph_s, atomic_numbers, BASIS_DEF, NAO_MAX, use_soc=SOC_ENABLED)
        else:
            H, H0, S = generate_hamiltonian_and_overlap(graph_h0, graph_h, graph_s, atomic_numbers, BASIS_DEF, NAO_MAX, use_soc=SOC_ENABLED)
    except Exception as e:
        print(f"Error preparing Hamiltonian or overlap matrices: {e}. Skipping...")
        return idx, False, None, None

    # Create a graph data object

    # save in Data
    if not SOC_ENABLED:
        graph = Data(z=torch.LongTensor(atomic_numbers),
                    cell = torch.Tensor(lattice[None,:,:]),
                    total_energy = torch.Tensor([energy]),
                    pos=torch.FloatTensor(pos),
                    node_counts=torch.LongTensor([len(atomic_numbers)]),
                    edge_index=torch.LongTensor(edge_index),
                    inv_edge_idx=torch.LongTensor(inv_edge_idx),
                    nbr_shift=torch.FloatTensor(nbr_shift),
                    cell_shift=torch.LongTensor(cell_shift),
                    hamiltonian=torch.FloatTensor(H),
                    overlap=torch.FloatTensor(S),
                    Hon = torch.FloatTensor(H[:pos.shape[0],:]),
                    Hoff = torch.FloatTensor(H[pos.shape[0]:,:]),
                    Hon0 = torch.FloatTensor(H0[:pos.shape[0],:]),
                    Hoff0 = torch.FloatTensor(H0[pos.shape[0]:,:]),
                    Son = torch.FloatTensor(S[:pos.shape[0],:]),
                    Soff = torch.FloatTensor(S[pos.shape[0]:,:]),
                    doping_charge=doping_charge_tensor)
    else:
        graph = Data(z=torch.LongTensor(atomic_numbers),
                    cell = torch.Tensor(lattice[None,:,:]),
                    total_energy = torch.Tensor([energy]),
                    pos=torch.FloatTensor(pos),
                    node_counts=torch.LongTensor([len(atomic_numbers)]),
                    edge_index=torch.LongTensor(edge_index),
                    inv_edge_idx=torch.LongTensor(inv_edge_idx),
                    nbr_shift=torch.FloatTensor(nbr_shift),
                    cell_shift=torch.LongTensor(cell_shift),
                    overlap=torch.FloatTensor(S),
                    Hon = torch.FloatTensor(H[:pos.shape[0],:]),
                    Hoff = torch.FloatTensor(H[pos.shape[0]:,:]),
                    iHon = torch.FloatTensor(iH[:pos.shape[0],:]),
                    iHoff = torch.FloatTensor(iH[pos.shape[0]:,:]),
                    Hon0 = torch.FloatTensor(H0[:pos.shape[0],:]),
                    Hoff0 = torch.FloatTensor(H0[pos.shape[0]:,:]),
                    iHon0 = torch.FloatTensor(iH0[:pos.shape[0],:]),
                    iHoff0 = torch.FloatTensor(iH0[pos.shape[0]:,:]),
                    Son = torch.FloatTensor(S[:pos.shape[0],:]),
                    Soff = torch.FloatTensor(S[pos.shape[0]:,:]),
                    doping_charge=doping_charge_tensor)

    if OUTPUT_FORMAT == 'lmdb':
        return idx, True, None, pickle.dumps(graph, protocol=pickle.HIGHEST_PROTOCOL)
    return idx, True, graph, None


def main():
    """
    Main function to generate graphs for all SCF calculations and save results.
    """
    args = parse_args()
    runtime_config = build_runtime_config(args)
    configure_runtime(runtime_config)

    multiprocessing.freeze_support()
    tasks = list(enumerate(SCF_OUTPUT_DIRS))
    total_tasks = len(tasks)
    num_processes = resolve_num_processes(NUM_PROCESSES, total_tasks)
    chunksize = resolve_chunksize(total_tasks, num_processes, POOL_CHUNKSIZE)
    os.makedirs(GRAPH_DATA_FOLDER, exist_ok=True)

    save_npz = OUTPUT_FORMAT in ('npz', 'both')
    save_lmdb = OUTPUT_FORMAT in ('lmdb', 'both')
    graph_data = {} if save_npz else None
    lmdb_path = os.path.join(GRAPH_DATA_FOLDER, LMDB_OUTPUT_FILENAME)
    lmdb_writer = LMDBGraphWriter(lmdb_path, commit_interval=LMDB_COMMIT_INTERVAL) if save_lmdb else None
    saved_graphs = 0

    print(f'Processing {total_tasks} SCF outputs with {num_processes} worker(s), worker_threads={max(1, WORKER_THREADS)}, chunksize={chunksize}.')

    def handle_graph_result(success: bool, graph: Data = None, payload: bytes = None) -> None:
        nonlocal saved_graphs
        if not success:
            return
        if graph_data is not None:
            graph_data[saved_graphs] = graph
        if lmdb_writer is not None:
            if payload is not None:
                lmdb_writer.write_payload(payload)
            else:
                lmdb_writer.write_graph(graph)
        saved_graphs += 1

    try:
        if num_processes <= 1:
            for task in tqdm(tasks, desc="Processing SCF Outputs", total=total_tasks):
                _, success, graph, payload = generate_graph(task)
                handle_graph_result(success, graph, payload)
        else:
            with multiprocessing.Pool(
                processes=num_processes,
                initializer=initialize_worker,
                initargs=(runtime_config,),
            ) as pool:
                for _, success, graph, payload in tqdm(
                    pool.imap_unordered(generate_graph, tasks, chunksize=chunksize),
                    desc="Processing SCF Outputs",
                    total=total_tasks,
                ):
                    handle_graph_result(success, graph, payload)

        if saved_graphs == 0:
            print('No valid data found! Please check the input paths or if the DFT calculations are converged.')
            return

        if graph_data is not None:
            graph_data_path = os.path.join(GRAPH_DATA_FOLDER, NPZ_OUTPUT_FILENAME)
            np.savez(graph_data_path, graph=graph_data)
            print(f'Saved {saved_graphs} graphs to {graph_data_path}')

        if lmdb_writer is not None:
            lmdb_writer.finalize()
            print(f'Saved {lmdb_writer.count} graphs to {lmdb_path}')
    finally:
        if lmdb_writer is not None:
            lmdb_writer.close()


if __name__ == "__main__":

    main()
