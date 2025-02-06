'''
Descripttion: 
version: 
Author: Yang Zhong
Date: 2024-01-16 13:00:43
Last Modified by:   Yang Zhong
Last Modified time: 2025-02-6 10:34:01 
'''
import os
import sys
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm
from copy import deepcopy
import multiprocessing
from pymatgen.core.periodic_table import Element
from read_abacus import STRU, ABACUSHS
from build_graph_from_coordinates import build_graph, compute_graph_difference, find_inverse_edge_index
from utils import *

################################ Input Parameters ##############################
# Maximum number of atomic orbitals (basis set size)
NAO_MAX = 27

# Scaling factor for radius, used for graph construction.
# Suggested scaling factors for different functionals:
# - For HSE: 1.5-2.0, to include long-range interactions. Users should tune this parameter based on their own systems.
# - For PBE: 1.0
RADIUS_SCALE_FACTOR = 1.8

# Flag to skip DFT Hamiltonian (useful for generating graphs for testing)
SKIP_DFT_HAMILTONIAN = True

# Paths for input and output data
GRAPH_DATA_FOLDER = '../graph/'
SCF_OUTPUT_PATHS = [f"/public/home/zhongyang/yzhong/Abacus_test/lcao_Si2/OUT.ABACUS"]
STRU_FILE_PATHS = [f"/public/home/zhongyang/yzhong/Abacus_test/lcao_Si2/STRU"]
SCF_LOG_FILENAME = "running_scf.log" # if SKIP_DFT_HAMILTONIAN is True, this file is not used

# Maximum SCF iterations (to check for convergence)
MAX_SCF_SKIP = 200

# SOC flag (Spin-Orbit Coupling)
SOC_ENABLED = False

# Number of processes for parallelization
NUM_PROCESSES = 1
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

# Create output folder if it doesn't exist
if not os.path.exists(GRAPH_DATA_FOLDER):
    os.makedirs(GRAPH_DATA_FOLDER)

# Ensure the number of SCF output paths matches the STRU file paths
if len(SCF_OUTPUT_PATHS) != len(STRU_FILE_PATHS):
    raise ValueError("Mismatch between SCF output paths and STRU file paths.")

# Dictionary to store graph data
graph_data = {}


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
            mask = np.zeros((nao_max, nao_max), dtype=int)
            mask[basis_definition[src][:, None], basis_definition[src][None, :]] = 1
            mask = mask.astype(bool)  # Ensure mask is binary

            # Populate matrices based on whether SOC is used
            if not use_soc:
                H[i][mask.flatten()] = Hon[0][i]
                H0[i][mask.flatten()] = Hon0[0][i]
            else:
                H[i], iH[i], H0[i], iH0[i] = _fill_soc_terms(
                    H[i], iH[i], H0[i], iH0[i], mask, Hon, Hon0, i
                )

            # Fill in overlap matrix
            S[i][mask.flatten()] = Son[i]

        # Fill in off-site terms for Hamiltonian and overlap
        for num, (src, tar) in enumerate(zip(edge_index[0], edge_index[1])):
            mask = np.zeros((nao_max, nao_max), dtype=int)
            mask[basis_definition[z_indices[src]][:, None], basis_definition[z_indices[tar]][None, :]] = 1
            mask = mask.astype(bool)

            # Populate matrices based on whether SOC is used
            if not use_soc:
                H[num + len(z_indices)][mask.flatten()] = Hoff[0][num]
                H0[num + len(z_indices)][mask.flatten()] = Hoff0[0][num]
            else:
                H[num + len(z_indices)], iH[num + len(z_indices)], H0[num + len(z_indices)], iH0[num + len(z_indices)] = _fill_soc_terms(
                    H[num + len(z_indices)], iH[num + len(z_indices)], H0[num + len(z_indices)], iH0[num + len(z_indices)], mask, Hoff, Hoff0, num
                )

            # Fill in overlap matrix for off-site terms
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
    
    # The number of bases of each species
    BASIS_NUM = np.zeros((99,), dtype=int)
    for k in BASIS_DEF.keys():
        BASIS_NUM[k] = len(BASIS_DEF[k])
    
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


def generate_graph(idx: int, scf_path: str) -> tuple:
    """
    Generates graph data for a given SCF calculation.

    Args:
        idx (int): Index of the SCF calculation.
        scf_path (str): Path to the SCF output folder.

    Returns:
        tuple: (success_flag, max_rcut, graph_data) where:
               - success_flag (bool): Indicates if graph generation was successful.
               - max_rcut (float): Maximum cutoff radius used in the calculation.
               - graph_data (torch_geometric.data.Data): Graph object with properties.
    """
    # Define paths for the required files
    scf_log_path = os.path.join(scf_path, SCF_LOG_FILENAME)
    stru_file_path = STRU_FILE_PATHS[idx]

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
            return False, None, None

    # Check SCF convergence
    if max_scf_iterations >= MAX_SCF_SKIP:
        print("Error: SCF did not converge. Skipping...")
        return False, None, None

    # Read crystal structure parameters
    try:
        crystal = STRU(stru_file_path)
        lattice = crystal.cell
        atomic_numbers = []
        for species, atom_count in zip(crystal.species, crystal.num_atoms_per_species):
            atomic_numbers += [Element(species).Z] * atom_count
        atomic_numbers = np.array(atomic_numbers, dtype=int)
    except Exception as e:
        print(f"Error reading STRU file: {e}. Skipping...")
        return False, None, None

    # Read hopping and overlap parameters
    try:
        # Load sparse Hamiltonian and overlap matrices
        h0_sparse = ABACUSHS(os.path.join(scf_path, 'data-H0R-sparse_SPIN0.csr'))
        if SKIP_DFT_HAMILTONIAN:
            h_sparse = None
        else:
            h_sparse = ABACUSHS(os.path.join(scf_path, 'data-HR-sparse_SPIN0.csr'))
        s_sparse = ABACUSHS(os.path.join(scf_path, 'data-SR-sparse_SPIN0.csr'))

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
        return False, None, None

    # Prepare Hamiltonian and overlap matrices
    try:
        if SOC_ENABLED:
            H, iH, H0, iH0 = generate_hamiltonian_and_overlap(graph_h0, graph_h, graph_s, atomic_numbers, BASIS_DEF, NAO_MAX, use_soc=SOC_ENABLED)
        else:
            H, H0, S = generate_hamiltonian_and_overlap(graph_h0, graph_h, graph_s, atomic_numbers, BASIS_DEF, NAO_MAX, use_soc=SOC_ENABLED)
    except Exception as e:
        print(f"Error preparing Hamiltonian or overlap matrices: {e}. Skipping...")
        return False, None, None

    # Create a graph data object

    # save in Data
    if not SOC_ENABLED:
        return True, Data(z=torch.LongTensor(atomic_numbers),
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
                        Soff = torch.FloatTensor(S[pos.shape[0]:,:]))
    else:
        return True, Data(z=torch.LongTensor(atomic_numbers),
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
                        Soff = torch.FloatTensor(S[pos.shape[0]:,:]))


def main():
    """
    Main function to generate graphs for all SCF calculations and save results.
    """
    multiprocessing.freeze_support()
    num_processes = min(multiprocessing.cpu_count(), NUM_PROCESSES)
    pool = multiprocessing.Pool(processes=num_processes)

    # Initialize cutoff radii
    crystal = STRU(STRU_FILE_PATHS[0])
    max_rcut = np.zeros([len(crystal.species), len(crystal.species)])
    min_rcut = np.zeros_like(max_rcut)
    for i, spec1 in enumerate(crystal.species):
        for j, spec2 in enumerate(crystal.species):
            min_rcut[i, j] = RCUT_dict[spec1] + RCUT_dict[spec2]

    # Process all SCF calculations
    results = []
    for idx, scf_path in enumerate(SCF_OUTPUT_PATHS):
        results.append(pool.apply_async(generate_graph, (idx, scf_path)))

    for idx, result in enumerate(tqdm(results, desc="Processing SCF Outputs")):
        success, graph = result.get()
        if success:
            graph_data[idx] = graph

    pool.close()
    pool.join()

    # Save graph data and cutoff radii
    graph_data_path = os.path.join(GRAPH_DATA_FOLDER, 'graph_data.npz')
    np.savez(graph_data_path, graph=graph_data)


if __name__ == "__main__":
    main()