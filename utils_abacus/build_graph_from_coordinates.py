'''
Descripttion: 
version: 
Author: Yang Zhong
Date: 2024-08-24 14:22:19
LastEditors: Yang Zhong
LastEditTime: 2024-08-24 21:05:41
'''

from __future__ import annotations

import torch
import torch.nn as nn
from torch_scatter import scatter
from easydict import EasyDict

import warnings
from ase import geometry, neighborlist
import numpy as np
from pymatgen.core.periodic_table import Element
from typing import List, Union

ATOMIC_RADII = {
    'openmx': {
        'H': 6.0, 'He': 8.0, 'Li': 8.0, 'Be': 7.0, 'B': 7.0, 'C': 6.0,
        'N': 6.0, 'O': 6.0, 'F': 6.0, 'Ne': 9.0, 'Na': 9.0, 'Mg': 9.0,
        'Al': 7.0, 'Si': 7.0, 'P': 7.0, 'S': 7.0, 'Cl': 7.0, 'Ar': 9.0,
        'K': 10.0, 'Ca': 9.0, 'Sc': 9.0, 'Ti': 7.0, 'V': 6.0, 'Cr': 6.0,
        'Mn': 6.0, 'Fe': 5.5, 'Co': 6.0, 'Ni': 6.0, 'Cu': 6.0, 'Zn': 6.0,
        'Ga': 7.0, 'Ge': 7.0, 'As': 7.0, 'Se': 7.0, 'Br': 7.0, 'Kr': 10.0,
        'Rb': 11.0, 'Sr': 10.0, 'Y': 10.0, 'Zr': 7.0, 'Nb': 7.0, 'Mo': 7.0,
        'Tc': 7.0, 'Ru': 7.0, 'Rh': 7.0, 'Pd': 7.0, 'Ag': 7.0, 'Cd': 7.0,
        'In': 7.0, 'Sn': 7.0, 'Sb': 7.0, 'Te': 7.0, 'I': 7.0, 'Xe': 11.0,
        'Cs': 12.0, 'Ba': 10.0, 'La': 8.0, 'Ce': 8.0, 'Pr': 8.0, 'Nd': 8.0,
        'Pm': 8.0, 'Sm': 8.0, 'Dy': 8.0, 'Ho': 8.0, 'Lu': 8.0, 'Hf': 9.0,
        'Ta': 7.0, 'W': 7.0, 'Re': 7.0, 'Os': 7.0, 'Ir': 7.0, 'Pt': 7.0,
        'Au': 7.0, 'Hg': 8.0, 'Tl': 8.0, 'Pb': 8.0, 'Bi': 8.0
    },
    'abacus': {
        'Ag':7,  'Cu':8,  'Mo':7,  'Sc':8,
        'Al':7,  'Fe':8,  'Na':8,  'Se':8,
        'Ar':7,  'F' :7,  'Nb':8,  'S' :7,
        'As':7,  'Ga':8,  'Ne':6,  'Si':7,
        'Au':7,  'Ge':8,  'N' :7,  'Sn':7,
        'Ba':10, 'He':6,  'Ni':8,  'Sr':9,
        'Be':7,  'Hf':7,  'O' :7,  'Ta':8,
        'B' :8,  'H' :6,  'Os':7,  'Tc':7,
        'Bi':7,  'Hg':9,  'Pb':7,  'Te':7,
        'Br':7,  'I' :7,  'Pd':7,  'Ti':8,
        'Ca':9,  'In':7,  'P' :7,  'Tl':7,
        'Cd':7,  'Ir':7,  'Pt':7,  'V' :8,
        'C' :7,  'K' :9,  'Rb':10, 'W' :8,
        'Cl':7,  'Kr':7,  'Re':7,  'Xe':8,
        'Co':8,  'Li':7,  'Rh':7,  'Y' :8,
        'Cr':8,  'Mg':8,  'Ru':7,  'Zn':8,
        'Cs':10, 'Mn':8,  'Sb':7,  'Zr':8
    }
}

DEFAULT_RADIUS = 10.0

def get_radii_from_atomic_numbers(atomic_numbers: Union[torch.Tensor, List[int]], 
                                  radius_scale: float = 1.5, radius_type: str = 'openmx') -> List[float]:
    """
    Retrieves the scaled atomic radii for a given list or tensor of atomic numbers.

    Parameters:
    - atomic_numbers (Union[torch.Tensor, List[int]]): A list or tensor containing atomic numbers.
    - radius_scale (float): A scaling factor to multiply the atomic radii. Default is 1.5.
    - radius_type (str): The software, in which the atomic radius is utilized, originates from a specific source. Default is openmx.

    Returns:
    - List[float]: A list of scaled atomic radii corresponding to the input atomic numbers.
    """
    
    if isinstance(atomic_numbers, torch.Tensor):
        atomic_numbers = atomic_numbers.tolist()

    # Convert atomic numbers to element symbols and then to scaled radii.
    # Use 0.0 as a default value for elements not found in the dictionary.
    return [radius_scale * ATOMIC_RADII[radius_type].get(Element.from_Z(z).symbol, DEFAULT_RADIUS) for z in atomic_numbers]


def create_neighbor_list_and_vectors(
    positions,
    max_radius,
    include_self_interaction=False,
    strict_self_interaction=True,
    cell_matrix=None,
    apply_pbc=False,
):
    """
    Create a neighbor list and corresponding relative position vectors based on a radial cutoff distance.

    Args:
        positions (torch.Tensor or np.ndarray): The positions of the particles (shape: (n_particles, 3)).
        max_radius (float): The maximum cutoff distance for neighbors.
        include_self_interaction (bool): Whether to include self-interactions. Default is False.
        strict_self_interaction (bool): If True, self-interactions are strictly excluded (based on exact position and no shift). Default is True.
        cell_matrix (torch.Tensor or np.ndarray, optional): The cell matrix for periodic boundary conditions (shape: (3, 3)).
        apply_pbc (bool or tuple): If True, applies periodic boundary conditions in all directions. 
                                    If tuple, specifies the PBC in each of the x, y, z directions.

    Returns:
        np.ndarray: The indices of the neighboring particles (shape: (2, n_neighbors)).
        np.ndarray: The relative shift vectors for each neighbor pair (shape: (n_neighbors, 3)).
    """
    # Ensure PBC is in the correct format (tuple of 3 boolean values)
    if isinstance(apply_pbc, bool):
        apply_pbc = (apply_pbc,) * 3

    # Handle input positions (convert to numpy if necessary)
    if isinstance(positions, torch.Tensor):
        positions = positions.detach().cpu().numpy()
    else:
        positions = np.asarray(positions)

    # Handle cell matrix input (convert to numpy if necessary)
    if isinstance(cell_matrix, torch.Tensor):
        cell_matrix = cell_matrix.detach().cpu().numpy()
    elif cell_matrix is not None:
        cell_matrix = np.asarray(cell_matrix)
    else:
        # Default to identity matrix if no cell matrix is provided
        cell_matrix = np.zeros((3, 3), dtype=positions.dtype)

    # Complete the cell matrix (if applicable) to handle PBC
    cell_matrix = geometry.complete_cell(cell_matrix)

    # Generate the raw neighbor list (pairs of indices and shift vectors)
    first_idx, second_idx, shifts = neighborlist.primitive_neighbor_list(
        "ijS", 
        apply_pbc, 
        cell_matrix, 
        positions, 
        cutoff=max_radius, 
        self_interaction=strict_self_interaction, 
        use_scaled_positions=False
    )

    # Remove self-interactions if required
    if not include_self_interaction:
        # Identify self-edges (same index and no shift)
        self_edges = (first_idx == second_idx) & np.all(shifts == 0, axis=1)
        valid_edges = ~self_edges
        
        # If no valid edges remain, raise an error
        if not np.any(valid_edges):
            raise ValueError("No edges remain after eliminating self-edges.")
        
        # Filter out invalid edges
        first_idx = first_idx[valid_edges]
        second_idx = second_idx[valid_edges]
        shifts = shifts[valid_edges]

    # Stack the edge indices into a 2xN array
    edge_indices = np.vstack(
        (np.array(first_idx, dtype=np.int64), np.array(second_idx, dtype=np.int64))
    )

    return edge_indices, shifts


def find_inverse_edge_index(edge_index, cell_shifts):
    """
    Find the indices of inverse edges in a given set of edges with associated cell shifts.

    This function takes a list of edges (defined by their start and end nodes) and a list of 
    corresponding periodic cell shifts. It returns an array of indices where each entry 
    contains the index of the inverse edge for the corresponding edge. If no inverse edge is found,
    the index will be -1.

    Parameters:
    -----------
    edge_index : np.ndarray
        A 2xN numpy array where the first row contains the start nodes of edges, and the second 
        row contains the end nodes of edges.

    cell_shifts : np.ndarray
        A Nx3 numpy array where each row represents a periodic shift vector associated with the 
        corresponding edge in edge_index.

    Returns:
    --------
    inverse_indices : np.ndarray
        A 1D numpy array where each element is the index of the inverse edge for the corresponding 
        edge. If an inverse edge does not exist, the entry is -1.
    """
    
    # Combine edge information with cell shifts and ensure the shifts are tuples (for immutability)
    edges_with_shifts = list(zip(edge_index[0], edge_index[1], map(tuple, cell_shifts)))
    
    # Create dictionaries to map edges and inverse edges to their indices
    edge_to_index = {edge: idx for idx, edge in enumerate(edges_with_shifts)}
    inverse_edge_to_index = {
        (end, start, tuple(-np.array(shift))): idx
        for idx, (start, end, shift) in enumerate(edges_with_shifts)
    }
    
    # Initialize an array to hold the indices of the inverse edges, defaulting to -1
    inverse_indices = np.full(len(edges_with_shifts), -1, dtype=int)
    
    # For each edge, check if its inverse exists and store the inverse's index
    for edge, idx in edge_to_index.items():
        if edge in inverse_edge_to_index:
            inverse_indices[idx] = inverse_edge_to_index[edge]
    
    # Raise an exception if any inverse index was not found (i.e., is still -1)
    if np.any(inverse_indices == -1):
        raise RuntimeError("Some edges do not have corresponding inverse edges.")
    
    return inverse_indices


def compute_graph_difference(edge_indices_1, cell_shifts_1, edge_indices_2, cell_shifts_2):
    """
    Compute the difference between two graphs based on their edges and corresponding cell shifts.

    The function calculates the edges and cell shifts present in the first graph
    but not in the second graph, and returns the resulting graph's edges and shifts.

    Parameters:
        edge_indices_1 (np.ndarray): A 2D numpy array of shape (2, E1), where E1 is the number of edges
                                     in the first graph. Each column represents a directed edge (start, end).
        cell_shifts_1 (np.ndarray): A 2D numpy array of shape (E1, 3), where each row represents the
                                    shift vector (x, y, z) for the corresponding edge in the first graph.
        edge_indices_2 (np.ndarray): A 2D numpy array of shape (2, E2), where E2 is the number of edges
                                     in the second graph. Each column represents a directed edge (start, end).
        cell_shifts_2 (np.ndarray): A 2D numpy array of shape (E2, 3), where each row represents the
                                    shift vector (x, y, z) for the corresponding edge in the second graph.

    Returns:
        tuple: A tuple containing:
            - edge_indices_diff (np.ndarray): A 2D numpy array of shape (2, E_diff), representing the edges
                                              present in the first graph but not in the second graph.
            - cell_shifts_diff (np.ndarray): A 2D numpy array of shape (E_diff, 3), representing the cell
                                             shifts for the edges in `edge_indices_diff`.

    Raises:
        ValueError: If the input arrays have mismatched shapes or dimensions.
    """
    # Validate input shapes
    if edge_indices_1.shape[0] != 2 or edge_indices_2.shape[0] != 2:
        raise ValueError("Edge indices must have shape (2, E).")
    if cell_shifts_1.shape[1] != 3 or cell_shifts_2.shape[1] != 3:
        raise ValueError("Cell shifts must have shape (E, 3).")
    if edge_indices_1.shape[1] != cell_shifts_1.shape[0]:
        raise ValueError("Number of edges and cell shifts must match for the first graph.")
    if edge_indices_2.shape[1] != cell_shifts_2.shape[0]:
        raise ValueError("Number of edges and cell shifts must match for the second graph.")

    # Combine edges and cell shifts into tuples for set operations
    edges_with_shifts_1 = set(zip(edge_indices_1[0], edge_indices_1[1], map(tuple, cell_shifts_1)))
    edges_with_shifts_2 = set(zip(edge_indices_2[0], edge_indices_2[1], map(tuple, cell_shifts_2)))

    # Compute the difference: edges and shifts in the first graph but not in the second
    difference_edges_with_shifts = edges_with_shifts_1 - edges_with_shifts_2

    # The sorted result may be empty, indicating the H0 graph already covers the reference graph
    sorted_edges_with_shifts = sorted(difference_edges_with_shifts)
    if not sorted_edges_with_shifts:
        edge_indices_diff = np.empty((2, 0), dtype=edge_indices_1.dtype)
        cell_shifts_diff = np.empty((0, 3), dtype=cell_shifts_1.dtype)
        return edge_indices_diff, cell_shifts_diff

    # Separate edges and cell shifts
    edge_indices, cell_shifts = zip(*[(edge[:2], edge[2]) for edge in sorted_edges_with_shifts])

    # Convert results back into numpy arrays
    edge_indices_diff = np.array(edge_indices).T  # Shape (2, E_diff)
    cell_shifts_diff = np.array(cell_shifts)  # Shape (E_diff, 3)

    return edge_indices_diff, cell_shifts_diff


def build_graph(radius_type, radius_scale, atomic_numbers, lattice, positions):
    """
    Generate a graph representation of atoms based on their positions, atomic numbers, and lattice information.

    Args:
        radius_type (str): Defines the type of radius calculation (e.g., "covalent", "vdw").
        radius_scale (float): Scale factor for radii based on atomic numbers.
        atomic_numbers (np.ndarray): Array of atomic numbers of the atoms (shape: (n_atoms,)).
        lattice (np.ndarray): The lattice matrix defining the simulation box (shape: (3, 3)).
        positions (np.ndarray): The atomic positions in the simulation box (shape: (n_atoms, 3)).

    Returns:
        EasyDict: A dictionary containing:
            - 'z': Atomic numbers of the atoms (np.ndarray, shape: (n_atoms,)).
            - 'pos': Atomic positions (np.ndarray, shape: (n_atoms, 3)).
            - 'edge_index': Neighboring atom indices (np.ndarray, shape: (2, n_neighbors)).
            - 'cell_shift': The shift vectors used for periodic boundary conditions (np.ndarray, shape: (n_neighbors, 3)).
            - 'nbr_shift': The relative shifts for each neighboring atom pair (np.ndarray, shape: (n_neighbors, 3)).
            - 'inv_edge_idx': The inverse neighbor index (np.ndarray, shape: (n_neighbors,)).
    """
    
    # Calculate cutoff radii based on atomic numbers and radius scaling
    cutoff_radii = get_radii_from_atomic_numbers(atomic_numbers, radius_scale=radius_scale, radius_type=radius_type)

    # Generate the neighbor list and associated shift vectors
    edge_index, cell_shift = create_neighbor_list_and_vectors(
        positions,
        max_radius=cutoff_radii,
        include_self_interaction=False,
        strict_self_interaction=True,
        cell_matrix=lattice,
        apply_pbc=True,
    )

    # Compute the neighbor shifts using lattice matrix for PBC correction
    neighbor_shifts = np.einsum('ni, ij -> nj', cell_shift, lattice)

    # Find inverse edge indices for graph processing
    inv_edge_index = find_inverse_edge_index(edge_index, cell_shift)

    # Create the graph structure using EasyDict for easy attribute access
    graph = EasyDict({
        'z': atomic_numbers,             
        'pos': positions,                
        'edge_index': edge_index,        
        'cell_shift': cell_shift,        
        'nbr_shift': neighbor_shifts,    
        'inv_edge_idx': inv_edge_index 
    })

    return graph

    
    