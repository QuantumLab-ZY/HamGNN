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


def neighbor_list_and_relative_vec(
    pos,
    r_max,
    self_interaction=False,
    strict_self_interaction=True,
    cell=None,
    pbc=False,
):
    """Create neighbor list and neighbor vectors based on radial cutoff.

    Edges are given by the following convention:
    - ``edge_index[0]`` is the *source* (convolution center).
    - ``edge_index[1]`` is the *target* (neighbor).

    Args:
        pos (shape [N, 3]): Positional coordinates; Tensor or numpy array.
        r_max (float): Radial cutoff distance for neighbor finding.
        cell (numpy shape [3, 3]): Cell for periodic boundary conditions.
        pbc (bool or 3-tuple of bool): Periodicity in each of the three dimensions.
        self_interaction (bool): Include same periodic image self-edges.
        strict_self_interaction (bool): Include any self interaction edges.

    Returns:
        edge_index (torch.Tensor [2, num_edges]): List of edges.
        shifts (torch.Tensor [num_edges, 3]): Relative cell shift vectors.
        cell_tensor (torch.Tensor [3, 3]): Cell tensor.
    """
    if isinstance(pbc, bool):
        pbc = (pbc,) * 3

    # Handle positional data
    if isinstance(pos, torch.Tensor):
        temp_pos = pos.detach().cpu().numpy()
        out_device = pos.device
        out_dtype = pos.dtype
    else:
        temp_pos = np.asarray(pos)
        out_device = torch.device("cpu")
        out_dtype = torch.get_default_dtype()

    if out_device.type != "cpu":
        warnings.warn(
            "Currently, neighborlists require a round trip to the CPU. Please pass CPU tensors if possible."
        )

    # Handle cell data
    if isinstance(cell, torch.Tensor):
        temp_cell = cell.detach().cpu().numpy()
        cell_tensor = cell.to(device=out_device, dtype=out_dtype)
    elif cell is not None:
        temp_cell = np.asarray(cell)
        cell_tensor = torch.as_tensor(temp_cell, device=out_device, dtype=out_dtype)
    else:
        temp_cell = np.zeros((3, 3), dtype=temp_pos.dtype)
        cell_tensor = torch.as_tensor(temp_cell, device=out_device, dtype=out_dtype)

    temp_cell = geometry.complete_cell(temp_cell)

    # Generate neighbor list
    first_index, second_index, shifts = neighborlist.primitive_neighbor_list(
        "ijS",
        pbc,
        temp_cell,
        temp_pos,
        cutoff=r_max,
        self_interaction=strict_self_interaction,
        use_scaled_positions=False,
    )

    # Filter self-edges
    if not self_interaction:
        bad_edge = first_index == second_index
        bad_edge &= np.all(shifts == 0, axis=1)
        keep_edge = ~bad_edge
        if not np.any(keep_edge):
            raise ValueError("No edges remain after eliminating self-edges.")
        first_index = first_index[keep_edge]
        second_index = second_index[keep_edge]
        shifts = shifts[keep_edge]

    # Build output
    edge_index = torch.vstack(
        (torch.LongTensor(first_index), torch.LongTensor(second_index))
    ).to(device=out_device)

    shifts = torch.as_tensor(
        shifts,
        dtype=torch.long,
        device=out_device,
    )

    return edge_index, shifts, cell_tensor

def find_matching_columns_of_A_in_B(A, B):
    """
    Finds matching columns between two matrices A and B.

    Parameters:
    - A (torch.Tensor): First matrix.
    - B (torch.Tensor): Second matrix.

    Returns:
    - torch.Tensor: Indices of matching columns in B.
    """
    assert A.shape[0] == B.shape[0], "The number of rows in A and B must be the same."
    assert A.shape[-1] <= B.shape[-1], "Please increase radius_scale factor!"

    # Transpose A and B to treat columns as rows for comparison
    A_rows = A.T.unsqueeze(1)  # Shape: (num_cols_A, 1, num_rows)
    B_rows = B.T.unsqueeze(0)  # Shape: (1, num_cols_B, num_rows)

    # Compare each row of A with each row of B
    matches = torch.all(A_rows == B_rows, dim=-1)  # Shape: (num_cols_A, num_cols_B)

    # Find the indices where the rows match
    matching_indices = matches.nonzero(as_tuple=True)[1]  # Take the second element of the tuple

    return matching_indices

class BaseModel(nn.Module):
    def __init__(self, radius_type: str = 'openmx', radius_scale: float = 1.5) -> None:
        super().__init__()
        self.radius_type = radius_type
        self.radius_scale = radius_scale

    def forward(self, data):
        raise NotImplementedError

    def generate_graph(
        self,
        data,
    ):
        graph = EasyDict()

        node_counts = scatter(torch.ones_like(data.batch), data.batch, dim=0).detach()

        latt_batch = data.cell.detach().reshape(-1, 3, 3)
        pos_batch = data.pos.detach()

        pos_batch = torch.split(pos_batch, node_counts.tolist(), dim=0)
        z_batch = torch.split(data.z.detach(), node_counts.tolist(), dim=0)
        
        nbr_shift = []
        edge_index = []
        cell_shift = []

        for idx_xtal, pos in enumerate(pos_batch):
            edge_index_temp, shifts_tmp, _ = neighbor_list_and_relative_vec(
                pos,
                r_max=get_radii_from_atomic_numbers(z_batch[idx_xtal], radius_scale=self.radius_scale, radius_type=self.radius_type),
                self_interaction=False,
                strict_self_interaction=True,
                cell=latt_batch[idx_xtal],
                pbc=True,
            )
            nbr_shift_temp = torch.einsum('ni, ij -> nj',  shifts_tmp.type_as(pos), latt_batch[idx_xtal])
            
            if idx_xtal > 0:
                edge_index_temp += node_counts[idx_xtal - 1]

            edge_index.append(edge_index_temp)
            cell_shift.append(shifts_tmp)
            nbr_shift.append(nbr_shift_temp)

        edge_index = torch.cat(edge_index, dim=-1).type_as(data.edge_index)
        cell_shift = torch.cat(cell_shift, dim=0).type_as(data.cell_shift)
        nbr_shift = torch.cat(nbr_shift, dim=0).type_as(data.nbr_shift)

        matching_edges = find_matching_columns_of_A_in_B(torch.cat([data.edge_index, data.cell_shift.t()], dim=0), 
                                                      torch.cat([edge_index, cell_shift.t()], dim=0))

        graph['z'] = data.z
        graph['pos'] = data.pos
        graph['edge_index'] = edge_index
        graph['cell_shift'] = cell_shift
        graph['nbr_shift'] = nbr_shift
        graph['batch'] = data.batch
        graph['matching_edges'] = matching_edges

        return graph

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())