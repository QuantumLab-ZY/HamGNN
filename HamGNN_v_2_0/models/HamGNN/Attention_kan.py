'''
Descripttion: 
version: 
Author: Yang Zhong
Date: 2024-08-24 20:42:41
LastEditors: Yang Zhong
LastEditTime: 2024-10-10 17:43:34
'''
from typing import Any, Callable, Dict, List, Optional, Type, Union, Tuple
import torch
from torch import nn
import math
from e3nn import o3
from e3nn.nn import FullyConnectedNet
from e3nn.math import soft_unit_step
from torch_scatter import scatter
from e3nn.util.jit import compile_mode
from ..Toolbox.nequip.data import AtomicDataDict
import torch.nn.functional as F 
from ..Toolbox.mace.modules.blocks import EquivariantProductBasisBlock

from ..Toolbox.mace.modules.irreps_tools import (
    linear_out_irreps,
    reshape_irreps,
    tp_out_irreps_with_instructions,
)
from torch_geometric.utils import softmax as edge_softmax
from ..Toolbox.efficient_kan import KAN
from ..Toolbox.nequip.nn.nonlinearities import ShiftedSoftPlus
from e3nn.nn import Gate, NormActivation
from ..Toolbox.nequip.nn import GraphModuleMixin
from ..layers import cuttoff_envelope, CosineCutoff

GRID_SIZE = 3
GRID_RANGE = [-1, 1]

class TensorExpansion(nn.Module):
    def __init__(self, ham_type, nao_max):
        """

        :param ham_type: Type of Hamiltonian ('openmx', 'siesta', 'abacus', 'pasp')
        :param nao_max: Maximum number of atomic orbitals
        """
        super().__init__()
        self.ham_type = ham_type
        self.nao_max = nao_max
        self.index_change = None
        self.minus_index = None
        self.row = None
        self.col = None
        self._set_basis_info()
        
        # Calculate maximum l for Clebsch-Gordan coefficients
        max_l = self.row.lmax + self.col.lmax
        self.cg_calculator = ClebschGordanCoefficients(max_l=max_l)

        irreps_combined = self._combine_irreps()
        self.irreps_out, self.permute_indices, self.inverse_permute_indices = o3.Irreps(irreps_combined).sort()
        self.irreps_out = self.irreps_out.simplify()

    def _combine_irreps(self):
        """
        Combine input irreps to determine output irreps.

        Returns:
            List of combined irreps.
        """
        combined_irreps = []
        for _, li in self.row:
            for _, lj in self.col:
                for L in range(abs(li.l - lj.l), li.l + lj.l + 1):
                    combined_irreps.append(o3.Irrep(L, (-1) ** (li.l + lj.l)))
        return o3.Irreps(combined_irreps)

    def _get_index_change_inv(self, index_change):
        """
        Get the inverse of an index change tensor.

        :param index_change: Tensor indicating the index change.
        :return: Tensor representing the inverse index change.
        """
        index_change_inv = torch.zeros_like(index_change)
        
        for i in range(len(index_change)):
            index_change_inv[index_change[i]] = i
        
        return index_change_inv

    def _set_basis_info(self):
        """
        Sets the basis information based on the Hamiltonian type and number of atomic orbitals.
        """
        if self.ham_type == 'openmx':
            self._set_openmx_basis()
        elif self.ham_type == 'siesta':
            self._set_siesta_basis()
        elif self.ham_type == 'abacus':
            self._set_abacus_basis()
        elif self.ham_type == 'pasp':
            self.row = self.col = o3.Irreps("1x1o")
        else:
            raise NotImplementedError(f"Hamiltonian type '{self.ham_type}' is not supported.")

    def _set_openmx_basis(self):
        """
        Sets basis information for 'openmx' Hamiltonian.
        """
        if self.nao_max == 14:
            self.index_change = torch.LongTensor([0, 1, 2, 5, 3, 4, 8, 6, 7, 11, 13, 9, 12, 10])
            self.row = self.col = o3.Irreps("1x0e+1x0e+1x0e+1x1o+1x1o+1x2e")
        elif self.nao_max == 13:
            self.index_change = torch.LongTensor([0, 1, 4, 2, 3, 7, 5, 6, 10, 12, 8, 11, 9])
            self.row = self.col = o3.Irreps("1x0e+1x0e+1x1o+1x1o+1x2e")
        elif self.nao_max == 19:
            self.index_change = torch.LongTensor([0, 1, 2, 5, 3, 4, 8, 6, 7, 11, 13, 9, 12, 10, 16, 18, 14, 17, 15])
            self.row = self.col = o3.Irreps("1x0e+1x0e+1x0e+1x1o+1x1o+1x2e+1x2e")
        elif self.nao_max == 26:
            self.index_change = torch.LongTensor([0, 1, 2, 5, 3, 4, 8, 6, 7, 11, 13, 9, 12, 10, 16, 18, 14, 17, 15, 22, 23, 21, 24, 20, 25, 19])
            self.row = self.col = o3.Irreps("1x0e+1x0e+1x0e+1x1o+1x1o+1x2e+1x2e+1x3o")
        else:
            raise NotImplementedError(f"NAO max '{self.nao_max}' not supported for 'openmx'.")

    def _set_siesta_basis(self):
        """
        Sets basis information for 'siesta' Hamiltonian.
        """
        if self.nao_max == 13:
            self.index_change = None
            self.row = self.col = o3.Irreps("1x0e+1x0e+1x1o+1x1o+1x2e")
            self.minus_index = torch.LongTensor([2, 4, 5, 7, 9, 11])
        elif self.nao_max == 19:
            self.index_change = None
            self.row = self.col = o3.Irreps("1x0e+1x0e+1x0e+1x1o+1x1o+1x2e+1x2e")
            self.minus_index = torch.LongTensor([3, 5, 6, 8, 10, 12, 15, 17])
        else:
            raise NotImplementedError(f"NAO max '{self.nao_max}' not supported for 'siesta'.")

    def _set_abacus_basis(self):
        """
        Sets basis information for 'abacus' Hamiltonian.
        """
        if self.nao_max == 13:
            self.index_change = torch.LongTensor([0, 1, 3, 4, 2, 6, 7, 5, 10, 11, 9, 12, 8])
            self.row = self.col = o3.Irreps("1x0e+1x0e+1x1o+1x1o+1x2e")
            self.minus_index = torch.LongTensor([3, 4, 6, 7, 9, 10])
        elif self.nao_max == 27:
            self.index_change = torch.LongTensor([0, 1, 2, 3, 5, 6, 4, 8, 9, 7, 12, 13, 11, 14, 10, 17, 18, 16, 19, 15, 23, 24, 22, 25, 21, 26, 20])
            self.row = self.col = o3.Irreps("1x0e+1x0e+1x0e+1x0e+1x1o+1x1o+1x2e+1x2e+1x3o")
            self.minus_index = torch.LongTensor([5, 6, 8, 9, 11, 12, 16, 17, 21, 22, 25, 26])
        elif self.nao_max == 40:
            self.index_change = torch.LongTensor([0, 1, 2, 3, 5, 6, 4, 8, 9, 7, 11, 12, 10, 14, 15, 13, 18, 19, 17, 20, 16, 23, 24, 22, 25, 21, 29, 30, 28, 31, 27, 32, 26, 36, 37, 35, 38, 34, 39, 33])
            self.row = self.col = o3.Irreps("1x0e+1x0e+1x0e+1x0e+1x1o+1x1o+1x1o+1x1o+1x2e+1x2e+1x3o+1x3o")
        else:
            raise NotImplementedError(f"NAO max '{self.nao_max}' not supported for 'abacus'.")

    def _change_index(self, hamiltonian):
        """
        Adjust the order of the output matrix elements to the atomic orbital order of openmx
        """
        if self.index_change is not None or hasattr(self, 'minus_index'):
            hamiltonian = hamiltonian.reshape(-1, self.nao_max, self.nao_max)   
            if self.index_change is not None:
                hamiltonian = hamiltonian[:, self.index_change[:,None], self.index_change[None,:]] 
            if hasattr(self, 'minus_index'):
                hamiltonian[:,self.minus_index,:] = -hamiltonian[:,self.minus_index,:]
                hamiltonian[:,:,self.minus_index] = -hamiltonian[:,:,self.minus_index]                
        return hamiltonian

    def _change_index_inv(self, hamiltonian):
        """
        Adjust the order of the output matrix elements to the atomic orbital order of openmx
        """
        if self.index_change is not None or hasattr(self, 'minus_index'):
            hamiltonian = hamiltonian.reshape(-1, self.nao_max, self.nao_max) 
            if hasattr(self, 'minus_index'):
                hamiltonian[:,self.minus_index,:] = -hamiltonian[:,self.minus_index,:]
                hamiltonian[:,:,self.minus_index] = -hamiltonian[:,:,self.minus_index]  
            if self.index_change is not None:
                index_change_inv = self._get_index_change_inv(self.index_change)
                hamiltonian = hamiltonian[:, index_change_inv[:,None], index_change_inv[None,:]]               
        return hamiltonian

    def forward(self, x):
        """
        Forward pass to compute the expanded tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (*, row.dim, col.dim).

        Returns:
            torch.Tensor: Expanded tensor.
        """
        x = x.reshape(-1, self.row.dim, self.col.dim)
        x = self._change_index_inv(x)
        
        output_blocks = []

        row_start = 0
        for _, li in self.row:
            num_rows = 2 * li.l + 1
            col_start = 0
            for _, lj in self.col:
                num_cols = 2 * lj.l + 1
                for L in range(abs(li.l - lj.l), li.l + lj.l + 1):
                    # Compute Clebsch-Gordan coefficients
                    cg_coeffs = self.cg_calculator(L, li.l, lj.l)
                    block = x.narrow(-2, row_start, num_rows).narrow(-1, col_start, num_cols)
                    output_blocks.append(torch.einsum('nij, kij -> nk', block, cg_coeffs))

                col_start += num_cols
            row_start += num_rows

        # Concatenate outputs and apply inverse permutation
        expanded_output = torch.cat([output_blocks[idx] for idx in self.inverse_permute_indices], dim=-1)
        return expanded_output

class OverlapExpand(nn.Module):
    def __init__(self, ham_type, nao_max) -> None:
        """
        Initialize the OverlapExpand module.

        :param ham_type: Type of Hamiltonian ('openmx', 'siesta', 'abacus', 'pasp').
        :param nao_max: Maximum number of atomic orbitals.
        """
        super().__init__()
        self.tensor_expansion = TensorExpansion(ham_type=ham_type, nao_max=nao_max)
        self.irreps_overlap = self.tensor_expansion.irreps_out

    def forward(self, data):
        """
        Forward pass to expand overlap data.

        Args:
            data: Object containing 'Son' and 'Soff' tensors to be expanded.

        Returns:
            Updated data object with expanded 'Son' and 'Soff'.
        """
        data['Son_expand'] = self.tensor_expansion(data.Son)
        data['Soff_expand'] = self.tensor_expansion(data.Soff)
        return data

@compile_mode("script")
class ClebschGordanCoefficients(nn.Module):
    """
    A PyTorch module for pre-computing and storing Clebsch-Gordan coefficients,
    which can then be accessed during the forward pass.
    """

    def __init__(self, max_l=8):
        """
        Initialize the module and pre-compute Clebsch-Gordan coefficients up to a maximum angular momentum value.

        :param max_l: Maximum angular momentum value for which to compute coefficients.
        """
        super().__init__()

        # Pre-compute and store all necessary Clebsch-Gordan coefficients
        for l1 in range(max_l + 1):
            for l2 in range(max_l + 1):
                for l3 in range(abs(l1 - l2), l1 + l2 + 1):
                    buffer_name = f'cg_{l1}_{l2}_{l3}'
                    self.register_buffer(buffer_name, o3.wigner_3j(l1, l2, l3))

    def forward(self, l1, l2, l3):
        """
        Retrieve the pre-computed Clebsch-Gordan coefficient for the given angular momenta.

        :param l1: First angular momentum value.
        :param l2: Second angular momentum value.
        :param l3: Third angular momentum value.
        :return: The Clebsch-Gordan coefficient tensor.
        """
        buffer_name = f'cg_{l1}_{l2}_{l3}'
        return getattr(self, buffer_name)

@compile_mode("script")
class LinearScaleWithWeights(nn.Module):
    def __init__(self, irreps_in, irreps_out):
        super().__init__()
        
        instructions =  [(i, 0, i, "uvu", True) for i in range(len(irreps_in))]
        
        self.tp = o3.TensorProduct(
            irreps_in,
            o3.Irreps('1x0e'),
            irreps_in,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )
        self.weight_numel = self.tp.weight_numel
        
        self.linear_out = o3.Linear(irreps_in, irreps_out, internal_weights=True, shared_weights=True)
        
    def forward(self, x, weight):
        y = torch.ones_like(x[:, 0:1])
        out = self.tp(x, y, weight)
        out = self.linear_out(out)
        return out

@compile_mode("script")
class SoftUnitStepCutoff(nn.Module):
    """
    A PyTorch module that applies a soft unit step function with a cutoff.
    
    Attributes:
        cutoff (float): The distance at which the cutoff is applied.
        cut_param (nn.Parameter): A learnable parameter influencing the softness of the step.
    """
    def __init__(self, cutoff):
        """
        Initializes the SoftUnitStepCutoff module.
        
        Args:
            cutoff (float): The cutoff distance for the step function.
        """
        super(SoftUnitStepCutoff, self).__init__()
        self.cutoff = cutoff
        self.cut_param = nn.Parameter(torch.tensor(10.0, dtype=torch.get_default_dtype()))

    def forward(self, edge_distance):
        """
        Forward pass for the module.
        
        Applies the soft unit step function to the input edge distances.
        
        Args:
            edge_distance (Tensor): A tensor containing edge distances.
        
        Returns:
            Tensor: A tensor with the calculated edge weights after applying the cutoff.
        """
        # Calculate the scaled difference and apply the soft unit step
        scaled_diff = self.cut_param * (1.0 - edge_distance / self.cutoff)
        edge_weight_cutoff = soft_unit_step(scaled_diff)
        
        return edge_weight_cutoff

def count_neighbors_per_node(source_nodes):
    """
    Calculate the number of neighbors for each node.

    Args:
        source_nodes (torch.Tensor): A tensor containing source node indices.

    Returns:
        torch.Tensor: A tensor where each index represents a node and the value
                      at that index is the count of its neighbors.
    """
    # Identify unique nodes and count their occurrences
    unique_nodes, counts = torch.unique(source_nodes, return_counts=True)

    # Determine the total number of nodes
    total_nodes = source_nodes.max().item() + 1

    # Initialize a tensor to store the neighbor counts for each node
    neighbor_counts = torch.zeros((total_nodes,)).type_as(source_nodes)

    # Assign the counts to their respective nodes
    neighbor_counts[unique_nodes] = counts

    # Ensure the output tensor has the same type as the input
    return neighbor_counts

@compile_mode("script")
class TensorProductWithMemoryOptimizationWithWeight(nn.Module):
    def __init__(self, irreps_input_1, irreps_input_2, irreps_out, irreps_scalar, radial_MLP, use_kan):
        """
        Initialize the TensorProductWithMemoryOptimization module.

        Args:
            irreps_input_1 (str): Irreducible representations for the first input.
            irreps_input_2 (str): Irreducible representations for the second input.
            irreps_out (str): Irreducible representations for the output.
            irreps_scalar (str): Irreducible representations for scalar inputs.
            radial_MLP (list[int]): List of hidden layer sizes for the radial MLP.
            use_kan (bool): Flag to use KAN instead of FullyConnectedNet.
        """
        super().__init__()

        # Initialize irreducible representations
        self.irreps_input_1 = o3.Irreps(irreps_input_1)
        self.irreps_input_2 = o3.Irreps(irreps_input_2)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_scalar = o3.Irreps(irreps_scalar)
        self.radial_MLP = radial_MLP
        self.use_kan = use_kan

        # Calculate intermediate irreps and instructions
        self.irreps_mid, self.instructions = self._tp_out_irreps_with_instructions(
            self.irreps_input_1,
            self.irreps_input_2,
            self.irreps_out,
        )

        # Initialize tensor product
        self.tensor_product = o3.TensorProduct(
            self.irreps_input_1,
            self.irreps_input_2,
            self.irreps_mid,
            instructions=self.instructions,
            internal_weights=True, 
            shared_weights=True
        )

        # Initialize linear scaling with weights
        self.linear_scaler = LinearScaleWithWeights(
            irreps_in=self.irreps_mid.simplify(),
            irreps_out=self.irreps_out
        )

        # Initialize the weight generator
        input_dim = self.irreps_scalar.num_irreps
        self.weight_generator = self._initialize_weight_generator(input_dim, self.linear_scaler.weight_numel)

    def _tp_out_irreps_with_instructions(
        self, irreps1: o3.Irreps, irreps2: o3.Irreps, target_irreps: o3.Irreps
    ) -> Tuple[o3.Irreps, List]:
        trainable = True

        # Collect possible irreps and their instructions
        irreps_out_list: List[Tuple[int, o3.Irreps]] = []
        instructions = []
        for i, (_, ir_in) in enumerate(irreps1):
            for j, (_, ir_edge) in enumerate(irreps2):  
                for _, (mul, ir_out) in enumerate(target_irreps):                  
                    if ir_out in ir_in * ir_edge:
                        k = len(irreps_out_list)
                        irreps_out_list.append((mul, ir_out))
                        instructions.append((i, j, k, 'uvw', trainable))

        # We sort the output irreps of the tensor product so that we can simplify them
        # when they are provided to the second o3.Linear
        irreps_out = o3.Irreps(irreps_out_list)
        irreps_out, permut, _ = irreps_out.sort()

        # Permute the output indexes of the instructions to match the sorted irreps:
        instructions = [
            (i_in1, i_in2, permut[i_out], mode, train)
            for i_in1, i_in2, i_out, mode, train in instructions
        ]

        instructions = sorted(instructions, key=lambda x: x[2])

        return irreps_out, instructions

    def _initialize_weight_generator(self, input_dim, weight_numel):
        """
        Initialize the weight generator module.

        Args:
            input_dim (int): Input dimension size for the weight generator.
            weight_numel (int): Number of elements in the weight vector.

        Returns:
            nn.Module: Initialized weight generator module.
        """
        if self.use_kan:
            return KAN([input_dim] + self.radial_MLP + [weight_numel], grid_size=GRID_SIZE, grid_range=GRID_RANGE)
        return FullyConnectedNet(
            [input_dim] + self.radial_MLP + [weight_numel],
            torch.nn.functional.silu,
        )

    def forward(self, x, y, scalars):
        """
        Forward pass of the TensorProductWithMemoryOptimization module.

        Args:
            x (torch.Tensor): Input tensor for the first irreps.
            y (torch.Tensor): Input tensor for the second irreps.
            scalars (torch.Tensor): Input tensor of scalars.

        Returns:
            torch.Tensor: Output tensor after applying tensor products and scaling.
        """
        # Generate weights using the scalar MLP
        weights = self.weight_generator(scalars)

        # Compute tensor products
        output = self.tensor_product(x, y)
        output = self.linear_scaler(output, weights)

        return output

@compile_mode("script")
class TensorProductWithScalarComponents(nn.Module):
    """
    A module for performing tensor products with memory optimization.

    Parameters:
    - irreps_input_1 (str): Irreducible representations for the first input.
    - irreps_input_2 (str): Irreducible representations for the second input.
    - irreps_out (str): Irreducible representations for the output.
    """

    def __init__(self, irreps_input_1, irreps_input_2, irreps_out):
        super().__init__()

        # Initialize irreducible representations
        self.irreps_input_1 = o3.Irreps(irreps_input_1)
        self.irreps_input_2 = o3.Irreps(irreps_input_2)
        self.irreps_out = o3.Irreps(irreps_out)

        # Calculate intermediate irreps and instructions
        irreps_mid_list = []
        instructions = []
        for i, (mul_1, ir_1) in enumerate(self.irreps_input_1):
            for j, (mul_2, ir_2) in enumerate(self.irreps_input_2):
                for _, (mul_o, ir_out) in enumerate(self.irreps_out):                  
                    if (ir_out in ir_1 * ir_2) and ((ir_1.l, ir_1.p) == (0, 1) or (ir_2.l, ir_2.p) == (0, 1)):
                        k = len(irreps_mid_list)
                        instructions += [(i, j, k, "uvw", True)]
                        irreps_mid_list.append((mul_o, ir_out))

        irreps_mid = o3.Irreps(irreps_mid_list)
        irreps_mid, permut, _ = irreps_mid.sort()

        # Permute the output indexes of the instructions to match the sorted irreps:
        instructions = [
            (i_in1, i_in2, permut[i_out], mode, train)
            for i_in1, i_in2, i_out, mode, train in instructions
        ]
    
        instructions = sorted(instructions, key=lambda x: x[2])

        # Initialize tensor product
        self.tensor_product = o3.TensorProduct(
            self.irreps_input_1,
            self.irreps_input_2,
            irreps_mid,
            instructions=instructions,
            internal_weights=True,
            shared_weights=True,
        )

        # Initialize linear layer
        self.linear_out = o3.Linear(
            irreps_in=irreps_mid.simplify(),
            irreps_out=self.irreps_out,
            internal_weights=True, 
            shared_weights=True
        )

    def forward(self, x, y):
        """
        Forward pass of the module.

        Args:
            x (torch.Tensor): Input tensor for the first irreps.
            y (torch.Tensor): Input tensor for the second irreps.

        Returns:
            torch.Tensor: Output tensor after applying tensor products and scaling.
        """
        # Compute tensor products
        output = self.tensor_product(x, y)
        output = self.linear_out(output)

        return output

def extract_scalar_irreps(irreps: o3.Irreps) -> o3.Irreps:
    """
    Extracts and returns the scalar irreducible representations (irreps) from the given irreps.

    A scalar irrep is defined as one with l=0 and p=1. This function calculates the total 
    multiplicity of such scalar irreps and constructs a new Irreps object containing only these.

    Parameters:
    - irreps (o3.Irreps): The input irreps from which to extract scalar components.

    Returns:
    - o3.Irreps: An Irreps object containing only the scalar components.
    """
    scalar_multiplicity = sum(
        multiplicity for multiplicity, irrep in irreps if irrep.l == 0 and irrep.p == 1
    )
    return o3.Irreps(f"{scalar_multiplicity}x0e")

@compile_mode("script")
class EdgeScalarEmbedding(nn.Module):
    """
    A layer to compute edge scalars from node attributes and edge embeddings.

    Args:
        irreps_node_attrs (Irreps): Irreps for node attributes.
        irreps_edge_embed (Irreps): Irreps for edge embeddings.
        irreps_edge_scalars (Irreps): Irreps for edge scalars.
    """
    def __init__(self, irreps_node_attrs, irreps_edge_embed, irreps_edge_scalars):
        super().__init__()
        self.linear_out = o3.Linear(
            irreps_node_attrs + irreps_node_attrs + irreps_edge_embed, irreps_edge_scalars
        )
        
    def forward(self, node_attr_src, node_attr_dst, edge_embed):
        """
        Forward pass to compute edge scalars.

        Args:
            node_attr_src (Tensor): Source node attributes.
            node_attr_dst (Tensor): Destination node attributes.
            edge_embed (Tensor): Edge embeddings.

        Returns:
            Tensor: Computed edge scalars.
        """
        combined_features = torch.cat([node_attr_src, node_attr_dst, edge_embed], dim=-1)
        return self.linear_out(combined_features)

@compile_mode("script")
class LocalEnvironmentEmbedding(nn.Module):
    """
    Embeds local environments using node and edge attributes, edge embeddings, and spherical harmonics.

    Args:
        irreps_edge_attrs (Irreps): Irreps for edge attributes.
        irreps_edge_embed (Irreps): Irreps for edge embeddings.
        irreps_node_attrs (Irreps): Irreps for node attributes.
        irreps_edge_scalars (Irreps): Irreps for edge scalars.
        irreps_env_sh (Irreps): Irreps for environment spherical harmonics.
        radial_mlp_dims (list[int]): Dimensions for the radial MLP.
        use_kan (bool): Whether to use the KAN model.
    """
    def __init__(self, irreps_edge_attrs, irreps_edge_embed, irreps_node_attrs,
                 irreps_edge_scalars, irreps_env_sh, radial_MLP=[64, 64], use_kan=False):
        super().__init__()

        self.edge_scalar_layer = EdgeScalarEmbedding(irreps_node_attrs, irreps_edge_embed, irreps_edge_scalars)
        
        instructions = [(i, 0, i, "uvw", True) for i in range(len(irreps_edge_attrs))]
        
        self.tensor_product = o3.TensorProduct(
            irreps_edge_attrs,
            o3.Irreps('1x0e'),
            irreps_env_sh,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )
        
        self.weight_numel = self.tensor_product.weight_numel

        input_dim = irreps_edge_embed.num_irreps
        self.weight_generator = self._initialize_weight_generator(input_dim, self.weight_numel, radial_MLP, use_kan)

    def _initialize_weight_generator(self, input_dim, weight_numel, radial_MLP, use_kan):
        """
        Initializes the weight generator.

        Args:
            input_dim (int): Input dimension for the generator.
            weight_numel (int): Number of elements in weights.
            radial_mlp_dims (list[int]): Dimensions for the radial MLP.
            use_kan (bool): Whether to use the KAN model.

        Returns:
            nn.Module: The weight generator model.
        """
        if use_kan:
            return KAN([input_dim] + radial_MLP + [weight_numel], grid_size=GRID_SIZE, grid_range=GRID_RANGE)
        return FullyConnectedNet(
            [input_dim] + radial_MLP + [weight_numel],
            torch.nn.functional.silu,
        )
        
    def forward(self, edge_index, node_attr, edge_attr, edge_embed):
        """
        Forward pass to compute local environment embeddings.

        Args:
            edge_index (Tensor): Indices of the edges.
            node_attr (Tensor): Node attributes.
            edge_attr (Tensor): Edge attributes.
            edge_embed (Tensor): Edge embeddings.

        Returns:
            Tensor: Local environment embeddings.
        """
        src, dst = edge_index
        pseudo_scalar = torch.ones_like(edge_embed[:, :1])
        
        edge_scalars = self.edge_scalar_layer(node_attr[src], node_attr[dst], edge_embed)
        weights = self.weight_generator(edge_scalars)
        local_env_edge = self.tensor_product(edge_attr, pseudo_scalar, weights)
        
        return local_env_edge

@compile_mode("script")
class ConcatenatedIrrepsTensorProduct(nn.Module):
    def __init__(self, irreps_in1, irreps_in2, num_tensors_in1, irreps_out, irreps_edge_scalars, radial_MLP, use_kan):
        """
        Initialize the ConcatenatedIrrepsTensorProduct module.

        Args:
            irreps_in1 (o3.Irreps): Input irreps for the first input tensor.
            irreps_in2 (o3.Irreps): Input irreps for the second input tensor.
            num_tensors_in1 (int): Number of tensors for the first input.
            irreps_out (o3.Irreps): Desired output irreps.
            irreps_edge_scalars (o3.Irreps): Edge scalar irreps.
            radial_mlp (List[int]): Dimensions for the radial MLP.
            use_kan (bool): Whether to use KAN for weight generation.
        """
        super().__init__()
        self.irreps_in1 = o3.Irreps(irreps_in1)
        self.irreps_in2 = o3.Irreps(irreps_in2)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_edge_scalars = o3.Irreps(irreps_edge_scalars)
        self.radial_MLP = radial_MLP
        self.use_kan = use_kan
        self.num_tensors_in1 = num_tensors_in1
        self.irreps_in1_combined = scale_irreps(self.irreps_in1, self.num_tensors_in1)

        self.fuse_in = AttentionHeadsToVector(self.irreps_in1)
        
        # Calculate intermediate irreps and instructions
        self.irreps_mid, self.instructions = self. _tp_out_irreps_with_instructions(
            self.irreps_in1_combined,
            self.irreps_in2,
            self.irreps_out,
        )

        # Initialize tensor product
        self.tensor_product = o3.TensorProduct(
            self.irreps_in1_combined,
            self.irreps_in2,
            self.irreps_mid,
            instructions=self.instructions,
            internal_weights=True,
            shared_weights=True
        )

        # Initialize linear scaling with weights
        self.linear_scaler = LinearScaleWithWeights(
            irreps_in=self.irreps_mid.simplify(),
            irreps_out=self.irreps_out
        )

        # Initialize the weight generator
        input_dim = self.irreps_edge_scalars.num_irreps
        self.weight_generator = self._initialize_weight_generator(input_dim, self.linear_scaler.weight_numel)

        # linear combination
        self.linear_out = o3.Linear(self.irreps_out, self.irreps_out, internal_weights=True, shared_weights=True)

    def _tp_out_irreps_with_instructions(
        self, irreps1: o3.Irreps, irreps2: o3.Irreps, target_irreps: o3.Irreps
    ) -> Tuple[o3.Irreps, List]:
        trainable = True

        # Collect possible irreps and their instructions
        irreps_out_list: List[Tuple[int, o3.Irreps]] = []
        instructions = []
        for i, (_, ir_in) in enumerate(irreps1):
            for j, (_, ir_edge) in enumerate(irreps2):  
                for _, (mul, ir_out) in enumerate(target_irreps):                  
                    if ir_out in ir_in * ir_edge:
                        k = len(irreps_out_list)
                        irreps_out_list.append((mul, ir_out))
                        instructions.append((i, j, k, 'uvw', trainable))

        # We sort the output irreps of the tensor product so that we can simplify them
        # when they are provided to the second o3.Linear
        irreps_out = o3.Irreps(irreps_out_list)
        irreps_out, permut, _ = irreps_out.sort()

        # Permute the output indexes of the instructions to match the sorted irreps:
        instructions = [
            (i_in1, i_in2, permut[i_out], mode, train)
            for i_in1, i_in2, i_out, mode, train in instructions
        ]

        instructions = sorted(instructions, key=lambda x: x[2])

        return irreps_out, instructions

    def _initialize_weight_generator(self, input_dim, weight_numel):
        """
        Initialize the weight generator module.

        Args:
            input_dim (int): Input dimension size for the weight generator.
            weight_numel (int): Number of elements in the weight vector.

        Returns:
            nn.Module: Initialized weight generator module.
        """
        if self.use_kan:
            return KAN([input_dim] + self.radial_MLP + [weight_numel], grid_size=GRID_SIZE, grid_range=GRID_RANGE)
        return FullyConnectedNet(
            [input_dim] + self.radial_MLP + [weight_numel],
            torch.nn.functional.silu,
        )

    def forward(self, input_tensors1_list: List[torch.Tensor], input_tensor2: torch.Tensor, scalars: torch.Tensor):
        """
        Forward pass for the ConcatenatedIrrepsTensorProduct module.

        Args:
            input_tensors1_list (List[torch.Tensor]): List of tensors for the first input.
            input_tensor2 (torch.Tensor): Tensor for the second input.
            scalars (torch.Tensor): Scalar inputs for weight generation.

        Returns:
            torch.Tensor: Processed output tensor.
        """
        input_tensor1 = self.fuse_in(torch.stack(input_tensors1_list, dim=-2))

        # Generate weights using the scalar MLP
        weights = self.weight_generator(scalars)

        # Compute tensor products
        output = self.tensor_product(input_tensor1, input_tensor2)
        output = self.linear_scaler(output, weights)

        # output
        output = self.linear_out(output)

        return output

@compile_mode("script")
class MessagePackBlock(nn.Module):
    def __init__(
        self,
        irreps_node_feats: str,
        irreps_edge_feats: str,
        irreps_local_env_edge: str,
        irreps_out: str,
        irreps_edge_scalars: str,
        radial_MLP: List[int] = [64, 64],
        use_kan: bool = False
    ):
        """
        Initializes the MessagePackBlock.

        Args:
            irreps_node_feats (str): Irreducible representations for node features.
            irreps_edge_feats (str): Irreducible representations for edge features.
            irreps_local_env_edge (str): Irreducible representations for local environment edges.
            irreps_out (str): Irreducible representations for outputs.
            irreps_edge_scalars (str): Irreducible representations for edge scalars.
            radial_mlp_layers (List[int]): Layers for radial MLP.
            use_kan (bool): Flag to use KAN for weight generation.
        """
        super().__init__()
        self.irreps_node_feats = o3.Irreps(irreps_node_feats)
        self.irreps_edge_feats = o3.Irreps(irreps_edge_feats)
        self.irreps_local_env_edge = o3.Irreps(irreps_local_env_edge)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_edge_scalars = o3.Irreps(irreps_edge_scalars)
        self.radial_MLP = radial_MLP
        self.use_kan = use_kan

        self.combined_node_irreps = scale_irreps(self.irreps_node_feats, 2)
        self.fuse_node = AttentionHeadsToVector(self.irreps_node_feats)

        # Calculate intermediate irreps and instructions
        self.mid_node_irreps, self.node_instructions = self._tp_out_irreps_with_instructions(
            self.combined_node_irreps,
            self.irreps_local_env_edge,
            self.irreps_out,
        )
        self.mid_edge_irreps, self.edge_instructions = self._tp_out_irreps_with_instructions(
            self.irreps_edge_feats,
            self.irreps_local_env_edge,
            self.irreps_out,
        )

        # Initialize tensor product
        self.node_tensor_product = o3.TensorProduct(
            self.combined_node_irreps,
            self.irreps_local_env_edge,
            self.mid_node_irreps,
            instructions=self.node_instructions,
            internal_weights=True,
            shared_weights=True
        )
        self.edge_tensor_product = o3.TensorProduct(
            self.irreps_edge_feats,
            self.irreps_local_env_edge,
            self.mid_edge_irreps,
            instructions=self.edge_instructions,
            internal_weights=True,
            shared_weights=True
        )

        # Initialize linear scaling with weights
        self.node_linear_scaler = LinearScaleWithWeights(
            irreps_in=self.mid_node_irreps.simplify(),
            irreps_out=self.irreps_out
        )
        self.edge_linear_scaler = LinearScaleWithWeights(
            irreps_in=self.mid_edge_irreps.simplify(),
            irreps_out=self.irreps_out
        )

        # Initialize the weight generator
        input_dim = self.irreps_edge_scalars.num_irreps
        self.node_weight_generator = self._initialize_weight_generator(input_dim, self.node_linear_scaler.weight_numel)
        self.edge_weight_generator = self._initialize_weight_generator(input_dim, self.edge_linear_scaler.weight_numel)

        # Linear output layers
        self.node_linear_out = o3.Linear(self.irreps_out, self.irreps_out, internal_weights=True, shared_weights=True)
        self.edge_linear_out = o3.Linear(self.irreps_out, self.irreps_out, internal_weights=True, shared_weights=True)

    def _tp_out_irreps_with_instructions(
        self, irreps1: o3.Irreps, irreps2: o3.Irreps, target_irreps: o3.Irreps
    ) -> Tuple[o3.Irreps, List]:
        trainable = True

        # Collect possible irreps and their instructions
        irreps_out_list: List[Tuple[int, o3.Irreps]] = []
        instructions = []
        for i, (_, ir_in) in enumerate(irreps1):
            for j, (_, ir_edge) in enumerate(irreps2):  
                for _, (mul, ir_out) in enumerate(target_irreps):                  
                    if ir_out in ir_in * ir_edge:
                        k = len(irreps_out_list)
                        irreps_out_list.append((mul, ir_out))
                        instructions.append((i, j, k, 'uvw', trainable))

        # We sort the output irreps of the tensor product so that we can simplify them
        # when they are provided to the second o3.Linear
        irreps_out = o3.Irreps(irreps_out_list)
        irreps_out, permut, _ = irreps_out.sort()

        # Permute the output indexes of the instructions to match the sorted irreps:
        instructions = [
            (i_in1, i_in2, permut[i_out], mode, train)
            for i_in1, i_in2, i_out, mode, train in instructions
        ]

        instructions = sorted(instructions, key=lambda x: x[2])

        return irreps_out, instructions

    def _initialize_weight_generator(self, input_dim, weight_numel):
        """
        Initialize the weight generator module.

        Args:
            input_dim (int): Input dimension size for the weight generator.
            weight_numel (int): Number of elements in the weight vector.

        Returns:
            nn.Module: Initialized weight generator module.
        """
        if self.use_kan:
            return KAN([input_dim] + self.radial_MLP + [weight_numel], grid_size=GRID_SIZE, grid_range=GRID_RANGE)
        return FullyConnectedNet(
            [input_dim] + self.radial_MLP + [weight_numel],
            torch.nn.functional.silu,
        )

    def forward(self, node_feats_src: torch.Tensor, 
                node_feats_dst: torch.Tensor, 
                edge_feats: torch.Tensor, 
                local_env_edge: torch.Tensor,
                edge_scalars: torch.Tensor):

        # Compute tensor products for node interaction
        node_inter = self.fuse_node(torch.stack([node_feats_src, node_feats_dst], dim=-2))
        weights_node = self.node_weight_generator(edge_scalars)
        node_inter_up = self.node_tensor_product(node_inter, local_env_edge)
        node_inter_dn = self.node_linear_scaler(node_inter_up, weights_node)
        
        # Compute tensor products for edge_features
        weights_edge = self.edge_weight_generator(edge_scalars)
        edge_feats_up = self.edge_tensor_product(edge_feats, local_env_edge)
        edge_feats_dn = self.edge_linear_scaler(edge_feats_up, weights_edge)        

        # output
        output = self.node_linear_out(node_inter_dn) + self.edge_linear_out(edge_feats_dn)

        return output

@compile_mode("script")
class MessagePackBlockV2(nn.Module):
    def __init__(
        self,
        irreps_node_feats: str,
        irreps_edge_feats: str,
        irreps_local_env_edge: str,
        irreps_out: str,
        irreps_edge_scalars: str,
        radial_MLP: List[int] = [64, 64],
        use_kan: bool = False
    ):
        """
        Initializes the MessagePackBlock.

        Args:
            irreps_node_feats (str): Irreducible representations for node features.
            irreps_edge_feats (str): Irreducible representations for edge features.
            irreps_local_env_edge (str): Irreducible representations for local environment edges.
            irreps_out (str): Irreducible representations for outputs.
            irreps_edge_scalars (str): Irreducible representations for edge scalars.
            radial_mlp_layers (List[int]): Layers for radial MLP.
            use_kan (bool): Flag to use KAN for weight generation.
        """
        super().__init__()
        self.irreps_node_feats = o3.Irreps(irreps_node_feats)
        self.irreps_edge_feats = o3.Irreps(irreps_edge_feats)
        self.irreps_local_env_edge = o3.Irreps(irreps_local_env_edge)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_edge_scalars = o3.Irreps(irreps_edge_scalars)
        self.radial_MLP = radial_MLP
        self.use_kan = use_kan

        self.combined_node_irreps = scale_irreps(self.irreps_node_feats, 2)
        self.fuse_node = AttentionHeadsToVector(self.irreps_node_feats)

        # Calculate intermediate irreps and instructions
        self.mid_node_irreps, self.node_instructions = self._tp_out_irreps_with_instructions(
            self.combined_node_irreps,
            self.irreps_local_env_edge,
            self.irreps_out,
        )
        self.mid_edge_irreps, self.edge_instructions = self._tp_out_irreps_with_instructions(
            self.irreps_edge_feats,
            self.irreps_local_env_edge,
            self.irreps_out,
        )
        self.mid_node_node_irreps, self.node_node_instructions = self._tp_out_irreps_with_instructions(
            self.irreps_node_feats,
            self.irreps_node_feats,
            self.irreps_out,
            mode='uvu'
        )

        # Initialize tensor product
        self.node_tensor_product = o3.TensorProduct(
            self.combined_node_irreps,
            self.irreps_local_env_edge,
            self.mid_node_irreps,
            instructions=self.node_instructions,
            internal_weights=True,
            shared_weights=True
        )
        self.edge_tensor_product = o3.TensorProduct(
            self.irreps_edge_feats,
            self.irreps_local_env_edge,
            self.mid_edge_irreps,
            instructions=self.edge_instructions,
            internal_weights=True,
            shared_weights=True
        )
        self.node_node_tensor_product = o3.TensorProduct(
            self.irreps_node_feats,
            self.irreps_node_feats,
            self.mid_node_node_irreps,
            instructions=self.node_node_instructions,
            internal_weights=True,
            shared_weights=True
        )

        # Initialize linear scaling with weights
        self.node_linear_scaler = LinearScaleWithWeights(
            irreps_in=self.mid_node_irreps.simplify(),
            irreps_out=self.irreps_out
        )
        self.edge_linear_scaler = LinearScaleWithWeights(
            irreps_in=self.mid_edge_irreps.simplify(),
            irreps_out=self.irreps_out
        )
        self.node_node_linear_scaler = LinearScaleWithWeights(
            irreps_in=self.mid_node_node_irreps.simplify(),
            irreps_out=self.irreps_out
        )

        # Initialize the weight generator
        input_dim = self.irreps_edge_scalars.num_irreps
        self.node_weight_generator = self._initialize_weight_generator(input_dim, self.node_linear_scaler.weight_numel)
        self.edge_weight_generator = self._initialize_weight_generator(input_dim, self.edge_linear_scaler.weight_numel)
        self.node_node_weight_generator = self._initialize_weight_generator(input_dim, self.node_node_linear_scaler.weight_numel)

        # Linear output layers
        self.node_linear_out = o3.Linear(self.irreps_out, self.irreps_out, internal_weights=True, shared_weights=True)
        self.edge_linear_out = o3.Linear(self.irreps_out, self.irreps_out, internal_weights=True, shared_weights=True)
        self.node_node_linear_out = o3.Linear(self.irreps_out, self.irreps_out, internal_weights=True, shared_weights=True)

    def _tp_out_irreps_with_instructions(
        self, irreps1: o3.Irreps, irreps2: o3.Irreps, target_irreps: o3.Irreps, mode: str='uvw'
    ) -> Tuple[o3.Irreps, List]:
        trainable = True

        # Collect possible irreps and their instructions
        irreps_out_list: List[Tuple[int, o3.Irreps]] = []
        instructions = []
        for i, (mul_i, ir_in) in enumerate(irreps1):
            for j, (mul_j, ir_edge) in enumerate(irreps2):  
                for _, (mul, ir_out) in enumerate(target_irreps):                  
                    if ir_out in ir_in * ir_edge:
                        k = len(irreps_out_list)
                        if mode=='uvw':
                            irreps_out_list.append((mul, ir_out))
                        elif mode=='uvu':
                            irreps_out_list.append((mul_i, ir_out))
                        else:
                            raise NotImplementedError
                        instructions.append((i, j, k, mode, trainable))

        # We sort the output irreps of the tensor product so that we can simplify them
        # when they are provided to the second o3.Linear
        irreps_out = o3.Irreps(irreps_out_list)
        irreps_out, permut, _ = irreps_out.sort()

        # Permute the output indexes of the instructions to match the sorted irreps:
        instructions = [
            (i_in1, i_in2, permut[i_out], m, train)
            for i_in1, i_in2, i_out, m, train in instructions
        ]

        instructions = sorted(instructions, key=lambda x: x[2])

        return irreps_out, instructions

    def _initialize_weight_generator(self, input_dim, weight_numel):
        """
        Initialize the weight generator module.

        Args:
            input_dim (int): Input dimension size for the weight generator.
            weight_numel (int): Number of elements in the weight vector.

        Returns:
            nn.Module: Initialized weight generator module.
        """
        if self.use_kan:
            return KAN([input_dim] + self.radial_MLP + [weight_numel], grid_size=GRID_SIZE, grid_range=GRID_RANGE)
        return FullyConnectedNet(
            [input_dim] + self.radial_MLP + [weight_numel],
            torch.nn.functional.silu,
        )

    def forward(self, node_feats_src: torch.Tensor, 
                node_feats_dst: torch.Tensor, 
                edge_feats: torch.Tensor, 
                local_env_edge: torch.Tensor,
                edge_scalars: torch.Tensor):

        # Compute tensor products for node interaction
        node_inter = self.fuse_node(torch.stack([node_feats_src, node_feats_dst], dim=-2))
        weights_node = self.node_weight_generator(edge_scalars)
        node_inter_up = self.node_tensor_product(node_inter, local_env_edge)
        node_inter_dn = self.node_linear_scaler(node_inter_up, weights_node)
        
        # node-node tensor product
        weights_node_node = self.node_node_weight_generator(edge_scalars)
        node_node_inter_up = self.node_node_tensor_product(node_feats_dst, node_feats_src)
        node_node_inter_dn = self.node_node_linear_scaler(node_node_inter_up, weights_node_node)
        
        # Compute tensor products for edge_features
        weights_edge = self.edge_weight_generator(edge_scalars)
        edge_feats_up = self.edge_tensor_product(edge_feats, local_env_edge)
        edge_feats_dn = self.edge_linear_scaler(edge_feats_up, weights_edge)        

        # output
        output = self.node_linear_out(node_inter_dn) + self.edge_linear_out(edge_feats_dn) + self.node_node_linear_out(node_node_inter_dn)

        return output

acts = {
    "abs": torch.abs,
    "tanh": torch.tanh,
    "ssp": ShiftedSoftPlus,
    "silu": torch.nn.functional.silu,
}

def irreps2gate(
    irreps: o3.Irreps,
    nonlinearity_scalars: Dict[int, str] = {1: "ssp", -1: "tanh"},
    nonlinearity_gates: Dict[int, str] = {1: "ssp", -1: "abs"},
) -> Tuple[o3.Irreps, o3.Irreps, o3.Irreps, List[Callable], List[Callable]]:
    """
    Splits irreducible representations into scalar and gated components and associates activation functions.

    Parameters:
    - irreps (o3.Irreps): The input irreducible representations.
    - nonlinearity_scalars (Dict[int, str]): Activation functions for scalar components.
    - nonlinearity_gates (Dict[int, str]): Activation functions for gate components.

    Returns:
    - Tuple containing:
        - irreps_scalars (o3.Irreps): Scalar irreps.
        - irreps_gates (o3.Irreps): Gate irreps.
        - irreps_gated (o3.Irreps): Gated irreps.
        - act_scalars (List[Callable]): Activation functions for scalars.
        - act_gates (List[Callable]): Activation functions for gates.
    """
    # Split the irreps into scalar and gated components
    irreps_scalars = o3.Irreps([(mul, ir) for mul, ir in irreps if ir.l == 0]).simplify()
    irreps_gated = o3.Irreps([(mul, ir) for mul, ir in irreps if ir.l != 0]).simplify()

    # Determine the gate irreps based on the presence of gated components
    irreps_gates = o3.Irreps([(mul, '0e') for mul, _ in irreps_gated]).simplify() if irreps_gated.dim > 0 else o3.Irreps([])

    # Retrieve the activation functions for scalars and gates
    act_scalars = [acts[nonlinearity_scalars[ir.p]] for _, ir in irreps_scalars]
    act_gates = [acts[nonlinearity_gates[ir.p]] for _, ir in irreps_gates]

    return irreps_scalars, irreps_gates, irreps_gated, act_scalars, act_gates

def scale_irreps(irreps: o3.Irreps, factor: float) -> o3.Irreps:
    """
    Scales the multiplicities of the irreducible representations (irreps) by a given factor,
    ensuring they remain at least 1.

    Parameters:
    - irreps (o3.Irreps): The input irreps.
    - factor (float): The scaling factor.

    Returns:
    - o3.Irreps: The scaled irreps.
    """
    return o3.Irreps([(max(1, int(mul * factor)), ir) for mul, ir in irreps])

def filter_and_split_irreps(irreps: o3.Irreps, num_channels: int, min_l: int, max_l: int) -> o3.Irreps:
    """
    Filters and splits irreducible representations (irreps) based on specified angular momentum range.

    Parameters:
    - irreps (o3.Irreps): The input irreducible representations.
    - num_channels (int): The number of channels to split the multiplicity by.
    - min_l (int): The minimum angular momentum (inclusive).
    - max_l (int): The maximum angular momentum (inclusive).

    Returns:
    - o3.Irreps: The resulting irreducible representations after filtering and splitting.
    """
    result_irreps = o3.Irreps()
    for multiplicity, irrep in irreps:
        if irrep.l < min_l or irrep.l > max_l:
            # Retain irreps outside the specified l range
            result_irreps += o3.Irreps([(multiplicity, irrep)])
        else:
            # Split multiplicity by num_channels for irreps within the range
            split_multiplicity = multiplicity // num_channels
            if split_multiplicity > 0:
                result_irreps += split_multiplicity * o3.Irreps([(num_channels, irrep)])
    
    return result_irreps

@compile_mode('script')
class RadialBasisEdgeEncoding(GraphModuleMixin, torch.nn.Module):
    """
    Encodes edge lengths using a specified radial basis.

    Attributes:
        out_field (str): The key for storing the encoded edge features.
    """

    def __init__(
        self,
        basis=None,
        cutoff=None,
        out_field: str = AtomicDataDict.EDGE_EMBEDDING_KEY,
        irreps_in=None,
    ):
        """
        Initializes the RadialBasisEdgeEncoding module.

        :param basis: The radial basis function used for encoding.
        :param out_field: The output field key for encoded edges.
        :param irreps_in: Input irreducible representations.
        """
        super().__init__()
        self.basis = basis
        self.cutoff = cutoff
        self.out_field = out_field

        # Determine the number of basis functions based on the basis type
        basis_type = type(basis).__name__.split(".")[-1]
        if basis_type in {'BesselBasis', 'GaussianSmearing'}:
            num_basis = basis.freqs.size(0) if basis_type == 'BesselBasis' else basis.offset.size(0)
        elif basis_type in {
            'ExponentialGaussianRadialBasisFunctions',
            'ExponentialBernsteinRadialBasisFunctions',
            'GaussianRadialBasisFunctions',
            'BernsteinRadialBasisFunctions'
        }:
            num_basis = basis.num_basis_functions
        else:
            raise NotImplementedError(f"Basis type {basis_type} is not supported.")

        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={self.out_field: o3.Irreps([(num_basis, (0, 1))])},
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        """
        Computes the edge encoding and updates the data dictionary.

        :param data: A dictionary containing graph data.
        :return: Updated graph data with encoded edge features.
        """
        j, i = data.edge_index
        nbr_shift = data.nbr_shift
        pos = data.pos

        # Calculate edge directions and lengths
        edge_dir = (pos[i] + nbr_shift) - pos[j]
        edge_length = edge_dir.norm(dim=-1)

        # Update data with computed edge vectors and lengths
        data[AtomicDataDict.EDGE_VECTORS_KEY] = edge_dir/edge_length[:,None]
        data[AtomicDataDict.EDGE_LENGTH_KEY] = edge_length

        # Apply the radial basis to the edge lengths
        edge_length_embedded = self.basis(edge_length)
        
        if self.cutoff is not None:
            edge_length_embedded = edge_length_embedded*self.cutoff(edge_length)[:, None]
            
        data[self.out_field] = edge_length_embedded

        return data

@compile_mode('script')
class VectorToAttentionHeads(nn.Module):
    """
    Reshapes vectors of shape [N, irreps_mid] to vectors of shape [N, num_heads, irreps_head].

    Attributes:
    - num_heads (int): Number of attention heads.
    - irreps_head (o3.Irreps): Irreps of each head.
    - irreps_mid_in (o3.Irreps): Intermediate irreps.
    - mid_in_indices (List[Tuple[int, int]]): Indices for reshaping.
    """

    def __init__(self, irreps_head: o3.Irreps, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.irreps_head = irreps_head
        self.irreps_mid_in = o3.Irreps([(mul * num_heads, ir) for mul, ir in irreps_head])
        self.mid_in_indices = []
        start_idx = 0
        for mul, ir in self.irreps_mid_in:
            self.mid_in_indices.append((start_idx, start_idx + mul * ir.dim))
            start_idx = start_idx + mul * ir.dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, _ = x.shape
        reshaped_tensors = [
            x.narrow(1, start_idx, end_idx - start_idx).view(N, self.num_heads, -1)
            for start_idx, end_idx in self.mid_in_indices
        ]
        return torch.cat(reshaped_tensors, dim=2)

    def __repr__(self):
        return f'{self.__class__.__name__}(irreps_head={self.irreps_head}, num_heads={self.num_heads})'

@compile_mode('script')
class AttentionHeadsToVector(nn.Module):
    """
    Converts vectors of shape [N, num_heads, irreps_head] into vectors of shape [N, irreps_head * num_heads].
    
    Attributes:
        irreps_head (o3.Irreps): A list of irreducible representations (irreps) that define
                                 the structure of the attention heads.
        head_sizes (List[int]): A list of sizes for each attention head, derived from the irreps.
    """

    def __init__(self, irreps_head: o3.Irreps):
        """
        Initialize the AttentionHeadsToVector module.

        Args:
            irreps_head (o3.Irreps): A list of irreducible representations (irreps) used to define
                                     the structure of attention heads. Each irrep specifies the
                                     multiplicity and dimension of a representation.
        """
        super().__init__()
        self.irreps_head = irreps_head

        # Compute the size of each attention head based on the irreps definitions.
        self.head_sizes = [multiplicity * irrep.dim for multiplicity, irrep in self.irreps_head]

    def __repr__(self):
        """
        Provide a string representation of the module for debugging.

        Returns:
            str: A string representation of the AttentionHeadProcessor instance.
        """
        return f'{self.__class__.__name__}(irreps_head={self.irreps_head})'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to process the attention heads and flatten them into a single vector.

        Args:
            x (torch.Tensor): Input tensor of shape (N, num_heads, input_dim), where:
                - N is the batch size.
                - num_heads is the number of attention heads.
                - input_dim is the total size of all heads.

        Returns:
            torch.Tensor: Output tensor of shape (N, flattened_dim), where `flattened_dim`
                          is the sum of the dimensions of all attention heads.

        Raises:
            ValueError: If the sum of `head_sizes` does not match `input_dim` of the input tensor.
        """
        # Extract the dimensions of the input tensor.
        batch_size, num_heads, input_dim = x.shape

        # Ensure the total size of all attention heads matches the input tensor's last dimension.
        if sum(self.head_sizes) != input_dim:
            raise ValueError(
                f"The sum of head_sizes ({sum(self.head_sizes)}) does not match the input_dim ({input_dim}) "
                "of the input tensor."
            )

        # Split the input tensor along the last dimension based on head_sizes.
        split_tensors = torch.split(x, self.head_sizes, dim=2)

        # Reshape each split tensor to flatten the attention heads into a single vector per batch.
        # Use `contiguous()` to ensure the tensor's memory layout is consistent.
        flattened_tensors = [sub_tensor.contiguous().view(batch_size, -1) for sub_tensor in split_tensors]

        # Concatenate the flattened tensors along the last dimension to produce the output.
        return torch.cat(flattened_tensors, dim=1)

@compile_mode("script")
class ConvBlockE3(nn.Module):
    """
    An equivariant convolutional block for processing node features using tensor products
    with optional skip connections.

    Parameters:
    - irreps_in (o3.Irreps): Input irreducible representations.
    - irreps_out (o3.Irreps): Output irreducible representations.
    - irreps_edge_attrs (o3.Irreps): Edge attribute irreducible representations.
    - irreps_edge_embed (o3.Irreps): Edge embedding irreducible representations.
    - radial_MLP (Optional[List[int]]): MLP architecture for radial embeddings. Defaults to [64, 64, 64].
    - use_skip_connections (bool): Whether to use skip connections. Defaults to True.
    - use_kan (bool): Whether to use the FastKAN module for weight generation. Defaults to False.
    - nonlinearity_type (str): Type of nonlinearity to use ("gate" or "norm"). Defaults to "gate".
    - nonlinearity_scalars (Dict[int, Callable]): Nonlinearity for scalar channels.
    - nonlinearity_gates (Dict[int, Callable]): Nonlinearity for gate channels.
    """

    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        irreps_node_attrs: o3.Irreps,
        irreps_edge_attrs: o3.Irreps,
        irreps_edge_embed: o3.Irreps,
        radial_MLP: Optional[list] = None,
        use_skip_connections: bool = True,
        use_kan: bool = False,
        nonlinearity_type: str = "gate",
        nonlinearity_scalars: dict = {"e": "ssp", "o": "tanh"},
        nonlinearity_gates: dict = {"e": "ssp", "o": "abs"},
    ):
        super().__init__()

        self.radial_MLP = radial_MLP or [64, 64, 64]
        self.use_kan = use_kan
        self.use_skip_connections = use_skip_connections

        assert nonlinearity_type in ("gate", "norm"), "Invalid nonlinearity type."

        # Convert nonlinearity mappings
        scalar_nonlinearities = {
            1: nonlinearity_scalars["e"],
            -1: nonlinearity_scalars["o"],
        }
        gate_nonlinearities = {
            1: nonlinearity_gates["e"],
            -1: nonlinearity_gates["o"],
        }

        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_node_attrs = o3.Irreps(irreps_node_attrs)
        self.irreps_edge_attrs = o3.Irreps(irreps_edge_attrs)
        self.irreps_edge_embed = o3.Irreps(irreps_edge_embed)

        # Residual block for processing features
        self.residual = ResidualBlock(self.irreps_in, self.irreps_out)

        # Convolution layers       
        self.conv_tp = MessagePackBlock(
            irreps_node_feats=self.irreps_in,
            irreps_edge_feats=self.irreps_in,
            irreps_local_env_edge=self.irreps_edge_attrs,
            irreps_out=self.irreps_out,
            irreps_edge_scalars=self.irreps_edge_embed, 
            radial_MLP=self.radial_MLP, 
            use_kan=self.use_kan
            )
        
        # Skip connection layer
        if self.use_skip_connections:
            self.skip_linear = self.create_linear(self.irreps_in, self.irreps_out)

    def create_linear(self, irreps_in, irreps_out=None):
        """
        Create a linear layer.

        Parameters:
        - irreps_in (o3.Irreps): Input irreps for the linear layer.
        - irreps_out (o3.Irreps, optional): Output irreps for the linear layer.

        Returns:
        - o3.Linear: A linear transformation layer.
        """
        return o3.Linear(
            irreps_in, irreps_out or irreps_in, internal_weights=True, shared_weights=True
        )

    def forward(self, data: dict) -> torch.Tensor:
        """
        Forward pass of the convolutional block.

        Parameters:
        - data (dict): Dictionary containing graph data.

        Returns:
        - torch.Tensor: Updated node features.
        """
        sender, receiver = data[AtomicDataDict.EDGE_INDEX_KEY]
        node_features = data[AtomicDataDict.NODE_FEATURES_KEY]
        edge_embedding = data[AtomicDataDict.EDGE_EMBEDDING_KEY]
        edge_attributes = data[AtomicDataDict.EDGE_ATTRS_KEY]
        num_nodes = len(data[AtomicDataDict.NODE_FEATURES_KEY])

        # Skip connection
        skip_connection = self.skip_linear(node_features) if self.use_skip_connections else None
        
        # Messages        
        messages = self.conv_tp(
            node_features[sender], 
            node_features[receiver],  
            data[AtomicDataDict.EDGE_FEATURES_KEY], 
            edge_attributes,
            edge_embedding
        )

        # Aggregate messages
        aggregated_messages = scatter(
            src=messages, index=receiver, dim=0, dim_size=num_nodes
        )
        
        # Apply residual block
        output_features = self.residual(aggregated_messages)

        # Apply skip connection if used
        if self.use_skip_connections:
            output_features += skip_connection

        data[AtomicDataDict.NODE_FEATURES_KEY] = output_features
        
        return output_features

@compile_mode("script")
class AttentionAggregationV2(nn.Module):
    """
    An equivariant attention mechanism that processes key, value, and query vectors
    and applies attention across edges in a graph.

    Parameters:
    - num_heads (int): Number of attention heads.
    - irreps_value (o3.Irreps): Irreducible representations for value vectors.
    """

    def __init__(
        self, 
        num_heads: int, 
        irreps_value: o3.Irreps, 
    ):
        super().__init__()
        self.num_heads = num_heads
        irreps_value = o3.Irreps(irreps_value)
        
        self.value_irreps_head = scale_irreps(irreps_value, 1/num_heads)
        self.unfuse_value = VectorToAttentionHeads(self.value_irreps_head, num_heads)
        self.fuse_value = AttentionHeadsToVector(self.value_irreps_head)
    
    def forward(
        self, 
        value,
        edge_weights: torch.Tensor,  # (num_edges, num_heads)
        edge_weights_cutoff: torch.Tensor, # (num_edges,)
        edge_index: torch.LongTensor
    ) -> torch.Tensor:
        """
        Forward pass of the attention mechanism.

        Parameters:
        - key (torch.Tensor): Key vectors.
        - value (torch.Tensor): Value vectors.
        - query (torch.Tensor): Query vectors.
        - edge_weight_cutoff (torch.Tensor): Cutoff weights for edges.
        - edge_index (torch.LongTensor): Edge indices.

        Returns:
        - torch.Tensor: Attended output vectors.
        """
        value = self.unfuse_value(value)
        
        edge_src, edge_dst = edge_index
        
        # Compute the attention weights per edge
        if edge_weights_cutoff is not None:
            edge_weights = edge_weights_cutoff[:, None] * edge_weights  # (num_edges, num_heads)
        edge_weights = edge_softmax(edge_weights, edge_dst)  # (num_edges, num_heads)
        edge_weights = edge_weights.unsqueeze(-1)  # (num_edges, num_heads, 1)

        # Compute the attended outputs per node
        f_out = scatter(edge_weights * value, edge_dst, dim=0)  # (num_nodes, num_heads, irreps_head)
        f_out = self.fuse_value(f_out)  # Merge heads
        return f_out

@compile_mode("script")
class AttentionAggregation(nn.Module):
    """
    An equivariant attention mechanism that processes key, value, and query vectors
    and applies attention across edges in a graph.

    Parameters:
    - num_heads (int): Number of attention heads.
    - irreps_key (o3.Irreps): Irreducible representations for key vectors.
    - irreps_value (o3.Irreps): Irreducible representations for value vectors.
    - irreps_query (o3.Irreps): Irreducible representations for query vectors.
    """

    def __init__(
        self, 
        num_heads: int, 
        irreps_key: o3.Irreps, 
        irreps_value: o3.Irreps, 
        irreps_query: o3.Irreps
    ):
        super().__init__()
        self.num_heads = num_heads
        self.irreps_key = o3.Irreps(irreps_key)
        irreps_value = o3.Irreps(irreps_value)
        irreps_query = o3.Irreps(irreps_query)
        
        self.key_irreps_head = scale_irreps(irreps_key, 1/num_heads)
        self.value_irreps_head = scale_irreps(irreps_value, 1/num_heads)
        self.query_irreps_head = scale_irreps(irreps_query, 1/num_heads)
        
        self.unfuse_key = VectorToAttentionHeads(self.key_irreps_head, num_heads)
        self.unfuse_value = VectorToAttentionHeads(self.value_irreps_head, num_heads)
        self.unfuse_query = VectorToAttentionHeads(self.query_irreps_head, num_heads)
        
        self.fuse_value = AttentionHeadsToVector(self.value_irreps_head)
    
    def forward(
        self, 
        key: torch.Tensor,  # (num_edges, hidden_feat_len)
        value: torch.Tensor, # (num_edges, hidden_feat_len) 
        query: torch.Tensor,  # (num_edges, hidden_feat_len) 
        edge_weight_cutoff: torch.Tensor, # (num_edges,)
        edge_index: torch.LongTensor
    ) -> torch.Tensor:
        """
        Forward pass of the attention mechanism.

        Parameters:
        - key (torch.Tensor): Key vectors.
        - value (torch.Tensor): Value vectors.
        - query (torch.Tensor): Query vectors.
        - edge_weight_cutoff (torch.Tensor): Cutoff weights for edges.
        - edge_index (torch.LongTensor): Edge indices.

        Returns:
        - torch.Tensor: Attended output vectors.
        """
        key = self.unfuse_key(key)
        value = self.unfuse_value(value)
        query = self.unfuse_query(query)
        
        edge_src, edge_dst = edge_index
        
        # Compute the attention weights per edge
        edge_weights = (query * key).sum(-1)  # (num_edges, num_heads)
        if edge_weight_cutoff is not None:
            edge_weights = edge_weight_cutoff[:, None] * edge_weights  # (num_edges, num_heads)
        edge_weights = edge_weights / math.sqrt(self.key_irreps_head.dim)
        edge_weights = edge_softmax(edge_weights, edge_dst)  # (num_edges, num_heads)
        edge_weights = edge_weights.unsqueeze(-1)  # (num_edges, num_heads, 1)

        # Compute the attended outputs per node
        f_out = scatter(edge_weights * value, edge_dst, dim=0)  # (num_nodes, num_heads, irreps_head)
        f_out = self.fuse_value(f_out)  # Merge heads
        return f_out

@compile_mode("script")    
class AttentionBlockE3(nn.Module):
    """
    An equivariant attention block for processing graph data with attention mechanisms.
    
    Parameters:
    - irreps_in (o3.Irreps): Input irreducible representations.
    - irreps_out (o3.Irreps): Output irreducible representations.
    - irreps_node_attrs (o3.Irreps): Node attribute irreducible representations.
    - irreps_edge_attrs (o3.Irreps): Edge attribute irreducible representations.
    - irreps_edge_embed (o3.Irreps): Edge embedding irreducible representations.
    - num_heads (int): Number of attention heads.
    - max_radius (float): Maximum radius for edge cutoff.
    - radial_MLP (Optional[List[int]]): Architecture of the radial MLP.
    - use_skip_connections (bool): Whether to use skip connections.
    - use_kan (bool): Whether to use KAN for radial MLP.
    - nonlinearity_type (str): Type of nonlinearity ('gate' or 'norm').
    - nonlinearity_scalars (Dict[int, Callable]): Scalar nonlinearity functions.
    - nonlinearity_gates (Dict[int, Callable]): Gate nonlinearity functions.
    """

    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        irreps_node_attrs: o3.Irreps,
        irreps_edge_feats: o3.Irreps,
        irreps_edge_attrs: o3.Irreps,
        irreps_edge_embed: o3.Irreps,
        num_heads: int,
        max_radius: float,
        radial_MLP: Optional[List[int]] = None,
        use_skip_connections: bool = True,
        use_kan: bool = False,
        nonlinearity_type: str = "gate",
        nonlinearity_scalars: Dict[int, Callable] = {"e": "ssp", "o": "tanh"},
        nonlinearity_gates: Dict[int, Callable] = {"e": "ssp", "o": "abs"},
    ):
        super().__init__()
        self.radial_MLP = radial_MLP or [64, 64, 64]
        self.use_kan = use_kan
        self.use_skip_connections = use_skip_connections

        assert nonlinearity_type in ("gate", "norm"), "Invalid nonlinearity type."

        # Convert nonlinearity mappings
        nonlinearity_scalars = {
            1: nonlinearity_scalars["e"],
            -1: nonlinearity_scalars["o"],
        }
        nonlinearity_gates = {
            1: nonlinearity_gates["e"],
            -1: nonlinearity_gates["o"],
        }

        # Assign irreps
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_edge_attrs = o3.Irreps(irreps_edge_attrs)
        self.irreps_edge_embed = o3.Irreps(irreps_edge_embed)
        self.irreps_edge_feats = o3.Irreps(irreps_edge_feats)

        self.irreps_node_attrs = o3.Irreps(irreps_node_attrs)

        self.register_buffer(
            "max_radius", torch.tensor(max_radius, dtype=torch.get_default_dtype())
        )
        self.cutoff_func = SoftUnitStepCutoff(cutoff=max_radius)
        
        # Linear transformations
        self.linear_up_src = self.create_linear(self.irreps_in)
        self.linear_up_tar = self.create_linear(self.irreps_in)
        self.linear_up_edge = self.create_linear(self.irreps_in)

        # Nonlinearity
        self.residual = ResidualBlock(self.irreps_in, self.irreps_out)

        # Create TensorProducts for value        
        self.conv_tp_value = MessagePackBlock(irreps_node_feats=self.irreps_in,
                                            irreps_edge_feats=self.irreps_edge_feats,
                                            irreps_local_env_edge=self.irreps_edge_attrs,
                                            irreps_out=self.irreps_out,
                                            irreps_edge_scalars=self.irreps_edge_embed,
                                            radial_MLP=self.radial_MLP,
                                            use_kan=self.use_kan)
        
        # Linear layers for key, query, and value
        self.linear_key = self.create_linear(self.irreps_in, self.irreps_in)
        self.linear_query = self.create_linear(self.irreps_in, self.irreps_in)

        # Attention mechanism
        self.attention = AttentionAggregation(
            num_heads=num_heads,
            irreps_key=self.irreps_in,
            irreps_value=self.irreps_in,
            irreps_query=self.irreps_in,
        )
        
        # Skip connection
        if self.use_skip_connections:
            self.skip_linear = self.create_linear(self.irreps_in, self.irreps_out)

    def create_linear(self, irreps_in, irreps_out=None):
        """Create a linear layer."""
        return o3.Linear(
            irreps_in, irreps_out or irreps_in, internal_weights=True, shared_weights=True
        )

    def create_tensor_product(self, irreps_mid, instructions):
        """Create a TensorProduct layer."""
        return o3.TensorProduct(
            self.irreps_in,
            self.irreps_edge_attrs,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )

    def init_weight_generator(self, input_dim, weight_numel):
        """Initialize weight generator."""
        if self.use_kan:
            return KAN([input_dim] + self.radial_MLP + [weight_numel], grid_size=GRID_SIZE, grid_range=GRID_RANGE)
        return FullyConnectedNet(
            [input_dim] + self.radial_MLP + [weight_numel],
            torch.nn.functional.silu,
        )

    def create_nonlinearity(self, nonlinearity_type, nonlinearity_scalars, nonlinearity_gates):
        """Create nonlinearity module."""
        if nonlinearity_type == "gate":
            irreps_scalars, irreps_gates, irreps_gated, act_scalars, act_gates = irreps2gate(
                self.irreps_in, nonlinearity_scalars, nonlinearity_gates
            )
            return Gate(
                irreps_scalars=irreps_scalars,
                act_scalars=act_scalars,
                irreps_gates=irreps_gates,
                act_gates=act_gates,
                irreps_gated=irreps_gated,
            )
        return NormActivation(
            irreps_in=self.irreps_in,
            scalar_nonlinearity=acts[nonlinearity_scalars[1]],
            normalize=True,
            epsilon=1e-8,
            bias=False,
        )

    def forward(
        self,
        data: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the attention block.

        Parameters:
        - data (Dict[str, torch.Tensor]): A dictionary containing the graph data.

        Returns:
        - Tuple[torch.Tensor, Optional[torch.Tensor]]: Updated node features and skip connection.
        """
        sender, receiver = data[AtomicDataDict.EDGE_INDEX_KEY]
        node_feats = data[AtomicDataDict.NODE_FEATURES_KEY]
        edge_embed = data[AtomicDataDict.EDGE_EMBEDDING_KEY]
        edge_attrs = data[AtomicDataDict.EDGE_ATTRS_KEY]
        edge_feats = data[AtomicDataDict.EDGE_FEATURES_KEY]
        
        # Skip connection
        sc = self.skip_linear(node_feats) if self.use_skip_connections else None

        # Process key, query, and value
        key = self.linear_key(node_feats)[sender]
        query = self.linear_key(node_feats)[receiver]
        
        value = self.conv_tp_value(self.linear_up_src(node_feats)[sender], 
                                   self.linear_up_tar(node_feats)[receiver],  
                                   self.linear_up_edge(edge_feats),
                                   edge_attrs, 
                                   edge_embed)

        # Attention mechanism      
        edge_weight_cutoff = self.cutoff_func(data[AtomicDataDict.EDGE_LENGTH_KEY])
        node_feats = self.attention(key, value, query, edge_weight_cutoff, edge_index=data[AtomicDataDict.EDGE_INDEX_KEY])

        # Apply nonlinearity
        node_feats = self.residual(node_feats)

        # Apply skip connection if used
        if self.use_skip_connections:
            node_feats += sc  

        data[AtomicDataDict.NODE_FEATURES_KEY] = node_feats

        return node_feats

@compile_mode("script")
class PairInteractionEmbeddingBlock(nn.Module):
    """
    A pair interaction block for updating edge features based on node features and edge attributes.

    Parameters:
    - irreps_node_feats (o3.Irreps): Irreducible representations for node features.
    - irreps_edge_attrs (o3.Irreps): Irreducible representations for edge attributes.
    - irreps_edge_embed (o3.Irreps): Irreducible representations for edge embeddings.
    - irreps_edge_feats (o3.Irreps): Irreducible representations for edge features.
    - use_skip_connections (bool): Whether to use skip connections.
    - use_kan (bool): Whether to use KAN for radial MLP.
    - radial_MLP (Optional[List[int]]): Architecture of the radial MLP.
    - nonlinearity_type (str): Type of nonlinearity to use ("gate" or "norm").
    - nonlinearity_scalars (Dict[int, Callable]): Nonlinearity for scalar channels.
    - nonlinearity_gates (Dict[int, Callable]): Nonlinearity for gate channels.
    """

    def __init__(
        self,
        irreps_node_feats: o3.Irreps,
        irreps_edge_attrs: o3.Irreps,
        irreps_node_attrs: o3.Irreps,
        irreps_edge_embed: o3.Irreps,
        irreps_edge_feats: o3.Irreps,
        use_kan: bool = False,
        radial_MLP: Optional[List[int]] = None,
        nonlinearity_type: str = "gate",
        nonlinearity_scalars: Dict[int, Callable] = {"e": "ssp", "o": "tanh"},
        nonlinearity_gates: Dict[int, Callable] = {"e": "ssp", "o": "abs"},
    ) -> None:
        super().__init__()

        self.radial_MLP = radial_MLP or [64, 64, 64]
        self.use_kan = use_kan

        # Assign irreps
        self.irreps_node_feats = o3.Irreps(irreps_node_feats)
        self.irreps_edge_attrs = o3.Irreps(irreps_edge_attrs)
        self.irreps_edge_embed = o3.Irreps(irreps_edge_embed)
        self.irreps_edge_feats = o3.Irreps(irreps_edge_feats)
        self.irreps_node_attrs = o3.Irreps(irreps_node_attrs)

        assert nonlinearity_type in ("gate", "norm"), "Invalid nonlinearity type."

        # Convert nonlinearity mappings
        nonlinearity_scalars = {
            1: nonlinearity_scalars["e"],
            -1: nonlinearity_scalars["o"],
        }
        nonlinearity_gates = {
            1: nonlinearity_gates["e"],
            -1: nonlinearity_gates["o"],
        }

        # Linear layers for lifting node features
        self.linear_up_src = self.create_linear(self.irreps_node_feats)
        self.linear_up_dst = self.create_linear(self.irreps_node_feats)

        # TensorProduct layer for edge feature mixing
        self.conv_tp = TensorProductWithMemoryOptimizationWithWeight(irreps_input_1=self.irreps_node_feats, 
                                                                      irreps_input_2=self.irreps_edge_attrs, 
                                                                      irreps_out=self.irreps_edge_feats, 
                                                                      irreps_scalar=self.irreps_edge_embed, 
                                                                      radial_MLP=self.radial_MLP, 
                                                                      use_kan=self.use_kan)

    def create_linear(self, irreps_in, irreps_out=None):
        """Create a linear layer."""
        return o3.Linear(
            irreps_in, irreps_out or irreps_in, internal_weights=True, shared_weights=True
        )

    def create_tensor_product(self, irreps_mid, instructions):
        """Create a TensorProduct layer."""
        return o3.TensorProduct(
            self.irreps_node_feats,
            self.irreps_edge_attrs,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )

    def init_weight_generator(self, input_dim, weight_numel):
        """Initialize weight generator."""
        if self.use_kan:
            return KAN([input_dim] + self.radial_MLP + [weight_numel], grid_size=GRID_SIZE, grid_range=GRID_RANGE)
        return FullyConnectedNet(
            [input_dim] + self.radial_MLP + [weight_numel],
            torch.nn.functional.silu,
        )

    def forward(
        self,
        data: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward pass of the pair interaction block.

        Parameters:
        - data (Dict[str, torch.Tensor]): A dictionary containing the graph data.

        Returns:
        - torch.Tensor: Updated edge features.
        """
        edge_src, edge_dst = data[AtomicDataDict.EDGE_INDEX_KEY]
        node_feats = data[AtomicDataDict.NODE_FEATURES_KEY]
        edge_embed = data[AtomicDataDict.EDGE_EMBEDDING_KEY]
        edge_attributes = data[AtomicDataDict.EDGE_ATTRS_KEY]
        
        node_feats_src = self.linear_up_src(node_feats[edge_src])
        node_feats_dst = self.linear_up_dst(node_feats[edge_dst])

        # Mixing node features for edge features
        edge_feats_mix_tp = self.conv_tp(
            node_feats_src + node_feats_dst, edge_attributes, edge_embed
        )

        data[AtomicDataDict.EDGE_FEATURES_KEY] = edge_feats_mix_tp
        return edge_feats_mix_tp

@compile_mode("script")
class PairInteractionBlock(nn.Module):
    """
    A pair interaction block for updating edge features based on node features and edge attributes.

    Parameters:
    - irreps_node_feats (o3.Irreps): Irreducible representations for node features.
    - irreps_edge_attrs (o3.Irreps): Irreducible representations for edge attributes.
    - irreps_edge_embed (o3.Irreps): Irreducible representations for edge embeddings.
    - irreps_edge_feats (o3.Irreps): Irreducible representations for edge features.
    - use_skip_connections (bool): Whether to use skip connections.
    - use_kan (bool): Whether to use KAN for radial MLP.
    - radial_MLP (Optional[List[int]]): Architecture of the radial MLP. Defaults to [64, 64, 64].
    - nonlinearity_type (str): Type of nonlinearity to use ("gate" or "norm").
    - nonlinearity_scalars (Dict[int, Callable]): Nonlinearity for scalar channels.
    - nonlinearity_gates (Dict[int, Callable]): Nonlinearity for gate channels.
    """

    def __init__(
        self,
        irreps_node_feats: o3.Irreps,
        irreps_node_attrs: o3.Irreps,
        irreps_edge_attrs: o3.Irreps,
        irreps_edge_embed: o3.Irreps,
        irreps_edge_feats: o3.Irreps,
        use_skip_connections: bool = False,
        use_kan: bool = False,
        radial_MLP: Optional[List[int]] = None,
        nonlinearity_type: str = "gate",
        nonlinearity_scalars: Dict[int, Callable] = {"e": "ssp", "o": "tanh"},
        nonlinearity_gates: Dict[int, Callable] = {"e": "ssp", "o": "abs"},
    ) -> None:
        super().__init__()

        self.radial_MLP = radial_MLP or [64, 64, 64]
        self.use_skip_connections = use_skip_connections
        self.use_kan = use_kan

        # Assign irreps
        self.irreps_node_feats = o3.Irreps(irreps_node_feats)
        self.irreps_edge_attrs = o3.Irreps(irreps_edge_attrs)
        self.irreps_edge_embed = o3.Irreps(irreps_edge_embed)
        self.irreps_edge_feats = o3.Irreps(irreps_edge_feats)
        self.irreps_node_attrs = o3.Irreps(irreps_node_attrs)

        assert nonlinearity_type in ("gate", "norm"), "Invalid nonlinearity type."

        # Convert nonlinearity mappings
        scalar_nonlinearities = {
            1: nonlinearity_scalars["e"],
            -1: nonlinearity_scalars["o"],
        }
        gate_nonlinearities = {
            1: nonlinearity_gates["e"],
            -1: nonlinearity_gates["o"],
        }

        # Linear transformations
        self.linear_up_src = self.create_linear(self.irreps_node_feats)
        self.linear_up_tar = self.create_linear(self.irreps_node_feats)

        # TensorProduct layer for edge feature mixing
        self.conv_tp = MessagePackBlock(
            irreps_node_feats=self.irreps_node_feats,
            irreps_edge_feats=self.irreps_edge_feats,
            irreps_local_env_edge=self.irreps_edge_attrs,
            irreps_out=self.irreps_edge_feats,
            irreps_edge_scalars=self.irreps_edge_embed, 
            radial_MLP=self.radial_MLP, 
            use_kan=self.use_kan
            )

        # Skip connection
        if self.use_skip_connections:
            self.skip_linear = self.create_linear(irreps_edge_feats, irreps_edge_feats)

    def create_linear(self, irreps_in, irreps_out=None):
        """
        Create a linear layer.

        Parameters:
        - irreps_in (o3.Irreps): Input irreps for the linear layer.
        - irreps_out (o3.Irreps, optional): Output irreps for the linear layer.

        Returns:
        - o3.Linear: A linear transformation layer.
        """
        return o3.Linear(
            irreps_in, irreps_out or irreps_in, internal_weights=True, shared_weights=True
        )

    def forward(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of the pair interaction block.

        Parameters:
        - data (Dict[str, torch.Tensor]): A dictionary containing the graph data.

        Returns:
        - torch.Tensor: Updated edge features.
        """
        edge_src, edge_dst = data[AtomicDataDict.EDGE_INDEX_KEY]
        node_feats = data[AtomicDataDict.NODE_FEATURES_KEY]
        edge_embed = data[AtomicDataDict.EDGE_EMBEDDING_KEY]
        edge_feats = data[AtomicDataDict.EDGE_FEATURES_KEY]

        # Mixing node features for edge features       
        edge_feats_mix = self.conv_tp(
            self.linear_up_src(node_feats)[edge_src], 
            self.linear_up_tar(node_feats)[edge_dst], 
            edge_feats, 
            data[AtomicDataDict.EDGE_ATTRS_KEY], 
            edge_embed
        )
        
        if self.use_skip_connections:
            edge_feats = edge_feats_mix + self.skip_linear(edge_feats)

        data[AtomicDataDict.EDGE_FEATURES_KEY] = edge_feats
        
        return edge_feats

@compile_mode("script")
class CorrProductBlock(nn.Module):
    """
    A correlation product block for updating node features using an equivariant product operation.

    Parameters:
    - irreps_node_feats (o3.Irreps): Irreducible representations for node features.
    - num_hidden_features (int): Number of hidden features.
    - correlation (int): Correlation level for the product operation.
    - use_skip_connections (bool): Whether to use skip connections.
    - num_elements (int): Number of elements for the product operation.
    """

    def __init__(
        self,
        irreps_node_feats: o3.Irreps,
        num_hidden_features: int,
        correlation: int,
        use_skip_connections: bool = True,
        num_elements: Optional[int] = None
    ) -> None:
        super().__init__()

        self.irreps_node_feats = o3.Irreps(irreps_node_feats).simplify()
        self.num_hidden_features = num_hidden_features
        self.correlation = correlation
        self.use_skip_connections = use_skip_connections
        self.num_elements = num_elements

        self.irreps_hidden_features = o3.Irreps(
            [(self.num_hidden_features, irrep.ir) for irrep in self.irreps_node_feats]
        )

        # Linear layers for lifting and skip connection
        self.linear_pre = o3.Linear(
            self.irreps_node_feats,
            self.irreps_hidden_features,
            internal_weights=True,
            shared_weights=True,
        )
        self.linear_sc = o3.Linear(
            self.irreps_node_feats,
            self.irreps_node_feats,
            internal_weights=True,
            shared_weights=True,
        )

        # Equivariant product operation
        self.prod = EquivariantProductBasisBlock(
            node_feats_irreps=self.irreps_hidden_features,
            target_irreps=self.irreps_hidden_features,
            correlation=correlation,
            num_elements=num_elements,
            use_sc=False,
        )

        # Linear layer for output
        self.linear_out = o3.Linear(
            self.irreps_hidden_features,
            self.irreps_node_feats,
            internal_weights=True,
            shared_weights=True,
        )
        
        self.reshape = reshape_irreps(self.irreps_hidden_features)

    def forward(
        self,
        data: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Forward pass of the correlation product block.

        Parameters:
        - data (Dict[str, torch.Tensor]): A dictionary containing the graph data.

        Returns:
        - torch.Tensor: Updated node features.
        """
        node_feats = self.linear_pre(data[AtomicDataDict.NODE_FEATURES_KEY])
        node_feats = self.reshape(node_feats) # [n_nodes, channels, (l + 1)**2]

        out = self.prod(node_feats, None, data[AtomicDataDict.NODE_ATTRS_KEY])
        out = self.linear_out(out)

        if self.use_skip_connections:
            sc = self.linear_sc(data[AtomicDataDict.NODE_FEATURES_KEY])
            data[AtomicDataDict.NODE_FEATURES_KEY] = out + sc
        else:
            data[AtomicDataDict.NODE_FEATURES_KEY] = out

        return out

@compile_mode("script")
class ResidualBlock(nn.Module):
    """
    A residual block used in equivariant neural networks.
    
    Args:
        irreps_in (str): The input irreducible representations (irreps).
        feature_irreps_hidden (str): The hidden feature irreps.
        resnet (bool): If True, apply a residual connection.
        nonlinearity_type (str): The type of nonlinearity to apply ('gate' or 'norm').
        nonlinearity_scalars (Dict[int, Callable]): A dictionary mapping parity to nonlinearity functions for scalar features.
        nonlinearity_gates (Dict[int, Callable]): A dictionary mapping parity to nonlinearity functions for gated features.
    """

    def __init__(
        self,
        irreps_in: str,
        feature_irreps_hidden: str,
        resnet: bool = True,
        nonlinearity_type: str = "gate",
        nonlinearity_scalars: Dict[int, Callable] = {"e": "ssp", "o": "tanh"},
        nonlinearity_gates: Dict[int, Callable] = {"e": "ssp", "o": "abs"},
    ):
        super().__init__()
        
        # Ensure valid nonlinearity type
        assert nonlinearity_type in ("gate", "norm"), "Invalid nonlinearity_type. Choose either 'gate' or 'norm'."

        # Convert scalar and gate nonlinearity based on parity
        nonlinearity_scalars = {1: nonlinearity_scalars["e"], -1: nonlinearity_scalars["o"]}
        nonlinearity_gates = {1: nonlinearity_gates["e"], -1: nonlinearity_gates["o"]}

        self.irreps_in = o3.Irreps(irreps_in)
        self.feature_irreps_hidden = o3.Irreps(feature_irreps_hidden)
        self.resnet = resnet
        
        self.equivariant_nonlin = self.create_nonlinearity(nonlinearity_type, self.feature_irreps_hidden, nonlinearity_scalars, nonlinearity_gates)
        
        # Define linear layers
        self.linear1 = o3.Linear(irreps_in=self.irreps_in, irreps_out=self.equivariant_nonlin.irreps_in)
        self.linear2 = o3.Linear(irreps_in=self.equivariant_nonlin.irreps_out, irreps_out=irreps_in)

    def create_nonlinearity(self, nonlinearity_type, irreps_mid, nonlinearity_scalars, nonlinearity_gates):
        """Create nonlinearity module."""
        if nonlinearity_type == "gate":
            irreps_scalars, irreps_gates, irreps_gated, act_scalars, act_gates = irreps2gate(
                irreps_mid, nonlinearity_scalars, nonlinearity_gates
            )
            return Gate(
                irreps_scalars=irreps_scalars,
                act_scalars=act_scalars,
                irreps_gates=irreps_gates,
                act_gates=act_gates,
                irreps_gated=irreps_gated,
            )
        return NormActivation(
            irreps_in=irreps_mid,
            scalar_nonlinearity=acts[nonlinearity_scalars[1]],
            normalize=True,
            epsilon=1e-8,
            bias=False,
        )

    def forward(self, x):
        """
        Forward pass of the residual block.
        
        Args:
            x (torch.Tensor): Input tensor with shape matching `irreps_in`.
        
        Returns:
            torch.Tensor: Output tensor with shape matching `irreps_in`.
        """
        # Store old input for resnet connection if applicable
        old_x = x
        
        # Apply first linear transformation
        x = self.linear1(x)
        
        # Apply nonlinearity
        x = self.equivariant_nonlin(x)
        
        # Apply second linear transformation
        x = self.linear2(x)
        
        # Apply residual connection if resnet is enabled
        if self.resnet:
            x = old_x + x
            
        return x

@compile_mode("script")
class HamLayer(nn.Module):
    def __init__(self, irreps_in, feature_irreps_hidden, irreps_out, nonlinearity_type: str = "gate", resnet: bool = True):
        super().__init__()
        
        # Define the residual block
        self.residual_block = ResidualBlock(irreps_in=irreps_in, 
                                            feature_irreps_hidden=feature_irreps_hidden, 
                                            nonlinearity_type=nonlinearity_type, 
                                            resnet=resnet)
        
        # Define the linear transformation
        self.linear_transform = o3.Linear(irreps_in=irreps_in, irreps_out=irreps_out)
    
    def forward(self, x):
        # Apply the residual block
        x = self.residual_block(x)
        
        # Apply the linear transformation
        x = self.linear_transform(x)
        
        return x
