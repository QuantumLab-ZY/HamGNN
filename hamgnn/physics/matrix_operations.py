import torch
import torch.nn as nn
import e3nn.o3 as o3
from e3nn.util.jit import compile_mode
from typing import Optional
from math import prod

from ..nn.interaction_blocks import ResidualBlock
from .Clebsch_Gordan_coefficients import ClebschGordanCoefficients

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

class TensorMerge(nn.Module):
    def __init__(self, irrep_in, irrep_out_1, irrep_out_2, internal_weights: Optional[bool] = False):
        super().__init__()
        self.irrep_in = irrep_in
        self.irrep_out_1 = irrep_out_1
        self.irrep_out_2 = irrep_out_2
        self.instructions = self.get_expansion_path(irrep_in, irrep_out_1, irrep_out_2)
        self.num_path_weight = sum(prod(ins[-1]) for ins in self.instructions if ins[3])
        self.num_bias = sum([prod(ins[-1][1:]) for ins in self.instructions if ins[0] == 0])
        self.num_weights = self.num_path_weight + self.num_bias
        self.internal_weights = internal_weights
        if self.internal_weights:
            self.weights = nn.Parameter(torch.rand(self.num_path_weight + self.num_bias))
        else:
            self.linear_weight_bias = o3.Linear(self.irrep_in, o3.Irreps([(self.num_weights, (0, 1))]))
    
    def forward(self, x_in):
        if self.internal_weights:
            weights, bias_weights = None
        else:
            weights, bias_weights = torch.split(self.linear_weight_bias(x_in),
                                               split_size_or_sections=[self.num_path_weight, self.num_bias], dim=-1)
        batch_num = x_in.shape[0]
        if len(self.irrep_in) == 1:
            x_in_s = [x_in.reshape(batch_num, self.irrep_in[0].mul, self.irrep_in[0].ir.dim)]
        else:
            x_in_s = [
                x_in[:, i].reshape(batch_num, mul_ir.mul, mul_ir.ir.dim)
            for i, mul_ir in zip(self.irrep_in.slices(), self.irrep_in)]
        outputs = {}
        flat_weight_index = 0
        bias_weight_index = 0
        for ins in self.instructions:
            mul_ir_in = self.irrep_in[ins[0]]
            mul_ir_out1 = self.irrep_out_1[ins[1]]
            mul_ir_out2 = self.irrep_out_2[ins[2]]
            x1 = x_in_s[ins[0]]
            x1 = x1.reshape(batch_num, mul_ir_in.mul, mul_ir_in.ir.dim)
            w3j_matrix = o3.wigner_3j(
                mul_ir_out1.ir.l, mul_ir_out2.ir.l, mul_ir_in.ir.l).type_as(x_in)
            if ins[3] is True or weights is not None:
                if weights is None:
                    weight = self.weights[flat_weight_index:flat_weight_index + prod(ins[-1])].reshape(ins[-1])
                    result = torch.einsum(
                        f"wuv, ijk, bwk-> buivj", weight, w3j_matrix, x1) / mul_ir_in.mul
                else:
                    weight = weights[:, flat_weight_index:flat_weight_index + prod(ins[-1])].reshape([-1] + ins[-1])
                    result = torch.einsum(f"bwuv, bwk-> buvk", weight, x1)
                    if ins[0] == 0 and bias_weights is not None:
                        bias_weight = bias_weights[:,bias_weight_index:bias_weight_index + prod(ins[-1][1:])].\
                            reshape([-1] + ins[-1][1:])
                        bias_weight_index += prod(ins[-1][1:])
                        result = result + bias_weight.unsqueeze(-1)
                    result = torch.einsum(f"ijk, buvk->buivj", w3j_matrix, result) / mul_ir_in.mul
                flat_weight_index += prod(ins[-1])
            else:
                result = torch.einsum(
                    f"uvw, ijk, bwk-> buivj", torch.ones(ins[-1]).type(x1.type()).to(self.device), w3j_matrix,
                    x1.reshape(batch_num, mul_ir_in.mul, mul_ir_in.ir.dim)
                )
            result = result.reshape(batch_num, mul_ir_out1.dim, mul_ir_out2.dim)
            key = (ins[1], ins[2])
            if key in outputs.keys():
                outputs[key] = outputs[key] + result
            else:
                outputs[key] = result
        rows = []
        for i in range(len(self.irrep_out_1)):
            blocks = []
            for j in range(len(self.irrep_out_2)):
                if (i, j) not in outputs.keys():
                    blocks += [torch.zeros((x_in.shape[0], self.irrep_out_1[i].dim, self.irrep_out_2[j].dim),
                                           device=x_in.device).type(x_in.type())]
                else:
                    blocks += [outputs[(i, j)]]
            rows.append(torch.cat(blocks, dim=-1))
        output = torch.cat(rows, dim=-2).reshape(batch_num, -1)
        return output
    
    def get_expansion_path(self, irrep_in, irrep_out_1, irrep_out_2):
        instructions = []
        for  i, (num_in, ir_in) in enumerate(irrep_in):
            for  j, (num_out1, ir_out1) in enumerate(irrep_out_1):
                for k, (num_out2, ir_out2) in enumerate(irrep_out_2):
                    if ir_in in ir_out1 * ir_out2:
                        instructions.append([i, j, k, True, 1.0, [num_in, num_out1, num_out2]])
        return instructions
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    def __repr__(self):
        return f'{self.irrep_in} -> {self.irrep_out_1}x{self.irrep_out_1} and bias {self.num_bias}' \
               f'with parameters {self.num_path_weight}'