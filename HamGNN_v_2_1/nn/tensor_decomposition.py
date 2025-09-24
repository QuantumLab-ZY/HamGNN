# Modified code based on [Nat. Commun. 14 2848 (2023)] for testing purposes.
import torch
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.utils import degree
from e3nn.o3 import Irrep, Irreps, wigner_3j, matrix_to_angles
from e3nn.nn import Extract
import sympy as sym
import numpy as np
from scipy.optimize import brentq
from scipy import special as sp
import math

def convert_float_to_complex_dtype(float_dtype):
    """Convert a float data type to the corresponding complex data type.
    
    Args:
        float_dtype (torch.dtype or numpy.dtype): Float data type to convert.
        
    Returns:
        torch.dtype or numpy.dtype: Corresponding complex data type.
        
    Raises:
        NotImplementedError: If the float data type is not supported.
    """
    if float_dtype == torch.float32:
        complex_dtype = torch.complex64
    elif float_dtype == torch.float64:
        complex_dtype = torch.complex128
    elif float_dtype == np.float32:
        complex_dtype = np.complex64
    elif float_dtype == np.float64:
        complex_dtype = np.complex128
    else:
        raise NotImplementedError(f'Unsupported float dtype: {float_dtype}')
    return complex_dtype

def irreps_from_l1l2(l1, l2, multiplicity, spinful, no_parity=False):
    """Generate irreducible representations from two angular momentum quantum numbers.
    
    This function computes the required irreducible representations for tensor products
    of two angular momentum states with quantum numbers l1 and l2.
    
    Args:
        l1 (int): First angular momentum quantum number.
        l2 (int): Second angular momentum quantum number.
        multiplicity (int): Multiplicity of the irreps.
        spinful (bool): Whether to include spin-1/2 coupling.
        no_parity (bool, optional): If True, ignore parity. Defaults to False.
        
    Returns:
        tuple: A tuple containing:
            - required_irreps_full: All required irreps.
            - required_irreps: Basic required irreps.
            - required_irreps_x1: List of irreps coupled with spin-1/2 (None if not spinful).
            
    Examples:
        Non-spinful example: l1=1, l2=2 (1x2) ->
        required_irreps_full=1+2+3, required_irreps=1+2+3, required_irreps_x1=None
        
        Spinful example: l1=1, l2=2 (1x0.5)x(2x0.5) ->
        required_irreps_full = 1+2+3 + 0+1+2 + 1+2+3 + 2+3+4
        required_irreps = (1+2+3)x0 = 1+2+3
        required_irreps_x1 = (1+2+3)x1 = [0+1+2, 1+2+3, 2+3+4]
    """
    parity = 1
    if not no_parity:
        parity = (-1) ** (l1 + l2)
    
    # Range of L values from |l1-l2| to l1+l2
    allowed_l_values = range(abs(l1 - l2), l1 + l2 + 1)
    required_irreps = Irreps([(multiplicity, (l, parity)) for l in allowed_l_values])
    required_irreps_full = required_irreps
    required_irreps_x1 = None
    
    if spinful:
        required_irreps_x1 = []
        for _, ir in required_irreps:
            # For each irrep, couple with spin-1/2 (l=1)
            spin_coupled_ls = range(abs(ir.l - 1), ir.l + 1 + 1)
            irx1 = Irreps([(multiplicity, (l, parity)) for l in spin_coupled_ls])
            required_irreps_x1.append(irx1)
            required_irreps_full += irx1
            
    return required_irreps_full, required_irreps, required_irreps_x1

class Rotate:
    """Class for handling rotations in 3D space with support for spherical harmonics.
    
    This class provides utilities for rotating tensors according to irreducible 
    representations of the rotation group SO(3).
    
    Attributes:
        spinful (bool): Whether to include spin-1/2 coupling.
        Us_openmx (dict): Transformation matrices from OpenMX to complex spherical harmonics.
        Us_openmx2wiki (dict): Transformation matrices from OpenMX to Wikipedia real spherical harmonics.
        Us_wiki2openmx (dict): Transformation matrices from Wikipedia to OpenMX real spherical harmonics.
        Us_openmx2wiki_sp (dict, optional): Spinful version of transformation matrices.
        dtype (torch.dtype): Default data type for tensors.
    """
    def __init__(self, default_dtype_torch, device_torch='cpu', spinful=False):
        """Initialize the rotation handler.
        
        Args:
            default_dtype_torch (torch.dtype): Default data type for tensors.
            device_torch (str, optional): Device for tensors. Defaults to 'cpu'.
            spinful (bool, optional): Whether to include spin-1/2 coupling. Defaults to False.
        """
        sqrt_2 = 1.4142135623730951
            
        self.spinful = spinful
        if spinful:
            assert default_dtype_torch in [torch.complex64, torch.complex128]
        else:
            assert default_dtype_torch in [torch.float32, torch.float64]
        
        # OpenMX real spherical harmonics to complex spherical harmonics
        self.Us_openmx = {
            0: torch.tensor([1], dtype=torch.cfloat, device=device_torch),
            1: torch.tensor([[-1 / sqrt_2, 1j / sqrt_2, 0], [0, 0, 1], [1 / sqrt_2, 1j / sqrt_2, 0]], dtype=torch.cfloat, device=device_torch),
            2: torch.tensor([[0, 1 / sqrt_2, -1j / sqrt_2, 0, 0],
                             [0, 0, 0, -1 / sqrt_2, 1j / sqrt_2],
                             [1, 0, 0, 0, 0],
                             [0, 0, 0, 1 / sqrt_2, 1j / sqrt_2],
                             [0, 1 / sqrt_2, 1j / sqrt_2, 0, 0]], dtype=torch.cfloat, device=device_torch),
            3: torch.tensor([[0, 0, 0, 0, 0, -1 / sqrt_2, 1j / sqrt_2],
                             [0, 0, 0, 1 / sqrt_2, -1j / sqrt_2, 0, 0],
                             [0, -1 / sqrt_2, 1j / sqrt_2, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0, 0, 0],
                             [0, 1 / sqrt_2, 1j / sqrt_2, 0, 0, 0, 0],
                             [0, 0, 0, 1 / sqrt_2, 1j / sqrt_2, 0, 0],
                             [0, 0, 0, 0, 0, 1 / sqrt_2, 1j / sqrt_2]], dtype=torch.cfloat, device=device_torch),
        }
        # OpenMX real spherical harmonics to Wikipedia real spherical harmonics
        # https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Real_spherical_harmonics
        self.Us_openmx2wiki = {
            0: torch.eye(1, dtype=default_dtype_torch).to(device=device_torch),
            1: torch.eye(3, dtype=default_dtype_torch)[[1, 2, 0]].to(device=device_torch),
            2: torch.eye(5, dtype=default_dtype_torch)[[2, 4, 0, 3, 1]].to(device=device_torch),
            3: torch.eye(7, dtype=default_dtype_torch)[[6, 4, 2, 0, 1, 3, 5]].to(device=device_torch)
        }
        self.Us_wiki2openmx = {k: v.T for k, v in self.Us_openmx2wiki.items()}
        if spinful:
            self.Us_openmx2wiki_sp = {}
            for k, v in self.Us_openmx2wiki.items():
                self.Us_openmx2wiki_sp[k] = torch.block_diag(v, v)
        
        self.dtype = default_dtype_torch 
        
    def rotate_e3nn_v(self, v, R, l, order_xyz=True):
        """Rotate a vector according to the e3nn convention.
        
        Args:
            v (torch.Tensor): Vector to rotate.
            R (torch.Tensor): Rotation matrix.
            l (int): Angular momentum quantum number.
            order_xyz (bool, optional): If True, R is in (x,y,z) order. Defaults to True.
            
        Returns:
            torch.Tensor: Rotated vector.
        """
        if order_xyz:
            # R is in (x, y, z) order
            R_e3nn = self.rotate_matrix_convert(R)
            # R_e3nn is in (y, z, x) order
        else:
            # R is already in (y, z, x) order
            R_e3nn = R
        return v @ Irrep(l, 1).D_from_matrix(R_e3nn)
    
    def rotate_openmx_H(self, H, R, l_left, l_right, order_xyz=True):
        """Rotate a Hamiltonian matrix from OpenMX.
        
        Args:
            H (torch.Tensor): Hamiltonian matrix.
            R (torch.Tensor): Rotation matrix.
            l_left (int): Angular momentum quantum number for left side.
            l_right (int): Angular momentum quantum number for right side.
            order_xyz (bool, optional): If True, R is in (x,y,z) order. Defaults to True.
            
        Returns:
            torch.Tensor: Rotated Hamiltonian matrix.
        """
        if order_xyz:
            # R is in (x, y, z) order
            R_e3nn = self.rotate_matrix_convert(R)
            # R_e3nn is in (y, z, x) order
        else:
            # R is already in (y, z, x) order
            R_e3nn = R
        return self.Us_openmx2wiki[l_left].T @ Irrep(l_left, 1).D_from_matrix(R_e3nn).transpose(-1, -2) @ self.Us_openmx2wiki[l_left] @ H \
               @ self.Us_openmx2wiki[l_right].T @ Irrep(l_right, 1).D_from_matrix(R_e3nn) @ self.Us_openmx2wiki[l_right]
    
    def rotate_openmx_H_full(self, H, R, orbital_types_left, orbital_types_right, order_xyz=True):
        """Rotate a full Hamiltonian matrix from OpenMX.
        
        Args:
            H (torch.Tensor): Hamiltonian matrix.
            R (torch.Tensor): Rotation matrix.
            orbital_types_left (list or int): Angular momentum quantum numbers for left side.
            orbital_types_right (list or int): Angular momentum quantum numbers for right side.
            order_xyz (bool, optional): If True, R is in (x,y,z) order. Defaults to True.
            
        Returns:
            torch.Tensor: Rotated Hamiltonian matrix.
            
        Raises:
            AssertionError: If R is not a 2D tensor.
        """
        assert len(R.shape) == 2  # TODO: does not support batch operation
        if order_xyz:
            # R is in (x, y, z) order
            R_e3nn = self.rotate_matrix_convert(R)
            # R_e3nn is in (y, z, x) order
        else:
            # R is already in (y, z, x) order
            R_e3nn = R
        irreps_left = Irreps([(1, (l, (- 1) ** l)) for l in orbital_types_left])
        irreps_right = Irreps([(1, (l, (- 1) ** l)) for l in orbital_types_right])
        U_left = irreps_left.D_from_matrix(R_e3nn)
        U_right = irreps_right.D_from_matrix(R_e3nn)
        openmx2wiki_left, openmx2wiki_right = self.openmx2wiki_left_right(orbital_types_left, orbital_types_right)
        if self.spinful:
            U_left = torch.kron(self.D_one_half(R_e3nn), U_left)
            U_right = torch.kron(self.D_one_half(R_e3nn), U_right)
        return openmx2wiki_left.T @ U_left.transpose(-1, -2).conj() @ openmx2wiki_left @ H \
               @ openmx2wiki_right.T @ U_right @ openmx2wiki_right
               
    def wiki2openmx_H_full(self, H, orbital_types_left, orbital_types_right):
        """Convert a Hamiltonian from Wikipedia to OpenMX format.
        
        Args:
            H (torch.Tensor): Hamiltonian matrix in Wikipedia format.
            orbital_types_left (list or int): Angular momentum quantum numbers for left side.
            orbital_types_right (list or int): Angular momentum quantum numbers for right side.
            
        Returns:
            torch.Tensor: Hamiltonian matrix in OpenMX format.
        """
        openmx2wiki_left, openmx2wiki_right = self.openmx2wiki_left_right(orbital_types_left, orbital_types_right)
        return openmx2wiki_left.T @ H @ openmx2wiki_right
        
    def openmx2wiki_H_full(self, H, orbital_types_left, orbital_types_right):
        """Convert a Hamiltonian from OpenMX to Wikipedia format.
        
        Args:
            H (torch.Tensor): Hamiltonian matrix in OpenMX format.
            orbital_types_left (list or int): Angular momentum quantum numbers for left side.
            orbital_types_right (list or int): Angular momentum quantum numbers for right side.
            
        Returns:
            torch.Tensor: Hamiltonian matrix in Wikipedia format.
        """
        openmx2wiki_left, openmx2wiki_right = self.openmx2wiki_left_right(orbital_types_left, orbital_types_right)
        return openmx2wiki_left @ H @ openmx2wiki_right.T
    
    def wiki2openmx_H(self, H, l_left, l_right):
        """Convert a Hamiltonian block from Wikipedia to OpenMX format.
        
        Args:
            H (torch.Tensor): Hamiltonian matrix block in Wikipedia format.
            l_left (int): Angular momentum quantum number for left side.
            l_right (int): Angular momentum quantum number for right side.
            
        Returns:
            torch.Tensor: Hamiltonian matrix block in OpenMX format.
        """
        return self.Us_openmx2wiki[l_left].T @ H @ self.Us_openmx2wiki[l_right]
        
    def openmx2wiki_H(self, H, l_left, l_right):
        """Convert a Hamiltonian block from OpenMX to Wikipedia format.
        
        Args:
            H (torch.Tensor): Hamiltonian matrix block in OpenMX format.
            l_left (int): Angular momentum quantum number for left side.
            l_right (int): Angular momentum quantum number for right side.
            
        Returns:
            torch.Tensor: Hamiltonian matrix block in Wikipedia format.
        """
        return self.Us_openmx2wiki[l_left] @ H @ self.Us_openmx2wiki[l_right].T
        
    def openmx2wiki_left_right(self, orbital_types_left, orbital_types_right):
        """Get transformation matrices for left and right sides.
        
        Args:
            orbital_types_left (list or int): Angular momentum quantum numbers for left side.
            orbital_types_right (list or int): Angular momentum quantum numbers for right side.
            
        Returns:
            tuple: Pair of transformation matrices (left, right).
        """
        if isinstance(orbital_types_left, int):
            orbital_types_left = [orbital_types_left]
        if isinstance(orbital_types_right, int):
            orbital_types_right = [orbital_types_right]
        openmx2wiki_left = torch.block_diag(*[self.Us_openmx2wiki[l] for l in orbital_types_left])
        openmx2wiki_right = torch.block_diag(*[self.Us_openmx2wiki[l] for l in orbital_types_right])
        if self.spinful:
            openmx2wiki_left = torch.block_diag(openmx2wiki_left, openmx2wiki_left)
            openmx2wiki_right = torch.block_diag(openmx2wiki_right, openmx2wiki_right)
        return openmx2wiki_left, openmx2wiki_right
    
    def rotate_matrix_convert(self, R):
        """Convert rotation matrix from (x,y,z) order to (y,z,x) order.
        
        Args:
            R (torch.Tensor): Rotation matrix in (x,y,z) order.
            
        Returns:
            torch.Tensor: Rotation matrix in (y,z,x) order.
        """
        # (x, y, z) order rotation matrix to (y, z, x) order
        # See e3nn.o3.spherical_harmonics() and https://docs.e3nn.org/en/stable/guide/change_of_basis.html
        return torch.eye(3)[[1, 2, 0]] @ R @ torch.eye(3)[[1, 2, 0]].T
    
    def D_one_half(self, R):
        """Compute the spin-1/2 representation of a rotation.
        
        Args:
            R (torch.Tensor): Rotation matrix in (y,z,x) order.
            
        Returns:
            torch.Tensor: Spin-1/2 representation matrix.
            
        Note:
            This assumes the l=1/2 irreducible representation has even parity.
            No spatial inversion is considered.
        """
        # Input R should be in y,z,x order
        # Assumes l=1/2 irreducible representation has even parity
        assert self.spinful
        d = torch.det(R).sign()
        R = d[..., None, None] * R
        k = (1 - d) / 2  # parity index
        alpha, beta, gamma = matrix_to_angles(R)
        J = torch.tensor([[1, 1], [1j, -1j]], dtype=self.dtype) / 1.4142135623730951  # <1/2 mz|1/2 my>
        Uz1 = self._sp_z_rot(alpha)
        Uy = J @ self._sp_z_rot(beta) @ J.T.conj()
        Uz2 = self._sp_z_rot(gamma)
        return Uz1 @ Uy @ Uz2
    
    def _sp_z_rot(self, angle):
        """Compute the spin-1/2 representation of a z-axis rotation.
        
        Args:
            angle (torch.Tensor): Rotation angle.
            
        Returns:
            torch.Tensor: Spin-1/2 representation matrix for z-rotation.
            
        Note:
            The matrix has the form [[e^{-ia/2}, 0], [0, e^{ia/2}]].
        """
        assert self.spinful
        M = torch.zeros([*angle.shape, 2, 2], dtype=self.dtype)
        inds = torch.tensor([0, 1])
        freqs = torch.tensor([0.5, -0.5], dtype=self.dtype)
        M[..., inds, inds] = torch.exp(- freqs * (1j) * angle[..., None])
        return M

class SortIrrepsTransform(torch.nn.Module):
    """Module for sorting irreducible representations.
    
    This module extracts and sorts irreducible representations to a canonical order.
    
    Attributes:
        irreps_in (e3nn.o3.Irreps): Input irreducible representations.
        irreps_out (e3nn.o3.Irreps): Output (sorted) irreducible representations.
        extr (e3nn.nn.Extract): Extractor for forward transformation.
        extr_inv (e3nn.nn.Extract): Extractor for inverse transformation.
    """
    def __init__(self, irreps_in):
        """Initialize the irreps sorting transform.
        
        Args:
            irreps_in (e3nn.o3.Irreps or str): Input irreducible representations.
        """
        super().__init__()
        irreps_in = Irreps(irreps_in)
        sorted_irreps = irreps_in.sort()
        
        # Create extractors for forward and inverse transformations
        irreps_out_list = [((mul, ir),) for mul, ir in sorted_irreps.irreps]
        instructions = [(i,) for i in sorted_irreps.inv]
        self.extr = Extract(irreps_in, irreps_out_list, instructions)
        
        irreps_in_list = [((mul, ir),) for mul, ir in irreps_in]
        instructions_inv = [(i,) for i in sorted_irreps.p]
        self.extr_inv = Extract(sorted_irreps.irreps, irreps_in_list, instructions_inv)
        
        self.irreps_in = irreps_in
        self.irreps_out = sorted_irreps.irreps.simplify()
    
    def forward(self, x):
        """Transform from input irreps to sorted irreps.
        
        Args:
            x (torch.Tensor): Input tensor with irreps_in structure.
            
        Returns:
            torch.Tensor: Output tensor with irreps_out structure.
        """
        extracted = self.extr(x)
        return torch.cat(extracted, dim=-1)
        
    def inverse(self, x):
        """Transform from sorted irreps back to input irreps.
        
        Args:
            x (torch.Tensor): Input tensor with irreps_out structure.
            
        Returns:
            torch.Tensor: Output tensor with irreps_in structure.
        """
        extracted_inv = self.extr_inv(x)
        return torch.cat(extracted_inv, dim=-1)

class E3TensorDecomposition:
    """Class for tensor decomposition in E(3) equivariant neural networks.
    
    This class handles the decomposition of tensors according to irreducible
    representations of the 3D rotation group.
    
    Attributes:
        dtype (torch.dtype): Data type for tensors.
        spinful (bool): Whether to include spin-1/2 coupling.
        nao_max (int): Maximum number of atomic orbitals.
        out_js_list (list): List of (l1, l2) pairs for output channels.
        in_slices (list): Slices for input tensor channels.
        wms (list): Wigner multipliers for tensor decomposition.
        H_slices (list): Slices for Hamiltonian tensor.
        wms_H (list): Wigner multipliers for Hamiltonian.
        rotate_kernel (Rotate): Helper for rotation operations.
        sort (SortIrrepsTransform, optional): Irreps sorting transform.
        required_irreps_out (e3nn.o3.Irreps): Required output irreps.
    """
    def __init__(self, net_irreps_out, out_js_list, default_dtype_torch=torch.float32, 
                 nao_max=26, spinful=False, no_parity=False, if_sort=False):
        """Initialize the tensor decomposition.
        
        Args:
            net_irreps_out (e3nn.o3.Irreps or str): Output irreducible representations.
            out_js_list (list): List of (l1, l2) pairs for output channels.
            default_dtype_torch (torch.dtype, optional): Default data type. Defaults to torch.float32.
            nao_max (int, optional): Maximum number of atomic orbitals. Defaults to 26.
            spinful (bool, optional): Whether to include spin-1/2 coupling. Defaults to False.
            no_parity (bool, optional): Whether to ignore parity. Defaults to False.
            if_sort (bool, optional): Whether to sort irreps. Defaults to False.
        """
        if spinful:
            default_dtype_torch = convert_float_to_complex_dtype(default_dtype_torch)
        self.dtype = default_dtype_torch
        self.spinful = spinful
        self.nao_max = nao_max
        
        self.out_js_list = out_js_list
        if net_irreps_out is not None:
            net_irreps_out = Irreps(net_irreps_out)
        required_irreps_out = Irreps(None)
        in_slices = [0]
        wigner_multipliers = []  # wigner multipliers
        H_slices = [0]
        wigner_multipliers_H = []
        if spinful:
            in_slices_sp = []
            H_slices_sp = []
            wigner_multipliers_sp = []
            wigner_multipliers_sp_H = []
            
        for angular_momentum_left, angular_momentum_right in out_js_list:
            
            # Construct required_irreps_out
            multiplicity = 1
            _, required_irreps_out_single, required_irreps_x1 = irreps_from_l1l2(
                angular_momentum_left, angular_momentum_right, 
                multiplicity, spinful, no_parity=no_parity
            )
            required_irreps_out += required_irreps_out_single
            
            # Handle spinful case
            # Example: (1x0.5)x(2x0.5) = (1+2+3)x(0+1) = (1+2+3)+(0+1+2)+(1+2+3)+(2+3+4)
            if spinful:
                in_slice_sp = [0, required_irreps_out_single.dim]
                H_slice_sp = [0]
                wm_sp = [None]
                wm_sp_H = []
                for (_a, ir), ir_times_1 in zip(required_irreps_out_single, required_irreps_x1):
                    required_irreps_out += ir_times_1
                    in_slice_sp.append(in_slice_sp[-1] + ir_times_1.dim)
                    H_slice_sp.append(H_slice_sp[-1] + ir.dim)
                    wm_irx1 = []
                    wm_irx1_H = []
                    for _b, ir_1 in ir_times_1:
                        for _c in range(multiplicity):
                            wm_irx1.append(wigner_3j(ir.l, 1, ir_1.l, dtype=default_dtype_torch))
                            wm_irx1_H.append(wigner_3j(ir_1.l, ir.l, 1, dtype=default_dtype_torch) * (2 * ir_1.l + 1))
                    wm_irx1 = torch.cat(wm_irx1, dim=-1)
                    wm_sp.append(wm_irx1)
                    wm_irx1_H = torch.cat(wm_irx1_H, dim=0)
                    wm_sp_H.append(wm_irx1_H)
            
            # Construct slices
            in_slices.append(required_irreps_out.dim)
            H_slices.append(H_slices[-1] + (2 * angular_momentum_left + 1) * (2 * angular_momentum_right + 1))
            if spinful:
                in_slices_sp.append(in_slice_sp)
                H_slices_sp.append(H_slice_sp)
            
            # Get CG coefficients multiplier to act on net_out
            wm = []
            wm_H = []
            for _a, ir in required_irreps_out_single:
                for _b in range(multiplicity):
                    # About this 2l+1: 
                    # We want the exact inverse of the w_3j symbol, i.e.,
                    # torch.einsum("ijk,jkl->il",w_3j(l,l1,l2),w_3j(l1,l2,l)) == torch.eye(...)
                    # But this is not the case since the CG coefficients are unitary and w_3j differ 
                    # from CG coefficients by a constant factor. From 
                    # https://en.wikipedia.org/wiki/3-j_symbol#Mathematical_relation_to_Clebsch%E2%80%93Gordan_coefficients
                    # we know that 2l+1 is exactly the factor we want.
                    wm.append(wigner_3j(angular_momentum_left, angular_momentum_right, ir.l, dtype=default_dtype_torch))
                    wm_H.append(wigner_3j(ir.l, angular_momentum_left, angular_momentum_right, dtype=default_dtype_torch) * (2 * ir.l + 1))
            wm = torch.cat(wm, dim=-1)
            wm_H = torch.cat(wm_H, dim=0)
            wigner_multipliers.append(wm)
            wigner_multipliers_H.append(wm_H)
            if spinful:
                wigner_multipliers_sp.append(wm_sp)
                wigner_multipliers_sp_H.append(wm_sp_H)
            
        # Check net irreps out
        if spinful:
            required_irreps_out = required_irreps_out + required_irreps_out
        if net_irreps_out is not None:
            if if_sort:
                assert net_irreps_out == required_irreps_out.sort().irreps.simplify(), \
                    f'requires {required_irreps_out.sort().irreps.simplify()} but got {net_irreps_out}'
            else:
                assert net_irreps_out == required_irreps_out, \
                    f'requires {required_irreps_out} but got {net_irreps_out}'
        
        self.in_slices = in_slices
        self.wms = wigner_multipliers
        self.H_slices = H_slices
        self.wms_H = wigner_multipliers_H
        if spinful:
            self.in_slices_sp = in_slices_sp
            self.H_slices_sp = H_slices_sp
            self.wms_sp = wigner_multipliers_sp
            self.wms_sp_H = wigner_multipliers_sp_H
            
        # Register rotate kernel
        self.rotate_kernel = Rotate(default_dtype_torch, spinful=spinful)
        
        if spinful:
            sqrt2 = 1.4142135623730951
            # Transformation matrix for spin components
            self.oyzx2spin = torch.tensor([[  1,   0,   1,   0],
                                           [  0, -1j,   0,   1],
                                           [  0,  1j,   0,   1],
                                           [  1,   0,  -1,   0]],
                                          dtype=default_dtype_torch) / sqrt2
        
        self.sort = None
        if if_sort:
            self.sort = SortIrrepsTransform(required_irreps_out)
        
        if self.sort is not None:
            self.required_irreps_out = self.sort.irreps_out
        else:
            self.required_irreps_out = required_irreps_out
    
    def get_H(self, net_out):
        """Get OpenMX-type Hamiltonian from network output.
        
        Args:
            net_out (torch.Tensor): Network output tensor.
            
        Returns:
            torch.Tensor: Hamiltonian matrix in OpenMX format.
        """
        if self.sort is not None:
            net_out = self.sort.inverse(net_out)
        if self.spinful:
            half_len = int(net_out.shape[-1] / 2)
            re = net_out[:, :half_len]
            im = net_out[:, half_len:]
            net_out = re + 1j * im
        
        if self.spinful:
            block = torch.zeros(net_out.shape[0], 4, self.nao_max, self.nao_max).type_as(net_out)
        else:
            block = torch.zeros(net_out.shape[0], self.nao_max, self.nao_max).type_as(net_out)
        num_irreps_row = int(math.sqrt(len(self.out_js_list)))
        
        start_i, start_j = 0, 0
        for i, (l_left, l_right) in enumerate(self.out_js_list):
            in_slice = slice(self.in_slices[i], self.in_slices[i + 1])
            net_out_block = net_out[:, in_slice]
            n_i, n_j = int(2*l_left+1), int(2*l_right+1)
            blockpart = block.narrow(-2, start_i, n_i).narrow(-1, start_j, n_j)  # shape: (Nbatch, (4,) n_i, n_j)
            
            if self.spinful:
                # (1+2+3)+(0+1+2)+(1+2+3)+(2+3+4) -> (1+2+3)x(0+1)
                H_block = []
                for j in range(len(self.wms_sp[i])):
                    in_slice_sp = slice(self.in_slices_sp[i][j], self.in_slices_sp[i][j + 1])
                    if j == 0:
                        H_block.append(net_out_block[:, in_slice_sp].unsqueeze(-1))
                    else:
                        H_block.append(torch.einsum('jkl,il->ijk', self.wms_sp[i][j].type_as(net_out), net_out_block[:, in_slice_sp]))
                H_block = torch.cat([H_block[0], torch.cat(H_block[1:], dim=-2)], dim=-1)
                # (1+2+3)x(0+1) -> (uu,ud,du,dd)x(1x2)
                H_block = torch.einsum('imn,klm,jn->ijkl', H_block, self.wms[i].type_as(net_out), self.oyzx2spin.type_as(net_out))
                blockpart += H_block.reshape(net_out.shape[0], 4, n_i, n_j)
            else:
                H_block = torch.sum(self.wms[i][None, :, :, :].type_as(net_out) * net_out_block[:, None, None, :], dim=-1)
                blockpart += H_block.reshape(net_out.shape[0], n_i, n_j)
                
            if (i+1) % num_irreps_row == 0:
                start_i += n_i
                start_j = 0
            else:
                start_j += n_j
        return block  # output shape: [edge, (4 spin components,) H_flattened_concatenated]
    
    def get_net_out(self, H):
        """Get network output from OpenMX-type Hamiltonian.
        
        Args:
            H (torch.Tensor): Hamiltonian matrix in OpenMX format.
            
        Returns:
            torch.Tensor: Network output tensor.
        """
        out = []
        for i in range(len(self.out_js_list)):
            H_slice = slice(self.H_slices[i], self.H_slices[i + 1])
            l1, l2 = self.out_js_list[i]
            if self.spinful:
                H_block = H[..., H_slice].reshape(-1, 4, 2 * l1 + 1, 2 * l2 + 1)
                H_block = self.rotate_kernel.openmx2wiki_H(H_block, *self.out_js_list[i])
                # (uu,ud,du,dd)x(1x2) -> (1+2+3)x(0+1)
                H_block = torch.einsum('ilmn,jmn,kl->ijk', H_block, self.wms_H[i], self.oyzx2spin.T.conj())
                # (1+2+3)x(0+1) -> (1+2+3)+(0+1+2)+(1+2+3)+(2+3+4)
                net_out_block = [H_block[:, :, 0]]
                for j in range(len(self.wms_sp_H[i])):
                    H_slice_sp = slice(self.H_slices_sp[i][j], self.H_slices_sp[i][j + 1])
                    net_out_block.append(torch.einsum('jlm,ilm->ij', self.wms_sp_H[i][j], H_block[:, H_slice_sp, 1:]))
                net_out_block = torch.cat(net_out_block, dim=-1)
                out.append(net_out_block)
            else:
                H_block = H[:, H_slice].reshape(-1, 2 * l1 + 1, 2 * l2 + 1)
                H_block = self.rotate_kernel.openmx2wiki_H(H_block, *self.out_js_list[i])
                net_out_block = torch.sum(self.wms_H[i][None, :, :, :] * H_block[:, None, :, :], dim=(-1, -2))
                out.append(net_out_block)
        out = torch.cat(out, dim=-1)
        if self.spinful:
            out = torch.cat([out.real, out.imag], dim=-1)
        if self.sort is not None:
            out = self.sort(out)
        return out
    
    def convert_mask(self, mask):
        """Convert a mask tensor to the appropriate format.
        
        Args:
            mask (torch.Tensor): Input mask tensor.
            
        Returns:
            torch.Tensor: Converted mask tensor.
            
        Raises:
            AssertionError: If not in spinful mode.
        """
        assert self.spinful
        num_edges = mask.shape[0]
        mask = mask.permute(0, 2, 1).reshape(num_edges, -1).repeat(1, 2)
        if self.sort is not None:
            mask = self.sort(mask)
        return mask

class E3LayerNorm(nn.Module):
    """Layer normalization for E(3) equivariant neural networks.
    
    This module performs layer normalization on tensor fields while respecting
    their transformation properties under rotations.
    
    Attributes:
        irreps_in (e3nn.o3.Irreps): Input irreducible representations.
        eps (float): Small constant to prevent division by zero.
        weight (nn.Parameter, optional): Learnable scale parameter.
        bias (nn.Parameter, optional): Learnable bias parameter.
        bias_slices (list): Slices for applying bias.
        weight_slices (list): Slices for applying weight.
        subtract_mean (bool): Whether to subtract mean from scalar fields.
        divide_norm (bool): Whether to divide by norm.
        normalization (str): Normalization method ('component' or 'norm').
    """
    def __init__(self, irreps_in, eps=1e-5, affine=True, normalization='component', 
                 subtract_mean=True, divide_norm=False):
        """Initialize the E3 layer normalization.
        
        Args:
            irreps_in (e3nn.o3.Irreps or str): Input irreducible representations.
            eps (float, optional): Small constant to prevent division by zero. Defaults to 1e-5.
            affine (bool, optional): Whether to learn affine parameters. Defaults to True.
            normalization (str, optional): Normalization method ('component' or 'norm'). 
                Defaults to 'component'.
            subtract_mean (bool, optional): Whether to subtract mean. Defaults to True.
            divide_norm (bool, optional): Whether to divide by norm. Defaults to False.
        """
        super().__init__()
        
        self.irreps_in = Irreps(irreps_in)
        self.eps = eps
        
        if affine:          
            bias_index, weight_index = 0, 0
            weight_slices, bias_slices = [], []
            for mul, ir in irreps_in:
                if ir.is_scalar():  # bias only to 0e (scalars)
                    bias_slices.append(slice(bias_index, bias_index + mul))
                    bias_index += mul
                else:
                    bias_slices.append(None)
                weight_slices.append(slice(weight_index, weight_index + mul))
                weight_index += mul
            self.weight = nn.Parameter(torch.ones([weight_index]))
            self.bias = nn.Parameter(torch.zeros([bias_index]))
            self.bias_slices = bias_slices
            self.weight_slices = weight_slices
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        
        self.subtract_mean = subtract_mean
        self.divide_norm = divide_norm
        assert normalization in ['component', 'norm'], "normalization must be 'component' or 'norm'"
        self.normalization = normalization
            
        self.reset_parameters()
    
    def reset_parameters(self):
        """Reset learnable parameters to default values."""
        if self.weight is not None:
            self.weight.data.fill_(1)
        if self.bias is not None:
            self.bias.data.fill_(0)
            
    def forward(self, x: torch.Tensor, batch: torch.Tensor = None):
        """Apply layer normalization to input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape [num_node(edge), dim].
            batch (torch.Tensor, optional): Batch assignment for nodes/edges.
                If None, assumes all nodes/edges belong to a single graph.
                
        Returns:
            torch.Tensor: Normalized tensor of the same shape as input.
            
        Note:
            - If the first dimension of x is node index, then batch should be batch.batch
            - If the first dimension of x is edge index, then batch should be batch.batch[batch.edge_index[0]]
        """
        if batch is None:
            batch = torch.full([x.shape[0]], 0, dtype=torch.int64)
        # From torch_geometric.nn.norm.LayerNorm
        batch_size = int(batch.max()) + 1 
        batch_degree = degree(batch, batch_size, dtype=torch.int64).clamp_(min=1).to(dtype=x.dtype)
        
        out = []
        index = 0
        for field_idx, (mul, ir) in enumerate(self.irreps_in):        
            field = x[:, index: index + mul * ir.dim].reshape(-1, mul, ir.dim)  # [node, mul, repr]
            
            # Compute and subtract mean
            if self.subtract_mean or ir.l == 0:  # Do not subtract mean for l>0 irreps if subtract_mean=False
                mean = scatter(field, batch, dim=0, dim_size=batch_size,
                            reduce='add').mean(dim=1, keepdim=True) / batch_degree[:, None, None]  
                # scatter_mean does not support complex numbers
                field = field - mean[batch]
                
            # Compute and divide norm
            if self.divide_norm or ir.l == 0:  # Do not divide norm for l>0 irreps if divide_norm=False
                norm = scatter(field.abs().pow(2), batch, dim=0, dim_size=batch_size,
                            reduce='mean').mean(dim=[1,2], keepdim=True)  # add abs for complex numbers
                if self.normalization == 'norm':
                    norm = norm * ir.dim
                field = field / (norm.sqrt()[batch] + self.eps)
            
            # Apply affine transformation
            if self.weight is not None:
                weight = self.weight[self.weight_slices[field_idx]]
                field = field * weight[None, :, None]
            if self.bias is not None and ir.is_scalar():
                bias = self.bias[self.bias_slices[field_idx]]
                field = field + bias[None, :, None]
            
            out.append(field.reshape(-1, mul * ir.dim))
            index += mul * ir.dim
            
        out = torch.cat(out, dim=-1)
                
        return out
