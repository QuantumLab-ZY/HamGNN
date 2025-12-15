
import copy
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import opt_einsum as oe
import torch
from e3nn import o3
from e3nn.util.jit import compile_mode
from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Structure
from pymatgen.symmetry.kpath import KPathSeek
from torch import nn
from torch_geometric.nn import global_mean_pool
from torch_scatter import scatter

from ..nn.interaction_blocks import ResidualBlock
from ..nn.tensor_decomposition import E3TensorDecomposition
from ..physics.Clebsch_Gordan_coefficients import ClebschGordanCoefficients
from ..physics.kpoints import kpoints_generator
from ..utils.constants import au2ang
from ..utils.math_utils import (
    blockwise_2x2_concat,
    extract_elements_above_threshold,
    upgrade_tensor_precision
)

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

class HamGNNPlusPlusOut(nn.Module):
    """
    Neural network module for computing Hamiltonian matrices using Graph Neural Networks.
    
    This class implements a GNN architecture for quantum chemistry applications, supporting
    different Hamiltonian types (openmx, siesta, abacus) with various configuration options
    for physics-informed neural network training.
    
    Args:
        irreps_in_node (Union[int, str, o3.Irreps]): Input irreps for node features.
        irreps_in_edge (Union[int, str, o3.Irreps]): Input irreps for edge features.
        nao_max (int): Maximum number of atomic orbitals.
        return_forces (bool): Whether to compute forces during forward pass.
        create_graph (bool): Whether to create computational graph for backpropagation.
        ham_type (str): Type of Hamiltonian ('openmx', 'siesta', 'abacus', or 'pasp').
        ham_only (bool): Whether to only compute the Hamiltonian matrix.
        symmetrize (bool): Whether to symmetrize the Hamiltonian matrix.
        include_triplet (bool): Whether to include triplet interactions.
        calculate_band_energy (bool): Whether to calculate band energies.
        num_k (int): Number of k-points for band structure calculations.
        k_path (Union[list, np.ndarray, tuple]): Path in k-space for band structure.
        band_num_control (dict or int): Control for band numbers.
        soc_switch (bool): Whether to include spin-orbit coupling.
        nonlinearity_type (str): Type of nonlinearity for neural network layers.
        export_reciprocal_values (bool): Whether to export reciprocal space values.
        add_H0 (bool): Whether to add the initial Hamiltonian term H0.
        soc_basis (str): Basis for spin-orbit coupling ('so3' or 'su2').
        spin_constrained (bool): Whether to apply spin constraints.
        use_learned_weight (bool): Whether to use learned weights.
        minMagneticMoment (float): Minimum magnetic moment threshold.
        collinear_spin (bool): Whether to consider collinear spin.
        zero_point_shift (bool): Whether to apply zero-point energy shift.
        add_H_nonsoc (bool): Whether to add non-SOC Hamiltonian.
        get_nonzero_mask_tensor (bool): Whether to get nonzero mask tensor.
    """
    
    def __init__(self, 
                 irreps_in_node: Union[int, str, o3.Irreps] = None, 
                 irreps_in_edge: Union[int, str, o3.Irreps] = None, 
                 nao_max: int = 14, 
                 return_forces: bool = False, 
                 create_graph: bool = False, 
                 ham_type: str = 'openmx', 
                 ham_only: bool = False, 
                 symmetrize: bool = True, 
                 include_triplet: bool = False, 
                 calculate_band_energy: bool = False, 
                 num_k: int = 8, 
                 k_path: Union[list, np.ndarray, tuple] = None, 
                 band_num_control: dict = None, 
                 soc_switch: bool = True, 
                 nonlinearity_type: str = 'gate', 
                 export_reciprocal_values: bool = False, 
                 add_H0: bool = False, 
                 soc_basis: str = 'so3',
                 spin_constrained: bool = False, 
                 use_learned_weight: bool = True, 
                 minMagneticMoment: float = 0.5, 
                 collinear_spin: bool = False,
                 zero_point_shift: bool = False,
                 add_H_nonsoc: bool = False,
                 get_nonzero_mask_tensor: bool = False,
                 calculate_sparsity: bool = True
                 ):
        
        super().__init__()
        # Store force computation settings
        self.derivative = return_forces
        self.create_graph = create_graph
        
        # Store model parameters
        self.nao_max = nao_max
        self.ham_type = ham_type.lower()
        self.ham_only = ham_only
        self.symmetrize = symmetrize
        self.include_triplet = include_triplet
        self.soc_switch = soc_switch
        self.nonlinearity_type = nonlinearity_type
        
        # Output and export settings
        self.export_reciprocal_values = export_reciprocal_values
        self.add_H0 = add_H0
        
        # Spin-related parameters
        self.spin_constrained = spin_constrained
        self.use_learned_weight = use_learned_weight
        self.min_magnetic_moment = minMagneticMoment
        self.collinear_spin = collinear_spin
        self.soc_basis = soc_basis.lower()

        # Adjust SOC basis for non-openmx Hamiltonians
        if soc_switch:
            if self.ham_type != 'openmx':
                self.soc_basis = 'su2'
        
        # Band structure calculations
        self.calculate_band_energy = calculate_band_energy
        self.num_k = num_k
        self.k_path = k_path
        
        # Additional physics parameters
        self.add_quartic = False
        self.zero_point_shift = zero_point_shift
        self.add_H_nonsoc = add_H_nonsoc
        self.get_nonzero_mask_tensor = get_nonzero_mask_tensor
        
        # Sparsity calculation flag
        self.calculate_sparsity = calculate_sparsity
        
        # Initialize configurations
        self._configure_band_num_control(band_num_control)
        self._initialize_basis_information()
        self._initialize_irreducible_representations()
        
        # Clebsch-Gordan coefficients
        self.cg_calculator = ClebschGordanCoefficients(max_l=self.hamiltonian_irreps.lmax)
        
        # Hamiltonian networks
        self.onsite_hamiltonian_network = self._create_hamiltonian_layer(
            irreps_in=irreps_in_node, 
            irreps_out=self.hamiltonian_irreps
        )
        self.offsite_hamiltonian_network = self._create_hamiltonian_layer(
            irreps_in=irreps_in_edge, 
            irreps_out=self.hamiltonian_irreps
        )
        
        # Spin-orbit coupling networks
        if soc_switch:            
            # Initialize based on selected basis
            if self.soc_basis == 'su2':
                self.onsite_hamiltonian_network = self._create_hamiltonian_layer(
                    irreps_in=irreps_in_node, 
                    irreps_out=2*self.hamiltonian_irreps_su2
                )
                self.offsite_hamiltonian_network = self._create_hamiltonian_layer(
                    irreps_in=irreps_in_edge, 
                    irreps_out=2*self.hamiltonian_irreps_su2
                )
            
            elif self.soc_basis == 'so3':                
                self.onsite_ksi_network = self._create_hamiltonian_layer(
                    irreps_in=irreps_in_node, 
                    irreps_out=(self.nao_max**2*o3.Irreps("0e")).simplify()
                )
                self.offsite_ksi_network = self._create_hamiltonian_layer(
                    irreps_in=irreps_in_edge, 
                    irreps_out=(self.nao_max**2*o3.Irreps("0e")).simplify()
                )
            
            else:
                raise NotImplementedError(f"SOC basis '{soc_basis}' not supported!")
        
        # Spin-constrained networks
        if self.spin_constrained:
            # J network
            self.onsite_J_network = self._create_hamiltonian_layer(
                irreps_in=irreps_in_node, 
                irreps_out=self.J_irreps
            )
            self.offsite_J_network = self._create_hamiltonian_layer(
                irreps_in=irreps_in_edge, 
                irreps_out=self.J_irreps
            )
            
            # K network (if quartic term is enabled)
            if self.add_quartic:
                self.onsite_K_network = self._create_hamiltonian_layer(
                    irreps_in=irreps_in_node, 
                    irreps_out=self.K_irreps
                )
                self.offsite_K_network = self._create_hamiltonian_layer(
                    irreps_in=irreps_in_edge, 
                    irreps_out=self.K_irreps
                )
            
            # Weight networks
            if self.use_learned_weight:
                self.onsite_weight_network = self._create_hamiltonian_layer(
                    irreps_in=irreps_in_node, 
                    irreps_out=self.hamiltonian_irreps
                )
                self.offsite_weight_network = self._create_hamiltonian_layer(
                    irreps_in=irreps_in_edge, 
                    irreps_out=self.hamiltonian_irreps
                )
        
        # Overlap networks (if not Hamiltonian-only mode)
        if not self.ham_only:            
            self.onsite_overlap_network = self._create_hamiltonian_layer(
                irreps_in=irreps_in_node, 
                irreps_out=self.hamiltonian_irreps
            )
            self.offsite_overlap_network = self._create_hamiltonian_layer(
                irreps_in=irreps_in_edge, 
                irreps_out=self.hamiltonian_irreps
            )

    def _initialize_irreducible_representations(self):
        """
        Initialize the irreducible representations for the Hamiltonian and related tensors.
        
        Sets up Hamiltonian irreps, their dimensions, SU(2) basis irreps (if spin-orbit 
        coupling is enabled), and J/K irreps (if spin-constrained mode is enabled).
        """
        self.hamiltonian_irreps_dimensions = []
        
        # Initialize main Hamiltonian irreps
        self.hamiltonian_irreps = o3.Irreps()
        for _, li in self.row:
            for _, lj in self.col:
                for L in range(abs(li.l-lj.l), li.l+lj.l+1):
                    self.hamiltonian_irreps += o3.Irrep(L, (-1)**(li.l+lj.l))
        
        # Store dimensions for each irrep
        for irrep in self.hamiltonian_irreps:
            self.hamiltonian_irreps_dimensions.append(irrep.dim)
        
        self.hamiltonian_irreps_dimensions = torch.LongTensor(self.hamiltonian_irreps_dimensions)
        
        # SU(2) basis irreps (for spin-orbit coupling)
        if self.soc_switch and (self.soc_basis == 'su2'): 
            out_js_list = []
            for _, li in self.row:
                for _, lj in self.col:
                    out_js_list.append((li.l, lj.l))
            self.hamiltonian_decomposition = E3TensorDecomposition(
                None, 
                out_js_list, 
                default_dtype_torch=torch.get_default_dtype(),
                nao_max=self.nao_max, 
                spinful=True
            )
            self.hamiltonian_irreps_su2 = self.hamiltonian_decomposition.required_irreps_out
        
        # Spin-constrained mode irreps
        if self.spin_constrained:
            # Initialize J and K irreps
            self.J_irreps = o3.Irreps()
            self.K_irreps = o3.Irreps()
            
            self.J_irreps_dimensions = []
            self.K_irreps_dimensions = []
            self.num_blocks = 0            
            
            for _, li in self.row:
                for _, lj in self.col:
                    self.num_blocks += 1
                    if self.soc_switch:
                        for L in range(0, 3):
                            self.J_irreps += o3.Irrep(L, 1)   # t=1, p=1
                            self.K_irreps += o3.Irrep(0, 1)   # t=1, p=1
                    else:
                        self.J_irreps += o3.Irrep(0, 1)   # t=1, p=1
            
            # Store dimensions for J and K irreps
            for irrep in self.J_irreps:
                self.J_irreps_dimensions.append(irrep.dim)
            for irrep in self.K_irreps:
                self.K_irreps_dimensions.append(irrep.dim)   
            
            self.J_irreps_dimensions = torch.LongTensor(self.J_irreps_dimensions)
            self.K_irreps_dimensions = torch.LongTensor(self.K_irreps_dimensions)

    def _initialize_basis_information(self):
        """
        Initialize basis information based on the selected Hamiltonian type.
        
        Configures basis sets for different Hamiltonian types:
        - openmx: Basis sets for OpenMX DFT code
        - siesta: Basis sets for SIESTA DFT code
        - abacus: Basis sets for ABACUS DFT code
        - pasp: Simplified basis set
        """
        if self.ham_type == 'openmx':
            self._initialize_openmx_basis()
        elif self.ham_type == 'siesta':
            self._initialize_siesta_basis()
        elif self.ham_type == 'abacus':
            self._initialize_abacus_basis()
        elif self.ham_type == 'pasp':
            self.row = self.col = o3.Irreps("1x1o")
        else:
            raise NotImplementedError(f"Hamiltonian type '{self.ham_type}' is not supported.")

    def _initialize_openmx_basis(self):
        """
        Sets basis information for 'openmx' Hamiltonian.
        """
        self.num_valence = {Element['H'].Z: 1, Element['He'].Z: 2, Element['Li'].Z: 3, Element['Be'].Z: 2, Element['B'].Z: 3,
                            Element['C'].Z: 4, Element['N'].Z: 5,  Element['O'].Z: 6,  Element['F'].Z: 7,  Element['Ne'].Z: 8,
                            Element['Na'].Z: 9, Element['Mg'].Z: 8, Element['Al'].Z: 3, Element['Si'].Z: 4, Element['P'].Z: 5,
                            Element['S'].Z: 6,  Element['Cl'].Z: 7, Element['Ar'].Z: 8, Element['K'].Z: 9,  Element['Ca'].Z: 10,
                            Element['Sc'].Z: 11, Element['Ti'].Z: 12, Element['V'].Z: 13, Element['Cr'].Z: 14, Element['Mn'].Z: 15,
                            Element['Fe'].Z: 16, Element['Co'].Z: 17, Element['Ni'].Z: 18, Element['Cu'].Z: 19, Element['Zn'].Z: 20,
                            Element['Ga'].Z: 13, Element['Ge'].Z: 4,  Element['As'].Z: 15, Element['Se'].Z: 6,  Element['Br'].Z: 7,
                            Element['Kr'].Z: 8,  Element['Rb'].Z: 9,  Element['Sr'].Z: 10, Element['Y'].Z: 11, Element['Zr'].Z: 12,
                            Element['Nb'].Z: 13, Element['Mo'].Z: 14, Element['Tc'].Z: 15, Element['Ru'].Z: 14, Element['Rh'].Z: 15,
                            Element['Pd'].Z: 16, Element['Ag'].Z: 17, Element['Cd'].Z: 12, Element['In'].Z: 13, Element['Sn'].Z: 14,
                            Element['Sb'].Z: 15, Element['Te'].Z: 16, Element['I'].Z: 7, Element['Xe'].Z: 8, Element['Cs'].Z: 9,
                            Element['Ba'].Z: 10, Element['La'].Z: 11, Element['Ce'].Z: 12, Element['Pr'].Z: 13, Element['Nd'].Z: 14,
                            Element['Pm'].Z: 15, Element['Sm'].Z: 16, Element['Dy'].Z: 20, Element['Ho'].Z: 21, Element['Lu'].Z: 11,
                            Element['Hf'].Z: 12, Element['Ta'].Z: 13, Element['W'].Z: 12,  Element['Re'].Z: 15, Element['Os'].Z: 14,
                            Element['Ir'].Z: 15, Element['Pt'].Z: 16, Element['Au'].Z: 17, Element['Hg'].Z: 18, Element['Tl'].Z: 19,
                            Element['Pb'].Z: 14, Element['Bi'].Z: 15
                        }
        
        if self.nao_max == 14:
            self.index_change = torch.LongTensor([0,1,2,5,3,4,8,6,7,11,13,9,12,10])       
            self.row = self.col = o3.Irreps("1x0e+1x0e+1x0e+1x1o+1x1o+1x2e")
            self.basis_def = {  1:[0,1,3,4,5], # H
                                2:[0,1,3,4,5], # He
                                3:[0,1,2,3,4,5,6,7,8], # Li
                                4:[0,1,3,4,5,6,7,8], # Be
                                5:[0,1,3,4,5,6,7,8,9,10,11,12,13], # B
                                6:[0,1,3,4,5,6,7,8,9,10,11,12,13], # C
                                7:[0,1,3,4,5,6,7,8,9,10,11,12,13], # N
                                8:[0,1,3,4,5,6,7,8,9,10,11,12,13], # O
                                9:[0,1,3,4,5,6,7,8,9,10,11,12,13], # F
                                10:[0,1,3,4,5,6,7,8,9,10,11,12,13], # Ne
                                11:[0,1,2,3,4,5,6,7,8,9,10,11,12,13], # Na
                                12:[0,1,2,3,4,5,6,7,8,9,10,11,12,13], # Mg
                                13:[0,1,3,4,5,6,7,8,9,10,11,12,13], # Al
                                14:[0,1,3,4,5,6,7,8,9,10,11,12,13], # Si
                                15:[0,1,3,4,5,6,7,8,9,10,11,12,13], # p
                                16:[0,1,3,4,5,6,7,8,9,10,11,12,13], # S
                                17:[0,1,3,4,5,6,7,8,9,10,11,12,13], # Cl
                                18:[0,1,3,4,5,6,7,8,9,10,11,12,13], # Ar
                                19:[0,1,2,3,4,5,6,7,8,9,10,11,12,13], # K
                                20:[0,1,2,3,4,5,6,7,8,9,10,11,12,13], # Ca
                                35:[0,1,2,3,4,5,6,7,8,9,10,11,12,13], # Br  
                                Element['V'].Z: [0,1,2,3,4,5,6,7,8,9,10,11,12,13], # V
                                Element['Mn'].Z: [0,1,2,3,4,5,6,7,8,9,10,11,12,13], # Mn
                            }
        
        elif self.nao_max == 13:
            self.basis_def = {  1:[0,1,2,3,4], # H
                                5:[0,1,2,3,4,5,6,7,8,9,10,11,12], # B
                                6:[0,1,2,3,4,5,6,7,8,9,10,11,12], # C
                                7:[0,1,2,3,4,5,6,7,8,9,10,11,12], # N
                                8:[0,1,2,3,4,5,6,7,8,9,10,11,12] # O
                            }
            self.index_change = torch.LongTensor([0,1,4,2,3,7,5,6,10,12,8,11,9])       
            self.row = self.col = o3.Irreps("1x0e+1x0e+1x1o+1x1o+1x2e")
        
        elif self.nao_max == 19:
            self.index_change = torch.LongTensor([0,1,2,5,3,4,8,6,7,11,13,9,12,10,16,18,14,17,15])       
            self.row = self.col = o3.Irreps("1x0e+1x0e+1x0e+1x1o+1x1o+1x2e+1x2e")
            self.basis_def = {  1:[0,1,3,4,5], # H
                2:[0,1,3,4,5], # He
                3:[0,1,2,3,4,5,6,7,8], # Li
                4:[0,1,3,4,5,6,7,8], # Be
                5:[0,1,3,4,5,6,7,8,9,10,11,12,13], # B
                6:[0,1,3,4,5,6,7,8,9,10,11,12,13], # C
                7:[0,1,3,4,5,6,7,8,9,10,11,12,13], # N
                8:[0,1,3,4,5,6,7,8,9,10,11,12,13], # O
                9:[0,1,3,4,5,6,7,8,9,10,11,12,13], # F
                10:[0,1,3,4,5,6,7,8,9,10,11,12,13], # Ne
                11:[0,1,2,3,4,5,6,7,8,9,10,11,12,13], # Na
                12:[0,1,2,3,4,5,6,7,8,9,10,11,12,13], # Mg
                13:[0,1,3,4,5,6,7,8,9,10,11,12,13], # Al
                14:[0,1,3,4,5,6,7,8,9,10,11,12,13], # Si
                15:[0,1,3,4,5,6,7,8,9,10,11,12,13], # p
                16:[0,1,3,4,5,6,7,8,9,10,11,12,13], # S
                17:[0,1,3,4,5,6,7,8,9,10,11,12,13], # Cl
                18:[0,1,3,4,5,6,7,8,9,10,11,12,13], # Ar
                19:[0,1,2,3,4,5,6,7,8,9,10,11,12,13], # K
                20:[0,1,2,3,4,5,6,7,8,9,10,11,12,13], # Ca
                25:[0,1,2,3,4,5,6,7,8,9,10,11,12,13], # Mn
                42:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], # Mo  
                83:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], # Bi  
                34:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], # Se 
                24:[0,1,2,3,4,5,6,7,8,9,10,11,12,13], # Cr 
                53:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], # I
                28:[0,1,2,3,4,5,6,7,8,9,10,11,12,13], # Ni
                35:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], # Br 
                26:[0,1,2,3,4,5,6,7,8,9,10,11,12,13], # Fe
                77:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], # Ir
                52:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], # Te
                Element['V'].Z: [0,1,2,3,4,5,6,7,8,9,10,11,12,13], # V
                Element['Sb'].Z: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], # Sb
            }
        
        elif self.nao_max == 26:
            self.index_change = torch.LongTensor([0,1,2,5,3,4,8,6,7,11,13,9,12,10,16,18,14,17,15,22,23,21,24,20,25,19])       
            self.row = self.col = o3.Irreps("1x0e+1x0e+1x0e+1x1o+1x1o+1x2e+1x2e+1x3o")
            self.basis_def = (lambda s1=[0],s2=[1],s3=[2],p1=[3,4,5],p2=[6,7,8],d1=[9,10,11,12,13],d2=[14,15,16,17,18],f1=[19,20,21,22,23,24,25]: {
                Element['H'].Z : s1+s2+p1,  # H6.0-s2p1
                Element['He'].Z : s1+s2+p1,  # He8.0-s2p1
                Element['Li'].Z : s1+s2+s3+p1+p2,  # Li8.0-s3p2
                Element['Be'].Z : s1+s2+p1+p2,  # Be7.0-s2p2
                Element['B'].Z : s1+s2+p1+p2+d1,  # B7.0-s2p2d1
                Element['C'].Z : s1+s2+p1+p2+d1,  # C6.0-s2p2d1
                Element['N'].Z : s1+s2+p1+p2+d1,  # N6.0-s2p2d1
                Element['O'].Z : s1+s2+p1+p2+d1,  # O6.0-s2p2d1
                Element['F'].Z : s1+s2+p1+p2+d1,  # F6.0-s2p2d1
                Element['Ne'].Z: s1+s2+p1+p2+d1,  # Ne9.0-s2p2d1
                Element['Na'].Z: s1+s2+s3+p1+p2+d1,  # Na9.0-s3p2d1
                Element['Mg'].Z: s1+s2+s3+p1+p2+d1,  # Mg9.0-s3p2d1
                Element['Al'].Z: s1+s2+p1+p2+d1,  # Al7.0-s2p2d1
                Element['Si'].Z: s1+s2+p1+p2+d1,  # Si7.0-s2p2d1
                Element['P'].Z: s1+s2+p1+p2+d1,  # P7.0-s2p2d1
                Element['S'].Z: s1+s2+p1+p2+d1,  # S7.0-s2p2d1
                Element['Cl'].Z: s1+s2+p1+p2+d1,  # Cl7.0-s2p2d1
                Element['Ar'].Z: s1+s2+p1+p2+d1,  # Ar9.0-s2p2d1
                Element['K'].Z: s1+s2+s3+p1+p2+d1,  # K10.0-s3p2d1
                Element['Ca'].Z: s1+s2+s3+p1+p2+d1,  # Ca9.0-s3p2d1
                Element['Sc'].Z: s1+s2+s3+p1+p2+d1,  # Sc9.0-s3p2d1
                Element['Ti'].Z: s1+s2+s3+p1+p2+d1,  # Ti7.0-s3p2d1
                Element['V'].Z: s1+s2+s3+p1+p2+d1,  # V6.0-s3p2d1
                Element['Cr'].Z: s1+s2+s3+p1+p2+d1,  # Cr6.0-s3p2d1
                Element['Mn'].Z: s1+s2+s3+p1+p2+d1,  # Mn6.0-s3p2d1
                Element['Fe'].Z: s1+s2+s3+p1+p2+d1,  # Fe5.5H-s3p2d1
                Element['Co'].Z: s1+s2+s3+p1+p2+d1,  # Co6.0H-s3p2d1
                Element['Ni'].Z: s1+s2+s3+p1+p2+d1,  # Ni6.0H-s3p2d1
                Element['Cu'].Z: s1+s2+s3+p1+p2+d1,  # Cu6.0H-s3p2d1
                Element['Zn'].Z: s1+s2+s3+p1+p2+d1,  # Zn6.0H-s3p2d1
                Element['Ga'].Z: s1+s2+s3+p1+p2+d1+d2,  # Ga7.0-s3p2d2
                Element['Ge'].Z: s1+s2+s3+p1+p2+d1+d2,  # Ge7.0-s3p2d2
                Element['As'].Z: s1+s2+s3+p1+p2+d1+d2,  # As7.0-s3p2d2
                Element['Se'].Z: s1+s2+s3+p1+p2+d1+d2,  # Se7.0-s3p2d2
                Element['Br'].Z: s1+s2+s3+p1+p2+d1+d2,  # Br7.0-s3p2d2
                Element['Kr'].Z: s1+s2+s3+p1+p2+d1+d2,  # Kr10.0-s3p2d2
                Element['Rb'].Z: s1+s2+s3+p1+p2+d1+d2,  # Rb11.0-s3p2d2
                Element['Sr'].Z: s1+s2+s3+p1+p2+d1+d2,  # Sr10.0-s3p2d2
                Element['Y'].Z: s1+s2+s3+p1+p2+d1+d2,  # Y10.0-s3p2d2
                Element['Zr'].Z: s1+s2+s3+p1+p2+d1+d2,  # Zr7.0-s3p2d2
                Element['Nb'].Z: s1+s2+s3+p1+p2+d1+d2,  # Nb7.0-s3p2d2
                Element['Mo'].Z: s1+s2+s3+p1+p2+d1+d2,  # Mo7.0-s3p2d2
                Element['Tc'].Z: s1+s2+s3+p1+p2+d1+d2,  # Tc7.0-s3p2d2
                Element['Ru'].Z: s1+s2+s3+p1+p2+d1+d2,  # Ru7.0-s3p2d2
                Element['Rh'].Z: s1+s2+s3+p1+p2+d1+d2,  # Rh7.0-s3p2d2
                Element['Pd'].Z: s1+s2+s3+p1+p2+d1+d2,  # Pd7.0-s3p2d2
                Element['Ag'].Z: s1+s2+s3+p1+p2+d1+d2,  # Ag7.0-s3p2d2
                Element['Cd'].Z: s1+s2+s3+p1+p2+d1+d2,  # Cd7.0-s3p2d2
                Element['In'].Z: s1+s2+s3+p1+p2+d1+d2,  # In7.0-s3p2d2
                Element['Sn'].Z: s1+s2+s3+p1+p2+d1+d2,  # Sn7.0-s3p2d2
                Element['Sb'].Z: s1+s2+s3+p1+p2+d1+d2,  # Sb7.0-s3p2d2
                Element['Te'].Z: s1+s2+s3+p1+p2+d1+d2+f1,  # Te7.0-s3p2d2f1
                Element['I'].Z: s1+s2+s3+p1+p2+d1+d2+f1,  # I7.0-s3p2d2f1
                Element['Xe'].Z: s1+s2+s3+p1+p2+d1+d2,  # Xe11.0-s3p2d2
                Element['Cs'].Z: s1+s2+s3+p1+p2+d1+d2,  # Cs12.0-s3p2d2
                Element['Ba'].Z: s1+s2+s3+p1+p2+d1+d2,  # Ba10.0-s3p2d2
                Element['La'].Z: s1+s2+s3+p1+p2+d1+d2+f1,  # La8.0-s3p2d2f1
                Element['Ce'].Z: s1+s2+s3+p1+p2+d1+d2+f1,  # Ce8.0-s3p2d2f1
                Element['Pr'].Z: s1+s2+s3+p1+p2+d1+d2+f1,  # Pr8.0-s3p2d2f1
                Element['Nd'].Z: s1+s2+s3+p1+p2+d1+d2+f1,  # Nd8.0-s3p2d2f1
                Element['Pm'].Z: s1+s2+s3+p1+p2+d1+d2+f1,  # Pm8.0-s3p2d2f1
                Element['Sm'].Z: s1+s2+s3+p1+p2+d1+d2+f1,  # Sm8.0-s3p2d2f1
                Element['Dy'].Z: s1+s2+s3+p1+p2+d1+d2+f1,  # Dy8.0-s3p2d2f1
                Element['Ho'].Z: s1+s2+s3+p1+p2+d1+d2+f1,  # Ho8.0-s3p2d2f1
                Element['Lu'].Z: s1+s2+s3+p1+p2+d1+d2+f1,  # Lu8.0-s3p2d2f1
                Element['Hf'].Z: s1+s2+s3+p1+p2+d1+d2+f1,  # Hf9.0-s3p2d2f1
                Element['Ta'].Z: s1+s2+s3+p1+p2+d1+d2+f1,  # Ta7.0-s3p2d2f1
                Element['W'].Z: s1+s2+s3+p1+p2+d1+d2+f1,  # W7.0-s3p2d2f1
                Element['Re'].Z: s1+s2+s3+p1+p2+d1+d2+f1,  # Re7.0-s3p2d2f1
                Element['Os'].Z: s1+s2+s3+p1+p2+d1+d2+f1,  # Os7.0-s3p2d2f1
                Element['Ir'].Z: s1+s2+s3+p1+p2+d1+d2+f1,  # Ir7.0-s3p2d2f1
                Element['Pt'].Z: s1+s2+s3+p1+p2+d1+d2+f1,  # Pt7.0-s3p2d2f1
                Element['Au'].Z: s1+s2+s3+p1+p2+d1+d2+f1,  # Au7.0-s3p2d2f1
                Element['Hg'].Z: s1+s2+s3+p1+p2+d1+d2+f1,  # Hg8.0-s3p2d2f1
                Element['Tl'].Z: s1+s2+s3+p1+p2+d1+d2+f1,  # Tl8.0-s3p2d2f1
                Element['Pb'].Z: s1+s2+s3+p1+p2+d1+d2+f1,  # Pb8.0-s3p2d2f1
                Element['Bi'].Z: s1+s2+s3+p1+p2+d1+d2+f1,  # Bi8.0-s3p2d2f1 
            })()
        else:
            raise NotImplementedError(f"NAO max '{self.nao_max}' not supported for 'openmx'.")

    def _initialize_siesta_basis(self):
        """
        Sets basis information for 'siesta' Hamiltonian.
        """
        self.num_valence = {
            1:1,2:2,
            3:1,4:2,5:3,6:4,7:5,8:6,9:7,10:8,
            11:1,12:2,13:3,14:4,15:5,16:6,17:7,18:8,
            19:1,20:2,22:12,31:3,33:5,72:4
        }
        if self.nao_max == 13:
            self.index_change = None       
            self.row = self.col = o3.Irreps("1x0e+1x0e+1x1o+1x1o+1x2e")
            self.minus_index = torch.LongTensor([2,4,5,7,9,11]) # this list should follow the order in siesta. See spher_harm.f
            self.basis_def = (lambda s1=[0],s2=[1],p1=[2,3,4],p2=[5,6,7],d1=[8,9,10,11,12]: {
                1 : s1+s2+p1, # H
                2 : s1+s2+p1, # He
                3 : s1+s2+p1, # Li
                4 : s1+s2+p1, # Be
                5 : s1+s2+p1+p2+d1, # B
                6 : s1+s2+p1+p2+d1, # C
                7 : s1+s2+p1+p2+d1, # N
                8 : s1+s2+p1+p2+d1, # O
                9 : s1+s2+p1+p2+d1, # F
                10: s1+s2+p1+p2+d1, # Ne
                11: s1+s2+p1, # Na
                12: s1+s2+p1, # Mg
                13: s1+s2+p1+p2+d1, # Al
                14: s1+s2+p1+p2+d1, # Si
                15: s1+s2+p1+p2+d1, # P
                16: s1+s2+p1+p2+d1, # S
                17: s1+s2+p1+p2+d1, # Cl
                18: s1+s2+p1+p2+d1, # Ar
                19: s1+s2+p1, # K
                20: s1+s2+p1, # Cl
                31: s1+s2+p1+p2+d1, # Ga
                33: s1+s2+p1+p2+d1, # As
            })()
        elif self.nao_max == 19:
            self.index_change = None
            self.row = self.col = o3.Irreps("1x0e+1x0e+1x0e+1x1o+1x1o+1x2e+1x2e")
            self.minus_index = torch.LongTensor([3,5,6,8,10,12,15,17]) # this list should follow the order in siesta. See spher_harm.f
            self.basis_def = (lambda s1=[0],s2=[1],s3=[2],p1=[3,4,5],p2=[6,7,8],d1=[9,10,11,12,13],d2=[14,15,16,17,18]: {
                1 : s1+s2+p1, # H
                2 : s1+s2+p1, # He
                3 : s1+s2+p1, # Li
                4 : s1+s2+p1, # Be
                5 : s1+s2+p1+p2+d1, # B
                6 : s1+s2+p1+p2+d1, # C
                7 : s1+s2+p1+p2+d1, # N
                8 : s1+s2+p1+p2+d1, # O
                9 : s1+s2+p1+p2+d1, # F
                10: s1+s2+p1+p2+d1, # Ne
                11: s1+s2+p1, # Na
                12: s1+s2+p1, # Mg
                13: s1+s2+p1+p2+d1, # Al
                14: s1+s2+p1+p2+d1, # Si
                15: s1+s2+p1+p2+d1, # P
                16: s1+s2+p1+p2+d1, # S
                17: s1+s2+p1+p2+d1, # Cl
                18: s1+s2+p1+p2+d1, # Ar
                19: s1+s2+p1, # K
                20: s1+s2+p1, # Cl
                22: s1+s2+s3+p1+p2+d1+d2, # Ti, created by Qin.
            })()
        else:
            raise NotImplementedError(f"NAO max '{self.nao_max}' not supported for 'siesta'.")

    def _initialize_abacus_basis(self):
        """
        Sets basis information for 'abacus' Hamiltonian.
        """
        self.num_valence = {1: 1,  2: 2,
                        3: 3,  4: 4,
                        5: 3,  6: 4,
                        7: 5,  8: 6,
                        9: 7,  10: 8,
                        11: 9, 12: 10,
                        13: 11, 14: 4,
                        15: 5,  16: 6,
                        17: 7,  18: 8,
                        19: 9,  20: 10,
                        21: 11, 22: 12,
                        23: 13, 24: 14,
                        25: 15, 26: 16,
                        27: 17, 28: 18,
                        29: 19, 30: 20,
                        31: 13, 32: 14,
                        33: 5,  34: 6,
                        35: 7,  36: 8,
                        37: 9,  38: 10,
                        39: 11, 40: 12,
                        41: 13, 42: 14,
                        43: 15, 44: 16,
                        45: 17, 46: 18,
                        47: 19, 48: 20,
                        49: 13, 50: 14,
                        51: 15, 52: 16,
                        53: 17, 54: 18,
                        55: 9, 56: 10,
                        57: 11, 72: 26,
                        73: 27, 74: 28,
                        75: 15, 76: 16,
                        77: 17, 78: 18,
                        79: 19, 80: 20,
                        81: 13, 82: 14,
                        83: 15}
        
        if self.nao_max == 13:
            self.index_change = torch.LongTensor([0,1,3,4,2,6,7,5,10,11,9,12,8])
            self.row = self.col = o3.Irreps("1x0e+1x0e+1x1o+1x1o+1x2e")
            self.minus_index = torch.LongTensor([3,4,6,7,9,10])
            self.basis_def = (lambda s1=[0],s2=[1],p1=[2,3,4],p2=[5,6,7],d1=[8,9,10,11,12]: {
                1 : np.array(s1+s2+p1, dtype=int), # H
                2 : np.array(s1+s2+p1, dtype=int), # He
                5 : np.array(s1+s2+p1+p2+d1, dtype=int), # B
                6 : np.array(s1+s2+p1+p2+d1, dtype=int), # C
                7 : np.array(s1+s2+p1+p2+d1, dtype=int), # N
                8 : np.array(s1+s2+p1+p2+d1, dtype=int), # O
                9 : np.array(s1+s2+p1+p2+d1, dtype=int), # F
                10: np.array(s1+s2+p1+p2+d1, dtype=int), # Ne
                14: np.array(s1+s2+p1+p2+d1, dtype=int), # Si
                15: np.array(s1+s2+p1+p2+d1, dtype=int), # P
                16: np.array(s1+s2+p1+p2+d1, dtype=int), # S
                17: np.array(s1+s2+p1+p2+d1, dtype=int), # Cl
                18: np.array(s1+s2+p1+p2+d1, dtype=int), # Ar
            })()           
        
        elif self.nao_max == 27:
            self.index_change = torch.LongTensor([0,1,2,3,5,6,4,8,9,7,12,13,11,14,10,17,18,16,19,15,23,24,22,25,21,26,20])       
            self.row = self.col = o3.Irreps("1x0e+1x0e+1x0e+1x0e+1x1o+1x1o+1x2e+1x2e+1x3o")
            self.minus_index = torch.LongTensor([5,6,8,9,11,12,16,17,21,22,25,26]) # this list should follow the order in abacus.
            self.basis_def = (lambda s1=[0],s2=[1],s3=[2],s4=[3],p1=[4,5,6],p2=[7,8,9],d1=[10,11,12,13,14],d2=[15,16,17,18,19],f1=[20,21,22,23,24,25,26]: {
            1 : s1+s2+p1, # H
            2 : s1+s2+p1, # He
            3 : s1+s2+s3+s4+p1, # Li
            4 : s1+s2+s3+s4+p1, # Bi
            5 : s1+s2+p1+p2+d1, # B
            6 : s1+s2+p1+p2+d1, # C
            7 : s1+s2+p1+p2+d1, # N
            8 : s1+s2+p1+p2+d1, # O
            9 : s1+s2+p1+p2+d1, # F
            10: s1+s2+p1+p2+d1, # Ne
            11: s1+s2+s3+s4+p1+p2+d1, # Na
            12: s1+s2+s3+s4+p1+p2+d1, # Mg
            # 13: Al
            14: s1+s2+p1+p2+d1, # Si
            15: s1+s2+p1+p2+d1, # P
            16: s1+s2+p1+p2+d1, # S
            17: s1+s2+p1+p2+d1, # Cl
            18: s1+s2+p1+p2+d1, # Ar
            19: s1+s2+s3+s4+p1+p2+d1, # K
            20: s1+s2+s3+s4+p1+p2+d1, # Ca
            21: s1+s2+s3+s4+p1+p2+d1+d2+f1, # Sc
            22: s1+s2+s3+s4+p1+p2+d1+d2+f1, # Ti
            23: s1+s2+s3+s4+p1+p2+d1+d2+f1, # V
            24: s1+s2+s3+s4+p1+p2+d1+d2+f1, # Cr
            25: s1+s2+s3+s4+p1+p2+d1+d2+f1, # Mn
            26: s1+s2+s3+s4+p1+p2+d1+d2+f1, # Fe
            27: s1+s2+s3+s4+p1+p2+d1+d2+f1, # Co
            28: s1+s2+s3+s4+p1+p2+d1+d2+f1, # Ni
            29: s1+s2+s3+s4+p1+p2+d1+d2+f1, # Cu
            30: s1+s2+s3+s4+p1+p2+d1+d2+f1, # Zn
            31: s1+s2+p1+p2+d1+d2+f1, # Ga
            32: s1+s2+p1+p2+d1+d2+f1, # Ge
            33: s1+s2+p1+p2+d1, # As
            34: s1+s2+p1+p2+d1, # Se
            35: s1+s2+p1+p2+d1, # Br
            36: s1+s2+p1+p2+d1, # Kr
            37: s1+s2+s3+s4+p1+p2+d1, # Rb
            38: s1+s2+s3+s4+p1+p2+d1, # Sr
            39: s1+s2+s3+s4+p1+p2+d1+d2+f1, # Y
            40: s1+s2+s3+s4+p1+p2+d1+d2+f1, # Zr
            41: s1+s2+s3+s4+p1+p2+d1+d2+f1, # Nb
            42: s1+s2+s3+s4+p1+p2+d1+d2+f1, # Mo
            43: s1+s2+s3+s4+p1+p2+d1+d2+f1, # Tc
            44: s1+s2+s3+s4+p1+p2+d1+d2+f1, # Ru
            45: s1+s2+s3+s4+p1+p2+d1+d2+f1, # Rh
            46: s1+s2+s3+s4+p1+p2+d1+d2+f1, # Pd
            47: s1+s2+s3+s4+p1+p2+d1+d2+f1, # Ag
            48: s1+s2+s3+s4+p1+p2+d1+d2+f1, # Cd
            49: s1+s2+p1+p2+d1+d2+f1, # In
            50: s1+s2+p1+p2+d1+d2+f1, # Sn
            51: s1+s2+p1+p2+d1+d2+f1, # Sb
            52: s1+s2+p1+p2+d1+d2+f1, # Te
            53: s1+s2+p1+p2+d1+d2+f1, # I
            54: s1+s2+p1+p2+d1+d2+f1, # Xe
            55: s1+s2+s3+s4+p1+p2+d1, # Cs
            56: s1+s2+s3+s4+p1+p2+d1+d2+f1, # Ba
            #
            79: s1+s2+s3+s4+p1+p2+d1+d2+f1, # Au
            80: s1+s2+s3+s4+p1+p2+d1+d2+f1, # Hg
            81: s1+s2+p1+p2+d1+d2+f1, # Tl
            82: s1+s2+p1+p2+d1+d2+f1, # Pb
            83: s1+s2+p1+p2+d1+d2+f1, # Bi
        })()
        elif self.nao_max == 40:
            self.index_change = torch.LongTensor([0,1,2,3,5,6,4,8,9,7,11,12,10,14,15,13,18,19,17,20,16,23,24,22,25,21,29,30,28,31,27,32,26,36,37,35,38,34,39,33])       
            self.row = self.col = o3.Irreps("1x0e+1x0e+1x0e+1x0e+1x1o+1x1o+1x1o+1x1o+1x2e+1x2e+1x3o+1x3o")
            self.minus_index = torch.LongTensor([5,6,8,9,11,12,14,15,17,18,22,23,27,28,31,32,34,35,38,39]) # this list should follow the order in abacus.
            self.basis_def = (lambda s1=[0],
                   s2=[1],
                   s3=[2],
                   s4=[3],
                   p1=[4,5,6],
                   p2=[7,8,9],
                   p3=[10,11,12],
                   p4=[13,14,15],
                   d1=[16,17,18,19,20],
                   d2=[21,22,23,24,25],
                   f1=[26,27,28,29,30,31,32],
                   f2=[33,34,35,36,37,38,39]: {
                Element('Ag').Z: s1+s2+s3+s4+p1+p2+d1+d2+f1, 
                Element('Al').Z: s1+s2+s3+s4+p1+p2+p3+p4+d1, 
                Element('Ar').Z: s1+s2+p1+p2+d1, 
                Element('As').Z: s1+s2+p1+p2+d1, 
                Element('Au').Z: s1+s2+s3+s4+p1+p2+d1+d2+f1, 
                Element('Ba').Z: s1+s2+s3+s4+p1+p2+d1+d2+f1, 
                Element('Be').Z: s1+s2+s3+s4+p1, 
                Element('B').Z: s1+s2+p1+p2+d1, 
                Element('Bi').Z: s1+s2+p1+p2+d1+d2+f1, 
                Element('Br').Z: s1+s2+p1+p2+d1, 
                Element('Ca').Z: s1+s2+s3+s4+p1+p2+d1, 
                Element('Cd').Z: s1+s2+s3+s4+p1+p2+d1+d2+f1, 
                Element('C').Z: s1+s2+p1+p2+d1, 
                Element('Cl').Z: s1+s2+p1+p2+d1, 
                Element('Co').Z: s1+s2+s3+s4+p1+p2+d1+d2+f1, 
                Element('Cr').Z: s1+s2+s3+s4+p1+p2+d1+d2+f1, 
                Element('Cs').Z: s1+s2+s3+s4+p1+p2+d1, 
                Element('Cu').Z: s1+s2+s3+s4+p1+p2+d1+d2+f1, 
                Element('Fe').Z: s1+s2+s3+s4+p1+p2+d1+d2+f1, 
                Element('F').Z: s1+s2+p1+p2+d1, 
                Element('Ga').Z: s1+s2+p1+p2+d1+d2+f1, 
                Element('Ge').Z: s1+s2+p1+p2+d1+d2+f1, 
                Element('He').Z: s1+s2+p1, 
                Element('Hf').Z: s1+s2+s3+s4+p1+p2+d1+d2+f1+f2,  # Hf_gga_10au_100Ry_4s2p2d2f.orb
                Element('H').Z: s1+s2+p1, 
                Element('Hg').Z: s1+s2+s3+s4+p1+p2+d1+d2+f1, 
                Element('I').Z: s1+s2+p1+p2+d1+d2+f1, 
                Element('In').Z: s1+s2+p1+p2+d1+d2+f1, 
                Element('Ir').Z: s1+s2+s3+s4+p1+p2+d1+d2+f1, 
                Element('K').Z: s1+s2+s3+s4+p1+p2+d1, 
                Element('Kr').Z: s1+s2+p1+p2+d1, 
                Element('Li').Z: s1+s2+s3+s4+p1, 
                Element('Mg').Z: s1+s2+s3+s4+p1+p2+d1, 
                Element('Mn').Z: s1+s2+s3+s4+p1+p2+d1+d2+f1, 
                Element('Mo').Z: s1+s2+s3+s4+p1+p2+d1+d2+f1, 
                Element('Na').Z: s1+s2+s3+s4+p1+p2+d1, 
                Element('Nb').Z: s1+s2+s3+s4+p1+p2+d1+d2+f1, 
                Element('Ne').Z: s1+s2+p1+p2+d1, 
                Element('N').Z: s1+s2+p1+p2+d1, 
                Element('Ni').Z: s1+s2+s3+s4+p1+p2+d1+d2+f1, 
                Element('O').Z: s1+s2+p1+p2+d1, 
                Element('Os').Z: s1+s2+s3+s4+p1+p2+d1+d2+f1, 
                Element('Pb').Z: s1+s2+p1+p2+d1+d2+f1, 
                Element('Pd').Z: s1+s2+s3+s4+p1+p2+d1+d2+f1, 
                Element('P').Z: s1+s2+p1+p2+d1, 
                Element('Pt').Z: s1+s2+s3+s4+p1+p2+d1+d2+f1, 
                Element('Rb').Z: s1+s2+s3+s4+p1+p2+d1, 
                Element('Re').Z: s1+s2+s3+s4+p1+p2+d1+d2+f1, 
                Element('Rh').Z: s1+s2+s3+s4+p1+p2+d1+d2+f1, 
                Element('Ru').Z: s1+s2+s3+s4+p1+p2+d1+d2+f1, 
                Element('Sb').Z: s1+s2+p1+p2+d1+d2+f1, 
                Element('Sc').Z: s1+s2+s3+s4+p1+p2+d1+d2+f1, 
                Element('Se').Z: s1+s2+p1+p2+d1, 
                Element('S').Z: s1+s2+p1+p2+d1, 
                Element('Si').Z: s1+s2+p1+p2+d1, 
                Element('Sn').Z: s1+s2+p1+p2+d1+d2+f1, 
                Element('Sr').Z: s1+s2+s3+s4+p1+p2+d1, 
                Element('Ta').Z: s1+s2+s3+s4+p1+p2+d1+d2+f1+f2,  # Ta_gga_10au_100Ry_4s2p2d2f.orb
                Element('Tc').Z: s1+s2+s3+s4+p1+p2+d1+d2+f1, 
                Element('Te').Z: s1+s2+p1+p2+d1+d2+f1, 
                Element('Ti').Z: s1+s2+s3+s4+p1+p2+d1+d2+f1, 
                Element('Tl').Z: s1+s2+p1+p2+d1+d2+f1, 
                Element('V').Z: s1+s2+s3+s4+p1+p2+d1+d2+f1, 
                Element('W').Z: s1+s2+s3+s4+p1+p2+d1+d2+f1+f2,  # W_gga_10au_100Ry_4s2p2d2f.orb
                Element('Xe').Z: s1+s2+p1+p2+d1+d2+f1, 
                Element('Y').Z: s1+s2+s3+s4+p1+p2+d1+d2+f1, 
                Element('Zn').Z: s1+s2+s3+s4+p1+p2+d1+d2+f1, 
                Element('Zr').Z: s1+s2+s3+s4+p1+p2+d1+d2+f1,
                })()
        else:
            raise NotImplementedError(f"NAO max '{self.nao_max}' not supported for 'abacus'.")

    def _configure_band_num_control(self, band_num_control):
        """
        Configure band number control settings.
        
        Args:
            band_num_control (dict or int): Band number control configuration.
                If dict, maps atomic numbers to band counts.
                If int, specifies a global band count.
        """
        if band_num_control is not None and not self.export_reciprocal_values:
            if isinstance(band_num_control, dict):
                # Convert atomic numbers to integers for consistency
                self.band_num_control = {int(k): v for k, v in band_num_control.items()}
            elif isinstance(band_num_control, int):
                self.band_num_control = band_num_control
            else:
                self.band_num_control = None
        else:
            self.band_num_control = None

    def _create_hamiltonian_layer(self, irreps_in, irreps_out):
        """
        Create a Hamiltonian neural network layer.
        
        Args:
            irreps_in (o3.Irreps): Input irreducible representations.
            irreps_out (o3.Irreps): Output irreducible representations.
            
        Returns:
            HamLayer: A neural network layer for Hamiltonian prediction.
        """
        return HamLayer(
            irreps_in=irreps_in,
            feature_irreps_hidden=irreps_in,
            irreps_out=irreps_out,
            nonlinearity_type=self.nonlinearity_type,
            resnet=True
        )

    def merge_tensor_components(self, spherical_components):
        """
        Merge spherical tensor components into a matrix using Clebsch-Gordan coefficients.

        Args:
            spherical_components (list of torch.Tensor): List of tensors representing spherical 
                components of irreducible representations. Each tensor has shape 
                (batch_size, component_dim).

        Returns:
            torch.Tensor: Merged matrix with shape (batch_size, nao_max * nao_max) representing
                flattened matrix of merged components.
        """
        batch_size = spherical_components[0].shape[0]
        result_matrix = torch.zeros(batch_size, self.nao_max, self.nao_max).type_as(spherical_components[0])

        component_index = 0  # Index for accessing the correct irreps
        row_start_idx = 0
        for _, row_orbital in self.row:
            row_dimension = 2 * row_orbital.l + 1  # Dimension based on angular momentum
            col_start_idx = 0
            for _, col_orbital in self.col:
                col_dimension = 2 * col_orbital.l + 1  # Dimension based on angular momentum

                # Iterate through allowed angular momentum values
                for angular_momentum in range(abs(row_orbital.l - col_orbital.l), row_orbital.l + col_orbital.l + 1):
                    # Compute inverse spherical tensor product             
                    clebsch_gordan_coef = math.sqrt(2 * angular_momentum + 1) * self.cg_calculator(
                        row_orbital.l, col_orbital.l, angular_momentum
                    ).unsqueeze(0)
                    tensor_product = (clebsch_gordan_coef * spherical_components[component_index].unsqueeze(-2).unsqueeze(-2)).sum(-1)

                    # Add product to appropriate part of the result matrix
                    submatrix = result_matrix.narrow(-2, row_start_idx, row_dimension).narrow(-1, col_start_idx, col_dimension)
                    submatrix += tensor_product
                    component_index += 1

                col_start_idx += col_dimension
            row_start_idx += row_dimension

        return result_matrix.reshape(-1, self.nao_max * self.nao_max)

    def merge_rank2_tensor_components(self, spherical_components):
        """
        Merge rank-2 tensor components into a block matrix representation.

        This function is specialized for rank-2 tensors and applies permutation to the result.

        Args:
            spherical_components (list of torch.Tensor): List of tensors representing spherical
                components of irreducible representations.

        Returns:
            torch.Tensor: Merged block matrix with shape (batch_size, n_blocks, 3, 3) after
                applying a specific permutation.
        """
        batch_size = spherical_components[0].shape[0]
        result_matrix = torch.zeros(batch_size, self.num_blocks, 3, 3).type_as(spherical_components[0])
        permutation_indices = torch.LongTensor([2, 0, 1])  # Indices for coordinate permutation

        component_index = 0  # Index for accessing the correct irreps
        block_index = 0
        for _, row_orbital in self.row:
            for _, col_orbital in self.col:
                for angular_momentum in range(0, 3):
                    # Compute inverse spherical tensor product             
                    clebsch_gordan_coef = math.sqrt(2 * angular_momentum + 1) * self.cg_calculator(1, 1, angular_momentum).unsqueeze(0)
                    tensor_product = (clebsch_gordan_coef * spherical_components[component_index].unsqueeze(-2).unsqueeze(-2)).sum(-1)

                    # Add product to appropriate part of the block
                    result_matrix[:, block_index, :, :] += tensor_product
                    component_index += 1
                block_index += 1

        # Apply permutation to the result
        permuted_result = result_matrix[:, :, permutation_indices[:, None], permutation_indices[None, :]]
        return permuted_result

    def merge_rank0_tensor_components(self, spherical_components):
        """
        Merge rank-0 (scalar) tensor components into a matrix representation.

        Args:
            spherical_components (list of torch.Tensor): List of tensors representing spherical
                components of irreducible representations.

        Returns:
            torch.Tensor: Merged matrix with shape (batch_size, nao_max, nao_max).
        """
        batch_size = spherical_components[0].shape[0]
        result_matrix = torch.zeros(batch_size, self.nao_max, self.nao_max).type_as(spherical_components[0])

        component_index = 0  # Index for accessing the correct irreps
        row_start_idx = 0
        for _, row_orbital in self.row:
            row_dimension = 2 * row_orbital.l + 1  # Dimension based on angular momentum
            col_start_idx = 0
            for _, col_orbital in self.col:
                col_dimension = 2 * col_orbital.l + 1  # Dimension based on angular momentum

                # For rank-0, we just expand the component to fill the submatrix
                expanded_component = spherical_components[component_index].unsqueeze(-1).expand(-1, row_dimension, col_dimension)

                # Add expanded component to appropriate part of the result matrix
                submatrix = result_matrix.narrow(-2, row_start_idx, row_dimension).narrow(-1, col_start_idx, col_dimension)
                submatrix += expanded_component
                component_index += 1
                col_start_idx += col_dimension
            row_start_idx += row_dimension

        return result_matrix

    def construct_j_coupling_matrix(self, coupling_coefficients):
        """
        Construct a matrix representation of J-coupling (spin-orbit interaction).

        Builds a matrix representation for J-coupling which can be either rank-2 
        (with spin-orbit coupling) or rank-0.

        Args:
            coupling_coefficients (torch.Tensor): Tensor containing coupling coefficients.

        Returns:
            torch.Tensor: The J-coupling matrix. If spin-orbit coupling is enabled,
                shape is (batch_size, nao_max, nao_max, 3, 3); otherwise, shape is
                (batch_size, nao_max, nao_max).
        """
        # Split coupling coefficients into spherical components
        spherical_components = torch.split(coupling_coefficients, self.J_irreps_dim.tolist(), dim=-1)

        if self.soc_switch:  # If spin-orbit coupling is enabled
            # Use rank-2 tensor merge for spin-orbit coupling
            processed_coefficients = self.merge_rank2_tensor_components(spherical_components)
            result_matrix = torch.zeros(
                processed_coefficients.shape[0], self.nao_max, self.nao_max, 3, 3
            ).type_as(processed_coefficients)

            # Distribute the blocks to the full matrix
            block_index = 0
            row_start_idx = 0
            for _, row_orbital in self.row:
                row_dimension = 2 * row_orbital.l + 1
                col_start_idx = 0
                for _, col_orbital in self.col:
                    col_dimension = 2 * col_orbital.l + 1

                    # Expand block to fill the corresponding submatrix
                    block_expanded = processed_coefficients[:, block_index].reshape(-1, 1, 1, 3, 3).expand(
                        processed_coefficients.shape[0], row_dimension, col_dimension, 3, 3
                    )
                    result_matrix[:, row_start_idx:row_start_idx+row_dimension, 
                                  col_start_idx:col_start_idx+col_dimension] = block_expanded

                    block_index += 1
                    col_start_idx += col_dimension
                row_start_idx += row_dimension
        else:
            # Use rank-0 tensor merge if spin-orbit coupling is disabled
            result_matrix = self.merge_rank0_tensor_components(spherical_components)

        return result_matrix

    def construct_k_coupling_matrix(self, coupling_coefficients):
        """
        Construct a matrix representation of exchange coupling (K-term).

        Args:
            coupling_coefficients (torch.Tensor): Tensor containing exchange coupling coefficients
                with shape (n_atoms_or_edges, coefficients_per_block * n_blocks).

        Returns:
            torch.Tensor: Exchange coupling matrix with shape (n_atoms_or_edges, nao_max, nao_max).
        """
        # Reshape to separate blocks
        reshaped_coefficients = coupling_coefficients.reshape(coupling_coefficients.shape[0], -1)

        # Initialize result matrix
        result_matrix = torch.zeros(
            reshaped_coefficients.shape[0], self.nao_max, self.nao_max
        ).type_as(reshaped_coefficients)

        # Distribute coefficients to blocks
        block_index = 0
        row_start_idx = 0
        for _, row_orbital in self.row:
            row_dimension = 2 * row_orbital.l + 1
            col_start_idx = 0
            for _, col_orbital in self.col:
                col_dimension = 2 * col_orbital.l + 1

                # Expand coefficient to fill the block
                expanded_coefficient = reshaped_coefficients[:, block_index].reshape(-1, 1, 1).expand(
                    reshaped_coefficients.shape[0], row_dimension, col_dimension
                )

                # Place in the appropriate position in the result matrix
                result_matrix[:, row_start_idx:row_start_idx+row_dimension, 
                              col_start_idx:col_start_idx+col_dimension] = expanded_coefficient

                block_index += 1
                col_start_idx += col_dimension
            row_start_idx += row_dimension

        return result_matrix

    def reorder_matrix(self, matrix):
        """
        Reorder matrix elements to match the atomic orbital convention used by DFT.

        This function performs two types of transformations:
        1. Reorders rows and columns according to a predefined permutation (if defined)
        2. Flips the sign of specific elements (if defined)

        These transformations ensure compatibility with DFT's atomic orbital ordering
        convention, which may differ from the internal representation.

        Args:
            matrix (torch.Tensor): Input matrix in flattened form,
                with shape (batch_size, nao_max2) where nao_max is the maximum
                number of atomic orbitals.

        Returns:
            torch.Tensor: Reordered matrix in the same shape as input,
                but with elements rearranged to match DFT conventions.
        """
        # Only perform reordering if necessary index maps are defined
        if self.index_change is not None or hasattr(self, 'minus_index'):
            # Reshape to 3D tensor for easier manipulation
            matrix_form = matrix.reshape(-1, self.nao_max, self.nao_max)

            # Apply index permutation if defined
            if self.index_change is not None:
                # Use advanced indexing to permute both rows and columns
                matrix_form = matrix_form[:, self.index_change[:, None], self.index_change[None, :]]

            # Apply sign flipping if defined
            if hasattr(self, 'minus_index'):
                # Flip signs for specified rows
                matrix_form[:, self.minus_index, :] = -matrix_form[:, self.minus_index, :]
                # Flip signs for specified columns
                matrix_form[:, :, self.minus_index] = -matrix_form[:, :, self.minus_index]

            # Return to original flattened shape
            matrix = matrix_form.reshape(-1, self.nao_max**2)

        return matrix

    def construct_molecular_hamiltonian(self, data, onsite_hamiltonian, offsite_hamiltonian):
        """
        Construct a complete molecular Hamiltonian matrix from on-site and off-site components.

        This function transforms separate on-site (diagonal) and off-site (interaction) Hamiltonian 
        components into a unified molecular Hamiltonian matrix. It handles atom-specific orbital 
        basis sets and masks invalid or padded orbitals.

        Args:
            data (DataObject): Object containing crystal structure information, including:
                - z: Atomic numbers
                - node_counts: Number of atoms in each crystal
                - edge_index: Edges between atoms (indices of connected atom pairs)
            onsite_hamiltonian (torch.Tensor): On-site Hamiltonian matrix elements for each atom,
                with shape (n_atoms, nao_max^2)
            offsite_hamiltonian (torch.Tensor): Off-site Hamiltonian matrix elements for each edge,
                with shape (n_edges, nao_max^2)

        Returns:
            torch.Tensor: Complete molecular Hamiltonian matrix with shape (n_molecules, n_orbitals, n_orbitals),
                where n_orbitals is the number of valid atomic orbitals per molecule after removing
                padding.
        """
        # Get the maximum number of atoms in any crystal in the batch
        max_atoms_per_crystal = torch.max(data.node_counts).item()
        n_atoms = data.z.shape[0]

        # Create orbital validity mask based on atomic numbers
        # (determine which orbitals are valid for each atom type)
        atom_orbital_mask = torch.zeros((99, self.nao_max), device=data.z.device).type_as(data.z)
        basis_definition_mapped = copy.deepcopy(self.basis_def)

        # Convert 1-indexed orbital indices to 0-indexed
        for atomic_number, orbital_indices in self.basis_def.items():
            zero_indexed_indices = [idx - 1 for idx in orbital_indices]  # Convert to 0-indexed
            basis_definition_mapped[atomic_number] = zero_indexed_indices
            atom_orbital_mask[atomic_number][zero_indexed_indices] = 1

        # Get the orbital mask for each atom in the batch
        # First reshape to (n_batch_crystals, max_atoms*nao_max)
        crystal_orbital_mask = atom_orbital_mask[data.z].view(-1, max_atoms_per_crystal * self.nao_max)

        # Create a 2D mask for interactions between all orbitals (outer product)
        # Shape: (n_batch_crystals, max_atoms*nao_max, max_atoms*nao_max)
        orbital_interaction_mask = crystal_orbital_mask[:, :, None] * crystal_orbital_mask[:, None, :]

        # Reshape mask to match the final Hamiltonian format
        orbital_interaction_mask = orbital_interaction_mask.view(-1, max_atoms_per_crystal * self.nao_max)

        # Initialize the combined Hamiltonian matrix
        # Shape: (n_atoms, max_atoms_per_crystal, nao_max^2)
        combined_hamiltonian = torch.zeros(
            [n_atoms, max_atoms_per_crystal, self.nao_max**2]
        ).type_as(onsite_hamiltonian)

        # Create atom index tensor for efficient indexing
        atom_indices = torch.arange(n_atoms, device=data.z.device).type_as(data.z)

        # Place on-site Hamiltonian elements
        # For each atom, place its on-site Hamiltonian at the corresponding position
        combined_hamiltonian[atom_indices, atom_indices % max_atoms_per_crystal] = onsite_hamiltonian

        # Place off-site Hamiltonian elements
        # For each edge (i,j), place the off-site Hamiltonian from i to j
        source_atoms, target_atoms = data.edge_index
        combined_hamiltonian[source_atoms, target_atoms % max_atoms_per_crystal] = offsite_hamiltonian

        # Reshape and reorder dimensions to get the correct Hamiltonian structure
        # From (n_atoms, max_atoms, nao_max, nao_max) to (n_atoms*nao_max, max_atoms*nao_max)
        combined_hamiltonian = combined_hamiltonian.reshape(
            n_atoms, max_atoms_per_crystal, self.nao_max, self.nao_max
        )
        combined_hamiltonian = combined_hamiltonian.permute((0, 2, 1, 3))
        combined_hamiltonian = combined_hamiltonian.reshape(
            n_atoms * self.nao_max, max_atoms_per_crystal * self.nao_max
        )

        # Apply mask to remove padded/invalid orbitals
        valid_elements = torch.masked_select(combined_hamiltonian, orbital_interaction_mask > 0)

        # Calculate the number of valid orbitals per molecule
        n_molecules = n_atoms / max_atoms_per_crystal
        orbitals_per_molecule = int(math.sqrt(valid_elements.shape[0] / n_molecules))

        # Reshape to final molecular Hamiltonian form
        molecular_hamiltonian = valid_elements.reshape(-1, orbitals_per_molecule)

        return molecular_hamiltonian
    
    def concatenate_hamiltonians_by_crystal(self, data, onsite_hamiltonians, offsite_hamiltonians):
        """
        Concatenate on-site and off-site Hamiltonian matrices for each crystal in a batch.

        This function organizes Hamiltonian matrices by crystal, interleaving on-site and
        off-site components in a specific order required for further processing.

        Args:
            data (DataObject): Object containing crystal structure information, including:
                - node_counts: Number of atoms in each crystal
                - edge_index: Indices of connected atom pairs
                - batch: Batch assignment for each node
            onsite_hamiltonians (torch.Tensor): On-site Hamiltonian matrices for all atoms,
                with shape (total_atoms, matrix_dimension)
            offsite_hamiltonians (torch.Tensor): Off-site Hamiltonian matrices for all edges,
                with shape (total_edges, matrix_dimension)

        Returns:
            torch.Tensor: Concatenated Hamiltonian matrices organized by crystal,
                with alternating on-site and off-site blocks.
        """
        # Split on-site Hamiltonians by crystal based on node counts
        atoms_per_crystal = data.node_counts
        onsite_by_crystal = torch.split(onsite_hamiltonians, atoms_per_crystal.tolist(), dim=0)

        # Get source and target indices for each edge
        source_nodes, target_nodes = data.edge_index

        # Count edges per crystal using scatter operation
        edge_counter = torch.ones_like(source_nodes)
        edges_per_crystal = scatter(edge_counter, data.batch[source_nodes], dim=0)

        # Split off-site Hamiltonians by crystal based on edge counts
        offsite_by_crystal = torch.split(offsite_hamiltonians, edges_per_crystal.tolist(), dim=0)

        # Interleave on-site and off-site Hamiltonians for each crystal
        concatenated_hamiltonians = []
        for crystal_idx in range(len(atoms_per_crystal)):
            concatenated_hamiltonians.append(onsite_by_crystal[crystal_idx])  # Add on-site Hamiltonians
            concatenated_hamiltonians.append(offsite_by_crystal[crystal_idx])  # Add off-site Hamiltonians

        # Concatenate all Hamiltonians into a single tensor
        return torch.cat(concatenated_hamiltonians, dim=0)
    
    def symmetrize_hamiltonian(self, hamiltonian, is_soc=False, inverse_edges=None, symmetry_type='hermitian'):
        """
        Apply symmetrization to a Hamiltonian matrix.

        This is a general-purpose function that handles various types of symmetrization for both
        on-site and off-site Hamiltonians, with or without spin-orbit coupling (SOC).

        Args:
            hamiltonian (torch.Tensor): Hamiltonian matrix elements in flattened form.
            is_soc (bool, optional): Whether this is a spin-orbit coupling Hamiltonian with
                double dimension. Defaults to False.
            inverse_edges (torch.Tensor, optional): Indices mapping each edge to its inverse
                for off-site Hamiltonians. Required for off-site symmetrization. Defaults to None.
            symmetry_type (str, optional): Type of symmetry to apply:
                - 'hermitian': Apply Hermitian symmetry H = 0.5*(H + H?)
                - 'anti-hermitian': Apply anti-Hermitian symmetry H = 0.5*(H - H?)
                Defaults to 'hermitian'.

        Returns:
            torch.Tensor: Symmetrized Hamiltonian matrix in the same format as input.
        """
        # Return original matrix if symmetrization is disabled
        if not self.symmetrize:
            # For SOC case, ensure consistent shape even when not symmetrizing
            if is_soc:
                dimension = 2 * self.nao_max
                return hamiltonian.reshape(-1, dimension**2)
            return hamiltonian

        # Determine matrix dimension based on whether this is SOC
        dimension = 2 * self.nao_max if is_soc else self.nao_max

        # Reshape for matrix operations
        matrix_form = hamiltonian.reshape(-1, dimension, dimension)

        # Determine symmetrization operation based on type
        if symmetry_type == 'hermitian':
            if inverse_edges is None:
                # On-site case: symmetrize with its own transpose
                symmetrized = 0.5 * (matrix_form + matrix_form.conj().permute(0, 2, 1))
            else:
                # Off-site case: symmetrize with transpose of inverse edges
                symmetrized = 0.5 * (matrix_form + matrix_form[inverse_edges].conj().permute(0, 2, 1))
        elif symmetry_type == 'anti-hermitian':
            if inverse_edges is None:
                # On-site case: anti-symmetrize with its own transpose
                symmetrized = 0.5 * (matrix_form - matrix_form.conj().permute(0, 2, 1))
            else:
                # Off-site case: anti-symmetrize with transpose of inverse edges
                symmetrized = 0.5 * (matrix_form - matrix_form[inverse_edges].conj().permute(0, 2, 1))
        else:
            raise ValueError(f"Unknown symmetry type: {symmetry_type}")

        # Reshape back to original format
        return symmetrized.reshape(-1, dimension**2)

    def symmetrize_onsite_hamiltonian(self, hamiltonian, hermitian=True):
        """
        Symmetrize on-site Hamiltonian matrices.

        Applies Hermitian or anti-Hermitian symmetrization to on-site Hamiltonians.

        Args:
            hamiltonian (torch.Tensor): On-site Hamiltonian matrix elements.
            hermitian (bool, optional): If True, apply Hermitian symmetry (H + H?),
                otherwise apply anti-Hermitian symmetry (H - H?). Defaults to True.

        Returns:
            torch.Tensor: Symmetrized Hamiltonian.
        """
        symmetry_type = 'hermitian' if hermitian else 'anti-hermitian'
        return self.symmetrize_hamiltonian(
            hamiltonian, is_soc=False, inverse_edges=None, symmetry_type=symmetry_type
        )

    def symmetrize_offsite_hamiltonian(self, hamiltonian, inverse_edges, hermitian=True):
        """
        Symmetrize off-site Hamiltonian matrices.

        Applies Hermitian or anti-Hermitian symmetrization to off-site Hamiltonians,
        using the inverse edge mapping to relate connected atom pairs.

        Args:
            hamiltonian (torch.Tensor): Off-site Hamiltonian matrix elements.
            inverse_edges (torch.Tensor): Tensor mapping each edge to its inverse edge index.
            hermitian (bool, optional): If True, apply Hermitian symmetry (H + H?),
                otherwise apply anti-Hermitian symmetry (H - H?). Defaults to True.

        Returns:
            torch.Tensor: Symmetrized Hamiltonian.
        """
        symmetry_type = 'hermitian' if hermitian else 'anti-hermitian'
        return self.symmetrize_hamiltonian(
            hamiltonian, is_soc=False, inverse_edges=inverse_edges, symmetry_type=symmetry_type
        )

    def symmetrize_onsite_hamiltonian_soc(self, hamiltonian, hermitian=True):
        """
        Symmetrize on-site Hamiltonian matrices with spin-orbit coupling.

        Applies Hermitian or anti-Hermitian symmetrization to on-site Hamiltonians
        that include spin-orbit coupling, which have double the dimension.

        Args:
            hamiltonian (torch.Tensor): On-site SOC Hamiltonian matrix elements.
            hermitian (bool, optional): If True, apply Hermitian symmetry (H + H?),
                otherwise apply anti-Hermitian symmetry (H - H?). Defaults to True.

        Returns:
            torch.Tensor: Symmetrized SOC Hamiltonian.
        """
        symmetry_type = 'hermitian' if hermitian else 'anti-hermitian'
        return self.symmetrize_hamiltonian(
            hamiltonian, is_soc=True, inverse_edges=None, symmetry_type=symmetry_type
        )

    def symmetrize_offsite_hamiltonian_soc(self, hamiltonian, inverse_edges, hermitian=True):
        """
        Symmetrize off-site Hamiltonian matrices with spin-orbit coupling.

        Applies Hermitian or anti-Hermitian symmetrization to off-site Hamiltonians
        that include spin-orbit coupling, which have double the dimension.

        Args:
            hamiltonian (torch.Tensor): Off-site SOC Hamiltonian matrix elements.
            inverse_edges (torch.Tensor): Tensor mapping each edge to its inverse edge index.
            hermitian (bool, optional): If True, apply Hermitian symmetry (H + H?),
                otherwise apply anti-Hermitian symmetry (H - H?). Defaults to True.

        Returns:
            torch.Tensor: Symmetrized SOC Hamiltonian.
        """
        symmetry_type = 'hermitian' if hermitian else 'anti-hermitian'
        return self.symmetrize_hamiltonian(
            hamiltonian, is_soc=True, inverse_edges=inverse_edges, symmetry_type=symmetry_type
        )

    def calculate_band_energies_with_overlap(self, onsite_hamiltonian, offsite_hamiltonian, 
                                             onsite_overlap, offsite_overlap, crystal_data, 
                                             export_reciprocal_values=False):
        """
        Calculate electronic band structure using provided Hamiltonian and overlap matrices.

        This function computes electronic band energies, wavefunctions, and band gaps for a set 
        of crystal structures using the generalized eigenvalue problem H���� = E��S����. This version
        allows debugging by accepting both reference and predicted overlap matrices.

        Args:
            onsite_hamiltonian (torch.Tensor): On-site Hamiltonian matrix elements with shape 
                (total_atoms, nao_max2).
            offsite_hamiltonian (torch.Tensor): Off-site Hamiltonian matrix elements with shape 
                (total_edges, nao_max2).
            onsite_overlap (torch.Tensor): Predicted on-site overlap matrix elements with shape 
                (total_atoms, nao_max2).
            offsite_overlap (torch.Tensor): Predicted off-site overlap matrix elements with shape 
                (total_edges, nao_max2).
            crystal_data (DataObject): Object containing crystal structure information including:
                - edge_index: Indices of connected atom pairs
                - cell: Unit cell vectors
                - z: Atomic numbers
                - node_counts: Number of atoms in each crystal
                - batch: Batch assignment for each atom
                - k_vecs: k-points for band structure calculation
                - nbr_shift: Neighbor cell shifts for periodic boundary conditions
                - Son/Soff: Reference overlap matrices
            export_reciprocal_values (bool, optional): Whether to export additional reciprocal 
                space matrices (H(k), S(k), dS(k)). Defaults to False.

        Returns:
            tuple: Contains band energies, wavefunctions, and optionally additional reciprocal space 
                   matrices depending on the export_reciprocal_values parameter.
        """
        source_indices, target_indices = crystal_data.edge_index
        lattice_vectors = crystal_data.cell  # shape: (batch_size, 3, 3)
        batch_size = lattice_vectors.shape[0]

        # Create orbital validity mask based on atomic numbers
        atomic_orbital_mask = torch.zeros((99, self.nao_max)).type_as(crystal_data.z)
        for atomic_number, orbital_indices in self.basis_def.items():
            atomic_orbital_mask[atomic_number][orbital_indices] = 1

        # Get orbital mask for each atom
        orbital_mask = atomic_orbital_mask[crystal_data.z]  # shape: [total_atoms, nao_max]
        orbital_mask_by_crystal = torch.split(orbital_mask, crystal_data.node_counts.tolist(), dim=0)

        # Create interaction masks for each crystal
        orbital_interaction_masks = []
        for idx in range(batch_size):
            # Create outer product of orbital masks for interaction masking
            orbital_interaction_masks.append(
                orbital_mask_by_crystal[idx].reshape(-1, 1) * 
                orbital_mask_by_crystal[idx].reshape(1, -1)
            )

        # Set the number of valence electrons per atom type
        valence_electron_counts = torch.zeros((99,)).type_as(crystal_data.z)
        for atomic_number, count in self.num_valence.items():
            valence_electron_counts[atomic_number] = count

        # Get valence electron count for each atom and then for each crystal
        atom_valence_counts = valence_electron_counts[crystal_data.z]  # shape: [total_atoms]
        crystal_valence_counts = scatter(atom_valence_counts, crystal_data.batch, dim=0)  # shape: [batch_size]

        # Initialize band window sizes if needed
        if self.band_num_control is not None:
            bands_per_atom_type = torch.zeros((99,)).type_as(crystal_data.z)
            for atomic_number, band_count in self.band_num_control.items():
                bands_per_atom_type[atomic_number] = band_count
            atom_band_counts = bands_per_atom_type[crystal_data.z]  # shape: [total_atoms]
            crystal_band_counts = scatter(atom_band_counts, crystal_data.batch, dim=0)  # shape: [batch_size]

        # Split data by crystal
        atoms_per_crystal = crystal_data.node_counts
        atom_index_offsets = torch.cumsum(atoms_per_crystal, dim=0) - atoms_per_crystal

        # Split Hamiltonians and overlaps by crystal
        onsite_hamiltonians_by_crystal = torch.split(onsite_hamiltonian, atoms_per_crystal.tolist(), dim=0)
        reference_onsite_overlaps = torch.split(crystal_data.Son, atoms_per_crystal.tolist(), dim=0)
        predicted_onsite_overlaps = torch.split(onsite_overlap, atoms_per_crystal.tolist(), dim=0)

        # Count edges per crystal and split edge data
        edge_counter = torch.ones_like(source_indices)
        edges_per_crystal = scatter(edge_counter, crystal_data.batch[source_indices], dim=0)
        edge_index_offsets = torch.cumsum(edges_per_crystal, dim=0) - edges_per_crystal

        offsite_hamiltonians_by_crystal = torch.split(offsite_hamiltonian, edges_per_crystal.tolist(), dim=0)
        reference_offsite_overlaps = torch.split(crystal_data.Soff, edges_per_crystal.tolist(), dim=0)
        predicted_offsite_overlaps = torch.split(offsite_overlap, edges_per_crystal.tolist(), dim=0)

        # Prepare derivatives if exporting reciprocal values
        if export_reciprocal_values:
            onsite_overlap_derivatives = torch.split(crystal_data.dSon, atoms_per_crystal.tolist(), dim=0)
            offsite_overlap_derivatives = torch.split(crystal_data.dSoff, edges_per_crystal.tolist(), dim=0)

        # Initialize results containers
        band_energies = []
        wavefunctions = []
        reciprocal_hamiltonians = []
        symmetrized_hamiltonians = []
        reciprocal_overlaps = []
        reciprocal_overlap_derivatives = []
        band_gaps = []

        # Process each crystal
        for crystal_idx in range(batch_size):
            k_points = crystal_data.k_vecs[crystal_idx]
            num_atoms = atoms_per_crystal[crystal_idx]

            # Calculate phase factors for k-points
            # Shape: (edges_in_crystal, num_k_points)
            phase_factors = torch.exp(
                2j * torch.pi * torch.sum(
                    crystal_data.nbr_shift[
                        edge_index_offsets[crystal_idx] + 
                        torch.arange(edges_per_crystal[crystal_idx]).type_as(source_indices),
                        None, :
                    ] * k_points[None, :, :],
                    dim=-1
                )
            )

            # Initialize k-space Hamiltonian and overlap matrices
            hamiltonian_k = torch.view_as_complex(
                torch.zeros((
                    self.num_k, num_atoms, num_atoms, self.nao_max, self.nao_max, 2
                )).type_as(onsite_hamiltonian)
            )
            reference_overlap_k = torch.view_as_complex(
                torch.zeros((
                    self.num_k, num_atoms, num_atoms, self.nao_max, self.nao_max, 2
                )).type_as(onsite_hamiltonian)
            )
            predicted_overlap_k = torch.view_as_complex(
                torch.zeros((
                    self.num_k, num_atoms, num_atoms, self.nao_max, self.nao_max, 2
                )).type_as(onsite_hamiltonian)
            )

            if export_reciprocal_values:
                overlap_derivatives_k = torch.view_as_complex(
                    torch.zeros((
                        self.num_k, num_atoms, num_atoms, self.nao_max, self.nao_max, 3, 2
                    )).type_as(onsite_hamiltonian)
                )

            # Add on-site terms to k-space matrices
            atom_indices = torch.arange(num_atoms).type_as(source_indices)

            # Add on-site Hamiltonian and overlap terms
            hamiltonian_k[:, atom_indices, atom_indices, :, :] += onsite_hamiltonians_by_crystal[crystal_idx].reshape(
                -1, self.nao_max, self.nao_max
            )[None, atom_indices, :, :].type_as(hamiltonian_k)

            reference_overlap_k[:, atom_indices, atom_indices, :, :] += reference_onsite_overlaps[crystal_idx].reshape(
                -1, self.nao_max, self.nao_max
            )[None, atom_indices, :, :].type_as(reference_overlap_k)

            predicted_overlap_k[:, atom_indices, atom_indices, :, :] += predicted_onsite_overlaps[crystal_idx].reshape(
                -1, self.nao_max, self.nao_max
            )[None, atom_indices, :, :].type_as(predicted_overlap_k)

            if export_reciprocal_values:
                overlap_derivatives_k[:, atom_indices, atom_indices, :, :, :] += onsite_overlap_derivatives[crystal_idx].reshape(
                    -1, self.nao_max, self.nao_max, 3
                )[None, atom_indices, :, :, :].type_as(overlap_derivatives_k)

            # Add off-site terms to k-space matrices
            for edge_idx in range(edges_per_crystal[crystal_idx]):
                # Get local indices within this crystal
                source_atom_idx = source_indices[edge_index_offsets[crystal_idx] + edge_idx] - atom_index_offsets[crystal_idx]
                target_atom_idx = target_indices[edge_index_offsets[crystal_idx] + edge_idx] - atom_index_offsets[crystal_idx]

                # Add contribution for each k-point with phase factor
                hamiltonian_k[:, source_atom_idx, target_atom_idx, :, :] += (
                    phase_factors[edge_idx, :, None, None] * 
                    offsite_hamiltonians_by_crystal[crystal_idx].reshape(-1, self.nao_max, self.nao_max)[None, edge_idx, :, :]
                )

                reference_overlap_k[:, source_atom_idx, target_atom_idx, :, :] += (
                    phase_factors[edge_idx, :, None, None] * 
                    reference_offsite_overlaps[crystal_idx].reshape(-1, self.nao_max, self.nao_max)[None, edge_idx, :, :]
                )

                predicted_overlap_k[:, source_atom_idx, target_atom_idx, :, :] += (
                    phase_factors[edge_idx, :, None, None] * 
                    predicted_offsite_overlaps[crystal_idx].reshape(-1, self.nao_max, self.nao_max)[None, edge_idx, :, :]
                )

            # Add derivative terms if needed
            if export_reciprocal_values:
                for edge_idx in range(edges_per_crystal[crystal_idx]):
                    source_atom_idx = source_indices[edge_index_offsets[crystal_idx] + edge_idx] - atom_index_offsets[crystal_idx]
                    target_atom_idx = target_indices[edge_index_offsets[crystal_idx] + edge_idx] - atom_index_offsets[crystal_idx]

                    overlap_derivatives_k[:, source_atom_idx, target_atom_idx, :, :, :] += (
                        phase_factors[edge_idx, :, None, None, None] * 
                        offsite_overlap_derivatives[crystal_idx].reshape(-1, self.nao_max, self.nao_max, 3)[None, edge_idx, :, :, :]
                    )

            # Reshape matrices to combine atom and orbital dimensions
            hamiltonian_k = torch.swapaxes(hamiltonian_k, -2, -3)  # Swap atom and orbital indices
            hamiltonian_k = hamiltonian_k.reshape(-1, num_atoms * self.nao_max, num_atoms * self.nao_max)

            reference_overlap_k = torch.swapaxes(reference_overlap_k, -2, -3)
            reference_overlap_k = reference_overlap_k.reshape(-1, num_atoms * self.nao_max, num_atoms * self.nao_max)

            predicted_overlap_k = torch.swapaxes(predicted_overlap_k, -2, -3)
            predicted_overlap_k = predicted_overlap_k.reshape(-1, num_atoms * self.nao_max, num_atoms * self.nao_max)

            if export_reciprocal_values:
                overlap_derivatives_k = torch.swapaxes(overlap_derivatives_k, -3, -4)
                overlap_derivatives_k = overlap_derivatives_k.reshape(-1, num_atoms * self.nao_max, num_atoms * self.nao_max, 3)

            # Apply orbital mask to select only valid orbitals
            mask = orbital_interaction_masks[crystal_idx].repeat(self.num_k, 1, 1) > 0

            hamiltonian_k = torch.masked_select(hamiltonian_k, mask)
            num_orbitals = int(math.sqrt(hamiltonian_k.numel() / self.num_k))
            hamiltonian_k = hamiltonian_k.reshape(self.num_k, num_orbitals, num_orbitals)

            reference_overlap_k = torch.masked_select(reference_overlap_k, mask)
            reference_overlap_k = reference_overlap_k.reshape(self.num_k, num_orbitals, num_orbitals)

            predicted_overlap_k = torch.masked_select(predicted_overlap_k, mask)
            predicted_overlap_k = predicted_overlap_k.reshape(self.num_k, num_orbitals, num_orbitals)

            if export_reciprocal_values:
                derivative_mask = orbital_interaction_masks[crystal_idx].unsqueeze(-1).repeat(self.num_k, 1, 1, 3) > 0
                overlap_derivatives_k = torch.masked_select(overlap_derivatives_k, derivative_mask)
                overlap_derivatives_k = overlap_derivatives_k.reshape(self.num_k, num_orbitals, num_orbitals, 3)

            # Solve generalized eigenvalue problem using Cholesky decomposition
            cholesky_factor = torch.linalg.cholesky(reference_overlap_k)
            cholesky_factor_conj_transpose = torch.transpose(cholesky_factor.conj(), dim0=-1, dim1=-2)
            cholesky_inverse = torch.linalg.inv(cholesky_factor)
            cholesky_conj_transpose_inverse = torch.linalg.inv(cholesky_factor_conj_transpose)

            # Transform to standard eigenvalue problem
            transformed_hamiltonian = torch.bmm(
                torch.bmm(cholesky_inverse, hamiltonian_k),
                cholesky_conj_transpose_inverse
            )

            # Solve eigenvalue problem
            eigenvalues, eigenvectors = torch.linalg.eigh(transformed_hamiltonian)

            # Transform eigenvectors back to original basis
            eigenvectors = torch.einsum('ijk,ika->iaj', cholesky_conj_transpose_inverse, eigenvectors)

            # Normalize wavefunctions if exporting reciprocal values
            if export_reciprocal_values:
                # Calculate normalization factors
                normalization_factors = torch.einsum(
                    'nai,nij,naj->na',
                    torch.conj(eigenvectors),
                    reference_overlap_k,
                    eigenvectors
                ).real

                # Apply normalization
                normalization_factors = 1 / torch.sqrt(normalization_factors)
                eigenvectors = eigenvectors * normalization_factors.unsqueeze(-1)

                # Store reciprocal space matrices
                reciprocal_hamiltonians.append(hamiltonian_k)
                reciprocal_overlaps.append(predicted_overlap_k)
                reciprocal_overlap_derivatives.append(overlap_derivatives_k)

            # Apply band number control if specified
            if self.band_num_control is not None:
                eigenvalues = eigenvalues[:, :crystal_band_counts[crystal_idx]]
                eigenvectors = eigenvectors[:, :crystal_band_counts[crystal_idx], :]

            # Store results
            band_energies.append(torch.transpose(eigenvalues, dim0=-1, dim1=-2))
            wavefunctions.append(eigenvectors)
            symmetrized_hamiltonians.append(transformed_hamiltonian.view(-1))

            # Calculate band gap
            half_filled_band_index = math.ceil(crystal_valence_counts[crystal_idx] / 2)
            band_gap = (
                torch.min(eigenvalues[:, half_filled_band_index]) - 
                torch.max(eigenvalues[:, half_filled_band_index - 1])
            ).unsqueeze(0)
            band_gaps.append(band_gap)

        # Combine results across all crystals
        band_energies = torch.cat(band_energies, dim=0)
        band_gaps = torch.cat(band_gaps, dim=0)

        if export_reciprocal_values:
            # Return reciprocal space matrices
            wavefunctions = torch.stack(wavefunctions, dim=0)
            hamiltonians_k_space = torch.stack(reciprocal_hamiltonians, dim=0)
            overlaps_k_space = torch.stack(reciprocal_overlaps, dim=0)
            overlap_derivatives_k_space = torch.stack(reciprocal_overlap_derivatives, dim=0)
            return band_energies, wavefunctions, hamiltonians_k_space, overlaps_k_space, overlap_derivatives_k_space, band_gaps
        else:
            # Flatten wavefunctions and return
            wavefunctions = [wf.reshape(-1) for wf in wavefunctions]
            wavefunctions = torch.cat(wavefunctions, dim=0)
            symmetrized_hamiltonians = torch.cat(symmetrized_hamiltonians, dim=0)
            return band_energies, wavefunctions, band_gaps, symmetrized_hamiltonians

    def calculate_band_energies(self, onsite_hamiltonian, offsite_hamiltonian, crystal_data, 
                                export_reciprocal_values=False):
        """
        Calculate electronic band structure using Hamiltonian matrices and reference overlap matrices.

        This function computes electronic band energies, wavefunctions, and band gaps for a set 
        of crystal structures by solving the generalized eigenvalue problem H���� = E��S����.

        Args:
            onsite_hamiltonian (torch.Tensor): On-site Hamiltonian matrix elements with shape 
                (total_atoms, nao_max2).
            offsite_hamiltonian (torch.Tensor): Off-site Hamiltonian matrix elements with shape 
                (total_edges, nao_max2).
            crystal_data (DataObject): Object containing crystal structure information including:
                - edge_index: Indices of connected atom pairs
                - cell: Unit cell vectors
                - z: Atomic numbers
                - node_counts: Number of atoms in each crystal
                - batch: Batch assignment for each atom
                - k_vecs: k-points for band structure calculation
                - nbr_shift: Neighbor cell shifts for periodic boundary conditions
                - Son/Soff: Reference overlap matrices
            export_reciprocal_values (bool, optional): Whether to export additional reciprocal 
                space matrices (H(k), S(k), dS(k)). Defaults to False.

        Returns:
            tuple: Contains band energies, wavefunctions, band gaps, and optionally additional 
                   reciprocal space matrices depending on the export_reciprocal_values parameter.
        """
        source_indices, target_indices = crystal_data.edge_index
        lattice_vectors = crystal_data.cell  # shape: (batch_size, 3, 3)
        batch_size = lattice_vectors.shape[0]

        # Create orbital validity mask based on atomic numbers
        atomic_orbital_mask = torch.zeros((99, self.nao_max)).type_as(crystal_data.z)
        for atomic_number, orbital_indices in self.basis_def.items():
            atomic_orbital_mask[atomic_number][orbital_indices] = 1

        # Get orbital mask for each atom
        orbital_mask = atomic_orbital_mask[crystal_data.z]  # shape: [total_atoms, nao_max]
        orbital_mask_by_crystal = torch.split(orbital_mask, crystal_data.node_counts.tolist(), dim=0)

        # Create interaction masks for each crystal
        orbital_interaction_masks = []
        for idx in range(batch_size):
            # Create outer product of orbital masks for interaction masking
            orbital_interaction_masks.append(
                orbital_mask_by_crystal[idx].reshape(-1, 1) * 
                orbital_mask_by_crystal[idx].reshape(1, -1)
            )

        # Set the number of valence electrons per atom type
        valence_electron_counts = torch.zeros((99,)).type_as(crystal_data.z)
        for atomic_number, count in self.num_valence.items():
            valence_electron_counts[atomic_number] = count

        # Get valence electron count for each atom and then for each crystal
        atom_valence_counts = valence_electron_counts[crystal_data.z]  # shape: [total_atoms]
        crystal_valence_counts = scatter(atom_valence_counts, crystal_data.batch, dim=0)  # shape: [batch_size]

        # Initialize band window sizes if needed
        if isinstance(self.band_num_control, dict):
            bands_per_atom_type = torch.zeros((99,)).type_as(crystal_data.z)
            for atomic_number, band_count in self.band_num_control.items():
                bands_per_atom_type[atomic_number] = band_count
            atom_band_counts = bands_per_atom_type[crystal_data.z]  # shape: [total_atoms]
            crystal_band_counts = scatter(atom_band_counts, crystal_data.batch, dim=0)  # shape: [batch_size]

        # Split data by crystal
        atoms_per_crystal = crystal_data.node_counts
        atom_index_offsets = torch.cumsum(atoms_per_crystal, dim=0) - atoms_per_crystal

        # Split Hamiltonians and overlaps by crystal
        onsite_hamiltonians_by_crystal = torch.split(onsite_hamiltonian, atoms_per_crystal.tolist(), dim=0)
        onsite_overlaps_by_crystal = torch.split(crystal_data.Son, atoms_per_crystal.tolist(), dim=0)

        # Count edges per crystal and split edge data
        edge_counter = torch.ones_like(source_indices)
        edges_per_crystal = scatter(edge_counter, crystal_data.batch[source_indices], dim=0)
        edge_index_offsets = torch.cumsum(edges_per_crystal, dim=0) - edges_per_crystal

        offsite_hamiltonians_by_crystal = torch.split(offsite_hamiltonian, edges_per_crystal.tolist(), dim=0)
        offsite_overlaps_by_crystal = torch.split(crystal_data.Soff, edges_per_crystal.tolist(), dim=0)

        # Prepare derivatives if exporting reciprocal values
        if export_reciprocal_values:
            onsite_overlap_derivatives = torch.split(crystal_data.dSon, atoms_per_crystal.tolist(), dim=0)
            offsite_overlap_derivatives = torch.split(crystal_data.dSoff, edges_per_crystal.tolist(), dim=0)

        # Initialize results containers
        band_energies = []
        wavefunctions = []
        reciprocal_hamiltonians = []
        symmetrized_hamiltonians = []
        reciprocal_overlaps = []
        reciprocal_overlap_derivatives = []
        band_gaps = []

        # Process each crystal
        for crystal_idx in range(batch_size):
            k_points = crystal_data.k_vecs[crystal_idx]
            num_atoms = atoms_per_crystal[crystal_idx]

            # Calculate phase factors for k-points
            # Shape: (edges_in_crystal, num_k_points)
            phase_factors = torch.exp(
                2j * torch.pi * torch.sum(
                    crystal_data.nbr_shift[
                        edge_index_offsets[crystal_idx] + 
                        torch.arange(edges_per_crystal[crystal_idx]).type_as(source_indices),
                        None, :
                    ] * k_points[None, :, :],
                    dim=-1
                )
            )

            # Initialize k-space Hamiltonian and overlap matrices
            hamiltonian_k = torch.view_as_complex(
                torch.zeros((
                    self.num_k, num_atoms, num_atoms, self.nao_max, self.nao_max, 2
                )).type_as(onsite_hamiltonian)
            )
            overlap_k = torch.view_as_complex(
                torch.zeros((
                    self.num_k, num_atoms, num_atoms, self.nao_max, self.nao_max, 2
                )).type_as(onsite_hamiltonian)
            )

            if export_reciprocal_values:
                overlap_derivatives_k = torch.view_as_complex(
                    torch.zeros((
                        self.num_k, num_atoms, num_atoms, self.nao_max, self.nao_max, 3, 2
                    )).type_as(onsite_hamiltonian)
                )

            # Add on-site terms to k-space matrices
            atom_indices = torch.arange(num_atoms).type_as(source_indices)

            # Add on-site Hamiltonian and overlap terms
            hamiltonian_k[:, atom_indices, atom_indices, :, :] += onsite_hamiltonians_by_crystal[crystal_idx].reshape(
                -1, self.nao_max, self.nao_max
            )[None, atom_indices, :, :].type_as(hamiltonian_k)

            overlap_k[:, atom_indices, atom_indices, :, :] += onsite_overlaps_by_crystal[crystal_idx].reshape(
                -1, self.nao_max, self.nao_max
            )[None, atom_indices, :, :].type_as(overlap_k)

            if export_reciprocal_values:
                overlap_derivatives_k[:, atom_indices, atom_indices, :, :, :] += onsite_overlap_derivatives[crystal_idx].reshape(
                    -1, self.nao_max, self.nao_max, 3
                )[None, atom_indices, :, :, :].type_as(overlap_derivatives_k)

            # Prepare edge data for vectorized operations
            edge_indices = torch.arange(edges_per_crystal[crystal_idx], device=source_indices.device)
            source_atom_indices = source_indices[edge_index_offsets[crystal_idx] + edge_indices] - atom_index_offsets[crystal_idx]
            target_atom_indices = target_indices[edge_index_offsets[crystal_idx] + edge_indices] - atom_index_offsets[crystal_idx]

            # Reshape matrices for more efficient operations
            offsite_h_reshaped = offsite_hamiltonians_by_crystal[crystal_idx].reshape(
                edges_per_crystal[crystal_idx], self.nao_max, self.nao_max
            )
            offsite_s_reshaped = offsite_overlaps_by_crystal[crystal_idx].reshape(
                edges_per_crystal[crystal_idx], self.nao_max, self.nao_max
            )

            # Add off-site terms to k-space matrices for each k-point
            for k_idx in range(self.num_k):
                # Pre-calculate phase factors for this k-point
                k_phase_factors = phase_factors[:edges_per_crystal[crystal_idx], k_idx].unsqueeze(-1).unsqueeze(-1)

                # Calculate contribution to Hamiltonian
                h_contributions = k_phase_factors * offsite_h_reshaped
                s_contributions = k_phase_factors * offsite_s_reshaped

                # Add contributions to k-space matrices using index_put
                hamiltonian_k[k_idx] = torch.index_put(
                    hamiltonian_k[k_idx], 
                    (source_atom_indices, target_atom_indices), 
                    h_contributions, 
                    accumulate=True
                )

                overlap_k[k_idx] = torch.index_put(
                    overlap_k[k_idx], 
                    (source_atom_indices, target_atom_indices), 
                    s_contributions, 
                    accumulate=True
                )

            # Add derivative terms if needed
            if export_reciprocal_values:
                offsite_ds_reshaped = offsite_overlap_derivatives[crystal_idx].reshape(
                    edges_per_crystal[crystal_idx], self.nao_max, self.nao_max, 3
                )

                for k_idx in range(self.num_k):
                    # Pre-calculate phase factors for this k-point
                    k_phase_factors = phase_factors[:edges_per_crystal[crystal_idx], k_idx].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

                    # Calculate contribution to derivatives
                    ds_contributions = k_phase_factors * offsite_ds_reshaped

                    # Add contributions to k-space matrices using index_put
                    overlap_derivatives_k[k_idx] = torch.index_put(
                        overlap_derivatives_k[k_idx], 
                        (source_atom_indices, target_atom_indices), 
                        ds_contributions, 
                        accumulate=True
                    )

            # Reshape matrices to combine atom and orbital dimensions
            hamiltonian_k = torch.swapaxes(hamiltonian_k, -2, -3)  # Swap atom and orbital indices
            hamiltonian_k = hamiltonian_k.reshape(-1, num_atoms * self.nao_max, num_atoms * self.nao_max)

            overlap_k = torch.swapaxes(overlap_k, -2, -3)
            overlap_k = overlap_k.reshape(-1, num_atoms * self.nao_max, num_atoms * self.nao_max)

            if export_reciprocal_values:
                overlap_derivatives_k = torch.swapaxes(overlap_derivatives_k, -3, -4)
                overlap_derivatives_k = overlap_derivatives_k.reshape(-1, num_atoms * self.nao_max, num_atoms * self.nao_max, 3)

            # Apply orbital mask to select only valid orbitals
            mask = orbital_interaction_masks[crystal_idx].repeat(self.num_k, 1, 1) > 0

            hamiltonian_k = torch.masked_select(hamiltonian_k, mask)
            num_orbitals = int(math.sqrt(hamiltonian_k.numel() / self.num_k))
            hamiltonian_k = hamiltonian_k.reshape(self.num_k, num_orbitals, num_orbitals)

            overlap_k = torch.masked_select(overlap_k, mask)
            overlap_k = overlap_k.reshape(self.num_k, num_orbitals, num_orbitals)

            if export_reciprocal_values:
                derivative_mask = orbital_interaction_masks[crystal_idx].unsqueeze(-1).repeat(self.num_k, 1, 1, 3) > 0
                overlap_derivatives_k = torch.masked_select(overlap_derivatives_k, derivative_mask)
                overlap_derivatives_k = overlap_derivatives_k.reshape(self.num_k, num_orbitals, num_orbitals, 3)

            # Solve generalized eigenvalue problem using Cholesky decomposition
            cholesky_factor = torch.linalg.cholesky(overlap_k)
            cholesky_factor_conj_transpose = torch.transpose(cholesky_factor.conj(), dim0=-1, dim1=-2)
            cholesky_inverse = torch.linalg.inv(cholesky_factor)
            cholesky_conj_transpose_inverse = torch.linalg.inv(cholesky_factor_conj_transpose)

            # Transform to standard eigenvalue problem
            transformed_hamiltonian = torch.bmm(
                torch.bmm(cholesky_inverse, hamiltonian_k),
                cholesky_conj_transpose_inverse
            )

            # Solve eigenvalue problem
            eigenvalues, eigenvectors = torch.linalg.eigh(transformed_hamiltonian)

            # Transform eigenvectors back to original basis
            eigenvectors = torch.einsum('ijk,ika->iaj', cholesky_conj_transpose_inverse, eigenvectors)

            # Calculate band gap
            half_filled_band_index = math.ceil(crystal_valence_counts[crystal_idx] / 2)
            band_gap = (
                torch.min(eigenvalues[:, half_filled_band_index]) - 
                torch.max(eigenvalues[:, half_filled_band_index - 1])
            ).unsqueeze(0)
            band_gaps.append(band_gap)

            # Apply band number control if specified
            if self.band_num_control is not None:
                if isinstance(self.band_num_control, dict):
                    # Use pre-calculated band counts
                    eigenvalues = eigenvalues[:, :crystal_band_counts[crystal_idx]]
                    eigenvectors = eigenvectors[:, :crystal_band_counts[crystal_idx], :]
                else:
                    # Calculate band window dynamically
                    if isinstance(self.band_num_control, float):
                        band_window = max(1, int(self.band_num_control * half_filled_band_index))
                    else:
                        band_window = min(self.band_num_control, half_filled_band_index)

                    # Select band window centered around half-filled band
                    start_band = half_filled_band_index - band_window
                    end_band = half_filled_band_index + band_window
                    eigenvalues = eigenvalues[:, start_band:end_band]
                    eigenvectors = eigenvectors[:, start_band:end_band, :]

            # Normalize wavefunctions if exporting reciprocal values
            if export_reciprocal_values:
                # Calculate normalization factors
                normalization_factors = torch.einsum(
                    'nai,nij,naj->na',
                    torch.conj(eigenvectors),
                    overlap_k,
                    eigenvectors
                ).real

                # Apply normalization
                normalization_factors = 1 / torch.sqrt(normalization_factors)
                eigenvectors = eigenvectors * normalization_factors.unsqueeze(-1)

                # Store reciprocal space matrices
                reciprocal_hamiltonians.append(hamiltonian_k)
                reciprocal_overlaps.append(overlap_k)
                reciprocal_overlap_derivatives.append(overlap_derivatives_k)

            # Store results
            band_energies.append(torch.transpose(eigenvalues, dim0=-1, dim1=-2))
            wavefunctions.append(eigenvectors)
            symmetrized_hamiltonians.append(transformed_hamiltonian.view(-1))

        # Combine results across all crystals
        band_energies = torch.cat(band_energies, dim=0)
        band_gaps = torch.cat(band_gaps, dim=0)

        if export_reciprocal_values:
            # Return reciprocal space matrices
            wavefunctions = torch.stack(wavefunctions, dim=0)
            hamiltonians_k_space = torch.stack(reciprocal_hamiltonians, dim=0)
            overlaps_k_space = torch.stack(reciprocal_overlaps, dim=0)
            overlap_derivatives_k_space = torch.stack(reciprocal_overlap_derivatives, dim=0)
            return band_energies, wavefunctions, hamiltonians_k_space, overlaps_k_space, overlap_derivatives_k_space, band_gaps
        else:
            # Flatten wavefunctions and return
            wavefunctions = [wf.reshape(-1) for wf in wavefunctions]
            wavefunctions = torch.cat(wavefunctions, dim=0)
            symmetrized_hamiltonians = torch.cat(symmetrized_hamiltonians, dim=0)
            return band_energies, wavefunctions, band_gaps, symmetrized_hamiltonians

    def calculate_band_energies_with_spin_orbit_coupling(self, real_onsite, imag_onsite, real_offsite, imag_offsite, crystal_data):
        """
        Calculate electronic band structure with spin-orbit coupling (SOC).

        This function computes electronic band energies and wavefunctions for systems with 
        spin-orbit coupling, which requires handling complex Hamiltonian matrices and doubling 
        the matrix dimensions to account for spin.

        Args:
            real_onsite (torch.Tensor): Real part of on-site SOC Hamiltonian with shape 
                (total_atoms, 2*nao_max, 2*nao_max).
            imag_onsite (torch.Tensor): Imaginary part of on-site SOC Hamiltonian with shape 
                (total_atoms, 2*nao_max, 2*nao_max).
            real_offsite (torch.Tensor): Real part of off-site SOC Hamiltonian with shape 
                (total_edges, 2*nao_max, 2*nao_max).
            imag_offsite (torch.Tensor): Imaginary part of off-site SOC Hamiltonian with shape 
                (total_edges, 2*nao_max, 2*nao_max).
            crystal_data (DataObject): Object containing crystal structure information including:
                - edge_index: Indices of connected atom pairs
                - cell: Unit cell vectors
                - z: Atomic numbers
                - node_counts: Number of atoms in each crystal
                - batch: Batch assignment for each atom
                - k_vecs: k-points for band structure calculation
                - cell_shift: Cell shift vectors for periodic images
                - nbr_shift: Neighbor cell shifts for periodic boundary conditions
                - Son/Soff: Reference overlap matrices (without SOC)

        Returns:
            tuple: (
                band_energies (torch.Tensor): Band energies with shape (total_bands, num_k_points),
                wavefunctions (torch.Tensor): Flattened wavefunctions
            )
        """
        source_indices, target_indices = crystal_data.edge_index
        lattice_vectors = crystal_data.cell  # shape: (batch_size, 3, 3)
        batch_size = lattice_vectors.shape[0]

        # Reshape Hamiltonian components to matrix form
        real_onsite = real_onsite.reshape(-1, 2*self.nao_max, 2*self.nao_max)
        imag_onsite = imag_onsite.reshape(-1, 2*self.nao_max, 2*self.nao_max)
        real_offsite = real_offsite.reshape(-1, 2*self.nao_max, 2*self.nao_max)
        imag_offsite = imag_offsite.reshape(-1, 2*self.nao_max, 2*self.nao_max)

        # Create orbital validity mask based on atomic numbers
        atomic_orbital_mask = torch.zeros((99, self.nao_max)).type_as(crystal_data.z)
        for atomic_number, orbital_indices in self.basis_def.items():
            atomic_orbital_mask[atomic_number][orbital_indices] = 1

        # Get orbital mask for each atom
        orbital_mask = atomic_orbital_mask[crystal_data.z]  # shape: [total_atoms, nao_max]
        orbital_mask_by_crystal = torch.split(orbital_mask, crystal_data.node_counts.tolist(), dim=0)

        # Create interaction masks for each crystal
        orbital_interaction_masks = []
        for idx in range(batch_size):
            # Create outer product of orbital masks for interaction masking
            orbital_interaction_masks.append(
                orbital_mask_by_crystal[idx].reshape(-1, 1) * 
                orbital_mask_by_crystal[idx].reshape(1, -1)
            )

        # Set the number of valence electrons per atom type
        valence_electron_counts = torch.zeros((99,)).type_as(crystal_data.z)
        for atomic_number, count in self.num_valence.items():
            valence_electron_counts[atomic_number] = count

        # Get valence electron count for each atom and then for each crystal
        atom_valence_counts = valence_electron_counts[crystal_data.z]  # shape: [total_atoms]
        crystal_valence_counts = scatter(atom_valence_counts, crystal_data.batch, dim=0)  # shape: [batch_size]

        # Initialize band window sizes if needed
        if isinstance(self.band_num_control, dict):
            bands_per_atom_type = torch.zeros((99,)).type_as(crystal_data.z)
            for atomic_number, band_count in self.band_num_control.items():
                bands_per_atom_type[atomic_number] = band_count
            atom_band_counts = bands_per_atom_type[crystal_data.z]  # shape: [total_atoms]
            crystal_band_counts = scatter(atom_band_counts, crystal_data.batch, dim=0)  # shape: [batch_size]

        # Split data by crystal
        atoms_per_crystal = crystal_data.node_counts

        # Split Hamiltonians by crystal
        real_onsite_by_crystal = torch.split(real_onsite, atoms_per_crystal.tolist(), dim=0)
        imag_onsite_by_crystal = torch.split(imag_onsite, atoms_per_crystal.tolist(), dim=0)
        onsite_overlaps_by_crystal = torch.split(
            crystal_data.Son.reshape(-1, self.nao_max, self.nao_max), 
            atoms_per_crystal.tolist(), 
            dim=0
        )

        # Count edges per crystal and split edge data
        edge_counter = torch.ones_like(source_indices)
        edges_per_crystal = scatter(edge_counter, crystal_data.batch[source_indices], dim=0)

        real_offsite_by_crystal = torch.split(real_offsite, edges_per_crystal.tolist(), dim=0)
        imag_offsite_by_crystal = torch.split(imag_offsite, edges_per_crystal.tolist(), dim=0)
        offsite_overlaps_by_crystal = torch.split(
            crystal_data.Soff.reshape(-1, self.nao_max, self.nao_max), 
            edges_per_crystal.tolist(), 
            dim=0
        )

        # Split cell shifts and neighbor shifts by crystal
        cell_shifts_by_crystal = torch.split(crystal_data.cell_shift, edges_per_crystal.tolist(), dim=0)
        nbr_shifts_by_crystal = torch.split(crystal_data.nbr_shift, edges_per_crystal.tolist(), dim=0)

        # Split edge indices by crystal and convert to local indices
        edge_indices_by_crystal = torch.split(crystal_data.edge_index, edges_per_crystal.tolist(), dim=1)
        atom_index_offsets = torch.cumsum(atoms_per_crystal, dim=0) - atoms_per_crystal

        # Convert global edge indices to local indices within each crystal
        local_edge_indices = []
        for idx, edge_indices in enumerate(edge_indices_by_crystal):
            local_edge_indices.append(edge_indices - atom_index_offsets[idx])

        # Initialize results containers
        band_energies = []
        wavefunctions = []

        # Process each crystal
        for crystal_idx in range(batch_size):
            k_points = crystal_data.k_vecs[crystal_idx]
            num_atoms = atoms_per_crystal[crystal_idx].item()

            # Create unique cell shift mapping for this crystal
            cell_shift_tuples = [tuple(shift.cpu().tolist()) for shift in cell_shifts_by_crystal[crystal_idx]]
            unique_cell_shifts = list(set(cell_shift_tuples))
            cell_to_index = {shift: idx for idx, shift in enumerate(unique_cell_shifts)}
            cell_indices = torch.tensor([cell_to_index[shift] for shift in cell_shift_tuples], device=source_indices.device).type_as(source_indices)
            num_cells = len(unique_cell_shifts)

            # Calculate phase factors for k-points
            # Shape: (num_k_points, num_cells)
            phase_factors = torch.view_as_complex(
                torch.zeros((self.num_k, num_cells, 2)).type_as(crystal_data.Son)
            )

            # Set phase factors for each cell shift
            phase_factors[:, cell_indices] = torch.exp(
                2j * torch.pi * torch.sum(
                    nbr_shifts_by_crystal[crystal_idx][None, :, :] * k_points[:, None, :],
                    dim=-1
                )
            )

            # Create atom index tensor
            atom_indices = torch.arange(num_atoms, device=source_indices.device).type_as(source_indices)

            # Initialize overlap matrix in k-space
            s_by_cell = torch.view_as_complex(
                torch.zeros((
                    num_cells, num_atoms, num_atoms, self.nao_max, self.nao_max, 2
                )).type_as(crystal_data.Son)
            )

            # Set off-site overlap elements
            source_atoms, target_atoms = local_edge_indices[crystal_idx]
            s_by_cell[cell_indices, source_atoms, target_atoms, :, :] += offsite_overlaps_by_crystal[crystal_idx]

            # Calculate k-space overlap matrix
            overlap_k = torch.einsum('ijklm,ni->njklm', s_by_cell, phase_factors)

            # Add on-site overlap elements
            overlap_k[:, atom_indices, atom_indices, :, :] += onsite_overlaps_by_crystal[crystal_idx][None, atom_indices, :, :]

            # Reshape and swap dimensions for consistent ordering
            overlap_k = torch.swapaxes(overlap_k, 2, 3)  # Swap atom and orbital indices
            overlap_k = overlap_k.reshape(self.num_k, num_atoms * self.nao_max, num_atoms * self.nao_max)

            # Apply orbital mask to select only valid orbitals
            overlap_k = overlap_k[:, orbital_interaction_masks[crystal_idx] > 0]
            num_orbitals = int(math.sqrt(overlap_k.numel() / self.num_k))
            overlap_k = overlap_k.reshape(self.num_k, num_orbitals, num_orbitals)

            # Create identity matrix for spin expansion
            identity = torch.eye(2, device=crystal_data.Hon.device).type_as(crystal_data.Hon)

            # Expand overlap matrix to account for spin
            overlap_k_soc = torch.kron(identity, overlap_k)

            # Process SOC Hamiltonian components
            # Extract spin blocks from on-site Hamiltonian
            h_up_up = real_onsite_by_crystal[crystal_idx][:, :self.nao_max, :self.nao_max] + \
                     1.0j * imag_onsite_by_crystal[crystal_idx][:, :self.nao_max, :self.nao_max]

            h_up_down = real_onsite_by_crystal[crystal_idx][:, :self.nao_max, self.nao_max:] + \
                       1.0j * imag_onsite_by_crystal[crystal_idx][:, :self.nao_max, self.nao_max:]

            h_down_up = real_onsite_by_crystal[crystal_idx][:, self.nao_max:, :self.nao_max] + \
                       1.0j * imag_onsite_by_crystal[crystal_idx][:, self.nao_max:, :self.nao_max]

            h_down_down = real_onsite_by_crystal[crystal_idx][:, self.nao_max:, self.nao_max:] + \
                         1.0j * imag_onsite_by_crystal[crystal_idx][:, self.nao_max:, self.nao_max:]

            onsite_soc_blocks = [h_up_up, h_up_down, h_down_up, h_down_down]

            # Extract spin blocks from off-site Hamiltonian
            h_up_up = real_offsite_by_crystal[crystal_idx][:, :self.nao_max, :self.nao_max] + \
                     1.0j * imag_offsite_by_crystal[crystal_idx][:, :self.nao_max, :self.nao_max]

            h_up_down = real_offsite_by_crystal[crystal_idx][:, :self.nao_max, self.nao_max:] + \
                       1.0j * imag_offsite_by_crystal[crystal_idx][:, :self.nao_max, self.nao_max:]

            h_down_up = real_offsite_by_crystal[crystal_idx][:, self.nao_max:, :self.nao_max] + \
                       1.0j * imag_offsite_by_crystal[crystal_idx][:, self.nao_max:, :self.nao_max]

            h_down_down = real_offsite_by_crystal[crystal_idx][:, self.nao_max:, self.nao_max:] + \
                         1.0j * imag_offsite_by_crystal[crystal_idx][:, self.nao_max:, self.nao_max:]

            offsite_soc_blocks = [h_up_up, h_up_down, h_down_up, h_down_down]

            # Initialize k-space Hamiltonian blocks
            hamiltonian_k_blocks = []

            # Process each spin block
            for onsite_block, offsite_block in zip(onsite_soc_blocks, offsite_soc_blocks):
                # Initialize Hamiltonian matrix in k-space for this spin block
                h_by_cell = torch.view_as_complex(
                    torch.zeros((
                        num_cells, num_atoms, num_atoms, self.nao_max, self.nao_max, 2
                    )).type_as(crystal_data.Son)
                )

                # Set off-site Hamiltonian elements
                h_by_cell[cell_indices, source_atoms, target_atoms, :, :] += offsite_block

                # Calculate k-space Hamiltonian matrix
                hamiltonian_k_block = torch.einsum('ijklm,ni->njklm', h_by_cell, phase_factors)

                # Add on-site Hamiltonian elements
                hamiltonian_k_block[:, atom_indices, atom_indices, :, :] += onsite_block[None, atom_indices, :, :]

                # Reshape and swap dimensions for consistent ordering
                hamiltonian_k_block = torch.swapaxes(hamiltonian_k_block, 2, 3)
                hamiltonian_k_block = hamiltonian_k_block.reshape(self.num_k, num_atoms * self.nao_max, num_atoms * self.nao_max)

                # Apply orbital mask to select only valid orbitals
                hamiltonian_k_block = hamiltonian_k_block[:, orbital_interaction_masks[crystal_idx] > 0]
                hamiltonian_k_block = hamiltonian_k_block.reshape(self.num_k, num_orbitals, num_orbitals)

                hamiltonian_k_blocks.append(hamiltonian_k_block)

            # Combine spin blocks into full SOC Hamiltonian
            hamiltonian_k_soc = torch.cat([
                torch.cat([hamiltonian_k_blocks[0], hamiltonian_k_blocks[1]], dim=-1),
                torch.cat([hamiltonian_k_blocks[2], hamiltonian_k_blocks[3]], dim=-1)
            ], dim=-2)

            # Solve generalized eigenvalue problem using Cholesky decomposition
            cholesky_factor = torch.linalg.cholesky(overlap_k_soc)
            cholesky_factor_conj_transpose = torch.transpose(cholesky_factor.conj(), dim0=-1, dim1=-2)
            cholesky_inverse = torch.linalg.inv(cholesky_factor)
            cholesky_conj_transpose_inverse = torch.linalg.inv(cholesky_factor_conj_transpose)

            # Transform to standard eigenvalue problem
            transformed_hamiltonian = torch.bmm(
                torch.bmm(cholesky_inverse, hamiltonian_k_soc),
                cholesky_conj_transpose_inverse
            )

            # Solve eigenvalue problem
            eigenvalues, eigenvectors = torch.linalg.eigh(transformed_hamiltonian)

            # Transform eigenvectors back to original basis
            eigenvectors = torch.bmm(cholesky_conj_transpose_inverse, eigenvectors)

            # Apply band number control if specified
            if self.band_num_control is not None:
                if isinstance(self.band_num_control, dict):
                    # Use pre-calculated band counts
                    eigenvalues = eigenvalues[:, :crystal_band_counts[crystal_idx]]
                    eigenvectors = eigenvectors[:, :crystal_band_counts[crystal_idx], :]
                else:
                    # Calculate band window centered around valence electron count
                    band_window_start = crystal_valence_counts[crystal_idx] - self.band_num_control
                    band_window_end = crystal_valence_counts[crystal_idx] + self.band_num_control
                    eigenvalues = eigenvalues[:, band_window_start:band_window_end]
                    eigenvectors = eigenvectors[:, band_window_start:band_window_end, :]

            # Store results
            band_energies.append(torch.transpose(eigenvalues, dim0=-1, dim1=-2))
            wavefunctions.append(eigenvectors)

        # Combine results across all crystals
        band_energies = torch.cat(band_energies, dim=0)
        wavefunctions = torch.cat(wavefunctions, dim=0).reshape(-1)

        return band_energies, wavefunctions

    def apply_orbital_masks_to_hamiltonians(self, onsite_hamiltonian, offsite_hamiltonian, data, return_masks=False):
        """
        Apply atomic orbital validity masks to on-site and off-site Hamiltonian matrices.

        This function zeroes out elements of Hamiltonian matrices that correspond to invalid 
        or non-existent atomic orbitals based on the atomic numbers. This ensures physical
        correctness by preventing interactions involving orbitals that shouldn't exist for
        particular atom types.

        Args:
            onsite_hamiltonian (torch.Tensor): On-site Hamiltonian matrix elements with shape
                (n_atoms, nao_max^2) or (n_atoms, nao_max, nao_max).
            offsite_hamiltonian (torch.Tensor): Off-site Hamiltonian matrix elements with shape 
                (n_edges, nao_max^2) or (n_edges, nao_max, nao_max).
            data (DataObject): Object containing:
                - z: Atomic numbers for each atom
                - edge_index: Indices of connected atom pairs (source, target)
            return_masks (bool, optional): Whether to return the masks along with masked 
                Hamiltonians. Defaults to False.

        Returns:
            tuple: If return_masks is False:
                (masked_onsite_hamiltonian, masked_offsite_hamiltonian)

                If return_masks is True:
                (masked_onsite_hamiltonian, masked_offsite_hamiltonian, 
                 onsite_orbital_mask, offsite_orbital_mask)
        """
        # Create atomic orbital validity mask based on atomic numbers
        atomic_orbital_mask = torch.zeros((99, self.nao_max), device=data.z.device).type_as(data.z)

        # Populate mask with 1s for valid orbitals of each atomic number
        for atomic_number, orbital_indices in self.basis_def.items():
            atomic_orbital_mask[atomic_number][orbital_indices] = 1

        # Save original shapes for reshaping at the end
        original_onsite_shape = onsite_hamiltonian.shape
        original_offsite_shape = offsite_hamiltonian.shape

        # ---- Create and apply mask for on-site Hamiltonian ----

        # Get orbital mask for each atom
        atom_orbital_masks = atomic_orbital_mask[data.z].view(-1, self.nao_max)  # [n_atoms, nao_max]

        # Create outer product of masks for each atom's interaction with itself
        # This gives a matrix where element (i,j) is 1 if both orbitals i and j are valid
        onsite_orbital_mask = atom_orbital_masks[:, :, None] * atom_orbital_masks[:, None, :]  # [n_atoms, nao_max, nao_max]

        # Reshape to match Hamiltonian shape
        onsite_orbital_mask = onsite_orbital_mask.reshape(original_onsite_shape)

        # Apply mask to on-site Hamiltonian
        masked_onsite_hamiltonian = onsite_hamiltonian * onsite_orbital_mask

        # ---- Create and apply mask for off-site Hamiltonian ----

        # Get source and target atom indices for each edge
        source_atoms, target_atoms = data.edge_index

        # Get orbital masks for source and target atoms of each edge
        source_orbital_masks = atomic_orbital_mask[data.z[source_atoms]].view(-1, self.nao_max)  # [n_edges, nao_max]
        target_orbital_masks = atomic_orbital_mask[data.z[target_atoms]].view(-1, self.nao_max)  # [n_edges, nao_max]

        # Create outer product of source and target masks for each edge
        # This gives a matrix where element (i,j) is 1 if both orbitals i (source) and j (target) are valid
        offsite_orbital_mask = source_orbital_masks[:, :, None] * target_orbital_masks[:, None, :]  # [n_edges, nao_max, nao_max]

        # Reshape to match Hamiltonian shape
        offsite_orbital_mask = offsite_orbital_mask.reshape(original_offsite_shape)

        # Apply mask to off-site Hamiltonian
        masked_offsite_hamiltonian = offsite_hamiltonian * offsite_orbital_mask

        # Return results based on return_masks flag
        if return_masks:
            return masked_onsite_hamiltonian, masked_offsite_hamiltonian, onsite_orbital_mask, offsite_orbital_mask
        else:
            return masked_onsite_hamiltonian, masked_offsite_hamiltonian

    def symmetrize_orbital_coefficients(self, coefficient_matrix):
        """
        Enforce spherical symmetry on orbital coefficient matrices by averaging within angular momentum blocks.

        This function applies orbital symmetrization to ensure that coefficients maintain proper
        rotational invariance within each angular momentum subspace (p, d, f orbitals). Each block
        of coefficients corresponding to orbitals with the same angular momentum is averaged to 
        enforce spherical symmetry constraints.

        Orbital index ranges by angular momentum:
        - s orbitals: 0:3 (single orbital with additional indices)
        - p orbitals: 3:6 (3 components)
        - p' orbitals: 6:9 (3 components)
        - d orbitals: 9:14 (5 components)
        - d' orbitals: 14:19 (5 components, only for nao_max �� 19)
        - f orbitals: 19:26 (7 components, only for nao_max = 26)

        Args:
            coefficient_matrix (torch.Tensor): Coefficient matrix with shape (batch_size, nao_max2)
                or (batch_size, nao_max, nao_max).

        Returns:
            torch.Tensor: Symmetrized coefficient matrix with shape (batch_size, nao_max2).
        """
        batch_size = coefficient_matrix.shape[0]

        # Reshape to 3D tensor if necessary
        matrix_3d = coefficient_matrix.reshape(batch_size, self.nao_max, self.nao_max)

        # Define orbital blocks based on angular momentum
        orbital_blocks = []

        if self.nao_max >= 14:  # All sizes have at least these blocks
            orbital_blocks = [
                (3, 6),    # p orbitals (3 components)
                (6, 9),    # p' orbitals (3 components)
                (9, 14),   # d orbitals (5 components)
            ]

        if self.nao_max >= 19:  # 19 and 26 have d' orbitals
            orbital_blocks.append((14, 19))  # d' orbitals (5 components)

        if self.nao_max == 26:  # Only 26 has f orbitals
            orbital_blocks.append((19, 26))  # f orbitals (7 components)

        # Apply row-wise averaging (across orbital m components)
        for start_idx, end_idx in orbital_blocks:
            # Calculate mean across the angular momentum block
            block_mean = torch.mean(matrix_3d[:, start_idx:end_idx], dim=1, keepdim=True)

            # Expand mean to match the original block shape and assign
            block_size = end_idx - start_idx
            matrix_3d[:, start_idx:end_idx] = block_mean.expand(batch_size, block_size, self.nao_max)

        # Apply column-wise averaging (across orbital m components)
        for start_idx, end_idx in orbital_blocks:
            # Calculate mean across the angular momentum block
            block_mean = torch.mean(matrix_3d[:, :, start_idx:end_idx], dim=2, keepdim=True)

            # Expand mean to match the original block shape and assign
            block_size = end_idx - start_idx
            matrix_3d[:, :, start_idx:end_idx] = block_mean.expand(batch_size, self.nao_max, block_size)

        # Flatten the matrix back to original shape
        return matrix_3d.view(batch_size, -1)
    
    def create_cell_index_mapping(self, unique_cell_vectors: List[List[int]]) -> dict:
        """
        Create a mapping from unique cell vectors to their indices.

        This function creates a dictionary that associates each unique cell vector 
        (representing a unit cell in a periodic lattice) with its index in the list
        of unique vectors. This mapping is used for efficient lookup of cell indices
        when constructing Hamiltonian matrices with periodic boundary conditions.

        Args:
            unique_cell_vectors (List[List[int]]): A list of cell vectors, where each 
                vector is represented as a list of integers (typically 3 integers for 
                3D periodic systems).

        Returns:
            dict: A dictionary mapping each cell vector (as a tuple) to its integer index.
        """
        # Create mapping using dictionary comprehension for efficiency
        return {tuple(cell_vector): index for index, cell_vector in enumerate(unique_cell_vectors)}

    def extract_unique_cell_vectors(self, data):
        """
        Extract unique cell vectors and create mappings for periodic boundary conditions.

        This function processes cell shift data to:
        1. Find all unique cell shift vectors
        2. Ensure the zero vector (0,0,0) is included (required for on-site interactions)
        3. Create a mapping from each cell shift to its unique index
        4. Create an index array that maps each edge to its corresponding cell index

        This information is essential for constructing Hamiltonian matrices with
        proper periodic boundary conditions.

        Args:
            data (DataObject): Object containing crystal structure information, including:
                - cell_shift: Tensor of cell shift vectors for each edge, with shape 
                  (n_edges, 3), representing the periodic image shifts.

        Returns:
            tuple:
                - unique_cell_vectors (torch.Tensor): Tensor of unique cell shift vectors
                  with shape (n_unique_cells, 3).
                - cell_vector_indices (torch.Tensor): Tensor mapping each edge to the
                  index of its corresponding unique cell vector, with shape (n_edges,).
                - cell_vector_map (dict): Dictionary mapping each cell vector tuple to
                  its index in unique_cell_vectors.
        """
        # Get all cell shift vectors from the data
        cell_shift_vectors = data.cell_shift

        # Find all unique cell shift vectors
        unique_cell_vectors = torch.unique(cell_shift_vectors, dim=0)

        # Create zero vector with the same type as unique vectors
        zero_vector = torch.tensor([[0, 0, 0]], device=unique_cell_vectors.device).type_as(unique_cell_vectors)

        # Check if zero vector is already present in the unique vectors
        is_zero_vector_present = torch.any(torch.all(unique_cell_vectors == zero_vector, dim=1))

        # Add zero vector if not present (required for on-site interactions)
        if not is_zero_vector_present:
            unique_cell_vectors = torch.cat((zero_vector, unique_cell_vectors), dim=0)

        # Create mapping from each edge's cell vector to its index in the unique vectors list
        # Method 1: Using tensor expansion (works for all PyTorch versions)
        # Expand tensors for broadcasting comparison
        expanded_cell_shifts = cell_shift_vectors.unsqueeze(1).expand(-1, unique_cell_vectors.size(0), -1)
        expanded_unique_vectors = unique_cell_vectors.unsqueeze(0).expand(cell_shift_vectors.size(0), -1, -1)

        # Find exact matches (where all dimensions are equal)
        matches = torch.all(expanded_cell_shifts == expanded_unique_vectors, dim=2)

        # Extract the index of the matching unique vector for each edge
        cell_vector_indices = matches.nonzero(as_tuple=True)[1]  # Shape: (n_edges,)

        # Create dictionary mapping each cell vector to its index
        cell_vector_map = self.create_cell_index_mapping(unique_cell_vectors.tolist())

        return unique_cell_vectors, cell_vector_indices, cell_vector_map

    def build_edge_lookup_structures(self, data, inverse_edge_indices=None):
        """
        Build efficient edge lookup structures for crystal graph operations.

        This function constructs two data structures that enable fast edge lookup:
        1. A mapping from each atom to all edges where that atom is the source
        2. A mapping from each atom and cell shift combination to corresponding edges

        These structures accelerate operations that require finding all edges connected 
        to specific atoms, particularly when periodic boundary conditions are involved.

        Args:
            data (DataObject): Crystal graph data containing:
                - edge_index: Tensor of shape [2, num_edges] with source and target indices
                - unique_cell_shift: Tensor of unique cell shift vectors
                - cell_shift_indices: Tensor mapping each edge to its cell shift index
                - cell_index_map: Dictionary mapping cell shift tuples to indices
                - z: Atomic numbers, used to determine number of atoms
            inverse_edge_indices (torch.Tensor, optional): Tensor mapping each edge to its
                inverse edge index (i��j maps to j��i). Required for building the target lookup.

        Returns:
            tuple:
                - source_edge_indices (list of torch.Tensor): For each atom, contains indices
                  of edges where that atom is the source.
                - target_edge_indices (list of list of torch.Tensor): For each atom and cell
                  shift combination, contains indices of edges where that atom is the target
                  in the specified cell shift.
        """
        # Extract necessary data
        source_nodes, target_nodes = data.edge_index
        unique_cell_shifts = data.unique_cell_shift
        cell_shift_indices = data.cell_shift_indices
        cell_index_map = data.cell_index_map

        # Get dimensions
        num_atoms = len(data.z)
        num_cell_shifts = len(unique_cell_shifts)

        # Build mapping from each atom to edges where it's the source
        # This finds all edges that originate from each atom
        source_edge_indices = [torch.where(source_nodes == atom_idx)[0] for atom_idx in range(num_atoms)]

        # Initialize the nested structure for target edge lookup
        # This will map each (atom, cell_shift) pair to a list of edge indices
        target_edge_indices = [[[] for _ in range(num_cell_shifts)] for _ in range(num_atoms)]

        # Populate target edge lookup structure
        for atom_idx in range(num_atoms):
            # Get the inverse edges for edges where this atom is the source
            # These are edges where this atom is the target
            inverse_edges = inverse_edge_indices[source_edge_indices[atom_idx]]

            # Get cell shift indices for these inverse edges
            cell_shifts_of_inverse_edges = cell_shift_indices[inverse_edges]

            # Group inverse edges by their cell shift
            for edge_idx, cell_shift_idx in zip(inverse_edges, cell_shifts_of_inverse_edges):
                target_edge_indices[atom_idx][cell_shift_idx.item()].append(edge_idx)

            # Convert lists to tensors for efficiency
            for cell_shift_idx in range(num_cell_shifts):
                if target_edge_indices[atom_idx][cell_shift_idx]:
                    # If edges exist for this cell shift, stack them into a tensor
                    target_edge_indices[atom_idx][cell_shift_idx] = torch.stack(
                        target_edge_indices[atom_idx][cell_shift_idx]
                    ).type_as(source_nodes)
                else:
                    # Otherwise create an empty tensor of the correct type
                    target_edge_indices[atom_idx][cell_shift_idx] = torch.tensor(
                        [], dtype=torch.long
                    ).type_as(source_nodes)

        return source_edge_indices, target_edge_indices

    def create_orbital_validity_mask(self, atomic_numbers):
        """
        Create a mask identifying valid atomic orbitals for each element type.

        This function generates a binary mask tensor where each row corresponds to an
        element in the periodic table (indexed by atomic number), and each column
        represents an atomic orbital. A value of 1 indicates that the orbital is
        valid for that element type, while 0 indicates an invalid orbital.

        Args:
            atomic_numbers (torch.Tensor): Tensor containing atomic numbers, used only
                for device and dtype information.

        Returns:
            torch.Tensor: Binary mask tensor of shape (99, nao_max) where 99 covers
                all possible elements in the periodic table and nao_max is the maximum
                number of atomic orbitals.
        """
        # Initialize all orbitals as invalid (zeros)
        orbital_mask = torch.zeros((99, self.nao_max), device=atomic_numbers.device).type_as(atomic_numbers)

        # Set valid orbitals to 1 for each element type based on the basis definition
        for atomic_number, valid_orbital_indices in self.basis_def.items():
            orbital_mask[atomic_number][valid_orbital_indices] = 1

        return orbital_mask

    def build_interaction_masks(self, data):
        """
        Build boolean masks for valid orbital interactions in Hamiltonian matrices.

        This function generates masks that identify which elements in the Hamiltonian
        matrices correspond to valid orbital-orbital interactions, based on the atomic
        species involved. It creates separate masks for on-site (same atom) and
        off-site (different atoms) interactions.

        Args:
            data (DataObject): Graph data containing:
                - edge_index: Indices of connected atom pairs
                - z: Atomic numbers for each atom

        Returns:
            torch.Tensor: Combined mask tensor with shape (n_atoms + n_edges, nao_max2)
                where the first n_atoms rows are for on-site interactions and the
                remaining n_edges rows are for off-site interactions.
        """
        source_indices, target_indices = data.edge_index
        atomic_numbers = data.z

        # Get orbital validity mask for each element type
        orbital_validity_mask = self.create_orbital_validity_mask(atomic_numbers)

        # Create on-site interaction masks using einsum for efficient outer product
        # Each mask has shape (n_atoms, nao_max, nao_max) where element (i,j,k) is True
        # if orbital j of atom i can interact with orbital k of the same atom
        onsite_masks = torch.einsum(
            'ni, nj -> nij', 
            orbital_validity_mask[atomic_numbers], 
            orbital_validity_mask[atomic_numbers]
        ).bool()

        # Create off-site interaction masks using einsum
        # Each mask has shape (n_edges, nao_max, nao_max) where element (e,j,k) is True
        # if orbital j of source atom can interact with orbital k of target atom
        offsite_masks = torch.einsum(
            'ni, nj -> nij', 
            orbital_validity_mask[atomic_numbers[source_indices]], 
            orbital_validity_mask[atomic_numbers[target_indices]]
        ).bool()

        # Reshape masks to 2D format and concatenate
        combined_masks = torch.cat(
            (onsite_masks.reshape(-1, self.nao_max**2), 
             offsite_masks.reshape(-1, self.nao_max**2)), 
            dim=0
        )

        return combined_masks

    def build_column_wise_interaction_masks(self, data):
        """
        Build interaction masks with an additional column dimension.

        Similar to build_interaction_masks, but creates masks with an additional
        dimension for column-wise operations (e.g., for real and imaginary parts).

        Args:
            data (DataObject): Graph data containing:
                - edge_index: Indices of connected atom pairs
                - z: Atomic numbers for each atom

        Returns:
            torch.Tensor: Combined mask tensor with shape 
                (n_atoms + n_edges, 2, nao_max2) where the additional
                dimension can be used for real and imaginary parts.
        """
        source_indices, target_indices = data.edge_index
        atomic_numbers = data.z

        # Get orbital validity mask for each element type
        orbital_validity_mask = self.create_orbital_validity_mask(atomic_numbers)

        # Create on-site interaction masks
        onsite_masks = torch.einsum(
            'ni, nj -> nij', 
            orbital_validity_mask[atomic_numbers], 
            orbital_validity_mask[atomic_numbers]
        ).bool()

        # Stack the mask along a new dimension (for real and imaginary parts)
        # Shape: (n_atoms, 2, nao_max, nao_max)
        onsite_masks_stacked = torch.stack([onsite_masks, onsite_masks], dim=1)

        # Create off-site interaction masks
        offsite_masks = torch.einsum(
            'ni, nj -> nij', 
            orbital_validity_mask[atomic_numbers[source_indices]], 
            orbital_validity_mask[atomic_numbers[target_indices]]
        ).bool()

        # Stack the mask along a new dimension
        # Shape: (n_edges, 2, nao_max, nao_max)
        offsite_masks_stacked = torch.stack([offsite_masks, offsite_masks], dim=1)

        # Reshape masks and concatenate
        combined_masks = torch.cat(
            (onsite_masks_stacked.reshape(-1, 2, self.nao_max**2), 
             offsite_masks_stacked.reshape(-1, 2, self.nao_max**2)), 
            dim=0
        )

        return combined_masks

    def build_spin_orbit_interaction_masks(self, data):
        """
        Build interaction masks that account for spin-orbit coupling.

        This function generates masks for systems with spin-orbit coupling, which
        requires handling interactions between different spin components. The resulting
        masks have double the orbital dimension to account for spin-up and spin-down
        components.

        Args:
            data (DataObject): Graph data containing:
                - edge_index: Indices of connected atom pairs
                - z: Atomic numbers for each atom

        Returns:
            tuple: A tuple containing:
                - real_imag_masks (torch.Tensor): Mask for real and imaginary
                  components with shape (n_atoms + n_edges, (2*nao_max)2)
                - combined_masks (torch.Tensor): Full mask tensor with shape 
                  (2*(n_atoms + n_edges), (2*nao_max)2)
        """
        source_indices, target_indices = data.edge_index
        atomic_numbers = data.z

        # Get orbital validity mask for each element type
        orbital_validity_mask = self.create_orbital_validity_mask(atomic_numbers)

        # Create base interaction masks without converting to boolean yet
        onsite_base_masks = torch.einsum(
            'ni, nj -> nij', 
            orbital_validity_mask[atomic_numbers], 
            orbital_validity_mask[atomic_numbers]
        )

        offsite_base_masks = torch.einsum(
            'ni, nj -> nij', 
            orbital_validity_mask[atomic_numbers[source_indices]], 
            orbital_validity_mask[atomic_numbers[target_indices]]
        )

        # Expand to include spin components using 2x2 block structure
        # This creates a block diagonal matrix where each block represents
        # interactions between different spin components
        onsite_expanded_masks = blockwise_2x2_concat(
            onsite_base_masks, onsite_base_masks, 
            onsite_base_masks, onsite_base_masks
        ).reshape(-1, (2*self.nao_max)**2).bool()

        offsite_expanded_masks = blockwise_2x2_concat(
            offsite_base_masks, offsite_base_masks, 
            offsite_base_masks, offsite_base_masks
        ).reshape(-1, (2*self.nao_max)**2).bool()

        # Combine on-site and off-site masks
        real_imag_masks = self.concatenate_hamiltonians_by_crystal(
            data, onsite_expanded_masks, offsite_expanded_masks
        )

        # Create full mask tensor by duplicating (for real and imaginary parts)
        combined_masks = torch.cat((real_imag_masks, real_imag_masks), dim=0)

        return real_imag_masks, combined_masks

    def calculate_sparsity_ratio(self, data):
        """
        Calculate the ratio between the total possible matrix elements and the effective matrix elements.

        This function computes the sparsity of Hamiltonian matrices by dividing the total number
        of possible matrix elements by the number of effective matrix elements based on the
        atomic basis definitions.

        Parameters
        ----------
        data : object
            Data object containing atomic information and Hamiltonian matrices.

        Returns
        -------
        torch.Tensor
            A scalar tensor on the same device as ``data.z``. The sparsity ratio is
            defined as the total number of matrix elements divided by the number of
            effective matrix elements. Returns ``inf`` if there are no effective elements.

        Notes
        -----
        The calculation considers both on-site and off-site Hamiltonian elements
        if they are present in the data object.
        """
        # Fast path: allow offline precompute/caching on the Data object.
        if hasattr(data, "sparsity_ratio"):
            cached = getattr(data, "sparsity_ratio")
            if torch.is_tensor(cached):
                return cached.to(device=data.z.device, dtype=torch.float32)
            return torch.tensor(float(cached), device=data.z.device, dtype=torch.float32)

        atomic_numbers = data.z
        device = atomic_numbers.device

        # Build (and cache) lookup tables on the correct device to avoid per-atom/edge Python loops.
        cache = getattr(self, "_sparsity_ratio_lookup_tables", None)
        if cache is None or cache[0].device != device:
            table_size = 256  # covers Z<=118 and avoids device sync from dynamic sizing
            n_orbital = torch.full((table_size,), int(self.nao_max), dtype=torch.long, device=device)
            is_defined = torch.zeros((table_size,), dtype=torch.bool, device=device)
            for z, basis_indices in self.basis_def.items():
                zi = int(z)
                if 0 <= zi < table_size:
                    n_orbital[zi] = int(len(basis_indices))
                    is_defined[zi] = True
            cache = (n_orbital, is_defined)
            setattr(self, "_sparsity_ratio_lookup_tables", cache)

        n_orbital, is_defined = cache
        z = atomic_numbers.to(torch.long)

        nao_max_squared = int(self.nao_max) ** 2
        total = z.new_zeros((), dtype=torch.float32)
        effective = z.new_zeros((), dtype=torch.float32)

        # On-site blocks: sum_i (n_i^2); unknown elements default to nao_max^2 (matches legacy behavior).
        if hasattr(data, "Hon"):
            num_atoms = int(z.numel())
            total = total + float(num_atoms * nao_max_squared)
            n_i = n_orbital[z]
            effective = effective + (n_i.to(torch.float32) ** 2).sum()

        # Off-site blocks: sum_e (n_src*n_dst) if both defined else nao_max^2 (matches legacy behavior).
        if hasattr(data, "Hoff") and hasattr(data, "edge_index"):
            edge_index = data.edge_index
            num_edges = int(edge_index.shape[1])
            total = total + float(num_edges * nao_max_squared)

            src = edge_index[0].to(torch.long)
            dst = edge_index[1].to(torch.long)
            z_src = z[src]
            z_dst = z[dst]

            both_defined = is_defined[z_src] & is_defined[z_dst]
            n_src = n_orbital[z_src]
            n_dst = n_orbital[z_dst]
            eff_edges = torch.where(
                both_defined,
                n_src * n_dst,
                n_src.new_full(n_src.shape, nao_max_squared),
            )
            effective = effective + eff_edges.to(torch.float32).sum()

        inf = torch.full_like(total, float("inf"))
        ratio = torch.where(effective > 0, total / effective, inf)
        return ratio.detach()

    def validate_elements_in_basis_def(self, data, raise_error=True):
        """
        Validate that all elements in the input data exist in the basis_def dictionary.

        Notes
        -----
        This function is used to ensure that all elements in the molecular system
        have corresponding basis set definitions before performing calculations.
        """
        # Get atomic numbers from data object
        if hasattr(data, 'z'):
            atomic_numbers = data.z
        elif hasattr(data, 'Z'):
            atomic_numbers = data.Z
        else:
            raise AttributeError("Input data object has no 'z' or 'Z' attribute containing atomic numbers")

        # Get unique atomic numbers and convert to standard Python list
        unique_atomic_numbers = atomic_numbers.unique().cpu().tolist()

        # Find atomic numbers that are missing from basis_def using set operations
        missing_atomic_numbers = [num for num in unique_atomic_numbers if num not in self.basis_def]

        # Check if all elements are valid
        all_elements_valid = len(missing_atomic_numbers) == 0

        # Handle case when invalid elements are found
        if not all_elements_valid and raise_error:
            # Try to convert atomic numbers to element symbols if possible
            try:
                # Attempt to use a periodic table package if available
                from qcelemental import periodictable
                element_symbols = [f"{periodictable.to_symbol(num)} (Z={num})" for num in missing_atomic_numbers]
            except (ImportError, AttributeError):
                # Fallback to just showing atomic numbers
                element_symbols = [f"Z={num}" for num in missing_atomic_numbers]

            raise ValueError(f"The following elements are missing from basis_def: {', '.join(element_symbols)}")

        # Return appropriate value based on raise_error parameter
        return all_elements_valid if raise_error else (all_elements_valid, missing_atomic_numbers)

    def forward(self, data, graph_representation: dict = None):
        """
        Forward pass of the Hamiltonian prediction model.

        This method constructs Hamiltonian matrices for crystal structures from graph representations,
        with support for various physical effects including spin-orbit coupling (SOC), magnetism,
        and long-range interactions. It can also calculate electronic band structures.

        The method handles several different physical regimes:
        1. Non-magnetic systems (standard DFT-like Hamiltonians)
        2. Collinear spin-polarized systems (separate Hamiltonians for up/down spins)
        3. Non-collinear magnetic systems (full 2x2 spin blocks with SOC)
        4. Systems with spin-orbit coupling (complex Hamiltonians with real/imaginary parts)

        Args:
            data (DataObject): Contains crystal structure information including:
                - edge_index: Connectivity information between atoms
                - z: Atomic numbers
                - pos: Atomic positions
                - cell: Unit cell vectors
                - Hon/Hoff: Reference on-site/off-site Hamiltonian matrices (if available)
                - Son/Soff: Reference on-site/off-site overlap matrices
                - node_counts: Number of atoms in each crystal
                - batch: Batch assignment for each atom
                - inv_edge_idx: Inverse edge indices

            graph_representation (dict): Output from the graph neural network containing:
                - node_attr: Node feature vectors from the GNN
                - edge_attr: Edge feature vectors from the GNN

        Returns:
            dict: Dictionary containing predicted quantities, which may include:
                - hamiltonian: Predicted Hamiltonian matrix
                - overlap: Predicted overlap matrix (if ham_only=False)
                - band_energy: Electronic band energies (if calculate_band_energy=True)
                - wavefunction: Eigenvectors of the Hamiltonian (if calculate_band_energy=True)
                - band_gap: Band gap values (if calculate_band_energy=True)
                - Various additional matrices for debugging or analysis

        Note:
            The exact contents of the return dictionary depend on the model configuration
            parameters (soc_switch, spin_constrained, collinear_spin, etc.)
        """
        # Validate that all elements in the data are present in basis_def
        self.validate_elements_in_basis_def(data)
        
        # Data format compatibility handling
        if 'H0_u' in data:
            # Handle legacy format from Hongyu yu's data
            Hon_u0 = data.H0_u[:len(data.z)].flatten(1)
            Hon_d0 = data.H0_d[:len(data.z)].flatten(1)
            Hoff_u0 = data.H0_u[len(data.z):].flatten(1)
            Hoff_d0 = data.H0_d[len(data.z):].flatten(1)
            data.Hon0 = torch.stack([Hon_u0, Hon_d0], dim=1)
            data.Hoff0 = torch.stack([Hoff_u0, Hoff_d0], dim=1)
            data.Hon = torch.stack([data.H_u[:len(data.z)], data.H_d[:len(data.z)]], dim=1).flatten(2)
            data.Hoff = torch.stack([data.H_u[len(data.z):], data.H_d[len(data.z):]], dim=1).flatten(2)

        # Prepare combined Hamiltonian and overlap matrices if not present
        if 'hamiltonian' not in data:
            data.hamiltonian = self.concatenate_hamiltonians_by_crystal(data, data.Hon, data.Hoff)
        if 'overlap' not in data:
            data.overlap = self.concatenate_hamiltonians_by_crystal(data, data.Son, data.Soff)

        # Extract node and edge attributes from graph representation
        node_attr = graph_representation['node_attr']
        edge_attr = graph_representation['edge_attr']  # mji
        source_indices, target_indices = data.edge_index

        # Calculate batch-specific inverse edge indices
        inverse_edge_indices = data.inv_edge_idx
        edge_count = torch.ones_like(source_indices)
        edge_count_by_batch = scatter(edge_count, data.batch[source_indices], dim=0)
        edge_count_offset = torch.cumsum(edge_count_by_batch, dim=0) - edge_count_by_batch
        inverse_edge_indices = inverse_edge_indices + edge_count_offset[data.batch[source_indices]]

        # Ensure irreps dimensions have correct data type
        self.hamiltonian_irreps_dimensions = self.hamiltonian_irreps_dimensions.type_as(source_indices)

        # === OVERLAP MATRIX CALCULATION (if enabled) ===
        if not self.ham_only:
            # Calculate on-site overlap matrix
            node_spherical_harmonics = self.onsite_overlap_network(node_attr)
            node_spherical_components = torch.split(node_spherical_harmonics, self.hamiltonian_irreps_dimensions.tolist(), dim=-1)
            onsite_overlap = self.merge_tensor_components(node_spherical_components)  # shape (n_atoms, nao_max2)

            # Reorder indices to match physical orbital ordering
            onsite_overlap = self.reorder_matrix(onsite_overlap)

            # Impose Hermitian symmetry
            onsite_overlap = self.symmetrize_onsite_hamiltonian(onsite_overlap)

            # Calculate off-site overlap matrix
            edge_spherical_harmonics = self.offsite_overlap_network(edge_attr)
            edge_spherical_components = torch.split(edge_spherical_harmonics, self.hamiltonian_irreps_dimensions.tolist(), dim=-1)
            offsite_overlap = self.merge_tensor_components(edge_spherical_components)

            # Reorder indices and impose symmetry
            offsite_overlap = self.reorder_matrix(offsite_overlap)
            offsite_overlap = self.symmetrize_offsite_hamiltonian(offsite_overlap, inverse_edge_indices)

            # Apply orbital masking for certain Hamiltonian types
            if self.ham_type in ['openmx', 'pasp', 'siesta', 'abacus']:
                onsite_overlap, offsite_overlap = self.apply_orbital_masks_to_hamiltonians(onsite_overlap, offsite_overlap, data)

        # === HAMILTONIAN CALCULATION WITH SPIN EFFECTS ===
        if self.soc_switch or self.spin_constrained:
            # Handle spin-orbit coupling
            if self.soc_switch:
                # Different SOC basis implementations
                if self.soc_basis == 'so3':
                    # SO(3) representation of spin-orbit coupling
                    if self.add_H_nonsoc:
                        # Use pre-defined non-SOC Hamiltonians
                        onsite_hamiltonian, offsite_hamiltonian = data.Hon_nonsoc, data.Hoff_nonsoc

                        # Process the H0 matrices for SOC
                        onsite_h0, offsite_h0 = data['Hon0'], data['Hoff0']
                        onsite_h0_resized = onsite_h0.reshape(-1, 2 * self.nao_max, 2 * self.nao_max)
                        offsite_h0_resized = offsite_h0.reshape(-1, 2 * self.nao_max, 2 * self.nao_max)

                        # Create zero blocks for the diagonal blocks
                        zero_onsite = torch.zeros_like(data['Son']).reshape(-1, self.nao_max, self.nao_max)
                        zero_offsite = torch.zeros_like(data['Soff']).reshape(-1, self.nao_max, self.nao_max)

                        # Zero out the diagonal blocks (spin-conserving terms)
                        onsite_h0_resized[:, :self.nao_max, :self.nao_max] = zero_onsite
                        onsite_h0_resized[:, self.nao_max:, self.nao_max:] = zero_onsite
                        offsite_h0_resized[:, :self.nao_max, :self.nao_max] = zero_offsite
                        offsite_h0_resized[:, self.nao_max:, self.nao_max:] = zero_offsite

                        # Store processed matrices
                        data['Hon0'] = onsite_h0_resized.reshape(-1, (2 * self.nao_max) ** 2)
                        data['Hoff0'] = offsite_h0_resized.reshape(-1, (2 * self.nao_max) ** 2)
                    else:
                        # Calculate Hamiltonians from spherical harmonics
                        node_spherical_harmonics = self.onsite_hamiltonian_network(node_attr)
                        node_spherical_components = torch.split(node_spherical_harmonics, self.hamiltonian_irreps_dimensions.tolist(), dim=-1)
                        onsite_hamiltonian = self.merge_tensor_components(node_spherical_components)

                        # Process on-site Hamiltonian
                        onsite_hamiltonian = self.reorder_matrix(onsite_hamiltonian)
                        onsite_hamiltonian = self.symmetrize_onsite_hamiltonian(onsite_hamiltonian)

                        # Calculate off-site Hamiltonian
                        edge_spherical_harmonics = self.offsite_hamiltonian_network(edge_attr)
                        edge_spherical_components = torch.split(edge_spherical_harmonics, self.hamiltonian_irreps_dimensions.tolist(), dim=-1)
                        offsite_hamiltonian = self.merge_tensor_components(edge_spherical_components)

                        # Process off-site Hamiltonian
                        offsite_hamiltonian = self.reorder_matrix(offsite_hamiltonian)
                        offsite_hamiltonian = self.symmetrize_offsite_hamiltonian(offsite_hamiltonian, inverse_edge_indices)
                        
                        # Apply orbital masking
                        onsite_hamiltonian, offsite_hamiltonian = self.apply_orbital_masks_to_hamiltonians(onsite_hamiltonian, offsite_hamiltonian, data)

                    # Calculate SOC coupling strength
                    ksi_onsite = self.onsite_ksi_network(node_attr)
                    ksi_onsite = self.symmetrize_orbital_coefficients(ksi_onsite)
                    ksi_offsite = self.offsite_ksi_network(edge_attr)
                    ksi_offsite = self.symmetrize_orbital_coefficients(ksi_offsite)

                    # Construct SOC Hamiltonian real part (2x2 block structure)
                    soc_onsite_real = torch.zeros((onsite_hamiltonian.shape[0], 2*self.nao_max, 2*self.nao_max)).type_as(onsite_hamiltonian)
                    # Diagonal blocks (spin-conserving terms)
                    soc_onsite_real[:, :self.nao_max, :self.nao_max] = onsite_hamiltonian.reshape(-1, self.nao_max, self.nao_max)
                    soc_onsite_real[:, self.nao_max:, self.nao_max:] = onsite_hamiltonian.reshape(-1, self.nao_max, self.nao_max)
                    # Off-diagonal blocks (spin-flip terms)
                    soc_onsite_real[:, :self.nao_max, self.nao_max:] = self.symmetrize_onsite_hamiltonian(
                        (ksi_onsite*data.Lon[:, :, 1]), hermitian=False
                    ).reshape(-1, self.nao_max, self.nao_max)
                    soc_onsite_real[:, self.nao_max:, :self.nao_max] = self.symmetrize_onsite_hamiltonian(
                        (ksi_onsite*data.Lon[:, :, 1]), hermitian=False
                    ).reshape(-1, self.nao_max, self.nao_max)

                    soc_onsite_real = soc_onsite_real.reshape(-1, (2*self.nao_max)**2)

                    # Construct SOC Hamiltonian imaginary part
                    soc_onsite_imag = torch.zeros((onsite_hamiltonian.shape[0], 2*self.nao_max, 2*self.nao_max)).type_as(onsite_hamiltonian)
                    # Diagonal blocks
                    soc_onsite_imag[:, :self.nao_max, :self.nao_max] = self.symmetrize_onsite_hamiltonian(
                        (ksi_onsite*data.Lon[:, :, 2]), hermitian=False
                    ).reshape(-1, self.nao_max, self.nao_max)
                    soc_onsite_imag[:, self.nao_max:, self.nao_max:] = -self.symmetrize_onsite_hamiltonian(
                        (ksi_onsite*data.Lon[:, :, 2]), hermitian=False
                    ).reshape(-1, self.nao_max, self.nao_max)
                    # Off-diagonal blocks
                    soc_onsite_imag[:, :self.nao_max, self.nao_max:] = self.symmetrize_onsite_hamiltonian(
                        (ksi_onsite*data.Lon[:, :, 0]), hermitian=False
                    ).reshape(-1, self.nao_max, self.nao_max)
                    soc_onsite_imag[:, self.nao_max:, :self.nao_max] = -self.symmetrize_onsite_hamiltonian(
                        (ksi_onsite*data.Lon[:, :, 0]), hermitian=False
                    ).reshape(-1, self.nao_max, self.nao_max)

                    soc_onsite_imag = soc_onsite_imag.reshape(-1, (2*self.nao_max)**2)

                    # Similar construction for off-site SOC Hamiltonian
                    soc_offsite_real = torch.zeros((offsite_hamiltonian.shape[0], 2*self.nao_max, 2*self.nao_max)).type_as(offsite_hamiltonian)
                    # Diagonal blocks
                    soc_offsite_real[:, :self.nao_max, :self.nao_max] = offsite_hamiltonian.reshape(-1, self.nao_max, self.nao_max)
                    soc_offsite_real[:, self.nao_max:, self.nao_max:] = offsite_hamiltonian.reshape(-1, self.nao_max, self.nao_max)
                    # Off-diagonal blocks
                    soc_offsite_real[:, :self.nao_max, self.nao_max:] = self.symmetrize_offsite_hamiltonian(
                        (ksi_offsite*data.Loff[:, :, 1]), inverse_edge_indices, hermitian=False
                    ).reshape(-1, self.nao_max, self.nao_max)
                    soc_offsite_real[:, self.nao_max:, :self.nao_max] = self.symmetrize_offsite_hamiltonian(
                        (ksi_offsite*data.Loff[:, :, 1]), inverse_edge_indices, hermitian=False
                    ).reshape(-1, self.nao_max, self.nao_max)

                    soc_offsite_real = soc_offsite_real.reshape(-1, (2*self.nao_max)**2)

                    # Off-site SOC Hamiltonian imaginary part
                    soc_offsite_imag = torch.zeros((offsite_hamiltonian.shape[0], 2*self.nao_max, 2*self.nao_max)).type_as(offsite_hamiltonian)
                    # Diagonal blocks
                    soc_offsite_imag[:, :self.nao_max, :self.nao_max] = self.symmetrize_offsite_hamiltonian(
                        (ksi_offsite*data.Loff[:, :, 2]), inverse_edge_indices, hermitian=False
                    ).reshape(-1, self.nao_max, self.nao_max)
                    soc_offsite_imag[:, self.nao_max:, self.nao_max:] = -self.symmetrize_offsite_hamiltonian(
                        (ksi_offsite*data.Loff[:, :, 2]), inverse_edge_indices, hermitian=False
                    ).reshape(-1, self.nao_max, self.nao_max)
                    # Off-diagonal blocks
                    soc_offsite_imag[:, :self.nao_max, self.nao_max:] = self.symmetrize_offsite_hamiltonian(
                        (ksi_offsite*data.Loff[:, :, 0]), inverse_edge_indices, hermitian=False
                    ).reshape(-1, self.nao_max, self.nao_max)
                    soc_offsite_imag[:, self.nao_max:, :self.nao_max] = -self.symmetrize_offsite_hamiltonian(
                        (ksi_offsite*data.Loff[:, :, 0]), inverse_edge_indices, hermitian=False
                    ).reshape(-1, self.nao_max, self.nao_max)

                    soc_offsite_imag = soc_offsite_imag.reshape(-1, (2*self.nao_max)**2)

                elif self.soc_basis == 'su2':
                    # SU(2) representation of spin-orbit coupling
                    node_spherical_harmonics = self.onsite_hamiltonian_network(node_attr)
                    onsite_hamiltonian = self.hamiltonian_decomposition.get_H(node_spherical_harmonics)
                    onsite_hamiltonian = self.reorder_matrix(onsite_hamiltonian)
                    onsite_hamiltonian = onsite_hamiltonian.reshape(-1, 2, 2, self.nao_max, self.nao_max)
                    onsite_hamiltonian = torch.swapaxes(onsite_hamiltonian, 2, 3)  # shape (n_atoms, 2, nao_max, 2, nao_max)
                    onsite_hamiltonian = self.symmetrize_onsite_hamiltonian_soc(onsite_hamiltonian).reshape(-1, 2, self.nao_max, 2, self.nao_max) # Ensure Hermiticity

                    # Calculate off-site Hamiltonian
                    edge_spherical_harmonics = self.offsite_hamiltonian_network(edge_attr)
                    offsite_hamiltonian = self.hamiltonian_decomposition.get_H(edge_spherical_harmonics)
                    offsite_hamiltonian = self.reorder_matrix(offsite_hamiltonian)
                    offsite_hamiltonian = offsite_hamiltonian.reshape(-1, 2, 2, self.nao_max, self.nao_max)
                    offsite_hamiltonian = torch.swapaxes(offsite_hamiltonian, 2, 3)  # shape (n_edges, 2, nao_max, 2, nao_max)
                    offsite_hamiltonian = self.symmetrize_offsite_hamiltonian_soc(offsite_hamiltonian, inverse_edge_indices).reshape(-1, 2, self.nao_max, 2, self.nao_max) # Ensure Hermiticity
                    
                    # Apply orbital masking to each spin block
                    for i in range(2):
                        for j in range(2):
                            onsite_hamiltonian[:, i, :, j, :], offsite_hamiltonian[:, i, :, j, :] = self.apply_orbital_masks_to_hamiltonians(
                                onsite_hamiltonian[:, i, :, j, :], offsite_hamiltonian[:, i, :, j, :], data
                            )

                    # Reshape to flattened form
                    onsite_hamiltonian = onsite_hamiltonian.reshape(-1, (2*self.nao_max)**2)
                    offsite_hamiltonian = offsite_hamiltonian.reshape(-1, (2*self.nao_max)**2)

                    # Extract real and imaginary parts
                    soc_onsite_real = onsite_hamiltonian.real
                    soc_offsite_real = offsite_hamiltonian.real
                    soc_onsite_imag = onsite_hamiltonian.imag
                    soc_offsite_imag = offsite_hamiltonian.imag

                else:
                    raise NotImplementedError("Unsupported SOC basis")

            else:  # No SOC, but spin-constrained
                # Calculate standard Hamiltonian
                node_spherical_harmonics = self.onsite_hamiltonian_network(node_attr)
                node_spherical_components = torch.split(node_spherical_harmonics, self.hamiltonian_irreps_dimensions.tolist(), dim=-1)
                onsite_hamiltonian = self.merge_tensor_components(node_spherical_components)

                onsite_hamiltonian = self.reorder_matrix(onsite_hamiltonian)
                onsite_hamiltonian = self.symmetrize_onsite_hamiltonian(onsite_hamiltonian)

                # Calculate off-site Hamiltonian
                edge_spherical_harmonics = self.offsite_hamiltonian_network(edge_attr)
                edge_spherical_components = torch.split(edge_spherical_harmonics, self.hamiltonian_irreps_dimensions.tolist(), dim=-1)
                offsite_hamiltonian = self.merge_tensor_components(edge_spherical_components)

                offsite_hamiltonian = self.reorder_matrix(offsite_hamiltonian)
                offsite_hamiltonian = self.symmetrize_offsite_hamiltonian(offsite_hamiltonian, inverse_edge_indices)
                
                # Apply orbital masking
                onsite_hamiltonian, offsite_hamiltonian = self.apply_orbital_masks_to_hamiltonians(onsite_hamiltonian, offsite_hamiltonian, data)

                # For non-collinear spin without SOC, create block diagonal matrices
                if not self.collinear_spin:
                    soc_onsite_real = torch.zeros_like(data.Hon).reshape(onsite_hamiltonian.shape[0], 2*self.nao_max, 2*self.nao_max)
                    soc_onsite_real[:, :self.nao_max, :self.nao_max] = onsite_hamiltonian.reshape(-1, self.nao_max, self.nao_max)
                    soc_onsite_real[:, self.nao_max:, self.nao_max:] = onsite_hamiltonian.reshape(-1, self.nao_max, self.nao_max)
                    soc_onsite_real = soc_onsite_real.reshape(onsite_hamiltonian.shape[0], (2*self.nao_max)**2)

                    soc_offsite_real = torch.zeros_like(data.Hoff).reshape(offsite_hamiltonian.shape[0], 2*self.nao_max, 2*self.nao_max)
                    soc_offsite_real[:, :self.nao_max, :self.nao_max] = offsite_hamiltonian.reshape(-1, self.nao_max, self.nao_max)
                    soc_offsite_real[:, self.nao_max:, self.nao_max:] = offsite_hamiltonian.reshape(-1, self.nao_max, self.nao_max)
                    soc_offsite_real = soc_offsite_real.reshape(offsite_hamiltonian.shape[0], (2*self.nao_max)**2)

                    # Zero imaginary parts
                    soc_onsite_imag = torch.zeros_like(data.iHon)
                    soc_offsite_imag = torch.zeros_like(data.iHoff)

            # === MAGNETIC INTERACTION CALCULATION ===
            if self.spin_constrained:
                # Identify magnetic atoms
                magnetic_atoms = (data.spin_length > self.min_magnetic_moment)

                # Extract cell shift information
                data.unique_cell_shift, data.cell_shift_indices, data.cell_index_map = self.extract_unique_cell_vectors(data)
                cell_shift_indices = data.cell_shift_indices.tolist()
                cell_index_map = data.cell_index_map

                # Learn weight matrices if enabled
                if self.use_learned_weight:
                    # Calculate on-site weight matrix
                    node_spherical_harmonics = self.onsite_weight_network(node_attr)
                    node_spherical_components = torch.split(node_spherical_harmonics, self.hamiltonian_irreps_dimensions.tolist(), dim=-1)
                    weight_onsite = self.merge_tensor_components(node_spherical_components)

                    weight_onsite = self.reorder_matrix(weight_onsite)
                    weight_onsite = self.symmetrize_onsite_hamiltonian(weight_onsite)

                    # Calculate off-site weight matrix
                    edge_spherical_harmonics = self.offsite_weight_network(edge_attr)
                    edge_spherical_components = torch.split(edge_spherical_harmonics, self.hamiltonian_irreps_dimensions.tolist(), dim=-1)
                    weight_offsite = self.merge_tensor_components(edge_spherical_components)

                    weight_offsite = self.reorder_matrix(weight_offsite)
                    weight_offsite = self.symmetrize_offsite_hamiltonian(weight_offsite, inverse_edge_indices)

                    # Apply orbital masking to weights
                    weight_onsite, weight_offsite = self.apply_orbital_masks_to_hamiltonians(weight_onsite, weight_offsite, data)
                    weight_onsite = weight_onsite.reshape(-1, self.nao_max, self.nao_max)
                    weight_offsite = weight_offsite.reshape(-1, self.nao_max, self.nao_max)

                    # Store weights for later use
                    data.weight_on = weight_onsite
                    data.weight_off = weight_offsite

                # Calculate magnetic interaction terms
                if self.soc_switch:
                    # Calculate Heisenberg J coupling for SOC
                    J_onsite = self.onsite_J_network(node_attr)
                    J_onsite = self.construct_j_coupling_matrix(J_onsite)  # shape: (n_atoms, nao_max, nao_max, 3, 3)

                    J_offsite = self.offsite_J_network(edge_attr)
                    J_offsite = self.construct_j_coupling_matrix(J_offsite)  # shape: (n_edges, nao_max, nao_max, 3, 3)

                    # Calculate quartic terms if enabled
                    if self.add_quartic:
                        K_onsite = self.onsite_K_network(node_attr)
                        K_onsite = self.construct_k_coupling_matrix(K_onsite)  # shape: (n_atoms, nao_max, nao_max)

                        K_offsite = self.offsite_K_network(edge_attr)
                        K_offsite = self.construct_k_coupling_matrix(K_offsite)  # shape: (n_edges, nao_max, nao_max)

                    # Define Pauli matrices for spin operators
                    sigma = torch.view_as_complex(torch.zeros((3, 2, 2, 2)).type_as(J_onsite))
                    sigma[0] = torch.Tensor([[0.0, 1.0], [1.0, 0.0]]).type_as(sigma)  # ��x
                    sigma[1] = torch.complex(
                        real=torch.zeros((2, 2)), 
                        imag=torch.Tensor([[0.0, -1.0], [1.0, 0.0]])
                    ).type_as(sigma)  # ��y
                    sigma[2] = torch.Tensor([[1.0, 0.0], [0.0, -1.0]]).type_as(sigma)  # ��z

                    # Get spin vectors
                    spin_vec = data.spin_vec

                    # Initialize Heisenberg interaction terms
                    H_heisen_J_onsite = torch.zeros(len(J_onsite), 2, self.nao_max, 2, self.nao_max).type_as(sigma)
                    H_heisen_J_offsite = torch.zeros(len(source_indices), 2, self.nao_max, 2, self.nao_max).type_as(sigma)

                    # Optimize performance with edge lookup structures
                    edge_matcher_src, edge_matcher_tar = self.build_edge_lookup_structures(data, inverse_edge_indices)

                    # Calculate on-site Heisenberg terms for magnetic atoms
                    # S��J��S terms using opt_einsum for efficient contraction
                    H_heisen_J_onsite[magnetic_atoms] += oe.contract(
                        'mijkl,mij,kop,ml->moipj',
                        J_onsite[magnetic_atoms].type_as(sigma),
                        weight_onsite[magnetic_atoms].type_as(sigma),
                        sigma,
                        spin_vec[magnetic_atoms].type_as(sigma)
                    )

                    H_heisen_J_onsite[magnetic_atoms] += oe.contract(
                        'mijkl,mij,lop,mk->moipj',
                        J_onsite[magnetic_atoms].type_as(sigma),
                        weight_onsite[magnetic_atoms].type_as(sigma),
                        sigma,
                        spin_vec[magnetic_atoms].type_as(sigma)
                    )

                    # Calculate off-site interactions via neighboring atoms
                    zero_shift_idx = cell_index_map[(0, 0, 0)]
                    for atom_idx in range(len(J_onsite)):
                        # Process source atom contributions
                        if magnetic_atoms[atom_idx]:
                            # Get all edges where this atom appears, including zero-shift cells
                            zero_shift_edges = edge_matcher_tar[atom_idx][zero_shift_idx]
                            source_edges = torch.cat([edge_matcher_src[atom_idx], zero_shift_edges])
                            offsite_weights = weight_offsite[source_edges]

                            # Calculate Heisenberg interactions for these edges
                            H_heisen_J_offsite[source_edges] += oe.contract(
                                'ijkl,mij,kop,l->moipj',
                                J_onsite[atom_idx].type_as(sigma),
                                offsite_weights.type_as(sigma),
                                sigma,
                                spin_vec[atom_idx].type_as(sigma)
                            )

                            H_heisen_J_offsite[source_edges] += oe.contract(
                                'ijkl,mij,lop,k->moipj',
                                J_onsite[atom_idx].type_as(sigma),
                                offsite_weights.type_as(sigma),
                                sigma,
                                spin_vec[atom_idx].type_as(sigma)
                            )

                    # Process edge-specific interactions
                    for edge_idx in range(len(source_indices)):
                        source_atom = source_indices[edge_idx].item()
                        target_atom = target_indices[edge_idx].item()

                        # Target atom contribution to source
                        if magnetic_atoms[target_atom]:
                            onsite_weight = weight_onsite[source_atom]
                            source_offsite_weights = weight_offsite[edge_matcher_src[source_atom]]

                            # Update on-site term at source atom
                            H_heisen_J_onsite[source_atom] += oe.contract(
                                'ijkl,ij,kop,l->oipj',
                                J_offsite[edge_idx].type_as(sigma),
                                onsite_weight.type_as(sigma),
                                sigma,
                                spin_vec[target_atom].type_as(sigma)
                            )

                            # Update off-site terms from source atom
                            H_heisen_J_offsite[edge_matcher_src[source_atom]] += oe.contract(
                                'ijkl,mij,kop,l->moipj',
                                J_offsite[edge_idx].type_as(sigma),
                                source_offsite_weights.type_as(sigma),
                                sigma,
                                spin_vec[target_atom].type_as(sigma)
                            )

                        # Source atom contribution to target
                        if magnetic_atoms[source_atom]:
                            target_offsite_weights = weight_offsite[edge_matcher_tar[target_atom][cell_shift_indices[edge_idx]]]

                            # Update off-site terms to target atom
                            H_heisen_J_offsite[edge_matcher_tar[target_atom][cell_shift_indices[edge_idx]]] += oe.contract(
                                'ijkl,mij,lop,k->moipj',
                                J_offsite[edge_idx].type_as(sigma),
                                target_offsite_weights.type_as(sigma),
                                sigma,
                                spin_vec[source_atom].type_as(sigma)
                            )

                            # For zero-cell-shift, also update on-site term at target
                            if cell_shift_indices[edge_idx] == data.cell_index_map[(0, 0, 0)]:
                                target_onsite_weight = weight_onsite[target_atom]
                                H_heisen_J_onsite[target_atom] += oe.contract(
                                    'ijkl,ij,lop,k->oipj',
                                    J_offsite[edge_idx].type_as(sigma),
                                    target_onsite_weight.type_as(sigma),
                                    sigma,
                                    spin_vec[source_atom].type_as(sigma)
                                )
                else:
                    # Non-SOC magnetic interactions
                    J_onsite = self.onsite_J_network(node_attr)
                    J_onsite = self.construct_j_coupling_matrix(J_onsite)  # shape: (n_atoms, nao_max, nao_max)

                    J_offsite = self.offsite_J_network(edge_attr)
                    J_offsite = self.construct_j_coupling_matrix(J_offsite)  # shape: (n_edges, nao_max, nao_max)

                    # Calculate quartic terms if enabled
                    if self.add_quartic:
                        K_onsite = self.onsite_K_network(node_attr)
                        K_onsite = self.construct_k_coupling_matrix(K_onsite)

                        K_offsite = self.offsite_K_network(edge_attr)
                        K_offsite = self.construct_k_coupling_matrix(K_offsite)

                    # Handle collinear vs non-collinear spin
                    if self.collinear_spin:
                        # Collinear spin uses only z-component
                        sigma_z = torch.Tensor([[1.0, 0.0], [0.0, -1.0]]).type_as(J_onsite)
                        spin_vec = data.spin_vec

                        # Initialize Heisenberg interaction matrices
                        H_heisen_J_onsite = torch.zeros(len(J_onsite), 2, self.nao_max, 2, self.nao_max).type_as(J_onsite)
                        H_heisen_J_offsite = torch.zeros(len(source_indices), 2, self.nao_max, 2, self.nao_max).type_as(J_offsite)

                        # Build edge lookup for efficient calculation
                        edge_matcher_src, edge_matcher_tar = self.build_edge_lookup_structures(data, inverse_edge_indices)

                        # Calculate on-site contributions for magnetic atoms
                        H_heisen_J_onsite[magnetic_atoms] += oe.contract(
                            'mij,mij,op,m->moipj',
                            J_onsite[magnetic_atoms],
                            weight_onsite[magnetic_atoms],
                            sigma_z,
                            spin_vec[magnetic_atoms, 2]  # z-component only
                        )

                        # Calculate off-site contributions
                        zero_shift_idx = cell_index_map[(0, 0, 0)]
                        for atom_idx in range(len(J_onsite)):
                            if magnetic_atoms[atom_idx]:
                                zero_shift_edges = edge_matcher_tar[atom_idx][zero_shift_idx]
                                source_edges = torch.cat([edge_matcher_src[atom_idx], zero_shift_edges])
                                offsite_weights = weight_offsite[source_edges]

                                H_heisen_J_offsite[source_edges] += oe.contract(
                                    'ij,mij,op->moipj',
                                    J_onsite[atom_idx],
                                    offsite_weights,
                                    sigma_z
                                ) * spin_vec[atom_idx, 2]

                        # Process edge-specific interactions
                        for edge_idx in range(len(source_indices)):
                            source_atom = source_indices[edge_idx].item()
                            target_atom = target_indices[edge_idx].item()

                            # Target atom contribution to source
                            if magnetic_atoms[target_atom]:
                                onsite_weight = weight_onsite[source_atom]
                                source_offsite_weights = weight_offsite[edge_matcher_src[source_atom]]

                                H_heisen_J_onsite[source_atom] += oe.contract(
                                    'ij,ij,op->oipj',
                                    J_offsite[edge_idx],
                                    onsite_weight,
                                    sigma_z
                                ) * spin_vec[target_atom, 2]

                                H_heisen_J_offsite[edge_matcher_src[source_atom]] += oe.contract(
                                    'ij,mij,op->moipj',
                                    J_offsite[edge_idx],
                                    source_offsite_weights,
                                    sigma_z
                                ) * spin_vec[target_atom, 2]

                            # Source atom contribution to target
                            if magnetic_atoms[source_atom]:
                                target_offsite_weights = weight_offsite[edge_matcher_tar[target_atom][cell_shift_indices[edge_idx]]]

                                H_heisen_J_offsite[edge_matcher_tar[target_atom][cell_shift_indices[edge_idx]]] += oe.contract(
                                    'ij,mij,op->moipj',
                                    J_offsite[edge_idx],
                                    target_offsite_weights,
                                    sigma_z
                                ) * spin_vec[source_atom, 2]

                                if cell_shift_indices[edge_idx] == data.cell_index_map[(0, 0, 0)]:
                                    target_onsite_weight = weight_onsite[target_atom]
                                    H_heisen_J_onsite[target_atom] += oe.contract(
                                        'ij,ij,op->oipj',
                                        J_offsite[edge_idx],
                                        target_onsite_weight,
                                        sigma_z
                                    ) * spin_vec[source_atom, 2]
                    else:
                        # Non-collinear spin handling (similar to SOC case but simpler)
                        # Define Pauli matrices
                        sigma = torch.view_as_complex(torch.zeros((3, 2, 2, 2)).type_as(J_onsite))
                        sigma[0] = torch.Tensor([[0.0, 1.0], [1.0, 0.0]]).type_as(sigma)
                        sigma[1] = torch.complex(
                            real=torch.zeros((2, 2)),
                            imag=torch.Tensor([[0.0, -1.0], [1.0, 0.0]])
                        ).type_as(sigma)
                        sigma[2] = torch.Tensor([[1.0, 0.0], [0.0, -1.0]]).type_as(sigma)

                        spin_vec = data.spin_vec

                        # Initialize Heisenberg matrices
                        H_heisen_J_onsite = torch.zeros(len(J_onsite), 2, self.nao_max, 2, self.nao_max).type_as(sigma)
                        H_heisen_J_offsite = torch.zeros(len(source_indices), 2, self.nao_max, 2, self.nao_max).type_as(sigma)

                        # Build edge lookup
                        edge_matcher_src, edge_matcher_tar = self.build_edge_lookup_structures(data, inverse_edge_indices)

                        # Calculate on-site contributions for magnetic atoms
                        H_heisen_J_onsite[magnetic_atoms] += oe.contract(
                            'mij,mij,kop,mk->moipj',
                            J_onsite[magnetic_atoms].type_as(sigma),
                            weight_onsite[magnetic_atoms].type_as(sigma),
                            sigma,
                            spin_vec[magnetic_atoms].type_as(sigma)
                        )

                        # Calculate off-site contributions
                        zero_shift_idx = cell_index_map[(0, 0, 0)]
                        for atom_idx in range(len(J_onsite)):
                            if magnetic_atoms[atom_idx]:
                                zero_shift_edges = edge_matcher_tar[atom_idx][zero_shift_idx]
                                source_edges = torch.cat([edge_matcher_src[atom_idx], zero_shift_edges])
                                offsite_weights = weight_offsite[source_edges]

                                H_heisen_J_offsite[source_edges] += oe.contract(
                                    'ij,mij,kop,k->moipj',
                                    J_onsite[atom_idx].type_as(sigma),
                                    offsite_weights.type_as(sigma),
                                    sigma,
                                    spin_vec[atom_idx].type_as(sigma)
                                )

                        # Process edge-specific interactions
                        for edge_idx in range(len(source_indices)):
                            source_atom = source_indices[edge_idx].item()
                            target_atom = target_indices[edge_idx].item()

                            # Target atom contribution to source
                            if magnetic_atoms[target_atom]:
                                onsite_weight = weight_onsite[source_atom]
                                source_offsite_weights = weight_offsite[edge_matcher_src[source_atom]]

                                H_heisen_J_onsite[source_atom] += oe.contract(
                                    'ij,ij,kop,k->oipj',
                                    J_offsite[edge_idx].type_as(sigma),
                                    onsite_weight.type_as(sigma),
                                    sigma,
                                    spin_vec[target_atom].type_as(sigma)
                                )

                                H_heisen_J_offsite[edge_matcher_src[source_atom]] += oe.contract(
                                    'ij,mij,kop,k->moipj',
                                    J_offsite[edge_idx].type_as(sigma),
                                    source_offsite_weights.type_as(sigma),
                                    sigma,
                                    spin_vec[target_atom].type_as(sigma)
                                )

                            # Source atom contribution to target
                            if magnetic_atoms[source_atom]:
                                target_offsite_weights = weight_offsite[edge_matcher_tar[target_atom][cell_shift_indices[edge_idx]]]

                                H_heisen_J_offsite[edge_matcher_tar[target_atom][cell_shift_indices[edge_idx]]] += oe.contract(
                                    'ij,mij,kop,k->moipj',
                                    J_offsite[edge_idx].type_as(sigma),
                                    target_offsite_weights.type_as(sigma),
                                    sigma,
                                    spin_vec[source_atom].type_as(sigma)
                                )

                                if cell_shift_indices[edge_idx] == data.cell_index_map[(0, 0, 0)]:
                                    target_onsite_weight = weight_onsite[target_atom]
                                    H_heisen_J_onsite[target_atom] += oe.contract(
                                        'ij,ij,kop,k->oipj',
                                        J_offsite[edge_idx].type_as(sigma),
                                        target_onsite_weight.type_as(sigma),
                                        sigma,
                                        spin_vec[source_atom].type_as(sigma)
                                    )

                # Combine Heisenberg terms with SOC Hamiltonian
                if not self.collinear_spin:
                    # Add real and imaginary parts separately
                    soc_onsite_real = soc_onsite_real + H_heisen_J_onsite.reshape(-1, (2*self.nao_max)**2).real
                    soc_offsite_real = soc_offsite_real + H_heisen_J_offsite.reshape(-1, (2*self.nao_max)**2).real
                    soc_onsite_imag = soc_onsite_imag + H_heisen_J_onsite.reshape(-1, (2*self.nao_max)**2).imag
                    soc_offsite_imag = soc_offsite_imag + H_heisen_J_offsite.reshape(-1, (2*self.nao_max)**2).imag

                    # Apply symmetrization if enabled
                    if self.symmetrize:
                        soc_onsite_real = self.symmetrize_onsite_hamiltonian_soc(soc_onsite_real, hermitian=True)
                        soc_offsite_real = self.symmetrize_offsite_hamiltonian_soc(soc_offsite_real, inverse_edge_indices, hermitian=True)
                        soc_onsite_imag = self.symmetrize_onsite_hamiltonian_soc(soc_onsite_imag, hermitian=False)
                        soc_offsite_imag = self.symmetrize_offsite_hamiltonian_soc(soc_offsite_imag, inverse_edge_indices, hermitian=False)
                else:
                    # For collinear spin, create separate up and down Hamiltonians
                    collinear_onsite = torch.stack([
                        onsite_hamiltonian.reshape(-1, self.nao_max, self.nao_max) + H_heisen_J_onsite[:, 0, :, 0, :],
                        onsite_hamiltonian.reshape(-1, self.nao_max, self.nao_max) + H_heisen_J_onsite[:, 1, :, 1, :]
                    ], dim=1).reshape(-1, 2, (self.nao_max)**2)

                    collinear_offsite = torch.stack([
                        offsite_hamiltonian.reshape(-1, self.nao_max, self.nao_max) + H_heisen_J_offsite[:, 0, :, 0, :],
                        offsite_hamiltonian.reshape(-1, self.nao_max, self.nao_max) + H_heisen_J_offsite[:, 1, :, 1, :]
                    ], dim=1).reshape(-1, 2, (self.nao_max)**2)

            # Add reference H0 terms if enabled
            if self.add_H0:
                if not self.collinear_spin:
                    soc_onsite_real = soc_onsite_real + data.Hon0
                    soc_offsite_real = soc_offsite_real + data.Hoff0
                    soc_onsite_imag = soc_onsite_imag + data.iHon0
                    soc_offsite_imag = soc_offsite_imag + data.iHoff0
                else:
                    collinear_onsite = collinear_onsite + data.Hon0
                    collinear_offsite = collinear_offsite + data.Hoff0

            # === COMBINE HAMILTONIANS AND CALCULATE BAND STRUCTURES ===
            if not self.collinear_spin:
                # Combine real and imaginary parts of SOC Hamiltonian
                soc_hamiltonian_real = self.concatenate_hamiltonians_by_crystal(data, soc_onsite_real, soc_offsite_real)
                soc_hamiltonian_imag = self.concatenate_hamiltonians_by_crystal(data, soc_onsite_imag, soc_offsite_imag)

                # Store reference Hamiltonians
                data.hamiltonian_real = self.concatenate_hamiltonians_by_crystal(data, data.Hon, data.Hoff)
                data.hamiltonian_imag = self.concatenate_hamiltonians_by_crystal(data, data.iHon, data.iHoff)

                # Combine real and imaginary parts
                soc_hamiltonian = torch.cat((soc_hamiltonian_real, soc_hamiltonian_imag), dim=0)
                data.hamiltonian = torch.cat((data.hamiltonian_real, data.hamiltonian_imag), dim=0)

                # Calculate band structure if requested
                if self.calculate_band_energy:
                    # Generate k-points for each crystal
                    k_vectors = []
                    for idx in range(data.batch[-1] + 1):
                        lattice_vectors = data.cell

                        # Generate k-point path based on configuration
                        if self.k_path is not None:
                            kpoints = kpoints_generator(dim_k=3, lat=lattice_vectors[idx].detach().cpu().numpy())
                            k_vec, k_dist, k_node, lat_per_inv = kpoints.k_path(self.k_path, self.num_k)
                        else:
                            # Random k-points if no path specified
                            lat_per_inv = np.linalg.inv(lattice_vectors[idx].detach().cpu().numpy()).T
                            k_vec = 2.0 * np.random.rand(self.num_k, 3) - 1.0  # (-1, 1)

                        # Transform k-points to reciprocal space
                        k_vec = k_vec.dot(lat_per_inv[np.newaxis, :, :])  # shape (nk, 1, 3)
                        k_vec = k_vec.reshape(-1, 3)  # shape (nk, 3)
                        k_vec = torch.Tensor(k_vec).type_as(onsite_hamiltonian)
                        k_vectors.append(k_vec)

                    # Store k-vectors for band structure calculation
                    data.k_vecs = torch.stack(k_vectors, dim=0)

                    # Calculate band energies and wavefunctions with SOC
                    band_energy, wavefunction = self.calculate_band_energies_with_spin_orbit_coupling(
                        soc_onsite_real, soc_onsite_imag, soc_offsite_real, soc_offsite_imag, data
                    )

                    # Calculate reference band structure
                    with torch.no_grad():
                        data.band_energy, data.wavefunction = self.calculate_band_energies_with_spin_orbit_coupling(
                            data.Hon, data.iHon, data.Hoff, data.iHoff, data
                        )
                else:
                    band_energy = None
                    wavefunction = None
            else:
                # For collinear spin or non-SOC case
                collinear_hamiltonian = self.concatenate_hamiltonians_by_crystal(data, collinear_onsite, collinear_offsite)
                data.hamiltonian = self.concatenate_hamiltonians_by_crystal(data, data.Hon, data.Hoff)

                # Calculate band structure if requested
                if self.calculate_band_energy:
                    k_vectors = []
                    for idx in range(data.batch[-1] + 1):
                        lattice_vectors = data.cell

                        # Generate k-point path based on configuration
                        if isinstance(self.k_path, list):
                            kpoints = kpoints_generator(dim_k=3, lat=lattice_vectors[idx].detach().cpu().numpy())
                            k_vec, k_dist, k_node, lat_per_inv = kpoints.k_path(self.k_path, self.num_k)
                        elif isinstance(self.k_path, str) and self.k_path.lower() == 'auto':
                            # Automatic k-path generation based on crystal symmetry
                            lattice = lattice_vectors[idx].detach().cpu().numpy() * au2ang
                            positions = torch.split(data.pos, data.node_counts.tolist(), dim=0)[idx].detach().cpu().numpy() * au2ang
                            species = torch.split(data.z, data.node_counts.tolist(), dim=0)[idx]

                            # Create pymatgen Structure
                            structure = Structure(
                                lattice=lattice, 
                                species=[Element.from_Z(k.item()).symbol for k in species], 
                                coords=positions, 
                                coords_are_cartesian=True
                            )

                            # Generate k-path using symmetry
                            kpath_seek = KPathSeek(structure=structure)
                            k_labels = []
                            for label_group in kpath_seek.kpath['path']:
                                k_labels += label_group

                            # Remove adjacent duplicates
                            unique_labels = [k_labels[0]]
                            [unique_labels.append(x) for x in k_labels[1:] if x != unique_labels[-1]]

                            k_path = [kpath_seek.kpath['kpoints'][k] for k in unique_labels]

                            try:
                                kpoints = kpoints_generator(dim_k=3, lat=lattice_vectors[idx].detach().cpu().numpy())
                                k_vec, k_dist, k_node, lat_per_inv = kpoints.k_path(k_path, self.num_k)
                            except:
                                # Fallback to random k-points if path generation fails
                                lat_per_inv = np.linalg.inv(lattice_vectors[idx].detach().cpu().numpy()).T
                                k_vec = 2.0 * np.random.rand(self.num_k, 3) - 1.0  # (-1, 1)
                        else:
                            # Random k-points if no path specified
                            lat_per_inv = np.linalg.inv(lattice_vectors[idx].detach().cpu().numpy()).T
                            k_vec = 2.0 * np.random.rand(self.num_k, 3) - 1.0  # (-1, 1)

                        # Transform k-points to reciprocal space
                        k_vec = k_vec.dot(lat_per_inv[np.newaxis, :, :])
                        k_vec = k_vec.reshape(-1, 3)
                        k_vec = torch.Tensor(k_vec).type_as(onsite_hamiltonian)
                        k_vectors.append(k_vec)

                    # Store k-vectors for band structure calculation
                    data.k_vecs = torch.stack(k_vectors, dim=0)

                    if self.export_reciprocal_values:
                        # Calculate band structure for both spin channels with reciprocal space matrices
                        band_energy_up, wavefunction_up, HK_up, SK_up, dSK_up, gap_up = self.calculate_band_energies(
                            collinear_onsite[:, 0, :], collinear_offsite[:, 0, :], data, True
                        )
                        band_energy_down, wavefunction_down, HK_down, SK_down, dSK_down, gap_down = self.calculate_band_energies(
                            collinear_onsite[:, 1, :], collinear_offsite[:, 1, :], data, True
                        )

                        H_sym = None
                        band_energy = torch.cat([band_energy_up, band_energy_down])
                        wavefunction = torch.cat([wavefunction_up, wavefunction_down])
                        HK = torch.cat([HK_up, HK_down])
                        gap = torch.cat([gap_up, gap_down])
                    else:
                        # Calculate band structure for both spin channels
                        band_energy_up, wavefunction_up, gap_up, H_sym = self.calculate_band_energies(
                            collinear_onsite[:, 0, :], collinear_offsite[:, 0, :], data
                        )
                        band_energy_down, wavefunction_down, gap_down, H_sym = self.calculate_band_energies(
                            collinear_onsite[:, 1, :], collinear_offsite[:, 1, :], data
                        )

                        band_energy = torch.cat([band_energy_up, band_energy_down])
                        wavefunction = torch.cat([wavefunction_up, wavefunction_down])
                        gap = torch.cat([gap_up, gap_down])

                    # Calculate reference band structure
                    with torch.no_grad():
                        data.band_energy_up, data.wavefunction, data.band_gap_up, data.H_sym = self.calculate_band_energies(
                            data.Hon[:, 0, :], data.Hoff[:, 0, :], data
                        )
                        data.band_energy_down, data.wavefunction, data.band_gap_down, data.H_sym = self.calculate_band_energies(
                            data.Hon[:, 1, :], data.Hoff[:, 1, :], data
                        )
                        data.band_energy = torch.cat([data.band_energy_up, data.band_energy_down])
                        data.band_gap = torch.cat([data.band_gap_up, data.band_gap_down])
                else:
                    band_energy = None
                    wavefunction = None
                    gap = None
                    H_sym = None

        # === NON-MAGNETIC, NON-SOC CASE ===
        else:
            # Calculate standard Hamiltonian
            node_spherical_harmonics = self.onsite_hamiltonian_network(node_attr)
            node_spherical_components = torch.split(node_spherical_harmonics, self.hamiltonian_irreps_dimensions.tolist(), dim=-1)
            onsite_hamiltonian = self.merge_tensor_components(node_spherical_components)

            onsite_hamiltonian = self.reorder_matrix(onsite_hamiltonian)
            onsite_hamiltonian = self.symmetrize_onsite_hamiltonian(onsite_hamiltonian)

            # Add reference H0 if available
            if self.add_H0:
                onsite_hamiltonian = onsite_hamiltonian + data.Hon0

            # Calculate off-site Hamiltonian
            edge_spherical_harmonics = self.offsite_hamiltonian_network(edge_attr)
            edge_spherical_components = torch.split(edge_spherical_harmonics, self.hamiltonian_irreps_dimensions.tolist(), dim=-1)
            offsite_hamiltonian = self.merge_tensor_components(edge_spherical_components)

            offsite_hamiltonian = self.reorder_matrix(offsite_hamiltonian)
            offsite_hamiltonian = self.symmetrize_offsite_hamiltonian(offsite_hamiltonian, inverse_edge_indices)

            # Add reference H0 if available
            if self.add_H0:
                offsite_hamiltonian = offsite_hamiltonian + data.Hoff0
                
            # Apply orbital masking for specific Hamiltonian types
            if self.ham_type in ['openmx', 'pasp', 'siesta', 'abacus']:
                onsite_hamiltonian, offsite_hamiltonian = self.apply_orbital_masks_to_hamiltonians(onsite_hamiltonian, offsite_hamiltonian, data)

            # Calculate band structure if requested
            if self.calculate_band_energy:
                # Generate k-points for each crystal
                k_vectors = []
                for idx in range(data.batch[-1] + 1):
                    lattice_vectors = data.cell

                    # Generate k-point path based on configuration
                    if isinstance(self.k_path, list):
                        kpoints = kpoints_generator(dim_k=3, lat=lattice_vectors[idx].detach().cpu().numpy())
                        k_vec, k_dist, k_node, lat_per_inv = kpoints.k_path(self.k_path, self.num_k)
                    elif isinstance(self.k_path, str) and self.k_path.lower() == 'auto':
                        # Automatic k-path generation
                        lattice = lattice_vectors[idx].detach().cpu().numpy() * au2ang
                        positions = torch.split(data.pos, data.node_counts.tolist(), dim=0)[idx].detach().cpu().numpy() * au2ang
                        species = torch.split(data.z, data.node_counts.tolist(), dim=0)[idx]

                        structure = Structure(
                            lattice=lattice,
                            species=[Element.from_Z(k.item()).symbol for k in species],
                            coords=positions,
                            coords_are_cartesian=True
                        )

                        kpath_seek = KPathSeek(structure=structure)
                        k_labels = []
                        for label_group in kpath_seek.kpath['path']:
                            k_labels += label_group

                        unique_labels = [k_labels[0]]
                        [unique_labels.append(x) for x in k_labels[1:] if x != unique_labels[-1]]

                        k_path = [kpath_seek.kpath['kpoints'][k] for k in unique_labels]

                        try:
                            kpoints = kpoints_generator(dim_k=3, lat=lattice_vectors[idx].detach().cpu().numpy())
                            k_vec, k_dist, k_node, lat_per_inv = kpoints.k_path(k_path, self.num_k)
                        except:
                            # Fallback to random k-points
                            lat_per_inv = np.linalg.inv(lattice_vectors[idx].detach().cpu().numpy()).T
                            k_vec = 2.0 * np.random.rand(self.num_k, 3) - 1.0
                    else:
                        # Random k-points
                        lat_per_inv = np.linalg.inv(lattice_vectors[idx].detach().cpu().numpy()).T
                        k_vec = 2.0 * np.random.rand(self.num_k, 3) - 1.0

                    # Transform k-points to reciprocal space
                    k_vec = k_vec.dot(lat_per_inv[np.newaxis, :, :])
                    k_vec = k_vec.reshape(-1, 3)
                    k_vec = torch.Tensor(k_vec).type_as(onsite_hamiltonian)
                    k_vectors.append(k_vec)

                # Store k-vectors for band structure calculation
                data.k_vecs = torch.stack(k_vectors, dim=0)

                if self.export_reciprocal_values:
                    if self.ham_only:
                        # Calculate band structure with reciprocal space matrices
                        band_energy, wavefunction, HK, SK, dSK, gap = self.calculate_band_energies(
                            onsite_hamiltonian, offsite_hamiltonian, data, True
                        )
                        H_sym = None
                    else:
                        # Calculate band structure with overlap matrices
                        band_energy, wavefunction, HK, SK, dSK, gap = self.calculate_band_energies_with_overlap(
                            onsite_hamiltonian, offsite_hamiltonian, onsite_overlap, offsite_overlap, data, True
                        )
                        H_sym = None
                else:
                    # Standard band structure calculation
                    band_energy, wavefunction, gap, H_sym = self.calculate_band_energies(
                        onsite_hamiltonian, offsite_hamiltonian, data
                    )

                # Calculate reference band structure
                with torch.no_grad():
                    data.band_energy, data.wavefunction, data.band_gap, data.H_sym = self.calculate_band_energies(
                        data.Hon, data.Hoff, data
                    )
            else:
                band_energy = None
                wavefunction = None
                gap = None
                H_sym = None

        # === PREPARE RESULTS ===
        # Process results based on Hamiltonian type
        if self.ham_type in ['openmx', 'pasp', 'siesta', 'abacus']:
            if self.soc_switch or self.spin_constrained:
                if not self.collinear_spin:
                    # Handle zero point energy shift if enabled
                    if self.zero_point_shift:
                        # Calculate energy shift to match reference
                        overlap_matrix = data.overlap.reshape(-1, self.nao_max, self.nao_max)
                        soc_hamiltonian_real_reshaped = soc_hamiltonian_real.reshape(-1, 2, self.nao_max, 2, self.nao_max)
                        real_up_up_block = soc_hamiltonian_real_reshaped[:, 0, :, 0, :]  
                        real_down_down_block = soc_hamiltonian_real_reshaped[:, 1, :, 1, :] 
                        
                        data_hamiltonian_real_reshaped = data.hamiltonian_real.reshape(-1, 2, self.nao_max, 2, self.nao_max)
                        data_real_up_up_block = data_hamiltonian_real_reshaped[:, 0, :, 0, :]  
                        data_real_down_down_block = data_hamiltonian_real_reshaped[:, 1, :, 1, :]  
                        
                        sum_overlap = torch.sum(overlap_matrix[overlap_matrix > 1e-6])
                        
                        diagonal_block_difference = (real_up_up_block + real_down_down_block) - (data_real_up_up_block + data_real_down_down_block)
                        
                        energy_shift_real = torch.sum(extract_elements_above_threshold(
                            overlap_matrix, diagonal_block_difference , 1e-6
                        )) / (2.0*sum_overlap)

                        # Apply energy shift
                        soc_hamiltonian_real_reshaped[:, 0, :, 0, :] = real_up_up_block - energy_shift_real * overlap_matrix
                        soc_hamiltonian_real_reshaped[:, 1, :, 1, :] = real_down_down_block - energy_shift_real * overlap_matrix
                        soc_hamiltonian_real = soc_hamiltonian_real_reshaped.reshape(-1, (2*self.nao_max)**2)
                        
                        soc_hamiltonian = torch.cat((soc_hamiltonian_real, soc_hamiltonian_imag), dim=0)

                        # Adjust band energies if calculated
                        if band_energy is not None:
                            band_energy = band_energy - torch.mean(band_energy - data.band_energy)

                    # Prepare result dictionary
                    result = {
                        'hamiltonian': soc_hamiltonian,
                        'hamiltonian_real': soc_hamiltonian_real,
                        'hamiltonian_imag': soc_hamiltonian_imag,
                        'band_energy': band_energy,
                        'wavefunction': wavefunction
                    }

                    # Add mask tensor if requested
                    if self.get_nonzero_mask_tensor:
                        mask_real_imag, mask_all = self.build_spin_orbit_interaction_masks(data)
                        result['mask_real_imag'] = mask_real_imag

                else:  # Collinear spin
                    # Handle zero point energy shift if enabled
                    if self.zero_point_shift:
                        # Calculate energy shift to match reference
                        overlap_matrix = data.overlap
                        collinear_overlap = torch.stack([overlap_matrix, overlap_matrix], dim=1)
                        sum_collinear_overlap = 2 * torch.sum(overlap_matrix[overlap_matrix > 1e-6])

                        energy_shift = torch.sum(extract_elements_above_threshold(
                            collinear_overlap, collinear_hamiltonian - data.hamiltonian, 1e-6
                        )) / sum_collinear_overlap

                        # Apply energy shift
                        collinear_hamiltonian = collinear_hamiltonian - energy_shift * collinear_overlap

                        # Adjust band energies if calculated
                        if band_energy is not None:
                            band_energy = band_energy - torch.mean(band_energy - data.band_energy)

                    # Prepare result dictionary
                    result = {
                        'hamiltonian': collinear_hamiltonian,
                        'band_energy': band_energy,
                        'wavefunction': wavefunction
                    }

                    # Add mask tensor if requested
                    if self.get_nonzero_mask_tensor:
                        mask_all = self.build_column_wise_interaction_masks(data)
                        result['mask'] = mask_all
            else:
                # Standard non-magnetic case
                combined_hamiltonian = self.concatenate_hamiltonians_by_crystal(data, onsite_hamiltonian, offsite_hamiltonian)

                # Handle zero point energy shift if enabled
                if self.zero_point_shift:
                    # Calculate energy shift to match reference
                    overlap_matrix = data.overlap
                    sum_overlap = torch.sum(overlap_matrix[overlap_matrix > 1e-6])

                    energy_shift = torch.sum(extract_elements_above_threshold(
                        overlap_matrix, combined_hamiltonian - data.hamiltonian, 1e-6
                    )) / sum_overlap

                    # Apply energy shift
                    combined_hamiltonian = combined_hamiltonian - energy_shift * data.overlap

                    # Adjust band energies if calculated
                    if band_energy is not None:
                        band_energy = band_energy - torch.mean(band_energy - data.band_energy)

                # Prepare result dictionary
                result = {
                    'hamiltonian': combined_hamiltonian,
                    'band_energy': band_energy,
                    'wavefunction': wavefunction,
                    'band_gap': gap,
                    'H_sym': H_sym
                }

                # Add reciprocal space matrices if requested
                if self.export_reciprocal_values:
                    result.update({'HK': HK, 'SK': SK, 'dSK': dSK})

                # Add mask tensor if requested
                if self.get_nonzero_mask_tensor:
                    mask_all = self.build_interaction_masks(data)
                    result['mask'] = mask_all

        else:
            raise NotImplementedError("Unsupported Hamiltonian type")

        # Add overlap matrix if calculating both Hamiltonian and overlap
        if not self.ham_only:
            if self.ham_type in ['openmx', 'pasp', 'siesta', 'abacus']:
                overlap = self.concatenate_hamiltonians_by_crystal(data, onsite_overlap, offsite_overlap)
            else:
                raise NotImplementedError("Unsupported Hamiltonian type for overlap calculation")

            result.update({'overlap': overlap})
        
        # Calculate and add sparsity ratio if requested
        if self.calculate_sparsity:
            result['sparsity_ratio'] = self.calculate_sparsity_ratio(data)
        
        return result
