

from numpy import zeros
import torch
import torch.nn as nn
from torch_geometric.data import Data, batch
from torch.nn import (Bilinear, Sigmoid, Softplus, ELU, ReLU, SELU, SiLU,
                      CELU, BatchNorm1d, ModuleList, Sequential, Tanh, Identity)
from ..utils import linear_bn_act
from ..layers import denseRegression
from torch_scatter import scatter
import sympy as sym
from e3nn import o3
from e3nn.o3 import Linear
from e3nn.nn import FullyConnectedNet
from e3nn.o3 import TensorProduct, Linear, FullyConnectedTensorProduct
from e3nn.nn import Gate, NormActivation
from easydict import EasyDict
from typing import Union
from ..layers import GaussianSmearing, cuttoff_envelope, CosineCutoff, BesselBasis, sph_harm_layer
from .nequip.data import AtomicDataDict, AtomicDataset
import math
from .clebsch_gordan import ClebschGordan
# from e3nn.o3._wigner import _so3_clebsch_gordan
import copy
from typing import Dict, Callable
from .nequip.nn.nonlinearities import ShiftedSoftPlus
from .kpoint_gen import kpoints_generator
import numpy as np
from pymatgen.core.periodic_table import Element
from torch_sparse import SparseTensor
from pymatgen.core.structure import Structure
from pymatgen.symmetry.kpath import KPathSeek
from ..e3_layers import e3TensorDecomp

au2ang = 0.5291772083

acts = {
    "abs": torch.abs,
    "tanh": torch.tanh,
    "ssp": ShiftedSoftPlus,
    "silu": torch.nn.functional.silu,
}

from .nequip.nn import (
    GraphModuleMixin,
    SequentialGraphNetwork,
    AtomwiseLinear,
    AtomwiseReduce,
    ConvNetLayer,
)
from .nequip.nn.embedding import (
    OneHotAtomEncoding,
    RadialBasisEdgeEncoding,
    SphericalHarmonicEdgeAttrs,
    Embedding_block,
    Embedding_block_q
)


class residual_block(torch.nn.Module):
    """
    Args:

    """

    def __init__(
        self,
        irreps_in,
        feature_irreps_hidden,
        resnet: bool = False,
        nonlinearity_type: str = "gate",
        nonlinearity_scalars: Dict[int, Callable] = {"e": "ssp", "o": "tanh"},
        nonlinearity_gates: Dict[int, Callable] = {"e": "ssp", "o": "abs"},
    ):
        super().__init__()
        # initialization
        assert nonlinearity_type in ("gate", "norm")
        # make the nonlin dicts from parity ints instead of convinience strs
        nonlinearity_scalars = {
            1: nonlinearity_scalars["e"],
            -1: nonlinearity_scalars["o"],
        }
        nonlinearity_gates = {
            1: nonlinearity_gates["e"],
            -1: nonlinearity_gates["o"],
        }

        self.irreps_in = o3.Irreps(irreps_in)
        self.feature_irreps_hidden = o3.Irreps(feature_irreps_hidden)
        self.resnet = resnet

        #irreps_layer_out_prev = self.feature_irreps_hidden

        irreps_scalars = o3.Irreps(
            [
                (mul, ir)
                for mul, ir in self.feature_irreps_hidden
                if ir.l == 0 and ir in self.irreps_in
            ]
        )

        irreps_gated = o3.Irreps(
            [
                (mul, ir)
                for mul, ir in self.feature_irreps_hidden
                if ir.l > 0 and ir in self.irreps_in
            ]
        )

        self.irreps_layer_out = (irreps_scalars + irreps_gated).simplify()

        if nonlinearity_type == "gate":
            ir = (
                "0e"
                if o3.Irrep("0e") in self.feature_irreps_hidden
                else "0o"
            )
            irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated])

            # TO DO, it's not that safe to directly use the
            # dictionary
            equivariant_nonlin = Gate(
                irreps_scalars=irreps_scalars,
                act_scalars=[
                    acts[nonlinearity_scalars[ir.p]] for _, ir in irreps_scalars
                ],
                irreps_gates=irreps_gates,
                act_gates=[acts[nonlinearity_gates[ir.p]] for _, ir in irreps_gates],
                irreps_gated=irreps_gated,
            )

            linear_irreps_out = equivariant_nonlin.irreps_in.simplify()

        else:
            linear_irreps_out = self.irreps_layer_out.simplify()

            equivariant_nonlin = NormActivation(
                irreps_in=linear_irreps_out,
                # norm is an even scalar, so use nonlinearity_scalars[1]
                scalar_nonlinearity=acts[nonlinearity_scalars[1]],
                normalize=True,
                epsilon=1e-8,
                bias=False,
            )

        self.equivariant_nonlin = equivariant_nonlin
        
        self.linear1 = Linear(
            irreps_in=self.irreps_in, irreps_out=linear_irreps_out
        )
        
        self.linear2 = Linear(
            irreps_in=self.equivariant_nonlin.irreps_out, irreps_out=irreps_in
        )

    def forward(self, x):
        # save old features for resnet
        old_x = x
        x = self.linear1(x)
        # do nonlinearity
        x = self.equivariant_nonlin(x)
        x = self.linear2(x)
        # do resnet
        if self.resnet:
            x = old_x + x
        return x

class Edge_builder(GraphModuleMixin, torch.nn.Module):

    def __init__(
        self,
        irreps_in,
        irreps_out,
        invariant_layers=1,
        invariant_neurons=8,
        nonlinearity_scalars: Dict[int, Callable] = {"e": "ssp"},
    ) -> None:

        super().__init__()

        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[
                AtomicDataDict.EDGE_EMBEDDING_KEY,
                AtomicDataDict.EDGE_ATTRS_KEY,
                AtomicDataDict.NODE_FEATURES_KEY
            ],
            my_irreps_in={
                AtomicDataDict.EDGE_EMBEDDING_KEY: o3.Irreps(
                    [
                        (
                            irreps_in[AtomicDataDict.EDGE_EMBEDDING_KEY].num_irreps,
                            (0, 1),
                        )
                    ]  # (0, 1) is even (invariant) scalars. We are forcing the EDGE_EMBEDDING to be invariant scalars so we can use a dense network
                )
            },
            irreps_out={AtomicDataDict.EDGE_FEATURES_KEY: irreps_out},
        )
        
        irreps_node_fea = self.irreps_in[AtomicDataDict.NODE_FEATURES_KEY]
        irreps_edge_attr = self.irreps_in[AtomicDataDict.EDGE_ATTRS_KEY]   
        feature_irreps_out = self.irreps_out[AtomicDataDict.EDGE_FEATURES_KEY]     

        # - Build modules -
        self.linear_node_src = Linear(
            irreps_in=irreps_node_fea,
            irreps_out=irreps_node_fea,
            internal_weights=True,
            shared_weights=True,
        )
        
        self.linear_node_dst = Linear(
            irreps_in=irreps_node_fea,
            irreps_out=irreps_node_fea,
            internal_weights=True,
            shared_weights=True,
        )

        irreps_mid = []
        instructions = []
        for i, (mul, ir_in1) in enumerate(irreps_node_fea):
            for j, (_, ir_in2) in enumerate(irreps_edge_attr):
                for ir_out in ir_in1 * ir_in2:
                    if ir_out in feature_irreps_out:
                        k = len(irreps_mid)
                        irreps_mid.append((mul, ir_out))
                        instructions.append((i, j, k, "uvu", True))

        # We sort the output irreps of the tensor product so that we can simplify them
        # when they are provided to the second o3.Linear
        irreps_mid = o3.Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()

        # Permute the output indexes of the instructions to match the sorted irreps:
        instructions = [
            (i_in1, i_in2, p[i_out], mode, train)
            for i_in1, i_in2, i_out, mode, train in instructions
        ]

        self.tp = TensorProduct(
            irreps_node_fea,
            irreps_edge_attr,
            irreps_mid,
            instructions,
            shared_weights=False,
            internal_weights=False,
        )

        # init_irreps already confirmed that the edge embeddding is all invariant scalars
        self.fc = FullyConnectedNet(
            [self.irreps_in[AtomicDataDict.EDGE_EMBEDDING_KEY].num_irreps]
            + invariant_layers * [invariant_neurons]
            + [self.tp.weight_numel],
            {
                "ssp": ShiftedSoftPlus,
                "silu": torch.nn.functional.silu,
            }[nonlinearity_scalars["e"]],
        )

        self.linear_edge = Linear(
            irreps_in=irreps_mid.simplify(),
            irreps_out=feature_irreps_out,
            internal_weights=True,
            shared_weights=True,
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        
        weight = self.fc(data[AtomicDataDict.EDGE_EMBEDDING_KEY])

        x = data[AtomicDataDict.NODE_FEATURES_KEY]
        edge_src = data[AtomicDataDict.EDGE_INDEX_KEY][1] # i
        edge_dst = data[AtomicDataDict.EDGE_INDEX_KEY][0] # j

        x_ij = self.linear_node_src(x[edge_src]) + self.linear_node_dst(x[edge_dst])
        
        edge_features = self.tp(
            x_ij, data[AtomicDataDict.EDGE_ATTRS_KEY], weight
        )

        edge_features = self.linear_edge(edge_features)
        
        data[AtomicDataDict.EDGE_FEATURES_KEY] = edge_features
        return data

class Edge_builder_tp(GraphModuleMixin, torch.nn.Module):
    
    def __init__(
        self,
        irreps_in,
        irreps_out,
        invariant_layers=1,
        invariant_neurons=8,
        nonlinearity_scalars: Dict[int, Callable] = {"e": "ssp"},
        irreps_node_prev = None
    ) -> None:

        super().__init__()

        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[
                AtomicDataDict.EDGE_EMBEDDING_KEY,
                AtomicDataDict.EDGE_ATTRS_KEY,
                AtomicDataDict.NODE_FEATURES_KEY
            ],
            my_irreps_in={
                AtomicDataDict.EDGE_EMBEDDING_KEY: o3.Irreps(
                    [
                        (
                            irreps_in[AtomicDataDict.EDGE_EMBEDDING_KEY].num_irreps,
                            (0, 1),
                        )
                    ]  # (0, 1) is even (invariant) scalars. We are forcing the EDGE_EMBEDDING to be invariant scalars so we can use a dense network
                )
            },
            irreps_out={AtomicDataDict.EDGE_FEATURES_KEY: irreps_out},
        )
        
        irreps_node_fea = self.irreps_in[AtomicDataDict.NODE_FEATURES_KEY]
        feature_irreps_out = self.irreps_out[AtomicDataDict.EDGE_FEATURES_KEY]     

        # - Build modules -
        self.linear_node_src = Linear(
            irreps_in=irreps_node_fea,
            irreps_out=irreps_node_fea,
            internal_weights=True,
            shared_weights=True,
        )
        
        if isinstance(irreps_node_prev, str):
            self.irreps_node_prev = o3.Irreps(irreps_node_prev)
        else:
            self.irreps_node_prev = irreps_node_prev
        
        self.linear_node_dst = Linear(
            irreps_in=irreps_node_fea,
            irreps_out=self.irreps_node_prev,
            internal_weights=True,
            shared_weights=True,
        )

        irreps_mid = []
        instructions = []
        for i, (mul, ir_in1) in enumerate(irreps_node_fea):
            for j, (_, ir_in2) in enumerate(self.irreps_node_prev):
                for ir_out in ir_in1 * ir_in2:
                    if ir_out in feature_irreps_out:
                        k = len(irreps_mid)
                        irreps_mid.append((mul, ir_out))
                        instructions.append((i, j, k, "uvu", True))

        # We sort the output irreps of the tensor product so that we can simplify them
        # when they are provided to the second o3.Linear
        irreps_mid = o3.Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()

        # Permute the output indexes of the instructions to match the sorted irreps:
        instructions = [
            (i_in1, i_in2, p[i_out], mode, train)
            for i_in1, i_in2, i_out, mode, train in instructions
        ]

        self.tp = TensorProduct(
            irreps_node_fea,
            self.irreps_node_prev,
            irreps_mid,
            instructions,
            shared_weights=False,
            internal_weights=False,
        )

        # init_irreps already confirmed that the edge embeddding is all invariant scalars
        self.fc = FullyConnectedNet(
            [self.irreps_in[AtomicDataDict.EDGE_EMBEDDING_KEY].num_irreps]
            + invariant_layers * [invariant_neurons]
            + [self.tp.weight_numel],
            {
                "ssp": ShiftedSoftPlus,
                "silu": torch.nn.functional.silu,
            }[nonlinearity_scalars["e"]],
        )

        self.linear_edge = Linear(
            irreps_in=irreps_mid.simplify(),
            irreps_out=feature_irreps_out,
            internal_weights=True,
            shared_weights=True,
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        
        weight = self.fc(data[AtomicDataDict.EDGE_EMBEDDING_KEY])

        x = data[AtomicDataDict.NODE_FEATURES_KEY]
        edge_src = data[AtomicDataDict.EDGE_INDEX_KEY][1] # i
        edge_dst = data[AtomicDataDict.EDGE_INDEX_KEY][0] # j

        x_i = self.linear_node_src(x[edge_src]) 
        x_j = self.linear_node_dst(x[edge_dst])
        
        edge_features = self.tp(
            x_i, x_j, weight
        )

        edge_features = self.linear_edge(edge_features)
          
        data[AtomicDataDict.EDGE_FEATURES_KEY] = data[AtomicDataDict.EDGE_FEATURES_KEY] + edge_features
        return data

class Triplet_builder(GraphModuleMixin, torch.nn.Module):
    
    def __init__(
        self,
        irreps_in,
        irreps_out,
        invariant_layers=1,
        invariant_neurons=8,
        nonlinearity_scalars: Dict[int, Callable] = {"e": "ssp"},
    ) -> None:

        super().__init__()

        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[
                AtomicDataDict.EDGE_FEATURES_KEY,
                AtomicDataDict.ANGLE_EMBEDDING_KEY
            ],
            my_irreps_in={
                AtomicDataDict.ANGLE_EMBEDDING_KEY: o3.Irreps(
                    [
                        (
                            irreps_in[AtomicDataDict.ANGLE_EMBEDDING_KEY].num_irreps,
                            (0, 1),
                        )
                    ]  # (0, 1) is even (invariant) scalars. We are forcing the ANGLE_EMBEDDING_KEY to be invariant scalars so we can use a dense network
                )
            },
            irreps_out={AtomicDataDict.TRIPLET_FEATURES_KEY: irreps_out},
        )
        
        self.ang_emb = sph_harm_layer(self.irreps_in[AtomicDataDict.ANGLE_EMBEDDING_KEY].num_irreps)
        
        irreps_edge_fea = self.irreps_in[AtomicDataDict.EDGE_FEATURES_KEY]   
        feature_irreps_out = self.irreps_out[AtomicDataDict.TRIPLET_FEATURES_KEY]  

        # - Build modules -
        self.linear_edge_kj = Linear(
            irreps_in=irreps_edge_fea,
            irreps_out=irreps_edge_fea,
            internal_weights=True,
            shared_weights=True,
        )
        
        self.linear_edge_ji = Linear(
            irreps_in=irreps_edge_fea,
            irreps_out=irreps_edge_fea,
            internal_weights=True,
            shared_weights=True,
        )   
        
        irreps_mid = []
        instructions = []
        for i, (mul, ir_in1) in enumerate(irreps_edge_fea):
            for j, (_, ir_in2) in enumerate(irreps_edge_fea):
                for ir_out in ir_in1 * ir_in2:
                    if ir_out in feature_irreps_out:
                        k = len(irreps_mid)
                        irreps_mid.append((mul, ir_out))
                        instructions.append((i, j, k, "uvu", True))

        # We sort the output irreps of the tensor product so that we can simplify them
        # when they are provided to the second o3.Linear
        irreps_mid = o3.Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()

        # Permute the output indexes of the instructions to match the sorted irreps:
        instructions = [
            (i_in1, i_in2, p[i_out], mode, train)
            for i_in1, i_in2, i_out, mode, train in instructions
        ]

        self.tp = TensorProduct(
            irreps_edge_fea,
            irreps_edge_fea,
            irreps_mid,
            instructions,
            shared_weights=False,
            internal_weights=False,
        )

        # init_irreps already confirmed that the edge embeddding is all invariant scalars
        self.fc = FullyConnectedNet(
            [self.irreps_in[AtomicDataDict.ANGLE_EMBEDDING_KEY].num_irreps]
            + invariant_layers * [invariant_neurons]
            + [self.tp.weight_numel],
            {
                "ssp": ShiftedSoftPlus,
                "silu": torch.nn.functional.silu,
            }[nonlinearity_scalars["e"]],
        )

        self.linear_triplet = Linear(
            irreps_in=irreps_mid.simplify(),
            irreps_out=feature_irreps_out,
            internal_weights=True,
            shared_weights=True,
        )
        
    def triplets(self, edge_index, num_nodes, cell_shift):
        row, col = edge_index  # j->i

        value = torch.arange(row.size(0), device=row.device)
        adj_t = SparseTensor(
            row=col, col=row, value=value, sparse_sizes=(num_nodes, num_nodes)
        )
        adj_t_row = adj_t[row]
        num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

        # Node indices (k->j->i) for triplets.
        idx_i = col.repeat_interleave(num_triplets)
        idx_j = row.repeat_interleave(num_triplets)
        idx_k = adj_t_row.storage.col()

        # Edge indices (k-j, j->i) for triplets.
        idx_kj = adj_t_row.storage.value()
        idx_ji = adj_t_row.storage.row()
 
        """
        idx_i -> pos[idx_i]
        idx_j -> pos[idx_j] - nbr_shift[idx_ji]
        idx_k -> pos[idx_k] - nbr_shift[idx_ji] - nbr_shift[idx_kj]
        """
        # Remove i == k triplets with the same cell_shift.
        relative_cell_shift = cell_shift[idx_kj] + cell_shift[idx_ji]
        mask = (idx_i != idx_k) | torch.any(relative_cell_shift != 0, dim=-1)
        idx_i, idx_j, idx_k, idx_kj, idx_ji = idx_i[mask], idx_j[mask], idx_k[mask], idx_kj[mask], idx_ji[mask]

        return col, row, idx_i, idx_j, idx_k, idx_kj, idx_ji

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        z = data.z
        pos = data.pos
        edge_index = data.edge_index
        nbr_shift = data.nbr_shift
        cell_shift = data.cell_shift # shape(Nedges, 3)

        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = self.triplets(edge_index, z.size(0), cell_shift)
        
        # Calculate angles -- revised version
        pos_i = pos[idx_i]
        pos_j = pos[idx_j] - nbr_shift[idx_ji]
        pos_k = pos[idx_k] - nbr_shift[idx_ji] - nbr_shift[idx_kj] 

        pos_ji = pos_j - pos_i
        pos_kj = pos_k - pos_j

        a = (pos_ji * pos_kj).sum(dim=-1)
        b = torch.cross(pos_ji, pos_kj).norm(dim=-1)
        angle = torch.atan2(b, a)
        
        ang_emb = self.ang_emb(angle)
        
        edge_kj = self.linear_edge_kj(data[AtomicDataDict.EDGE_FEATURES_KEY])[idx_kj]
        edge_ji = self.linear_edge_ji(data[AtomicDataDict.EDGE_FEATURES_KEY])[idx_ji]
        
        weight = self.fc(ang_emb)
        
        triplet_fea = self.tp(
            edge_kj, edge_ji, weight
        )

        triplet_fea = self.linear_triplet(triplet_fea)
        
        data[AtomicDataDict.TRIPLET_FEATURES_KEY] = triplet_fea
        data[AtomicDataDict.TRIPLET_INDEX_KEY] = (idx_i, idx_j, idx_k, idx_kj, idx_ji)
        return data

class Ham_layer(nn.Module):
    def __init__(self, irreps_in, feature_irreps_hidden, irreps_out, nonlinearity_type: str = "gate", resnet: bool = True):
        super().__init__()
        self.residual = residual_block(irreps_in=irreps_in, feature_irreps_hidden=feature_irreps_hidden, 
                                                 nonlinearity_type = nonlinearity_type, resnet=resnet) 
        self.linear = Linear(irreps_in=irreps_in, irreps_out=irreps_out) 
        
    def forward(self, x):
        x = self.residual(x)
        x = self.linear(x)
        return x

class HamGNN_pre(nn.Module):
    def __init__(self, config):
        super(HamGNN_pre, self).__init__()
        
        self.num_types = config.HamGNN_pre.num_types # number of atomic species in the dataset
        self.set_features = config.HamGNN_pre.set_features # Whether to set one_hot encoding as node_features in data
        
        self.export_triplet = config.HamGNN_pre.export_triplet # Whether to export the features of the triplet
        #
        self.irreps_edge_sh = config.HamGNN_pre.irreps_edge_sh
        self.edge_sh_normalization = config.HamGNN_pre.edge_sh_normalization
        self.edge_sh_normalize = config.HamGNN_pre.edge_sh_normalize
        #
        self.irreps_node_output = config.HamGNN_pre.irreps_node_output
        self.irreps_edge_output = config.HamGNN_pre.irreps_edge_output
        
        # Cosine basis function expansion layer
        self.cutoff = config.HamGNN_pre.cutoff
        self.cutoff_func = config.HamGNN_pre.cutoff_func
        if 'p' == self.cutoff_func.lower()[0]:  # "envelope func used in Dimnet"
            self.cutoff_func = cuttoff_envelope(cutoff=self.cutoff, exponent=6)
        elif 'c' == self.cutoff_func.lower()[0]:  # "cosinecutoff" func
            self.cutoff_func = CosineCutoff(cutoff=self.cutoff)
        else:
            print(f'There is no {self.cutoff_func} cutoff function!')
            quit()
            
        self.rbf_func = config.HamGNN_pre.rbf_func
        self.num_radial = config.HamGNN_pre.num_radial
        if self.rbf_func.lower() == 'gaussian': 
            self.rbf_func = GaussianSmearing(start=0.0, stop=self.cutoff, num_gaussians=self.num_radial, cutoff_func=self.cutoff_func)
        elif self.rbf_func.lower() == 'bessel':
            self.rbf_func = BesselBasis(cutoff=self.cutoff, n_rbf=self.num_radial, cutoff_func=self.cutoff_func)
        else:
            print(f'There is no {self.rbf_func} rbf function!')
            quit()
        # 
        self.num_interaction_layers = config.HamGNN_pre.num_interaction_layers # number of interaction layers
        self.resnet = config.HamGNN_pre.resnet # Whether to add residual layer       
        #
        self.irreps_node_features = config.HamGNN_pre.irreps_node_features # irreducible representations of the node
        #
        self.feature_irreps_hidden = config.HamGNN_pre.feature_irreps_hidden # Hidden irreducible representation of nodes in convolutional layers
        self.invariant_layers = config.HamGNN_pre.invariant_layers 
        self.invariant_neurons = config.HamGNN_pre.invariant_neurons
        convolution_kwargs : dict = {'invariant_layers':self.invariant_layers, 'invariant_neurons': self.invariant_neurons} # Additional initialization parameters for the conv layer
        
        self.one_hot = OneHotAtomEncoding(num_types=self.num_types, set_features=self.set_features) # The atomic number of the node is mapped to the one_hot encoding of "num_types*0e"
        
        # Embed the direction of the edge as a spherical harmonic feature, irreps_edge_sh is the irreducible representations of the direction of the edge
        self.spharm_edges = SphericalHarmonicEdgeAttrs(irreps_edge_sh=self.irreps_edge_sh, edge_sh_normalization=self.edge_sh_normalization,
                                                       edge_sh_normalize = self.edge_sh_normalize) 
        
        # Embed edge distances as features of 'num_basis*0e'
        self.radial_basis = RadialBasisEdgeEncoding(basis=self.rbf_func, cutoff=self.cutoff_func)
        
        self.chemical_embedding = AtomwiseLinear(irreps_in={AtomicDataDict.NODE_FEATURES_KEY: self.one_hot.irreps_out['node_attrs']}, 
                                                 irreps_out=self.irreps_node_features)
        
        self.convnet = nn.ModuleList([ConvNetLayer(irreps_in={AtomicDataDict.EDGE_ATTRS_KEY: self.irreps_edge_sh, AtomicDataDict.EDGE_EMBEDDING_KEY:self.radial_basis.irreps_out[AtomicDataDict.EDGE_EMBEDDING_KEY], 
                                                              AtomicDataDict.NODE_FEATURES_KEY: self.irreps_node_features, AtomicDataDict.NODE_ATTRS_KEY:self.one_hot.irreps_out[AtomicDataDict.NODE_ATTRS_KEY]}, 
                                                   feature_irreps_hidden=self.feature_irreps_hidden, 
                                                   convolution_kwargs = convolution_kwargs, resnet=self.resnet) for _ in range(self.num_interaction_layers)])
        
        self.conv_to_output_node = AtomwiseLinear(irreps_in={AtomicDataDict.NODE_FEATURES_KEY: self.irreps_node_features}, irreps_out=self.irreps_node_output)
        
        #"""
        self.conv_to_output_edge = Edge_builder(irreps_in={AtomicDataDict.NODE_FEATURES_KEY: self.irreps_node_output, 
                                                           AtomicDataDict.EDGE_ATTRS_KEY: self.irreps_edge_sh,
                                                           AtomicDataDict.EDGE_EMBEDDING_KEY:self.radial_basis.irreps_out[AtomicDataDict.EDGE_EMBEDDING_KEY]}, 
                                                irreps_out= self.irreps_edge_output, **convolution_kwargs)
        
        if self.export_triplet:
            self.num_spherical = config.HamGNN_pre.num_spherical
            self.irreps_triplet_output = config.HamGNN_pre.irreps_triplet_output
            self.conv_to_output_triplet = Triplet_builder(irreps_in={AtomicDataDict.EDGE_FEATURES_KEY: self.irreps_edge_output, 
                                                           AtomicDataDict.ANGLE_EMBEDDING_KEY: o3.Irreps([(self.num_spherical, (0, 1))])}, 
                                                          irreps_out= self.irreps_triplet_output, **convolution_kwargs)
    
    def forward(self, data, batch=None):
        self.one_hot(data)
        self.spharm_edges(data)
        self.radial_basis(data)
        self.chemical_embedding(data)
        # orbital convolution
        for i in range(self.num_interaction_layers):
            self.convnet[i](data)
        self.conv_to_output_node(data)    
        self.conv_to_output_edge(data)    
        graph_representation = EasyDict()
        graph_representation['node_attr'] = data[AtomicDataDict.NODE_FEATURES_KEY]
        graph_representation['edge_attr'] = data[AtomicDataDict.EDGE_FEATURES_KEY]
        if self.export_triplet:
            self.conv_to_output_triplet(data)
            graph_representation['triplet_attr'] = data[AtomicDataDict.TRIPLET_FEATURES_KEY] 
            graph_representation['triplet_index'] = data[AtomicDataDict.TRIPLET_INDEX_KEY]    
        return graph_representation

class HamGNN_pre2(nn.Module):
    def __init__(self, config):
        super(HamGNN_pre2, self).__init__()
        
        #self.num_types = config.HamGNN_pre.num_types # number of atomic species in the dataset
        self.set_features = config.HamGNN_pre.set_features # Whether to set one_hot encoding as node_features in data
        
        self.export_triplet = config.HamGNN_pre.export_triplet # Whether to export the features of the triplet
        #
        self.irreps_edge_sh = config.HamGNN_pre.irreps_edge_sh
        self.edge_sh_normalization = config.HamGNN_pre.edge_sh_normalization
        self.edge_sh_normalize = config.HamGNN_pre.edge_sh_normalize
        #
        self.irreps_node_output = config.HamGNN_pre.irreps_node_output
        self.irreps_edge_output = config.HamGNN_pre.irreps_edge_output
        
        # Cosine basis function expansion layer
        self.cutoff = config.HamGNN_pre.cutoff
        self.cutoff_func = config.HamGNN_pre.cutoff_func
        if 'e' == self.cutoff_func.lower()[0]:  # "envelope func used in Dimnet"
            self.cutoff_func = cuttoff_envelope(cutoff=self.cutoff, exponent=6)
        elif 'c' == self.cutoff_func.lower()[0]:  # "cosinecutoff" func
            self.cutoff_func = CosineCutoff(cutoff=self.cutoff)
        else:
            print(f'There is no {self.cutoff_func} cutoff function!')
            quit()
            
        self.rbf_func = config.HamGNN_pre.rbf_func
        self.num_radial = config.HamGNN_pre.num_radial
        if self.rbf_func.lower() == 'gaussian': 
            self.rbf_func = GaussianSmearing(start=0.0, stop=self.cutoff, num_gaussians=self.num_radial, cutoff_func=self.cutoff_func)
        elif self.rbf_func.lower() == 'bessel':
            self.rbf_func = BesselBasis(cutoff=self.cutoff, n_rbf=self.num_radial, cutoff_func=self.cutoff_func)
        else:
            print(f'There is no {self.rbf_func} rbf function!')
            quit()
        # 
        self.num_interaction_layers = config.HamGNN_pre.num_interaction_layers # number of interaction layers
        self.resnet = config.HamGNN_pre.resnet # Whether to add residual layer       
        #
        self.irreps_node_features = config.HamGNN_pre.irreps_node_features # irreducible representations of the node
        #
        self.feature_irreps_hidden = config.HamGNN_pre.feature_irreps_hidden # Hidden irreducible representation of nodes in convolutional layers
        self.invariant_layers = config.HamGNN_pre.invariant_layers 
        self.invariant_neurons = config.HamGNN_pre.invariant_neurons
        convolution_kwargs : dict = {'invariant_layers':self.invariant_layers, 'invariant_neurons': self.invariant_neurons} # Additional initialization parameters for the conv layer
        
        #self.one_hot = OneHotAtomEncoding(num_types=self.num_types, set_features=self.set_features) # The atomic number of the node is mapped to the one_hot encoding of "num_types*0e"
        
        self.num_node_attr_feas = config.HamGNN_pre.num_node_attr_feas
        self.emb = Embedding_block(num_node_attr_feas = self.num_node_attr_feas, set_features=self.set_features)
        
        # Embed the direction of the edge as a spherical harmonic feature, irreps_edge_sh is the irreducible representations of the direction of the edge
        self.spharm_edges = SphericalHarmonicEdgeAttrs(irreps_edge_sh=self.irreps_edge_sh, edge_sh_normalization=self.edge_sh_normalization,
                                                       edge_sh_normalize = self.edge_sh_normalize) 
        
        # Embed edge distances as features of 'num_basis*0e'
        self.radial_basis = RadialBasisEdgeEncoding(basis=self.rbf_func, cutoff=self.cutoff_func)
        
        self.chemical_embedding = AtomwiseLinear(irreps_in={AtomicDataDict.NODE_FEATURES_KEY: self.emb.irreps_out['node_attrs']}, 
                                                 irreps_out=self.irreps_node_features)
        
        self.convnet = nn.ModuleList([ConvNetLayer(irreps_in={AtomicDataDict.EDGE_ATTRS_KEY: self.irreps_edge_sh, AtomicDataDict.EDGE_EMBEDDING_KEY:self.radial_basis.irreps_out[AtomicDataDict.EDGE_EMBEDDING_KEY], 
                                                              AtomicDataDict.NODE_FEATURES_KEY: self.irreps_node_features, AtomicDataDict.NODE_ATTRS_KEY:self.emb.irreps_out[AtomicDataDict.NODE_ATTRS_KEY]}, 
                                                   feature_irreps_hidden=self.feature_irreps_hidden, 
                                                   convolution_kwargs = convolution_kwargs, resnet=self.resnet) for _ in range(self.num_interaction_layers)])
        
        self.conv_to_output_node = AtomwiseLinear(irreps_in={AtomicDataDict.NODE_FEATURES_KEY: self.irreps_node_features}, irreps_out=self.irreps_node_output)
        
        #"""
        self.conv_to_output_edge = Edge_builder(irreps_in={AtomicDataDict.NODE_FEATURES_KEY: self.irreps_node_output, 
                                                           AtomicDataDict.EDGE_ATTRS_KEY: self.irreps_edge_sh,
                                                           AtomicDataDict.EDGE_EMBEDDING_KEY:self.radial_basis.irreps_out[AtomicDataDict.EDGE_EMBEDDING_KEY]}, 
                                                irreps_out= self.irreps_edge_output, **convolution_kwargs)
        
        self.add_edge_tp = config.HamGNN_pre.add_edge_tp
        if self.add_edge_tp:
            self.irreps_node_prev = config.HamGNN_pre.irreps_node_prev
            self.conv_to_output_edge_tp = Edge_builder_tp(irreps_in={AtomicDataDict.NODE_FEATURES_KEY: self.irreps_node_output, 
                                                           AtomicDataDict.EDGE_ATTRS_KEY: self.irreps_edge_sh,
                                                           AtomicDataDict.EDGE_EMBEDDING_KEY:self.radial_basis.irreps_out[AtomicDataDict.EDGE_EMBEDDING_KEY]}, 
                                                            irreps_node_prev=self.irreps_node_prev,
                                                            irreps_out= self.irreps_edge_output, **convolution_kwargs)
        
        if self.export_triplet:
            self.num_spherical = config.HamGNN_pre.num_spherical
            self.irreps_triplet_output = config.HamGNN_pre.irreps_triplet_output
            self.conv_to_output_triplet = Triplet_builder(irreps_in={AtomicDataDict.EDGE_FEATURES_KEY: self.irreps_edge_output, 
                                                           AtomicDataDict.ANGLE_EMBEDDING_KEY: o3.Irreps([(self.num_spherical, (0, 1))])}, 
                                                          irreps_out= self.irreps_triplet_output, **convolution_kwargs)
        #"""
        
        #self.conv_to_output_edge = AtomwiseLinear(field=AtomicDataDict.EDGE_FEATURES_KEY, out_field=AtomicDataDict.EDGE_FEATURES_KEY, 
                                                  #irreps_in={AtomicDataDict.EDGE_FEATURES_KEY: self.convnet[-1].conv.linear_2.irreps_in}, irreps_out= self.irreps_edge_output)
    
    def forward(self, data, batch=None):
        #self.one_hot(data)
        self.emb(data)
        self.spharm_edges(data)
        self.radial_basis(data)
        self.chemical_embedding(data)
        # orbital convolution
        for i in range(self.num_interaction_layers):
            self.convnet[i](data)
        self.conv_to_output_node(data)    
        self.conv_to_output_edge(data)  
        if self.add_edge_tp:
            self.conv_to_output_edge_tp(data)  
        graph_representation = EasyDict()
        graph_representation['node_attr'] = data[AtomicDataDict.NODE_FEATURES_KEY]
        graph_representation['edge_attr'] = data[AtomicDataDict.EDGE_FEATURES_KEY]
        if self.export_triplet:
            self.conv_to_output_triplet(data)
            graph_representation['triplet_attr'] = data[AtomicDataDict.TRIPLET_FEATURES_KEY] 
            graph_representation['triplet_index'] = data[AtomicDataDict.TRIPLET_INDEX_KEY]    
        return graph_representation

class HamGNN_pre_charge(nn.Module):
    def __init__(self, config):
        super(HamGNN_pre_charge, self).__init__()
        
        #self.num_types = config.HamGNN_pre.num_types # number of atomic species in the dataset
        self.set_features = config.HamGNN_pre.set_features # Whether to set one_hot encoding as node_features in data
        
        self.export_triplet = config.HamGNN_pre.export_triplet # Whether to export the features of the triplet
        #
        self.irreps_edge_sh = config.HamGNN_pre.irreps_edge_sh
        self.edge_sh_normalization = config.HamGNN_pre.edge_sh_normalization
        self.edge_sh_normalize = config.HamGNN_pre.edge_sh_normalize
        #
        self.irreps_node_output = config.HamGNN_pre.irreps_node_output
        self.irreps_edge_output = config.HamGNN_pre.irreps_edge_output
        
        # Cosine basis function expansion layer
        self.cutoff = config.HamGNN_pre.cutoff
        self.cutoff_func = config.HamGNN_pre.cutoff_func
        if 'e' == self.cutoff_func.lower()[0]:  # "envelope func used in Dimnet"
            self.cutoff_func = cuttoff_envelope(cutoff=self.cutoff, exponent=6)
        elif 'c' == self.cutoff_func.lower()[0]:  # "cosinecutoff" func
            self.cutoff_func = CosineCutoff(cutoff=self.cutoff)
        else:
            print(f'There is no {self.cutoff_func} cutoff function!')
            quit()
            
        self.rbf_func = config.HamGNN_pre.rbf_func
        self.num_radial = config.HamGNN_pre.num_radial
        if self.rbf_func.lower() == 'gaussian': 
            self.rbf_func = GaussianSmearing(start=0.0, stop=self.cutoff, num_gaussians=self.num_radial, cutoff_func=self.cutoff_func)
        elif self.rbf_func.lower() == 'bessel':
            self.rbf_func = BesselBasis(cutoff=self.cutoff, n_rbf=self.num_radial, cutoff_func=self.cutoff_func)
        else:
            print(f'There is no {self.rbf_func} rbf function!')
            quit()
        # 
        self.num_interaction_layers = config.HamGNN_pre.num_interaction_layers # number of interaction layers
        self.resnet = config.HamGNN_pre.resnet # Whether to add residual layer       
        #
        self.irreps_node_features = config.HamGNN_pre.irreps_node_features # irreducible representations of the node
        #
        self.feature_irreps_hidden = config.HamGNN_pre.feature_irreps_hidden # Hidden irreducible representation of nodes in convolutional layers
        self.invariant_layers = config.HamGNN_pre.invariant_layers 
        self.invariant_neurons = config.HamGNN_pre.invariant_neurons
        convolution_kwargs : dict = {'invariant_layers':self.invariant_layers, 'invariant_neurons': self.invariant_neurons} # Additional initialization parameters for the conv layer
        
        #self.one_hot = OneHotAtomEncoding(num_types=self.num_types, set_features=self.set_features) # The atomic number of the node is mapped to the one_hot encoding of "num_types*0e"

        self.num_node_attr_feas = config.HamGNN_pre.num_node_attr_feas
        self.apply_charge_doping = True 
        self.num_charge_attr_feas = config.HamGNN_pre.num_charge_attr_feas
        self.emb = Embedding_block_q(num_node_attr_feas = self.num_node_attr_feas, apply_charge_doping=self.apply_charge_doping, 
                                   num_charge_attr_feas = self.num_charge_attr_feas, set_features=self.set_features)
        
        # Embed the direction of the edge as a spherical harmonic feature, irreps_edge_sh is the irreducible representations of the direction of the edge
        self.spharm_edges = SphericalHarmonicEdgeAttrs(irreps_edge_sh=self.irreps_edge_sh, edge_sh_normalization=self.edge_sh_normalization,
                                                       edge_sh_normalize = self.edge_sh_normalize) 
        
        # Embed edge distances as features of 'num_basis*0e'
        self.radial_basis = RadialBasisEdgeEncoding(basis=self.rbf_func, cutoff=self.cutoff_func)
        
        self.chemical_embedding = AtomwiseLinear(irreps_in={AtomicDataDict.NODE_FEATURES_KEY: self.emb.irreps_out['node_attrs']}, 
                                                 irreps_out=self.irreps_node_features)
        
        self.convnet = nn.ModuleList([ConvNetLayer(irreps_in={AtomicDataDict.EDGE_ATTRS_KEY: self.irreps_edge_sh, AtomicDataDict.EDGE_EMBEDDING_KEY:self.radial_basis.irreps_out[AtomicDataDict.EDGE_EMBEDDING_KEY], 
                                                              AtomicDataDict.NODE_FEATURES_KEY: self.irreps_node_features, AtomicDataDict.NODE_ATTRS_KEY:self.emb.irreps_out[AtomicDataDict.NODE_ATTRS_KEY]}, 
                                                   feature_irreps_hidden=self.feature_irreps_hidden, 
                                                   convolution_kwargs = convolution_kwargs, resnet=self.resnet) for _ in range(self.num_interaction_layers)])
        
        self.conv_to_output_node = AtomwiseLinear(irreps_in={AtomicDataDict.NODE_FEATURES_KEY: self.irreps_node_features}, irreps_out=self.irreps_node_output)
        
        #"""
        self.conv_to_output_edge = Edge_builder(irreps_in={AtomicDataDict.NODE_FEATURES_KEY: self.irreps_node_output, 
                                                           AtomicDataDict.EDGE_ATTRS_KEY: self.irreps_edge_sh,
                                                           AtomicDataDict.EDGE_EMBEDDING_KEY:self.radial_basis.irreps_out[AtomicDataDict.EDGE_EMBEDDING_KEY]}, 
                                                irreps_out= self.irreps_edge_output, **convolution_kwargs)
        
        self.add_edge_tp = config.HamGNN_pre.add_edge_tp
        if self.add_edge_tp:
            self.irreps_node_prev = config.HamGNN_pre.irreps_node_prev
            self.conv_to_output_edge_tp = Edge_builder_tp(irreps_in={AtomicDataDict.NODE_FEATURES_KEY: self.irreps_node_output, 
                                                           AtomicDataDict.EDGE_ATTRS_KEY: self.irreps_edge_sh,
                                                           AtomicDataDict.EDGE_EMBEDDING_KEY:self.radial_basis.irreps_out[AtomicDataDict.EDGE_EMBEDDING_KEY]}, 
                                                            irreps_node_prev=self.irreps_node_prev,
                                                            irreps_out= self.irreps_edge_output, **convolution_kwargs)
        
        if self.export_triplet:
            self.num_spherical = config.HamGNN_pre.num_spherical
            self.irreps_triplet_output = config.HamGNN_pre.irreps_triplet_output
            self.conv_to_output_triplet = Triplet_builder(irreps_in={AtomicDataDict.EDGE_FEATURES_KEY: self.irreps_edge_output, 
                                                           AtomicDataDict.ANGLE_EMBEDDING_KEY: o3.Irreps([(self.num_spherical, (0, 1))])}, 
                                                          irreps_out= self.irreps_triplet_output, **convolution_kwargs)
        #"""
        
        #self.conv_to_output_edge = AtomwiseLinear(field=AtomicDataDict.EDGE_FEATURES_KEY, out_field=AtomicDataDict.EDGE_FEATURES_KEY, 
                                                  #irreps_in={AtomicDataDict.EDGE_FEATURES_KEY: self.convnet[-1].conv.linear_2.irreps_in}, irreps_out= self.irreps_edge_output)
    
    def forward(self, data, batch=None):
        #self.one_hot(data)
        self.emb(data)
        self.spharm_edges(data)
        self.radial_basis(data)
        self.chemical_embedding(data)
        # orbital convolution
        for i in range(self.num_interaction_layers):
            self.convnet[i](data)
        self.conv_to_output_node(data)    
        self.conv_to_output_edge(data)  
        if self.add_edge_tp:
            self.conv_to_output_edge_tp(data)  
        graph_representation = EasyDict()
        graph_representation['node_attr'] = data[AtomicDataDict.NODE_FEATURES_KEY]
        graph_representation['edge_attr'] = data[AtomicDataDict.EDGE_FEATURES_KEY]
        if self.export_triplet:
            self.conv_to_output_triplet(data)
            graph_representation['triplet_attr'] = data[AtomicDataDict.TRIPLET_FEATURES_KEY] 
            graph_representation['triplet_index'] = data[AtomicDataDict.TRIPLET_INDEX_KEY]    
        return graph_representation

class HamGNN_out(nn.Module):
    
    def __init__(self, irreps_in_node: Union[int, str, o3.Irreps]=None, irreps_in_edge: Union[int, str, o3.Irreps]=None, irreps_in_triplet: Union[int, str, o3.Irreps]=None, nao_max: int = 14, return_forces=False, create_graph=False, 
                 ham_type: str='openmx', ham_only: bool = False, symmetrize: bool=True, include_triplet: bool = False, calculate_band_energy: bool = False, num_k: int = 8, 
                 k_path:Union[list, np.array, tuple]=None, band_num_control:dict=None, soc_switch:bool=True, nonlinearity_type:str='norm', export_reciprocal_values: bool = False, add_H0:bool= False, soc_basis: str='so3'):

        
        """
        nao_max (int): Maximum total number of atomic orbits
        """
        super(HamGNN_out, self).__init__()
        if return_forces:
            self.derivative = True
        else:
            self.derivative = False

        self.create_graph = create_graph

        self.nao_max = nao_max
        self.ham_type = ham_type.lower()
        self.ham_only = ham_only
        self.symmetrize = symmetrize
        self.include_triplet = include_triplet
        self.soc_switch = soc_switch
        self.nonlinearity_type = nonlinearity_type
        self.export_reciprocal_values = export_reciprocal_values
        self.add_H0 = add_H0
        # band
        self.calculate_band_energy = calculate_band_energy
        self.num_k = num_k
        self.k_path = k_path
        
        self.soc_basis = soc_basis.lower()
        
        # band_num_control      
        if (band_num_control is not None) and (not self.export_reciprocal_values) and (isinstance(band_num_control, dict)):      
            self.band_num_control = {Element[k].Z: band_num_control[k] for k in band_num_control.keys()}
        elif isinstance(band_num_control, int):
            self.band_num_control = band_num_control
        else:
            self.band_num_control = None
        
        # openmx
        if self.ham_type == 'openmx':
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
                                    42:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], # Mo   
                                    83:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], # Bi  
                                    34:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], # Se
                                    24:[0,1,2,3,4,5,6,7,8,9,10,11,12,13], # Cr 
                                    53:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], # I  
                                    82:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], # pb
                                    55:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], # Cs
                                    33:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], # As
                                    31:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], # Ga  
                                    32:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], # Ge
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
                raise NotImplementedError
        
        elif self.ham_type == 'siesta':
            self.num_valence = {
                1:1,2:2,
                3:1,4:2,5:3,6:4,7:5,8:6,9:7,10:8,
                11:1,12:2,13:3,14:4,15:5,16:6,17:7,18:8,
                19:1,20:2,22:12
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
                raise NotImplementedError
        elif self.ham_type == 'abacus':
            # this dict is for abacus calculation.
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
                raise NotImplementedError
        # pasp
        elif self.ham_type == 'pasp':   
            self.row = self.col = o3.Irreps("1x1o")
        else:
            raise NotImplementedError
        
        self._init_irreps()
        self.cg_cal = ClebschGordan()

        # hamiltonian     
        if soc_switch:
            if self.ham_type is not 'openmx':
                self.soc_basis == 'su2'
            
            if self.soc_basis == 'su2':
                self.onsitenet_residual = residual_block(irreps_in=irreps_in_node, feature_irreps_hidden=irreps_in_node, 
                                                         nonlinearity_type = self.nonlinearity_type, resnet=True) 
                self.onsitenet_linear = Linear(irreps_in=irreps_in_node, irreps_out=2*self.ham_irreps)

                self.offsitenet_residual = residual_block(irreps_in=irreps_in_edge, feature_irreps_hidden=irreps_in_edge, 
                                                         nonlinearity_type = self.nonlinearity_type, resnet=True)  
                self.offsitenet_linear = Linear(irreps_in=irreps_in_edge, irreps_out=2*self.ham_irreps)
            
            elif self.soc_basis == 'so3':
                self.onsitenet_residual = residual_block(irreps_in=irreps_in_node, feature_irreps_hidden=irreps_in_node, 
                                                         nonlinearity_type = self.nonlinearity_type, resnet=True) 
                self.onsitenet_linear = Linear(irreps_in=irreps_in_node, irreps_out=self.ham_irreps)

                self.offsitenet_residual = residual_block(irreps_in=irreps_in_edge, feature_irreps_hidden=irreps_in_edge, 
                                                         nonlinearity_type = self.nonlinearity_type, resnet=True)  
                self.offsitenet_linear = Linear(irreps_in=irreps_in_edge, irreps_out=self.ham_irreps)
                
                self.onsitenet_residual_ksi = residual_block(irreps_in=irreps_in_node, feature_irreps_hidden=irreps_in_node, 
                                                         nonlinearity_type = self.nonlinearity_type, resnet=True) 
                self.ksi_on_scalar = Linear(irreps_in=irreps_in_node, irreps_out=(self.nao_max**2*o3.Irreps("0e")).simplify())
    
                self.offsitenet_residual_ksi = residual_block(irreps_in=irreps_in_edge, feature_irreps_hidden=irreps_in_edge, 
                                                         nonlinearity_type = self.nonlinearity_type, resnet=True)
                self.ksi_off_scalar = Linear(irreps_in=irreps_in_edge, irreps_out=(self.nao_max**2*o3.Irreps("0e")).simplify()) 
            
            else:
                raise NotImplementedError(f"{soc_basis} not supportted!")
                                
        else: # non-soc
            self.onsitenet_residual = residual_block(irreps_in=irreps_in_node, feature_irreps_hidden=irreps_in_node, 
                                                     nonlinearity_type = self.nonlinearity_type, resnet=True) 
            self.onsitenet_linear = Linear(irreps_in=irreps_in_node, irreps_out=self.ham_irreps)
               
            self.offsitenet_residual = residual_block(irreps_in=irreps_in_edge, feature_irreps_hidden=irreps_in_edge, 
                                                     nonlinearity_type = self.nonlinearity_type, resnet=True)  
            self.offsitenet_linear = Linear(irreps_in=irreps_in_edge, irreps_out=self.ham_irreps)
        
        if not self.ham_only:            
            self.onsitenet_s = Ham_layer(irreps_in=irreps_in_node, feature_irreps_hidden=irreps_in_node,irreps_out=self.ham_irreps, 
                                                 nonlinearity_type = self.nonlinearity_type, resnet=True)
            self.offsitenet_s = Ham_layer(irreps_in=irreps_in_edge, feature_irreps_hidden=irreps_in_edge, irreps_out=self.ham_irreps, 
                                                 nonlinearity_type = self.nonlinearity_type, resnet=True)
                 
    def _init_irreps(self):
        """
        Initialize the irreducible representation of the Hamiltonian
        """
        self.ham_irreps_dim = []
        self.ham_irreps = o3.Irreps()

        if self.soc_switch and (self.soc_basis == 'su2'): 
            out_js_list = []
            for _, li in self.row:
                for _, lj in self.col:
                    out_js_list.append((li.l, lj.l))

            self.hamDecomp = e3TensorDecomp(None, out_js_list, default_dtype_torch=torch.float32, nao_max=self.nao_max, spinful=True)
            self.ham_irreps = self.hamDecomp.required_irreps_out
        else:
            for _, li in self.row:
                for _, lj in self.col:
                    for L in range(abs(li.l-lj.l), li.l+lj.l+1):
                        self.ham_irreps += o3.Irrep(L, (-1)**(li.l+lj.l))

        for irs in self.ham_irreps:
            self.ham_irreps_dim.append(irs.dim)
        self.ham_irreps_dim = torch.LongTensor(self.ham_irreps_dim)

    def matrix_merge(self, sph_split):   
        """
        Incorporate irreducible representations into matrix blocks
        """
        block = torch.zeros(sph_split[0].shape[0], self.nao_max, self.nao_max).type_as(sph_split[0])
        
        idx = 0 #index for accessing the correct irreps
        start_i = 0
        for _, li in self.row:
            n_i = 2*li.l+1
            start_j = 0
            for _, lj in self.col:
                n_j = 2*lj.l+1
                for L in range(abs(li.l-lj.l), li.l+lj.l+1):
                    #compute inverse spherical tensor product             
                    cg = math.sqrt(2*L+1)*self.cg_cal(li.l, lj.l, L).unsqueeze(0)
                    product = (cg*sph_split[idx].unsqueeze(-2).unsqueeze(-2)).sum(-1)

                    #add product to appropriate part of the block
                    blockpart = block.narrow(-2,start_i,n_i).narrow(-1,start_j,n_j)
                    blockpart += product

                    idx += 1
                start_j += n_j
            start_i += n_i
            
        return block.reshape(-1, self.nao_max*self.nao_max)
    
    def change_index(self, hamiltonian):
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
            hamiltonian = hamiltonian.reshape(-1, self.nao_max**2)                
        return hamiltonian
    
    def convert_to_mole_Ham(self, data, Hon, Hoff):
        # Get the number of nodes in each crystal
        max_atoms = torch.max(data.node_counts).item()
                
        # parse the Atomic Orbital Basis Sets
        basis_definition = torch.zeros((99, self.nao_max)).type_as(data.z)
        basis_def_temp = copy.deepcopy(self.basis_def)
        # key is the atomic number, value is the occupied orbits.
        for k in self.basis_def.keys():
            basis_def_temp[k] = [num-1 for num in self.basis_def[k]]
            basis_definition[k][basis_def_temp[k]] = 1
            
        orb_mask = basis_definition[data.z].view(-1, max_atoms*self.nao_max) # shape: [Nbatch, max_atoms*nao_max]  
        orb_mask = orb_mask[:,:,None] * orb_mask[:,None,:]       # shape: [Nbatch, max_atoms*nao_max, max_atoms*nao_max]
        orb_mask = orb_mask.view(-1, max_atoms*self.nao_max) # shape: [Natoms*nao_max, max_atoms*nao_max]
        
        atom_idx = torch.arange(data.z.shape[0]).type_as(data.z)
        H = torch.zeros([data.z.shape[0], max_atoms, self.nao_max**2]).type_as(Hon) # shape: [Natoms, max_atoms, nao_max**2]
        H[atom_idx, atom_idx%max_atoms] = Hon
        H[data.edge_index[0], data.edge_index[1]%max_atoms] = Hoff
        H = H.reshape(
            data.z.shape[0], max_atoms, self.nao_max, self.nao_max) # shape: [Natoms, max_atoms, nao_max, nao_max]

        # reshape the dimension of the hamiltonian.
        H = H.permute((0, 2, 1, 3))
        H = H.reshape(data.z.shape[0] * self.nao_max, max_atoms * self.nao_max)

        # mask padded orbitals
        H = torch.masked_select(H, orb_mask > 0)
        orbs = int(math.sqrt(H.shape[0] / (data.z.shape[0]/max_atoms)))
        H = H.reshape(-1, orbs)              
        return H
    
    def cat_onsite_and_offsite(self, data, Hon, Hoff):
        # Get the number of nodes in each crystal
        node_counts = data.node_counts
        Hon_split = torch.split(Hon, node_counts.tolist(), dim=0)
        #
        j, i = data.edge_index
        edge_num = torch.ones_like(j)
        edge_num = scatter(edge_num, data.batch[j], dim=0)
        Hoff_split = torch.split(Hoff, edge_num.tolist(), dim=0)
        #
        H = []
        for i in range(len(node_counts)):
            H.append(Hon_split[i])
            H.append(Hoff_split[i])
        H = torch.cat(H, dim=0)
        return H 
    
    def symmetrize_Hon(self, Hon, sign:str='+'):
        if self.symmetrize:
            Hon = Hon.reshape(-1, self.nao_max, self.nao_max)
            if sign == '+':
                Hon = 0.5*(Hon + Hon.permute((0,2,1)))
            else:
                Hon = 0.5*(Hon - Hon.permute((0,2,1)))
            Hon = Hon.reshape(-1, self.nao_max**2)
            return Hon
        else:
            return Hon
    
    def symmetrize_Hoff(self, Hoff, inv_edge_idx, sign:str='+'):
        if self.symmetrize:
            Hoff = Hoff.reshape(-1, self.nao_max, self.nao_max)
            if sign == '+':
                Hoff = 0.5*(Hoff + Hoff[inv_edge_idx].permute((0,2,1)))
            else:
                Hoff = 0.5*(Hoff - Hoff[inv_edge_idx].permute((0,2,1)))
            Hoff = Hoff.reshape(-1, self.nao_max**2)
            return Hoff
        else:
            return Hoff
    
    def cal_band_energy_debug(self, Hon, Hoff, Son, Soff, data, export_reciprocal_values:bool=False):
        """
        Currently this function can only be used to calculate the energy band of the openmx Hamiltonian.
        """
        j, i = data.edge_index
        cell = data.cell # shape:(Nbatch, 3, 3)
        Nbatch = cell.shape[0]
        
        # parse the Atomic Orbital Basis Sets
        basis_definition = torch.zeros((99, self.nao_max)).type_as(data.z)
        # key is the atomic number, value is the index of the occupied orbits.
        for k in self.basis_def.keys():
            basis_definition[k][self.basis_def[k]] = 1
            
        orb_mask = basis_definition[data.z] # shape: [Natoms, nao_max] 
        orb_mask = torch.split(orb_mask, data.node_counts.tolist(), dim=0) # shape: [natoms, nao_max]
        orb_mask_batch = []
        for idx in range(Nbatch):
            orb_mask_batch.append(orb_mask[idx].reshape(-1, 1)* orb_mask[idx].reshape(1, -1)) # shape: [natoms*nao_max, natoms*nao_max]
        
        # set the number of valence electrons
        num_val = torch.zeros((99,)).type_as(data.z)
        for k in self.num_valence.keys():
            num_val[k] = self.num_valence[k]
        num_val = num_val[data.z] # shape: [Natoms]
        num_val = scatter(num_val, data.batch, dim=0) # shape: [Nbatch]
                
        # Initialize band_num_win
        if self.band_num_control is not None:
            band_num_win = torch.zeros((99,)).type_as(data.z)
            for k in self.band_num_control.keys():
                band_num_win[k] = self.band_num_control[k]
            band_num_win = band_num_win[data.z] # shape: [Natoms,]   
            band_num_win = scatter(band_num_win, data.batch, dim=0) # shape: (Nbatch,)
             
        # Separate Hon and Hoff for each batch
        node_counts = data.node_counts
        node_counts_shift = torch.cumsum(node_counts, dim=0) - node_counts
        Hon_split = torch.split(Hon, node_counts.tolist(), dim=0)
        Son_split = torch.split(data.Son, node_counts.tolist(), dim=0)
        Son_pred_split = torch.split(Son, node_counts.tolist(), dim=0)
        #
        edge_num = torch.ones_like(j)
        edge_num = scatter(edge_num, data.batch[j], dim=0) # shape: (Nbatch,)
        edge_num_shift = torch.cumsum(edge_num, dim=0) - edge_num
        Hoff_split = torch.split(Hoff, edge_num.tolist(), dim=0)
        Soff_split = torch.split(data.Soff, edge_num.tolist(), dim=0)
        Soff_pred_split = torch.split(Soff, edge_num.tolist(), dim=0)
        if export_reciprocal_values:
            dSon_split = torch.split(data.dSon, node_counts.tolist(), dim=0)
            dSoff_split = torch.split(data.dSoff, edge_num.tolist(), dim=0)
        
        band_energy = []
        wavefunction = []
        H_reciprocal = []
        H_sym = []
        S_reciprocal = []
        dS_reciprocal = []
        gap = []
        for idx in range(Nbatch):
            k_vec = data.k_vecs[idx]   
            natoms = data.node_counts[idx]
            
            # Initialize HK and SK       
            coe = torch.exp(2j*torch.pi*torch.sum(data.nbr_shift[edge_num_shift[idx]+torch.arange(edge_num[idx]).type_as(j),None,:]*k_vec[None,:,:], axis=-1)) # (nedges, 1, 3)*(1, num_k, 3) -> (nedges, num_k)     
            
            HK = torch.view_as_complex(torch.zeros((self.num_k, natoms, natoms, self.nao_max, self.nao_max, 2)).type_as(Hon))
            SK = torch.view_as_complex(torch.zeros((self.num_k, natoms, natoms, self.nao_max, self.nao_max, 2)).type_as(Hon))  
            SK_pred = torch.view_as_complex(torch.zeros((self.num_k, natoms, natoms, self.nao_max, self.nao_max, 2)).type_as(Hon))           
            if export_reciprocal_values:
                dSK = torch.view_as_complex(torch.zeros((self.num_k, natoms, natoms, self.nao_max, self.nao_max, 3, 2)).type_as(Hon))

            na = torch.arange(natoms).type_as(j)
            HK[:,na,na,:,:] +=  Hon_split[idx].reshape(-1, self.nao_max, self.nao_max)[None,na,:,:].type_as(HK) # shape (num_k, natoms, nao_max, nao_max)
            SK[:,na,na,:,:] +=  Son_split[idx].reshape(-1, self.nao_max, self.nao_max)[None,na,:,:].type_as(SK)
            SK_pred[:,na,na,:,:] +=  Son_pred_split[idx].reshape(-1, self.nao_max, self.nao_max)[None,na,:,:].type_as(SK_pred)
            if export_reciprocal_values:
                dSK[:,na,na,:,:,:] +=  dSon_split[idx].reshape(-1, self.nao_max, self.nao_max, 3)[None,na,:,:,:].type_as(dSK)

            
            for iedge in range(edge_num[idx]):
                # shape (num_k, nao_max, nao_max) += (num_k, 1, 1)*(1, nao_max, nao_max)
                j_idx = j[edge_num_shift[idx]+iedge] - node_counts_shift[idx]
                i_idx = i[edge_num_shift[idx]+iedge] - node_counts_shift[idx]
                HK[:,j_idx,i_idx,:,:] += coe[iedge,:,None,None] * Hoff_split[idx].reshape(-1, self.nao_max, self.nao_max)[None,iedge,:,:]
                SK[:,j_idx,i_idx,:,:] += coe[iedge,:,None,None] * Soff_split[idx].reshape(-1, self.nao_max, self.nao_max)[None,iedge,:,:]
                SK_pred[:,j_idx,i_idx,:,:] += coe[iedge,:,None,None] * Soff_pred_split[idx].reshape(-1, self.nao_max, self.nao_max)[None,iedge,:,:]
            
            if export_reciprocal_values:
                for iedge in range(edge_num[idx]):
                    j_idx = j[edge_num_shift[idx]+iedge] - node_counts_shift[idx]
                    i_idx = i[edge_num_shift[idx]+iedge] - node_counts_shift[idx]
                    dSK[:,j_idx,i_idx,:,:,:] += coe[iedge,:,None,None,None] * dSoff_split[idx].reshape(-1, self.nao_max, self.nao_max, 3)[None,iedge,:,:,:]

            HK = torch.swapaxes(HK,-2,-3) #(nk, natoms, nao_max, natoms, nao_max)
            HK = HK.reshape(-1, natoms*self.nao_max, natoms*self.nao_max)
            SK = torch.swapaxes(SK,-2,-3) #(nk, natoms, nao_max, natoms, nao_max)
            SK = SK.reshape(-1, natoms*self.nao_max, natoms*self.nao_max)
            SK_pred = torch.swapaxes(SK_pred,-2,-3) #(nk, natoms, nao_max, natoms, nao_max)
            SK_pred = SK_pred.reshape(-1, natoms*self.nao_max, natoms*self.nao_max)
            if export_reciprocal_values:
                dSK = torch.swapaxes(dSK,-3,-4) #(nk, natoms, nao_max, natoms, nao_max, 3)
                dSK = dSK.reshape(-1, natoms*self.nao_max, natoms*self.nao_max, 3)
            
            # mask HK and SK
            HK = torch.masked_select(HK, orb_mask_batch[idx].repeat(self.num_k,1,1) > 0)
            norbs = int(math.sqrt(HK.numel()/self.num_k))
            HK = HK.reshape(self.num_k, norbs, norbs)
            
            SK = torch.masked_select(SK, orb_mask_batch[idx].repeat(self.num_k,1,1) > 0)
            norbs = int(math.sqrt(SK.numel()/self.num_k))
            SK = SK.reshape(self.num_k, norbs, norbs)

            SK_pred = torch.masked_select(SK_pred, orb_mask_batch[idx].repeat(self.num_k,1,1) > 0)
            norbs = int(math.sqrt(SK_pred.numel()/self.num_k))
            SK_pred = SK_pred.reshape(self.num_k, norbs, norbs)
            
            if export_reciprocal_values:
                dSK = torch.masked_select(dSK, orb_mask_batch[idx].unsqueeze(-1).repeat(self.num_k,1,1,3) > 0)
                dSK = dSK.reshape(self.num_k, norbs, norbs, 3)            
            
            # Calculate band energies
            L = torch.linalg.cholesky(SK)
            L_t = torch.transpose(L.conj(), dim0=-1, dim1=-2)
            L_inv = torch.linalg.inv(L)
            L_t_inv = torch.linalg.inv(L_t)
            Hs = torch.bmm(torch.bmm(L_inv, HK), L_t_inv)
            orbital_energies, orbital_coefficients = torch.linalg.eigh(Hs)        
            
            # Convert the wavefunction coefficients back to the original basis
            orbital_coefficients = torch.einsum('ijk, ika -> iaj', L_t_inv, orbital_coefficients)
            
            # Numpy
            """
            HK_t = HK.detach().cpu().numpy()
            SK_t = SK.detach().cpu().numpy()
            from scipy.linalg import eigh
            eigen = []
            eigen_vecs = []
            for ik in range(self.num_k):
                w, v = eigh(a=HK_t[ik], b=SK_t[ik])
                eigen.append(w)
                eigen_vecs.append(v)
            eigen_vecs = np.array(eigen_vecs) # (nk, nbands, nbands)
            eigen_vecs = np.swapaxes(eigen_vecs, -1, -2)
            
            lamda = np.einsum('nai, nij, naj -> na', np.conj(eigen_vecs), SK_t, eigen_vecs).real
            lamda = 1/np.sqrt(lamda) # shape: (numk, norbs)
            eigen_vecs = eigen_vecs*lamda[:,:,None]  
            orbital_energies, orbital_coefficients = torch.Tensor(eigen).type_as(data.pos), torch.complex(torch.Tensor(eigen_vecs.real), torch.Tensor(eigen_vecs.imag)).type_as(HK)
            """
            
            if export_reciprocal_values:
                # Normalize wave function
                lamda = torch.einsum('nai, nij, naj -> na', torch.conj(orbital_coefficients), SK, orbital_coefficients).real
                lamda = 1/torch.sqrt(lamda) # shape: (numk, norbs)
                orbital_coefficients = orbital_coefficients*lamda.unsqueeze(-1)    
                        
                H_reciprocal.append(HK)
                S_reciprocal.append(SK_pred)
                dS_reciprocal.append(dSK)
            
            if self.band_num_control is not None:
                orbital_energies = orbital_energies[:,:band_num_win[idx]]
                orbital_coefficients = orbital_coefficients[:,:band_num_win[idx],:]                
            band_energy.append(torch.transpose(orbital_energies, dim0=-1, dim1=-2)) # [shape:(Nbands, num_k)]
            wavefunction.append(orbital_coefficients)  
            H_sym.append(Hs.view(-1)) 
            numc = math.ceil(num_val[idx]/2)
            gap.append((torch.min(orbital_energies[:,numc]) - torch.max(orbital_energies[:,numc-1])).unsqueeze(0))    
            
        band_energy = torch.cat(band_energy, dim=0) # [shape:(Nbands, num_k)]
        
        gap = torch.cat(gap, dim=0)
        
        if export_reciprocal_values:
            wavefunction = torch.stack(wavefunction, dim=0) # shape:[Nbatch, num_k, norbs, norbs]
            HK = torch.stack(H_reciprocal, dim=0) # shape:[Nbatch, num_k, norbs, norbs]
            SK = torch.stack(S_reciprocal, dim=0) # shape:[Nbatch, num_k, norbs, norbs]  
            dSK = torch.stack(dS_reciprocal, dim=0) # shape:[Nbatch, num_k, norbs, norbs, 3]   
            return band_energy, wavefunction, HK, SK, dSK, gap
        else:
            wavefunction = [wavefunction[idx].reshape(-1) for idx in range(Nbatch)]
            wavefunction = torch.cat(wavefunction, dim=0) # shape:[Nbatch*num_k*norbs*norbs]
            H_sym = torch.cat(H_sym, dim=0) # shape:(Nbatch*num_k*norbs*norbs)
            return band_energy, wavefunction, gap, H_sym   
    
    def cal_band_energy(self, Hon, Hoff, data, export_reciprocal_values:bool=False):
        """
        Currently this function can only be used to calculate the energy band of the openmx Hamiltonian.
        """
        j, i = data.edge_index
        cell = data.cell # shape:(Nbatch, 3, 3)
        Nbatch = cell.shape[0]
        
        # parse the Atomic Orbital Basis Sets
        basis_definition = torch.zeros((99, self.nao_max)).type_as(data.z)
        # key is the atomic number, value is the index of the occupied orbits.
        for k in self.basis_def.keys():
            basis_definition[k][self.basis_def[k]] = 1
            
        orb_mask = basis_definition[data.z] # shape: [Natoms, nao_max] 
        orb_mask = torch.split(orb_mask, data.node_counts.tolist(), dim=0) # shape: [natoms, nao_max]
        orb_mask_batch = []
        for idx in range(Nbatch):
            orb_mask_batch.append(orb_mask[idx].reshape(-1, 1)* orb_mask[idx].reshape(1, -1)) # shape: [natoms*nao_max, natoms*nao_max]
        
        # set the number of valence electrons
        num_val = torch.zeros((99,)).type_as(data.z)
        for k in self.num_valence.keys():
            num_val[k] = self.num_valence[k]
        num_val = num_val[data.z] # shape: [Natoms]
        num_val = scatter(num_val, data.batch, dim=0) # shape: [Nbatch]
                
        # Initialize band_num_win
        if isinstance(self.band_num_control, dict):
            band_num_win = torch.zeros((99,)).type_as(data.z)
            for k in self.band_num_control.keys():
                band_num_win[k] = self.band_num_control[k]
            band_num_win = band_num_win[data.z] # shape: [Natoms,]   
            band_num_win = scatter(band_num_win, data.batch, dim=0) # shape: (Nbatch,)   
             
        # Separate Hon and Hoff for each batch
        node_counts = data.node_counts
        node_counts_shift = torch.cumsum(node_counts, dim=0) - node_counts
        Hon_split = torch.split(Hon, node_counts.tolist(), dim=0)
        Son_split = torch.split(data.Son, node_counts.tolist(), dim=0)
        #
        edge_num = torch.ones_like(j)
        edge_num = scatter(edge_num, data.batch[j], dim=0) # shape: (Nbatch,)
        edge_num_shift = torch.cumsum(edge_num, dim=0) - edge_num
        Hoff_split = torch.split(Hoff, edge_num.tolist(), dim=0)
        Soff_split = torch.split(data.Soff, edge_num.tolist(), dim=0)
        if export_reciprocal_values:
            dSon_split = torch.split(data.dSon, node_counts.tolist(), dim=0)
            dSoff_split = torch.split(data.dSoff, edge_num.tolist(), dim=0)
        
        band_energy = []
        wavefunction = []
        H_reciprocal = []
        H_sym = []
        S_reciprocal = []
        dS_reciprocal = []
        gap = []
        for idx in range(Nbatch):
            k_vec = data.k_vecs[idx]   
            natoms = data.node_counts[idx]
            
            # Initialize HK and SK       
            coe = torch.exp(2j*torch.pi*torch.sum(data.nbr_shift[edge_num_shift[idx]+torch.arange(edge_num[idx]).type_as(j),None,:]*k_vec[None,:,:], axis=-1)) # (nedges, 1, 3)*(1, num_k, 3) -> (nedges, num_k)     
            
            HK = torch.view_as_complex(torch.zeros((self.num_k, natoms, natoms, self.nao_max, self.nao_max, 2)).type_as(Hon))
            SK = torch.view_as_complex(torch.zeros((self.num_k, natoms, natoms, self.nao_max, self.nao_max, 2)).type_as(Hon))            
            if export_reciprocal_values:
                dSK = torch.view_as_complex(torch.zeros((self.num_k, natoms, natoms, self.nao_max, self.nao_max, 3, 2)).type_as(Hon))

            na = torch.arange(natoms).type_as(j)
            HK[:,na,na,:,:] +=  Hon_split[idx].reshape(-1, self.nao_max, self.nao_max)[None,na,:,:].type_as(HK) # shape (num_k, natoms, nao_max, nao_max)
            SK[:,na,na,:,:] +=  Son_split[idx].reshape(-1, self.nao_max, self.nao_max)[None,na,:,:].type_as(SK)
            if export_reciprocal_values:
                dSK[:,na,na,:,:,:] +=  dSon_split[idx].reshape(-1, self.nao_max, self.nao_max, 3)[None,na,:,:,:].type_as(dSK)

            
            for iedge in range(edge_num[idx]):
                # shape (num_k, nao_max, nao_max) += (num_k, 1, 1)*(1, nao_max, nao_max)
                j_idx = j[edge_num_shift[idx]+iedge] - node_counts_shift[idx]
                i_idx = i[edge_num_shift[idx]+iedge] - node_counts_shift[idx]
                HK[:,j_idx,i_idx,:,:] += coe[iedge,:,None,None] * Hoff_split[idx].reshape(-1, self.nao_max, self.nao_max)[None,iedge,:,:]
                SK[:,j_idx,i_idx,:,:] += coe[iedge,:,None,None] * Soff_split[idx].reshape(-1, self.nao_max, self.nao_max)[None,iedge,:,:]
            
            if export_reciprocal_values:
                for iedge in range(edge_num[idx]):
                    j_idx = j[edge_num_shift[idx]+iedge] - node_counts_shift[idx]
                    i_idx = i[edge_num_shift[idx]+iedge] - node_counts_shift[idx]
                    dSK[:,j_idx,i_idx,:,:,:] += coe[iedge,:,None,None,None] * dSoff_split[idx].reshape(-1, self.nao_max, self.nao_max, 3)[None,iedge,:,:,:]

            HK = torch.swapaxes(HK,-2,-3) #(nk, natoms, nao_max, natoms, nao_max)
            HK = HK.reshape(-1, natoms*self.nao_max, natoms*self.nao_max)
            SK = torch.swapaxes(SK,-2,-3) #(nk, natoms, nao_max, natoms, nao_max)
            SK = SK.reshape(-1, natoms*self.nao_max, natoms*self.nao_max)
            if export_reciprocal_values:
                dSK = torch.swapaxes(dSK,-3,-4) #(nk, natoms, nao_max, natoms, nao_max, 3)
                dSK = dSK.reshape(-1, natoms*self.nao_max, natoms*self.nao_max, 3)
            
            # mask HK and SK
            HK = torch.masked_select(HK, orb_mask_batch[idx].repeat(self.num_k,1,1) > 0)
            norbs = int(math.sqrt(HK.numel()/self.num_k))
            HK = HK.reshape(self.num_k, norbs, norbs)
            
            SK = torch.masked_select(SK, orb_mask_batch[idx].repeat(self.num_k,1,1) > 0)
            norbs = int(math.sqrt(SK.numel()/self.num_k))
            SK = SK.reshape(self.num_k, norbs, norbs)
            if export_reciprocal_values:
                dSK = torch.masked_select(dSK, orb_mask_batch[idx].unsqueeze(-1).repeat(self.num_k,1,1,3) > 0)
                dSK = dSK.reshape(self.num_k, norbs, norbs, 3)            
            
            # Calculate band energies
            L = torch.linalg.cholesky(SK)
            L_t = torch.transpose(L.conj(), dim0=-1, dim1=-2)
            L_inv = torch.linalg.inv(L)
            L_t_inv = torch.linalg.inv(L_t)
            Hs = torch.bmm(torch.bmm(L_inv, HK), L_t_inv)
            orbital_energies, orbital_coefficients = torch.linalg.eigh(Hs)        
            
            # Convert the wavefunction coefficients back to the original basis
            orbital_coefficients = torch.einsum('ijk, ika -> iaj', L_t_inv, orbital_coefficients)
            
            # Numpy
            """
            HK_t = HK.detach().cpu().numpy()
            SK_t = SK.detach().cpu().numpy()
            from scipy.linalg import eigh
            eigen = []
            eigen_vecs = []
            for ik in range(self.num_k):
                w, v = eigh(a=HK_t[ik], b=SK_t[ik])
                eigen.append(w)
                eigen_vecs.append(v)
            eigen_vecs = np.array(eigen_vecs) # (nk, nbands, nbands)
            eigen_vecs = np.swapaxes(eigen_vecs, -1, -2)
            
            lamda = np.einsum('nai, nij, naj -> na', np.conj(eigen_vecs), SK_t, eigen_vecs).real
            lamda = 1/np.sqrt(lamda) # shape: (numk, norbs)
            eigen_vecs = eigen_vecs*lamda[:,:,None]  
            orbital_energies, orbital_coefficients = torch.Tensor(eigen).type_as(data.pos), torch.complex(torch.Tensor(eigen_vecs.real), torch.Tensor(eigen_vecs.imag)).type_as(HK)
            """
            
            if export_reciprocal_values:
                # Normalize wave function
                lamda = torch.einsum('nai, nij, naj -> na', torch.conj(orbital_coefficients), SK, orbital_coefficients).real
                lamda = 1/torch.sqrt(lamda) # shape: (numk, norbs)
                orbital_coefficients = orbital_coefficients*lamda.unsqueeze(-1)    
                        
                H_reciprocal.append(HK)
                S_reciprocal.append(SK)
                dS_reciprocal.append(dSK)
            
            numc = math.ceil(num_val[idx]/2)
            gap.append((torch.min(orbital_energies[:,numc]) - torch.max(orbital_energies[:,numc-1])).unsqueeze(0))
            if self.band_num_control is not None:
                if isinstance(self.band_num_control, dict):
                    orbital_energies = orbital_energies[:,:band_num_win[idx]]   
                    orbital_coefficients = orbital_coefficients[:,:band_num_win[idx],:]
                else:
                    if isinstance(self.band_num_control, float):
                        self.band_num_control = max([1, int(self.band_num_control*numc)])
                    else:
                        self.band_num_control = min([self.band_num_control, numc])
                    orbital_energies = orbital_energies[:,numc-self.band_num_control:numc+self.band_num_control]   
                    orbital_coefficients = orbital_coefficients[:,numc-self.band_num_control:numc+self.band_num_control,:]               
            band_energy.append(torch.transpose(orbital_energies, dim0=-1, dim1=-2)) # [shape:(Nbands, num_k)]
            wavefunction.append(orbital_coefficients)  
            H_sym.append(Hs.view(-1))   
            
        band_energy = torch.cat(band_energy, dim=0) # [shape:(Nbands, num_k)]
        
        gap = torch.cat(gap, dim=0)
        
        if export_reciprocal_values:
            wavefunction = torch.stack(wavefunction, dim=0) # shape:[Nbatch, num_k, norbs, norbs]
            HK = torch.stack(H_reciprocal, dim=0) # shape:[Nbatch, num_k, norbs, norbs]
            SK = torch.stack(S_reciprocal, dim=0) # shape:[Nbatch, num_k, norbs, norbs]  
            dSK = torch.stack(dS_reciprocal, dim=0) # shape:[Nbatch, num_k, norbs, norbs, 3]   
            return band_energy, wavefunction, HK, SK, dSK, gap
        else:
            wavefunction = [wavefunction[idx].reshape(-1) for idx in range(Nbatch)]
            wavefunction = torch.cat(wavefunction, dim=0) # shape:[Nbatch*num_k*norbs*norbs]
            H_sym = torch.cat(H_sym, dim=0) # shape:(Nbatch*num_k*norbs*norbs)
            return band_energy, wavefunction, gap, H_sym  
    
    def cal_band_energy_soc(self, Hsoc_on_real, Hsoc_on_imag, Hsoc_off_real, Hsoc_off_imag, data):
        """
        Currently this function can only be used to calculate the energy band of the openmx Hamiltonian.
        """
        j, i = data.edge_index
        cell = data.cell # shape:(Nbatch, 3, 3)
        Nbatch = cell.shape[0]
        
        Hsoc_on_real = Hsoc_on_real.reshape(-1, 2*self.nao_max, 2*self.nao_max)
        Hsoc_on_imag = Hsoc_on_imag.reshape(-1, 2*self.nao_max, 2*self.nao_max)
        Hsoc_off_real = Hsoc_off_real.reshape(-1, 2*self.nao_max, 2*self.nao_max) 
        Hsoc_off_imag = Hsoc_off_imag.reshape(-1, 2*self.nao_max, 2*self.nao_max)
        
        # parse the Atomic Orbital Basis Sets
        basis_definition = torch.zeros((99, self.nao_max)).type_as(data.z)
        # key is the atomic number, value is the index of the occupied orbits.
        for k in self.basis_def.keys():
            basis_definition[k][self.basis_def[k]] = 1
            
        orb_mask = basis_definition[data.z] # shape: [Natoms, nao_max] 
        orb_mask = torch.split(orb_mask, data.node_counts.tolist(), dim=0) # shape: [natoms, nao_max]
        orb_mask_batch = []
        for idx in range(Nbatch):
            orb_mask_batch.append(orb_mask[idx].reshape(-1, 1)* orb_mask[idx].reshape(1, -1)) # shape: [natoms*nao_max, natoms*nao_max]
        
        # Set the number of valence electrons
        num_val = torch.zeros((99,)).type_as(data.z)
        for k in self.num_valence.keys():
            num_val[k] = self.num_valence[k]
        num_val = num_val[data.z] # shape: [Natoms]
        num_val = scatter(num_val, data.batch, dim=0) # shape: [Nbatch]
                
        # Initialize band_num_win
        if isinstance(self.band_num_control, dict):
            band_num_win = torch.zeros((99,)).type_as(data.z)
            for k in self.band_num_control.keys():
                band_num_win[k] = self.band_num_control[k]
            band_num_win = band_num_win[data.z] # shape: [Natoms,]   
            band_num_win = scatter(band_num_win, data.batch, dim=0) # shape: (Nbatch,)       
            
        # Separate Hon and Hoff for each batch
        node_counts = data.node_counts
        Hon_split = torch.split(Hsoc_on_real, node_counts.tolist(), dim=0)
        iHon_split = torch.split(Hsoc_on_imag, node_counts.tolist(), dim=0)
        Son_split = torch.split(data.Son.reshape(-1, self.nao_max, self.nao_max), node_counts.tolist(), dim=0)
        #
        edge_num = torch.ones_like(j)
        edge_num = scatter(edge_num, data.batch[j], dim=0)
        Hoff_split = torch.split(Hsoc_off_real, edge_num.tolist(), dim=0)
        iHoff_split = torch.split(Hsoc_off_imag, edge_num.tolist(), dim=0)
        Soff_split = torch.split(data.Soff.reshape(-1, self.nao_max, self.nao_max), edge_num.tolist(), dim=0)
        
        cell_shift_split = torch.split(data.cell_shift, edge_num.tolist(), dim=0)
        nbr_shift_split = torch.split(data.nbr_shift, edge_num.tolist(), dim=0)
        edge_index_split = torch.split(data.edge_index, edge_num.tolist(), dim=1)
        node_num = torch.cumsum(node_counts, dim=0) - node_counts
        edge_index_split = [edge_index_split[idx]-node_num[idx] for idx in range(len(node_num))]
        
        band_energy = []
        wavefunction = []
        for idx in range(Nbatch):
            k_vec = data.k_vecs[idx]   
            natoms = data.node_counts[idx].item() 
            
            # Initialize cell index
            cell_shift_tuple = [tuple(c) for c in cell_shift_split[idx].detach().cpu().tolist()]
            cell_shift_set = set(cell_shift_tuple)
            cell_shift_list = list(cell_shift_set)
            cell_index = [cell_shift_list.index(icell) for icell in cell_shift_tuple]
            cell_index = torch.LongTensor(cell_index).type_as(data.edge_index)
            ncells = len(cell_shift_set)
            
            # Initialize SK
            phase = torch.view_as_complex(torch.zeros((self.num_k, ncells, 2)).type_as(data.Son))
            phase[:, cell_index] = torch.exp(2j*torch.pi*torch.sum(nbr_shift_split[idx][None,:,:]*k_vec[:,None,:], dim=-1))
            na = torch.arange(natoms).type_as(j)

            S_cell = torch.view_as_complex(torch.zeros((ncells, natoms, natoms, self.nao_max, self.nao_max, 2)).type_as(data.Son))
            S_cell[cell_index, edge_index_split[idx][0], edge_index_split[idx][1], :, :] += Soff_split[idx]

            SK = torch.einsum('ijklm, ni->njklm', S_cell, phase) # (nk, natoms, natoms, nao_max, nao_max)
            SK[:,na,na,:,:] +=  Son_split[idx][None,na,:,:]
            SK = torch.swapaxes(SK,2,3) #(nk, natoms, nao_max, natoms, nao_max)
            SK = SK.reshape(self.num_k, natoms*self.nao_max, natoms*self.nao_max)
            SK = SK[:,orb_mask_batch[idx] > 0]
            norbs = int(math.sqrt(SK.numel()/self.num_k))
            SK = SK.reshape(self.num_k, norbs, norbs)
            I = torch.eye(2).type_as(data.Hon)
            SK = torch.kron(I, SK)
            
            # Initialize Hsoc
            # on-site term 
            H11 = Hon_split[idx][:,:self.nao_max,:self.nao_max] + 1.0j*iHon_split[idx][:,:self.nao_max,:self.nao_max] # up-up
            H12 = Hon_split[idx][:,:self.nao_max, self.nao_max:] + 1.0j*iHon_split[idx][:,:self.nao_max,self.nao_max:] # up-down
            H21 = Hon_split[idx][:,self.nao_max:,:self.nao_max] + 1.0j*iHon_split[idx][:,self.nao_max:,:self.nao_max] # down-up
            H22 = Hon_split[idx][:,self.nao_max:,self.nao_max:] + 1.0j*iHon_split[idx][:,self.nao_max:,self.nao_max:] # down-down
            Hon_soc = [H11, H12, H21, H22]
            # off-site term
            H11 = Hoff_split[idx][:,:self.nao_max,:self.nao_max] + 1.0j*iHoff_split[idx][:,:self.nao_max,:self.nao_max] # up-up
            H12 = Hoff_split[idx][:,:self.nao_max, self.nao_max:] + 1.0j*iHoff_split[idx][:,:self.nao_max,self.nao_max:] # up-down
            H21 = Hoff_split[idx][:,self.nao_max:,:self.nao_max] + 1.0j*iHoff_split[idx][:,self.nao_max:,:self.nao_max] # down-up
            H22 = Hoff_split[idx][:,self.nao_max:,self.nao_max:] + 1.0j*iHoff_split[idx][:,self.nao_max:,self.nao_max:] # down-down
            Hoff_soc = [H11, H12, H21, H22]
            
            # Initialize HK
            HK_list = []
            for Hon, Hoff in zip(Hon_soc, Hoff_soc):
                H_cell = torch.view_as_complex(torch.zeros((ncells, natoms, natoms, self.nao_max, self.nao_max, 2)).type_as(data.Son))
                H_cell[cell_index, edge_index_split[idx][0], edge_index_split[idx][1], :, :] += Hoff    

                HK = torch.einsum('ijklm, ni->njklm', H_cell, phase) # (nk, natoms, natoms, nao_max, nao_max)
                HK[:,na,na,:,:] +=  Hon[None,na,:,:] # shape (nk, natoms, nao_max, nao_max)

                HK = torch.swapaxes(HK,2,3) #(nk, natoms, nao_max, natoms, nao_max)
                HK = HK.reshape(self.num_k, natoms*self.nao_max, natoms*self.nao_max)

                # mask HK
                HK = HK[:, orb_mask_batch[idx] > 0]
                norbs = int(math.sqrt(HK.numel()/self.num_k))
                HK = HK.reshape(self.num_k, norbs, norbs)
        
                HK_list.append(HK)

            HK = torch.cat([torch.cat([HK_list[0],HK_list[1]], dim=-1), torch.cat([HK_list[2],HK_list[3]], dim=-1)],dim=-2)
        
            # Calculate band energies
            L = torch.linalg.cholesky(SK)
            L_t = torch.transpose(L.conj(), dim0=-1, dim1=-2)
            L_inv = torch.linalg.inv(L)
            L_t_inv = torch.linalg.inv(L_t)
            Hs = torch.bmm(torch.bmm(L_inv, HK), L_t_inv)
            orbital_energies, orbital_coefficients = torch.linalg.eigh(Hs)   
            # Convert the wavefunction coefficients back to the original basis
            orbital_coefficients = torch.bmm(L_t_inv, orbital_coefficients) # shape:(num_k, Nbands, Nbands)
            if self.band_num_control is not None:
                if isinstance(self.band_num_control, dict):
                    orbital_energies = orbital_energies[:,:band_num_win[idx]]   
                    orbital_coefficients = orbital_coefficients[:,:band_num_win[idx],:]
                else:
                    orbital_energies = orbital_energies[:,num_val[idx]-self.band_num_control:num_val[idx]+self.band_num_control]   
                    orbital_coefficients = orbital_coefficients[:,num_val[idx]-self.band_num_control:num_val[idx]+self.band_num_control,:]
            band_energy.append(torch.transpose(orbital_energies, dim0=-1, dim1=-2)) # [shape:(Nbands, num_k)]
            wavefunction.append(orbital_coefficients)
        return torch.cat(band_energy, dim=0), torch.cat(wavefunction, dim=0).reshape(-1)
    
    def mask_Ham(self, Hon, Hoff, data):
        # parse the Atomic Orbital Basis Sets
        basis_definition = torch.zeros((99, self.nao_max)).type_as(data.z)
        # key is the atomic number, value is the index of the occupied orbits.
        for k in self.basis_def.keys():
            basis_definition[k][self.basis_def[k]] = 1
        
        # Save the original shape
        original_shape_on = Hon.shape
        original_shape_off = Hoff.shape
        
        if len(original_shape_on) > 2:
            Hon = Hon.reshape(original_shape_on[0], -1)
        if len(original_shape_off) > 2:
            Hoff = Hoff.reshape(original_shape_off[0], -1)
        
        # mask Hon first        
        orb_mask = basis_definition[data.z].view(-1, self.nao_max) # shape: [Natoms, nao_max] 
        orb_mask = orb_mask[:,:,None] * orb_mask[:,None,:]       # shape: [Natoms, nao_max, nao_max]
        orb_mask = orb_mask.reshape(-1, int(self.nao_max*self.nao_max)) # shape: [Natoms, nao_max*nao_max]
        
        Hon_mask = torch.zeros_like(Hon)
        Hon_mask[orb_mask>0] = Hon[orb_mask>0]
        
        # mask Hoff
        j, i = data.edge_index        
        orb_mask_j = basis_definition[data.z[j]].view(-1, self.nao_max) # shape: [Nedges, nao_max]
        orb_mask_i = basis_definition[data.z[i]].view(-1, self.nao_max) # shape: [Nedges, nao_max] 
        orb_mask = orb_mask_j[:,:,None] * orb_mask_i[:,None,:]       # shape: [Nedges, nao_max, nao_max]
        orb_mask = orb_mask.reshape(-1, int(self.nao_max*self.nao_max)) # shape: [Nedges, nao_max*nao_max]
        
        Hoff_mask = torch.zeros_like(Hoff)
        Hoff_mask[orb_mask>0] = Hoff[orb_mask>0]

        # Output the result in the original shape
        Hon_mask = Hon_mask.reshape(original_shape_on)
        Hoff_mask = Hoff_mask.reshape(original_shape_off)
        
        return Hon_mask, Hoff_mask    
    
    def construct_Hsoc(self, H, iH):
        Hsoc = torch.view_as_complex(torch.zeros((H.shape[0], (2*self.nao_max)**2, 2)).type_as(H))
        Hsoc = H + 1.0j*iH
        return Hsoc
    
    def reduce(self, coefficient):
        if self.nao_max == 14:
            coefficient = coefficient.reshape(coefficient.shape[0], self.nao_max, self.nao_max)
            coefficient[:, 3:6] = torch.mean(coefficient[:, 3:6], dim=1, keepdim=True).expand(coefficient.shape[0], 3, self.nao_max)
            coefficient[:, 6:9] = torch.mean(coefficient[:, 6:9], dim=1, keepdim=True).expand(coefficient.shape[0], 3, self.nao_max)
            coefficient[:, 9:14] = torch.mean(coefficient[:, 9:14], dim=1, keepdim=True).expand(coefficient.shape[0], 5, self.nao_max)
            #
            coefficient[:, :, 3:6] = torch.mean(coefficient[:, :, 3:6], dim=2, keepdim=True).expand(coefficient.shape[0], self.nao_max, 3)
            coefficient[:, :, 6:9] = torch.mean(coefficient[:, :, 6:9], dim=2, keepdim=True).expand(coefficient.shape[0], self.nao_max, 3)
            coefficient[:, :, 9:14] = torch.mean(coefficient[:, :, 9:14], dim=2, keepdim=True).expand(coefficient.shape[0], self.nao_max, 5)
            
        elif self.nao_max == 19:
            coefficient = coefficient.reshape(coefficient.shape[0], self.nao_max, self.nao_max)
            coefficient[:, 3:6] = torch.mean(coefficient[:, 3:6], dim=1, keepdim=True).expand(coefficient.shape[0], 3, self.nao_max)
            coefficient[:, 6:9] = torch.mean(coefficient[:, 6:9], dim=1, keepdim=True).expand(coefficient.shape[0], 3, self.nao_max)
            coefficient[:, 9:14] = torch.mean(coefficient[:, 9:14], dim=1, keepdim=True).expand(coefficient.shape[0], 5, self.nao_max)
            coefficient[:, 14:19] = torch.mean(coefficient[:, 14:19], dim=1, keepdim=True).expand(coefficient.shape[0], 5, self.nao_max)
            #
            coefficient[:, :, 3:6] = torch.mean(coefficient[:, :, 3:6], dim=2, keepdim=True).expand(coefficient.shape[0], self.nao_max, 3)
            coefficient[:, :, 6:9] = torch.mean(coefficient[:, :, 6:9], dim=2, keepdim=True).expand(coefficient.shape[0], self.nao_max, 3)
            coefficient[:, :, 9:14] = torch.mean(coefficient[:, :, 9:14], dim=2, keepdim=True).expand(coefficient.shape[0], self.nao_max, 5)
            coefficient[:, :, 14:19] = torch.mean(coefficient[:, :, 14:19], dim=2, keepdim=True).expand(coefficient.shape[0], self.nao_max, 5)

        elif self.nao_max == 26:
            coefficient = coefficient.reshape(coefficient.shape[0], self.nao_max, self.nao_max)
            coefficient[:, 3:6] = torch.mean(coefficient[:, 3:6], dim=1, keepdim=True).expand(coefficient.shape[0], 3, self.nao_max)
            coefficient[:, 6:9] = torch.mean(coefficient[:, 6:9], dim=1, keepdim=True).expand(coefficient.shape[0], 3, self.nao_max)
            coefficient[:, 9:14] = torch.mean(coefficient[:, 9:14], dim=1, keepdim=True).expand(coefficient.shape[0], 5, self.nao_max)
            coefficient[:, 14:19] = torch.mean(coefficient[:, 14:19], dim=1, keepdim=True).expand(coefficient.shape[0], 5, self.nao_max)
            coefficient[:, 19:26] = torch.mean(coefficient[:, 19:26], dim=1, keepdim=True).expand(coefficient.shape[0], 7, self.nao_max)
            #
            coefficient[:, :, 3:6] = torch.mean(coefficient[:, :, 3:6], dim=2, keepdim=True).expand(coefficient.shape[0], self.nao_max, 3)
            coefficient[:, :, 6:9] = torch.mean(coefficient[:, :, 6:9], dim=2, keepdim=True).expand(coefficient.shape[0], self.nao_max, 3)
            coefficient[:, :, 9:14] = torch.mean(coefficient[:, :, 9:14], dim=2, keepdim=True).expand(coefficient.shape[0], self.nao_max, 5)
            coefficient[:, :, 14:19] = torch.mean(coefficient[:, :, 14:19], dim=2, keepdim=True).expand(coefficient.shape[0], self.nao_max, 5)
            coefficient[:, :, 19:26] = torch.mean(coefficient[:, :, 19:26], dim=2, keepdim=True).expand(coefficient.shape[0], self.nao_max, 7)
        return coefficient.view(coefficient.shape[0], -1)
         
    def forward(self, data, graph_representation: dict = None):
        # prepare data.hamiltonian & data.overlap
        if 'hamiltonian' not in data:
            data.hamiltonian = self.cat_onsite_and_offsite(data, data.Hon, data.Hoff)
        if 'overlap' not in data:
            data.overlap = self.cat_onsite_and_offsite(data, data.Son, data.Soff)
        
        node_attr = graph_representation['node_attr']
        edge_attr = graph_representation['edge_attr']  # mji
        j, i = data.edge_index
        
        # Calculate inv_edge_index in batch
        inv_edge_idx = data.inv_edge_idx
        edge_num = torch.ones_like(j)
        edge_num = scatter(edge_num, data.batch[j], dim=0)
        edge_num = torch.cumsum(edge_num, dim=0) - edge_num
        inv_edge_idx = inv_edge_idx + edge_num[data.batch[j]]
        
        # Calculate the on-site Hamiltonian 
        self.ham_irreps_dim = self.ham_irreps_dim.type_as(j)  
        
        if not self.ham_only:
            node_sph = self.onsitenet_s(node_attr)
            node_sph = torch.split(node_sph, self.ham_irreps_dim.tolist(), dim=-1)
            Son = self.matrix_merge(node_sph) # shape (Nnodes, nao_max**2)
            
            Son = self.change_index(Son)
        
            # Impose Hermitian symmetry for Son
            Son = self.symmetrize_Hon(Son)

            # Calculate the off-site overlap
            # Calculate the contribution of the edges       
            edge_sph = self.offsitenet_s(edge_attr)
            edge_sph = torch.split(edge_sph, self.ham_irreps_dim.tolist(), dim=-1)        
            Soff = self.matrix_merge(edge_sph)
        
            Soff = self.change_index(Soff)        
            # Impose Hermitian symmetry for Soff
            Soff = self.symmetrize_Hoff(Soff, inv_edge_idx)
        
            if self.ham_type in ['openmx','pasp', 'siesta', 'abacus']:
                Son, Soff = self.mask_Ham(Son, Soff, data)

        if self.soc_switch:
            if self.soc_basis == 'so3':
                node_sph = self.onsitenet_residual(node_attr)     
                node_sph = self.onsitenet_linear(node_sph) 
                node_sph = torch.split(node_sph, self.ham_irreps_dim.tolist(), dim=-1)
                Hon = self.matrix_merge(node_sph) # shape (Nnodes, nao_max**2)
                
                Hon = self.change_index(Hon)
    
                # Impose Hermitian symmetry for Hon
                Hon = self.symmetrize_Hon(Hon)            
    
                # Calculate the off-site Hamiltonian
                # Calculate the contribution of the edges       
                edge_sph = self.offsitenet_residual(edge_attr)
                edge_sph = self.offsitenet_linear(edge_sph)
                edge_sph = torch.split(edge_sph, self.ham_irreps_dim.tolist(), dim=-1)        
                Hoff = self.matrix_merge(edge_sph)
            
                Hoff = self.change_index(Hoff)        
                # Impose Hermitian symmetry for Hoff
                Hoff = self.symmetrize_Hoff(Hoff, inv_edge_idx)
                
                Hon, Hoff = self.mask_Ham(Hon, Hoff, data)
                
                # build Hsoc
                node_sph = self.onsitenet_residual_ksi(node_attr)
                ksi_on = self.ksi_on_scalar(node_sph)
                ksi_on = self.reduce(ksi_on)
                
                edge_sph = self.offsitenet_residual_ksi(edge_attr)
                ksi_off = self.ksi_off_scalar(edge_sph)
                ksi_off = self.reduce(ksi_off)  
                
                Hsoc_on_real = torch.zeros((Hon.shape[0], 2*self.nao_max, 2*self.nao_max)).type_as(Hon)
                Hsoc_on_real[:,:self.nao_max,:self.nao_max] = Hon.reshape(-1, self.nao_max, self.nao_max)
                Hsoc_on_real[:,:self.nao_max,self.nao_max:] = self.symmetrize_Hon((ksi_on*data.Lon[:,:,1]), sign='-').reshape(-1, self.nao_max, self.nao_max)
                Hsoc_on_real[:,self.nao_max:,:self.nao_max] = self.symmetrize_Hon((ksi_on*data.Lon[:,:,1]), sign='-').reshape(-1, self.nao_max, self.nao_max)
                Hsoc_on_real[:,self.nao_max:,self.nao_max:] = Hon.reshape(-1, self.nao_max, self.nao_max)
                Hsoc_on_real = Hsoc_on_real.reshape(-1, (2*self.nao_max)**2)
                
                Hsoc_on_imag = torch.zeros((Hon.shape[0], 2*self.nao_max, 2*self.nao_max)).type_as(Hon)
                Hsoc_on_imag[:,:self.nao_max,:self.nao_max] = self.symmetrize_Hon((ksi_on*data.Lon[:,:,2]), sign='-').reshape(-1, self.nao_max, self.nao_max)
                Hsoc_on_imag[:,:self.nao_max, self.nao_max:] = self.symmetrize_Hon((ksi_on*data.Lon[:,:,0]), sign='-').reshape(-1, self.nao_max, self.nao_max)
                Hsoc_on_imag[:,self.nao_max:,:self.nao_max] = -self.symmetrize_Hon((ksi_on*data.Lon[:,:,0]), sign='-').reshape(-1, self.nao_max, self.nao_max)
                Hsoc_on_imag[:,self.nao_max:,self.nao_max:] = -self.symmetrize_Hon((ksi_on*data.Lon[:,:,2]), sign='-').reshape(-1, self.nao_max, self.nao_max)
                Hsoc_on_imag = Hsoc_on_imag.reshape(-1, (2*self.nao_max)**2)
                
                Hsoc_off_real = torch.zeros((Hoff.shape[0], 2*self.nao_max, 2*self.nao_max)).type_as(Hoff)
                Hsoc_off_real[:,:self.nao_max,:self.nao_max] = Hoff.reshape(-1, self.nao_max, self.nao_max)
                Hsoc_off_real[:,:self.nao_max,self.nao_max:] = self.symmetrize_Hoff((ksi_off*data.Loff[:,:,1]), inv_edge_idx, sign='-').reshape(-1, self.nao_max, self.nao_max)
                Hsoc_off_real[:,self.nao_max:,:self.nao_max] = self.symmetrize_Hoff((ksi_off*data.Loff[:,:,1]), inv_edge_idx, sign='-').reshape(-1, self.nao_max, self.nao_max)
                Hsoc_off_real[:,self.nao_max:,self.nao_max:] = Hoff.reshape(-1, self.nao_max, self.nao_max)
                Hsoc_off_real = Hsoc_off_real.reshape(-1, (2*self.nao_max)**2)
                
                Hsoc_off_imag = torch.zeros((Hoff.shape[0], 2*self.nao_max, 2*self.nao_max)).type_as(Hoff)
                Hsoc_off_imag[:,:self.nao_max,:self.nao_max] = self.symmetrize_Hoff((ksi_off*data.Loff[:,:,2]), inv_edge_idx, sign='-').reshape(-1, self.nao_max, self.nao_max)
                Hsoc_off_imag[:,:self.nao_max, self.nao_max:] = self.symmetrize_Hoff((ksi_off*data.Loff[:,:,0]), inv_edge_idx, sign='-').reshape(-1, self.nao_max, self.nao_max)
                Hsoc_off_imag[:,self.nao_max:,:self.nao_max] = -self.symmetrize_Hoff((ksi_off*data.Loff[:,:,0]), inv_edge_idx, sign='-').reshape(-1, self.nao_max, self.nao_max)
                Hsoc_off_imag[:,self.nao_max:,self.nao_max:] = -self.symmetrize_Hoff((ksi_off*data.Loff[:,:,2]), inv_edge_idx, sign='-').reshape(-1, self.nao_max, self.nao_max)
                Hsoc_off_imag = Hsoc_off_imag.reshape(-1, (2*self.nao_max)**2)
            
            elif self.soc_basis == 'su2':
                node_sph = self.onsitenet_residual(node_attr)
                node_sph = self.onsitenet_linear(node_sph) 
                
                Hon = self.hamDecomp.get_H(node_sph) # shape [Nbatchs, (4 spin components,) H_flattened_concatenated]
                Hon = self.change_index(Hon)
                Hon = Hon.reshape(-1, 2, 2, self.nao_max, self.nao_max)                
                Hon = torch.swapaxes(Hon, 2, 3) # shape (Nnodes, 2, nao_max, 2, nao_max)
    
                # Calculate the off-site Hamiltonian
                # Calculate the contribution of the edges       
                edge_sph = self.offsitenet_residual(edge_attr)
                edge_sph = self.offsitenet_linear(edge_sph)
                
                Hoff = self.hamDecomp.get_H(edge_sph) # shape [Nbatchs, (4 spin components,) H_flattened_concatenated]
                Hoff = self.change_index(Hoff)
                Hoff = Hoff.reshape(-1, 2, 2, self.nao_max, self.nao_max)
                Hoff = torch.swapaxes(Hoff, 2, 3) # shape (Nedges, 2, nao_max, 2, nao_max)    
                
                # mask zeros         
                for i in range(2):
                    for j in range(2):
                        Hon[:,i,:,j,:], Hoff[:,i,:,j,:] = self.mask_Ham(Hon[:,i,:,j,:], Hoff[:,i,:,j,:], data)
                Hon = Hon.reshape(-1, (2*self.nao_max)**2)
                Hoff = Hoff.reshape(-1, (2*self.nao_max)**2)
                # build four parts
                Hsoc_on_real =  Hon.real
                Hsoc_off_real = Hoff.real
                Hsoc_on_imag = Hon.imag
                Hsoc_off_imag = Hoff.imag
                
            else:
                raise NotImplementedError
            
            if self.add_H0:
                Hsoc_on_real =  Hsoc_on_real + data.Hon0
                Hsoc_off_real = Hsoc_off_real + data.Hoff0
                Hsoc_on_imag = Hsoc_on_imag + data.iHon0
                Hsoc_off_imag = Hsoc_off_imag + data.iHoff0
            
            Hsoc_real = self.cat_onsite_and_offsite(data, Hsoc_on_real, Hsoc_off_real)
            Hsoc_imag = self.cat_onsite_and_offsite(data, Hsoc_on_imag, Hsoc_off_imag)
            
            data.hamiltonian_real = self.cat_onsite_and_offsite(data, data.Hon, data.Hoff)
            data.hamiltonian_imag = self.cat_onsite_and_offsite(data, data.iHon, data.iHoff)
            
            #Hsoc = self.construct_Hsoc(Hsoc_real, Hsoc_imag)
            #data.hamiltonian = self.construct_Hsoc(data.hamiltonian_real, data.hamiltonian_imag)
            
            Hsoc = torch.cat((Hsoc_real, Hsoc_imag), dim=0)
            data.hamiltonian = torch.cat((data.hamiltonian_real, data.hamiltonian_imag), dim=0)
            
            if self.calculate_band_energy:
                k_vecs = []
                for idx in range(data.batch[-1]+1):
                    cell = data.cell
                    # Generate K point path
                    if self.k_path is not None:
                        kpts=kpoints_generator(dim_k=3, lat=cell[idx].detach().cpu().numpy())
                        k_vec, k_dist, k_node, lat_per_inv = kpts.k_path(self.k_path, self.num_k)
                    else:
                        lat_per_inv=np.linalg.inv(cell[idx].detach().cpu().numpy()).T
                        k_vec = 2.0*np.random.rand(self.num_k, 3)-1.0 #(-1, 1)
                    k_vec = k_vec.dot(lat_per_inv[np.newaxis,:,:]) # shape (nk,1,3)
                    k_vec = k_vec.reshape(-1,3) # shape (nk, 3)
                    k_vec = torch.Tensor(k_vec).type_as(Hon)
                    k_vecs.append(k_vec)  
                data.k_vecs = torch.stack(k_vecs, dim=0)
                band_energy, wavefunction = self.cal_band_energy_soc(Hsoc_on_real, Hsoc_on_imag, Hsoc_off_real, Hsoc_off_imag, data) 
                with torch.no_grad():
                    data.band_energy, data.wavefunction = self.cal_band_energy_soc(data.Hon, data.iHon, data.Hoff, data.iHoff, data)
            else:
                band_energy = None
                wavefunction = None
        else:                
            node_sph = self.onsitenet_residual(node_attr)
            node_sph = self.onsitenet_linear(node_sph) 
            node_sph = torch.split(node_sph, self.ham_irreps_dim.tolist(), dim=-1)
            Hon = self.matrix_merge(node_sph) # shape (Nnodes, nao_max**2)
            
            Hon = self.change_index(Hon)
        
            # Impose Hermitian symmetry for Hon
            Hon = self.symmetrize_Hon(Hon)
            if self.add_H0:
                Hon = Hon + data.Hon0
               
            # Calculate the off-site Hamiltonian
            # Calculate the contribution of the edges       
            edge_sph = self.offsitenet_residual(edge_attr)
            edge_sph = self.offsitenet_linear(edge_sph)
            edge_sph = torch.split(edge_sph, self.ham_irreps_dim.tolist(), dim=-1)        
            Hoff = self.matrix_merge(edge_sph)
        
            Hoff = self.change_index(Hoff)        
            # Impose Hermitian symmetry for Hoff
            Hoff = self.symmetrize_Hoff(Hoff, inv_edge_idx)
            if self.add_H0:
                Hoff = Hoff + data.Hoff0
        
            if self.ham_type in ['openmx','pasp', 'siesta', 'abacus']:
                Hon, Hoff = self.mask_Ham(Hon, Hoff, data)
        
            if self.calculate_band_energy:
                k_vecs = []
                for idx in range(data.batch[-1]+1):
                    cell = data.cell
                    # Generate K point path
                    if isinstance(self.k_path, list):
                        kpts=kpoints_generator(dim_k=3, lat=cell[idx].detach().cpu().numpy())
                        k_vec, k_dist, k_node, lat_per_inv = kpts.k_path(self.k_path, self.num_k)
                    elif isinstance(self.k_path, str) and self.k_path.lower() == 'auto':
                        # build crystal structure
                        latt = cell[idx].detach().cpu().numpy()*au2ang
                        pos = torch.split(data.pos, data.node_counts.tolist(), dim=0)[idx].detach().cpu().numpy()*au2ang
                        species = torch.split(data.z, data.node_counts.tolist(), dim=0)[idx]
                        struct = Structure(lattice=latt, species=[Element.from_Z(k.item()).symbol for k in species], coords=pos, coords_are_cartesian=True)
                        # Initialize k_path and label
                        kpath_seek = KPathSeek(structure = struct)
                        klabels = []
                        for lbs in kpath_seek.kpath['path']:
                            klabels += lbs
                        # remove adjacent duplicates   
                        res = [klabels[0]]
                        [res.append(x) for x in klabels[1:] if x != res[-1]]
                        klabels = res
                        k_path = [kpath_seek.kpath['kpoints'][k] for k in klabels]
                        try:
                            kpts=kpoints_generator(dim_k=3, lat=cell[idx].detach().cpu().numpy())
                            k_vec, k_dist, k_node, lat_per_inv = kpts.k_path(k_path, self.num_k)
                        except:
                            lat_per_inv=np.linalg.inv(cell[idx].detach().cpu().numpy()).T
                            k_vec = 2.0*np.random.rand(self.num_k, 3)-1.0 #(-1, 1)
                    else:
                        lat_per_inv=np.linalg.inv(cell[idx].detach().cpu().numpy()).T
                        k_vec = 2.0*np.random.rand(self.num_k, 3)-1.0 #(-1, 1)
                    k_vec = k_vec.dot(lat_per_inv[np.newaxis,:,:]) # shape (nk,1,3)
                    k_vec = k_vec.reshape(-1,3) # shape (nk, 3)
                    k_vec = torch.Tensor(k_vec).type_as(Hon)
                    k_vecs.append(k_vec)  
                data.k_vecs = torch.stack(k_vecs, dim=0)
                if self.export_reciprocal_values:
                    if self.ham_only:
                        band_energy, wavefunction, HK, SK, dSK, gap = self.cal_band_energy(Hon, Hoff, data, True)
                        H_sym = None
                    else:
                        band_energy, wavefunction, HK, SK, dSK, gap = self.cal_band_energy_debug(Hon, Hoff, Son, Soff, data, True)
                        H_sym = None
                else:
                    band_energy, wavefunction, gap, H_sym = self.cal_band_energy(Hon, Hoff, data)
                with torch.no_grad():
                    data.band_energy, data.wavefunction, data.band_gap, data.H_sym = self.cal_band_energy(data.Hon, data.Hoff, data)
            else:
                band_energy = None
                wavefunction = None
                gap = None
                H_sym = None
        
        # Combining on-site and off-site Hamiltonians
        # openmx
        if self.ham_type in ['openmx','pasp', 'siesta', 'abacus']:
            if self.soc_switch:
                result = {'hamiltonian': Hsoc, 'hamiltonian_real':Hsoc_real, 
                          'hamiltonian_imag':Hsoc_imag, 'band_energy': band_energy, 
                          'wavefunction': wavefunction, 'iHon': Hsoc_on_imag, 'iHoff': Hsoc_off_imag}
            else:
                H = self.cat_onsite_and_offsite(data, Hon, Hoff)
                result = {'hamiltonian': H, 'band_energy': band_energy, 'wavefunction': wavefunction, 'band_gap':gap, 'H_sym': H_sym}
                if self.export_reciprocal_values:
                    result.update({'HK':HK, 'SK':SK, 'dSK': dSK})
        else:
            raise NotImplementedError
        
        if not self.ham_only:                
            # openmx
            if self.ham_type in ['openmx','pasp', 'siesta','abacus']:
                S = self.cat_onsite_and_offsite(data, Son, Soff)
            else:
                raise NotImplementedError
            result.update({'overlap': S})
        
        return result
