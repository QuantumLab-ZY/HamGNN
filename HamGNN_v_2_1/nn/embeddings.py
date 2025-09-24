from typing import Callable, Dict, List, Optional, Tuple

import torch
import numpy as np
from e3nn import o3
from e3nn.nn import FullyConnectedNet
from e3nn.util.jit import compile_mode
from torch import nn

from ..toolbox.efficient_kan import KAN
from ..toolbox.nequip.data import AtomicDataDict
from ..toolbox.nequip.nn import GraphModuleMixin
from ..utils.macro import GRID_RANGE, GRID_SIZE
from .electron_configurations import electron_configurations
from .tensor_products import TensorProductWithMemoryOptimizationWithWeight

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


"""
Embedding layer which takes scalar nuclear charges Z and transforms 
them to vectors of size num_features
"""
class Embedding(nn.Module):
    def __init__(self, num_features, Zmax=87):
        super(Embedding, self).__init__()
        self.num_features = num_features
        self.Zmax = Zmax
        self.register_buffer('electron_config', torch.tensor(electron_configurations))
        self.register_parameter('element_embedding', nn.Parameter(torch.Tensor(self.Zmax, self.num_features))) 
        self.config_linear = nn.Linear(self.electron_config.size(1), self.num_features, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.element_embedding, -np.sqrt(3), np.sqrt(3))
        nn.init.orthogonal_(self.config_linear.weight)

    def forward(self, Z):
        embedding = self.element_embedding + self.config_linear(self.electron_config)
        return embedding[Z]