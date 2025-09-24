from typing import Callable, Dict, List, Optional, Tuple

import torch
from e3nn import o3
from e3nn.nn import Gate, NormActivation
from e3nn.util.jit import compile_mode
from torch import nn

from .message_passing import MessagePackBlock
from ..toolbox.nequip.data import AtomicDataDict
from ..toolbox.mace.modules.blocks import EquivariantProductBasisBlock
from ..toolbox.mace.modules.irreps_tools import (
    linear_out_irreps,
    reshape_irreps,
    tp_out_irreps_with_instructions,
)
from ..utils.irreps_utils import acts, irreps2gate

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


