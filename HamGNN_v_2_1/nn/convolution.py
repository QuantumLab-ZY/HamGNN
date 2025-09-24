from typing import Callable, Dict, List, Optional, Tuple

import torch
from e3nn import o3
from e3nn.util.jit import compile_mode
from torch import nn
from torch_scatter import scatter

from .interaction_blocks import ResidualBlock
from .message_passing import MessagePackBlock
from ..toolbox.nequip.data import AtomicDataDict

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
            src=messages, index=receiver, dim=0, dim_size=receiver.max().item() + 1
        )
        
        # Apply residual block
        output_features = self.residual(aggregated_messages)

        # Apply skip connection if used
        if self.use_skip_connections:
            output_features += skip_connection

        data[AtomicDataDict.NODE_FEATURES_KEY] = output_features
        
        return output_features
