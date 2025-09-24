import math
from typing import Callable, Dict, List, Optional, Tuple

import torch
from e3nn import o3
from e3nn.nn import FullyConnectedNet, Gate, NormActivation
from e3nn.util.jit import compile_mode
from torch import nn
from torch_geometric.utils import softmax as edge_softmax
from torch_scatter import scatter

from .attention_utils import AttentionHeadsToVector, VectorToAttentionHeads
from .interaction_blocks import ResidualBlock
from .message_passing import MessagePackBlock
from ..toolbox.efficient_kan import KAN
from ..toolbox.nequip.data import AtomicDataDict
from ..toolbox.nequip.nn import GraphModuleMixin
from ..utils.cutoff_functions import SoftUnitStepCutoff
from ..utils.irreps_utils import acts, irreps2gate, scale_irreps
from ..utils.macro import GRID_RANGE, GRID_SIZE

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
