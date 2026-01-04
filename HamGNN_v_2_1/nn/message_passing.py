from typing import Callable, Dict, List, Optional, Tuple

import torch
from e3nn import o3
from e3nn.util.jit import compile_mode
from e3nn.nn import FullyConnectedNet
from torch import nn

from .attention_utils import AttentionHeadsToVector
from ..toolbox.efficient_kan import KAN
from ..utils.irreps_utils import scale_irreps
from ..utils.macro import GRID_RANGE, GRID_SIZE
from .tensor_products import LinearScaleWithWeights


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
        use_kan: bool = False,
        lite_mode: bool = False
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
            lite_mode (bool): The mode with the fewest model parameters and the fastest running speed.
        """
        super().__init__()
        self.irreps_node_feats = o3.Irreps(irreps_node_feats)
        self.irreps_edge_feats = o3.Irreps(irreps_edge_feats)
        self.irreps_local_env_edge = o3.Irreps(irreps_local_env_edge)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_edge_scalars = o3.Irreps(irreps_edge_scalars)
        self.radial_MLP = radial_MLP
        self.use_kan = use_kan
        self.lite_mode = lite_mode
        if self.lite_mode:
            self.tp_mode = 'uvu'
        else:
            self.tp_mode = 'uvw'

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
            internal_weights=False if self.lite_mode else True,
            shared_weights=False if self.lite_mode else True
        )
        self.edge_tensor_product = o3.TensorProduct(
            self.irreps_edge_feats,
            self.irreps_local_env_edge,
            self.mid_edge_irreps,
            instructions=self.edge_instructions,
            internal_weights=False if self.lite_mode else True,
            shared_weights=False if self.lite_mode else True
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
        if self.lite_mode:
            trainable = False
        else:
            trainable = True

        # Collect possible irreps and their instructions
        irreps_out_list: List[Tuple[int, o3.Irreps]] = []
        instructions = []
        for i, (mul_in, ir_in) in enumerate(irreps1):
            for j, (_, ir_edge) in enumerate(irreps2):  
                for _, (mul_out, ir_out) in enumerate(target_irreps):                  
                    if ir_out in ir_in * ir_edge:
                        k = len(irreps_out_list)
                        if self.tp_mode == 'uvw':
                            irreps_out_list.append((mul_out, ir_out))
                        else:
                            irreps_out_list.append((mul_in, ir_out))
                        instructions.append((i, j, k, self.tp_mode, trainable))

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

