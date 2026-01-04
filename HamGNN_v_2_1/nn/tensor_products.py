from typing import Callable, Dict, List, Optional, Tuple

import torch
from e3nn import o3
from e3nn.nn import FullyConnectedNet
from e3nn.util.jit import compile_mode
from torch import nn

from .attention_utils import AttentionHeadsToVector
from ..toolbox.efficient_kan import KAN
from ..utils.irreps_utils import scale_irreps
from ..utils.macro import GRID_RANGE, GRID_SIZE


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
class TensorProductWithMemoryOptimizationWithWeight(nn.Module):
    def __init__(self,
                 irreps_input_1,
                 irreps_input_2,
                 irreps_out,
                 irreps_scalar,
                 radial_MLP,
                 use_kan,
                 lite_mode):
        """
        Initialize the TensorProductWithMemoryOptimization module.

        Args:
            irreps_input_1 (str): Irreducible representations for the first input.
            irreps_input_2 (str): Irreducible representations for the second input.
            irreps_out (str): Irreducible representations for the output.
            irreps_scalar (str): Irreducible representations for scalar inputs.
            radial_MLP (list[int]): List of hidden layer sizes for the radial MLP.
            use_kan (bool): Flag to use KAN instead of FullyConnectedNet.
            lite_mode (bool): The mode with the fewest model parameters and the fastest running speed.
        """
        super().__init__()

        # Initialize irreducible representations
        self.irreps_input_1 = o3.Irreps(irreps_input_1)
        self.irreps_input_2 = o3.Irreps(irreps_input_2)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_scalar = o3.Irreps(irreps_scalar)
        self.radial_MLP = radial_MLP
        self.use_kan = use_kan
        self.lite_mode = lite_mode
        if self.lite_mode:
            self.tp_mode = 'uvu'
        else:
            self.tp_mode = 'uvw'

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
            internal_weights=False if self.lite_mode else True,
            shared_weights=False if self.lite_mode else True
        )

        # Initialize linear scaling with weights
        self.linear_scaler = LinearScaleWithWeights(
            irreps_in=self.irreps_mid.simplify(),
            irreps_out=self.irreps_out
        )

        # Initialize the weight generator
        input_dim = self.irreps_scalar.num_irreps
        self.weight_generator = self._initialize_weight_generator(
            input_dim, self.linear_scaler.weight_numel)

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

