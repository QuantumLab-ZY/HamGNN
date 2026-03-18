import torch
from e3nn import o3
from e3nn.util.jit import compile_mode
from torch import nn


@compile_mode('script')
class VectorToAttentionHeads(nn.Module):
    """
    Reshapes vectors of shape [N, irreps_mid] to vectors of shape [N, num_heads, irreps_head].

    Attributes:
    - num_heads (int): Number of attention heads.
    - irreps_head (o3.Irreps): Irreps of each head.
    - irreps_mid_in (o3.Irreps): Intermediate irreps.
    - mid_in_indices (List[Tuple[int, int]]): Indices for reshaping.
    """

    def __init__(self, irreps_head: o3.Irreps, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.irreps_head = irreps_head
        self.irreps_mid_in = o3.Irreps([(mul * num_heads, ir) for mul, ir in irreps_head])
        self.mid_in_indices = []
        start_idx = 0
        for mul, ir in self.irreps_mid_in:
            self.mid_in_indices.append((start_idx, start_idx + mul * ir.dim))
            start_idx = start_idx + mul * ir.dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, _ = x.shape
        reshaped_tensors = [
            x.narrow(1, start_idx, end_idx - start_idx).view(N, self.num_heads, -1)
            for start_idx, end_idx in self.mid_in_indices
        ]
        return torch.cat(reshaped_tensors, dim=2)

    def __repr__(self):
        return f'{self.__class__.__name__}(irreps_head={self.irreps_head}, num_heads={self.num_heads})'

@compile_mode('script')
class AttentionHeadsToVector(nn.Module):
    """
    Converts vectors of shape [N, num_heads, irreps_head] into vectors of shape [N, irreps_head * num_heads].
    
    Attributes:
        irreps_head (o3.Irreps): A list of irreducible representations (irreps) that define
                                 the structure of the attention heads.
        head_sizes (List[int]): A list of sizes for each attention head, derived from the irreps.
    """

    def __init__(self, irreps_head: o3.Irreps):
        """
        Initialize the AttentionHeadsToVector module.

        Args:
            irreps_head (o3.Irreps): A list of irreducible representations (irreps) used to define
                                     the structure of attention heads. Each irrep specifies the
                                     multiplicity and dimension of a representation.
        """
        super().__init__()
        self.irreps_head = irreps_head

        # Compute the size of each attention head based on the irreps definitions.
        self.head_sizes = [multiplicity * irrep.dim for multiplicity, irrep in self.irreps_head]

    def __repr__(self):
        """
        Provide a string representation of the module for debugging.

        Returns:
            str: A string representation of the AttentionHeadProcessor instance.
        """
        return f'{self.__class__.__name__}(irreps_head={self.irreps_head})'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to process the attention heads and flatten them into a single vector.

        Args:
            x (torch.Tensor): Input tensor of shape (N, num_heads, input_dim), where:
                - N is the batch size.
                - num_heads is the number of attention heads.
                - input_dim is the total size of all heads.

        Returns:
            torch.Tensor: Output tensor of shape (N, flattened_dim), where `flattened_dim`
                          is the sum of the dimensions of all attention heads.

        Raises:
            ValueError: If the sum of `head_sizes` does not match `input_dim` of the input tensor.
        """
        # Extract the dimensions of the input tensor.
        batch_size, num_heads, input_dim = x.shape

        # Ensure the total size of all attention heads matches the input tensor's last dimension.
        if sum(self.head_sizes) != input_dim:
            raise ValueError(
                f"The sum of head_sizes ({sum(self.head_sizes)}) does not match the input_dim ({input_dim}) "
                "of the input tensor."
            )

        # Split the input tensor along the last dimension based on head_sizes.
        split_tensors = torch.split(x, self.head_sizes, dim=2)

        # Reshape each split tensor to flatten the attention heads into a single vector per batch.
        # Use `contiguous()` to ensure the tensor's memory layout is consistent.
        flattened_tensors = [sub_tensor.contiguous().view(batch_size, -1) for sub_tensor in split_tensors]

        # Concatenate the flattened tensors along the last dimension to produce the output.
        return torch.cat(flattened_tensors, dim=1)
