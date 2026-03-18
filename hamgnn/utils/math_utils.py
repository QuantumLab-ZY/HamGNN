import torch

def count_neighbors_per_node(source_nodes):
    """
    Calculate the number of neighbors for each node.

    Args:
        source_nodes (torch.Tensor): A tensor containing source node indices.

    Returns:
        torch.Tensor: A tensor where each index represents a node and the value
                      at that index is the count of its neighbors.
    """
    # Identify unique nodes and count their occurrences
    unique_nodes, counts = torch.unique(source_nodes, return_counts=True)

    # Determine the total number of nodes
    total_nodes = source_nodes.max().item() + 1

    # Initialize a tensor to store the neighbor counts for each node
    neighbor_counts = torch.zeros((total_nodes,)).type_as(source_nodes)

    # Assign the counts to their respective nodes
    neighbor_counts[unique_nodes] = counts

    # Ensure the output tensor has the same type as the input
    return neighbor_counts


def prod(x):
    """Compute the product of a sequence."""
    out = 1
    for a in x:
        out *= a
    return out


def blockwise_2x2_concat(
    top_left: torch.Tensor,
    top_right: torch.Tensor,
    bottom_left: torch.Tensor,
    bottom_right: torch.Tensor
) -> torch.Tensor:
    """
    Concatenates four tensors in a 2x2 block pattern to form a doubled-size tensor.
    The concatenation pattern follows:
    [top_left | top_right]
    ----------------------
    [bottom_left | bottom_right]
    Parameters:
        top_left (torch.Tensor):     Tensor of shape [N, H, W]
        top_right (torch.Tensor):    Tensor of same shape as top_left
        bottom_left (torch.Tensor):  Tensor of same shape as top_left
        bottom_right (torch.Tensor): Tensor of same shape as top_left
    Returns:
        torch.Tensor: Concatenated tensor of shape [N, 2H, 2W]
    Raises:
        ValueError: If input tensors have mismatching shapes
    Example:
        >>> a = torch.ones(2, 3, 3)
        >>> b = torch.zeros(2, 3, 3)
        >>> result = blockwise_2x2_concat(a, b, b, a)
        >>> result.shape
        torch.Size([2, 6, 6])
    """
    # Validate input tensor dimensions
    expected_shape = top_left.shape
    for i, tensor in enumerate([top_right, bottom_left, bottom_right], start=2):
        if tensor.shape != expected_shape:
            raise ValueError(
                f"Tensor {i} shape {tensor.shape} doesn't match "
                f"first tensor's shape {expected_shape}"
            )
    # Horizontal concatenation first (dimension W)
    top_row = torch.cat([top_left, top_right], dim=-1)
    bottom_row = torch.cat([bottom_left, bottom_right], dim=-1)
    # Vertical concatenation (dimension H)
    return torch.cat([top_row, bottom_row], dim=-2)


def extract_elements_above_threshold(
    condition_tensor: torch.Tensor,
    source_tensor: torch.Tensor,
    threshold: float = 0.0
) -> torch.Tensor:
    """Extracts elements from source tensor where condition tensor exceeds threshold.
    Args:
        condition_tensor: Tensor[Nbatch, N, N] used for threshold comparison
        source_tensor: Tensor[Nbatch, N, N] from which values are extracted
        threshold: Minimum value for elements in condition_tensor to trigger extraction
    Returns:
        torch.Tensor: 1D tensor of extracted values from source_tensor
    Raises:
        ValueError: If input tensors have mismatching shapes
    Example:
        >>> S = torch.randn(2, 3, 3)
        >>> H = torch.randn(2, 3, 3)
        >>> result = extract_elements_above_threshold(S, H, 0.5)
    """
    # Validate input shapes
    if condition_tensor.shape != source_tensor.shape:
        raise ValueError(f"Shape mismatch: {condition_tensor.shape} vs {source_tensor.shape}")
    # Create boolean mask for threshold condition
    threshold_mask = condition_tensor > threshold
    # Extract corresponding elements from source tensor
    extracted_values = source_tensor[threshold_mask]
    return extracted_values


def upgrade_tensor_precision(tensor_dict):
    """
    Upgrades the precision of specific tensor types in the provided dictionary.
    This function iterates through the given dictionary and converts:
    - torch.float32 (float) tensors to torch.float64 (double)
    - torch.complex64 tensors to torch.complex128
    All other tensor types remain unchanged. The original device of each tensor
    is preserved during conversion.
    Args:
        tensor_dict (dict): Dictionary containing torch tensors with string keys
                           and torch tensor values.
    Returns:
        None: The function modifies the dictionary in-place.
    Notes:
        For float32 tensors, either `.to(dtype=torch.float64)` or `.double()`
        can be used for conversion. This function uses the `.to()` method for
        consistency with complex tensor conversion.
    Example:
        >>> data = {'float_tensor': torch.tensor([1.0, 2.0], dtype=torch.float32)}
        >>> upgrade_tensor_precision(data)
        >>> print(data['float_tensor'].dtype)
        torch.float64
    """
    for key, value in tensor_dict.items():
        if isinstance(value, torch.Tensor):
            if value.dtype == torch.float32:
                tensor_dict[key] = value.to(dtype=torch.float64)
            elif value.dtype == torch.complex64:
                tensor_dict[key] = value.to(dtype=torch.complex128)