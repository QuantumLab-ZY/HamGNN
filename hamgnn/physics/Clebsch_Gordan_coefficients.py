
import torch.nn as nn
import e3nn.o3 as o3
from e3nn.util.jit import compile_mode


@compile_mode("script")
class ClebschGordanCoefficients(nn.Module):
    """
    A PyTorch module for pre-computing and storing Clebsch-Gordan coefficients,
    which can then be accessed during the forward pass.
    """

    def __init__(self, max_l=8):
        """
        Initialize the module and pre-compute Clebsch-Gordan coefficients up to a maximum angular momentum value.

        :param max_l: Maximum angular momentum value for which to compute coefficients.
        """
        super().__init__()

        # Pre-compute and store all necessary Clebsch-Gordan coefficients
        for l1 in range(max_l + 1):
            for l2 in range(max_l + 1):
                for l3 in range(abs(l1 - l2), l1 + l2 + 1):
                    buffer_name = f'cg_{l1}_{l2}_{l3}'
                    self.register_buffer(buffer_name, o3.wigner_3j(l1, l2, l3))

    def forward(self, l1, l2, l3):
        """
        Retrieve the pre-computed Clebsch-Gordan coefficient for the given angular momenta.

        :param l1: First angular momentum value.
        :param l2: Second angular momentum value.
        :param l3: Third angular momentum value.
        :return: The Clebsch-Gordan coefficient tensor.
        """
        buffer_name = f'cg_{l1}_{l2}_{l3}'
        return getattr(self, buffer_name)
