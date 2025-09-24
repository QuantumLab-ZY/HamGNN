import torch
import torch.nn as nn
from math import pi
from e3nn.math import soft_unit_step
from e3nn.util.jit import compile_mode

"""
cutoff function that smoothly goes from y = 1..0 in the interval x = 0..cutoff
(this cutoff function has infinitely many smooth derivatives)
"""
def cutoff_function(x, cutoff):
    zeros = torch.zeros_like(x) 
    x_ = torch.where(x < cutoff, x, zeros)
    return torch.where(x < cutoff, torch.exp(-x_**2/((cutoff-x_)*(cutoff+x_))), zeros)


class cuttoff_envelope(nn.Module):
    def __init__(self, cutoff, exponent=6):
        super(cuttoff_envelope, self).__init__()
        self.p = exponent
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2
        self.cutoff = cutoff
    
    def forward(self, x):
        p, a, b, c = self.p, self.a, self.b, self.c
        x = x/self.cutoff
        x_pow_p0 = x.pow(p)
        x_pow_p1 = x_pow_p0 * x
        x_pow_p2 = x_pow_p1 * x
        return (1. + a * x_pow_p0 + b * x_pow_p1 + c * x_pow_p2) * (x < self.cutoff).float()


class CosineCutoff(nn.Module):
    r"""Class of Behler cosine cutoff. From schnetpack
    .. math::
       f(r) = \begin{cases}
        0.5 \times \left[1 + \cos\left(\frac{\pi r}{r_\text{cutoff}}\right)\right]
          & r < r_\text{cutoff} \\
        0 & r \geqslant r_\text{cutoff} \\
        \end{cases}
    Args:
        cutoff (float, optional): cutoff radius.
    """
    def __init__(self, cutoff=5.0):
        super(CosineCutoff, self).__init__()
        self.register_buffer("cutoff", torch.FloatTensor([cutoff]))
    
    def forward(self, distances):
        """Compute cutoff.
        Args:
            distances (torch.Tensor): values of interatomic distances.
        Returns:
            torch.Tensor: values of cutoff function.
        """
        # Compute values of cutoff function
        cutoffs = 0.5 * (torch.cos(distances * pi / self.cutoff) + 1.0)
        # Remove contributions beyond the cutoff radius
        cutoffs *= (distances < self.cutoff).float()
        return cutoffs


@compile_mode("script")
class SoftUnitStepCutoff(nn.Module):
    """
    A PyTorch module that applies a soft unit step function with a cutoff.
    
    Attributes:
        cutoff (float): The distance at which the cutoff is applied.
        cut_param (nn.Parameter): A learnable parameter influencing the softness of the step.
    """
    def __init__(self, cutoff):
        """
        Initializes the SoftUnitStepCutoff module.
        
        Args:
            cutoff (float): The cutoff distance for the step function.
        """
        super(SoftUnitStepCutoff, self).__init__()
        self.cutoff = cutoff
        self.cut_param = nn.Parameter(torch.tensor(10.0, dtype=torch.get_default_dtype()))

    def forward(self, edge_distance):
        """
        Forward pass for the module.
        
        Applies the soft unit step function to the input edge distances.
        
        Args:
            edge_distance (Tensor): A tensor containing edge distances.
        
        Returns:
            Tensor: A tensor with the calculated edge weights after applying the cutoff.
        """
        # Calculate the scaled difference and apply the soft unit step
        scaled_diff = self.cut_param * (1.0 - edge_distance / self.cutoff)
        edge_weight_cutoff = soft_unit_step(scaled_diff)
        
        return edge_weight_cutoff