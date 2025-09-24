import math
import re


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CELU, ELU, ReLU, SELU, SiLU, Softplus, Tanh
from e3nn.math import soft_unit_step
from e3nn.util.jit import compile_mode


"""
IMPORTANT NOTE: The cutoff and the switch function are numerically a bit tricky:
Right at the "seems" of these functions, i.e. where the piecewise definition changes,
there is formally a division by 0 (i.e. 0/0). This is of no issue for the function
itself, but when automatic differentiation is used, this division will lead to NaN 
gradients. In order to circumvent this, the input needs to be masked as well.
"""

"""
shifted softplus activation function
"""
_log2 = math.log(2)
def shifted_softplus(x):
    return F.softplus(x) - _log2

"""
switch function that smoothly and symmetrically goes from y = 1..0 in the interval x = cuton..cutoff and
is 1 for x <= cuton and 0 for x >= cutoff (this switch function has infinitely many smooth derivatives)
(when cuton < cutoff, it goes from 1 to 0, if cutoff < cuton, it goes from 0 to 1)
NOTE: the implementation with the "_switch_component" function is numerically more stable than
a simplified version, DO NOT CHANGE THIS!
"""
def _switch_component(x, ones, zeros):
    x_ = torch.where(x <= 0, ones, x)
    return torch.where(x <= 0, zeros, torch.exp(-ones/x_))

def switch_function(x, cuton, cutoff):
    x = (x-cuton)/(cutoff-cuton)
    ones  = torch.ones_like(x)
    zeros = torch.zeros_like(x)
    fp = _switch_component(x, ones, zeros)
    fm = _switch_component(1-x, ones, zeros)
    return torch.where(x <= 0, ones, torch.where(x >= 1, zeros, fm/(fp+fm)))

"""
inverse softplus transformation, this is useful for initialization of parameters that are constrained to be positive
"""
def softplus_inverse(x):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    return x + torch.log(-torch.expm1(-x))


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

def swish(x):
    return x * x.sigmoid()

class SSP(nn.Module):
    r"""Applies element-wise :math:`\text{SSP}(x)=\text{Softplus}(x)-\text{Softplus}(0)`

    Shifted SoftPlus (SSP)

    Args:
        beta: the :math:`\beta` value for the Softplus formulation. Default: 1
        threshold: values above this revert to a linear function. Default: 20

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input
    """

    def __init__(self, beta=1, threshold=20):
        super(SSP, self).__init__()
        self.beta = beta
        self.threshold = threshold

    def forward(self, input):
        sp0 = F.softplus(torch.Tensor([0]), self.beta, self.threshold).item()
        return F.softplus(input, self.beta, self.threshold) - sp0

    def extra_repr(self):
        return 'beta={}, threshold={}'.format(self.beta, self.threshold)

class SWISH(nn.Module):
    def __init__(self):
        super(SWISH, self).__init__()

    def forward(self, input):
        return swish(input)

def get_activation(name):
    act_name = name.lower()
    m = re.match(r"(\w+)\((\d+\.\d+)\)", act_name)
    if m is not None:
        act_name, alpha = m.groups()
        alpha = float(alpha)
        print(act_name, alpha)
    else:
        alpha = 1.0
    if act_name == 'softplus':
        return Softplus()
    elif act_name == 'ssp':
        return SSP()
    elif act_name == 'elu':
        return ELU(alpha)
    elif act_name == 'relu':
        return ReLU()
    elif act_name == 'selu':
        return SELU()
    elif act_name == 'swish':
        return SWISH()
    elif act_name == 'tanh':
        return Tanh()
    elif act_name == 'silu':
        return SiLU()
    elif act_name == 'celu':
        return CELU(alpha)
    else:
        raise NameError("Not supported activation: {}".format(name))
