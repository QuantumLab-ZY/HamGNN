import torch
import torch.nn as nn
from torch.nn import Linear
from torch.nn.init import xavier_uniform_, constant_
from functools import partial
from torch.nn import Linear, BatchNorm1d, ELU

zeros_initializer = partial(constant_, val=0.0)


def linear_bn_act(in_features: int, out_features: int, lbias: bool = True,
                 activation = ELU(), use_batch_norm: bool = True):
    """
    Create a sequential module that includes a linear layer, optional batch normalization, and activation functions
    
    Args:
        in_features (int): Number of input features
        out_features (int): Number of output features
        lbias (bool): Whether a bias is included in the linear layer
        activation (callable): The activation function to be used
        use_batch_norm (bool): Whether it includes batch normalization or not
    
    Returns:
        torch.nn.Sequential: A sequential module containing a linear layer, an optional batch normalization, and an activation function
    """
    layers = []
    layers.append(Linear(in_features, out_features, bias=lbias))
    
    if use_batch_norm:
        layers.append(BatchNorm1d(out_features))
    
    if activation is not None:
        layers.append(activation)
    
    return nn.Sequential(*layers)

class denseLayer(nn.Module):
    def __init__(self, in_features: int=None, out_features: int=None, bias:bool=True,
                 use_batch_norm:bool=True, activation=nn.ELU()):
        super().__init__()
        self.lba = linear_bn_act(in_features=in_features, out_features=out_features, lbias=bias,
                        activation=activation, use_batch_norm=use_batch_norm)
        self.linear = Linear(out_features, out_features, bias=bias)
    
    def forward(self, x):
        out = self.linear(self.lba(x))
        return out


class Dense(nn.Linear):
    r"""From schnetpack
    Fully connected linear layer with activation function.
    .. math::
       y = activation(xW^T + b)
    Args:
        in_features (int): number of input feature :math:`x`.
        out_features (int): number of output features :math:`y`.
        bias (bool, optional): if False, the layer will not adapt bias :math:`b`.
        activation (callable, optional): if None, no activation function is used.
        weight_init (callable, optional): weight initializer from current weight.
        bias_init (callable, optional): bias initializer from current bias.
    """
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        activation=None,
        weight_init=xavier_uniform_,
        bias_init=zeros_initializer,
    ):
        self.weight_init = weight_init
        self.bias_init = bias_init
        super(Dense, self).__init__(in_features, out_features, bias)
        self.activation = activation
        # initialize linear layer y = xW^T + b
    
    def reset_parameters(self):
        """Reinitialize model weight and bias values."""
        self.weight_init(self.weight)
        if self.bias is not None:
            self.bias_init(self.bias)
    
    def forward(self, inputs):
        """Compute layer output.
        Args:
            inputs (dict of torch.Tensor): batch of input values.
        Returns:
            torch.Tensor: layer output.
        """
        # compute linear layer y = xW^T + b
        y = super(Dense, self).forward(inputs)
        # add activation function
        if self.activation:
            y = self.activation(y)
        return y