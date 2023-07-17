"""
/*
 * @Author: Yang Zhong 
 * @Date: 2021-10-09 10:18:50 
 * @Last Modified by: Yang Zhong
 * @Last Modified time: 2021-10-28 20:19:52
 */
"""
import torch
import torch.nn as nn
from .utils import linear_bn_act
from torch_geometric.data import Data, batch
from torch.nn import (Linear, Bilinear, Sigmoid, Softplus, ELU, ReLU, SELU, SiLU,
                      CELU, BatchNorm1d, ModuleList, Sequential, Tanh)
from typing import Callable
import sympy as sym
import math
from torch_geometric.nn.models.dimenet_utils import real_sph_harm
from torch_geometric.nn.acts import swish
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from math import pi
from functools import partial

zeros_initializer = partial(constant_, val=0.0)


class denseLayer(nn.Module):
    def __init__(self, in_features: int=None, out_features: int=None, bias:bool=True, 
                    use_batch_norm:bool=True, activation:callable=ELU()):
        super().__init__()
        self.lba = linear_bn_act(in_features=in_features, out_features=out_features, lbias=bias, 
                        activation=activation, use_batch_norm=use_batch_norm)
        self.linear = Linear(out_features, out_features, bias=bias)
    def forward(self, x):
        out = self.linear(self.lba(x))
        return out

class denseRegression(nn.Module):
    def __init__(self, in_features: int=None, out_features: int=None, bias:bool=True, 
                    use_batch_norm:bool=True, activation:callable=Softplus(), n_h:int=3):
        super().__init__()
        if n_h > 1:
            self.fcs = nn.ModuleList([linear_bn_act(in_features=in_features, out_features=in_features, lbias=bias, 
                        activation=activation, use_batch_norm=use_batch_norm) for _ in range(n_h-1)])
        self.fc_out = nn.Linear(in_features, out_features)

    def forward(self, x):
        for fc in self.fcs:
            x = fc(x)
        out = self.fc_out(x)
        return out

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

class MLPRegression(nn.Module):
    def __init__(self, num_in_features: int, num_out_features: int, num_mlp: int=3, lbias: bool = False,
                 activation: Callable = ELU(), use_batch_norm: bool = False):
        super(MLPRegression, self).__init__()
        self.linear_regression = [linear_bn_act(int(num_in_features/2**(i-1)), int(num_in_features/2**i), 
                                   lbias, activation, use_batch_norm) for i in range(1, num_mlp)]
        self.linear_regression += [linear_bn_act(int(num_in_features/2**(num_mlp-1)), num_out_features, 
                                    lbias, activation, use_batch_norm)]                           
        self.linear_regression = ModuleList(self.linear_regression)

    def forward(self, x):
        for lr in self.linear_regression:
            x = lr(x)
        return x

class sph_harm_layer(nn.Module):
    def __init__(self, num_spherical):
        super(sph_harm_layer, self).__init__()
        self.num_spherical = num_spherical
        sph_harm_forms = real_sph_harm(num_spherical)
        self.sph_funcs = []

        theta = sym.symbols('theta')
        modules = {'sin': torch.sin, 'cos': torch.cos}
        for i in range(num_spherical):
            if i == 0:
                sph1 = sym.lambdify([theta], sph_harm_forms[i][0], modules)(0)
                self.sph_funcs.append(lambda x: torch.zeros_like(x) + sph1)
            else:
                sph = sym.lambdify([theta], sph_harm_forms[i][0], modules)
                self.sph_funcs.append(sph)

    def forward(self, angle):
        out = torch.cat([f(angle.unsqueeze(-1)) for f in self.sph_funcs], dim=-1)
        return out

class BesselBasis(nn.Module):
    """
    Sine for radial basis expansion with coulomb decay. (0th order Bessel from DimeNet)
    """
    def __init__(self, cutoff=5.0, n_rbf:int=None, cutoff_func:callable=None):
        """
        Args:
            cutoff: radial cutoff
            n_rbf: number of basis functions.
        """
        super(BesselBasis, self).__init__()
        # compute offset and width of Gaussian functions
        freqs = torch.arange(1, n_rbf + 1) * math.pi / cutoff
        self.register_buffer("freqs", freqs)
        self.cutoff_func = cutoff_func

    def forward(self, dist):
        r"""Computes the 0th order Bessel expansion of inter-atomic distances.

            Args:
                dist (torch.Tensor):
                    inter-atomic distances with (N_edge,) shape

            Returns:
                rbf (torch.Tensor):
                    the 0th order Bessel expansion of inter-atomic distances
                    with (N_edge, n_rbf) shape.
            """
        a = self.freqs[None, :]
        ax = dist.unsqueeze(-1) * a
        rbf = torch.sin(ax) / dist.unsqueeze(-1)
        if self.cutoff_func is not None:
            rbf = rbf * self.cutoff_func(dist.unsqueeze(-1))
        return rbf

class GaussianSmearing(nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50, cutoff_func=None):
        super(GaussianSmearing, self).__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)
        self.cutoff_func = cutoff_func
    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        expansion = torch.exp(self.coeff * torch.pow(dist, 2))
        if self.cutoff_func is not None:
            expansion = expansion*self.cutoff_func(dist)
        return expansion

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
