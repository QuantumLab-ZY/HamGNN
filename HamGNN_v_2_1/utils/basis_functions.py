import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.special import binom
import math
import sympy as sym
from torch_geometric.nn.models.dimenet_utils import real_sph_harm

from .cutoff_functions import cutoff_function
from .activation import softplus_inverse

"""
computes radial basis functions with Bernstein polynomials
"""
class BernsteinRadialBasisFunctions(nn.Module):
    def __init__(self, num_basis_functions, cutoff):
        super(BernsteinRadialBasisFunctions, self).__init__()
        self.num_basis_functions = num_basis_functions
        #compute values to initialize buffers
        logfactorial = np.zeros((num_basis_functions))
        for i in range(2,num_basis_functions):
            logfactorial[i] = logfactorial[i-1] + np.log(i)
        v = np.arange(0,num_basis_functions)
        n = (num_basis_functions-1)-v
        logbinomial = logfactorial[-1]-logfactorial[v]-logfactorial[n]
        #register buffers and parameters
        self.register_buffer('cutoff', torch.tensor(cutoff, dtype=torch.float64))
        self.register_buffer('logc', torch.tensor(logbinomial, dtype=torch.float64))
        self.register_buffer('n', torch.tensor(n, dtype=torch.float64))
        self.register_buffer('v', torch.tensor(v, dtype=torch.float64))
        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, r):
        x = torch.log(r/self.cutoff)
        x = self.logc + self.n*x + self.v*torch.log(-torch.expm1(x))
        rbf = cutoff_function(r, self.cutoff) * torch.exp(x)
        return rbf 

"""
computes radial basis functions with exponential Bernstein polynomials
"""
class ExponentialBernsteinRadialBasisFunctions(nn.Module):
    def __init__(self, num_basis_functions, cutoff, ini_alpha=0.5):
        super(ExponentialBernsteinRadialBasisFunctions, self).__init__()
        self.num_basis_functions = num_basis_functions
        self.ini_alpha = ini_alpha
        #compute values to initialize buffers
        logfactorial = np.zeros((num_basis_functions))
        for i in range(2,num_basis_functions):
            logfactorial[i] = logfactorial[i-1] + np.log(i)
        v = np.arange(0,num_basis_functions)
        n = (num_basis_functions-1)-v
        logbinomial = logfactorial[-1]-logfactorial[v]-logfactorial[n]
        #register buffers and parameters
        self.register_buffer('cutoff', torch.tensor(cutoff, dtype=torch.float64))
        self.register_buffer('logc', torch.tensor(logbinomial, dtype=torch.float64))
        self.register_buffer('n', torch.tensor(n, dtype=torch.float64))
        self.register_buffer('v', torch.tensor(v, dtype=torch.float64))
        self.register_parameter('_alpha', nn.Parameter(torch.tensor(1.0, dtype=torch.float64)))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self._alpha,  softplus_inverse(self.ini_alpha))

    def forward(self, r):
        alpha = F.softplus(self._alpha)
        x = -alpha*r
        x = self.logc + self.n*x + self.v*torch.log(-torch.expm1(x))
        rbf = cutoff_function(r, self.cutoff) * torch.exp(x)
        return rbf 


"""
computes radial basis functions with exponential Gaussians
"""
class ExponentialGaussianRadialBasisFunctions(nn.Module):
    def __init__(self, num_basis_functions, cutoff, ini_alpha=0.5):
        super(ExponentialGaussianRadialBasisFunctions, self).__init__()
        self.num_basis_functions = num_basis_functions
        self.ini_alpha = ini_alpha
        self.register_buffer('cutoff', torch.tensor(cutoff, dtype=torch.float64))
        self.register_buffer('center', torch.linspace(1, 0, self.num_basis_functions, dtype=torch.float64))
        self.register_buffer('width', torch.tensor(1.0*self.num_basis_functions, dtype=torch.float64))
        self.register_parameter('_alpha', nn.Parameter(torch.tensor(1.0, dtype=torch.float64)))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self._alpha,  softplus_inverse(self.ini_alpha))

    def forward(self, r):
        alpha = F.softplus(self._alpha)
        rbf = cutoff_function(r, self.cutoff) * torch.exp(-self.width*(torch.exp(-alpha*r)-self.center)**2)
        return rbf 


"""
computes radial basis functions with exponential Gaussians
"""
class GaussianRadialBasisFunctions(nn.Module):
    def __init__(self, num_basis_functions, cutoff):
        super(GaussianRadialBasisFunctions, self).__init__()
        self.num_basis_functions = num_basis_functions
        self.register_buffer('cutoff', torch.tensor(cutoff, dtype=torch.float64))
        self.register_buffer('center', torch.linspace(0, cutoff, self.num_basis_functions, dtype=torch.float64))
        self.register_buffer('width', torch.tensor(self.num_basis_functions/cutoff, dtype=torch.float64))
        #for compatibility with other basis functions on tensorboard, doesn't do anything
        self.register_parameter('_alpha', nn.Parameter(torch.tensor(1.0, dtype=torch.float64)))
        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, r):
        rbf = cutoff_function(r, self.cutoff) * torch.exp(-self.width*(r-self.center)**2)
        return rbf 


"""
computes radial basis functions with overlap Bernstein polynomials
"""
class OverlapBernsteinRadialBasisFunctions(nn.Module):
    def __init__(self, num_basis_functions, cutoff, ini_alpha=0.5):
        super(OverlapBernsteinRadialBasisFunctions, self).__init__()
        self.num_basis_functions = num_basis_functions
        self.ini_alpha = ini_alpha
        #compute values to initialize buffers
        logfactorial = np.zeros((num_basis_functions))
        for i in range(2,num_basis_functions):
            logfactorial[i] = logfactorial[i-1] + np.log(i)
        v = np.arange(0,num_basis_functions)
        n = (num_basis_functions-1)-v
        logbinomial = logfactorial[-1]-logfactorial[v]-logfactorial[n]
        #register buffers and parameters
        self.register_buffer('cutoff', torch.tensor(cutoff, dtype=torch.float64))
        self.register_buffer('logc', torch.tensor(logbinomial, dtype=torch.float64))
        self.register_buffer('n', torch.tensor(n, dtype=torch.float64))
        self.register_buffer('v', torch.tensor(v, dtype=torch.float64))
        self.register_parameter('_alpha', nn.Parameter(torch.tensor(1.0, dtype=torch.float64)))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self._alpha,  softplus_inverse(self.ini_alpha))

    def forward(self, r):
        alpha_r = F.softplus(self._alpha)*r
        x = torch.log1p(alpha_r)-alpha_r
        x = self.logc + self.n*x + self.v*torch.log(-torch.expm1(x))
        rbf = cutoff_function(r, self.cutoff) * torch.exp(x)
        return rbf


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
    def __init__(self, cutoff=5.0, n_rbf:int=None, cutoff_func=None):
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