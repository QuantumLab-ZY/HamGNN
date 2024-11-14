'''
Descripttion: 
version: 
Author: Yang Zhong
Date: 2024-06-20 21:30:56
LastEditors: Yang Zhong
LastEditTime: 2024-06-20 21:32:28
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.special import binom
from .functional import cutoff_function, softplus_inverse

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
