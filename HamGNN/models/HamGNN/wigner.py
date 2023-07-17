
from numpy import zeros
import torch
import torch.nn as nn
from torch_geometric.data import Data, batch
from torch.nn import (Bilinear, Sigmoid, Softplus, ELU, ReLU, SELU, SiLU,
                      CELU, BatchNorm1d, ModuleList, Sequential, Tanh)
from ..utils import linear_bn_act
from ..layers import denseRegression
from torch_scatter import scatter
import sympy as sym
from e3nn import o3
from e3nn.o3 import Linear
from e3nn.nn import Gate, NormActivation
from easydict import EasyDict
from typing import Union
from ..layers import GaussianSmearing, cuttoff_envelope, CosineCutoff, BesselBasis
from .nequip.data import AtomicDataDict, AtomicDataset
import math
from ..PhiSNet.modules.clebsch_gordan import ClebschGordan
import copy
from typing import Dict, Callable

def wigner(l, axis, angle):
    if l == 0:
        w = torch.Tensor([1.0]).type_as(angle)
    elif l == 1:
        w = o3.Irreps("1x1o").D_from_axis_angle(axis, angle).reshape(3, 3)
    elif l == 2:
        R = o3.Irreps("1x1o").D_from_axis_angle(axis, angle).reshape(3, 3)
        w = torch.Tensor([[R[0,0]*R[1,1]+R[0,1]*R[1,0], R[0,1]*R[1,2]+R[0,2]*R[1,1], R[0,2]*R[1,2], R[0,0]*R[1,2]+R[0,2]*R[1,0], R[0,0]*R[1,0]-R[0,1]*R[1,1]],
                 [R[1,0]*R[2,1]+R[1,1]*R[2,0], R[1,1]*R[2,2]+R[1,2]*R[2,1], R[1,2]*R[2,2], R[1,0]*R[2,2]+R[1,2]*R[2,0], R[1,0]*R[2,0]-R[1,1]*R[2,1]],
                 [2.0*R[2,0]*R[2,1]-R[0,0]*R[0,1]-R[1,0]*R[1,1], 2.0*R[2,1]*R[2,2]-R[0,1]*R[0,2]-R[1,1]*R[1,2], R[2,2]*R[2,2]-0.5*R[0,2]*R[0,2]-0.5*R[1,2]*R[1,2], 2.0*R[2,0]*R[2,2]-R[0,0]*R[0,2]-R[1,0]*R[1,2], R[2,0]*R[2,0]+0.5*R[0,1]*R[0,1]+0.5*R[1,1]*R[1,1]-0.5*R[0,0]*R[0,0]-0.5*R[1,0]*R[1,0]-R[2,1]*R[2,1]],
                 [R[0,0]*R[2,1]+R[0,1]*R[2,0], R[0,1]*R[2,2]+R[0,2]*R[2,1], R[0,2]*R[2,2], R[0,0]*R[2,2]+R[0,2]*R[2,0], R[0,0]*R[2,0]-R[0,1]*R[2,1]],
                 [R[0,0]*R[0,1]-R[1,0]*R[1,1], R[0,1]*R[0,2]-R[1,1]*R[1,2], 0.5*(R[0,2]*R[0,2]-R[1,2]*R[1,2]), R[0,0]*R[0,2]-R[1,0]*R[1,2], 0.5*(R[0,0]*R[0,0]+R[1,1]*R[1,1]-R[1,0]*R[1,0]-R[0,1]*R[0,1])]]).type_as(angle)
    else:
        raise ValueError
    return w
            
        