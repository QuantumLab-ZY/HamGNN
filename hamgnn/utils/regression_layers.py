import torch
import torch.nn as nn
from torch.nn import ModuleList, ELU, Softplus
from typing import Callable
from .mlp import linear_bn_act

class denseRegression(nn.Module):
    def __init__(self, in_features: int=None, out_features: int=None, bias:bool=True,
                 use_batch_norm:bool=True, activation=Softplus(), n_h:int=3):
        super().__init__()
        self.fcs = nn.ModuleList()
        if n_h > 1:
            self.fcs = nn.ModuleList([linear_bn_act(in_features=in_features, out_features=in_features, lbias=bias,
                        activation=activation, use_batch_norm=use_batch_norm) for _ in range(n_h-1)])
        self.fc_out = nn.Linear(in_features, out_features)
    
    def forward(self, x):
        for fc in self.fcs:
            x = fc(x)
        out = self.fc_out(x)
        return out


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