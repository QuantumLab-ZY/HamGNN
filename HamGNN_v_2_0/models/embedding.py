'''
Descripttion: 
version: 
Author: Yang Zhong
Date: 2024-06-20 21:33:29
LastEditors: Yang Zhong
LastEditTime: 2024-06-20 21:35:59
'''
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .electron_configurations import *

"""
Embedding layer which takes scalar nuclear charges Z and transforms 
them to vectors of size num_features
"""
class Embedding(nn.Module):
    def __init__(self, num_features, Zmax=87):
        super(Embedding, self).__init__()
        self.num_features = num_features
        self.Zmax = Zmax
        self.register_buffer('electron_config', torch.tensor(electron_configurations))
        self.register_parameter('element_embedding', nn.Parameter(torch.Tensor(self.Zmax, self.num_features))) 
        self.config_linear = nn.Linear(self.electron_config.size(1), self.num_features, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.element_embedding, -np.sqrt(3), np.sqrt(3))
        nn.init.orthogonal_(self.config_linear.weight)

    def forward(self, Z):
        embedding = self.element_embedding + self.config_linear(self.electron_config)
        return embedding[Z]
