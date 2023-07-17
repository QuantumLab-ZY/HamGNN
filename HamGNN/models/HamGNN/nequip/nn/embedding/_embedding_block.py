'''
Descripttion: 
version: 
Author: Yang Zhong
Date: 2022-11-29 10:49:47
LastEditors: Yang Zhong
LastEditTime: 2022-11-29 10:55:12
'''
import torch
import torch.nn.functional

from e3nn.o3 import Irreps
from e3nn.util.jit import compile_mode
from torch.nn import Embedding

from ....nequip.data import AtomicDataDict
from .._graph_mixin import GraphModuleMixin


class Embedding_block(GraphModuleMixin, torch.nn.Module):
    num_types: int
    set_features: bool

    # TODO: use torch.unique?
    # TODO: type annotation
    # Docstrings
    def __init__(
        self,
        num_node_attr_feas: int,
        set_features: bool = True,
        irreps_in=None,
    ):
        super().__init__()
        self.num_node_attr_feas = num_node_attr_feas
        self.set_features = set_features
        # Output irreps are num_types even (invariant) scalars
        irreps_out = {AtomicDataDict.NODE_ATTRS_KEY: Irreps([(self.num_node_attr_feas, (0, 1))])}
        if self.set_features:
            irreps_out[AtomicDataDict.NODE_FEATURES_KEY] = irreps_out[
                AtomicDataDict.NODE_ATTRS_KEY
            ]
        self._init_irreps(irreps_in=irreps_in, irreps_out=irreps_out)
        
        self.emb = Embedding(119, self.num_node_attr_feas)

    def forward(self, data: AtomicDataDict.Type):
        type_numbers = data[AtomicDataDict.ATOMIC_NUMBERS_KEY]
        node_attrs = self.emb(type_numbers)
        data[AtomicDataDict.NODE_ATTRS_KEY] = node_attrs
        if self.set_features:
            data[AtomicDataDict.NODE_FEATURES_KEY] = node_attrs
        return data
