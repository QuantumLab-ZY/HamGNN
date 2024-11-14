'''
Descripttion: 
version: 
Author: Yang Zhong
Date: 2022-11-29 10:49:47
LastEditors: Yang Zhong
LastEditTime: 2024-05-10 09:28:13
'''
import torch
import torch.nn.functional

from e3nn.o3 import Irreps
from e3nn.util.jit import compile_mode
from torch.nn import Embedding

from ....nequip.data import AtomicDataDict
from .._graph_mixin import GraphModuleMixin
from .....layers import denseRegression


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

class Embedding_block_q(GraphModuleMixin, torch.nn.Module):
    num_types: int
    set_features: bool

    # TODO: use torch.unique?
    # TODO: type annotation
    # Docstrings
    def __init__(
        self,
        num_node_attr_feas: int,
        num_charge_attr_feas: int,
        apply_charge_doping: bool = False,
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
        self.num_charge_attr_feas = num_charge_attr_feas
        
        self.emb = Embedding(119, self.num_node_attr_feas)
        
        self.apply_charge_doping = apply_charge_doping
        if self.apply_charge_doping:
            self.emb_q = Embedding(18, self.num_charge_attr_feas)
            self.mlp_node = denseRegression(in_features=self.num_node_attr_feas, out_features=self.num_node_attr_feas,bias=True,
                                            use_batch_norm=False, n_h=2)
            self.mlp_q = denseRegression(in_features=self.num_charge_attr_feas, out_features=self.num_node_attr_feas,bias=True,
                                        use_batch_norm=False, n_h=2)

    def forward(self, data: AtomicDataDict.Type):
        type_numbers = data[AtomicDataDict.ATOMIC_NUMBERS_KEY]
        node_attrs = self.emb(type_numbers) # Vi
        
        # embed the charge attr
        if self.apply_charge_doping:
            q_num = data.doping_charge.long() + 9 # 假设背景电荷只取-5到5之间
            q_attrs = self.emb_q(q_num)
            node_attrs = self.mlp_node(node_attrs) + self.mlp_q(q_attrs)
        
        data[AtomicDataDict.NODE_ATTRS_KEY] = node_attrs
        if self.set_features:
            data[AtomicDataDict.NODE_FEATURES_KEY] = node_attrs
        return data