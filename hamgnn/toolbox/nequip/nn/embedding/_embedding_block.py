'''
Descripttion: 
version: 
Author: Yang Zhong
Date: 2022-11-29 10:49:47
LastEditors: Wenhai Lu
LastEditTime: 2026-03-18 16:29:28
'''
import torch
import torch.nn.functional

from e3nn.o3 import Irreps
from e3nn.util.jit import compile_mode
from torch.nn import Embedding

from ....nequip.data import AtomicDataDict
from .._graph_mixin import GraphModuleMixin
from .....utils.regression_layers import denseRegression


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

    def __init__(
        self,
        num_types: int,
        num_charge_attr_feas: int,
        apply_charge_doping: bool = False,
        set_features: bool = True,
        irreps_in=None,
    ):
        super().__init__()
        self.num_types = num_types
        self.set_features = set_features
        self.apply_charge_doping = apply_charge_doping
        irreps_out = {AtomicDataDict.NODE_ATTRS_KEY: Irreps([(self.num_types, (0, 1))])}
        if self.set_features:
            irreps_out[AtomicDataDict.NODE_FEATURES_KEY] = irreps_out[AtomicDataDict.NODE_ATTRS_KEY]
        self._init_irreps(irreps_in=irreps_in, irreps_out=irreps_out)
        self.num_charge_attr_feas = num_charge_attr_feas

        if self.apply_charge_doping:
            self.charge_min = -8.0
            self.charge_max = 8.0
            if self.num_charge_attr_feas > 1:
                charge_width = (self.charge_max - self.charge_min) / (self.num_charge_attr_feas - 1)
            else:
                charge_width = 1.0
            charge_centers = torch.linspace(self.charge_min, self.charge_max, steps=self.num_charge_attr_feas)
            self.register_buffer("charge_centers", charge_centers)
            self.register_buffer("charge_gamma", torch.tensor(1.0 / (charge_width ** 2)))
            neutral_charge_attrs = torch.exp(-(1.0 / (charge_width ** 2)) * charge_centers * charge_centers)
            self.register_buffer("neutral_charge_attrs", neutral_charge_attrs.view(1, -1))
            self.mlp_q = denseRegression(in_features=self.num_charge_attr_feas, out_features=self.num_types,
                                         bias=True, use_batch_norm=False, n_h=2)

    def _expand_charge_to_nodes(self, data, node_attrs):
        if AtomicDataDict.DOPING_CHARGE_KEY in data:
            doping_charge = data[AtomicDataDict.DOPING_CHARGE_KEY]
        else:
            doping_charge = getattr(data, AtomicDataDict.DOPING_CHARGE_KEY)
        doping_charge = doping_charge.to(device=node_attrs.device, dtype=node_attrs.dtype)
        if doping_charge.dim() == 0:
            doping_charge = doping_charge.view(1)
        if doping_charge.dim() == 1:
            doping_charge = doping_charge.view(-1, 1)

        if AtomicDataDict.BATCH_KEY in data:
            batch = data[AtomicDataDict.BATCH_KEY]
        else:
            batch = getattr(data, "batch", None)

        if batch is not None and doping_charge.size(0) != node_attrs.size(0):
            return doping_charge[batch.view(-1)]
        if doping_charge.size(0) == node_attrs.size(0):
            return doping_charge
        return doping_charge[:1].expand(node_attrs.size(0), -1)

    def _build_charge_attrs(self, per_node_charge):
        per_node_charge = per_node_charge.clamp(self.charge_min, self.charge_max)
        diff = per_node_charge - self.charge_centers.view(1, -1)
        return torch.exp(-self.charge_gamma * diff * diff)

    def forward(self, data: AtomicDataDict.Type):
        type_numbers = data[AtomicDataDict.ATOMIC_NUMBERS_KEY]
        node_attrs = torch.nn.functional.one_hot(
            type_numbers,
            num_classes=self.num_types,
        ).to(device=type_numbers.device, dtype=data[AtomicDataDict.POSITIONS_KEY].dtype)

        if self.apply_charge_doping:
            per_node_charge = self._expand_charge_to_nodes(data, node_attrs)
            q_attrs = self._build_charge_attrs(per_node_charge)
            neutral_q_attrs = self.neutral_charge_attrs.expand(q_attrs.size(0), -1)
            # Treat charge as a residual correction on top of the original one-hot node encoding.
            node_attrs = node_attrs + self.mlp_q(q_attrs) - self.mlp_q(neutral_q_attrs)

        data[AtomicDataDict.NODE_ATTRS_KEY] = node_attrs
        if self.set_features:
            data[AtomicDataDict.NODE_FEATURES_KEY] = node_attrs
        return data
