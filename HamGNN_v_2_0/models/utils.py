"""
/*
 * @Author: Yang Zhong 
 * @Date: 2021-11-29 22:13:49 
 * @Last Modified by: Yang Zhong
 * @Last Modified time: 2021-11-29 22:26:42
 */
"""
from torch_sparse import SparseTensor
import torch
import torch.nn as nn
import numpy as np
from torch.nn import (Linear, Bilinear, Sigmoid, Softplus, ELU, ReLU, SELU, SiLU,
                      CELU, BatchNorm1d, ModuleList, Sequential, Tanh, BatchNorm1d as BN)
from typing import Callable, Union
import re
import torch.nn.functional as F
import matplotlib.pyplot as plt
from easydict import EasyDict
from scipy.stats import gaussian_kde
from typing import Optional
from e3nn import o3

def swish(x):
    return x * x.sigmoid()

def linear_bn_act(in_features: int, out_features: int, lbias: bool = False, activation: Callable = None, use_batch_norm: bool = False):
    if use_batch_norm:
        if activation is None:
            return Sequential(Linear(in_features, out_features, lbias), BN(out_features))
        else:
            return Sequential(Linear(in_features, out_features, lbias), BN(out_features), activation)
    else:
        if activation is None:
            return Linear(in_features, out_features, lbias)
        else:
            return Sequential(Linear(in_features, out_features, lbias), activation)

class SSP(nn.Module):
    r"""Applies element-wise :math:`\text{SSP}(x)=\text{Softplus}(x)-\text{Softplus}(0)`

    Shifted SoftPlus (SSP)

    Args:
        beta: the :math:`\beta` value for the Softplus formulation. Default: 1
        threshold: values above this revert to a linear function. Default: 20

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input
    """

    def __init__(self, beta=1, threshold=20):
        super(SSP, self).__init__()
        self.beta = beta
        self.threshold = threshold

    def forward(self, input):
        sp0 = F.softplus(torch.Tensor([0]), self.beta, self.threshold).item()
        return F.softplus(input, self.beta, self.threshold) - sp0

    def extra_repr(self):
        return 'beta={}, threshold={}'.format(self.beta, self.threshold)

class SWISH(nn.Module):
    def __init__(self):
        super(SWISH, self).__init__()

    def forward(self, input):
        return swish(input)

def get_activation(name):
    act_name = name.lower()
    m = re.match(r"(\w+)\((\d+\.\d+)\)", act_name)
    if m is not None:
        act_name, alpha = m.groups()
        alpha = float(alpha)
        print(act_name, alpha)
    else:
        alpha = 1.0
    if act_name == 'softplus':
        return Softplus()
    elif act_name == 'ssp':
        return SSP()
    elif act_name == 'elu':
        return ELU(alpha)
    elif act_name == 'relu':
        return ReLU()
    elif act_name == 'selu':
        return SELU()
    elif act_name == 'swish':
        return SWISH()
    elif act_name == 'tanh':
        return Tanh()
    elif act_name == 'silu':
        return SiLU()
    elif act_name == 'celu':
        return CELU(alpha)
    else:
        raise NameError("Not supported activation: {}".format(name))

def scatter_plot(pred: np.ndarray = None, target: np.ndarray = None):
    fig, ax = plt.subplots()
    """
        try:
        # Calculate the point density
        xy = np.vstack([pred, target])
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        pred, target, z = pred[idx], target[idx], z[idx]
        # scatter plot
        ax.scatter(x=pred, y=target, s=25, c=z, marker=".")
    except:
        ax.scatter(x=pred, y=target, s=25, c='g', alpha=0.5, marker=".")
    """
    ax.scatter(x=pred, y=target, s=25, c='g', alpha=0.5, marker=".")
    ax.set_title('Prediction VS Target')
    ax.set_aspect('equal')
    min_val, max_val = np.min([target, pred]), np.max([target, pred])
    ax.plot([min_val, max_val], [min_val, max_val],
            ls="--", linewidth=1, c='r')
    plt.xlabel('Prediction', fontsize=15)
    plt.ylabel('Target', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    return fig

class cosine_similarity_loss(nn.Module):
    def __init__(self):
        super(cosine_similarity_loss, self).__init__()

    def forward(self, pred, target):
        vec_product = torch.sum(pred*target, dim=-1)
        pred_norm = torch.norm(pred, p=2, dim=-1)
        target_norm = torch.norm(target, p=2, dim=-1)
        loss = torch.tensor(1.0).type_as(
            pred) - vec_product/(pred_norm*target_norm)
        loss = torch.mean(loss)
        return loss

class sum_zero_loss(nn.Module):
    def __init__(self):
        super(sum_zero_loss, self).__init__()

    def forward(self, pred, target):
        loss = torch.sum(pred, dim=0).pow(2).sum(dim=-1).sqrt()
        return loss

class Euclidean_loss(nn.Module):
    def __init__(self):
        super(Euclidean_loss, self).__init__()

    def forward(self, pred, target):
        dist = (pred - target).pow(2).sum(dim=-1).sqrt()
        loss = torch.mean(dist)
        return loss

class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        return torch.sqrt(self.mse(pred, target))

def parse_metric_func(losses_list: Union[list, tuple] = None):
    for loss_dict in losses_list:
        if loss_dict['metric'].lower() == 'mse':
            loss_dict['metric'] = nn.MSELoss()
        elif loss_dict['metric'].lower() == 'mae':
            loss_dict['metric'] = nn.L1Loss()
        elif loss_dict['metric'].lower() == 'cosine_similarity':
            loss_dict['metric'] = cosine_similarity_loss()
        elif loss_dict['metric'].lower() == 'sum_zero':
            loss_dict['metric'] = sum_zero_loss()
        elif loss_dict['metric'].lower() == 'euclidean_loss':
            loss_dict['metric'] = Euclidean_loss()
        elif loss_dict['metric'].lower() == 'rmse':
            loss_dict['metric'] = RMSELoss()
        else:
            print(f'This metric function is not supported!')
    return losses_list

def get_hparam_dict(config: dict = None):
    if config.setup.GNN_Net.lower() == 'dimnet':
        hparam_dict = config.representation_nets.dimnet_params
    elif config.setup.GNN_Net.lower() == 'edge_gnn':
        hparam_dict = config.representation_nets.Edge_GNN
    elif config.setup.GNN_Net.lower() == 'schnet':
        hparam_dict = config.representation_nets.SchNet
    elif config.setup.GNN_Net.lower() == 'cgcnn':
        hparam_dict = config.representation_nets.cgcnn
    elif config.setup.GNN_Net.lower() == 'cgcnn_edge':
        hparam_dict = config.representation_nets.cgcnn_edge
    elif config.setup.GNN_Net.lower() == 'painn':
        hparam_dict = config.representation_nets.painn
    elif config.setup.GNN_Net.lower() == 'cgcnn_triplet':
        hparam_dict = config.representation_nets.cgcnn_triplet
    elif config.setup.GNN_Net.lower() == 'dimenet_triplet':
        hparam_dict = config.representation_nets.dimenet_triplet
    elif config.setup.GNN_Net.lower() == 'dimeham':
        hparam_dict = config.representation_nets.dimeham
    elif config.setup.GNN_Net.lower() == 'dimeorb':
        hparam_dict = config.representation_nets.dimeorb
    elif config.setup.GNN_Net.lower() == 'schnorb':
        hparam_dict = config.representation_nets.schnorb
    elif config.setup.GNN_Net.lower() == 'nequip':
        hparam_dict = config.representation_nets.nequip
    elif config.setup.GNN_Net.lower() == 'hamgnn_pre':
        hparam_dict = config.representation_nets.HamGNN_pre
    elif config.setup.GNN_Net.lower()[:6] == 'hamgnn':
        hparam_dict = config.representation_nets.HamGNN_pre
    else:
        print(f"The network: {config.setup.GNN_Net} is not yet supported!")
        quit()
    for key in hparam_dict:
        if type(hparam_dict[key]) not in [str, float, int, bool, None]:
            hparam_dict[key] = type(hparam_dict[key]).__name__.split(".")[-1]
    out = {'GNN_Name': config.setup.GNN_Net}
    out.update(dict(hparam_dict))
    return out

def triplets(edge_index, num_nodes, cell_shift):
    row, col = edge_index  # j->i
            
    value = torch.arange(row.size(0), device=row.device)
    adj_t = SparseTensor(
        row=col, col=row, value=value, sparse_sizes=(num_nodes, num_nodes)
    )
    adj_t_row = adj_t[row]
    num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)
                    
    # Node indices (k->j->i) for triplets.
    idx_i = col.repeat_interleave(num_triplets)
    idx_j = row.repeat_interleave(num_triplets)
    idx_k = adj_t_row.storage.col()
                   
    # Edge indices (k-j, j->i) for triplets.
    idx_kj = adj_t_row.storage.value()
    idx_ji = adj_t_row.storage.row()
                       
    """
    idx_i -> pos[idx_i]
    idx_j -> pos[idx_j] - nbr_shift[idx_ji]
    idx_k -> pos[idx_k] - nbr_shift[idx_ji] - nbr_shift[idx_kj]
    """
    # Remove i == k triplets with the same cell_shift.
    relative_cell_shift = cell_shift[idx_kj] + cell_shift[idx_ji]
    mask = (idx_i != idx_k) | torch.any(relative_cell_shift != 0, dim=-1)
    idx_i, idx_j, idx_k, idx_kj, idx_ji = idx_i[mask], idx_j[mask], idx_k[mask], idx_kj[mask], idx_ji[mask]
               
    return col, row, idx_i, idx_j, idx_k, idx_kj, idx_ji

def prod(x):
    """Compute the product of a sequence."""
    out = 1
    for a in x:
        out *= a
    return out

class Expansion(nn.Module):
    def __init__(self, irrep_in, irrep_out_1, irrep_out_2, internal_weights: Optional[bool] = False):
        super().__init__()
        self.irrep_in = irrep_in
        self.irrep_out_1 = irrep_out_1
        self.irrep_out_2 = irrep_out_2
        self.instructions = self.get_expansion_path(irrep_in, irrep_out_1, irrep_out_2)
        self.num_path_weight = sum(prod(ins[-1]) for ins in self.instructions if ins[3])
        self.num_bias = sum([prod(ins[-1][1:]) for ins in self.instructions if ins[0] == 0])
        self.num_weights = self.num_path_weight + self.num_bias
        self.internal_weights = internal_weights
        if self.internal_weights:
            self.weights = nn.Parameter(torch.rand(self.num_path_weight + self.num_bias))
        else:
            self.linear_weight_bias = o3.Linear(self.irrep_in, o3.Irreps([(self.num_weights, (0, 1))]))

    def forward(self, x_in):
        if self.internal_weights:
            weights, bias_weights = None
        else:
            weights, bias_weights = torch.split(self.linear_weight_bias(x_in), 
                                               split_size_or_sections=[self.num_path_weight, self.num_bias], dim=-1)
        batch_num = x_in.shape[0]
        if len(self.irrep_in) == 1:
            x_in_s = [x_in.reshape(batch_num, self.irrep_in[0].mul, self.irrep_in[0].ir.dim)]
        else:
            x_in_s = [
                x_in[:, i].reshape(batch_num, mul_ir.mul, mul_ir.ir.dim)
            for i, mul_ir in zip(self.irrep_in.slices(), self.irrep_in)]

        outputs = {}
        flat_weight_index = 0
        bias_weight_index = 0
        for ins in self.instructions:
            mul_ir_in = self.irrep_in[ins[0]]
            mul_ir_out1 = self.irrep_out_1[ins[1]]
            mul_ir_out2 = self.irrep_out_2[ins[2]]
            x1 = x_in_s[ins[0]]
            x1 = x1.reshape(batch_num, mul_ir_in.mul, mul_ir_in.ir.dim)
            w3j_matrix = o3.wigner_3j(
                mul_ir_out1.ir.l, mul_ir_out2.ir.l, mul_ir_in.ir.l).type_as(x_in)
            if ins[3] is True or weights is not None:
                if weights is None:
                    weight = self.weights[flat_weight_index:flat_weight_index + prod(ins[-1])].reshape(ins[-1])
                    result = torch.einsum(
                        f"wuv, ijk, bwk-> buivj", weight, w3j_matrix, x1) / mul_ir_in.mul
                else:
                    weight = weights[:, flat_weight_index:flat_weight_index + prod(ins[-1])].reshape([-1] + ins[-1])
                    result = torch.einsum(f"bwuv, bwk-> buvk", weight, x1)
                    if ins[0] == 0 and bias_weights is not None:
                        bias_weight = bias_weights[:,bias_weight_index:bias_weight_index + prod(ins[-1][1:])].\
                            reshape([-1] + ins[-1][1:])
                        bias_weight_index += prod(ins[-1][1:])
                        result = result + bias_weight.unsqueeze(-1)
                    result = torch.einsum(f"ijk, buvk->buivj", w3j_matrix, result) / mul_ir_in.mul
                flat_weight_index += prod(ins[-1])
            else:
                result = torch.einsum(
                    f"uvw, ijk, bwk-> buivj", torch.ones(ins[-1]).type(x1.type()).to(self.device), w3j_matrix,
                    x1.reshape(batch_num, mul_ir_in.mul, mul_ir_in.ir.dim)
                )

            result = result.reshape(batch_num, mul_ir_out1.dim, mul_ir_out2.dim)
            key = (ins[1], ins[2])
            if key in outputs.keys():
                outputs[key] = outputs[key] + result
            else:
                outputs[key] = result

        rows = []
        for i in range(len(self.irrep_out_1)):
            blocks = []
            for j in range(len(self.irrep_out_2)):
                if (i, j) not in outputs.keys():
                    blocks += [torch.zeros((x_in.shape[0], self.irrep_out_1[i].dim, self.irrep_out_2[j].dim),
                                           device=x_in.device).type(x_in.type())]
                else:
                    blocks += [outputs[(i, j)]]
            rows.append(torch.cat(blocks, dim=-1))
        output = torch.cat(rows, dim=-2).reshape(batch_num, -1)
        return output

    def get_expansion_path(self, irrep_in, irrep_out_1, irrep_out_2):
        instructions = []
        for  i, (num_in, ir_in) in enumerate(irrep_in):
            for  j, (num_out1, ir_out1) in enumerate(irrep_out_1):
                for k, (num_out2, ir_out2) in enumerate(irrep_out_2):
                    if ir_in in ir_out1 * ir_out2:
                        instructions.append([i, j, k, True, 1.0, [num_in, num_out1, num_out2]])
        return instructions

    @property
    def device(self):
        return next(self.parameters()).device

    def __repr__(self):
        return f'{self.irrep_in} -> {self.irrep_out_1}x{self.irrep_out_1} and bias {self.num_bias}' \
               f'with parameters {self.num_path_weight}'
