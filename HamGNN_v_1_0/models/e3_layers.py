
# The following codes borrowed from https://github.com/Xiaoxun-Gong/DeepH-E3 
"""
MIT License

Copyright (c) 2023 Xiaoxun-Gong

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.utils import degree

from e3nn.o3 import Irrep, Irreps, wigner_3j, matrix_to_angles, Linear, FullyConnectedTensorProduct, TensorProduct, SphericalHarmonics
from e3nn.nn import Extract

import sympy as sym

import numpy as np
from scipy.optimize import brentq
from scipy import special as sp
import math


class PolynomialCutoff(nn.Module):
    def __init__(self, r_max, p=6):
        r"""Polynomial cutoff, as proposed in DimeNet: https://arxiv.org/abs/2003.03123


        Parameters
        ----------
        r_max : float
            Cutoff radius

        p : int
            Power used in envelope function
        """
        super(PolynomialCutoff, self).__init__()

        self.register_buffer("p", torch.Tensor([p]))
        self.register_buffer("r_max", torch.Tensor([r_max]))

    def forward(self, x):
        """
        Evaluate cutoff function.

        x: torch.Tensor, input distance
        """
        envelope = (
            1.0
            - ((self.p + 1.0) * (self.p + 2.0) / 2.0)
            * torch.pow(x / self.r_max, self.p)
            + self.p * (self.p + 2.0) * torch.pow(x / self.r_max, self.p + 1.0)
            - (self.p * (self.p + 1.0) / 2) * torch.pow(x / self.r_max, self.p + 2.0)
        )

        envelope *= (x < self.r_max).float()
        return envelope

def Jn(r, n):
    """
    numerical spherical bessel functions of order n
    """
    return np.sqrt(np.pi/(2*r)) * sp.jv(n+0.5, r)

def Jn_zeros(n, k):
    """
    Compute the first k zeros of the spherical bessel functions up to order n (excluded)
    """
    zerosj = np.zeros((n, k), dtype="float32")
    zerosj[0] = np.arange(1, k + 1) * np.pi
    points = np.arange(1, k + n) * np.pi
    racines = np.zeros(k + n - 1, dtype="float32")
    for i in range(1, n):
        for j in range(k + n - 1 - i):
            foo = brentq(Jn, points[j], points[j + 1], (i,))
            racines[j] = foo
        points = racines
        zerosj[i][:k] = racines[:k]

    return zerosj

def spherical_bessel_formulas(n):
    """
    Computes the sympy formulas for the spherical bessel functions up to order n (excluded)
    """
    x = sym.symbols('x')

    f = [sym.sin(x)/x]
    a = sym.sin(x)/x
    for i in range(1, n):
        b = sym.diff(a, x)/x
        f += [sym.simplify(b*(-x)**i)]
        a = sym.simplify(b)
    return f

def bessel_basis(n, k):
    """
    Compute the sympy formulas for the normalized and rescaled spherical bessel functions up to
    order n (excluded) and maximum frequency k (excluded).
    """

    zeros = Jn_zeros(n, k)
    normalizer = []
    for order in range(n):
        normalizer_tmp = []
        for i in range(k):
            normalizer_tmp += [0.5*Jn(zeros[order, i], order+1)**2]
        normalizer_tmp = 1/np.array(normalizer_tmp)**0.5
        normalizer += [normalizer_tmp]

    f = spherical_bessel_formulas(n)
    x = sym.symbols('x')
    bess_basis = []
    for order in range(n):
        bess_basis_tmp = []
        for i in range(k):
            bess_basis_tmp += [sym.simplify(normalizer[order]
                                            [i]*f[order].subs(x, zeros[order, i]*x))]
        bess_basis += [bess_basis_tmp]
    return bess_basis

def flt2cplx(flt_dtype):
    if flt_dtype == torch.float32:
        cplx_dtype = torch.complex64
    elif flt_dtype == torch.float64:
        cplx_dtype = torch.complex128
    elif flt_dtype == np.float32:
        cplx_dtype = np.complex64
    elif flt_dtype == np.float64:
        cplx_dtype = np.complex128
    else:
        raise NotImplementedError(f'Unsupported float dtype: {flt_dtype}')
    return cplx_dtype

def irreps_from_l1l2(l1, l2, mul, spinful, no_parity=False):
    r'''
    non-spinful example: l1=1, l2=2 (1x2) ->
    required_irreps_full=1+2+3, required_irreps=1+2+3, required_irreps_x1=None
    
    spinful example: l1=1, l2=2 (1x0.5)x(2x0.5) ->
    required_irreps_full = 1+2+3 + 0+1+2 + 1+2+3 + 2+3+4
    required_irreps = (1+2+3)x0 = 1+2+3
    required_irreps_x1 = (1+2+3)x1 = [0+1+2, 1+2+3, 2+3+4]
    
    notice that required_irreps_x1 is a list of Irreps
    '''
    p = 1
    if not no_parity:
        p = (-1) ** (l1 + l2)
    required_ls = range(abs(l1 - l2), l1 + l2 + 1)
    required_irreps = Irreps([(mul, (l, p)) for l in required_ls])
    required_irreps_full = required_irreps
    required_irreps_x1 = None
    if spinful:
        required_irreps_x1 = []
        for _, ir in required_irreps:
            required_ls_irx1 = range(abs(ir.l - 1), ir.l + 1 + 1)
            irx1 = Irreps([(mul, (l, p)) for l in required_ls_irx1])
            required_irreps_x1.append(irx1)
            required_irreps_full += irx1
    return required_irreps_full, required_irreps, required_irreps_x1

class Rotate:
    def __init__(self, default_dtype_torch, device_torch='cpu', spinful=False):
        sqrt_2 = 1.4142135623730951
            
        self.spinful = spinful
        if spinful:
            assert default_dtype_torch in [torch.complex64, torch.complex128]
        else:
            assert default_dtype_torch in [torch.float32, torch.float64]
        
        # openmx的实球谐函数基组变复球谐函数
        self.Us_openmx = {
            0: torch.tensor([1], dtype=torch.cfloat, device=device_torch),
            1: torch.tensor([[-1 / sqrt_2, 1j / sqrt_2, 0], [0, 0, 1], [1 / sqrt_2, 1j / sqrt_2, 0]], dtype=torch.cfloat, device=device_torch),
            2: torch.tensor([[0, 1 / sqrt_2, -1j / sqrt_2, 0, 0],
                             [0, 0, 0, -1 / sqrt_2, 1j / sqrt_2],
                             [1, 0, 0, 0, 0],
                             [0, 0, 0, 1 / sqrt_2, 1j / sqrt_2],
                             [0, 1 / sqrt_2, 1j / sqrt_2, 0, 0]], dtype=torch.cfloat, device=device_torch),
            3: torch.tensor([[0, 0, 0, 0, 0, -1 / sqrt_2, 1j / sqrt_2],
                             [0, 0, 0, 1 / sqrt_2, -1j / sqrt_2, 0, 0],
                             [0, -1 / sqrt_2, 1j / sqrt_2, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0, 0, 0],
                             [0, 1 / sqrt_2, 1j / sqrt_2, 0, 0, 0, 0],
                             [0, 0, 0, 1 / sqrt_2, 1j / sqrt_2, 0, 0],
                             [0, 0, 0, 0, 0, 1 / sqrt_2, 1j / sqrt_2]], dtype=torch.cfloat, device=device_torch),
        }
        # openmx的实球谐函数基组变wiki的实球谐函数 https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Real_spherical_harmonics
        self.Us_openmx2wiki = {
            0: torch.eye(1, dtype=default_dtype_torch).to(device=device_torch),
            1: torch.eye(3, dtype=default_dtype_torch)[[1, 2, 0]].to(device=device_torch),
            2: torch.eye(5, dtype=default_dtype_torch)[[2, 4, 0, 3, 1]].to(device=device_torch),
            3: torch.eye(7, dtype=default_dtype_torch)[[6, 4, 2, 0, 1, 3, 5]].to(device=device_torch)
        }
        self.Us_wiki2openmx = {k: v.T for k, v in self.Us_openmx2wiki.items()}
        if spinful:
            self.Us_openmx2wiki_sp = {}
            for k, v in self.Us_openmx2wiki.items():
                self.Us_openmx2wiki_sp[k] = torch.block_diag(v, v)
        
        self.dtype = default_dtype_torch 

    def rotate_e3nn_v(self, v, R, l, order_xyz=True):
        if order_xyz:
            # R是(x, y, z)顺序
            R_e3nn = self.rotate_matrix_convert(R)
            # R_e3nn是(y, z, x)顺序
        else:
            # R是(y, z, x)顺序
            R_e3nn = R
        return v @ Irrep(l, 1).D_from_matrix(R_e3nn)
    # 注: 在e3nn的规范中, 旋转矩阵是作用在向量左边的. 
    # 但是这里旋转矩阵默认是作用在向量右边的, 这是为了方便处理向量开头可能还有的额外维度.
    # 解决这个问题很容易, 只要让输入到这里的矩阵R是作用在向量右边的就可以. 因为R转置后, 输出的D也恰好被转置.

    def rotate_openmx_H(self, H, R, l_left, l_right, order_xyz=True):
        if order_xyz:
            # R是(x, y, z)顺序
            R_e3nn = self.rotate_matrix_convert(R)
            # R_e3nn是(y, z, x)顺序
        else:
            # R是(y, z, x)顺序
            R_e3nn = R
        return self.Us_openmx2wiki[l_left].T @ Irrep(l_left, 1).D_from_matrix(R_e3nn).transpose(-1, -2) @ self.Us_openmx2wiki[l_left] @ H \
               @ self.Us_openmx2wiki[l_right].T @ Irrep(l_right, 1).D_from_matrix(R_e3nn) @ self.Us_openmx2wiki[l_right]
    
    def rotate_openmx_H_full(self, H, R, orbital_types_left, orbital_types_right, order_xyz=True):
        assert len(R.shape) == 2 # TODO: does not support batch operation
        if order_xyz:
            # R是(x, y, z)顺序
            R_e3nn = self.rotate_matrix_convert(R)
            # R_e3nn是(y, z, x)顺序
        else:
            # R是(y, z, x)顺序
            R_e3nn = R
        irreps_left = Irreps([(1, (l, (- 1) ** l)) for l in orbital_types_left])
        irreps_right = Irreps([(1, (l, (- 1) ** l)) for l in orbital_types_right])
        U_left = irreps_left.D_from_matrix(R_e3nn)
        U_right = irreps_right.D_from_matrix(R_e3nn)
        openmx2wiki_left, openmx2wiki_right = self.openmx2wiki_left_right(orbital_types_left, orbital_types_right)
        if self.spinful:
            U_left = torch.kron(self.D_one_half(R_e3nn), U_left)
            U_right = torch.kron(self.D_one_half(R_e3nn), U_right)
        return openmx2wiki_left.T @ U_left.transpose(-1, -2).conj() @ openmx2wiki_left @ H \
               @ openmx2wiki_right.T @ U_right @ openmx2wiki_right

    def wiki2openmx_H_full(self, H, orbital_types_left, orbital_types_right):
        openmx2wiki_left, openmx2wiki_right = self.openmx2wiki_left_right(orbital_types_left, orbital_types_right)
        return openmx2wiki_left.T @ H @ openmx2wiki_right

    def openmx2wiki_H_full(self, H, orbital_types_left, orbital_types_right):
        openmx2wiki_left, openmx2wiki_right = self.openmx2wiki_left_right(orbital_types_left, orbital_types_right)
        return openmx2wiki_left @ H @ openmx2wiki_right.T
    
    def wiki2openmx_H(self, H, l_left, l_right):
        return self.Us_openmx2wiki[l_left].T @ H @ self.Us_openmx2wiki[l_right]

    def openmx2wiki_H(self, H, l_left, l_right):
        return self.Us_openmx2wiki[l_left] @ H @ self.Us_openmx2wiki[l_right].T

    def openmx2wiki_left_right(self, orbital_types_left, orbital_types_right):
        if isinstance(orbital_types_left, int):
            orbital_types_left = [orbital_types_left]
        if isinstance(orbital_types_right, int):
            orbital_types_right = [orbital_types_right]
        openmx2wiki_left = torch.block_diag(*[self.Us_openmx2wiki[l] for l in orbital_types_left])
        openmx2wiki_right = torch.block_diag(*[self.Us_openmx2wiki[l] for l in orbital_types_right])
        if self.spinful:
            openmx2wiki_left = torch.block_diag(openmx2wiki_left, openmx2wiki_left)
            openmx2wiki_right = torch.block_diag(openmx2wiki_right, openmx2wiki_right)
        return openmx2wiki_left, openmx2wiki_right
    
    def rotate_matrix_convert(self, R):
        # (x, y, z)顺序排列的旋转矩阵转换为(y, z, x)顺序(see e3nn.o3.spherical_harmonics() and https://docs.e3nn.org/en/stable/guide/change_of_basis.html)
        return torch.eye(3)[[1, 2, 0]] @ R @ torch.eye(3)[[1, 2, 0]].T # todo: cuda
    
    def D_one_half(self, R):
        # 输入到这个函数的R应为y,z,x顺序
        # 这里没有考虑空间反演, 即假设l=1/2的不等价不可约表示带有偶宇称
        assert self.spinful
        d = torch.det(R).sign()
        R = d[..., None, None] * R
        k = (1 - d) / 2 # parity index
        alpha, beta, gamma = matrix_to_angles(R)
        J = torch.tensor([[1, 1], [1j, -1j]], dtype=self.dtype) / 1.4142135623730951 # <1/2 mz|1/2 my>
        Uz1 = self._sp_z_rot(alpha)
        Uy = J @ self._sp_z_rot(beta) @ J.T.conj()
        Uz2 = self._sp_z_rot(gamma)
        return Uz1 @ Uy @ Uz2
        # return torch.eye(2, dtype=self.dtype)
    
    def _sp_z_rot(self, angle):
        # [[e^{-ia/2}, 0], [0, e^{ia/2}]]
        assert self.spinful
        M = torch.zeros([*angle.shape, 2, 2], dtype=self.dtype)
        inds = torch.tensor([0, 1])
        freqs = torch.tensor([0.5, -0.5], dtype=self.dtype)
        M[..., inds, inds] = torch.exp(- freqs * (1j) * angle[..., None])
        return M

class sort_irreps(torch.nn.Module):
    def __init__(self, irreps_in):
        super().__init__()
        irreps_in = Irreps(irreps_in)
        sorted_irreps = irreps_in.sort()
        
        irreps_out_list = [((mul, ir),) for mul, ir in sorted_irreps.irreps]
        instructions = [(i,) for i in sorted_irreps.inv]
        self.extr = Extract(irreps_in, irreps_out_list, instructions)
        
        irreps_in_list = [((mul, ir),) for mul, ir in irreps_in]
        instructions_inv = [(i,) for i in sorted_irreps.p]
        self.extr_inv = Extract(sorted_irreps.irreps, irreps_in_list, instructions_inv)
        
        self.irreps_in = irreps_in
        self.irreps_out = sorted_irreps.irreps.simplify()
    
    def forward(self, x):
        r'''irreps_in -> irreps_out'''
        extracted = self.extr(x)
        return torch.cat(extracted, dim=-1)

    def inverse(self, x):
        r'''irreps_out -> irreps_in'''
        extracted_inv = self.extr_inv(x)
        return torch.cat(extracted_inv, dim=-1)

class e3TensorDecomp:
    def __init__(self, net_irreps_out, out_js_list, default_dtype_torch = torch.float32, nao_max=26, spinful=False, no_parity=False, if_sort=False):
        if spinful:
            default_dtype_torch = flt2cplx(default_dtype_torch)
        self.dtype = default_dtype_torch
        self.spinful = spinful
        self.nao_max = nao_max
        
        self.out_js_list = out_js_list
        if net_irreps_out is not None:
            net_irreps_out = Irreps(net_irreps_out)

        required_irreps_out = Irreps(None)
        in_slices = [0]
        wms = [] # wm = wigner_multiplier
        H_slices = [0]
        wms_H = []
        if spinful:
            in_slices_sp = []
            H_slices_sp = []
            wms_sp = []
            wms_sp_H = []
            
        for H_l1, H_l2 in out_js_list:
            
            # = construct required_irreps_out =
            mul = 1
            # if net_irreps_out is not None:
            #     if len(net_irreps_out) < len(required_irreps_out) + 1:
            #         raise ValueError('Net irreps out and target does not match')
            #     mul = net_irreps_out[len(required_irreps_out)].mul
            _, required_irreps_out_single, required_irreps_x1 = irreps_from_l1l2(H_l1, H_l2, mul, spinful, no_parity=no_parity)
            required_irreps_out += required_irreps_out_single
            
            # spinful case, example: (1x0.5)x(2x0.5) = (1+2+3)x(0+1) = (1+2+3)+(0+1+2)+(1+2+3)+(2+3+4)
            # everything on r.h.s. as a whole constitutes a slice in in_slices
            # each bracket on r.h.s. above corresponds to a slice in in_slice_sp
            if spinful:
                in_slice_sp = [0, required_irreps_out_single.dim]
                H_slice_sp = [0]
                wm_sp = [None]
                wm_sp_H = []
                for (_a, ir), ir_times_1 in zip(required_irreps_out_single, required_irreps_x1):
                    required_irreps_out += ir_times_1
                    in_slice_sp.append(in_slice_sp[-1] + ir_times_1.dim)
                    H_slice_sp.append(H_slice_sp[-1] + ir.dim)
                    wm_irx1 = []
                    wm_irx1_H = []
                    for _b, ir_1 in ir_times_1:
                        for _c in range(mul):
                            wm_irx1.append(wigner_3j(ir.l, 1, ir_1.l, dtype=default_dtype_torch))
                            wm_irx1_H.append(wigner_3j(ir_1.l, ir.l, 1, dtype=default_dtype_torch) * (2 * ir_1.l + 1))
                            # wm_irx1.append(wigner_3j(ir.l, 1, ir_1.l, dtype=default_dtype_torch, device=device_torch) * sqrt(2 * ir_1.l + 1))
                            # wm_irx1_H.append(wigner_3j(ir_1.l, ir.l, 1, dtype=default_dtype_torch, device=device_torch) * sqrt(2 * ir_1.l + 1))
                    wm_irx1 = torch.cat(wm_irx1, dim=-1)
                    wm_sp.append(wm_irx1)
                    wm_irx1_H = torch.cat(wm_irx1_H, dim=0)
                    wm_sp_H.append(wm_irx1_H)
            
            # = construct slices =
            in_slices.append(required_irreps_out.dim)
            H_slices.append(H_slices[-1] + (2 * H_l1 + 1) * (2 * H_l2 + 1))
            if spinful:
                in_slices_sp.append(in_slice_sp)
                H_slices_sp.append(H_slice_sp)
            
            # = get CG coefficients multiplier to act on net_out =
            wm = []
            wm_H = []
            for _a, ir in required_irreps_out_single:
                for _b in range(mul):
                    # about this 2l+1: 
                    # we want the exact inverse of the w_3j symbol, i.e. torch.einsum("ijk,jkl->il",w_3j(l,l1,l2),w_3j(l1,l2,l))==torch.eye(...). but this is not the case, since the CG coefficients are unitary and w_3j differ from CG coefficients by a constant factor. but we know from https://en.wikipedia.org/wiki/3-j_symbol#Mathematical_relation_to_Clebsch%E2%80%93Gordan_coefficients that 2l+1 is exactly the factor we want.
                    wm.append(wigner_3j(H_l1, H_l2, ir.l, dtype=default_dtype_torch))
                    wm_H.append(wigner_3j(ir.l, H_l1, H_l2, dtype=default_dtype_torch) * (2 * ir.l + 1))
                    # wm.append(wigner_3j(H_l1, H_l2, ir.l, dtype=default_dtype_torch, device=device_torch) * sqrt(2 * ir.l + 1))
                    # wm_H.append(wigner_3j(ir.l, H_l1, H_l2, dtype=default_dtype_torch, device=device_torch) * sqrt(2 * ir.l + 1))
            wm = torch.cat(wm, dim=-1)
            wm_H = torch.cat(wm_H, dim=0)
            wms.append(wm)
            wms_H.append(wm_H)
            if spinful:
                wms_sp.append(wm_sp)
                wms_sp_H.append(wm_sp_H)
            
        # = check net irreps out =
        if spinful:
            required_irreps_out = required_irreps_out + required_irreps_out
        if net_irreps_out is not None:
            if if_sort:
                assert net_irreps_out == required_irreps_out.sort().irreps.simplify(), f'requires {required_irreps_out.sort().irreps.simplify()} but got {net_irreps_out}'
            else:
                assert net_irreps_out == required_irreps_out, f'requires {required_irreps_out} but got {net_irreps_out}'
        
        self.in_slices = in_slices
        self.wms = wms
        self.H_slices = H_slices
        self.wms_H = wms_H
        if spinful:
            self.in_slices_sp = in_slices_sp
            self.H_slices_sp = H_slices_sp
            self.wms_sp = wms_sp
            self.wms_sp_H = wms_sp_H

        # = register rotate kernel =
        self.rotate_kernel = Rotate(default_dtype_torch, spinful=spinful)
        
        if spinful:
            sqrt2 = 1.4142135623730951
                                            #  0,   y,   z,   x
            # self.oyzx2spin = torch.tensor([[   0, -1, 1j,   0,  ],  # uu
            #                                [   1,  0,  0,   1,  ],  # ud
            #                                [  -1,  0,  0,   1,  ],  # du
            #                                [   0,  1, 1j,   0,  ]], # dd
            #                                dtype=default_dtype_torch) / sqrt2
            self.oyzx2spin = torch.tensor([[  1,   0,   1,   0],
                                           [  0, -1j,   0,   1],
                                           [  0,  1j,   0,   1],
                                           [  1,   0,  -1,   0]],
                                            dtype=default_dtype_torch) / sqrt2
        
        self.sort = None
        if if_sort:
            self.sort = sort_irreps(required_irreps_out)
        
        if self.sort is not None:
            self.required_irreps_out = self.sort.irreps_out
        else:
            self.required_irreps_out = required_irreps_out
    
    def get_H(self, net_out):
        r''' get openmx type H from net output '''
        if self.sort is not None:
            net_out = self.sort.inverse(net_out)
        if self.spinful:
            half_len = int(net_out.shape[-1] / 2)
            re = net_out[:, :half_len]
            im = net_out[:, half_len:]
            net_out = re + 1j * im
        
        if self.spinful:
            block = torch.zeros(net_out.shape[0], 4, self.nao_max, self.nao_max).type_as(net_out)
        else:
            block = torch.zeros(net_out.shape[0], self.nao_max, self.nao_max).type_as(net_out)
        num_irreps_row = int(math.sqrt(len(self.out_js_list)))
        
        start_i, start_j = 0, 0
        for i, (li, lj) in enumerate(self.out_js_list):
            in_slice = slice(self.in_slices[i], self.in_slices[i + 1])
            net_out_block = net_out[:, in_slice]
            n_i, n_j = int(2*li+1), int(2*lj+1)
            blockpart = block.narrow(-2,start_i,n_i).narrow(-1,start_j,n_j) # shape: (Nbatch, (4,) n_i, n_j)
            
            if self.spinful:
                # (1+2+3)+(0+1+2)+(1+2+3)+(2+3+4) -> (1+2+3)x(0+1)
                H_block = []
                for j in range(len(self.wms_sp[i])):
                    in_slice_sp = slice(self.in_slices_sp[i][j], self.in_slices_sp[i][j + 1])
                    if j == 0:
                        H_block.append(net_out_block[:, in_slice_sp].unsqueeze(-1))
                    else:
                        H_block.append(torch.einsum('jkl,il->ijk', self.wms_sp[i][j].type_as(net_out), net_out_block[:, in_slice_sp]))
                H_block = torch.cat([H_block[0], torch.cat(H_block[1:], dim=-2)], dim=-1)
                # (1+2+3)x(0+1) -> (uu,ud,du,dd)x(1x2)
                H_block = torch.einsum('imn,klm,jn->ijkl', H_block, self.wms[i].type_as(net_out), self.oyzx2spin.type_as(net_out))
                # H_block = self.rotate_kernel.wiki2openmx_H(H_block, *self.out_js_list[i])
                blockpart += H_block.reshape(net_out.shape[0], 4, n_i, n_j)
            else:
                H_block = torch.sum(self.wms[i][None, :, :, :].type_as(net_out) * net_out_block[:, None, None, :], dim=-1)
                # H_block = self.rotate_kernel.wiki2openmx_H(H_block, *self.out_js_list[i])
                blockpart += H_block.reshape(net_out.shape[0], n_i, n_j)
                
            if (i+1) % num_irreps_row == 0:
                start_i += n_i
                start_j = 0
            else:
                start_j += n_j
        return block # output shape: [edge, (4 spin components,) H_flattened_concatenated]

    def get_net_out(self, H):
        r'''get net output from openmx type H'''
        out = []
        for i in range(len(self.out_js_list)):
            H_slice = slice(self.H_slices[i], self.H_slices[i + 1])
            l1, l2 = self.out_js_list[i]
            if self.spinful:
                H_block = H[..., H_slice].reshape(-1, 4, 2 * l1 + 1, 2 * l2 + 1)
                H_block = self.rotate_kernel.openmx2wiki_H(H_block, *self.out_js_list[i])
                # (uu,ud,du,dd)x(1x2) -> (1+2+3)x(0+1)
                H_block = torch.einsum('ilmn,jmn,kl->ijk', H_block, self.wms_H[i], self.oyzx2spin.T.conj())
                # (1+2+3)x(0+1) -> (1+2+3)+(0+1+2)+(1+2+3)+(2+3+4)
                net_out_block = [H_block[:, :, 0]]
                for j in range(len(self.wms_sp_H[i])):
                    H_slice_sp = slice(self.H_slices_sp[i][j], self.H_slices_sp[i][j + 1])
                    net_out_block.append(torch.einsum('jlm,ilm->ij', self.wms_sp_H[i][j], H_block[:, H_slice_sp, 1:]))
                net_out_block = torch.cat(net_out_block, dim=-1)
                out.append(net_out_block)
            else:
                H_block = H[:, H_slice].reshape(-1, 2 * l1 + 1, 2 * l2 + 1)
                H_block = self.rotate_kernel.openmx2wiki_H(H_block, *self.out_js_list[i])
                net_out_block = torch.sum(self.wms_H[i][None, :, :, :] * H_block[:, None, :, :], dim=(-1, -2))
                out.append(net_out_block)
        out = torch.cat(out, dim=-1)
        if self.spinful:
            out = torch.cat([out.real, out.imag], dim=-1)
        if self.sort is not None:
            out = self.sort(out)
        return out

    def convert_mask(self, mask):
        assert self.spinful
        num_edges = mask.shape[0]
        mask = mask.permute(0, 2, 1).reshape(num_edges, -1).repeat(1, 2)
        if self.sort is not None:
            mask = self.sort(mask)
        return mask

class e3LayerNorm(nn.Module):
    def __init__(self, irreps_in, eps=1e-5, affine=True, normalization='component', subtract_mean=True, divide_norm=False):
        super().__init__()
        
        self.irreps_in = Irreps(irreps_in)
        self.eps = eps
        
        if affine:          
            ib, iw = 0, 0
            weight_slices, bias_slices = [], []
            for mul, ir in irreps_in:
                if ir.is_scalar(): # bias only to 0e
                    bias_slices.append(slice(ib, ib + mul))
                    ib += mul
                else:
                    bias_slices.append(None)
                weight_slices.append(slice(iw, iw + mul))
                iw += mul
            self.weight = nn.Parameter(torch.ones([iw]))
            self.bias = nn.Parameter(torch.zeros([ib]))
            self.bias_slices = bias_slices
            self.weight_slices = weight_slices
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        
        self.subtract_mean = subtract_mean
        self.divide_norm = divide_norm
        assert normalization in ['component', 'norm']
        self.normalization = normalization
            
        self.reset_parameters()
    
    def reset_parameters(self):
        if self.weight is not None:
            self.weight.data.fill_(1)
            # nn.init.uniform_(self.weight)
        if self.bias is not None:
            self.bias.data.fill_(0)
            # nn.init.uniform_(self.bias)

    def forward(self, x: torch.Tensor, batch: torch.Tensor = None):
        # input x must have shape [num_node(edge), dim]
        # if first dimension of x is node index, then batch should be batch.batch
        # if first dimension of x is edge index, then batch should be batch.batch[batch.edge_index[0]]
        
        if batch is None:
            batch = torch.full([x.shape[0]], 0, dtype=torch.int64)

        # from torch_geometric.nn.norm.LayerNorm

        batch_size = int(batch.max()) + 1 
        batch_degree = degree(batch, batch_size, dtype=torch.int64).clamp_(min=1).to(dtype=x.dtype)
        
        out = []
        ix = 0
        for index, (mul, ir) in enumerate(self.irreps_in):        
            field = x[:, ix: ix + mul * ir.dim].reshape(-1, mul, ir.dim) # [node, mul, repr]
            
            # compute and subtract mean
            if self.subtract_mean or ir.l == 0: # do not subtract mean for l>0 irreps if subtract_mean=False
                mean = scatter(field, batch, dim=0, dim_size=batch_size,
                            reduce='add').mean(dim=1, keepdim=True) / batch_degree[:, None, None] # scatter_mean does not support complex number
                field = field - mean[batch]
                
            # compute and divide norm
            if self.divide_norm or ir.l == 0: # do not divide norm for l>0 irreps if subtract_mean=False
                norm = scatter(field.abs().pow(2), batch, dim=0, dim_size=batch_size,
                            reduce='mean').mean(dim=[1,2], keepdim=True) # add abs here to deal with complex numbers
                if self.normalization == 'norm':
                    norm = norm * ir.dim
                field = field / (norm.sqrt()[batch] + self.eps)
            
            # affine
            if self.weight is not None:
                weight = self.weight[self.weight_slices[index]]
                field = field * weight[None, :, None]
            if self.bias is not None and ir.is_scalar():
                bias = self.bias[self.bias_slices[index]]
                field = field + bias[None, :, None]
            
            out.append(field.reshape(-1, mul * ir.dim))
            ix += mul * ir.dim
            
        out = torch.cat(out, dim=-1)
                
        return out

class e3ElementWise:
    def __init__(self, irreps_in):
        self.irreps_in = Irreps(irreps_in)
        
        len_weight = 0
        for mul, ir in self.irreps_in:
            len_weight += mul
        
        self.len_weight = len_weight
    
    def __call__(self, x: torch.Tensor, weight: torch.Tensor):
        # x should have shape [edge/node, channels]
        # weight should have shape [edge/node, self.len_weight]
        
        ix = 0
        iw = 0
        out = []
        for mul, ir in self.irreps_in:
            field = x[:, ix: ix + mul * ir.dim]
            field = field.reshape(-1, mul, ir.dim)
            field = field * weight[:, iw: iw + mul][:, :, None]
            field = field.reshape(-1, mul * ir.dim)
            
            ix += mul * ir.dim
            iw += mul
            out.append(field)
        
        return torch.cat(out, dim=-1)

class SkipConnection(nn.Module):
    def __init__(self, irreps_in, irreps_out, is_complex=False):
        super().__init__()
        irreps_in = Irreps(irreps_in)
        irreps_out = Irreps(irreps_out)
        self.sc = None
        if irreps_in == irreps_out:
            self.sc = None
        else:
            self.sc = Linear(irreps_in=irreps_in, irreps_out=irreps_out)
    
    def forward(self, old, new):
        if self.sc is not None:
            old = self.sc(old)
        
        return old + new

class SelfTp(nn.Module):
    def __init__(self, irreps_in, irreps_out, **kwargs):
        '''z_i = W'_{ij}x_j W''_{ik}x_k (k>=j)'''
        super().__init__()
        
        assert not kwargs.pop('internal_weights', False) # internal weights must be True
        assert kwargs.pop('shared_weights', True) # shared weights must be false
        
        irreps_in = Irreps(irreps_in)
        irreps_out = Irreps(irreps_out)
        
        instr_tp = []
        weights1, weights2 = [], []
        for i1, (mul1, ir1) in enumerate(irreps_in):
            for i2 in range(i1, len(irreps_in)):
                mul2, ir2 = irreps_in[i2]
                for i_out, (mul_out, ir3) in enumerate(irreps_out):
                    if ir3 in ir1 * ir2:
                        weights1.append(nn.Parameter(torch.randn(mul1, mul_out)))
                        weights2.append(nn.Parameter(torch.randn(mul2, mul_out)))
                        instr_tp.append((i1, i2, i_out, 'uvw', True, 1.0))
        
        self.tp = TensorProduct(irreps_in, irreps_in, irreps_out, instr_tp, internal_weights=False, shared_weights=True, **kwargs)
        
        self.weights1 = nn.ParameterList(weights1)
        self.weights2 = nn.ParameterList(weights2)
        
    def forward(self, x):
        weights = []
        for weight1, weight2 in zip(self.weights1, self.weights2):
            weight = weight1[:, None, :] * weight2[None, :, :]
            weights.append(weight.view(-1))
        weights = torch.cat(weights)
        return self.tp(x, x, weights)

class SeparateWeightTensorProduct(nn.Module):
    def __init__(self, irreps_in1, irreps_in2, irreps_out, **kwargs):
        '''z_i = W'_{ij}x_j W''_{ik}y_k'''
        super().__init__()
        
        assert not kwargs.pop('internal_weights', False) # internal weights must be True
        assert kwargs.pop('shared_weights', True) # shared weights must be false
        
        irreps_in1 = Irreps(irreps_in1)
        irreps_in2 = Irreps(irreps_in2)
        irreps_out = Irreps(irreps_out)
                
        instr_tp = []
        weights1, weights2 = [], []
        for i1, (mul1, ir1) in enumerate(irreps_in1):
            for i2, (mul2, ir2) in enumerate(irreps_in2):
                for i_out, (mul_out, ir3) in enumerate(irreps_out):
                    if ir3 in ir1 * ir2:
                        weights1.append(nn.Parameter(torch.randn(mul1, mul_out)))
                        weights2.append(nn.Parameter(torch.randn(mul2, mul_out)))
                        instr_tp.append((i1, i2, i_out, 'uvw', True, 1.0))
        
        self.tp = TensorProduct(irreps_in1, irreps_in2, irreps_out, instr_tp, internal_weights=False, shared_weights=True, **kwargs)
        
        self.weights1 = nn.ParameterList(weights1)
        self.weights2 = nn.ParameterList(weights2)
        
    def forward(self, x1, x2):
        weights = []
        for weight1, weight2 in zip(self.weights1, self.weights2):
            weight = weight1[:, None, :] * weight2[None, :, :]
            weights.append(weight.view(-1))
        weights = torch.cat(weights)
        return self.tp(x1, x2, weights)

class SphericalBasis(nn.Module):
    def __init__(self, target_irreps, rcutoff, eps=1e-7, dtype=torch.get_default_dtype()):
        super().__init__()
        
        target_irreps = Irreps(target_irreps)
        
        self.sh = SphericalHarmonics(
            irreps_out=target_irreps,
            normalize=True,
            normalization='component',
        )
        
        max_order = max(map(lambda x: x[1].l, target_irreps)) # maximum angular momentum l
        max_freq = max(map(lambda x: x[0], target_irreps)) # maximum multiplicity
        
        basis = bessel_basis(max_order + 1, max_freq)
        lambdify_torch = {
            # '+': torch.add,
            # '-': torch.sub,
            # '*': torch.mul,
            # '/': torch.div,
            # '**': torch.pow,
            'sin': torch.sin,
            'cos': torch.cos
        }
        x = sym.symbols('x')
        funcs = []
        for mul, ir in target_irreps:
            for freq in range(mul):
                funcs.append(sym.lambdify([x], basis[ir.l][freq], [lambdify_torch]))
                
        self.bessel_funcs = funcs
        self.multiplier = e3ElementWise(target_irreps)
        self.dtype = dtype
        self.cutoff = PolynomialCutoff(rcutoff, p=6)
        self.register_buffer('rcutoff', torch.Tensor([rcutoff]))
        self.irreps_out = target_irreps
        self.register_buffer('eps', torch.Tensor([eps]))
        
    def forward(self, length, direction):
        # direction should be in y, z, x order
        sh = self.sh(direction).type(self.dtype)
        sbf = torch.stack([f((length + self.eps) / self.rcutoff) for f in self.bessel_funcs], dim=-1)
        return self.multiplier(sh, sbf) * self.cutoff(length)[:, None]
