'''
Descripttion: 
version: 
Author: Yang Zhong & ChangWei Zhang
Date: 2023-01-16 13:00:43
Last Modified by:   Yang Zhong
Last Modified time: 2023-05-19 10:34:01 
'''

from ase import Atoms
import numpy as np
from ctypes import Union
import numpy as np
from typing import Tuple, Union, Optional, List, Set, Dict, Any
from pymatgen.core.periodic_table import Element
import re

def _nice_float(x,just,rnd):
    return str(round(x,rnd)).rjust(just)

class kpoints_generator:
    """
    Used to generate K point path
    """
    def __init__(self, dim_k: int=3, lat: Union[np.array, list]=None, per: Union[List, Tuple] = None):
        self._dim_k = dim_k
        self._lat = lat
        # choose which self._dim_k out of self._dim_r dimensions are
        # to be considered periodic.        
        if per==None:
            # by default first _dim_k dimensions are periodic
            self._per=list(range(self._dim_k))
        else:
            if len(per)!=self._dim_k:
                raise Exception("\n\nWrong choice of periodic/infinite direction!")
            # store which directions are the periodic ones
            self._per=per
        
    def k_path(self,kpts,nk,report=True):
    
        # processing of special cases for kpts
        if kpts=='full':
            # full Brillouin zone for 1D case
            k_list=np.array([[0.],[0.5],[1.]])
        elif kpts=='fullc':
            # centered full Brillouin zone for 1D case
            k_list=np.array([[-0.5],[0.],[0.5]])
        elif kpts=='half':
            # half Brillouin zone for 1D case
            k_list=np.array([[0.],[0.5]])
        else:
            k_list=np.array(kpts)
    
        # in 1D case if path is specified as a vector, convert it to an (n,1) array
        if len(k_list.shape)==1 and self._dim_k==1:
            k_list=np.array([k_list]).T

        # make sure that k-points in the path have correct dimension
        if k_list.shape[1]!=self._dim_k:
            print('input k-space dimension is',k_list.shape[1])
            print('k-space dimension taken from model is',self._dim_k)
            raise Exception("\n\nk-space dimensions do not match")

        # must have more k-points in the path than number of nodes
        if nk<k_list.shape[0]:
            raise Exception("\n\nMust have more points in the path than number of nodes.")

        # number of nodes
        n_nodes=k_list.shape[0]
    
        # extract the lattice vectors from the TB model
        lat_per=np.copy(self._lat)
        # choose only those that correspond to periodic directions
        lat_per=lat_per[self._per]    
        # compute k_space metric tensor
        k_metric = np.linalg.inv(np.dot(lat_per,lat_per.T))

        # Find distances between nodes and set k_node, which is
        # accumulated distance since the start of the path
        #  initialize array k_node
        k_node=np.zeros(n_nodes,dtype=float)
        for n in range(1,n_nodes):
            dk = k_list[n]-k_list[n-1]
            dklen = np.sqrt(np.dot(dk,np.dot(k_metric,dk)))
            k_node[n]=k_node[n-1]+dklen
    
        # Find indices of nodes in interpolated list
        node_index=[0]
        for n in range(1,n_nodes-1):
            frac=k_node[n]/k_node[-1]
            node_index.append(int(round(frac*(nk-1))))
        node_index.append(nk-1)
    
        # initialize two arrays temporarily with zeros
        #   array giving accumulated k-distance to each k-point
        k_dist=np.zeros(nk,dtype=float)
        #   array listing the interpolated k-points    
        k_vec=np.zeros((nk,self._dim_k),dtype=float)
    
        # go over all kpoints
        k_vec[0]=k_list[0]
        for n in range(1,n_nodes):
            n_i=node_index[n-1]
            n_f=node_index[n]
            kd_i=k_node[n-1]
            kd_f=k_node[n]
            k_i=k_list[n-1]
            k_f=k_list[n]
            for j in range(n_i,n_f+1):
                frac=float(j-n_i)/float(n_f-n_i)
                k_dist[j]=kd_i+frac*(kd_f-kd_i)
                k_vec[j]=k_i+frac*(k_f-k_i)
    
        if report==True:
            if self._dim_k==1:
                print(' Path in 1D BZ defined by nodes at '+str(k_list.flatten()))
            else:
                print('----- k_path report begin ----------')
                original=np.get_printoptions()
                np.set_printoptions(precision=5)
                print('real-space lattice vectors\n', lat_per)
                print('k-space metric tensor\n', k_metric)
                print('internal coordinates of nodes\n', k_list)
                if (lat_per.shape[0]==lat_per.shape[1]):
                    # lat_per is invertible
                    lat_per_inv=np.linalg.inv(lat_per).T
                    print('reciprocal-space lattice vectors\n', lat_per_inv)
                    # cartesian coordinates of nodes
                    kpts_cart=np.tensordot(k_list,lat_per_inv,axes=1)
                    print('cartesian coordinates of nodes\n',kpts_cart)
                print('list of segments:')
                for n in range(1,n_nodes):
                    dk=k_node[n]-k_node[n-1]
                    dk_str=_nice_float(dk,7,5)
                    print('  length = '+dk_str+'  from ',k_list[n-1],' to ',k_list[n])
                print('node distance list:', k_node)
                print('node index list:   ', np.array(node_index))
                np.set_printoptions(precision=original["precision"])
                print('----- k_path report end ------------')
            print()

        return (k_vec,k_dist,k_node,lat_per_inv, node_index)

RCUT_dict = { # unit: au
    'Ag':7,  'Cu':8,  'Mo':7,  'Sc':8,
    'Al':7,  'Fe':8,  'Na':8,  'Se':8,
    'Ar':7,  'F' :7,  'Nb':8,  'S' :7,
    'As':7,  'Ga':8,  'Ne':6,  'Si':7,
    'Au':7,  'Ge':8,  'N' :7,  'Sn':7,
    'Ba':10, 'He':6,  'Ni':8,  'Sr':9,
    'Be':7,  'Hf':7,  'O' :7,  'Ta':8,
    'B' :8,  'H' :6,  'Os':7,  'Tc':7,
    'Bi':7,  'Hg':9,  'Pb':7,  'Te':7,
    'Br':7,  'I' :7,  'Pd':7,  'Ti':8,
    'Ca':9,  'In':7,  'P' :7,  'Tl':7,
    'Cd':7,  'Ir':7,  'Pt':7,  'V' :8,
    'C' :7,  'K' :9,  'Rb':10, 'W' :8,
    'Cl':7,  'Kr':7,  'Re':7,  'Xe':8,
    'Co':8,  'Li':7,  'Rh':7,  'Y' :8,
    'Cr':8,  'Mg':8,  'Ru':7,  'Zn':8,
    'Cs':10, 'Mn':8,  'Sb':7,  'Zr':8
}

# Warning: this dict is not complete!!!
# openmx
# s1, s2, s3, px1, py1, pz1, px2, py2, pz2, d3z^2-r^2, dx^2-y^2, dxy, dxz, dyz
# siesta
# s1, s2,-py1, pz1,-px1,-py1, pz1,-px1, dxy,-dyz, dz2,-dxz, dx2-y2
# abacus
# see: https://github.com/abacusmodeling/abacus-develop/issues/267

basis_def_13_abacus = (lambda s1=[0],s2=[1],p1=[2,3,4],p2=[5,6,7],d1=[8,9,10,11,12]: {
    1 : np.array(s1+s2+p1, dtype=int), # H
    2 : np.array(s1+s2+p1, dtype=int), # He
    5 : np.array(s1+s2+p1+p2+d1, dtype=int), # B
    6 : np.array(s1+s2+p1+p2+d1, dtype=int), # C
    7 : np.array(s1+s2+p1+p2+d1, dtype=int), # N
    8 : np.array(s1+s2+p1+p2+d1, dtype=int), # O
    9 : np.array(s1+s2+p1+p2+d1, dtype=int), # F
    10: np.array(s1+s2+p1+p2+d1, dtype=int), # Ne
    14: np.array(s1+s2+p1+p2+d1, dtype=int), # Si
    15: np.array(s1+s2+p1+p2+d1, dtype=int), # P
    16: np.array(s1+s2+p1+p2+d1, dtype=int), # S
    17: np.array(s1+s2+p1+p2+d1, dtype=int), # Cl
    18: np.array(s1+s2+p1+p2+d1, dtype=int), # Ar
})()

basis_def_15_abacus = (lambda s1=[0],s2=[1],s3=[2],s4=[3],p1=[4,5,6],p2=[7,8,9],d1=[10,11,12,13,14]: {
    1 : np.array(s1+s2+p1, dtype=int), # H
    2 : np.array(s1+s2+p1, dtype=int), # He
    3 : np.array(s1+s2+s3+s4+p1, dtype=int), # Li
    4 : np.array(s1+s2+s3+s4+p1, dtype=int), # Bi
    5 : np.array(s1+s2+p1+p2+d1, dtype=int), # B
    6 : np.array(s1+s2+p1+p2+d1, dtype=int), # C
    7 : np.array(s1+s2+p1+p2+d1, dtype=int), # N
    8 : np.array(s1+s2+p1+p2+d1, dtype=int), # O
    9 : np.array(s1+s2+p1+p2+d1, dtype=int), # F
    10: np.array(s1+s2+p1+p2+d1, dtype=int), # Ne
    11: np.array(s1+s2+s3+s4+p1+p2+d1, dtype=int), # Na
    12: np.array(s1+s2+s3+s4+p1+p2+d1, dtype=int), # Mg
    # 13: Al
    14: np.array(s1+s2+p1+p2+d1, dtype=int), # Si
    15: np.array(s1+s2+p1+p2+d1, dtype=int), # P
    16: np.array(s1+s2+p1+p2+d1, dtype=int), # S
    17: np.array(s1+s2+p1+p2+d1, dtype=int), # Cl
    18: np.array(s1+s2+p1+p2+d1, dtype=int), # Ar
})()

# this dict is for abacus calculation.
basis_def_27_abacus = (lambda s1=[0],s2=[1],s3=[2],s4=[3],p1=[4,5,6],p2=[7,8,9],d1=[10,11,12,13,14],d2=[15,16,17,18,19],f1=[20,21,22,23,24,25,26]: {
    1 : np.array(s1+s2+p1, dtype=int), # H
    2 : np.array(s1+s2+p1, dtype=int), # He
    3 : np.array(s1+s2+s3+s4+p1, dtype=int), # Li
    4 : np.array(s1+s2+s3+s4+p1, dtype=int), # Bi
    5 : np.array(s1+s2+p1+p2+d1, dtype=int), # B
    6 : np.array(s1+s2+p1+p2+d1, dtype=int), # C
    7 : np.array(s1+s2+p1+p2+d1, dtype=int), # N
    8 : np.array(s1+s2+p1+p2+d1, dtype=int), # O
    9 : np.array(s1+s2+p1+p2+d1, dtype=int), # F
    10: np.array(s1+s2+p1+p2+d1, dtype=int), # Ne
    11: np.array(s1+s2+s3+s4+p1+p2+d1, dtype=int), # Na
    12: np.array(s1+s2+s3+s4+p1+p2+d1, dtype=int), # Mg
    # 13: Al
    14: np.array(s1+s2+p1+p2+d1, dtype=int), # Si
    15: np.array(s1+s2+p1+p2+d1, dtype=int), # P
    16: np.array(s1+s2+p1+p2+d1, dtype=int), # S
    17: np.array(s1+s2+p1+p2+d1, dtype=int), # Cl
    18: np.array(s1+s2+p1+p2+d1, dtype=int), # Ar
    19: np.array(s1+s2+s3+s4+p1+p2+d1, dtype=int), # K
    20: np.array(s1+s2+s3+s4+p1+p2+d1, dtype=int), # Ca
    21: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int), # Sc
    22: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int), # Ti
    23: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int), # V
    24: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int), # Cr
    25: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int), # Mn
    26: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int), # Fe
    27: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int), # Co
    28: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int), # Ni
    29: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int), # Cu
    30: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int), # Zn
    31: np.array(s1+s2+p1+p2+d1+d2+f1, dtype=int), # Ga
    32: np.array(s1+s2+p1+p2+d1+d2+f1, dtype=int), # Ge
    33: np.array(s1+s2+p1+p2+d1, dtype=int), # As
    34: np.array(s1+s2+p1+p2+d1, dtype=int), # Se
    35: np.array(s1+s2+p1+p2+d1, dtype=int), # Br
    36: np.array(s1+s2+p1+p2+d1, dtype=int), # Kr
    37: np.array(s1+s2+s3+s4+p1+p2+d1, dtype=int), # Rb
    38: np.array(s1+s2+s3+s4+p1+p2+d1, dtype=int), # Sr
    39: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int), # Y
    40: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int), # Zr
    41: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int), # Nb
    42: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int), # Mo
    43: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int), # Tc
    44: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int), # Ru
    45: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int), # Rh
    46: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int), # Pd
    47: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int), # Ag
    48: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int), # Cd
    49: np.array(s1+s2+p1+p2+d1+d2+f1, dtype=int), # In
    50: np.array(s1+s2+p1+p2+d1+d2+f1, dtype=int), # Sn
    51: np.array(s1+s2+p1+p2+d1+d2+f1, dtype=int), # Sb
    52: np.array(s1+s2+p1+p2+d1+d2+f1, dtype=int), # Te
    53: np.array(s1+s2+p1+p2+d1+d2+f1, dtype=int), # I
    54: np.array(s1+s2+p1+p2+d1+d2+f1, dtype=int), # Xe
    55: np.array(s1+s2+s3+s4+p1+p2+d1, dtype=int), # Cs
    56: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int), # Ba
    #
    79: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int), # Au
    80: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int), # Hg
    81: np.array(s1+s2+p1+p2+d1+d2+f1, dtype=int), # Tl
    82: np.array(s1+s2+p1+p2+d1+d2+f1, dtype=int), # Pb
    83: np.array(s1+s2+p1+p2+d1+d2+f1, dtype=int), # Bi
})()

basis_def_40_abacus = (lambda s1=[0],
                       s2=[1],
                       s3=[2],
                       s4=[3],
                       p1=[4,5,6],
                       p2=[7,8,9],
                       p3=[10,11,12],
                       p4=[13,14,15],
                       d1=[16,17,18,19,20],
                       d2=[21,22,23,24,25],
                       f1=[26,27,28,29,30,31,32],
                       f2=[33,34,35,36,37,38,39]: {
    Element('Ag').Z: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int),
    Element('Al').Z: np.array(s1+s2+s3+s4+p1+p2+p3+p4+d1, dtype=int),
    Element('Ar').Z: np.array(s1+s2+p1+p2+d1, dtype=int),
    Element('As').Z: np.array(s1+s2+p1+p2+d1, dtype=int),
    Element('Au').Z: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int),
    Element('Ba').Z: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int),
    Element('Be').Z: np.array(s1+s2+s3+s4+p1, dtype=int),
    Element('B').Z: np.array(s1+s2+p1+p2+d1, dtype=int),
    Element('Bi').Z: np.array(s1+s2+p1+p2+d1+d2+f1, dtype=int),
    Element('Br').Z: np.array(s1+s2+p1+p2+d1, dtype=int),
    Element('Ca').Z: np.array(s1+s2+s3+s4+p1+p2+d1, dtype=int),
    Element('Cd').Z: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int),
    Element('C').Z: np.array(s1+s2+p1+p2+d1, dtype=int),
    Element('Cl').Z: np.array(s1+s2+p1+p2+d1, dtype=int),
    Element('Co').Z: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int),
    Element('Cr').Z: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int),
    Element('Cs').Z: np.array(s1+s2+s3+s4+p1+p2+d1, dtype=int),
    Element('Cu').Z: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int),
    Element('Fe').Z: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int),
    Element('F').Z: np.array(s1+s2+p1+p2+d1, dtype=int),
    Element('Ga').Z: np.array(s1+s2+p1+p2+d1+d2+f1, dtype=int),
    Element('Ge').Z: np.array(s1+s2+p1+p2+d1+d2+f1, dtype=int),
    Element('He').Z: np.array(s1+s2+p1, dtype=int),
    Element('Hf').Z: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1+f2, dtype=int), # Hf_gga_10au_100Ry_4s2p2d2f.orb
    Element('H').Z: np.array(s1+s2+p1, dtype=int),
    Element('Hg').Z: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int),
    Element('I').Z: np.array(s1+s2+p1+p2+d1+d2+f1, dtype=int),
    Element('In').Z: np.array(s1+s2+p1+p2+d1+d2+f1, dtype=int),
    Element('Ir').Z: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int),
    Element('K').Z: np.array(s1+s2+s3+s4+p1+p2+d1, dtype=int),
    Element('Kr').Z: np.array(s1+s2+p1+p2+d1, dtype=int),
    Element('Li').Z: np.array(s1+s2+s3+s4+p1, dtype=int),
    Element('Mg').Z: np.array(s1+s2+s3+s4+p1+p2+d1, dtype=int),
    Element('Mn').Z: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int),
    Element('Mo').Z: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int),
    Element('Na').Z: np.array(s1+s2+s3+s4+p1+p2+d1, dtype=int),
    Element('Nb').Z: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int),
    Element('Ne').Z: np.array(s1+s2+p1+p2+d1, dtype=int),
    Element('N').Z: np.array(s1+s2+p1+p2+d1, dtype=int),
    Element('Ni').Z: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int),
    Element('O').Z: np.array(s1+s2+p1+p2+d1, dtype=int),
    Element('Os').Z: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int),
    Element('Pb').Z: np.array(s1+s2+p1+p2+d1+d2+f1, dtype=int),
    Element('Pd').Z: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int),
    Element('P').Z: np.array(s1+s2+p1+p2+d1, dtype=int),
    Element('Pt').Z: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int),
    Element('Rb').Z: np.array(s1+s2+s3+s4+p1+p2+d1, dtype=int),
    Element('Re').Z: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int),
    Element('Rh').Z: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int),
    Element('Ru').Z: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int),
    Element('Sb').Z: np.array(s1+s2+p1+p2+d1+d2+f1, dtype=int),
    Element('Sc').Z: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int),
    Element('Se').Z: np.array(s1+s2+p1+p2+d1, dtype=int),
    Element('S').Z: np.array(s1+s2+p1+p2+d1, dtype=int),
    Element('Si').Z: np.array(s1+s2+p1+p2+d1, dtype=int),
    Element('Sn').Z: np.array(s1+s2+p1+p2+d1+d2+f1, dtype=int),
    Element('Sr').Z: np.array(s1+s2+s3+s4+p1+p2+d1, dtype=int),
    Element('Ta').Z: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1+f2, dtype=int), # Ta_gga_10au_100Ry_4s2p2d2f.orb
    Element('Tc').Z: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int),
    Element('Te').Z: np.array(s1+s2+p1+p2+d1+d2+f1, dtype=int),
    Element('Ti').Z: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int),
    Element('Tl').Z: np.array(s1+s2+p1+p2+d1+d2+f1, dtype=int),
    Element('V').Z: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int),
    Element('W').Z: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1+f2, dtype=int), # W_gga_10au_100Ry_4s2p2d2f.orb
    Element('Xe').Z: np.array(s1+s2+p1+p2+d1+d2+f1, dtype=int),
    Element('Y').Z: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int),
    Element('Zn').Z: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int),
    Element('Zr').Z: np.array(s1+s2+s3+s4+p1+p2+d1+d2+f1, dtype=int)
    })()

# this dict is for abacus calculation.
num_valence = {
    'Ag':19, 'Al':11, 'Ar': 8, 'As': 5, 'Au':19, 'Ba':10, 'Be': 4, 'Bi':15, 'B' : 3, 'Br': 7, 
    'Ca':10, 'Cd':20, 'Cl': 7, 'C' : 4, 'Co':17, 'Cr':14, 'Cs': 9, 'Cu':19, 'Fe':16, 'F' : 7, 
    'Ga':13, 'Ge':14, 'He': 2, 'Hf':26, 'Hg':20, 'H' : 1, 'In':13, 'I' :17, 'Ir':17, 'K' : 9, 
    'Kr': 8, 'La':11, 'Li': 3, 'Mg':10, 'Mn':15, 'Mo':14, 'Na': 9, 'Nb':13, 'Ne': 8, 'Ni':18, 
    'N' : 5, 'O' : 6, 'Os':16, 'Pb':14, 'Pd':18, 'P' : 5, 'Pt':18, 'Rb': 9, 'Re':15, 'Rh':17, 
    'Ru':16, 'Sb':15, 'Sc':11, 'Se': 6, 'Si': 4, 'Sn':14, 'S' : 6, 'Sr':10, 'Ta':27, 'Tc':15, 
    'Te':16, 'Ti':12, 'Tl':13, 'V' :13, 'W' :28, 'Xe':18, 'Y' :11, 'Zn':20, 'Zr':12
}
num_val = np.zeros((99,), dtype=int)
for k in num_valence.keys():
    num_val[Element(k).Z] = num_valence[k]

au2ang = 0.5291772490000065
au2ev = 27.211324570273
pattern_eng = re.compile(r'Enpy  =(\W+)(\-\d+\.?\d*)')
pattern_md = re.compile(r'MD= 1  SCF=(\W*)(\d+)')
pattern_latt = re.compile(r'<Atoms.UnitVectors.+\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+Atoms.UnitVectors>')
pattern_coor = re.compile(r'\s+\d+\s+(\w+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+\-?\d+\.?\d+\s+\-?\d+\.?\d+')
pattern_eng_siesta = re.compile(r'siesta: Etot\s+=\s+(\-\d+\.?\d*)')
pattern_md_siesta = re.compile(r'scf:\s+(\d+)')
pattern_eng_abacus = re.compile(r'final etot is\s*(-?\d+\.?\d*)')
pattern_md_abacus = re.compile(r'ELEC=\s*(\d+)')

# default values
max_SCF_skip = 200
device = 'cpu'


