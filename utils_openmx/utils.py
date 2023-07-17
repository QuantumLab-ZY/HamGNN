'''
Descripttion: 
version: 
Author: Yang Zhong
Date: 2023-01-16 13:00:43
LastEditors: Yang Zhong
LastEditTime: 2023-01-16 14:53:37
'''

from ase import Atoms
import numpy as np
from ctypes import Union
import numpy as np
from typing import Tuple, Union, Optional, List, Set, Dict, Any
from pymatgen.core.periodic_table import Element
from ase import Atoms
import re

def ase_atoms_to_openmxfile(atoms:Atoms, basic_commad:str, spin_set:dict, PAO_dict:dict, PBE_dict:dict, filename:str):
    chemical_symbols = atoms.get_chemical_symbols()
    species = set(chemical_symbols)
    positions = atoms.get_array(name='positions')
    cell = atoms.get_cell().array
    openmx = basic_commad
    openmx += "#\n# Definition of Atomic Species\n#\n"
    openmx += f'Species.Number       {len(species)}\n'
    openmx += '<Definition.of.Atomic.Species\n'
    for s in species:
        openmx += f"{s}   {PAO_dict[s]}       {PBE_dict[s]}\n"    
    openmx += "Definition.of.Atomic.Species>\n\n"
    openmx += "#\n# Atoms\n#\n"
    openmx += "Atoms.Number%12d" % len(chemical_symbols)
    openmx += "\nAtoms.SpeciesAndCoordinates.Unit   Ang # Ang|AU"
    openmx += "\n<Atoms.SpeciesAndCoordinates           # Unit=Ang."
    for num, sym in enumerate(chemical_symbols):
        openmx += "\n%3d  %s  %10.7f  %10.7f  %10.7f   %.2f   %.2f" % (num+1, sym, *positions[num], *spin_set[chemical_symbols[num]])
    openmx += "\nAtoms.SpeciesAndCoordinates>"
    openmx += "\nAtoms.UnitVectors.Unit             Ang #  Ang|AU"
    openmx += "\n<Atoms.UnitVectors                     # unit=Ang."
    openmx += "\n      %10.7f  %10.7f  %10.7f\n      %10.7f  %10.7f  %10.7f\n      %10.7f  %10.7f  %10.7f" % (*cell[0], *cell[1], *cell[2])
    openmx += "\nAtoms.UnitVectors>"
    with open(filename,'w') as wf:
        wf.write(openmx)

# Warning: this dict is not complete!!!
spin_set = {'H':[0.5, 0.5],
            'He':[1.0,1.0],
            'Li':[1.5,1.5],
            'Be':[1.0,1.0],
            'B':[1.5,1.5], 
            'C':[2.0, 2.0], 
            'N': [2.5,2.5], 
            'O':[3.0,3.0], 
            'F':[3.5,3.5],
            'Ne':[4.0,4.0],
            'Na':[4.5,4.5],
            'Mg':[4.0,4.0],
            'Al':[1.5,1.5],
            'Si':[2.0,2.0],
            'P':[2.5,2.5],
            'S':[3.0,3.0],
            'Cl':[3.5,3.5],
            'Ar':[4.0,4.0],
            'K':[4.5,4.5],
            'Ca':[5.0,5.0],
            'Si': [2.0, 2.0], 
            'Mo': [7.0, 7.0], 
            'S': [3.0, 3.0],
            'Bi':[7.5, 7.5],
            'Se':[3.0, 3.0],
            'Cr':[9.0, 5.0],
            'I':[3.5, 3.5],
            'As':[7.5, 7.5],
            'Ga':[6.5, 6.5],
            'Mo': [7.0, 7.0],
            'Cs':[4.5, 4.5],
            'Pb':[7.0, 7.0],
            'Te':[8.0, 8.0]
            }

# Warning: this dict is not complete!!!
PAO_dict = {'H':'H6.0-s2p1',
            'He':'He8.0-s2p1',
            'Li':'Li8.0-s3p2',
            'Be':'Be7.0-s2p2',
            'B':'B7.0-s2p2d1',
            'C':'C6.0-s2p2d1',
            'N':'N6.0-s2p2d1',
            'O':'O6.0-s2p2d1',
            'F':'F6.0-s2p2d1',
            'Ne':'Ne9.0-s2p2d1',
            'Na':'Na9.0-s3p2d1',
            'Mg':'Mg9.0-s3p2d1',
            'Al':'Al7.0-s2p2d1',
            'Si':'Si7.0-s2p2d1',
            'P':'P7.0-s2p2d1',
            'S':'S7.0-s2p2d1',
            'Cl':'Cl7.0-s2p2d1',
            'Ar':'Ar9.0-s2p2d1',
            'K':'K10.0-s3p2d1',
            'Ca':'Ca9.0-s3p2d1',
            'Bi':'Bi8.0-s3p2d2',
            'Se':'Se7.0-s3p2d2',
            'Cr':'Cr6.0-s3p2d1',
            'I':'I7.0-s3p2d2',
            'As':'As7.0-s3p2d2',
            'Ga':'Ga7.0-s3p2d2',
            'Mo':'Mo7.0-s3p2d2',
            'Cs':'Cs12.0-s3p2d2',
            'Pb':'Pb8.0-s3p2d2',
            'Te': 'Te7.0-s3p2d2'
            }

# Warning: this dict is not complete!!!
PBE_dict = {'H':'H_PBE19',
            'He':'He_PBE19',
            'Li':'Li_PBE19',
            'Be':'Be_PBE19',
            'B':'B_PBE19',
            'C':'C_PBE19',
            'N':'N_PBE19',
            'O':'O_PBE19',
            'F':'F_PBE19',
            'Ne':'Ne_PBE19',
            'Na':'Na_PBE19',
            'Mg':'Mg_PBE19',
            'Al':'Al_PBE19',
            'Si':'Si_PBE19',
            'P':'P_PBE19',
            'S':'S_PBE19',
            'Cl':'Cl_PBE19',
            'Ar':'Ar_PBE19',
            'K':'K_PBE19',
            'Ca':'Ca_PBE19',
            'Bi':'Bi_PBE19',
            'Se':'Se_PBE19',
            'Cr':'Cr_PBE19',
            'I':'I_PBE19',
            'As':'As_PBE19',
            'Ga':'Ga_PBE19',
            'Mo':'Mo_PBE19',
            'Cs':'Cs_PBE19',
            'Pb':'Pb_PBE19',
            'Te':'Te_PBE19'}

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

# Warning: this dict is not complete!!!
# openmx
# s1, s2, s3, px1, py1, pz1, px2, py2, pz2, d3z^2-r^2, dx^2-y^2, dxy, dxz, dyz
# siesta
# .........., py1, pz1, px1, ............., dxy, dyz, dz2, dxz, dx2-y2
#              4    5    3                  11   13    9   12    10
basis_def_19 = {1:np.array([0,1,3,4,5], dtype=int), # H
             2:np.array([0,1,3,4,5], dtype=int), # He
             3:np.array([0,1,2,3,4,5,6,7,8], dtype=int), # Li
             4:np.array([0,1,3,4,5,6,7,8], dtype=int), # Be
             5:np.array([0,1,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # B
             6:np.array([0,1,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # C
             7:np.array([0,1,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # N
             8:np.array([0,1,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # O
             9:np.array([0,1,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # F
             10:np.array([0,1,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # Ne
             11:np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # Na
             12:np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # Mg
             13:np.array([0,1,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # Al
             14:np.array([0,1,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # Si
             15:np.array([0,1,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # p
             16:np.array([0,1,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # S
             17:np.array([0,1,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # Cl
             18:np.array([0,1,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # Ar
             19:np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # K
             20:np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # Ca 
             42:np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], dtype=int), # Mo  
             83:np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], dtype=int), # Bi  
             34:np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], dtype=int), # Se 
             24:np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # Cr 
             53:np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], dtype=int), # I   
             82:np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], dtype=int), # pb
             55:np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], dtype=int) # Cs
             }

basis_def_19_siesta = {55:np.array([0,1,4,5,3], dtype=int), # Cs
             51:np.array([0,1,4,5,3,7,8,6,11,13,9,12,10], dtype=int), # Sb
             47:np.array([0,1,11,13,9,12,10,16,18,14,17,15,4,5,3], dtype=int), # Ag
             17:np.array([0,1,4,5,3,7,8,6,11,13,9,12,10], dtype=int), # Cl
             14:np.array([0,1,4,5,3,7,8,6,11,13,9,12,10], dtype=int) # Si
             }

# Warning: this dict is not complete!!!
basis_def_14 = {1:np.array([0,1,3,4,5], dtype=int), # H
             2:np.array([0,1,3,4,5], dtype=int), # He
             3:np.array([0,1,2,3,4,5,6,7,8], dtype=int), # Li
             4:np.array([0,1,3,4,5,6,7,8], dtype=int), # Be
             5:np.array([0,1,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # B
             6:np.array([0,1,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # C
             7:np.array([0,1,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # N
             8:np.array([0,1,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # O
             9:np.array([0,1,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # F
             10:np.array([0,1,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # Ne
             11:np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # Na
             12:np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # Mg
             13:np.array([0,1,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # Al
             14:np.array([0,1,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # Si
             15:np.array([0,1,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # p
             16:np.array([0,1,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # S
             17:np.array([0,1,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # Cl
             18:np.array([0,1,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # Ar
             19:np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # K
             20:np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13], dtype=int), # Ca 
             }

# Warning: this dict is not complete!!!
num_valence = {1:1,2:2,3:3,4:2,5:3,6:4,7:5,8:6,9:7,10:8,11:9,12:8,13:3,14:4,15:5,16:6,17:7,18:8,19:9,20:10,42:14,83:15,34:6,
               35:7,55:9,82:14,53:7}
num_val = np.zeros((99,), dtype=int)
for k in num_valence.keys():
    num_val[k] = num_valence[k]

au2ang = 0.5291772490000065
au2ev = 27.211324570273
pattern_eng = re.compile(r'Enpy  =(\W+)(\-\d+\.?\d*)')
pattern_md = re.compile(r'MD= 1  SCF=(\W*)(\d+)')
pattern_latt = re.compile(r'<Atoms.UnitVectors.+\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+Atoms.UnitVectors>')
pattern_coor = re.compile(r'\s+\d+\s+(\w+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+\-?\d+\.?\d+\s+\-?\d+\.?\d+')
num = r'-?\d+\.?\d*'
wht = r'\s+'
pattern_eng_siesta = re.compile(r'siesta: Etot\s+=\s+(\-\d+\.?\d*)')
pattern_md_siesta = re.compile(r'scf:\s+(\d+)')
pattern_latt_siesta = re.compile(r'%block LatticeVectors.*' + f'{wht}({num}){wht}({num}){wht}({num}){wht}({num}){wht}({num}){wht}({num}){wht}({num}){wht}({num}){wht}({num})' + r'\s+%endblock LatticeVectors')
pattern_cblk_siesta = re.compile(r'%block AtomicCoordinatesAndAtomicSpecies(.+)%endblock AtomicCoordinatesAndAtomicSpecies', flags=re.S)
pattern_coor_siesta = re.compile(f'{wht}({num}){wht}({num}){wht}({num}){wht}(\d+)')
pattern_sblk_siesta = re.compile(r'%block ChemicalSpeciesLabel(.+)%endblock ChemicalSpeciesLabel', flags=re.S)
pattern_spec_siesta = re.compile(r'\s+(\d+)\s+(\d+)\s+(\w+)')

# default values
max_SCF_skip = 200
device = 'cpu'

def read_openmx_dat(filename: str = None):

    with open(filename,'r') as f:
        content = f.read()
        speciesAndCoordinates = pattern_coor.findall((content).strip())
        latt = pattern_latt.findall((content).strip())[0]
        latt = np.array([float(var) for var in latt]).reshape(-1, 3)/au2ang    
        species = []
        coordinates = []
        for item in speciesAndCoordinates:
            species.append(item[0])
            coordinates += item[1:]
        atomic_numbers = np.array([Element[s].Z for s in species])
        coordinates = np.array([float(pos) for pos in coordinates]).reshape(-1, 3)/au2ang
    
    return atomic_numbers, latt, coordinates
    