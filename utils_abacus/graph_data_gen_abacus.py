'''
Descripttion: The scripts used to generate the input file graph_data.npz for HamNet.
version: 0.1
Author: Yang Zhong & ChangWei Zhang
Date: 2022-11-24 19:07:54
Last Modified by:   Changwei Zhang 
Last Modified time: 2023-05-20 15:24:36 
'''

import numpy as np
from copy import deepcopy
import os
import sys
from torch_geometric.data import Data
import torch
from tqdm import tqdm
import re
import multiprocessing
from read_abacus import STRU, ABACUSHS
from pymatgen.core.periodic_table import Element
from utils import *

################################ Input parameters begin ####################
nao_max = 27
graph_data_folder = '../graph/'
max_SCF_skip = 200 # default is 200
scfout_paths = ["../Si_abacus/{:03d}/OUT.Si".format(i) for i in range(1,501)]
dat_file_name = ["../Si_abacus/{:03d}/STRU".format(i) for i in range(1,501)] # STRU 
std_file_name = "running_scf.log"
soc = False
nproc = 1
################################ Input parameters end ######################

if nao_max == 13:
    basis_def = basis_def_13_abacus
elif nao_max == 15:
    basis_def = basis_def_15_abacus
elif nao_max == 27:
    basis_def = basis_def_27_abacus
elif nao_max == 40:
    basis_def = basis_def_40_abacus
else:
    raise NotImplementedError

graphs = dict()
if not os.path.exists(graph_data_folder):
    os.makedirs(graph_data_folder)

if len(scfout_paths) != len(dat_file_name):
    raise IndexError('The number of scfout_paths and dat_file not match.')

def generate_graph_data(idx:int, scf_path:str):
    # file paths
    f_std = os.path.join(scf_path, std_file_name)
    f_dat = dat_file_name[idx]
    
    # read energy
    try:
        with open(f_std, 'r') as f:
            content = f.read()
            Enpy = float(pattern_eng_abacus.findall((content).strip())[0])
            max_SCF = int(pattern_md_abacus.findall((content).strip())[-1])
    except:
        print('Error: scf.log. This may result from unconvergent calcs. Continue...')
        return False, None
    
    # check if the calculation is converged
    if max_SCF >= max_SCF_skip:
        print('Error: max_SCF. Continue...')
        return False, None  
    
    # Read crystal parameters
    try:
        crystal = STRU(f_dat)
        latt = crystal.cell
        z = []
        for spec, na in zip(crystal.species, crystal.na_s):
            z += [Element(spec).Z] * na
        z = atomic_numbers = np.array(z, dtype=int)
    except:
        print('Error: STRU. Continue...')
        return False, None
    
    # read hopping parameters
    try:
        fH0 = ABACUSHS(os.path.join(scf_path, 'data-H0R-sparse_SPIN0.csr'))
        graphH0 = fH0.getGraph(crystal, graph={}, isH=True, isSOC=soc)
        fH = ABACUSHS(os.path.join(scf_path, 'data-HR-sparse_SPIN0.csr'))
        graphH = fH.getGraph(crystal, graph=graphH0, isH=True, calcRcut=True, isSOC=soc, skip=True)
        fS = ABACUSHS(os.path.join(scf_path, 'data-SR-sparse_SPIN0.csr'))
        graphS = fS.getGraph(crystal, graph=graphH, skip=True, isSOC=soc)
        fH0.close()
        fH.close()
        fS.close()
    
        pos = graphH['pos']
        edge_index = graphH['edge_index']
        inv_edge_idx = graphH['inv_edge_idx']
        #
        Hon = graphH['Hon']
        Hoff = graphH['Hoff']
        Son = graphS['Hon'][0]
        Soff = graphS['Hoff'][0]
        nbr_shift = graphH['nbr_shift']
        cell_shift = graphH['cell_shift']
        
        # Find inverse edge_index
        if len(inv_edge_idx) != len(edge_index[0]):
            print('Wrong info: len(inv_edge_idx) != len(edge_index[0]) !')
            sys.exit()

        #
        num_sub_matrix = pos.shape[0] + edge_index.shape[1]
        if not soc:
            H = np.zeros((num_sub_matrix, nao_max**2), dtype=np.float32)
        else:
            H = np.zeros((num_sub_matrix, (2*nao_max)**2), dtype=np.float32)
            iH= np.zeros((num_sub_matrix, (2*nao_max)**2), dtype=np.float32)
        S = np.zeros((num_sub_matrix, nao_max**2), dtype=np.float32)
        
        # on-site
        for i in range(len(z)):
            src = z[i]
            mask = np.zeros((nao_max, nao_max), dtype=int)
            mask[basis_def[src][:,None], basis_def[src][None,:]] = 1
            mask = (mask > 0)
            if not soc:
                H[i][mask.flatten()] = Hon[0][i]
            else:
                tH = np.zeros((2*nao_max, 2*nao_max), dtype=np.complex64)
                tH[:nao_max,:nao_max][mask] = Hon[0][i] # uu
                tH[:nao_max,nao_max:][mask] = Hon[1][i] # ud
                tH[nao_max:,:nao_max][mask] = Hon[2][i] # du
                tH[nao_max:,nao_max:][mask] = Hon[3][i] # dd
                H[i] = tH.real.flatten()
                iH[i]= tH.imag.flatten()
            S[i][mask.flatten()] = Son[i].real
        
        # off-site
        num = 0
        for i in range(len(edge_index[0])):
            src, tar = z[edge_index[0,num]], z[edge_index[1,num]]
            mask = np.zeros((nao_max, nao_max), dtype=int)
            mask[basis_def[src][:,None], basis_def[tar][None,:]] = 1
            mask = (mask > 0)
            if not soc:
                H[num + len(z)][mask.flatten()] = Hoff[0][i]
            else:
                tH = np.zeros((2*nao_max, 2*nao_max), dtype=np.complex64)
                tH[:nao_max,:nao_max][mask] = Hoff[0][i] # uu
                tH[:nao_max,nao_max:][mask] = Hoff[1][i] # ud
                tH[nao_max:,:nao_max][mask] = Hoff[2][i] # du
                tH[nao_max:,nao_max:][mask] = Hoff[3][i] # dd
                H[num + len(z)] = tH.real.flatten()
                iH[num + len(z)]= tH.imag.flatten()
            S[num + len(z)][mask.flatten()] = Soff[i].real
            num = num + 1
    except:
        print('Error: H and S. Continue...')
        return False, None
    
    # read H0
    try:    
        Hon0 = graphH0['Hon']
        Hoff0 = graphH0['Hoff']
        #
        num_sub_matrix = pos.shape[0] + edge_index.shape[1]
        if not soc:
            H0 = np.zeros((num_sub_matrix, nao_max**2), dtype=np.float32)
        else:
            H0 = np.zeros((num_sub_matrix, (2*nao_max)**2), dtype=np.float32)
            iH0= np.zeros((num_sub_matrix, (2*nao_max)**2), dtype=np.float32)

        # on-site
        for i in range(len(z)):
            src = z[i]
            mask = np.zeros((nao_max, nao_max), dtype=int)
            mask[basis_def[src][:,None], basis_def[src][None,:]] = 1
            mask = (mask > 0)
            if not soc:
                H0[i][mask.flatten()] = Hon0[0][i]
            else:
                tH = np.zeros((2*nao_max, 2*nao_max), dtype=np.complex64)
                tH[:nao_max,:nao_max][mask] = Hon0[0][i] # uu
                tH[:nao_max,nao_max:][mask] = Hon0[1][i] # ud
                tH[nao_max:,:nao_max][mask] = Hon0[2][i] # du
                tH[nao_max:,nao_max:][mask] = Hon0[3][i] # dd
                H0[i] = tH.real.flatten()
                iH0[i]= tH.imag.flatten()
        
        # off-site
        num = 0
        for i in range(len(edge_index[0])):
            src, tar = z[edge_index[0,num]], z[edge_index[1,num]]
            mask = np.zeros((nao_max, nao_max), dtype=int)
            mask[basis_def[src][:,None], basis_def[tar][None,:]] = 1
            mask = (mask > 0)
            if not soc:
                H0[num + len(z)][mask.flatten()] = Hoff0[0][i]
            else:
                tH = np.zeros((2*nao_max, 2*nao_max), dtype=np.complex64)
                tH[:nao_max,:nao_max][mask] = Hoff0[0][i] # uu
                tH[:nao_max,nao_max:][mask] = Hoff0[1][i] # ud
                tH[nao_max:,:nao_max][mask] = Hoff0[2][i] # du
                tH[nao_max:,nao_max:][mask] = Hoff0[3][i] # dd
                H0[num + len(z)] = tH.real.flatten()
                iH0[num + len(z)]= tH.imag.flatten()
            num = num + 1
    except:
        print('Error: H0. Continue...')
        return False, None
    
    # save in Data
    if not soc:
        return True, fH.max_rcut, Data(z=torch.LongTensor(z),
                        cell = torch.Tensor(latt[None,:,:]),
                        total_energy = torch.Tensor([Enpy]),
                        pos=torch.FloatTensor(pos),
                        node_counts=torch.LongTensor([len(z)]),
                        edge_index=torch.LongTensor(edge_index),
                        inv_edge_idx=torch.LongTensor(inv_edge_idx),
                        nbr_shift=torch.FloatTensor(nbr_shift),
                        cell_shift=torch.LongTensor(cell_shift),
                        hamiltonian=torch.FloatTensor(H),
                        overlap=torch.FloatTensor(S),
                        Hon = torch.FloatTensor(H[:pos.shape[0],:]),
                        Hoff = torch.FloatTensor(H[pos.shape[0]:,:]),
                        Hon0 = torch.FloatTensor(H0[:pos.shape[0],:]),
                        Hoff0 = torch.FloatTensor(H0[pos.shape[0]:,:]),
                        Son = torch.FloatTensor(S[:pos.shape[0],:]),
                        Soff = torch.FloatTensor(S[pos.shape[0]:,:]))
    else:
        return True, fH.max_rcut, Data(z=torch.LongTensor(z),
                        cell = torch.Tensor(latt[None,:,:]),
                        total_energy = torch.Tensor([Enpy]),
                        pos=torch.FloatTensor(pos),
                        node_counts=torch.LongTensor([len(z)]),
                        edge_index=torch.LongTensor(edge_index),
                        inv_edge_idx=torch.LongTensor(inv_edge_idx),
                        nbr_shift=torch.FloatTensor(nbr_shift),
                        cell_shift=torch.LongTensor(cell_shift),
                        overlap=torch.FloatTensor(S),
                        Hon = torch.FloatTensor(H[:pos.shape[0],:]),
                        Hoff = torch.FloatTensor(H[pos.shape[0]:,:]),
                        iHon = torch.FloatTensor(iH[:pos.shape[0],:]),
                        iHoff = torch.FloatTensor(iH[pos.shape[0]:,:]),
                        Hon0 = torch.FloatTensor(H0[:pos.shape[0],:]),
                        Hoff0 = torch.FloatTensor(H0[pos.shape[0]:,:]),
                        iHon0 = torch.FloatTensor(iH0[:pos.shape[0],:]),
                        iHoff0 = torch.FloatTensor(iH0[pos.shape[0]:,:]),
                        Son = torch.FloatTensor(S[:pos.shape[0],:]),
                        Soff = torch.FloatTensor(S[pos.shape[0]:,:]))
    
results = []
multiprocessing.freeze_support()
nproc = min(multiprocessing.cpu_count(), nproc)
pool = multiprocessing.Pool(processes=nproc)

crystal = STRU(dat_file_name[0])
max_rcut = np.zeros([crystal.nspecies, crystal.nspecies])
min_rcut = np.zeros_like(max_rcut)
for is_, ispec in enumerate(crystal.species):
    for js, jspec in enumerate(crystal.species):
        min_rcut[is_,js] = RCUT_dict[ispec] + RCUT_dict[jspec]

for idx, scf_path in enumerate(scfout_paths):
    results.append(
        pool.apply_async(generate_graph_data, (idx, scf_path))
    )
for idx, res in enumerate(tqdm(results)):
    tmp = res.get()
    success = tmp[0]
    if success:
        max_rcut = np.max([tmp[1], max_rcut], axis=0)
        graphs[idx] = tmp[2]
pool.close()
pool.join()

graph_data_path = os.path.join(graph_data_folder, 'graph_data.npz')
rcut_path = os.path.join(graph_data_folder, 'rcut.txt')
np.savez(graph_data_path, graph=graphs)
np.savetxt(rcut_path, np.max([min_rcut, max_rcut], axis=0))