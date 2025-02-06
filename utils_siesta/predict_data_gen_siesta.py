'''
Descripttion: The scripts used to generate the input file graph_data.npz for HamNet.
version: 0.1
Author: Yang Zhong & Changwei Zhang
Date: 2023-05-18 17:08:30 
Last Modified by:   Changwei Zhang 
Last Modified time: 2023-05-18 17:08:30 
'''

import numpy as np
import os
import sys
from torch_geometric.data import Data
import torch
from tqdm import tqdm
import multiprocessing
from pymatgen.core.periodic_table import Element
from read_siesta import FDF, HSX
from utils import *

################################ Input parameters begin ####################
nao_max = 13
graph_data_path = '../example/graph/'
hsxdump_path = './hsxdump'
max_SCF_skip = 95 # default is 200
scfout_paths = ["../example/traj/{:d}".format(i) for i in range(0,1000)]
dat_file_name = "cell.fdf"
fix_edge_index = False
nproc = 8
################################ Input parameters end ######################

if nao_max == 13:
    basis_def = basis_def_13_siesta
elif nao_max == 19:
    basis_def = basis_def_19_siesta
else:
    raise NotImplementedError

graphs = dict()
if not os.path.exists(graph_data_path):
    os.makedirs(graph_data_path)

def generate_graph_data(idx:int, scf_path:str):
    # file paths
    f_dat = os.path.join(scf_path, dat_file_name)
    f_H0 = os.path.join(scf_path, 'overlap.HSX')
    
    try:
        assert os.path.isfile(os.path.join(scf_path, 'overlap.HSX'))
    except AssertionError:
        print('Error: file missing. Continue...')
        return False, None
    
    # Read crystal parameters
    try:
        crystal = FDF(f_dat)
        latt = crystal.cell
        z = atomic_numbers = crystal.z
    except:
        print('Error: STRU. Continue...')
        return False, None
    
    # read hopping parameters
    os.system(f'{hsxdump_path} {f_H0} {idx} > /dev/null')
    hsx = HSX('HSX' + str(idx))
    graphH = hsx.getGraph2(crystal, graph={}) # siesta H0 should be stored in the exact same order!
    os.system('rm HSX' + str(idx))
    try:
        pos = graphH['pos']
        edge_index = graphH['edge_index']
        inv_edge_idx = graphH['inv_edge_idx']
        #
        Hon = graphH['Hon'][0]
        Hoff = graphH['Hoff'][0]
        Son = graphH['Son']
        Soff = graphH['Soff']
        nbr_shift = graphH['nbr_shift']
        cell_shift = graphH['cell_shift']
        
        # Find inverse edge_index
        if len(inv_edge_idx) != len(edge_index[0]):
            print('Wrong info: len(inv_edge_idx) != len(edge_index[0]) !')
            sys.exit()

        #
        num_sub_matrix = pos.shape[0] + edge_index.shape[1]
        H = np.zeros((num_sub_matrix, nao_max**2))
        S = np.zeros((num_sub_matrix, nao_max**2))
        
        for i, (sub_maxtrix_H, sub_maxtrix_S) in enumerate(zip(Hon, Son)):
            tH = np.zeros((nao_max, nao_max))
            tS = np.zeros((nao_max, nao_max))
            src = z[i]
            nao = len(basis_def[src])
            tH[basis_def[src][:,None], basis_def[src][None,:]] = sub_maxtrix_H.reshape(nao, nao)
            tS[basis_def[src][:,None], basis_def[src][None,:]] = sub_maxtrix_S.reshape(nao, nao)
            H[i] = tH.flatten()
            S[i] = tS.flatten()
        
        num = 0
        for i, (sub_maxtrix_H, sub_maxtrix_S) in enumerate(zip(Hoff, Soff)):
            tH = np.zeros((nao_max, nao_max))
            tS = np.zeros((nao_max, nao_max))
            src, tar = z[edge_index[0,num]], z[edge_index[1,num]]
            nao_i = len(basis_def[src])
            nao_j = len(basis_def[tar])
            tH[basis_def[src][:,None], basis_def[tar][None,:]] = sub_maxtrix_H.reshape(nao_i, nao_j)
            tS[basis_def[src][:,None], basis_def[tar][None,:]] = sub_maxtrix_S.reshape(nao_i, nao_j)
            H[num + len(z)] = tH.flatten()
            S[num + len(z)] = tS.flatten()
            num = num + 1
    except:
        print('Error: H and S. Continue...')
        return False, None
    
    # save in Data
    return True, Data(z=torch.LongTensor(z),
                        cell = torch.Tensor(latt[None,:,:]),
                        # total_energy=torch.Tensor([Enpy]),
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
                        Hon0 = torch.FloatTensor(H[:pos.shape[0],:]),
                        Hoff0 = torch.FloatTensor(H[pos.shape[0]:,:]),
                        Son = torch.FloatTensor(S[:pos.shape[0],:]),
                        Soff = torch.FloatTensor(S[pos.shape[0]:,:]))

results = []
multiprocessing.freeze_support()
nproc = min(multiprocessing.cpu_count(), nproc)
pool = multiprocessing.Pool(processes=nproc)
for idx, scf_path in enumerate(scfout_paths):
    results.append(
        pool.apply_async(generate_graph_data, (idx, scf_path))
    )
for idx, res in enumerate(tqdm(results)):
    success, graph = res.get()
    if idx == 0:
        assert success
    if success:
        if fix_edge_index:
            if idx == 0:
                graph_fixed = graph
                graphs[idx] = graph
            else:
                graph.edge_index = graph_fixed.edge_index
                graph.inv_edge_idx = graph_fixed.inv_edge_idx
                graph.nbr_shift = graph_fixed.nbr_shift
                graph.cell_shift = graph_fixed.cell_shift
                graph.hamiltonian = graph_fixed.hamiltonian
                graph.Hon = graph_fixed.Hon
                graph.Hoff = graph_fixed.Hoff
                graphs[idx] = graph
        else:
            graphs[idx] = graph
pool.close()
pool.join()

graph_data_path = os.path.join(graph_data_path, 'graph_data.npz')
np.savez(graph_data_path, graph=graphs)
