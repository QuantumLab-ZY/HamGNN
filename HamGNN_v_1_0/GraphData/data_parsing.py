"""
/*
 * @Author: Yang Zhong 
 * @Date: 2021-10-07 20:30:29 
 * @Last Modified by: Yang Zhong
 * @Last Modified time: 2021-10-29 15:52:53
 */
"""
import numpy as np
import torch
from tqdm import tqdm
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from torch_geometric.data import Data
import os
import csv
import glob, json


def cal_shfit_vec(image:np.array = None, lattice:np.array = None):
    """
    This function is used to compute the periodic shift vector of the end nodes of the edges.
    image: np.array, shape: (3,)
    lattice: np.array, shape: (3,3)
    """
    return np.sum(image[:, None]*lattice, axis=0)

def build_config(config):
    """
    Input all crystal files, Then build one-hot encoding and save as a json file.
    """
    crystal_path = config.dataset_params.csv_params.crystal_path
    file_type = config.dataset_params.csv_params.file_type
    if file_type.lower() == 'poscar':
        file_extension = '.vasp'
    elif file_type.lower() == 'cif':
        file_extension = '.cif'
    else:
        print(f'The file type: {file_type} has not been supported yet!')
    config_path = config.dataset_params.graph_data_path
    config_path = os.path.join(config_path, 'config_onehot.json')
    atoms=[]
    all_files = sorted(glob.glob(os.path.join(crystal_path,'*'+file_extension)))
    for path in tqdm(all_files):
        crystal = Structure.from_file(path)
        atoms += list(crystal.atomic_numbers)
    unique_z = np.unique(atoms)
    num_z = len(unique_z)
    print('unique_z:', num_z)
    print('min z:', np.min(unique_z))
    print('max z:', np.max(unique_z))
    # Configuration file
    config = dict()
    config["atomic_numbers"] = unique_z.tolist()
    config["node_vectors"] = np.eye(num_z,num_z).tolist() # One-hot encoding
    with open(config_path, 'w') as f:
        json.dump(config, f)
    return config

def get_init_atomfea(config:dict=None, crystal:Structure=None):
    atoms=crystal.atomic_numbers
    atomnum=config['atomic_numbers']
    z_dict = {z:i for i, z in enumerate(atomnum)}
    one_hotvec = np.array(config["node_vectors"])
    atom_fea = np.vstack([one_hotvec[z_dict[atoms[i]]] for i in range(len(crystal))])
    return atom_fea

def cif_parse(config):
    crystal_path = config.dataset_params.csv_params.crystal_path
    id_prop_path = config.dataset_params.csv_params.id_prop_path
    graph_data_path = config.dataset_params.graph_data_path
    radius = config.dataset_params.radius
    max_num_nbr = config.dataset_params.max_num_nbr
    l_pred_atomwise_tensor = config.setup.csv_params.l_pred_atomwise_tensor
    l_pred_crystal_tensor = config.setup.csv_params.l_pred_crystal_tensor
    rank_tensor = config.dataset_params.csv_params.rank_tensor

    file_type = config.dataset_params.csv_params.file_type
    if file_type.lower() == 'poscar':
        file_extension = '.vasp'
    elif file_type.lower() == 'cif':
        file_extension = '.cif'
    else:
        print(f'The file type: {file_type} has not been supported yet!')

    # build one hot vectors. (This will be deprecated in the future!)
    config_onehot_file_path = os.path.join(graph_data_path, "config_onehot.json")
    if os.path.exists(config_onehot_file_path):
        config_onehot = json.load(open(config_onehot_file_path))
    else:
        config_onehot = build_config(config)

    assert os.path.exists(id_prop_path), 'id_prop_path does not exist!'
    id_prop_file = os.path.join(id_prop_path, 'id_prop.csv')
    assert os.path.exists(id_prop_file), 'id_prop.csv does not exist!'

    with open(id_prop_file) as f:
        reader = csv.reader(f)
        id_prop_data = [row for row in reader]

    cif_ids = [id_prop[0] for id_prop in id_prop_data]
    # Parse the data in the id_prop.csv file
    if l_pred_atomwise_tensor or l_pred_crystal_tensor:
        targets = []
        length_tensor = 3**rank_tensor
        for idx in range(len(id_prop_data)):
            target_idx = list(map(lambda x: float(x), id_prop_data[idx][1:]))
            target_idx = np.array(target_idx).reshape(-1, length_tensor) # (N_atom, length)
            targets.append(target_idx) 

    else:
        targets = [float(id_prop[1]) for id_prop in id_prop_data]

    pbar_cif_ids = tqdm(cif_ids)
    graphs = dict()
    for i, cif_id in enumerate(pbar_cif_ids):
        #pbar_cif_ids.set_description("Processing %s" % cif_id)
        cif_path = os.path.join(crystal_path, cif_id+file_extension)
        crystal = Structure.from_file(cif_path)

        node_attr = get_init_atomfea(config_onehot, crystal)

        all_nbrs = crystal.get_all_neighbors(radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]

        edge_src_index, edge_tar_index, nbr_shift, nbr_counts = [], [], [], []
        for ni, nbr in enumerate(all_nbrs):
            if len(nbr) < max_num_nbr:
                nbr_counts.append(len(nbr))
                edge_src_index += [ni]*len(nbr)
                edge_tar_index += list(map(lambda x: x[2], nbr))
                nbr_shift += list(map(lambda x: cal_shfit_vec(
                    np.array(x[3], dtype=np.float32), x.lattice.matrix), nbr))
            else:
                nbr_counts.append(max_num_nbr)
                edge_src_index += [ni]*max_num_nbr
                edge_tar_index += list(map(lambda x: x[2], nbr[:max_num_nbr]))
                nbr_shift += list(map(lambda x: cal_shfit_vec(
                    np.array(x[3], dtype=np.float32), x.lattice.matrix), nbr[:max_num_nbr]))
        
        edge_index = [edge_src_index, edge_tar_index]  # 2*nedges
            
        if l_pred_atomwise_tensor or l_pred_crystal_tensor:
            y_label = targets[i]
        else:
            y_label = [targets[i]]

        graphs[cif_id] = Data(z=torch.LongTensor(crystal.atomic_numbers),
                              node_attr=torch.FloatTensor(node_attr),
                              y=torch.FloatTensor(y_label),
                              pos=torch.FloatTensor(crystal.cart_coords),
                              node_counts=torch.LongTensor([len(crystal)]),
                              nbr_counts=torch.LongTensor(nbr_counts),
                              edge_index=torch.LongTensor(edge_index),
                              nbr_shift=torch.FloatTensor(nbr_shift))

    graph_data_path = os.path.join(graph_data_path, 'graph_data.npz')
    np.savez(graph_data_path, graph=graphs)

if __name__ == '__main__':
    from input.config_parsing import read_config

    config = read_config(config_file_name='config.yaml')
    cif_parse(config)
