'''
Descripttion: 
version: 
Author: Yang Zhong
Date: 2023-06-28 20:36:48
LastEditors: Yang Zhong
LastEditTime: 2023-09-22 12:51:46
'''
import numpy as np
import os
import yaml
import argparse

au2ang = 0.5291772490000065


def write_kpoint(file, k_vec):
    file.write(np.asarray(k_vec, dtype=np.float64).tobytes())


def write_coefficients(file, wfn):
    coefficients = np.empty((len(wfn), 2), dtype=np.float64)
    coefficients[:, 0] = wfn.real
    coefficients[:, 1] = wfn.imag
    file.write(coefficients.tobytes())


def main():
    parser = argparse.ArgumentParser(description='Wavefunction export')
    parser.add_argument('--config', default='wfn_export.yaml', type=str, metavar='N')
    args = parser.parse_args()
    
    with open(args.config, encoding='utf-8') as rstream:
        input = yaml.load(rstream, yaml.SafeLoader)
    
    ##################### Input parameters ###################
    eig_vecs = np.load(input['eigen_vecs_path'])    
    latt = np.array(input['latt'])/au2ang
    
    save_dir = input['save_dir'] 
    soc_switch=input['soc_switch']
    integration = input['integration']
    ##########################################################
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if integration:
        wfn_min = input['wfn_min']
        wfn_max = input['wfn_max']
        k_vecs = np.load(input['k_vecs_path'])
        
        eig_vecs = eig_vecs.astype(np.complex128)
        
        lat_per_inv = np.linalg.inv(latt).T
        k_vecs = np.tensordot(k_vecs, lat_per_inv, axes=1)
        
        if soc_switch:
            filename = os.path.join(save_dir, 'wfn_up.bin')
            fw_u = open(filename, "wb")
        
            filename = os.path.join(save_dir, 'wfn_down.bin')
            fw_d = open(filename, "wb")
            
            for ik in range(len(k_vecs)):
                idx_k = ik
                k_vec = k_vecs[ik]
                write_kpoint(fw_u, k_vec)
                write_kpoint(fw_d, k_vec)
                for wfn_idx in range(wfn_min, wfn_max+1):
                    wfn = eig_vecs[idx_k, wfn_idx]
                    norbs = int(eig_vecs.shape[2]/2)
                    # output wavefunction
                    wfn_u = wfn[:norbs]
                    write_coefficients(fw_u, wfn_u)
                    wfn_d = wfn[norbs:]
                    write_coefficients(fw_d, wfn_d)
            fw_u.close()
            fw_d.close()
        else:
            filename = os.path.join(save_dir, 'wfn.bin')
            fw = open(filename, "wb")
            
            for ik in range(len(k_vecs)):
                idx_k = ik
                k_vec = k_vecs[ik]
                write_kpoint(fw, k_vec)
                for wfn_idx in range(wfn_min, wfn_max+1):
                    wfn = eig_vecs[idx_k, wfn_idx]
                    norbs = eig_vecs.shape[2]
                    # output wavefunction
                    write_coefficients(fw, wfn)
            fw.close()
    else:
        idx_k = input['k_idx'] 
        wfn_idx = input['wfn_idx']
        k_vec = np.array(input['k_vec'])
    
        eig_vecs = eig_vecs.astype(np.complex128)
        
        lat_per_inv = np.linalg.inv(latt).T
        k_vec = np.tensordot(k_vec, lat_per_inv, axes=1)
        wfn = eig_vecs[idx_k, wfn_idx]
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        if soc_switch:
            norbs = int(eig_vecs.shape[2]/2)
            # output wavefunction
            filename = os.path.join(save_dir, 'wfn_up.bin')
            fw = open(filename, "wb")
            idx_k = 0
            write_kpoint(fw, k_vec)
            wfn_u = wfn[:norbs]
            write_coefficients(fw, wfn_u)
            fw.close()
            
            filename = os.path.join(save_dir, 'wfn_down.bin')
            fw = open(filename, "wb")
            idx_k = 0
            write_kpoint(fw, k_vec)
            wfn_d = wfn[norbs:]
            write_coefficients(fw, wfn_d)
            fw.close()
    
        else:
            norbs = int(eig_vecs.shape[2])
            # output wavefunction
            filename = os.path.join(save_dir, 'wfn.bin')
            fw = open(filename, "wb")
            idx_k = 0
            write_kpoint(fw, k_vec)
            write_coefficients(fw, wfn)
            fw.close()

if __name__ == '__main__':
    main()
