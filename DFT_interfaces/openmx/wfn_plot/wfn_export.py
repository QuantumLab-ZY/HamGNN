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
                for i in k_vec.astype(np.float64):
                    fw_u.write(i)
                    fw_d.write(i)
                for wfn_idx in range(wfn_min, wfn_max+1):
                    wfn = eig_vecs[idx_k, wfn_idx]
                    norbs = int(eig_vecs.shape[2]/2)
                    # output wavefunction
                    wfn_u = wfn[:norbs]
                    for (i, j) in zip(wfn_u.real.astype(np.float64), wfn_u.imag.astype(np.float64)):
                        fw_u.write(i)
                        fw_u.write(j)
                    wfn_d = wfn[norbs:]
                    for (i, j) in zip(wfn_d.real.astype(np.float64), wfn_d.imag.astype(np.float64)):
                        fw_d.write(i)
                        fw_d.write(j)
            fw_u.close()
            fw_d.close()
        else:
            filename = os.path.join(save_dir, 'wfn.bin')
            fw = open(filename, "wb")
            
            for ik in range(len(k_vecs)):
                idx_k = ik
                k_vec = k_vecs[ik]
                for i in k_vec.astype(np.float64):
                    fw.write(i)
                for wfn_idx in range(wfn_min, wfn_max+1):
                    wfn = eig_vecs[idx_k, wfn_idx]
                    norbs = eig_vecs.shape[2]
                    # output wavefunction
                    for (i, j) in zip(wfn.real.astype(np.float64), wfn.imag.astype(np.float64)):
                        fw.write(i)
                        fw.write(j)
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
            for i in k_vec.astype(np.float64):
                fw.write(i)
            wfn_u = wfn[:norbs]
            for (i, j) in zip(wfn_u.real.astype(np.float64), wfn_u.imag.astype(np.float64)):
                fw.write(i)
                fw.write(j)
            fw.close()
            
            filename = os.path.join(save_dir, 'wfn_down.bin')
            fw = open(filename, "wb")
            idx_k = 0
            for i in k_vec.astype(np.float64):
                fw.write(i)
            wfn_d = wfn[norbs:]
            for (i, j) in zip(wfn_d.real.astype(np.float64), wfn_d.imag.astype(np.float64)):
                fw.write(i)
                fw.write(j)
            fw.close()
    
        else:
            norbs = int(eig_vecs.shape[2])
            # output wavefunction
            filename = os.path.join(save_dir, 'wfn.bin')
            fw = open(filename, "wb")
            idx_k = 0
            for i in k_vec.astype(np.float64):
                fw.write(i)
            for (i, j) in zip(wfn.real.astype(np.float64), wfn.imag.astype(np.float64)):
                fw.write(i)
                fw.write(j)
            fw.close()

if __name__ == '__main__':
    main()