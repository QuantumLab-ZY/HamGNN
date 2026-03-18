'''
Author: Changwei Zhang 
Date: 2023-05-20 15:32:19 
Last Modified by:   Changwei Zhang 
Last Modified time: 2023-05-20 15:32:19 
'''

import glob
from pymatgen.core.structure import Structure
from pymatgen.core.periodic_table import Element
from pymatgen.io.ase import AseAtomsAdaptor
import os
import natsort
import numpy as np
from ase import Atoms
from numba import njit

PP_dict = {
    'Ag':'Ag_ONCV_PBE-1.0.upf',  'Co':'Co_ONCV_PBE-1.0.upf',  'Ir':'Ir_ONCV_PBE-1.0.upf',  'Os':'Os_ONCV_PBE-1.0.upf',  'S' :'S_ONCV_PBE-1.0.upf',
    'Al':'Al_ONCV_PBE-1.0.upf',  'Cr':'Cr_ONCV_PBE-1.0.upf',  'K' :'K_ONCV_PBE-1.0.upf',   'Pb':'Pb_ONCV_PBE-1.0.upf',  'Sr':'Sr_ONCV_PBE-1.0.upf',
    'Ar':'Ar_ONCV_PBE-1.0.upf',  'Cs':'Cs_ONCV_PBE-1.0.upf',  'Kr':'Kr_ONCV_PBE-1.0.upf',  'Pd':'Pd_ONCV_PBE-1.0.upf',  'Ta':'Ta_ONCV_PBE-1.0.upf',
    'As':'As_ONCV_PBE-1.0.upf',  'Cu':'Cu_ONCV_PBE-1.0.upf',  'La':'La_ONCV_PBE-1.0.upf',  'P' :'P_ONCV_PBE-1.0.upf',   'Tc':'Tc_ONCV_PBE-1.0.upf',
    'Au':'Au_ONCV_PBE-1.0.upf',  'Fe':'Fe_ONCV_PBE-1.0.upf',  'Li':'Li_ONCV_PBE-1.0.upf',  'Pt':'Pt_ONCV_PBE-1.0.upf',  'Te':'Te_ONCV_PBE-1.0.upf',
    'Ba':'Ba_ONCV_PBE-1.0.upf',  'F' :'F_ONCV_PBE-1.0.upf',   'Mg':'Mg_ONCV_PBE-1.0.upf',  'Rb':'Rb_ONCV_PBE-1.0.upf',  'Ti':'Ti_ONCV_PBE-1.0.upf',
    'Be':'Be_ONCV_PBE-1.0.upf',  'Ga':'Ga_ONCV_PBE-1.0.upf',  'Mn':'Mn_ONCV_PBE-1.0.upf',  'Re':'Re_ONCV_PBE-1.0.upf',  'Tl':'Tl_ONCV_PBE-1.0.upf',
    'Bi':'Bi_ONCV_PBE-1.0.upf',  'Ge':'Ge_ONCV_PBE-1.0.upf',  'Mo':'Mo_ONCV_PBE-1.0.upf',  'Rh':'Rh_ONCV_PBE-1.0.upf',  'V' :'V_ONCV_PBE-1.0.upf',
    'B' :'B_ONCV_PBE-1.0.upf',   'He':'He_ONCV_PBE-1.0.upf',  'Na':'Na_ONCV_PBE-1.0.upf',  'Ru':'Ru_ONCV_PBE-1.0.upf',  'W' :'W_ONCV_PBE-1.0.upf',
    'Br':'Br_ONCV_PBE-1.0.upf',  'Hf':'Hf_ONCV_PBE-1.0.upf',  'Nb':'Nb_ONCV_PBE-1.0.upf',  'Sb':'Sb_ONCV_PBE-1.0.upf',  'Xe':'Xe_ONCV_PBE-1.0.upf',
    'Ca':'Ca_ONCV_PBE-1.0.upf',  'Hg':'Hg_ONCV_PBE-1.0.upf',  'Ne':'Ne_ONCV_PBE-1.0.upf',  'Sc':'Sc_ONCV_PBE-1.0.upf',  'Y' :'Y_ONCV_PBE-1.0.upf',
    'Cd':'Cd_ONCV_PBE-1.0.upf',  'H' :'H_ONCV_PBE-1.0.upf',   'Ni':'Ni_ONCV_PBE-1.0.upf',  'Se':'Se_ONCV_PBE-1.0.upf',  'Zn':'Zn_ONCV_PBE-1.0.upf',
    'Cl':'Cl_ONCV_PBE-1.0.upf',  'In':'In_ONCV_PBE-1.0.upf',  'N' :'N_ONCV_PBE-1.0.upf',   'Si':'Si_ONCV_PBE-1.0.upf',  'Zr':'Zr_ONCV_PBE-1.0.upf',
    'C' :'C_ONCV_PBE-1.0.upf',   'I' :'I_ONCV_PBE-1.0.upf',   'O' :'O_ONCV_PBE-1.0.upf',   'Sn':'Sn_ONCV_PBE-1.0.upf'
}

ORB_dict = {
    'Ag':'Ag_gga_7au_100Ry_4s2p2d1f.orb',   'Cu':'Cu_gga_8au_100Ry_4s2p2d1f.orb',    'Mo':'Mo_gga_7au_100Ry_4s2p2d1f.orb',  'Sc':'Sc_gga_8au_100Ry_4s2p2d1f.orb',
    'Al':'Al_gga_7au_100Ry_4s4p1d.orb',     'Fe':'Fe_gga_8au_100Ry_4s2p2d1f.orb',    'Na':'Na_gga_8au_100Ry_4s2p1d.orb',    'Se':'Se_gga_8au_100Ry_2s2p1d.orb',
    'Ar':'Ar_gga_7au_100Ry_2s2p1d.orb',     'F' :'F_gga_7au_100Ry_2s2p1d.orb',       'Nb':'Nb_gga_8au_100Ry_4s2p2d1f.orb',  'S' :'S_gga_7au_100Ry_2s2p1d.orb',
    'As':'As_gga_7au_100Ry_2s2p1d.orb',     'Ga':'Ga_gga_8au_100Ry_2s2p2d1f.orb',    'Ne':'Ne_gga_6au_100Ry_2s2p1d.orb',    'Si':'Si_gga_7au_100Ry_2s2p1d.orb',
    'Au':'Au_gga_7au_100Ry_4s2p2d1f.orb',   'Ge':'Ge_gga_8au_100Ry_2s2p2d1f.orb',    'N' :'N_gga_7au_100Ry_2s2p1d.orb',     'Sn':'Sn_gga_7au_100Ry_2s2p2d1f.orb',
    'Ba':'Ba_gga_10au_100Ry_4s2p2d1f.orb',  'He':'He_gga_6au_100Ry_2s1p.orb',        'Ni':'Ni_gga_8au_100Ry_4s2p2d1f.orb',  'Sr':'Sr_gga_9au_100Ry_4s2p1d.orb',
    'Be':'Be_gga_7au_100Ry_4s1p.orb',       'Hf':'Hf_gga_7au_100Ry_4s2p2d2f1g.orb',  'O' :'O_gga_7au_100Ry_2s2p1d.orb',     'Ta':'Ta_gga_8au_100Ry_4s2p2d2f1g.orb',
    'B' :'B_gga_8au_100Ry_2s2p1d.orb',      'H' :'H_gga_6au_100Ry_2s1p.orb',         'Os':'Os_gga_7au_100Ry_4s2p2d1f.orb',  'Tc':'Tc_gga_7au_100Ry_4s2p2d1f.orb',
    'Bi':'Bi_gga_7au_100Ry_2s2p2d1f.orb',   'Hg':'Hg_gga_9au_100Ry_4s2p2d1f.orb',    'Pb':'Pb_gga_7au_100Ry_2s2p2d1f.orb',  'Te':'Te_gga_7au_100Ry_2s2p2d1f.orb',
    'Br':'Br_gga_7au_100Ry_2s2p1d.orb',     'I' :'I_gga_7au_100Ry_2s2p2d1f.orb',     'Pd':'Pd_gga_7au_100Ry_4s2p2d1f.orb',  'Ti':'Ti_gga_8au_100Ry_4s2p2d1f.orb',
    'Ca':'Ca_gga_9au_100Ry_4s2p1d.orb',     'In':'In_gga_7au_100Ry_2s2p2d1f.orb',    'P' :'P_gga_7au_100Ry_2s2p1d.orb',     'Tl':'Tl_gga_7au_100Ry_2s2p2d1f.orb',
    'Cd':'Cd_gga_7au_100Ry_4s2p2d1f.orb',   'Ir':'Ir_gga_7au_100Ry_4s2p2d1f.orb',    'Pt':'Pt_gga_7au_100Ry_4s2p2d1f.orb',  'V' :'V_gga_8au_100Ry_4s2p2d1f.orb',
    'C' :'C_gga_7au_100Ry_2s2p1d.orb',      'K' :'K_gga_9au_100Ry_4s2p1d.orb',       'Rb':'Rb_gga_10au_100Ry_4s2p1d.orb',   'W' :'W_gga_8au_100Ry_4s2p2d2f1g.orb',
    'Cl':'Cl_gga_7au_100Ry_2s2p1d.orb',     'Kr':'Kr_gga_7au_100Ry_2s2p1d.orb',      'Re':'Re_gga_7au_100Ry_4s2p2d1f.orb',  'Xe':'Xe_gga_8au_100Ry_2s2p2d1f.orb',
    'Co':'Co_gga_8au_100Ry_4s2p2d1f.orb',   'Li':'Li_gga_7au_100Ry_4s1p.orb',        'Rh':'Rh_gga_7au_100Ry_4s2p2d1f.orb',  'Y' :'Y_gga_8au_100Ry_4s2p2d1f.orb',
    'Cr':'Cr_gga_8au_100Ry_4s2p2d1f.orb',   'Mg':'Mg_gga_8au_100Ry_4s2p1d.orb',      'Ru':'Ru_gga_7au_100Ry_4s2p2d1f.orb',  'Zn':'Zn_gga_8au_100Ry_4s2p2d1f.orb',
    'Cs':'Cs_gga_10au_100Ry_4s2p1d.orb',    'Mn':'Mn_gga_8au_100Ry_4s2p2d1f.orb',    'Sb':'Sb_gga_7au_100Ry_2s2p2d1f.orb',  'Zr':'Zr_gga_8au_100Ry_4s2p2d1f.orb'
}

def check_bound(poses:np.ndarray, cell:np.ndarray, bound:np.ndarray=np.zeros(3)):
    invcell = np.linalg.inv(cell)
    direct = np.zeros_like(poses)
    for i, pos in enumerate(poses):
        dir = pos @ invcell
        direct[i] = dir
    showmessage = np.all(direct < 1) and np.all(direct >= 0)
    direct = direct + bound
    for i, dir in enumerate(direct):
        pos = dir @ cell
        poses[i] = pos
    return poses, showmessage

def ase_atoms_to_abacusfile(atoms:Atoms, filename:str):
    chemical_symbols = atoms.get_chemical_symbols()
    species = list(set(chemical_symbols))
    species.sort()
    positions = atoms.get_array(name='positions')
    cell = atoms.get_cell().array
    positions, showmessage = check_bound(positions, cell)

    abacus = "ATOMIC_SPECIES\n"
    for idx, s in enumerate(species):
        abacus += "{:2s} {:8.4f}  {:s}\n".format(s, Element(s).atomic_mass, PP_dict[s])

    abacus += "\nNUMERICAL_ORBITAL\n"
    for idx, s in enumerate(species):
        abacus += f"{ORB_dict[s]}\n"

    abacus += "\nLATTICE_CONSTANT\n1.8897259886\n"

    abacus += "\nLATTICE_VECTORS\n"
    abacus += " %19.15f %19.15f %19.15f\n %19.15f %19.15f %19.15f\n %19.15f %19.15f %19.15f\n" % (*cell[0], *cell[1], *cell[2])

    abacus += "\nATOMIC_POSITIONS\nCartesian\n"
    num = 0
    for sym in species:
        na = chemical_symbols.count(sym)
        abacus += f"{sym}\n0.0\n{na}\n"
        for i, isym in enumerate(chemical_symbols):
            if isym != sym:
                continue
            abacus += " %15.10f %15.10f %15.10f 0 0 0\n" % (*positions[i],)
            num += 1

    with open(filename,'w') as wf:
        wf.write(abacus)

################################ ABACUS calculation parameters Set end #######################

if __name__ == '__main__':
    ######################## Input parameters begin #######################
    system_name = 'Si'
    poscar_path = "../Si_in/*.vasp" # The path of poscar or cif files
    filepath = '../Si_abacus' # abacus file directory to save
    ######################## Input parameters end #########################

    if not os.path.exists(filepath):
        os.mkdir(filepath)

    f_vasp = glob.glob(poscar_path) # poscar or cif file directory
    f_vasp = natsort.natsorted(f_vasp)

    for i, poscar in enumerate(f_vasp):
        cif_id = str(i+1)

        crystal = Structure.from_file(poscar)
        ase_atoms = AseAtomsAdaptor.get_atoms(crystal)
        filename =  os.path.join(filepath, 'STRU_'+ cif_id)
        ase_atoms_to_abacusfile(ase_atoms, filename)

