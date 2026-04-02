# Copyright (c) 2021-2026 HamGNN Team
# SPDX-License-Identifier: GPL-3.0-only

"""Convert POSCAR/CIF structures to OpenMX input decks using shared PAO/pseudopotential tables.

Entry point is YAML-configured; delegates formatting to :mod:`interfaces.openmx.utils`.
"""

from pymatgen.core.structure import Structure
import glob
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
import os
import natsort
from interfaces.openmx.utils import *
import argparse
import yaml


def main():
    parser = argparse.ArgumentParser(description='openmx dat file generation')
    parser.add_argument('--config', default='poscar2openmx.yaml', type=str, metavar='N')
    args = parser.parse_args()
    
    with open(args.config, encoding='utf-8') as rstream:
        input = yaml.load(rstream, yaml.SafeLoader)
    
    system_name = input['system_name']
    poscar_path = input['poscar_path'] # The path of poscar or cif files
    filepath = input['filepath'] # openmx file directory to save
    basic_command = input['basic_command']
    
    if not os.path.exists(filepath):
        os.mkdir(filepath)

    f_vasp = glob.glob(poscar_path) # poscar or cif file directory
    f_vasp = natsort.natsorted(f_vasp)

    for i, poscar in enumerate(f_vasp):
        cif_id = str(i+1)
        crystal = Structure.from_file(poscar)
        ase_atoms = AseAtomsAdaptor.get_atoms(crystal)
        cell = ase_atoms.get_cell().array
        filename =  os.path.join(filepath, f'{system_name}_'+ cif_id + ".dat")
        ase_atoms_to_openmxfile(ase_atoms, basic_command, spin_set, PAO_dict, PBE_dict, filename)

if __name__ == '__main__':
    main()