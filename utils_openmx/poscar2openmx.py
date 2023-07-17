'''
Descripttion: Script for converting poscar to openmx input file
version: 0.1
Author: Yang Zhong
Date: 2022-11-24 19:03:36
LastEditors: Yang Zhong
LastEditTime: 2023-07-18 03:09:31
'''

from pymatgen.core.structure import Structure
import glob
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
import os
import natsort
from utils_openmx.utils import *

######################## Input parameters begin #######################
system_name = 'Si'

poscar_path = "../PC/*.vasp" # The path of poscar or cif files
filepath = '../PC' # openmx file directory to save
######################## Input parameters end #########################

################################ OPENMX calculation parameters Set begin #####################

basic_commad = f"""#
#      File Name      
#

System.CurrrentDirectory         ./    # default=./
System.Name                      {system_name}
DATA.PATH           /public/home/cwzhang/templates/DFT_DATA19   # default=../DFT_DATA19
level.of.stdout                   1    # default=1 (1-3)
level.of.fileout                  1    # default=1 (0-2)
HS.fileout                   on       # on|off, default=off

#
# SCF or Electronic System
#

scf.XcType                  GGA-PBE    # LDA|LSDA-CA|LSDA-PW|GGA-PBE
scf.SpinPolarization        off        # On|Off|NC
scf.ElectronicTemperature  100.0       # default=300 (K)
scf.energycutoff           200.0       # default=150 (Ry)
scf.maxIter                 300         # default=40
scf.EigenvalueSolver        Band      # DC|GDC|Cluster|Band
scf.Kgrid                  5 5 5       # means 4x4x4
scf.Mixing.Type           rmm-diis     # Simple|Rmm-Diis|Gr-Pulay|Kerker|Rmm-Diisk
scf.Init.Mixing.Weight     0.10        # default=0.30 
scf.Min.Mixing.Weight      0.001       # default=0.001 
scf.Max.Mixing.Weight      0.400       # default=0.40 
scf.Mixing.History          7          # default=5
scf.Mixing.StartPulay       5          # default=6
scf.criterion             1.0e-7      # default=1.0e-6 (Hartree)

#
# MD or Geometry Optimization
#

MD.Type                      Nomd        # Nomd|Opt|NVE|NVT_VS|NVT_NH
                                       # Constraint_Opt|DIIS2|Constraint_DIIS2
MD.Opt.DIIS.History          4
MD.Opt.StartDIIS             5         # default=5
MD.maxIter                 100         # default=1
MD.TimeStep                1.0         # default=0.5 (fs)
MD.Opt.criterion          1.0e-4       # default=1.0e-4 (Hartree/bohr)

#
# MO output
#

MO.fileout                  off        # on|off, default=off
num.HOMOs                    2         # default=1
num.LUMOs                    2         # default=1

#
# DOS and PDOS
#

Dos.fileout                  off       # on|off, default=off
Dos.Erange              -10.0  10.0    # default = -20 20 
Dos.Kgrid                 1  1  1      # default = Kgrid1 Kgrid2 Kgrid3

\n"""

################################ OPENMX calculation parameters Set end #######################

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
    ase_atoms_to_openmxfile(ase_atoms, basic_commad, spin_set, PAO_dict, PBE_dict, filename)

