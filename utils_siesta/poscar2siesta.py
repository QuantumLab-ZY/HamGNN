'''
Author: Changwei Zhang 
Date: 2023-05-23 22:38:22 
Last Modified by:   Changwei Zhang 
Last Modified time: 2023-05-23 22:38:22 
'''

from pymatgen.core.structure import Structure
from pymatgen.core.periodic_table import Element
from pymatgen.io.ase import AseAtomsAdaptor
import os
import numpy as np
from ase import Atoms
from numba import njit

################################ SIESTA calculation parameters Set begin #####################

basic_commad = """
SystemName      {:s}
SystemLabel     {:s}

PAO.BasisSize           DZP
PAO.SplitNorm           0.26

# %block PS.lmax
#   C  2
# %endblock PS.lmax

#  >>>>> KPOINTS <<<<<

%block kgrid_Monkhorst_Pack
   1  0  0  0.0
   0  1  0  0.0
   0  0  1  0.0
%endblock kgrid_Monkhorst_Pack

xc.functional      GGA    # Default vaxc.authors         HSE-06    # Default value
xc.authors         PBE    # Default value

# >>>>> self-consistent-field stuff <<<<<

ElectronicTemperature   500. K
SCF.Mix                 Hamiltonian
SCF.Mix.First          .true.
SCF.Mix.First.Force    .false.
SCF.Mixer.Method        Pulay
SCF.Mixer.Weight        0.1
SCF.Mixer.History       6

DM.UseSaveDM           .false.
MaxSCFIteration         1000
SCF.FreeE.Converge     .true.
SCF.FreeE.Tolerance     1.d-6 eV
SCF.DM.Converge        .true.
SCF.DM.Tolerance        1.d-4 eV
# Write.DM.end.of.cycle  .true.
# Write.H.end.of.cycle   .true.
SaveHS                 .true.
SaveRho                .true.
# OR COOP.Write .true.

Mesh.Cutoff             300. Ry     # Reduce eggbox effect


# DM.UseSaveDM            T
# MeshCutoff              250. Ry     # Equivalent planewave cutoff for the grid
# MaxSCFIterations        50         # Maximum number of SCF iterations per step
# DM.MixingWeight         0.1         # New DM amount for next SCF cycle
# DM.Tolerance            1.d-4       # Tolerance in maximum difference
#                                     # between input and output DM
# DM.NumberPulay          6          # Number of SCF steps between pulay mixing

# >>>>> Eigenvalue problem: order-N or diagonalization <<<<<

SolutionMethod          diagon      # OrderN or Diagon

MD.TypeOfRun            CG          # Type of dynamics:
MD.Steps                0
# MD.VariableCell        .true.
# MD.MaxForceTol          0.01 eV/Ang
# #WriteMDXmol           .true.
# WriteForces            .true.
# WriteCoorStep

HFX.UseFittedNAOs      .true.
HFX.Dynamic_parallel   .true.

\n"""

@njit
def check_bound(poses:np.ndarray, cell:np.ndarray):
    invcell = np.linalg.inv(cell)
    direct = np.zeros_like(poses)
    for i, pos in enumerate(poses):
        dir = pos @ invcell
        direct[i] = dir
    showmessage = not(np.all(direct < 1) and np.all(direct >= 0))
    direct = direct % 1
    for i, dir in enumerate(direct):
        pos = dir @ cell
        poses[i] = pos
    return poses, showmessage

def ase_atoms_to_siestafile(atoms:Atoms, basic_commad:str, filename:str):
    chemical_symbols = atoms.get_chemical_symbols()
    species = ['Ga', 'As'] # list(set(chemical_symbols))
    positions = atoms.get_array(name='positions')
    cell = atoms.get_cell().array
    positions, showmessage = check_bound(positions, cell)
    siesta = basic_commad
    siesta += "#\n# Definition of Atomic Species\n#\n"
    siesta += f'NumberOfSpecies       {len(species)}\n'
    siesta += '%block ChemicalSpeciesLabel\n'
    for idx, s in enumerate(species):
        siesta += f"  {idx+1}  {Element(s).Z}  {s}\n"    
    siesta += "%endblock ChemicalSpeciesLabel\n\n"

    siesta += "#\n# Atoms\n#\n"
    siesta += f"NumberOfAtoms         {len(chemical_symbols)}\n\n"
    siesta += "AtomicCoordinatesFormat   Ang # Ang|Bohr|Fractional\n"
    siesta += "%block AtomicCoordinatesAndAtomicSpecies\n"
    for num, sym in enumerate(chemical_symbols):
        siesta += "  %10.7f  %10.7f  %10.7f   %d\n" % (*positions[num], species.index(sym)+1)
    siesta += "%endblock AtomicCoordinatesAndAtomicSpecies\n\n"
    siesta += "LatticeConstant      1.00 Ang\n"
    siesta += "%block LatticeVectors\n"
    siesta += "      %10.7f  %10.7f  %10.7f\n      %10.7f  %10.7f  %10.7f\n      %10.7f  %10.7f  %10.7f\n" % (*cell[0], *cell[1], *cell[2])
    siesta += "%endblock LatticeVectors"
    with open(filename,'w') as wf:
        wf.write(siesta)

################################ SIESTA calculation parameters Set end #######################

if __name__ == '__main__':
    ######################## Input parameters begin #######################
    system_name = 'cell'
    poscar_path = ["{:d}.vasp".format(i) for i in range(0,259)] # The path of poscar or cif files
    filepath = 'output/' # siesta file directory to save
    ######################## Input parameters end #########################

    if not os.path.exists(filepath):
        os.mkdir(filepath)

    for i, poscar in enumerate(poscar_path):
        cif_id = str(i)

        crystal = Structure.from_file(poscar)
        ase_atoms = AseAtomsAdaptor.get_atoms(crystal)
        cell = ase_atoms.get_cell().array
        filename =  os.path.join(filepath, f'{system_name}_'+ cif_id + ".fdf")
        ase_atoms_to_siestafile(ase_atoms, 
                                '', # basic_commad.format(system_name, system_name), # or ''
                                filename)

