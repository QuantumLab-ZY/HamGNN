===================================
Construction of graph_data.npz File
===================================

``graph_data.npz`` is the core input format for HamGNN, containing material structure information and Hamiltonian matrix data. The following describes its construction method.

General Process
===============

Regardless of which DFT software is used, the basic process for constructing ``graph_data.npz`` is as follows:

1. **Structure File Conversion**: Convert material structure files (such as POSCAR, CIF) to the corresponding DFT software input format
2. **Non-self-consistent Hamiltonian Calculation**: Use post-processing tools to generate files containing non-self-consistent Hamiltonian H0 (matrices that do not depend on self-consistent charge density, such as kinetic energy matrices)
3. **DFT Calculation** (Optional): Run complete DFT calculations to obtain real Hamiltonians (required for training set construction, can be skipped for pure prediction)
4. **Data Packaging**: Call the appropriate ``graph_data_gen`` script to package structure and Hamiltonian matrix data into ``graph_data.npz``

OpenMX Process
==============

1. **Structure Conversion**: Edit the ``poscar2openmx.yaml`` configuration file, setting appropriate paths and DFT parameters (for non-SOC data, set ``scf.SpinPolarization: Off`` and ``scf.SpinOrbit.Coupling: off``; for SOC data, set ``scf.SpinPolarization: nc`` and ``scf.SpinOrbit.Coupling: on``):

   .. code-block:: yaml

       system_name: 'Si'
       poscar_path: "/path/to/poscar/*.vasp" # Path to POSCAR or CIF files
       filepath: '/path/to/save/dat/file' # Directory to save OpenMX files
       basic_command: |+  # OpenMX calculation parameters
         #
         #      File Name      
         #
         System.CurrrentDirectory         ./    # default=./
         System.Name                     Si
         DATA.PATH           /path/to/DFT_DATA   # default=../DFT_DATA19
         # ...other OpenMX parameters...

2. **Non-self-consistent Hamiltonian Calculation**: Run ``openmx_postprocess`` on the generated ``.dat`` files:

   .. code-block:: bash

      mpirun -np ncpus ./openmx_postprocess openmx.dat

   This will generate an ``overlap.scfout`` file containing overlap matrices and non-self-consistent Hamiltonian information.

3. **DFT Calculation** (Optional, required for training set):

   .. code-block:: bash

      mpirun -np ncpus openmx openmx.dat > openmx.std

   This will generate a ``.scfout`` file containing self-consistent Hamiltonians.

4. **Data Packaging**: Configure ``graph_data_gen.yaml``:

   .. code-block:: yaml

      nao_max: 26  # Maximum number of atomic orbitals
      graph_data_save_path: '/path/to/save/graph_data'
      read_openmx_path: '/path/to/HamGNN/utils_openmx/read_openmx'
      max_SCF_skip: 200
      scfout_paths: '/path/to/scfout/files'  # Directory of .scfout files
      dat_file_name: 'openmx.dat'
      std_file_name: 'openmx.std'  # Set to null if no DFT calculation
      scfout_file_name: 'openmx.scfout'  # Use 'overlap.scfout' for prediction
      soc_switch: False  # Set to True for SOC data

   Then run:

   .. code-block:: bash

      graph_data_gen --config graph_data_gen.yaml

SIESTA/HONPAS Process
=====================

1. **Structure Conversion**: Edit the ``poscar_path`` and ``filepath`` parameters in the ``poscar2siesta.py`` script, then run:

   .. code-block:: bash

      python poscar2siesta.py

2. **DFT Calculation**: Use HONPAS to perform DFT calculations to obtain ``.HSX`` Hamiltonian matrix files.

3. **Non-self-consistent Hamiltonian Calculation**: Run the following command to generate non-self-consistent Hamiltonians and overlap matrices:

   .. code-block:: bash

      mpirun -np Ncores honpas_1.2_H0 < input.fdf

   This will generate an ``overlap.HSX`` file.

4. **Data Packaging**: Modify the parameters in the ``graph_data_gen_siesta.py`` script, then run:

   .. code-block:: bash

      python graph_data_gen_siesta.py

ABACUS Process
==============

1. **Structure Conversion**: Use ``poscar2abacus.py`` to generate ABACUS input files.

2. **DFT Calculation and Post-processing**: Run ABACUS calculations and use ``abacus_postprocess`` to extract H0 matrices.

3. **Data Packaging**: Use the ``graph_data_gen_abacus.py`` script to integrate data and generate ``graph_data.npz``.