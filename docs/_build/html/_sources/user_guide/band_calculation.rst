==============================
Band Structure Calculation
==============================

Band calculation is used to verify the accuracy of predicted Hamiltonians, checking if they can correctly reproduce the electronic band structure of materials.

Operation Process
===================

1. **Edit configuration file**: Create a ``band_cal.yaml`` file, specifying the following parameters:

   .. code-block:: yaml

       nao_max: 26
       graph_data_path: '/path/to/graph_data.npz' # Path to graph_data.npz
       hamiltonian_path: '/path/to/prediction_hamiltonian.npy'  # Path to Hamiltonian matrix
       nk: 120          # the number of k points
       save_dir: '/path/to/save/band/calculation/result' # The directory to save the results
       strcture_name: 'Si'  # The name of each cif file saved is strcture_name_idx.cif after band calculation
       soc_switch: False
       spin_colinear: False
       auto_mode: True # If the auto_mode is used, users can omit providing k_path and label, as the program will automatically generate them based on the crystal symmetry.
       k_path: [[0.,0.,-0.5],[0.,0.,0.0],[0.,0.,0.5]] # High-symmetry point path
       label: ['$Mbar$','$G$','$M$'] # The lable for each k points in K_path

2. **Run calculation**:

   .. code-block:: bash

      band_cal --config band_cal.yaml

Band Calculation for Large Systems
=====================================

For large systems, you can use the parallel version of the band calculation tool ``band_cal_parallel``:

1. **Install parallel tools**:

   .. code-block:: bash

      pip install mpitool-0.0.1-cp39-cp39-manylinux1_x86_64.whl
      pip install band_cal_parallel-0.1.12-py3-none-any.whl

2. **Run parallel calculation**:

   .. code-block:: bash

      mpirun -np N_CORES band_cal_parallel --config band_cal_parallel.yaml

.. note::
   Some MKL environments may encounter the error ``Intel MKL FATAL ERROR: Cannot load symbol MKLMPI_Get_wrappers``. Refer to GitHub Issues `#18 <https://github.com/QuantumLab-ZY/HamGNN/issues/18>`_ and `#12 <https://github.com/QuantumLab-ZY/HamGNN/issues/12>`_ for solutions.