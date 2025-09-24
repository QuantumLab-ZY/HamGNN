=============================
Uni-HamGNN Universal Model
=============================

Model Introduction
==================

Uni-HamGNN is a universal spin-orbit coupling Hamiltonian model designed to accelerate quantum material discovery. The model addresses the major challenge of modeling spin-orbit coupling (SOC) effects in various complex systems, which traditionally requires computationally expensive density functional theory (DFT) calculations.

Uni-HamGNN eliminates the need for system-specific retraining and costly SOC-DFT calculations, enabling high-throughput screening of quantum materials across systems of different dimensions. This makes it a powerful tool for quantum material design and property studies, significantly accelerating the pace of discovery in condensed matter physics and materials science.

Input Requirements
==================

The universal SOC Hamiltonian model requires two ``graph_data.npz`` files as input data:

- One file in non-SOC mode
- One file in SOC mode

The preparation method for these files is the same as described in previous chapters, but attention should be paid to the differences between SOC and non-SOC parameters.

Usage Process
=============

1. **Prepare input data**:
   
   - Convert structures to be predicted into two ``graph_data.npz`` files (non-SOC and SOC modes)
   - Place these files in the specified directories

2. **Configure parameters**:
   
   Edit the ``Input.yaml`` configuration file:

   .. code-block:: yaml

      # HamGNN prediction configuration
      model_pkl_path: '/path/to/universal_model.pkl'
      non_soc_data_dir: '/path/to/non_soc_graph_data'
      soc_data_dir: '/path/to/soc_graph_data'  # Optional, only needed for SOC calculations
      output_dir: './results'
      device: 'cuda'  # Use GPU, fallback to CPU if not available
      calculate_mae: true  # Calculate mean absolute error

3. **Run prediction**:

   .. code-block:: bash

      python Uni-HamiltonianPredictor.py --config Input.yaml

Output Results
==============

After execution, the script will generate the following files in the specified ``output_dir``:

- ``hamiltonian.npy``: Predicted Hamiltonian matrix in NumPy array format
- If ``calculate_mae`` is enabled, MAE statistics will be printed to the console