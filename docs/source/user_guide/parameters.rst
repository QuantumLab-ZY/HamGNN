========================
HamGNN Parameter Details
========================

This section explains in detail the parameter modules and parameters in the ``config.yaml`` configuration file.

setup (Basic Settings)
----------------------

.. list-table::
   :widths: 20 15 50 15
   :header-rows: 1

   * - Parameter
     - Type
     - Description
     - Default/Recommended Value
   * - ``GNN_Net``
     - String
     - Type of GNN network used
     - ``HamGNN_pre`` (or ``HamGNNpre``) for normal Hamiltonian fitting, ``HamGNN_pre_charge`` for charged defect Hamiltonian fitting, ``HamGNNTransformer`` for transformer-based model
   * - ``accelerator``
     - null or String
     - Accelerator type
     - ``null``
   * - ``ignore_warnings``
     - Boolean
     - Whether to ignore warnings
     - ``true``
   * - ``checkpoint_path``
     - String
     - Checkpoint path for resuming training or path used during testing
     - ``'./'`` (no default checkpoint)
   * - ``load_from_checkpoint``
     - Boolean
     - Whether to load model parameters from checkpoint
     - ``false`` (for new training), ``true`` (when loading pre-trained model)
   * - ``resume``
     - Boolean
     - Whether to continue training from interruption
     - ``false`` (for new training), ``true`` (to continue training)
   * - ``num_gpus``
     - null, Integer, or List
     - Number or ID of GPUs to use
     - ``1`` (first GPU), ``null`` for CPU
   * - ``precision``
     - Integer
     - Computation precision
     - ``32`` (32-bit precision), optional ``64`` (64-bit precision)
   * - ``property``
     - String
     - Type of physical quantity output by the network
     - ``hamiltonian``
   * - ``stage``
     - String
     - Execution stage
     - ``fit`` (training), ``test`` (testing/prediction)
   * - ``hostname``
     - String
     - Host identifier (auto-detected)
     - Auto-detected system hostname
   * - ``job_id``
     - String
     - Job identifier for tracking
     - Auto-generated (e.g., ``time_2025``)

profiler_params (Profiler Parameters)
--------------------------------------

.. list-table::
   :widths: 20 15 50 15
   :header-rows: 1

   * - Parameter
     - Type
     - Description
     - Default/Recommended Value
   * - ``train_dir``
     - String
     - Training output directory (tensorboard logs, checkpoints)
     - ``'./'`` (current directory)
   * - ``progress_bar_refresh_rat``
     - Integer
     - Progress bar refresh rate
     - ``1``

dataset_params (Dataset Parameters)
-----------------------------------

.. list-table::
   :widths: 20 15 50 15
   :header-rows: 1

   * - Parameter
     - Type
     - Description
     - Default/Recommended Value
   * - ``batch_size``
     - Integer
     - Number of samples processed per batch
     - ``1`` (typically kept as 1)
   * - ``test_ratio``
     - Float
     - Proportion of test set in the entire dataset
     - ``0.2`` (20%)
   * - ``train_ratio``
     - Float
     - Proportion of training set in the entire dataset
     - ``0.6`` (60%)
   * - ``val_ratio``
     - Float
     - Proportion of validation set in the entire dataset
     - ``0.2`` (20%)
   * - ``split_file``
     - String or null
     - Path to save/load pre-defined dataset split indices
     - ``None`` (auto-split based on ratios)
   * - ``graph_data_path``
     - String
     - Directory of processed compressed graph data files
     - No default value, must be manually set
   * - ``num_workers``
     - Integer
     - Number of parallel DataLoader worker processes
     - ``4``
   * - ``preload``
     - Integer
     - Number of graphs to preload into memory on startup
     - ``0`` (load on demand)
   * - ``data_format``
     - String
     - Input data format: ``'auto'``, ``'lmdb'`` (LMDB format), or ``'npz'`` (NPZ format)
     - ``'auto'`` (auto-detect based on file extension)
   * - ``test_mode``
     - Boolean
     - Use entire dataset as test set (skip train/val split)
     - ``false``

.. note::
   For large-scale datasets, using LMDB format (``data_format: lmdb``) with ``npz_to_lmdb.py`` conversion provides significantly faster I/O compared to NPZ format.

losses_metrics (Loss Functions and Evaluation Metrics)
------------------------------------------------------

.. list-table::
   :widths: 20 15 50 15
   :header-rows: 1

   * - Parameter
     - Type
     - Description
     - Default/Recommended Value
   * - ``losses``
     - List
     - List of loss function definitions
     - Must include at least Hamiltonian loss
   * - ``losses[].loss_weight``
     - Float
     - Loss function weight
     - ``27.211`` (Hamiltonian), ``0.27211`` (band energy)
   * - ``losses[].metric``
     - String
     - Loss calculation method
     - ``mae`` (mean absolute error), optional ``mse`` (mean squared error), ``rmse`` (root mean squared error), ``cosine_similarity``, ``sum_zero``, or ``euclidean_loss``
   * - ``losses[].prediction``
     - String
     - Prediction output
     - ``hamiltonian``, ``band_energy``, or other task-specific targets
   * - ``losses[].target``
     - String
     - Target data
     - ``hamiltonian``, ``band_energy``, ``band_gap``, ``overlap``, ``peak``, ``hamiltonian_imag``, or ``wavefunction``
   * - ``metrics``
     - List
     - List of evaluation metric definitions
     - Usually the same as losses

optim_params (Optimizer Parameters)
------------------------------------

.. list-table::
   :widths: 20 15 50 15
   :header-rows: 1

   * - Parameter
     - Type
     - Description
     - Default/Recommended Value
   * - ``lr``
     - Float
     - Learning rate
     - ``0.01`` (primary training), ``0.0001`` (secondary training)
   * - ``lr_decay``
     - Float
     - Learning rate decay factor
     - ``0.5``
   * - ``lr_patience``
     - Integer
     - Number of epochs to tolerate before triggering learning rate decay
     - ``5``
   * - ``gradient_clip_val``
     - Float
     - Gradient clipping value
     - ``0.0`` (no clipping)
   * - ``max_epochs``
     - Integer
     - Maximum number of training epochs
     - ``3000``
   * - ``min_epochs``
     - Integer
     - Minimum number of training epochs before early stopping can trigger
     - ``100``
   * - ``stop_patience``
     - Integer
     - Number of epochs to tolerate for early stopping
     - ``30``

output_nets.HamGNN_out (Output Network Parameters)
--------------------------------------------------

.. list-table::
   :widths: 20 15 50 15
   :header-rows: 1

   * - Parameter
     - Type
     - Description
     - Default/Recommended Value
   * - ``ham_type``
     - String
     - Type of Hamiltonian to fit
     - ``openmx`` (OpenMX), ``abacus`` (ABACUS), ``siesta`` (SIESTA/HONPAS), or ``pasp``
   * - ``nao_max``
     - Integer
     - Maximum number of orbitals per atom
     - ``14`` (short-period elements), ``19`` (common elements), ``26`` (all OpenMX supported elements); for ABACUS options are ``27`` or ``40``
   * - ``add_H0``
     - Boolean
     - Whether to add H_nonscf to predicted H_scf
     - ``true``
   * - ``add_H_nonsoc``
     - Boolean
     - Add non-SOC Hamiltonian (for SOC-coupled systems)
     - ``false``
   * - ``symmetrize``
     - Boolean
     - Whether to apply Hermitian constraints to Hamiltonian
     - ``true``
   * - ``calculate_band_energy``
     - Boolean
     - Whether to calculate bands for band training
     - ``false`` (primary training), ``true`` (secondary training)
   * - ``num_k``
     - Integer
     - Number of k-points used for band calculation
     - ``5``
   * - ``band_num_control``
     - Integer, Dictionary, or null
     - Control the number of orbitals considered in band calculation
     - ``8`` (VBM±8 bands), ``dict`` (specify number of bases for each atom type), ``null`` (all bands)
   * - ``k_path``
     - List, String, or null
     - k-space path for band structure calculation; ``auto`` = auto-generate based on crystal symmetry
     - ``null`` (generate random k-points)
   * - ``soc_switch``
     - Boolean
     - Whether to fit SOC Hamiltonian
     - ``false``
   * - ``soc_basis``
     - String
     - SOC basis type
     - ``'so3'`` (SO(3)), ``'su2'`` (SU(2))
   * - ``nonlinearity_type``
     - String
     - Type of non-linear activation function
     - ``gate`` (gated nonlinearity) or ``norm`` (norm-based)
   * - ``zero_point_shift``
     - Boolean
     - Whether to apply zero-point potential correction to Hamiltonian matrix
     - ``true``
   * - ``spin_constrained``
     - Boolean
     - Whether to constrain spin
     - ``false``
   * - ``collinear_spin``
     - Boolean
     - Whether for collinear spin
     - ``false``
   * - ``minMagneticMoment``
     - Float
     - Minimum magnetic moment
     - ``0.5``
   * - ``ham_only``
     - Boolean
     - When ``true``, only compute Hamiltonian H; when ``false``, fit both H and overlap S
     - ``true``
   * - ``include_triplet``
     - Boolean
     - Include triplet interaction terms in output
     - ``false``
   * - ``export_reciprocal_values``
     - Boolean
     - Export reciprocal space values during forward pass
     - ``false``
   * - ``use_learned_weight``
     - Boolean
     - Use learned weight factors in the output
     - ``true``
   * - ``get_nonzero_mask_tensor``
     - Boolean
     - Compute nonzero element mask tensor for sparse Hamiltonian
     - ``false``
   * - ``return_forces``
     - Boolean
     - Compute atomic forces during forward pass
     - ``false``
   * - ``create_graph``
     - Boolean
     - Create computational graph for backprop (required for force derivatives)
     - ``false``
   * - ``calculate_sparsity``
     - Boolean
     - Calculate sparsity ratio for loss correction
     - ``true``

representation_nets.HamGNN_pre (Representation Network Parameters)
-------------------------------------------------------------------

.. list-table::
   :widths: 20 15 50 15
   :header-rows: 1

   * - Parameter
     - Type
     - Description
     - Default/Recommended Value
   * - ``cutoff``
     - Float
     - Atomic distance cutoff radius
     - ``26.0``
   * - ``cutoff_func``
     - String
     - Type of distance cutoff function
     - ``cos`` (cosine function), optional ``pol`` (polynomial function)
   * - ``radius_type``
     - String
     - Atomic radius table source
     - ``'openmx'`` (OpenMX radii) or ``'abacus'`` (ABACUS radii)
   * - ``radius_scale``
     - Float
     - Radius scaling factor
     - ``1.01``
   * - ``edge_sh_normalization``
     - String
     - Edge spherical harmonic normalization method
     - ``component``
   * - ``edge_sh_normalize``
     - Boolean
     - Whether to normalize edge spherical harmonics
     - ``true``
   * - ``irreps_edge_sh``
     - String
     - Spherical harmonic representation of edges
     - ``0e + 1o + 2e + 3o + 4e + 5o``
   * - ``irreps_node_features``
     - String
     - O(3) irreducible representation of initial atomic features
     - ``64x0e+64x0o+32x1o+16x1e+12x2o+25x2e+18x3o+9x3e+4x4o+9x4e+4x5o+4x5e+2x6e``
   * - ``num_layers``
     - Integer
     - Number of interaction layers or orbital convolution layers
     - ``3``
   * - ``num_radial``
     - Integer
     - Number of Bessel bases
     - ``64``
   * - ``num_types``
     - Integer
     - Maximum number of atom types
     - ``96``
   * - ``rbf_func``
     - String
     - Type of radial basis function
     - ``bessel`` (also supports ``gaussian``, ``exp-gaussian``, ``exp-bernstein``, ``bernstein``)
   * - ``set_features``
     - Boolean
     - Whether to set features
     - ``true``
   * - ``radial_MLP``
     - List
     - Hidden layer sizes of radial multilayer perceptron
     - ``[64, 64]``
   * - ``use_corr_prod``
     - Boolean
     - Whether to use correlation product
     - ``false``
   * - ``correlation``
     - Integer
     - Correlation parameter
     - ``2``
   * - ``num_hidden_features``
     - Integer
     - Number of hidden features
     - ``16``
   * - ``use_kan``
     - Boolean
     - Whether to use KAN (Kolmogorov-Arnold Network) activation function
     - ``false``
   * - ``build_internal_graph``
     - Boolean
     - Whether to build internal neighbor graph
     - ``false``
   * - ``legacy_edge_update``
     - Boolean
     - Legacy edge update behavior (for old checkpoint compatibility only)
     - ``false``
   * - ``lite_mode``
     - Boolean
     - Minimal parameter mode for faster inference (reduces parameter count)
     - ``false``
   * - ``apply_charge_doping``
     - Boolean
     - Enable charge doping atom embedding for charged defect systems
     - ``false``
   * - ``num_charge_attr_feas``
     - Integer
     - Number of charge attribution Gaussian features (only used if ``apply_charge_doping=True``)
     - ``8``
   * - ``use_gradient_checkpointing``
     - Boolean
     - Enable gradient checkpointing to reduce memory usage during training
     - ``false``

Parameter Adjustment Recommendations
------------------------------------

1. **Initial Configuration**:

   - When using for the first time, it's recommended to first use default parameters for primary training, and then adjust based on results

2. **Learning Rate Adjustment**:

   - For primary training, set initial learning rate to ``0.01``
   - For secondary training, set initial learning rate to ``0.0001``
   - If training is unstable, lower the learning rate; if convergence is too slow, increase the learning rate appropriately

3. **Network Depth Adjustment**:

   - For simple systems, ``num_layers=3`` is usually sufficient
   - For complex systems, try increasing ``num_layers`` to 4-5

4. **Orbital Number Adjustment**:

   - ``nao_max`` needs to be selected based on the maximum number of atomic orbitals in the system
   - Use 14 for short-period elements (such as C, Si, O, etc.)
   - Use 19 for most common elements
   - Use 26 for all elements supported by OpenMX

5. **Cutoff Radius Adjustment**:

   - The default value of ``cutoff`` 26.0 is suitable for most systems; too small a ``cutoff`` sometimes has poor effects
   - Note that the ``cutoff`` here only controls the decay factor of atomic interactions, without affecting the structure of the graph, i.e., not changing the number of edges

6. **Loss Function Weight Adjustment**:

   - In secondary training, the loss function weight for ``band_energy`` is typically set to 0.001~0.01 times that of ``hamiltonian``
   - If band fitting effect is poor, the weight of ``band_energy`` can be increased appropriately, but should not be too large to avoid affecting the prediction accuracy of the Hamiltonian

References
==========

1. `Transferable equivariant graph neural networks for the Hamiltonians of molecules and solids <https://doi.org/10.1038/s41524-023-01130-4>`_
2. `Universal Machine Learning Kohn-Sham Hamiltonian for Materials <https://cpl.iphy.ac.cn/article/10.1088/0256-307X/41/7/077103>`_
3. `A Universal Spin-Orbit-Coupled Hamiltonian Model for Accelerated Quantum Material Discovery <https://arxiv.org/abs/2504.19586>`_
4. `Accelerating the electronic-structure calculation of magnetic systems by equivariant neural networks <https://arxiv.org/abs/2306.01558>`_
5. `GitHub - QuantumLab-ZY/HamGNN <https://github.com/QuantumLab-ZY/HamGNN>`_
