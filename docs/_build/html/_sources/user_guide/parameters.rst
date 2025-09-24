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
     - ``HamGNN_pre`` for normal Hamiltonian fitting, ``HamGNN_pre_charge`` for charged defect Hamiltonian fitting
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
     - No default value, must be manually set
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
     - ``null`` (CPU), ``[0]`` (use first GPU)
   * - ``precision``
     - Integer
     - Computation precision
     - ``32`` (32-bit precision), optional ``64`` (64-bit precision)
   * - ``property``
     - String
     - Type of physical quantity output by the network
     - ``Hamiltonian`` (Hamiltonian)
   * - ``stage``
     - String
     - Execution stage
     - ``fit`` (training), ``test`` (testing/prediction)

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
     - ``0.1`` (10%)
   * - ``train_ratio``
     - Float
     - Proportion of training set in the entire dataset
     - ``0.8`` (80%)
   * - ``val_ratio``
     - Float
     - Proportion of validation set in the entire dataset
     - ``0.1`` (10%)
   * - ``graph_data_path``
     - String
     - Directory of processed compressed graph data files
     - No default value, must be manually set

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
     - ``mae`` (mean absolute error), optional ``mse`` (mean squared error) or ``rmse`` (root mean squared error)
   * - ``losses[].prediction``
     - String
     - Prediction output
     - ``Hamiltonian`` or ``band_energy``
   * - ``losses[].target``
     - String
     - Target data
     - ``hamiltonian`` or ``band_energy``
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
     - ``4``
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
     - Minimum number of training epochs
     - ``30``
   * - ``stop_patience``
     - Integer
     - Number of epochs to tolerate for early stopping
     - ``10``

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
     - ``openmx`` (OpenMX Hamiltonian), ``abacus`` (ABACUS Hamiltonian)
   * - ``nao_max``
     - Integer
     - Maximum number of orbitals per atom
     - ``14`` (short-period elements), ``19`` (common elements), ``26`` (all OpenMX supported elements); for ABACUS options are ``27`` or ``40``
   * - ``add_H0``
     - Boolean
     - Whether to add H_nonscf to predicted H_scf
     - ``true``
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
     - ``4``
   * - ``band_num_control``
     - Integer, Dictionary, or null
     - Control the number of orbitals considered in band calculation
     - ``8`` (VBM±8 bands), ``dict`` (specify number of bases for each atom type), ``null`` (all bands)
   * - ``soc_switch``
     - Boolean
     - Whether to fit SOC Hamiltonian
     - ``false``
   * - ``nonlinearity_type``
     - String
     - Type of non-linear activation function
     - ``gate``
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
     - ``bessel``
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
     - Whether to use KAN activation function
     - ``false``
   * - ``radius_scale``
     - Float
     - Radius scaling factor
     - ``1.01``
   * - ``build_internal_graph``
     - Boolean
     - Whether to build internal graph
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