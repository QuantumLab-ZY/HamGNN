<p align="center">
  <img height="130" src="logo/logo.png"/>
</p>

# 🚀 HamGNN v2.1 Now Available! 

Getting Started with HamGNN: [Online Documentation](https://hamgnn.readthedocs.io/en/latest/)

## Table of Contents
- [1. Introduction to HamGNN](#1-introduction-to-hamgnn)
- [2. Environment Configuration Requirements](#2-environment-configuration-requirements)
  - [Python Environment](#python-environment)
  - [Third-party DFT Tool Support](#third-party-dft-tool-support)
  - [Compiling `openmx_postprocess` and `read_openmx`](#compiling-openmx_postprocess-and-read_openmx)
- [3. HamGNN Installation Steps](#3-hamgnn-installation-steps)
  - [Step One: Install Conda Environment](#step-one-install-conda-environment)
  - [Step Two: Install HamGNN from Source](#step-two-install-hamgnn-from-source)
- [4. Construction Method for graph_data.npz Files](#4-construction-method-for-graph_datanpz-files)
  - [General Process](#general-process)
  - [OpenMX Process](#openmx-process)
  - [SIESTA/HONPAS Process](#siestahonpas-process)
  - [ABACUS Process](#abacus-process)
- [5. Model Training Process](#5-model-training-process)
  - [Training Mode Classification](#training-mode-classification)
  - [Training Commands](#training-commands)
  - [Key Configuration Item Descriptions](#key-configuration-item-descriptions)
  - [Training Monitoring](#training-monitoring)
- [6. Model Prediction Operations](#6-model-prediction-operations)
  - [Preparation Before Prediction](#preparation-before-prediction)
  - [Execute Prediction](#execute-prediction)
  - [Output Results](#output-results)
- [7. Band Structure Calculation](#7-band-structure-calculation)
  - [Operation Process](#operation-process)
  - [Band Calculation for Large Systems](#band-calculation-for-large-systems)
- [8. Introduction to and Usage of Uni-HamGNN Universal Model](#8-introduction-to-and-usage-of-uni-hamgnn-universal-model)
  - [Model Introduction](#model-introduction)
  - [Input Requirements](#input-requirements)
  - [Usage Process](#usage-process)
  - [Output Results](#output-results-1)
- [9. HamGNN Parameter Details](#9-hamgnn-parameter-details)
  - [setup (Basic Settings)](#setup-basic-settings)
  - [dataset_params (Dataset Parameters)](#dataset_params-dataset-parameters)
  - [losses_metrics (Loss Functions and Evaluation Metrics)](#losses_metrics-loss-functions-and-evaluation-metrics)
  - [optim_params (Optimizer Parameters)](#optim_params-optimizer-parameters)
  - [output_nets.HamGNN_out (Output Network Parameters)](#output_netshamgnn_out-output-network-parameters)
  - [representation_nets.HamGNN_pre (Representation Network Parameters)](#representation_netshamgnn_pre-representation-network-parameters)
  - [Parameter Adjustment Recommendations](#parameter-adjustment-recommendations)
- [References](#references)
- [Code contributors](#code-contributors)
- [Project leaders](#project-leaders)

## 1. Introduction to HamGNN
HamGNN (Hamiltonian Graph Neural Network) is an E(3) equivariant graph neural network framework designed specifically for quantum materials simulation. Its core functionality is to train and predict ab initio tight-binding Hamiltonians, which are fundamental quantum mechanical operators describing the electronic structure of materials.

Key features of HamGNN include:
- **Support for a wide range of physical scenarios**: Applicable to molecules, solids, and multi-dimensional material systems with Spin-Orbit Coupling (SOC) effects, capable of handling structures from zero to three dimensions.
- **Compatibility with multiple DFT software**: Supports integration with mainstream density functional theory software based on Numerical Atomic Orbitals (NAO), including OpenMX, SIESTA/HONPAS, and ABACUS.
- **Theoretical foundation**: Employs a graph neural network architecture with E(3) rotational and translational equivariance, ensuring the predicted Hamiltonian matrices satisfy fundamental physical symmetries.
- **High fidelity**: Capable of approximating density functional theory (DFT) calculation results with high precision, while maintaining good cross-material structure prediction capabilities.
- **Computational efficiency**: Significantly improves computational efficiency for large-scale systems (such as systems containing thousands of atoms) compared to traditional DFT methods.

HamGNN's application domains primarily focus on high-throughput material design and discovery, large-scale electronic structure calculations, and quantum material property predictions, providing an efficient research tool for materials science, condensed matter physics, and computational chemistry.

According to recent research advances, HamGNN has been extended to a universal model called Uni-HamGNN, which can predict spin-orbit coupling effects across the periodic table without the need for retraining for new material systems, significantly accelerating the discovery and design process of quantum materials.

## 2. Environment Configuration Requirements
### Python Environment
The HamGNN framework recommends Python 3.9 and depends on the following key Python libraries:
- `numpy == 1.21.2`
- `PyTorch == 1.11.0`
- `PyTorch Geometric == 2.0.4`
- `pytorch_lightning == 1.5.10`
- `e3nn == 0.5.0`
- `pymatgen == 2022.3.7`
- `tensorboard == 2.8.0`
- `tqdm`
- `scipy == 1.7.3`
- `yaml`

### Third-party DFT Tool Support
#### OpenMX
- **OpenMX**: HamGNN requires tight-binding Hamiltonians generated by OpenMX. Users should be familiar with basic OpenMX parameter settings and usage methods. Available for download from [OpenMX official website](https://www.openmx-square.org/).
- **openmx_postprocess**: This is a modified version of OpenMX used to parse calculated overlap matrices and other Hamiltonian matrices. It stores computational data in a binary file named `overlap.scfout`.
- **read_openmx**: This is a binary executable used to export matrices from the `overlap.scfout` file to `HS.json`.

#### SIESTA/HONPAS
- **honpas_1.2_H0**: This is a modified version of HONPAS used to parse calculated overlap matrices and non-self-consistent Hamiltonian matrices `H0`, similar to the `openmx_postprocess` tool. The output is a binary file containing Hamiltonian data (`overlap.HSX`).
- **hsxdump**: This is a binary executable that generates intermediate Hamiltonian files, essential for converting HONPAS output to HamGNN-readable format.

#### ABACUS
- **abacus_postprocess**: A tool used to export Hamiltonian matrices `H0`.

### Compiling `openmx_postprocess` and `read_openmx`
To install `openmx_postprocess`:
1. First, install the [GSL](https://www.gnu.org/software/gsl/) library.
2. Modify the `makefile` in the `openmx_postprocess` directory:
   - Set `GSL_lib` to the path of the GSL library.
   - Set `GSL_include` to the include path of GSL.
   - Set `MKLROOT` to the Intel MKL path.
   - Set `CMPLR_ROOT` to the Intel compiler path.
After modifying the `makefile`, execute `make` to generate the executable programs: `openmx_postprocess` and `read_openmx`.

## 3. HamGNN Installation Steps
### Step One: Install Conda Environment
To avoid library version conflicts, it is recommended to use one of the following two methods:

#### Method 1: Use Pre-built Environment (Recommended)
1. Download the pre-built HamGNN Conda environment (`ML.tar.gz`) from [Zenodo](https://zenodo.org/records/11064223)
2. Extract it to the `envs` folder in your Conda installation directory:
   ```bash
   tar -xzvf ML.tar.gz -C $HOME/miniconda3/envs/
   ```
3. Activate the environment:
   ```bash
   conda activate ML
   ```

#### Method 2: Create Environment Using Configuration File
1. Create an environment using the YAML configuration file provided by HamGNN:
   ```bash
   conda env create -f ./HamGNN.yaml
   ```

### Step Two: Install HamGNN from Source
1. Clone the HamGNN repository:
   ```bash
   git clone https://github.com/QuantumLab-ZY/HamGNN.git
   ```
2. Enter the HamGNN directory and execute the installation:
   ```bash
   cd HamGNN
   python setup.py install
   ```
3. Verify that the installation was successful:
   ```bash
   python -c "import HamGNN_v_2_1; print('HamGNN installed successfully')"
   ```
4. To upgrade HamGNN version, first uninstall the old version:
   ```bash
   pip uninstall HamGNN
   ```
   Ensure that related files in the `site-packages` directory (such as `HamGNN-x.x.x-py3.9.egg/HamGNN`) have been completely removed, then reinstall the new version.

## 4. Construction Method for graph_data.npz Files
`graph_data.npz` is the core input format for HamGNN, containing material structure information and Hamiltonian matrix data. Below is an introduction to its construction method.

### General Process
Regardless of which DFT software is used, the basic process for constructing `graph_data.npz` is as follows:
1. **Structure File Conversion**: Convert material structure files (such as POSCAR, CIF) to the input format of the corresponding DFT software
2. **Non-self-consistent Hamiltonian Calculation**: Use post-processing tools to generate files containing non-self-consistent Hamiltonian H0 (matrices that do not depend on self-consistent charge density, such as kinetic energy matrices)
3. **DFT Calculation** (optional): Run a complete DFT calculation to obtain the true Hamiltonian (needed for training set construction, can be skipped for pure prediction)
4. **Data Packaging**: Call the adapted `graph_data_gen` script to package the structure and Hamiltonian matrix data into `graph_data.npz`

### OpenMX Process
1. **Structure Conversion**: Edit the `poscar2openmx.yaml` configuration file, set appropriate paths and DFT parameters (for non-SOC data, set `scf.SpinPolarization: Off` and `scf.SpinOrbit.Coupling: off`; for SOC data, set `scf.SpinPolarization: nc` and `scf.SpinOrbit.Coupling: on`):
    ```yaml
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
    ```
2. **Non-self-consistent Hamiltonian Calculation**: Run `openmx_postprocess` on the generated `.dat` files:
   ```bash
   mpirun -np ncpus ./openmx_postprocess openmx.dat
   ```
   This will generate an `overlap.scfout` file, containing overlap matrix and non-self-consistent Hamiltonian information.
3. **DFT Calculation** (optional, required for training sets):
   ```bash
   mpirun -np ncpus openmx openmx.dat > openmx.std
   ```
   This will generate a `.scfout` file containing self-consistent Hamiltonian.
4. **Data Packaging**: Configure `graph_data_gen.yaml`:
   ```yaml
   nao_max: 26  # Maximum atomic orbital number
   graph_data_save_path: '/path/to/save/graph_data'
   read_openmx_path: '/path/to/HamGNN/utils_openmx/read_openmx'
   max_SCF_skip: 200
   scfout_paths: '/path/to/scfout/files'  # Directory of .scfout files
   dat_file_name: 'openmx.dat'
   std_file_name: 'openmx.std'  # Set to null if no DFT calculation
   scfout_file_name: 'openmx.scfout'  # Use 'overlap.scfout' for prediction
   soc_switch: False  # Set to True for SOC data
   ```
   Then run:
   ```bash
   graph_data_gen --config graph_data_gen.yaml
   ```

### SIESTA/HONPAS Process
1. **Structure Conversion**: Edit the `poscar2siesta.py` script with `poscar_path` and `filepath` parameters, then run:
   ```bash
   python poscar2siesta.py
   ```
2. **DFT Calculation**: Use HONPAS to perform DFT calculations and obtain the `.HSX` Hamiltonian matrix files.
3. **Non-self-consistent Hamiltonian Calculation**: Run the following command to generate non-self-consistent Hamiltonian and overlap matrices:
   ```bash
   mpirun -np Ncores honpas_1.2_H0 < input.fdf
   ```
   This will generate an `overlap.HSX` file.
4. **Data Packaging**: Modify parameters in the `graph_data_gen_siesta.py` script, then run:
   ```bash
   python graph_data_gen_siesta.py
   ```

### ABACUS Process
1. **Structure Conversion**: Use `poscar2abacus.py` to generate ABACUS input files.
2. **DFT Calculation and Post-processing**: Run ABACUS calculations and use `abacus_postprocess` to extract H0 matrices.
3. **Data Packaging**: Use the `graph_data_gen_abacus.py` script to integrate data and generate `graph_data.npz`.

## 5. Model Training Process
HamGNN training typically consists of two phases: primary training (Hamiltonian optimization) and optional secondary training (including band energy optimization).

### Training Mode Classification
1. **Primary Training**:
   - Uses only Hamiltonian loss function to train the model
   - Trains until Hamiltonian error reaches about 10^-5 Hartree
   - If accuracy already meets requirements, secondary training may not be necessary
2. **Secondary Training** (optional):
   - Based on primary training, adds band energy loss function
   - Uses a smaller learning rate to fine-tune the model
   - Improves model performance in band structure prediction

### Training Commands
```bash
HamGNN --config config.yaml > log.out 2>&1
```
For cluster environments, job scheduling systems (such as SLURM) can be used to submit training tasks. Example script:
```bash
#!/bin/bash
#SBATCH --job-name=HamGNN_train
#SBATCH --partition=gpu
#SBATCH --time=999:00:00
#SBATCH --mem=100G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
export OMP_NUM_THREADS=8
source /path/to/your/miniconda3/bin/activate your_env_name
HamGNN --config ./config.yaml > log1.out 2>&1
```

### Key Configuration Item Descriptions
Below are descriptions of key configuration items in `config.yaml`:
1. **dataset_params**:
   ```yaml
   dataset_params:
     batch_size: 1  # Number of samples processed per batch
     test_ratio: 0.1  # Test set ratio
     train_ratio: 0.8  # Training set ratio
     val_ratio: 0.1  # Validation set ratio
     graph_data_path: './Examples/pentadiamond/'  # Path to graph_data.npz file
   ```
2. **losses_metrics**:
   ```yaml
   losses_metrics:
     losses:  # Loss function definition
     - loss_weight: 27.211  # Hamiltonian loss weight
       metric: mae  # Mean absolute error
       prediction: Hamiltonian
       target: hamiltonian
     # Uncomment for secondary training
     #- loss_weight: 0.27211  # Band energy loss weight (typically 0.001~0.01 of Hamiltonian weight)
     #  metric: mae
     #  prediction: band_energy
     #  target: band_energy
     metrics:  # Evaluation metric definition
     - metric: mae
       prediction: hamiltonian
       target: hamiltonian
     # Uncomment for secondary training
     #- metric: mae
     #  prediction: band_energy
     #  target: band_energy
   ```
3. **optim_params**:
   ```yaml
   optim_params:
     lr: 0.01  # Learning rate (recommended 0.01 for primary training, 0.0001 for secondary training)
     lr_decay: 0.5  # Learning rate decay rate
     lr_patience: 4  # Number of epochs to wait before adjusting learning rate
     gradient_clip_val: 0.0  # Gradient clipping value
     max_epochs: 3000  # Maximum number of training epochs
     min_epochs: 30  # Minimum number of training epochs
     stop_patience: 10  # Number of epochs to wait for early stopping
   ```
4. **setup**:
   ```yaml
   setup:
     GNN_Net: HamGNN_pre  # Type of network to use
     accelerator: null  # Accelerator type
     ignore_warnings: true  # Whether to ignore warnings
     checkpoint_path: /path/to/ckpt  # Checkpoint path
     load_from_checkpoint: false  # Whether to load model parameters from checkpoint
     resume: false  # Whether to continue training from interruption
     num_gpus: [0]  # GPU device numbers to use, null indicates CPU
     precision: 32  # Computation precision (32 or 64 bit)
     property: Hamiltonian  # Type of physical quantity output
     stage: fit  # Stage: fit (training) or test (testing)
   ```
5. **output_nets**:
   ```yaml
   output_nets:
     output_module: HamGNN_out
     HamGNN_out:
       ham_type: openmx  # Type of Hamiltonian to fit: openmx or abacus
       nao_max: 19  # Maximum atomic orbital number (14/19/26 for openmx)
       add_H0: true  # Whether to add non-self-consistent Hamiltonian
       symmetrize: true  # Whether to apply Hermitian constraints to Hamiltonian
       calculate_band_energy: false  # Whether to calculate bands (set to true for secondary training)
       #soc_switch: false  # Whether to fit SOC Hamiltonian
       # Parameters used in secondary training
       #num_k: 4  # Number of k-points used for band calculation
       #band_num_control: 8  # Number of orbitals considered in band calculation
       #k_path: null # Generate random k-points
   ```

### Training Monitoring
Use TensorBoard to monitor the training process:
```bash
tensorboard --logdir train_dir --port=6006
```
When training on a remote server, you can access TensorBoard through an Xshell tunnel:
1. In Xshell, click "Server->Properties->Tunneling", add a new tunnel
2. Set source host to localhost, port to 16006
3. Set target host to localhost, target port to 6006
4. Access http://localhost:16006/ in your browser to view training progress

## 6. Model Prediction Operations
After completing model training, you can use the trained model to predict Hamiltonians for new structures.

### Preparation Before Prediction
1. **Construct graph_data.npz for structures to be predicted**:
   - Convert structures to the corresponding DFT software format
   - Use post-processing tools (such as `openmx_postprocess`) to generate `overlap.scfout`
   - Configure `graph_data_gen.yaml`, set `scfout_file_name` to `'overlap.scfout'`
   - Run `graph_data_gen` to generate `graph_data.npz`
2. **Edit configuration file**: Modify the following parameters in `config.yaml`:
   ```yaml
   setup:
     checkpoint_path: /path/to/trained/model.ckpt  # Path to trained model
     num_gpus: null  # Set to null or 0 to use CPU for prediction
     stage: test  # Set to test for prediction mode
   ```
3. **Set environment variables**: If running on CPU, you can accelerate with multithreading:
   ```bash
   export OMP_NUM_THREADS=64  # Set number of threads
   ```

### Execute Prediction
```bash
HamGNN --config config.yaml > predict.log 2>&1
```
For large systems or batch predictions, it is recommended to use a job scheduling system to submit tasks:
```bash
#!/bin/bash
#SBATCH --partition=compute
#SBATCH --job-name=HamGNN_predict
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
export OMP_NUM_THREADS=64
source /path/to/your/miniconda3/bin/activate your_env_name
HamGNN --config ./config.yaml > predict.log 2>&1
```

### Output Results
After prediction is completed, the following files will be generated in the directory specified by `train_dir`:
- **prediction_hamiltonian.npy**: Predicted Hamiltonian matrix
- **target_hamiltonian.npy**: If the input `graph_data.npz` contains the real Hamiltonian matrix, this will be the input real value

## 7. Band Structure Calculation
Band structure calculation is used to verify the accuracy of predicted Hamiltonians, checking whether they can correctly reproduce the electronic band structure of materials.

### Operation Process
1. **Edit configuration file**: Create a `band_cal.yaml` file, specifying the following parameters:
    ```yaml
    nao_max: 26
    graph_data_path: '/path/to/graph_data.npz' # Path to graph_data.npz
    hamiltonian_path: '/path/to/prediction_hamiltonian.npy'  # Path to Hamiltonian matrix
    nk: 120          # the number of k points
    save_dir: '/path/to/save/band/calculation/result' # The directory to save the results
    strcture_name: 'Si'  # The name of each cif file saved is strcture_name_idx.cif after band calculation
    soc_switch: False
    spin_colinear: False
    auto_mode: True # If the auto_mode is used, users can omit providing k_path and label, as the program will automatically generate them based on the crystal symmetry.
    k_path: [[0.,0.,-0.5],[0.,0.,0.0],[0.,0.,0.5]] # High symmetry point path
    label: ['$Mbar$','$G$','$M$'] # The lable for each k points in K_path
    ```
2. **Run calculation**:
   ```bash
   band_cal --config band_cal.yaml
   ```

### Band Calculation for Large Systems
For large systems, a parallel version of the band calculation tool `band_cal_parallel` can be used:
1. **Install parallel tools**:
   ```bash
   pip install mpitool-0.0.1-cp39-cp39-manylinux1_x86_64.whl
   pip install band_cal_parallel-0.1.12-py3-none-any.whl
   ```
2. **Run parallel calculation**:
   ```bash
   mpirun -np N_CORES band_cal_parallel --config band_cal_parallel.yaml
   ```
Note: Some MKL environments may encounter the `Intel MKL FATAL ERROR: Cannot load symbol MKLMPI_Get_wrappers` error. Refer to GitHub Issues [#18](https://github.com/QuantumLab-ZY/HamGNN/issues/18) and [#12](https://github.com/QuantumLab-ZY/HamGNN/issues/12) for solutions.

## 8. Introduction to and Usage of Uni-HamGNN Universal Model
### Model Introduction
Uni-HamGNN is a universal spin-orbit coupling Hamiltonian model designed to accelerate quantum material discovery. The model addresses the major challenge of modeling spin-orbit coupling (SOC) effects in various complex systems, which traditionally requires computationally expensive density functional theory (DFT) calculations.

Uni-HamGNN eliminates the need for system-specific retraining and costly SOC-DFT calculations, making high-throughput screening of quantum materials possible across systems of different dimensions. This makes it a powerful tool for quantum material design and property research, significantly accelerating the pace of discovery in condensed matter physics and materials science.

### Input Requirements
The universal SOC Hamiltonian model requires two `graph_data.npz` files as input data:
- One file in non-SOC mode
- One file in SOC mode
The preparation method for these files is the same as described in previous sections, but note the differences between SOC and non-SOC parameters.

### Usage Process
1. **Prepare input data**:
   - Convert structures to be predicted into two `graph_data.npz` files (non-SOC and SOC modes)
   - Place these files in the specified directories
2. **Configure parameters**:
   Edit the `Input.yaml` configuration file:
   ```yaml
   # HamGNN prediction configuration
   model_pkl_path: '/path/to/universal_model.pkl'
   non_soc_data_dir: '/path/to/non_soc_graph_data'
   soc_data_dir: '/path/to/soc_graph_data'  # Optional, only needed for SOC calculations
   output_dir: './results'
   device: 'cuda'  # Use GPU, fall back to CPU if not available
   calculate_mae: true  # Calculate mean absolute error
   ```
3. **Run prediction**:
   ```bash
   python Uni-HamiltonianPredictor.py --config Input.yaml
   ```

### Output Results
After execution, the script will generate the following files in the specified `output_dir`:
- `hamiltonian.npy`: Predicted Hamiltonian matrix in NumPy array format
- If `calculate_mae` is enabled, MAE statistics will be printed to the console

## 9. HamGNN Parameter Details
This section provides detailed explanations of the parameter modules and parameters in the `config.yaml` configuration file.

### setup (Basic Settings)
| Parameter | Type | Description | Default/Recommended Value |
|-----|-----|------|-------------|
| `GNN_Net` | string | Type of GNN network to use | `HamGNN_pre` for normal Hamiltonian fitting, `HamGNN_pre_charge` for charged defect Hamiltonian fitting |
| `accelerator` | null or string | Accelerator type | `null` |
| `ignore_warnings` | boolean | Whether to ignore warnings | `true` |
| `checkpoint_path` | string | Checkpoint path for resuming training or for testing | No default value, must be set manually |
| `load_from_checkpoint` | boolean | Whether to load model parameters from checkpoint | `false` (for new training), `true` (when loading pre-trained model) |
| `resume` | boolean | Whether to continue training from last interruption | `false` (for new training), `true` (to continue training) |
| `num_gpus` | null, integer, or list | Number or ID of GPUs to use | `null` (CPU), `[0]` (use first GPU) |
| `precision` | integer | Computation precision | `32` (32-bit precision), optional `64` (64-bit precision) |
| `property` | string | Type of physical quantity output by the network | `Hamiltonian` (Hamiltonian) |
| `stage` | string | Execution stage | `fit` (training), `test` (testing/prediction) |

### dataset_params (Dataset Parameters)
| Parameter | Type | Description | Default/Recommended Value |
|-----|-----|------|-------------|
| `batch_size` | integer | Number of samples processed per batch | `1` (typically kept at 1) |
| `test_ratio` | float | Proportion of test set in the entire dataset | `0.1` (10%) |
| `train_ratio` | float | Proportion of training set in the entire dataset | `0.8` (80%) |
| `val_ratio` | float | Proportion of validation set in the entire dataset | `0.1` (10%) |
| `graph_data_path` | string | Directory of processed compressed graph data files | No default value, must be set manually |

### losses_metrics (Loss Functions and Evaluation Metrics)
| Parameter | Type | Description | Default/Recommended Value |
|-----|-----|------|-------------|
| `losses` | list | List of loss function definitions | Must include at least Hamiltonian loss |
| `losses[].loss_weight` | float | Loss function weight | `27.211` (Hamiltonian), `0.27211` (band energy) |
| `losses[].metric` | string | Loss calculation method | `mae` (mean absolute error), optional `mse` (mean squared error) or `rmse` (root mean squared error) |
| `losses[].prediction` | string | Prediction output | `Hamiltonian` or `band_energy` |
| `losses[].target` | string | Target data | `hamiltonian` or `band_energy` |
| `metrics` | list | List of evaluation metric definitions | Usually the same as losses |

### optim_params (Optimizer Parameters)
| Parameter | Type | Description | Default/Recommended Value |
|-----|-----|------|-------------|
| `lr` | float | Learning rate | `0.01` (primary training), `0.0001` (secondary training) |
| `lr_decay` | float | Learning rate decay factor | `0.5` |
| `lr_patience` | integer | Number of epochs to wait before triggering learning rate decay | `4` |
| `gradient_clip_val` | float | Gradient clipping value | `0.0` (no clipping) |
| `max_epochs` | integer | Maximum number of training epochs | `3000` |
| `min_epochs` | integer | Minimum number of training epochs | `30` |
| `stop_patience` | integer | Number of epochs to wait for early stopping | `10` |

### output_nets.HamGNN_out (Output Network Parameters)
| Parameter | Type | Description | Default/Recommended Value |
|-----|-----|------|-------------|
| `ham_type` | string | Type of Hamiltonian to fit | `openmx` (OpenMX Hamiltonian), `abacus` (ABACUS Hamiltonian) |
| `nao_max` | integer | Maximum number of atomic orbitals | `14` (short-period elements), `19` (common elements), `26` (all elements supported by OpenMX); for ABACUS, `27` or `40` are options |
| `add_H0` | boolean | Whether to add predicted H_scf to H_nonscf | `true` |
| `symmetrize` | boolean | Whether to apply Hermitian constraints to Hamiltonian | `true` |
| `calculate_band_energy` | boolean | Whether to calculate bands for band training | `false` (primary training), `true` (secondary training) |
| `num_k` | integer | Number of k-points used for band calculation | `4` |
| `band_num_control` | integer, dictionary, or null | Controls the number of orbitals considered in band calculation | `8` (VBM±8 bands), `dict` (specifies basis number for each atom type), `null` (all bands) |
| `soc_switch` | boolean | Whether to fit SOC Hamiltonian | `false` |
| `nonlinearity_type` | string | Type of non-linear activation function | `gate` |
| `zero_point_shift` | boolean | Whether to apply zero-point potential correction to Hamiltonian matrix | `true` |
| `spin_constrained` | boolean | Whether to constrain spin | `false` |
| `collinear_spin` | boolean | Whether it is collinear spin | `false` |
| `minMagneticMoment` | float | Minimum magnetic moment | `0.5` |

### representation_nets.HamGNN_pre (Representation Network Parameters)
| Parameter | Type | Description | Default/Recommended Value |
|-----|-----|------|-------------|
| `cutoff` | float | Cutoff radius for interatomic distances | `26.0` |
| `cutoff_func` | string | Type of distance cutoff function | `cos` (cosine function), optional `pol` (polynomial function) |
| `edge_sh_normalization` | string | Normalization method for edge spherical harmonics | `component` |
| `edge_sh_normalize` | boolean | Whether to normalize edge spherical harmonics | `true` |
| `irreps_edge_sh` | string | Spherical harmonic representation of edges | `0e + 1o + 2e + 3o + 4e + 5o` |
| `irreps_node_features` | string | O(3) irreducible representation of initial atomic features | `64x0e+64x0o+32x1o+16x1e+12x2o+25x2e+18x3o+9x3e+4x4o+9x4e+4x5o+4x5e+2x6e` |
| `num_layers` | integer | Number of interaction or orbital convolution layers | `3` |
| `num_radial` | integer | Number of Bessel bases | `64` |
| `num_types` | integer | Maximum number of atom types | `96` |
| `rbf_func` | string | Type of radial basis function | `bessel` |
| `set_features` | boolean | Whether to set features | `true` |
| `radial_MLP` | list | Hidden layer sizes of the radial multilayer perceptron | `[64, 64]` |
| `use_corr_prod` | boolean | Whether to use correlation product | `false` |
| `correlation` | integer | Correlation parameter | `2` |
| `num_hidden_features` | integer | Number of hidden features | `16` |
| `use_kan` | boolean | Whether to use KAN activation function | `false` |
| `radius_scale` | float | Radius scaling factor | `1.01` |
| `build_internal_graph` | boolean | Whether to build internal graph | `false` |

### Parameter Adjustment Recommendations
1. **Initial Configuration**:
   - When using for the first time, it is recommended to first use default parameters for primary training, then adjust based on needs after observing the effect
2. **Learning Rate Adjustment**:
   - For primary training, set initial learning rate to `0.01`
   - For secondary training, set initial learning rate to `0.0001`
   - If training is unstable, lower the learning rate; if convergence is too slow, appropriately increase the learning rate
3. **Network Depth Adjustment**:
   - For simple systems, `num_layers=3` is usually sufficient
   - For complex systems, try increasing `num_layers` to 4-5
4. **Orbital Number Adjustment**:
   - `nao_max` needs to be chosen based on the maximum atomic orbital number in the system
   - Use 14 for short-period elements (such as C, Si, O, etc.)
   - Use 19 for most common elements
   - Use 26 for all elements supported by OpenMX
5. **Cutoff Radius Adjustment**:
   - The default value of `cutoff` 26.0 is suitable for most systems, too small a `cutoff` sometimes has poor effect
   - Note that `cutoff` here only controls the decay factor of interatomic interactions, not affecting the structure of the graph, i.e., not changing the number of edges
6. **Loss Function Weight Adjustment**:
   - In secondary training, the loss function weight for `band_energy` is typically set to 0.001~0.01 times that of `hamiltonian`
   - If band fitting effect is poor, the weight of `band_energy` can be appropriately increased, but should not be too large to avoid affecting the prediction accuracy of the Hamiltonian
7. **Minimum Irreps for Node and Edge Features in config.yaml**：
    - The different atomic orbital basis sets used to expand the Hamiltonian matrix may require different combinations of $l$ in the equivariant features, and the number of channels may also vary. If the basis set contains f orbitals, the maximum value of $l$ in the equivariant feature is as high as 6. The minimum equivariant feature settings can be determined through the following code. For example, for the $ssppd$ basis set, the minimum irreducible representations are `17x0e+20x1o+8x1e+8x2o+20x2e+8x3o+4x3e+4x4e`:
        ```
        from e3nn import o3

        row=col=o3.Irreps("1x0e+1x0e+1x0e+1x1o+1x1o+1x2e+1x2e") # for 'sssppd'
        ham_irreps_dim = []
        ham_irreps = o3.Irreps()

        for _, li in row:
            for _, lj in col:
                for L in range(abs(li.l-lj.l), li.l+lj.l+1):
                    ham_irreps += o3.Irrep(L, (-1)**(li.l+lj.l)) 

        print(ham_irreps.sort()[0].simplify())
        ```
        ```
        Output: 17x0e+20x1o+8x1e+8x2o+20x2e+8x3o+4x3e+4x4e
        ```

## References
The papers related to HamGNN:
1. [Transferable equivariant graph neural networks for the Hamiltonians of molecules and solids](https://doi.org/10.1038/s41524-023-01130-4)
2. [Universal Machine Learning Kohn-Sham Hamiltonian for Materials](https://cpl.iphy.ac.cn/article/10.1088/0256-307X/41/7/077103)
3. [A Universal Spin-Orbit-Coupled Hamiltonian Model for Accelerated Quantum Material Discovery](https://arxiv.org/abs/2504.19586)
4. [Accelerating the electronic-structure calculation of magnetic systems by equivariant neural networks](https://arxiv.org/abs/2306.01558)
5. [Topological interfacial states in ferroelectric domain walls of two-dimensional bismuth](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.111.075407)
6. [Transferable Machine Learning Approach for Predicting Electronic Structures of Charged Defects](https://pubs.aip.org/aip/apl/article-abstract/126/4/044103/3332348/Transferable-machine-learning-approach-for?redirectedFrom=fulltext)
7. [Advancing nonadiabatic molecular dynamics simulations in solids with E(3) equivariant deep neural hamiltonians](https://www.nature.com/articles/s41467-025-57328-1)
8. [Revealing higher-order topological bulk-boundary correspondence in bismuth crystal with spin-helical hinge state loop and proximity superconductivity](https://www.sciencedirect.com/science/article/abs/pii/S2095927325008783)
9. [Silicon Nanowire Gate‐All‐Around Cold Source MOSFET With Ultralow Power Dissipation: A Machine‐Learning‐Hamiltonian Accelerated Design](https://advanced.onlinelibrary.wiley.com/doi/10.1002/adfm.202513807)

## Code contributors:
+ Yang Zhong (Fudan University)
+ Changwei Zhang (Fudan University)
+ Zhenxing Dai (Fudan University)
+ Shixu Liu (Fudan University)
+ Hongyu Yu (Fudan University)
+ Yuxing Ma (Fudan University)

## Project leaders: 
+ Hongjun Xiang  (Fudan University)
+ Xingao Gong  (Fudan University)

