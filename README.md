# HamGNN (Hamiltonian prediction via Graph Neural Network)

## Introduction to HamGNN
The HamGNN model is an E(3) equivariant graph neural network designed for the purpose of training and predicting tight-binding (TB) Hamiltonians of molecules and solids. Currently, HamGNN can be used in common ab initio DFT software that is based on numerical atomic orbitals, such as OpenMX, Siesta, and Abacus. HamGNN supports predictions of SU(2) equivariant Hamiltonians with spin-orbit coupling effects. HamGNN not only achieves a high fidelity approximation of DFT but also enables transferable predictions across material structures, making it suitable for high-throughput electronic structure calculations and accelerating computations on large-scale systems.

## Requirements

The following environments and packages are required to use HamGNN:

### Python libraries
We recommend using the Python 3.9 interpreter. HamGNN needs the following python libraries:
- NumPy
- PyTorch = 1.11.0
- PyTorch Geometric = 2.0.4
- pytorch_lightning = 1.5.10
- e3nn = 0.5.0
- pymatgen
- TensorBoard
- tqdm
- scipy
- yaml

### openmx_postprocess
openmx_postprocess is a modified OpenMX package used for computing overlap matrices and other Hamiltonian matrices that can be calculated analytically. The data computed by openmx_postprocess will be stored in a binary file `overlap.scfout`. The installation and usage of openmx_postprocess is essentially the same as that of OpenMX. To install openmx_postprocess, you need to install the [GSL](https://www.gnu.org/software/gsl/) library first.Then enter the openmx_postprocess directory and modify the following parameters in the makefile:
+ `GSL_lib`: The lib path of GSL
+ `GSL_include`: The include path of GSL
+ `MKLROOT`: The intel MKL path
+ `CMPLR_ROOT`: The path where the intel compiler is installed

After modifying the makefile, you can directly execute the make command to generate two executable programs, `openmx_postprocess` and `read_openmx`.

### read_openmx
read_openmx is a binary executable that can be used to export the matrices from the binary file overlap.scfout to a file called `HS.json`.

## Installation
Run the following command to install HamGNN:
```bash
git clone git@github.com:XXXXXXX.git
cd HamGNN
python setup.py install
```

## Usage
### Preparation of Hamiltonian Training Data:
First, generate a set of structure files (POSCAR or CIF files) using molecular dynamics or random perturbation. After setting the appropriate path parameters in the `poscar2openmx.py` file,
run `python poscar2openmx.py` to convert these structures into OpenMX's `.dat` file format. Run OpenMX to perform static calculations on these structure files and obtain the `.scfout` binary files, which store the Hamiltonian and overlap matrix information for each structure. These files serve as the target Hamiltonians during training. Next, run `openmx_postprocess` to process each structure and obtain the `overlap.scfout` file, which contains the Hamiltonian matrix H0 that is independent of the self-consistent charge density. If the constructed dataset is only used for prediction purposes and not for training (i.e., no target Hamiltonian is needed), run `openmx_postprocess` to obtain the `overlap.scfout` file merely. `openmx_postprocess` is executed similarly to OpenMX and supports MPI parallelism.

### Graph Data Conversion:
After setting the appropriate path information in a `graph_data_gen.yaml` file, run `graph_data_gen --config graph_data_gen.yaml` to package the structural information and Hamiltonian data from all `.scfout` files into a single `graph_data.npz` file, which serves as the input data for the HamGNN network.

### HamGNN Network Training and Prediction:
Prepare the `config.yaml` configuration file and set the network parameters, training parameters, and other details in this file. To run HamGNN, simply enter `HamGNN --config config.yaml`. Running `tensorboard --logdir train_dir` allows real-time monitoring of the training progress, where `train_dir` is the folder where HamGNN saves the training data, corresponding to the `train_dir` parameter in `config.yaml`. To enhance the transferability and prediction accuracy of the network, the training is divided into two steps. The first step involves training with only the loss value of the Hamiltonian in the loss function until the Hamiltonian training converges or the error reaches around 10^-5 Hartree, at which point the training can be stopped. Then, the band energy error is added to the loss function, and the network parameters obtained from the previous step are loaded for further training. After obtaining the final network parameters, the network can be used for prediction. First, convert the structures to be predicted into the input data format (`graph_data.npz`) for the network, following similar steps and procedures as preparing the training set. Then, in the `config.yaml` file, set the `checkpoint_path` to the path of the network parameter file and set the `stage` parameter to `test`. After configuring the parameters in `config.yaml`, running `HamGNN --config config.yaml` will perform the prediction.

### Band Structure Calculation:
Set the parameters in band_cal.yaml, mainly the path to the Hamiltonian data, then run `band_cal --config band_cal.yaml`

##  How to set the options in config.yaml
The input parameters in config.yaml are divided into different modules, which mainly include `'setup'`, `'dataset_params'`, `'losses_metrics'`, `'optim_params'` and network-related parameters (`'HamGNN_pre'` and `'HamGNN_out'`). Most of the parameters work well using the default values. The following introduces some commonly used parameters in each module.
+ `setup`:
    + `stage`: Select the state of the network: training (`fit`) or testing (`test`).
    + `property`：Select the type of physical quantity to be output by the network, generally set to `hamiltonian`
    + `num_gpus`: number of gpus to train on (`int`) or which GPUs to train on (`list` or `str`) applied per node.
    + `resume`: resume training (`true`) or start from scratch (`false`).
    + `checkpoint_path`: Path of the checkpoint from which training is resumed (`stage` = `fit`) or path to the checkpoint you wish to test (`stage` = `test`).
+ `dataset_params`:
    + `graph_data_path`: The directory where the processed compressed graph data files (`grah_data.npz`) are stored.
    + `batch_size`: The number of samples or data points that are processed together in a single forward and backward pass during the training of a neural network. defaut: 1. 
    + `train_ratio`: The proportion of the training samples in the entire data set.
    + `val_ratio`: The proportion of the validation samples in the entire data set.
    + `test_ratio`：The proportion of the test samples in the entire data set.
+ `losses_metrics`：
    + `losses`: define multiple loss functions and their respective weights in the total loss value. Currently, HamGNN supports `mse`, `mae`, and `rmse`. 
    + `metrics`：A variety of metric functions can be defined to evaluate the accuracy of the model on the validation set and test set.
+ `optim_params`：
    + `min_epochs`: Force training for at least these many epochs.
    + `max_epochs`: Stop training once this number of epochs is reached.
    + `lr`：learning rate, the default value is 0.001.

+ `profiler_params`:
    + `train_dir`: The folder for saving training information and prediction results. This directory can be read by tensorboard to monitor the training process.

+ `HamGNN_pre`: The representation network to generate the node and pair interaction features
    + `num_types`：The maximum number of atomic types used to build the one-hot vectors for atoms
    + `cutoff`: The cutoff radius adopted in the envelope function for interatomic distances.
    + `cutoff_func`: which envelope function is used for interatomic distances. Options: `cos` refers to cosine envelope function, `pol` refers to the polynomial envelope function.
    + `rbf_func`: The radial basis function type used to expand the interatomic distances
    + `num_radial`: The number of Bessel basis.
    + `num_interaction_layers`: The number of interaction layers or orbital convolution layers.
    + `add_edge_tp`: Whether to utilize the tensor product of i and j to construct pair interaction features. This option requires a significant amount of memory, but it can sometimes improve accuracy.
    + `irreps_edge_sh`: Spherical harmonic representation of the orientation of an edge
    + `irreps_node_features`: O(3) irreducible representations of the initial atomic features
    + `irreps_edge_output`: O(3) irreducible representations of the edge features to output
    + `irreps_node_output`: O(3) irreducible representations of the atomic features to output
    + `feature_irreps_hidden`: intermediate O(3) irreducible representations of the atomic features in convelution
    + `irreps_triplet_output(deprecated)`: O(3) irreducible representations of the triplet features to output
    + `invariant_layers`: The layers of the MLP used to map the invariant edge embeddings to the weights of each tensor product path
    + `invariant_neurons`: The number of the neurons of the MLP used to map the invariant edge embeddings to the weights of each tensor product path

+ `HamGNN_out`: The output layer to transform the representation of crystals into Hamiltonian matrix
    + `nao_max`: It is modified according to the maximum number of atomic orbitals in the data set, which can be `14`, `19`
    + `add_H0`: Generally true, the complete Hamiltonian is predicted as the sum of H_scf plus H_nonscf (H0)
    + `symmetrize`：if set to true, the Hermitian symmetry constraint is imposed on the Hamiltonian
    + `calculate_band_energy`: Whether to calculate the energy bands to train the model 
    + `num_k`: When calculating the energy bands, the number of K points to use
    + `band_num_control`: `dict`: controls how many orbitals are considered for each atom in energy bands; `int`: [vbm-num, vbm+num]; `null`: all bands
    + `k_path`: `auto`: Automatically determine the k-point path; `null`: random k-point path; `list`: list of k-point paths provided by the user
    + `soc_switch`: if true, Fit the SOC Hamiltonian
    + `nonlinearity_type`: `norm` activation or `gate` activation as the nonlinear activation function
