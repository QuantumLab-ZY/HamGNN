======================
Model Training Process
======================

HamGNN training is typically divided into two phases: primary training (Hamiltonian optimization) and optional secondary training (including band energy optimization).

Training Mode Classification
============================

1. **Primary Training**:
   
   - Only uses Hamiltonian loss function to train the model
   - Trains until Hamiltonian error reaches approximately 10^-5 Hartree
   - Secondary training may not be necessary if accuracy already meets requirements

2. **Secondary Training** (Optional):
   
   - Based on primary training, adds band energy loss function
   - Uses a smaller learning rate to fine-tune the model
   - Improves model performance in band structure prediction

Training Command
================

.. code-block:: bash

   HamGNN --config config.yaml > log.out 2>&1

For cluster environments, you can use job scheduling systems (like SLURM) to submit training tasks, example script:

.. code-block:: bash

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

Key Configuration Items
=======================

Here's an introduction to key configuration items in ``config.yaml``:

1. **dataset_params**:

   .. code-block:: yaml

      dataset_params:
        batch_size: 1  # Number of samples processed per batch
        test_ratio: 0.1  # Test set ratio
        train_ratio: 0.8  # Training set ratio
        val_ratio: 0.1  # Validation set ratio
        graph_data_path: './Examples/pentadiamond/'  # Path to graph_data.npz file

2. **losses_metrics**:

   .. code-block:: yaml

      losses_metrics:
        losses:  # Loss function definition
        - loss_weight: 27.211  # Hamiltonian loss weight
          metric: mae  # Mean absolute error
          prediction: Hamiltonian
          target: hamiltonian
        # Uncomment the following for secondary training
        #- loss_weight: 0.27211  # Band energy loss weight (typically 0.001~0.01 of Hamiltonian weight)
        #  metric: mae
        #  prediction: band_energy
        #  target: band_energy
        metrics:  # Evaluation metric definition
        - metric: mae
          prediction: hamiltonian
          target: hamiltonian
        # Uncomment the following for secondary training
        #- metric: mae
        #  prediction: band_energy
        #  target: band_energy

3. **optim_params**:

   .. code-block:: yaml

      optim_params:
        lr: 0.01  # Learning rate (recommend 0.01 for primary training, 0.0001 for secondary training)
        lr_decay: 0.5  # Learning rate decay rate
        lr_patience: 4  # Number of epochs to wait before adjusting learning rate
        gradient_clip_val: 0.0  # Gradient clipping value
        max_epochs: 3000  # Maximum number of training epochs
        min_epochs: 30  # Minimum number of training epochs
        stop_patience: 10  # Number of epochs to wait for early stopping

4. **setup**:

   .. code-block:: yaml

      setup:
        GNN_Net: HamGNN_pre  # Type of network to use
        accelerator: null  # Accelerator type
        ignore_warnings: true  # Whether to ignore warnings
        checkpoint_path: /path/to/ckpt  # Checkpoint path
        load_from_checkpoint: false  # Whether to load model parameters from checkpoint
        resume: false  # Whether to continue training from interruption
        num_gpus: [0]  # GPU device numbers to use, null indicates using CPU
        precision: 32  # Computation precision (32 or 64 bit)
        property: Hamiltonian  # Type of physical quantity output
        stage: fit  # Stage: fit (training) or test (testing)

5. **output_nets**:

   .. code-block:: yaml

      output_nets:
        output_module: HamGNN_out
        HamGNN_out:
          ham_type: openmx  # Type of Hamiltonian to fit: openmx or abacus
          nao_max: 19  # Maximum number of atomic orbitals (14/19/26 for openmx)
          add_H0: true  # Whether to add non-self-consistent Hamiltonian
          symmetrize: true  # Whether to apply Hermitian constraints to Hamiltonian
          calculate_band_energy: false  # Whether to calculate bands (set to true for secondary training)
          #soc_switch: false  # Whether to fit SOC Hamiltonian
          # The following parameters are used in secondary training
          #num_k: 4  # Number of k-points used for band calculation
          #band_num_control: 8  # Number of orbitals considered in band calculation
          #k_path: null # Generate random k-points

Training Monitoring
===================

Use TensorBoard to monitor the training process:

.. code-block:: bash

   tensorboard --logdir train_dir --port=6006

When training on a remote server, you can access TensorBoard through an Xshell tunnel:

1. In Xshell, click "Server->Properties->Tunneling" and add a new tunnel
2. Select localhost as the source host, set the port to 16006
3. Select localhost as the destination host, set the destination port to 6006
4. Access http://localhost:16006/ in your browser to view training progress