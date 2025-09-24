==========================
Model Prediction Operation
==========================

After completing model training, you can use the trained model to predict Hamiltonians for new structures.

Preparation for Prediction
==========================

1. **Construct graph_data.npz for structures to be predicted**:
   
   - Convert structures to the corresponding DFT software format
   - Use post-processing tools (such as ``openmx_postprocess``) to generate ``overlap.scfout``
   - Configure ``graph_data_gen.yaml``, setting ``scfout_file_name`` to ``'overlap.scfout'``
   - Run ``graph_data_gen`` to generate ``graph_data.npz``

2. **Edit configuration file**: Modify the following parameters in ``config.yaml``:

   .. code-block:: yaml

      setup:
        checkpoint_path: /path/to/trained/model.ckpt  # Path to trained model
        num_gpus: null  # Set to null or 0 to use CPU for prediction
        stage: test  # Set to test for prediction mode

3. **Set environment variables**: For CPU execution, you can accelerate with multi-threading:

   .. code-block:: bash

      export OMP_NUM_THREADS=64  # Set number of threads

Execute Prediction
==================

.. code-block:: bash

   HamGNN --config config.yaml > predict.log 2>&1

For large systems or batch predictions, it's recommended to submit tasks using a job scheduling system:

.. code-block:: bash

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

Output Results
==============

After prediction completion, the following files will be generated in the directory specified by ``train_dir``:

- **prediction_hamiltonian.npy**: Predicted Hamiltonian matrix in NumPy array format
- **target_hamiltonian.npy**: If the input ``graph_data.npz`` contains the actual Hamiltonian matrix, this will be the true value from the input