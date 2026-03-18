"""
HamGNN - Hamiltonian Graph Neural Network

An E(3) equivariant graph neural network framework for quantum materials simulation.
Train and predict ab initio tight-binding Hamiltonians for molecules and solids.

Main Components:
    - models: GNN model architectures (HamGNN_pre, HamGNN_out)
    - nn: Neural network components (convolution, attention, tensor products)
    - physics: Physical operations (k-points, matrix operations)
    - toolbox: Additional tools (MACE, NequIP, KAN implementations)
    - utils: Utility functions (basis functions, cutoffs, activations)
    - config: Configuration parsing
    - data: Data processing

Usage:
    import hamgnn
    from hamgnn.main import Model

Website: https://github.com/QuantumLab-ZY/HamGNN
Documentation: https://hamgnn.readthedocs.io
"""