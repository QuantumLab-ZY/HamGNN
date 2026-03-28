"""
Interfaces - DFT Software Interface Package

A collection of interfaces for external electronic structure software used with
HamGNN. Convert simulation inputs and outputs from supported packages into graph
representations and Hamiltonian data for training and prediction workflows.

Main Components:
    - abacus: ABACUS interface for input conversion and graph generation
    - openmx: OpenMX interface for input conversion, postprocessing, and band calculations
    - siesta: SIESTA/HONPAS interface for input conversion and graph generation

Usage:
    import interfaces
    from interfaces.abacus import poscar2abacus
    from interfaces.openmx import graph_data_gen

Website: https://github.com/QuantumLab-ZY/HamGNN
Documentation: https://hamgnn.readthedocs.io
"""
