"""
Interfaces for OpenMX DFT software.

This module provides tools for:
    - Converting POSCAR/CIF files to OpenMX input format
    - Generating graph_data.npz from OpenMX calculations
    - Band structure calculations using OpenMX Hamiltonians

Main Scripts:
    - poscar2openmx: Convert structure files to OpenMX format
    - graph_data_gen: Generate graph data from OpenMX outputs
    - band_cal: Calculate band structures

The openmx_postprocess directory contains modified OpenMX source code
for parsing overlap matrices and Hamiltonian data.
"""