"""
Interfaces for ABACUS DFT software.

This module provides tools for:
    - Converting POSCAR files to ABACUS input format
    - Generating graph_data.npz from ABACUS calculations

Main Scripts:
    - poscar2abacus: Convert structure files to ABACUS format
    - graph_data_gen_abacus: Generate graph data from ABACUS outputs
    - read_abacus: Read ABACUS Hamiltonian files
    - build_graph_from_coordinates: Build graph directly from coordinates

The abacus_H0_export directory contains abacus-postprocess tool.
"""