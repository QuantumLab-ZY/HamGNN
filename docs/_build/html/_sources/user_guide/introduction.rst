===================
HamGNN Introduction
===================

HamGNN (Hamiltonian Graph Neural Network) is an E(3) equivariant Graph Neural Network framework designed specifically for quantum materials simulation. Its core functionality is to train and predict ab initio tight-binding Hamiltonians, which are the fundamental quantum mechanical operators describing the electronic structure of materials.

Key Features
============

- **Wide Physical Scenario Support**: Applicable to molecules, solids, and multi-dimensional material systems with Spin-Orbit Coupling (SOC) effects, handling structures from zero to three dimensions.
- **Multiple DFT Software Compatibility**: Supports integration with mainstream Density Functional Theory software based on Numerical Atomic Orbitals (NAO), including OpenMX, SIESTA/HONPAS, and ABACUS.
- **Theoretical Foundation**: Uses a Graph Neural Network architecture with E(3) rotational and translational equivariance, ensuring that the predicted Hamiltonian matrices satisfy basic physical symmetries.
- **High Fidelity**: Capable of high-precision approximation of Density Functional Theory (DFT) calculation results, with good cross-material structure prediction capabilities.
- **Efficient Computation**: Significantly improves computational efficiency for large-scale systems (such as systems containing thousands of atoms) compared to traditional DFT methods.

Application Areas
=================

HamGNN's applications are primarily focused on high-throughput material design and discovery, large-scale electronic structure calculations, and quantum material property predictions, providing an efficient research tool for materials science, condensed matter physics, and computational chemistry.

Latest Developments
===================

According to recent research progress, HamGNN has been extended to a universal model called Uni-HamGNN, which can predict spin-orbit coupling effects across the periodic table without the need for retraining for new material systems, significantly accelerating the discovery and design process of quantum materials.