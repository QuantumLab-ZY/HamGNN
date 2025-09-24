.. HamGNN_v_2_1 documentation master file

HamGNN v2.1 Documentation
=========================

Welcome to HamGNN v2.1 documentation!

Introduction to HamGNN
----------------------

HamGNN is an E(3) equivariant graph neural network designed to train and predict ab initio tight-binding (TB) Hamiltonians for molecules and solids. It can be used with common ab initio DFT software that rely on numerical atomic orbitals, such as OpenMX, Siesta, and ABACUS. Additionally, it supports predictions of SU(2) equivariant Hamiltonians with spin-orbit coupling effects. HamGNN provides a high-fidelity approximation of DFT results and offers transferable predictions across material structures. This makes it ideal for high-throughput electronic structure calculations, accelerating computations on large-scale systems.

.. toctree::
   :maxdepth: 3
   :caption: Contents:

   user_guide/index
   api/model_structure 
   api/gnn_core
   api/model_components
   api/data_processing
   api/configuration
   api/utilities

Other
-----

Tutorials, etc (todo)

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`