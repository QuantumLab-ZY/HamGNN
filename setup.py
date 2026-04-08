'''
Descripttion: 
version: 
Author: Yang Zhong
Date: 2024-10-29 21:19:36
LastEditors: Yang Zhong
LastEditTime: 2025-09-24 00:15:49
'''

import os.path
import codecs
from setuptools import setup, find_packages


setup(
    name="HamGNN",
    version='2.1.0',
    description="Hamiltonian prediction via Graph Neural Network",
    download_url="",
    author="Yang Zhong",
    python_requires=">=3.8",
    packages=find_packages(),
    package_dir={},
    package_data={'': ['*.npz', '*.json'],},
    entry_points={
        "console_scripts": [
            "HamGNN2.0 = hamgnn.main:HamGNN",
            "band_cal = hamgnn.interfaces.openmx.band_cal:main",
            "graph_data_gen = hamgnn.interfaces.openmx.graph_data_gen:main",
            "poscar2openmx = hamgnn.interfaces.openmx.poscar2openmx:main",
            "npz_to_lmdb = hamgnn.tools.npz_to_lmdb:main"
        ]
    },
    install_requires=[
        "numpy",
        "torch",
        "torch_geometric",
        "e3nn",
        "pymatgen",
        "tqdm",
        "tensorboard",
        "natsort",
        "numba",
        "lmdb"
    ],
    license="MIT",
    license_files="LICENSE",
    zip_safe=False,
)
