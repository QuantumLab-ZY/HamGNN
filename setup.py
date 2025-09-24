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
            "HamGNN2.0 = HamGNN_v_2_1.main:HamGNN",
            "band_cal = utils_openmx.band_cal:main",
            "graph_data_gen = utils_openmx.graph_data_gen:main",
            "poscar2openmx = utils_openmx.poscar2openmx:main"
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
        "numba"
    ],
    license="MIT",
    license_files="LICENSE",
    zip_safe=False,
)
