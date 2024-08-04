
import os.path
import codecs
from setuptools import setup, find_packages


setup(
    name="HamGNN",
    version='0.1.0',
    description="Hamiltonian prediction via Graph Neural Network",
    download_url="",
    author="Yang Zhong",
    python_requires=">=3.9",
    packages=find_packages(),
    package_dir={},
    package_data={'': ['*.npz', '*.json'],},
    entry_points={
        "console_scripts": [
            "HamGNN = HamGNN.main:HamGNN",
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
