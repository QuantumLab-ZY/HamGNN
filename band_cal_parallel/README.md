# The script for calculating the band structures for large systems
## Installation
pip install mpitool-0.0.1-cp39-cp39-manylinux1_x86_64.whl

pip install band_cal_parallel-0.1.12-py3-none-any.whl

## Usage
In the Python environment with `band_cal_parallel` installed, execute the following command with multiple cpus to compute the band structure:
mpirun -np ncpus band_cal_parallel --config band_cal_parallel.yaml