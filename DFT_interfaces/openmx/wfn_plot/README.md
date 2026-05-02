# wfn_plot — OpenMX Wavefunction Visualization Tool

## Overview

`wfn_plot` is a toolkit for exporting and visualizing wavefunction data from [OpenMX](https://www.openmx-square.org/) DFT calculations. It converts wavefunction coefficients into Gaussian Cube format, which can be visualized with tools like VESTA, VMD, or PyMOL.

The toolkit consists of two components:

| Component | Language | Purpose |
|-----------|----------|---------|
| `wfn_export.py` | Python | Extracts specific band wavefunction coefficients from `eigen_vecs.npy` (output by `band_cal_parallel`) and writes binary `.bin` files |
| `wfn2cube` / `wfn2cube_mpi` | C (serial + MPI+OpenMP) | Reads binary wavefunction files + OpenMX input (`.dat`) + PAO basis files, evaluates wavefunctions on a real-space grid, and outputs `.cube` files |

---

## Installation

### Prerequisites

| Dependency | Required For | Notes |
|------------|-------------|-------|
| **GCC** (or compatible C compiler) | `wfn2cube` (serial) | `gcc >= 4.8` recommended |
| **MPI compiler** (`mpicc`) | `wfn2cube_mpi` (parallel) | OpenMPI or MPICH |
| **OpenMP** (optional) | `wfn2cube_mpi` parallel performance | Usually bundled with GCC |
| **Python 3** | `wfn_export.py` | Python 3.6+ |
| **NumPy** | `wfn_export.py` | `pip install numpy` |
| **PyYAML** | `wfn_export.py` | `pip install pyyaml` |

### Build Steps

```bash
cd DFT_interfaces/openmx/wfn_plot

# Build serial version
make all

# Build MPI+OpenMP parallel version (recommended for large systems)
make mpi

# Run built-in tests (optional)
make test
```

This produces:
- `wfn2cube` — serial executable
- `wfn2cube_mpi` — MPI+OpenMP parallel executable

### Clean

```bash
make clean
```

---

## Usage Workflow

```
+--------------------------+     +----------------------------------------+     +----------------+     +---------------+
|  band_cal_parallel       |---->|  wfn_export.py                         |---->|   wfn2cube     |---->|  .cube file   |
|  (outputs eigen_vecs.npy)|     |  (extract specific band wavefunction)  |     |  (grid eval)   |     |  (visualize)  |
+--------------------------+     +----------------------------------------+     +----------------+     +---------------+
```

### Step 1: Run Band Calculation with `band_cal_parallel`

The `wfn_export.py` script is designed to work with the `eigen_vecs.npy` output from HamGNN's `tools/band_cal_parallel` tool. Run a band structure calculation first:

```bash
band_cal_parallel --config band_cal.yaml
```

This produces `eigen_vecs.npy` containing eigenvector coefficients at each k-point. You also need:
- `openmx.dat` — Original OpenMX input file (used by `wfn2cube` for crystal structure and basis information)
- `eigen_vecs.npy` — Eigenvector output from `band_cal_parallel`

### Step 2: Export Wavefunction Coefficients (`wfn_export.py`)

`wfn_export.py` extracts a specific wavefunction (defined by k-point index and band index) from `eigen_vecs.npy` into a binary `wfn.bin` file that `wfn2cube` can read.

Create a configuration file `wfn_export.yaml`:

#### Single Wavefunction Mode

Export a specific band at a specific k-point:

```yaml
eigen_vecs_path: "/path/to/eigen_vecs.npy"  # Output from band_cal_parallel
latt: [[ax, ay, az], [bx, by, bz], [cx, cy, cz]]  # Lattice vectors in Angstrom
save_dir: "./output"  # Output directory
soc_switch: false  # Set true for spin-orbit coupling
integration: false  # Single mode (export one wavefunction)
k_idx: 0  # Index of k-point in eigen_vecs.npy
wfn_idx: 3  # Band index to export (0-based). Corresponds to `num_wfns` in band_cal_parallel.yaml. If `num_wfns=3`, `wfn_idx=3` exports the HOMO wavefunction.
k_vec: [0.0, 0.0, 0.0]  # k-point in reciprocal coordinates (for cube header)
```

Run:
```bash
python wfn_export.py --config wfn_export.yaml
```

Output: `wfn.bin` (or `wfn_up.bin` + `wfn_down.bin` for SOC)

### Step 3: Convert to Cube Format (`wfn2cube`)

#### Serial Mode

```bash
./wfn2cube <wfn.bin> <openmx.dat> <DFT_DATA_path> [output_prefix]
```

Arguments:
| Argument | Description | Example |
|----------|-------------|---------|
| `wfn.bin` | Binary wavefunction file from step 2 | `output/wfn.bin` |
| `openmx.dat` | OpenMX input file | `example_input/openmx.dat` |
| `DFT_DATA_path` | Path to OpenMX DFT_DATA directory (contains PAO/ and VPS/) | `example_input/DFT_DATA19` |
| `output_prefix` | (Optional) Prefix for output cube files | `wfn` |

Output files:
- `<prefix>_real.cube` — Real part of wavefunction
- `<prefix>_imag.cube` — Imaginary part of wavefunction
- `<prefix>_abs.cube` — Absolute value squared (|ψ|²)

#### Parallel Mode (MPI+OpenMP)

```bash
mpirun -np 8 ./wfn2cube_mpi output/wfn.bin example_input/openmx.dat example_input/DFT_DATA19 wfn_mpi
```

Output files:
- `wfn_mpi_real.cube`
- `wfn_mpi_imag.cube`
- `wfn_mpi_abs.cube`

---

## Complete Example

```bash
cd DFT_interfaces/openmx/wfn_plot

# 1. Build wfn2cube
make all mpi

# 2. Run band structure calculation with band_cal_parallel (prerequisite)
band_cal_parallel --config band_cal.yaml
# This produces eigen_vecs.npy containing eigenvectors

# 3. Export specific band wavefunction to binary format
python wfn_export.py --config wfn_export.yaml
# This reads eigen_vecs.npy and produces output/wfn.bin

# 4. Convert to cube format (serial)
./wfn2cube output/wfn.bin /path/to/openmx.dat /path/to/DFT_DATA19 wfn

# 4b. Or convert to cube format (parallel with 8 MPI ranks)
mpirun -np 8 ./wfn2cube_mpi output/wfn.bin /path/to/openmx.dat /path/to/DFT_DATA19 wfn_mpi

# 5. Visualize the .cube files with VESTA, VMD, etc.
```

---

## File Format Details

### Binary Wavefunction File (`wfn.bin`)

| Offset | Content | Type |
|--------|---------|------|
| 0–23 | k-point (kx, ky, kz) | 3 × float64 |
| 24– | Wavefunction coefficients (Re, Im pairs) | N × 2 × float64 |

### Cube File Format

Standard Gaussian Cube format:
- Line 1-2: Comment/header
- Line 3: Number of atoms + origin
- Lines 4-6: Grid dimensions and step vectors
- Lines 7+: Atomic numbers and positions
- Remaining: Grid data (6 values per line)

---

## Grid Determination

The real-space grid is determined by (in priority order):

1. **Explicit grid**: If `scf.Ngrid1/2/3` are set in the `.dat` file, those values are used
2. **Energy cutoff**: If `scf.energycutoff` is set (in Rydberg), the grid is calculated as `N = ceil(π × lattice_length / √(ecut))`, rounded up to FFT-friendly numbers (factors of 2, 3, 5)

---

## System Limits

| Limit | Value | Defined In |
|-------|-------|------------|
| Maximum species | 64 | `MAX_SPECIES` |
| Maximum atoms | 1024 | `MAX_ATOMS` |
| Maximum angular momentum | L = 6 | `MAX_L` |
| Maximum zeta | 6 | `MAX_MUL` |
| Maximum grid dimension | 3000 | `MAX_MESH` |

For very large systems exceeding these limits, modify the `#define` constants in `wfn2cube.c` and recompile.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Cannot open PAO file` | Verify `DFT_DATA_path` contains the `PAO/` subdirectory with correct `.pao` files |
| `Neither scf.Ngrid nor scf.energycutoff found` | Add either `scf.Ngrid1/2/3` or `scf.energycutoff` to your `.dat` file |
| `No atoms found in dat file` | Check that `<Atoms.SpeciesAndCoordinates>` section exists in `.dat` |
| `Too many species` | Increase `MAX_SPECIES` in `wfn2cube.c` (default: 64) |
| MPI build fails | Ensure `mpicc` is in PATH and OpenMPI/MPICH is installed |
