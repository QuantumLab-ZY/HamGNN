# wfn_plot — OpenMX Wavefunction Visualization Tool

## Overview

`wfn_plot` is a toolkit for exporting and visualizing wavefunction data from [OpenMX](https://www.openmx-square.org/) DFT calculations. It converts wavefunction coefficients into Gaussian Cube format, which can be visualized with tools like VESTA, VMD, or PyMOL.

The toolkit consists of two components:

| Component | Language | Purpose |
|-----------|----------|---------|
| `wfn_export.py` | Python | Extracts specific band wavefunction coefficients from `eigen_vecs.npy` (output by `band_cal_parallel`) and writes binary `.bin` files |
| `wfn2cube` / `wfn2cube_mpi` | C (serial + MPI+OpenMP) | Reads one wavefunction from a binary file plus an OpenMX input (`.dat`) and its PAO basis files, evaluates it on a real-space grid, and outputs `.cube` files |

---

## Installation

### Prerequisites

| Dependency | Required For | Notes |
|------------|-------------|-------|
| **GCC** (or compatible C compiler) | `wfn2cube` (serial) | `gcc >= 4.8` recommended |
| **MPI compiler** (`mpicc`) | `wfn2cube_mpi` (parallel) | OpenMPI or MPICH |
| **OpenMP** | `wfn2cube_mpi` | The Makefile builds the MPI target with `-fopenmp` |
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
mpirun -np <ncpus> band_cal_parallel --config band_cal_parallel.yaml
```

This produces `eigen_vecs.npy` containing eigenvector coefficients at each k-point. You also need:
- `openmx.dat` — Original OpenMX input file. Its `DATA.PATH` setting must point to an OpenMX DFT data directory containing `PAO/`.
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
wfn_idx: 3  # 0-based index along eigen_vecs.npy axis 1
k_vec: [0.0, 0.0, 0.0]  # Input k-point; transformed by inv(latt).T before being written
```

Run:
```bash
python wfn_export.py --config wfn_export.yaml
```

Output: `wfn.bin` (or `wfn_up.bin` + `wfn_down.bin` for SOC).

`latt` is converted from Angstrom to Bohr before the reciprocal-space k-vector is calculated. `wfn_idx` is a direct array index; it is not automatically mapped to HOMO/VBM from `num_wfns`. The converted k-vector is stored in the binary file and printed by `wfn2cube`, but is not written to the Cube header.

For SOC output, run `wfn2cube` separately for `wfn_up.bin` and `wfn_down.bin`.

#### Integration Mode

When `integration: true`, the script additionally requires `wfn_min`, `wfn_max`, and `k_vecs_path`, and concatenates all selected bands for all k-points into one binary file. The current `wfn2cube` reader consumes only the first record in a file, so integration-mode output is **not directly supported by `wfn2cube`**. Use `integration: false` for the conversion workflow documented below.

### Step 3: Convert to Cube Format (`wfn2cube`)

#### Serial Mode

```bash
./wfn2cube <openmx.dat> <wfn.bin> [output.cube]
```

Arguments:
| Argument | Description | Example |
|----------|-------------|---------|
| `openmx.dat` | OpenMX input file | `example_input/openmx.dat` |
| `wfn.bin` | Single-wavefunction binary file from step 2 | `output/wfn.bin` |
| `output.cube` | Optional output base name; the `.cube` suffix is removed before adding component suffixes | `wfn.cube` |

The DFT data path is not a command-line argument. `wfn2cube` reads it from `DATA.PATH` in `openmx.dat` and loads PAO files from `<DATA.PATH>/PAO/`. A relative `DATA.PATH` is resolved relative to the process working directory.

Output files:
- `<prefix>_real.cube` — Real part of wavefunction
- `<prefix>_imag.cube` — Imaginary part of wavefunction
- `<prefix>_abs.cube` — Absolute value squared (|ψ|²)

#### Parallel Mode (MPI+OpenMP)

```bash
mpirun -np 8 ./wfn2cube_mpi example_input/openmx.dat output/wfn.bin wfn_mpi.cube
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
mpirun -np 8 band_cal_parallel --config band_cal_parallel.yaml
# This produces eigen_vecs.npy containing eigenvectors

# 3. Export specific band wavefunction to binary format
python wfn_export.py --config wfn_export.yaml
# This reads eigen_vecs.npy and produces output/wfn.bin

# 4. Convert to cube format (serial)
./wfn2cube /path/to/openmx.dat output/wfn.bin wfn.cube

# 4b. Or convert to cube format (parallel with 8 MPI ranks)
mpirun -np 8 ./wfn2cube_mpi /path/to/openmx.dat output/wfn.bin wfn_mpi.cube

# 5. Visualize the .cube files with VESTA, VMD, etc.
```

---

## File Format Details

### Binary Wavefunction File (`wfn.bin`)

| Offset | Content | Type |
|--------|---------|------|
| 0-23 | Converted k-point (kx, ky, kz) | 3 x native-endian float64 |
| 24- | Wavefunction coefficients as interleaved (Re, Im) pairs | N x 2 x native-endian float64 |

The file has no magic number, dimensions, or orbital count. `wfn2cube` derives the expected coefficient count from `openmx.dat` and the PAO files, so the binary file and OpenMX input must describe the same orbital basis. Extra records, such as those written by integration mode, are ignored after the first wavefunction.

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

1. **Explicit grid**: If all of `scf.Ngrid1`, `scf.Ngrid2`, and `scf.Ngrid3` are nonzero in the `.dat` file, those values are used.
2. **Energy cutoff**: Otherwise, if `scf.energycutoff` is set in Rydberg, each dimension is calculated as `N = ceil(lattice_length_Bohr × sqrt(ecut) / π)`, then increased to the next FFT-friendly integer whose prime factors are only 2, 3, and 5.

The generated Cube dimensions are `(Ngrid1 + 1) x (Ngrid2 + 1) x (Ngrid3 + 1)`. The extra point includes the periodic boundary endpoint, while each step vector is the corresponding lattice vector divided by `Ngrid`.

---

## System Limits

| Limit or constant | Value | Actual behavior |
|-------------------|-------|-----------------|
| Maximum cached species | 64 (`MAX_SPECIES`) | Enforced with an error |
| Maximum stored atoms | 6000 (`MAX_ATOMS`) | Additional atom records are currently ignored |
| Angular-index array bound | L = 6 (`MAX_L`) | Used for per-atom zeta storage; not a hard PAO parser limit |
| Declared multiplicity constant | 6 (`MAX_MUL`) | Currently not enforced at runtime |
| Declared radial-mesh constant | 3000 (`MAX_MESH`) | Currently not enforced at runtime |

`AngularF` is implemented only for L = 0 through 3. L = 4 through 6 currently emit a warning and contribute zero to the evaluated wavefunction; higher values also contribute zero. Supporting those channels requires implementing their angular functions; increasing `MAX_L` alone is not sufficient.

For systems exceeding an enforced compile-time limit, modify the corresponding `#define` constant in `wfn2cube.c` and recompile. The PAO loader allocates from the file's `PAO.Mul` and `grid.num.output` values, so `MAX_MUL` and `MAX_MESH` do not currently protect those allocations. `MAX_MESH` is not a real-space Cube grid-dimension limit.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Cannot open PAO file` | Verify that `DATA.PATH` in `openmx.dat` resolves to a directory containing the correct files under `PAO/` |
| `Neither scf.Ngrid nor scf.energycutoff found` | Set all three of `scf.Ngrid1/2/3`, or set a positive `scf.energycutoff`, in the `.dat` file |
| `No atoms found in dat file` | Check that `<Atoms.SpeciesAndCoordinates>` section exists in `.dat` |
| `Too many species` | Increase `MAX_SPECIES` in `wfn2cube.c` (default: 64) |
| MPI build fails | Ensure `mpicc` is in PATH and OpenMPI/MPICH is installed |
