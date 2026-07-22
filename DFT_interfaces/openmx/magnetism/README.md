# Purpose

This package provides the magnetism-oriented OpenMX CLI surface for collinear and non-collinear input generation, XSF spin generation, and graph-data packaging.

# Dependencies

- Python 3 with the project dependencies already used by the OpenMX interfaces.
- `PyYAML` for config loading.
- `read_openmx` on disk when you move beyond dry-run planning for graph-data workflows.
- OpenMX output directories and structure inputs that match the configured glob patterns.

# Workflow overview

The CLI exposes four workflows: `convert-collinear`, `convert-noncollinear`, `make-xsf-spin`, and `pack-graph-data`. All four accept a YAML config with shared `inputs.patterns`, `outputs.directory`, and optional runtime overrides, plus a workflow-specific section for command-specific settings. `convert-collinear`, `convert-noncollinear`, and `make-xsf-spin` write files when `runtime.dry_run` is false. `pack-graph-data` remains a packaging command that reads existing OpenMX outputs and writes `graph_data.npz`.

# `convert-collinear` usage

Example: [examples/convert-collinear.example.yml](examples/convert-collinear.example.yml)

```bash
python -m DFT_interfaces.openmx.magnetism.cli convert-collinear --config DFT_interfaces/openmx/magnetism/examples/convert-collinear.example.yml --dry-run
```

Set `runtime.dry_run: false` to write `.dat` files. Initial spins can be set for a species and then overridden per atom:

```yaml
convert_collinear:
  template: |
    System.Name example
    DATA.PATH /path/to/DFT_DATA19
    scf.XcType GGA-PBE
    scf.SpinPolarization On
    scf.EigenvalueSolver Band
    scf.Kgrid 3 3 3
species:
  overrides:
    Fe:
      spin: [8.0, 6.0]
      spin_constraint: on
atom_spins:
  1: [8.0, 6.0]
  2: [6.0, 8.0]
```

`atom_spins` keys are 1-based atom indices in the same order as the generated OpenMX `Atoms.SpeciesAndCoordinates` block.

# `convert-noncollinear` usage

Example: [examples/convert-noncollinear.example.yml](examples/convert-noncollinear.example.yml)

```bash
python -m DFT_interfaces.openmx.magnetism.cli convert-noncollinear --config DFT_interfaces/openmx/magnetism/examples/convert-noncollinear.example.yml --dry-run
```

Set `runtime.dry_run: false` to write `.dat` files from XSF inputs that contain magnetic vectors in the `PRIMCOORD` rows. The XSF spin vector controls the non-collinear `theta/phi` direction, while `species.overrides` and `atom_spins` control the OpenMX `spin_up/spin_down` values:

```yaml
convert_noncollinear:
  template: |
    System.Name example
    DATA.PATH /path/to/DFT_DATA19
    scf.XcType GGA-PBE
    scf.SpinPolarization NC
    scf.EigenvalueSolver Band
    scf.Kgrid 3 3 3
  nonmagnetic_threshold: 0.01
species:
  overrides:
    Fe:
      spin: [8.0, 8.0]
      spin_constraint: on
atom_spins:
  1: [8.0, 6.0]
  2: [6.0, 8.0]
```

`atom_spins` does not set the 3D non-collinear direction. Use XSF magnetic-vector columns for direction and `atom_spins` for per-atom `[spin_up, spin_down]` values.

# `make-xsf-spin` usage

Example: [examples/make-xsf-spin.example.yml](examples/make-xsf-spin.example.yml)

Explicit magnetic vectors are also supported via [examples/make-xsf-spin-vectors.example.yml](examples/make-xsf-spin-vectors.example.yml).

```bash
python -m DFT_interfaces.openmx.magnetism.cli make-xsf-spin --config DFT_interfaces/openmx/magnetism/examples/make-xsf-spin.example.yml --dry-run
```

Set `runtime.dry_run: false` to write `.xsf` files with magnetic-vector columns in `PRIMCOORD` rows.

# `pack-graph-data` usage

Example: [examples/pack-graph-data-collinear.example.yml](examples/pack-graph-data-collinear.example.yml)

```bash
# Dry-run preflight (recommended first)
python -m DFT_interfaces.openmx.magnetism.cli pack-graph-data \
    --config DFT_interfaces/openmx/magnetism/examples/pack-graph-data-collinear.example.yml \
    --mode collinear --dry-run

# Actual packaging
python -m DFT_interfaces.openmx.magnetism.cli pack-graph-data \
    --config DFT_interfaces/openmx/magnetism/examples/pack-graph-data-collinear.example.yml \
    --mode collinear
```

For collinear packaging the convenience script fixes `--mode=collinear`:

```bash
python DFT_interfaces/openmx/magnetism/graph_data_gen_spin_collinear.py \
    --config DFT_interfaces/openmx/magnetism/examples/pack-graph-data-collinear.example.yml
```

For non-collinear packaging the counterpart fixes `--mode=non_collinear`:

```bash
python DFT_interfaces/openmx/magnetism/graph_data_gen_non_collinear.py \
    --config DFT_interfaces/openmx/magnetism/examples/pack-graph-data-noncollinear.example.yml
```

Several shortcut scripts are also available in the package directory for other workflows:

| Script | Delegates to |
|---|---|
| `poscar2openmx_col.py` | `convert-collinear` |
| `xsf2openmx_spin.py` | `convert-noncollinear` |
| `poscar2xsf.py` | `make-xsf-spin` |

The non-collinear variant is documented in [examples/pack-graph-data-noncollinear.example.yml](examples/pack-graph-data-noncollinear.example.yml) and uses `--mode non_collinear`. Mode selection comes from the CLI flag, not from a YAML key.

The `spin_length` and `spin_vec` graph attributes are automatically parsed from the OpenMX `.out` file (configured via `graph_data.out_file_name`). The parser extracts the per-atom Mulliken population summary — for collinear mode the spin vector is `[0, 0, sign(Up − Down)]` and for non-collinear mode the spherical angles `theta`/`phi` are converted to Cartesian unit vectors. When no `.out` file is found, these attributes fall back to zero.

# Configuration reference

Minimum config shape:

```yaml
inputs:
  patterns:
    - ./structures/*.vasp
outputs:
  directory: ./openmx_outputs
runtime:
  dry_run: true
  skip_errors: false
convert_collinear:
  data_path: /path/to/DFT_DATA19
graph_data:
  read_openmx: /path/to/DFT_DATA19/read_openmx
  nao_max: 19
  dat_file_name: openmx.dat
  std_file_name: openmx.std
  scfout_file_name: system.scfout
  h0_scfout_file_name: overlap.scfout
  out_file_name: openmx.out
  max_scf_iterations: 250
```

- `inputs.patterns`: one or more glob patterns relative to the repo root.
- `outputs.directory`: destination directory for planned outputs.
- `runtime.dry_run`: when false, `convert-collinear` and `convert-noncollinear` write `.dat` files; dry-run remains recommended for the planning-only workflows.
- `runtime.skip_errors`: continue after per-file failures when a workflow supports execution later.
- `runtime.workers`: optional parallel worker count (default: 1). Set to a higher number to process multiple inputs concurrently.
- `convert_collinear`, `convert_noncollinear`, `make_xsf_spin`: workflow-specific sections used to hold command-specific settings in the examples.
- `convert_collinear.template`: OpenMX header and calculation settings prepended before generated species, coordinates, and lattice blocks.
- `convert_noncollinear.template`: OpenMX header and calculation settings prepended before generated species, coordinates, spin angles, and lattice blocks.
- `graph_data.dat_file_name`, `graph_data.std_file_name`: optional explicit input filenames; otherwise the best matching file is discovered automatically.
- `graph_data.scfout_file_name`: optional explicit SCF Hamiltonian filename; `overlap.scfout` is excluded from automatic discovery.
- `graph_data.h0_scfout_file_name`: optional explicit reference-Hamiltonian filename. When set, it is read independently and populates `Hon0/Hoff0` in collinear mode. In non-collinear mode it also populates `iHon0/iHoff0`, and `Lon/Loff` are taken from this payload instead of the main Hamiltonian.
- `graph_data.out_file_name`: optional explicit OpenMX output filename (default: auto-discovered `.out`). The Mulliken population summary in this file is parsed to populate `spin_length` and `spin_vec` in the output graph. If the file is missing or lacks a Mulliken section, spin attributes fall back to zero.
- `graph_data.max_scf_iterations`: optional inclusive maximum SCF iteration count. Samples above this value are skipped or reported as errors according to `runtime.skip_errors`.
- `convert_noncollinear.nonmagnetic_threshold`: vector-norm threshold below which XSF spin vectors are treated as nonmagnetic and assigned zero angles.
- `make_xsf_spin.base_direction`: 3-vector used to generate magnetic vectors for every atom before masking.
- `make_xsf_spin.mask`: optional per-atom scalar mask applied to `base_direction`.
- `make_xsf_spin.vectors`: optional explicit `N x 3` magnetic vectors used instead of `base_direction`/`mask`.
- `species.overrides`: optional per-element PAO, PBE, spin, spin constraint, and basis overrides. Note that `pack-graph-data` does not read spin from this field — `spin_length`/`spin_vec` are extracted from the `.out` file Mulliken analysis instead.
- `atom_spins`: optional 1-based per-atom `[spin_up, spin_down]` overrides for `convert-collinear` and `convert-noncollinear`.
- `graph_data.read_openmx`: path to the `read_openmx` executable or its containing directory.
- `graph_data.nao_max`: basis size selector used by graph-data packing.

# Troubleshooting

- `Unknown element`: your structure contains a symbol that is not covered by the built-in OpenMX defaults. Add a species override in the config before attempting execution.
- `read_openmx`: verify the path points to an executable file, or to a directory that contains `read_openmx`.
- No files matched the configured patterns: check `inputs.patterns` and whether the paths are relative to the repository root.
- Config validation errors: make sure `inputs.patterns` and `outputs.directory` are present, because the CLI requires both to plan work.
- If `pack-graph-data` does not list the files you expect in `--dry-run`, confirm the `--mode` flag matches the example you are using and that the OpenMX output tree exists.
