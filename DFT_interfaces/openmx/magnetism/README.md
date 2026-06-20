# Purpose

This package provides the magnetism-oriented OpenMX CLI surface for collinear and non-collinear input generation, planning conversions, and graph-data packaging. XSF spin generation and graph-data packaging remain dry-run oriented unless noted below.

# Dependencies

- Python 3 with the project dependencies already used by the OpenMX interfaces.
- `PyYAML` for config loading.
- `read_openmx` on disk when you move beyond dry-run planning for graph-data workflows.
- OpenMX output directories and structure inputs that match the configured glob patterns.

# Workflow overview

The CLI exposes four workflows: `convert-collinear`, `convert-noncollinear`, `make-xsf-spin`, and `pack-graph-data`. All four accept a YAML config with shared `inputs.patterns`, `outputs.directory`, and optional runtime overrides, plus a workflow-specific section for command-specific settings. `convert-collinear` and `convert-noncollinear` write OpenMX `.dat` files when `runtime.dry_run` is false. The other commands only print planned work when `--dry-run` is set.

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

```bash
python -m DFT_interfaces.openmx.magnetism.cli make-xsf-spin --config DFT_interfaces/openmx/magnetism/examples/make-xsf-spin.example.yml --dry-run
```

# `pack-graph-data` usage

Example: [examples/pack-graph-data-collinear.example.yml](examples/pack-graph-data-collinear.example.yml)

```bash
python -m DFT_interfaces.openmx.magnetism.cli pack-graph-data --config DFT_interfaces/openmx/magnetism/examples/pack-graph-data-collinear.example.yml --mode collinear --dry-run
```

The non-collinear variant is documented in [examples/pack-graph-data-noncollinear.example.yml](examples/pack-graph-data-noncollinear.example.yml) and uses `--mode non_collinear`. In the current Task 7 CLI, mode selection comes from the CLI flag, not from a YAML key.

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
```

- `inputs.patterns`: one or more glob patterns relative to the repo root.
- `outputs.directory`: destination directory for planned outputs.
- `runtime.dry_run`: when false, `convert-collinear` and `convert-noncollinear` write `.dat` files; dry-run remains recommended for the planning-only workflows.
- `runtime.skip_errors`: continue after per-file failures when a workflow supports execution later.
- `convert_collinear`, `convert_noncollinear`, `make_xsf_spin`: workflow-specific sections used to hold command-specific settings in the examples.
- `convert_collinear.template`: OpenMX header and calculation settings prepended before generated species, coordinates, and lattice blocks.
- `convert_noncollinear.template`: OpenMX header and calculation settings prepended before generated species, coordinates, spin angles, and lattice blocks.
- `convert_noncollinear.nonmagnetic_threshold`: vector-norm threshold below which XSF spin vectors are treated as nonmagnetic and assigned zero angles.
- `species.overrides`: optional per-element PAO, PBE, spin, spin constraint, and basis overrides.
- `atom_spins`: optional 1-based per-atom `[spin_up, spin_down]` overrides for `convert-collinear` and `convert-noncollinear`.
- `graph_data.read_openmx`: path to the `read_openmx` executable or its containing directory.
- `graph_data.nao_max`: basis size selector used by graph-data packing.

# Troubleshooting

- `Unknown element`: your structure contains a symbol that is not covered by the built-in OpenMX defaults. Add a species override in the config before attempting execution.
- `read_openmx`: verify the path points to an executable file, or to a directory that contains `read_openmx`.
- No files matched the configured patterns: check `inputs.patterns` and whether the paths are relative to the repository root.
- Config validation errors: make sure `inputs.patterns` and `outputs.directory` are present, because the CLI requires both to plan work.
- If `pack-graph-data` does not list the files you expect in `--dry-run`, confirm the `--mode` flag matches the example you are using and that the OpenMX output tree exists.
