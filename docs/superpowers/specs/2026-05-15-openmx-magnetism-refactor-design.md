# OpenMX Magnetism Refactor Design

## Goal

Improve the Python scripts under `DFT_interfaces/openmx/magnetism/` so ordinary users can configure and run OpenMX magnetic-structure conversion and magnetic Hamiltonian graph-data packaging workflows without editing source code. The refactor should also make the code easier to read, maintain, test, and extend.

## Current Problems

The current magnetism scripts mix user parameters, file discovery, OpenMX text generation, OpenMX output parsing, graph assembly, and batch execution in the same files. Important parameters are hard-coded in Python source, including absolute paths, input globs, element defaults, spin patterns, `read_openmx` locations, `nao_max`, and OpenMX template blocks. Users have no clear entry point, no README, and no examples that explain what to edit or how to run each workflow.

The affected workflows are:

- `poscar2openmx_col.py`: POSCAR/CIF to collinear OpenMX `.dat` input.
- `xsf2openmx_spin.py`: XSF spin-vector input to non-collinear OpenMX `.dat` input.
- `poscar2xsf.py`: POSCAR/CIF to XSF with magnetic vectors.
- `graph_data_gen_spin_collinear.py`: collinear OpenMX outputs to `graph_data.npz`.
- `graph_data_gen_non_collinear.py`: non-collinear/SOC OpenMX outputs to `graph_data.npz`.

Backward compatibility with old source-edited entry behavior is not required, as long as the existing functionality is covered by clearer workflows.

## Chosen Approach

Use a medium refactor: keep the magnetism tools in `DFT_interfaces/openmx/magnetism/`, introduce clearer CLI entry points, move shared logic into focused modules, and make YAML configuration the primary user interface with CLI overrides for common parameters.

This is preferred over a thin wrapper because it removes the main maintainability problems instead of hiding them. It is also preferred over a full package redesign because the requested scope is focused on this tool directory and should not require broad packaging or install changes.

## Architecture

The directory should become a small, documented toolset rather than a collection of source-edited research scripts. User-facing commands parse YAML and CLI arguments, validate inputs, print useful progress and errors, and call pure or mostly-pure core functions.

Proposed shared modules:

- `config.py`: load YAML, merge CLI overrides, validate required fields, and report user-facing configuration errors.
- `defaults.py`: define built-in PAO, PBE, spin, and basis defaults, plus helper functions for applying element overrides.
- `spin.py`: parse XSF spin vectors, create configurable spin patterns, and convert spin vectors to OpenMX magnetic moments and spherical angles.
- `openmx_input.py`: generate collinear and non-collinear OpenMX `.dat` text from structures, species defaults, spin settings, and OpenMX template settings.
- `graph_data.py`: parse OpenMX output artifacts and `HS.json`, build collinear or non-collinear PyTorch Geometric `Data` objects, assemble basis masks, and save `graph_data.npz`.
- `io_utils.py`: discover input files, create output paths safely, run external commands via `subprocess.run(..., check=True)`, and manage temporary or specified working directories for `HS.json`.

The design has two clear subflows:

- Structure conversion: read structures and configuration, then write `.dat` or `.xsf`; it does not run OpenMX.
- Graph-data packaging: read completed OpenMX output files, call `read_openmx`, consume `HS.json`, then write `graph_data.npz`.

## User-Facing Workflows

Provide four documented commands or entry scripts:

- `convert-collinear`: convert POSCAR/CIF inputs into collinear OpenMX `.dat` files.
- `convert-noncollinear`: convert XSF or spin-bearing structure inputs into non-collinear/SOC OpenMX `.dat` files.
- `make-xsf-spin`: create XSF files from POSCAR/CIF inputs with configurable magnetic vectors, replacing the current hard-coded MBT mask behavior.
- `pack-graph-data`: package OpenMX outputs into `graph_data.npz`, with `mode: collinear` or `mode: non_collinear`.

YAML is the complete configuration surface. CLI overrides should cover common batch-run edits:

- `--input`
- `--output-dir`
- `--read-openmx`
- `--nao-max`
- `--workers`
- `--dry-run`
- `--skip-errors`

## Configuration Model

YAML files should use these top-level sections:

- `inputs`: input globs, explicit files, or directories.
- `outputs`: output directory, output filename behavior, and overwrite policy.
- `openmx`: OpenMX `system_name`, OpenMX data path, SCF settings, K-grid settings, and optional template fields.
- `species`: PAO/PBE/spin/basis defaults with per-element overrides.
- `spin`: collinear or non-collinear moment source, constraints, masks, and generated spin-pattern settings.
- `graph_data`: `mode`, `nao_max`, basis selection, `read_openmx` path, SCF threshold, and output filename.

Every parameter that users currently need to edit in source should move into YAML or documented built-in defaults. Unknown elements should fail with a clear message telling users to add the missing PAO/PBE/spin/basis information under `species.overrides`.

## Data Flow

For structure conversion commands:

1. Load YAML and apply CLI overrides.
2. Validate input files, output directory, OpenMX settings, species defaults, and spin settings.
3. Read each structure file.
4. Resolve per-element PAO/PBE/spin/basis configuration.
5. Generate `.dat` or `.xsf` text.
6. Write outputs according to the configured naming and overwrite policy.

For graph-data packaging:

1. Load YAML and apply CLI overrides.
2. Validate OpenMX output file discovery, `read_openmx`, `nao_max`, basis definitions, output path, and graph mode.
3. Match required `.std`, `.dat`, `.out`, and `.scfout` files per system.
4. Reject or skip entries whose SCF error exceeds the configured threshold.
5. Run `read_openmx` using `subprocess.run(..., check=True)` in an isolated temporary directory or configured working directory.
6. Read and validate `HS.json` without relying on current-working-directory side effects.
7. Parse collinear or non-collinear spin and Hamiltonian data.
8. Build graph objects and save `graph_data.npz`.

## Error Handling And UX

Commands should preflight before batch processing:

- Check that input globs match files.
- Check that output directories are writable.
- Check that all structure elements have required defaults or overrides.
- Check that the OpenMX template/configuration is sufficient.
- Check that `read_openmx` exists and is executable for graph packaging.
- Check that `nao_max` and requested basis masks are supported.

Failures should not be silently swallowed. By default, commands stop on the first actionable error and show the file path, failure category, and fix suggestion. Batch processing can continue only when users pass `--skip-errors`; in that mode, the command should summarize skipped files and reasons at the end.

Graph-data packaging should classify common failures:

- Missing `.std`, `.dat`, `.out`, or `.scfout` file.
- SCF error above threshold.
- Spin parse failure.
- `read_openmx` execution failure.
- Invalid or missing `HS.json` schema.
- Missing basis mask for the configured `nao_max` or element.

Each command should print the number of planned inputs, output location, important parameters, progress, generated count, and failed/skipped count. `--dry-run` should parse and validate configuration, list planned inputs and outputs, and perform no writes.

## Testing Strategy

Testing should validate the refactor without requiring real OpenMX calculations.

Required checks:

- Import checks for all new modules and CLI entry points.
- `--help` checks for `convert-collinear`, `convert-noncollinear`, `make-xsf-spin`, and `pack-graph-data`.
- Config parsing tests for required fields, defaults, CLI overrides, and unknown-element errors.
- Unit tests for XSF spin-vector parsing.
- Unit tests for Cartesian spin vector to OpenMX magnetic moment and angle conversion.
- Unit tests for collinear and non-collinear OpenMX `.dat` text generation from minimal structures.
- Unit tests for species/default override resolution.
- Unit tests for basis mask assembly.
- Minimal mock-I/O tests for graph packaging using small fake `.std/.dat/.out/.scfout` inputs and mocked `HS.json` data.

Normal tests should not require the actual `read_openmx` executable or OpenMX binaries. An integration-style path may be documented or skipped unless a user supplies a real executable.

## Documentation

Create `DFT_interfaces/openmx/magnetism/README.md` as the main user guide, English first.

The README should include:

- What each workflow does and when to use it.
- Required Python dependencies and external OpenMX/`read_openmx` dependency.
- Input and output files for each command.
- A minimal YAML example for each workflow.
- Copyable command examples.
- A configuration reference table explaining each key, default, and when ordinary users should change it.
- Troubleshooting entries for no input matched, unknown element, missing basis, missing OpenMX files, failed `read_openmx`, SCF threshold skip, and spin parse failure.

Add example config files under `DFT_interfaces/openmx/magnetism/examples/` so users have concrete files to edit instead of editing Python source.

## Scope Boundaries

In scope:

- Refactor the five magnetism workflows listed above.
- Introduce shared modules inside `DFT_interfaces/openmx/magnetism/`.
- Add YAML-driven configuration and common CLI overrides.
- Add README and example config files.
- Add tests for configuration, pure functions, CLI help, and mock graph packaging.

Out of scope:

- Running real OpenMX calculations.
- Requiring old hard-coded source-edited entry behavior to keep working.
- Reorganizing the broader `DFT_interfaces/openmx/` package outside what the magnetism workflows need.
- Adding a full installable Python package or global console-script distribution unless existing project conventions already make that minimal.

## Acceptance Criteria

The implementation is complete when:

- Users can run each workflow from a documented command using YAML configuration.
- Common parameters can be overridden from the CLI without editing source files.
- The previous hard-coded absolute paths and source-edited parameters are represented in YAML, documented defaults, or explicit user-provided overrides.
- Structure conversion writes expected `.dat` or `.xsf` outputs in dry-run and mock/minimal test paths.
- Graph packaging can be tested with mocked `HS.json` and minimal OpenMX-like artifacts without real OpenMX.
- Errors are actionable and no longer silently skipped by default.
- `DFT_interfaces/openmx/magnetism/README.md` and example configs explain the workflows clearly enough for ordinary users to start from examples.
- The agreed medium validation suite passes.
