from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
from copy import deepcopy
from pathlib import Path

from ase.io import read as read_ase
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from DFT_interfaces.openmx.magnetism.config import ConfigError, load_config, merge_cli_overrides
from DFT_interfaces.openmx.magnetism.defaults import resolve_species_settings
from DFT_interfaces.openmx.magnetism.graph_data import (
    _parse_dat_text,
    build_collinear_graph,
    build_non_collinear_graph,
    save_graphs_npz,
)
from DFT_interfaces.openmx.magnetism.io_utils import (
    discover_files,
    ensure_output_dir,
    output_path_for,
    resolve_read_openmx,
    run_read_openmx,
)
from DFT_interfaces.openmx.magnetism.openmx_input import (
    build_collinear_dat_text,
    build_xsf_text,
    build_noncollinear_dat_text,
    write_text,
)
from DFT_interfaces.openmx.magnetism.spin import generate_spin_vectors, read_xsf_spin, spin_to_spherical


def add_common_options(parser):
    """Attach shared config and override options to a subcommand parser."""
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--input", dest="input_pattern", help="Override input file or glob pattern.")
    parser.add_argument("--output-dir", help="Override output directory.")
    parser.add_argument("--read-openmx", help="Override read_openmx executable path.")
    parser.add_argument("--nao-max", type=int, help="Override nao_max setting.")
    parser.add_argument("--workers", type=int, help="Override worker count.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=None,
        help="Validate config and print planned work.",
    )
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        default=None,
        help="Continue after per-file errors.",
    )


def build_parser():
    """Return the top-level argparse parser with all magnetism subcommands."""
    parser = argparse.ArgumentParser(prog="python -m DFT_interfaces.openmx.magnetism.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert_collinear = subparsers.add_parser(
        "convert-collinear",
        help="Convert POSCAR/CIF inputs to collinear OpenMX .dat files.",
    )
    add_common_options(convert_collinear)
    convert_collinear.set_defaults(func=command_convert_collinear)

    convert_noncollinear = subparsers.add_parser(
        "convert-noncollinear",
        help="Convert XSF spin inputs to non-collinear OpenMX .dat files.",
    )
    add_common_options(convert_noncollinear)
    convert_noncollinear.set_defaults(func=command_convert_noncollinear)

    make_xsf_spin = subparsers.add_parser(
        "make-xsf-spin",
        help="Convert POSCAR/CIF inputs to XSF-with-spin files.",
    )
    add_common_options(make_xsf_spin)
    make_xsf_spin.set_defaults(func=command_make_xsf_spin)

    pack_graph_data = subparsers.add_parser(
        "pack-graph-data",
        help="Package OpenMX outputs into graph_data.npz.",
    )
    add_common_options(pack_graph_data)
    pack_graph_data.add_argument(
        "--mode",
        choices=("collinear", "non_collinear"),
        default="collinear",
        help="Graph-data packaging mode.",
    )
    pack_graph_data.set_defaults(func=command_pack_graph_data)
    return parser


def _load_merged_config(args):
    config = load_config(args.config)
    return merge_cli_overrides(
        config,
        input_pattern=getattr(args, "input_pattern", None),
        output_dir=getattr(args, "output_dir", None),
        read_openmx=getattr(args, "read_openmx", None),
        nao_max=getattr(args, "nao_max", None),
        workers=getattr(args, "workers", None),
        dry_run=args.dry_run,
        skip_errors=args.skip_errors,
    )


def _planned_paths(config, suffix, output_name=None):
    patterns = config.get("inputs", {}).get("patterns")
    output_dir = config.get("outputs", {}).get("directory")
    if not patterns:
        raise ConfigError("Config must define inputs.patterns.")
    if not output_dir:
        raise ConfigError("Config must define outputs.directory.")

    inputs = discover_files(patterns)
    output_dir_path = Path(output_dir)
    plans = []
    for input_path in inputs:
        if output_name is None:
            output_path = output_path_for(input_path, output_dir_path, suffix, overwrite=True)
        else:
            output_path = output_dir_path / output_name
        plans.append((Path(input_path), output_path))
    return plans


def _pack_input_roots(patterns):
    if isinstance(patterns, (str, Path)):
        pattern_list = [patterns]
    else:
        pattern_list = list(patterns)

    roots = set()
    for pattern in pattern_list:
        pattern_path = Path(pattern)
        if pattern_path.exists():
            roots.add(pattern_path.resolve() if pattern_path.is_dir() else pattern_path.resolve().parent)
            continue

        for match in discover_files(pattern):
            roots.add(match.resolve() if match.is_dir() else match.resolve().parent)

    if not roots:
        raise ConfigError(f"No files or directories matched pattern(s): {', '.join(str(pattern) for pattern in pattern_list)}")

    return sorted(roots)


def _shared_prefix_length(left: str, right: str) -> int:
    prefix = 0
    for left_char, right_char in zip(left, right):
        if left_char != right_char:
            break
        prefix += 1
    return prefix


def _select_best_file(directory: Path, suffix: str, root_name: str, *, exclude_names: set[str] | None = None) -> Path:
    exclude_names = exclude_names or set()
    candidates = [path for path in sorted(directory.glob(f"*{suffix}")) if path.name not in exclude_names]
    if not candidates:
        raise ConfigError(f"No '{suffix}' files found in '{directory}'.")

    def score(path: Path):
        stem = path.stem
        return (
            1 if stem == root_name else 0,
            1 if stem.startswith(root_name) else 0,
            _shared_prefix_length(stem, root_name),
            -abs(len(stem) - len(root_name)),
            -len(path.name),
            path.name,
        )

    return max(candidates, key=score)


def _pack_graph_for_root(root: Path, config, mode: str, read_openmx, nao_max: int):
    dat_file = _select_best_file(root, ".dat", root.name)
    std_file = _select_best_file(root, ".std", root.name)
    scfout_file = _select_best_file(root, ".scfout", root.name, exclude_names={"overlap.scfout"})

    dat_text = dat_file.read_text(encoding="utf-8")
    std_text = std_file.read_text(encoding="utf-8")
    species, _, _, _ = _parse_dat_text(dat_text)
    species_settings = resolve_species_settings(species, config.get("species"), nao_max=nao_max)
    hs_payload = run_read_openmx(read_openmx, scfout_file, root)

    if mode == "non_collinear":
        return build_non_collinear_graph(dat_text, std_text, hs_payload, species_settings, nao_max)

    return build_collinear_graph(dat_text, std_text, hs_payload, species_settings, nao_max)


def _print_dry_run(command_name, plans):
    print(f"Dry run: {command_name}")
    for input_path, output_path in plans:
        print(f"  input: {input_path}")
        print(f"  output: {output_path}")


def _resolve_nao_max(config, default=26):
    graph_data = config.get("graph_data", {})
    if graph_data and not isinstance(graph_data, dict):
        raise ConfigError("Config section 'graph_data' must be a mapping.")

    try:
        return int(graph_data.get("nao_max", default))
    except (TypeError, ValueError) as exc:
        raise ConfigError("graph_data.nao_max must be an integer.") from exc


def _print_skip_summary(command_name, plans, skipped):
    if not skipped:
        return
    print(
        f"{command_name}: processed {len(plans) - len(skipped)}/{len(plans)} file(s); skipped {len(skipped)}.",
        file=sys.stderr,
    )
    for input_path, error in skipped:
        print(f"  {input_path}: {error}", file=sys.stderr)


def _is_batch_error(error):
    return isinstance(error, (ConfigError, OSError, ValueError, TypeError, IndexError, KeyError))


def _resolve_workers(config):
    workers = config.get("runtime", {}).get("workers", 1)
    if workers is None:
        return 1
    workers = int(workers)
    return max(workers, 1)


def _run_batch(plans, worker_count, runner):
    if worker_count <= 1 or len(plans) <= 1:
        return [runner(item) for item in plans]

    results = [None] * len(plans)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_map = {executor.submit(runner, item): index for index, item in enumerate(plans)}
        for future in as_completed(future_map):
            index = future_map[future]
            results[index] = future.result()
    return results


def _run_batch_with_skip_errors(command_name, plans, worker_count, runner, *, skip_errors):
    skipped = []
    results = [None] * len(plans)

    def wrapped(item):
        try:
            return runner(item)
        except Exception as exc:  # noqa: BLE001 - converted into user-facing batch diagnostics.
            if not skip_errors or not _is_batch_error(exc):
                raise
            return exc

    batch_results = _run_batch(plans, worker_count, wrapped)
    for index, result in enumerate(batch_results):
        if isinstance(result, Exception):
            skipped.append((plans[index][0], result))
        else:
            results[index] = result

    _print_skip_summary(command_name, plans, skipped)
    return results, skipped


def _run_preflight_batch(command_name, plans, worker_count, preflight, *, skip_errors, empty_message):
    def runner(item):
        preflight(item)
        return item

    results, _ = _run_batch_with_skip_errors(
        command_name,
        plans,
        worker_count,
        runner,
        skip_errors=skip_errors,
    )
    valid_plans = [plan for plan, result in zip(plans, results) if result is not None]
    if not valid_plans:
        raise ConfigError(empty_message)
    return valid_plans


def _load_structure(path):
    try:
        structure = Structure.from_file(path)
    except Exception as exc:  # noqa: BLE001 - parsed into ConfigError for uniform CLI diagnostics.
        raise ConfigError(f"Cannot read structure file '{path}': {exc}") from exc
    return AseAtomsAdaptor.get_atoms(structure)


def _validate_species_settings(atoms, species_config, nao_max):
    symbols = atoms.get_chemical_symbols()
    species_settings = resolve_species_settings(symbols, species_config, nao_max=nao_max)
    return symbols, species_settings


def _preflight_convert_collinear(input_path, config):
    atoms = _load_structure(input_path)
    species_config = config.get("species", {})
    _validate_species_settings(atoms, species_config, 26)
    return atoms


def _preflight_convert_noncollinear(input_path, config):
    atoms = read_ase(input_path)
    _, spin_vectors = read_xsf_spin(input_path)
    if len(spin_vectors) != len(atoms):
        raise ConfigError(
            f"Spin vector count {len(spin_vectors)} does not match atom count {len(atoms)} for '{input_path}'."
        )
    noncollinear_config = config.get("convert_noncollinear", {})
    threshold = float(noncollinear_config.get("nonmagnetic_threshold", 0.01))
    spin_to_spherical(spin_vectors, nonmagnetic_threshold=threshold)
    species_config = config.get("species", {})
    _validate_species_settings(atoms, species_config, 26)
    return atoms


def _preflight_make_xsf_spin(input_path, config):
    atoms = _load_structure(input_path)
    workflow = config.get("make_xsf_spin", {})
    if not isinstance(workflow, dict):
        raise ConfigError("Config section 'make_xsf_spin' must be a mapping.")
    species_config = config.get("species", {})
    _validate_species_settings(atoms, species_config, 26)
    magnetic_vectors = generate_spin_vectors(len(atoms), workflow)
    build_xsf_text(atoms, magnetic_vectors)
    return atoms


def _preflight_pack_graph_root(root, config, mode, read_openmx, nao_max):
    dat_file = _select_best_file(root, ".dat", root.name)
    std_file = _select_best_file(root, ".std", root.name)
    scfout_file = _select_best_file(root, ".scfout", root.name, exclude_names={"overlap.scfout"})
    if not dat_file.exists() or not std_file.exists() or not scfout_file.exists():
        raise ConfigError(f"Missing required graph files in '{root}'.")

    dat_text = dat_file.read_text(encoding="utf-8")
    std_text = std_file.read_text(encoding="utf-8")
    species, _, _, _ = _parse_dat_text(dat_text)
    resolve_species_settings(species, config.get("species"), nao_max=nao_max)
    _ = std_text
    return root


def _config_dry_run_enabled(config):
    return bool(config.get("runtime", {}).get("dry_run"))


def _collinear_template(config):
    section = config.get("convert_collinear", {})
    if not isinstance(section, dict):
        raise ConfigError("Config section 'convert_collinear' must be a mapping.")
    template = section.get("template", "")
    if not isinstance(template, str):
        raise ConfigError("convert_collinear.template must be a string.")
    if template and not template.endswith("\n"):
        template += "\n"
    return template


def _noncollinear_template(config):
    section = config.get("convert_noncollinear", {})
    if not isinstance(section, dict):
        raise ConfigError("Config section 'convert_noncollinear' must be a mapping.")
    template = section.get("template", "")
    if not isinstance(template, str):
        raise ConfigError("convert_noncollinear.template must be a string.")
    if template and not template.endswith("\n"):
        template += "\n"
    return template


def _atom_settings(symbols, species_settings, atom_spins):
    settings = [deepcopy(species_settings[symbol]) for symbol in symbols]
    if atom_spins is None:
        return settings
    if not isinstance(atom_spins, dict):
        raise ConfigError("atom_spins must be a mapping of 1-based atom index to [spin_up, spin_down].")

    for raw_index, spin in atom_spins.items():
        try:
            atom_index = int(raw_index)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"atom_spins key '{raw_index}' is not a valid 1-based atom index.") from exc
        if atom_index < 1 or atom_index > len(settings):
            raise ConfigError(f"atom_spins index {atom_index} is outside the atom range 1..{len(settings)}.")
        if not isinstance(spin, (list, tuple)) or len(spin) != 2:
            raise ConfigError(f"atom_spins.{raw_index} must be a two-value [spin_up, spin_down] list.")
        settings[atom_index - 1]["spin"] = [float(spin[0]), float(spin[1])]
    return settings


def command_convert_collinear(args):
    """Run POSCAR/CIF to collinear OpenMX .dat conversion."""
    config = _load_merged_config(args)
    plans = _planned_paths(config, ".dat")
    workers = _resolve_workers(config)
    skip_errors = bool(config.get("runtime", {}).get("skip_errors"))
    if _config_dry_run_enabled(config):
        valid_plans = _run_preflight_batch(
            "convert-collinear",
            plans,
            workers,
            lambda item: _preflight_convert_collinear(item[0], config),
            skip_errors=skip_errors,
            empty_message="No valid collinear inputs could be converted.",
        )
        _print_dry_run("convert-collinear", valid_plans)
        return 0

    ensure_output_dir(config["outputs"]["directory"])
    template = _collinear_template(config)
    species_config = config.get("species", {})

    def runner(item):
        input_path, output_path = item
        atoms = _load_structure(input_path)
        symbols = atoms.get_chemical_symbols()
        species_settings = resolve_species_settings(symbols, species_config)
        atom_settings = _atom_settings(symbols, species_settings, config.get("atom_spins"))
        text = build_collinear_dat_text(atoms, template, species_settings, atom_settings=atom_settings)
        return write_text(output_path, text, overwrite=True)

    results, _ = _run_batch_with_skip_errors(
        "convert-collinear",
        plans,
        workers,
        runner,
        skip_errors=skip_errors,
    )
    if not any(result is not None for result in results):
        raise ConfigError("No valid collinear inputs could be converted.")
    return 0


def command_convert_noncollinear(args):
    """Run XSF spin to non-collinear OpenMX .dat conversion."""
    config = _load_merged_config(args)
    plans = _planned_paths(config, ".dat")
    workers = _resolve_workers(config)
    skip_errors = bool(config.get("runtime", {}).get("skip_errors"))
    if _config_dry_run_enabled(config):
        valid_plans = _run_preflight_batch(
            "convert-noncollinear",
            plans,
            workers,
            lambda item: _preflight_convert_noncollinear(item[0], config),
            skip_errors=skip_errors,
            empty_message="No valid non-collinear inputs could be converted.",
        )
        _print_dry_run("convert-noncollinear", valid_plans)
        return 0

    ensure_output_dir(config["outputs"]["directory"])
    template = _noncollinear_template(config)
    species_config = config.get("species", {})
    noncollinear_config = config.get("convert_noncollinear", {})
    threshold = float(noncollinear_config.get("nonmagnetic_threshold", 0.01))

    def runner(item):
        input_path, output_path = item
        atoms = read_ase(input_path)
        _, spin_vectors = read_xsf_spin(input_path)
        symbols = atoms.get_chemical_symbols()
        if len(spin_vectors) != len(symbols):
            raise ConfigError(
                f"Spin vector count {len(spin_vectors)} does not match atom count {len(symbols)} for '{input_path}'."
            )
        _, theta, phi = spin_to_spherical(spin_vectors, nonmagnetic_threshold=threshold)
        species_settings = resolve_species_settings(symbols, species_config)
        atom_settings = _atom_settings(symbols, species_settings, config.get("atom_spins"))
        text = build_noncollinear_dat_text(
            atoms,
            template,
            species_settings,
            theta=theta,
            phi=phi,
            atom_settings=atom_settings,
        )
        return write_text(output_path, text, overwrite=True)

    results, _ = _run_batch_with_skip_errors(
        "convert-noncollinear",
        plans,
        workers,
        runner,
        skip_errors=skip_errors,
    )
    if not any(result is not None for result in results):
        raise ConfigError("No valid non-collinear inputs could be converted.")
    return 0


def command_make_xsf_spin(args):
    """Run POSCAR/CIF to XSF-with-spin conversion."""
    config = _load_merged_config(args)
    plans = _planned_paths(config, ".xsf")
    workers = _resolve_workers(config)
    skip_errors = bool(config.get("runtime", {}).get("skip_errors"))
    if _config_dry_run_enabled(config):
        valid_plans = _run_preflight_batch(
            "make-xsf-spin",
            plans,
            workers,
            lambda item: _preflight_make_xsf_spin(item[0], config),
            skip_errors=skip_errors,
            empty_message="No valid XSF inputs could be converted.",
        )
        _print_dry_run("make-xsf-spin", valid_plans)
        return 0

    ensure_output_dir(config["outputs"]["directory"])
    species_config = config.get("species", {})
    workflow = config.get("make_xsf_spin", {})
    if not isinstance(workflow, dict):
        raise ConfigError("Config section 'make_xsf_spin' must be a mapping.")

    def runner(item):
        input_path, output_path = item
        atoms = _load_structure(input_path)
        symbols = atoms.get_chemical_symbols()
        resolve_species_settings(symbols, species_config)
        magnetic_vectors = generate_spin_vectors(len(symbols), workflow)
        text = build_xsf_text(atoms, magnetic_vectors)
        return write_text(output_path, text, overwrite=True)

    results, _ = _run_batch_with_skip_errors(
        "make-xsf-spin",
        plans,
        workers,
        runner,
        skip_errors=skip_errors,
    )
    if not any(result is not None for result in results):
        raise ConfigError("No valid XSF inputs could be converted.")
    return 0


def command_pack_graph_data(args):
    """Run OpenMX output packaging to graph_data.npz."""
    config = _load_merged_config(args)
    patterns = config.get("inputs", {}).get("patterns")
    output_dir = config.get("outputs", {}).get("directory")
    if not patterns:
        raise ConfigError("Config must define inputs.patterns.")
    if not output_dir:
        raise ConfigError("Config must define outputs.directory.")

    roots = _pack_input_roots(patterns)
    skip_errors = bool(config.get("runtime", {}).get("skip_errors"))
    workers = _resolve_workers(config)
    if _config_dry_run_enabled(config):
        read_openmx = resolve_read_openmx(config.get("graph_data", {}).get("read_openmx"))
        nao_max = _resolve_nao_max(config)
        output_path = Path(output_dir) / "graph_data.npz"
        plans = [(root, output_path) for root in roots]
        valid_plans = _run_preflight_batch(
            "pack-graph-data",
            plans,
            workers,
            lambda item: _preflight_pack_graph_root(item[0], config, args.mode, read_openmx, nao_max),
            skip_errors=skip_errors,
            empty_message="No valid graph data could be packaged.",
        )
        _print_dry_run(f"pack-graph-data ({args.mode})", valid_plans)
        return 0

    read_openmx = resolve_read_openmx(config.get("graph_data", {}).get("read_openmx"))
    nao_max = _resolve_nao_max(config)
    output_dir_path = ensure_output_dir(output_dir)
    output_path = output_dir_path / "graph_data.npz"

    def runner(item):
        root, _ = item
        return _pack_graph_for_root(root, config, args.mode, read_openmx, nao_max)

    plans = [(root, output_path) for root in roots]
    results, _ = _run_batch_with_skip_errors(
        "pack-graph-data",
        plans,
        workers,
        runner,
        skip_errors=skip_errors,
    )
    graphs = {index: graph for index, graph in enumerate(results) if graph is not None}

    if not graphs:
        raise ConfigError("No valid graph data could be packaged.")

    save_graphs_npz(graphs, output_path)
    return 0


def main(argv=None):
    """Parse arguments and dispatch the selected subcommand."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ConfigError as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
