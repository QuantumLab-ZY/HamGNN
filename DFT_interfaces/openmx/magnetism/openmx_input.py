from __future__ import annotations

from pathlib import Path

import numpy as np


def ordered_unique(items):
    """Return items in first-seen order without duplicates."""
    seen = set()
    return [item for item in items if not (item in seen or seen.add(item))]


def _species_block(symbols, species_settings):
    species = ordered_unique(symbols)
    text = "#\n# Definition of Atomic Species\n#\n"
    text += f"Species.Number       {len(species)}\n"
    text += "<Definition.of.Atomic.Species\n"
    for symbol in species:
        settings = species_settings[symbol]
        text += f"{symbol}   {settings['pao']}       {settings['pbe']}\n"
    text += "Definition.of.Atomic.Species>\n\n"
    return text


def _unit_vectors_block(cell):
    text = "Atoms.UnitVectors.Unit             Ang #  Ang|AU"
    text += "\n<Atoms.UnitVectors                     # unit=Ang."
    text += (
        "\n      %10.7f  %10.7f  %10.7f\n      %10.7f  %10.7f  %10.7f\n      %10.7f  %10.7f  %10.7f"
        % (*cell[0], *cell[1], *cell[2])
    )
    text += "\nAtoms.UnitVectors>"
    return text


def _spin_constraint_enabled(value):
    """Normalize spin-constraint values shared by collinear/noncollinear builders."""
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "on":
            return True
        if normalized == "off":
            return False
    return bool(int(value))


def build_collinear_dat_text(atoms, template, species_settings, atom_settings=None):
    """Return OpenMX collinear input text for an ASE Atoms object."""
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    cell = atoms.get_cell().array

    text = template
    text += _species_block(symbols, species_settings)
    text += "#\n# Atoms\n#\n"
    text += "Atoms.Number%12d" % len(symbols)
    text += "\nAtoms.SpeciesAndCoordinates.Unit   Ang # Ang|AU"
    text += "\n<Atoms.SpeciesAndCoordinates           # Unit=Ang."
    for index, symbol in enumerate(symbols, start=1):
        settings = atom_settings[index - 1] if atom_settings is not None else species_settings[symbol]
        spin_up, spin_down = settings["spin"]
        spin_constraint = "on" if _spin_constraint_enabled(settings["spin_constraint"]) else "off"
        text += (
            "\n%3d  %s  %10.7f  %10.7f  %10.7f   %.2f   %.2f  %s"
            % (
                index,
                symbol,
                *positions[index - 1],
                spin_up,
                spin_down,
                spin_constraint,
            )
        )
    text += "\nAtoms.SpeciesAndCoordinates>\n"
    text += _unit_vectors_block(cell)
    return text


def build_noncollinear_dat_text(atoms, template, species_settings, theta, phi, atom_settings=None):
    """Return OpenMX non-collinear input text with theta/phi spin angles."""
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    cell = atoms.get_cell().array
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)

    if theta.shape != (len(symbols),) or phi.shape != (len(symbols),):
        raise ValueError("theta and phi must match atom count")

    text = template
    text += _species_block(symbols, species_settings)
    text += "#\n# Atoms\n#\n"
    text += "Atoms.Number%12d" % len(symbols)
    text += "\nAtoms.SpeciesAndCoordinates.Unit   Ang # Ang|AU"
    text += "\n<Atoms.SpeciesAndCoordinates           # Unit=Ang."
    for index, symbol in enumerate(symbols, start=1):
        settings = atom_settings[index - 1] if atom_settings is not None else species_settings[symbol]
        spin_up, spin_down = settings["spin"]
        spin_constraint = int(_spin_constraint_enabled(settings["spin_constraint"]))
        text += (
            "\n%3d  %s  %10.7f  %10.7f  %10.7f   %.2f   %.2f  %.3f  %.3f  %.3f %.3f  %d  off"
            % (
                index,
                symbol,
                *positions[index - 1],
                spin_up,
                spin_down,
                theta[index - 1],
                phi[index - 1],
                theta[index - 1],
                phi[index - 1],
                spin_constraint,
            )
        )
    text += "\nAtoms.SpeciesAndCoordinates>\n"
    text += _unit_vectors_block(cell)
    return text


def build_xsf_text(atoms, magnetic_vectors=None):
    """Return XSF text with optional magnetic vectors in PRIMCOORD rows."""
    cell = atoms.get_cell()
    positions = atoms.get_positions()
    numbers = atoms.get_atomic_numbers()

    vectors = None
    if magnetic_vectors is not None:
        vectors = np.asarray(magnetic_vectors, dtype=float)
        if vectors.shape != (len(positions), 3):
            raise ValueError("magnetic_vectors must have shape (N, 3)")

    text = "CRYSTAL\n"
    text += "PRIMVEC\n"
    for row in cell:
        text += " %.14f %.14f %.14f\n" % tuple(row)
    text += "PRIMCOORD\n"
    text += " %d 1\n" % len(positions)

    for index, position in enumerate(positions):
        text += " %2d" % numbers[index]
        text += " %20.14f %20.14f %20.14f" % tuple(position)
        if vectors is None:
            text += "\n"
        else:
            text += " %20.14f %20.14f %20.14f\n" % tuple(vectors[index])

    return text


def write_text(path, text, overwrite=False):
    """Write text to path, refusing existing files unless overwrite is true."""
    output_path = Path(path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"File already exists: {output_path}")
    output_path.write_text(text, encoding="utf-8")
    return output_path
