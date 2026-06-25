from __future__ import annotations

from pathlib import Path

import numpy as np

from DFT_interfaces.openmx.magnetism.config import ConfigError


def read_xsf_spin(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()

    for index, line in enumerate(lines):
        if line.strip() != "PRIMCOORD":
            continue

        if index + 1 >= len(lines):
            break

        try:
            atom_count = int(lines[index + 1].split()[0])
        except (IndexError, ValueError) as exc:
            raise ConfigError(f"Invalid PRIMCOORD header in '{path}'.") from exc

        coords = []
        spins = []
        for atom_line in lines[index + 2 : index + 2 + atom_count]:
            fields = atom_line.split()
            if len(fields) < 7:
                raise ConfigError(f"Missing spin columns in PRIMCOORD block for '{path}'.")
            try:
                coords.append([float(value) for value in fields[1:4]])
                spins.append([float(value) for value in fields[4:7]])
            except ValueError as exc:
                raise ConfigError(
                    f"Invalid numeric field in PRIMCOORD row '{atom_line}' for '{path}'."
                ) from exc

        if len(coords) != atom_count:
            raise ConfigError(f"Incomplete PRIMCOORD block in '{path}'.")

        return np.asarray(coords, dtype=float), np.asarray(spins, dtype=float)

    raise ConfigError(f"Missing PRIMCOORD block in '{path}'.")


def spin_to_spherical(spin, nonmagnetic_threshold=0.01):
    spin = np.asarray(spin, dtype=float)
    if spin.ndim != 2 or spin.shape[1] != 3:
        raise ConfigError(
            f"spin must be an N x 3 array; got shape {spin.shape}."
        )
    moments = np.linalg.norm(spin, axis=1)
    theta = np.zeros_like(moments)
    phi = np.zeros_like(moments)

    magnetic = moments >= nonmagnetic_threshold
    safe_moments = np.where(magnetic, moments, 1.0)
    cos_theta = np.clip(spin[:, 2] / safe_moments, -1.0, 1.0)

    theta[magnetic] = np.degrees(np.arccos(cos_theta[magnetic]))
    phi[magnetic] = np.degrees(np.arctan2(spin[magnetic, 1], spin[magnetic, 0]))

    return moments, theta, phi


def generate_spin_vectors(atom_count, config):
    if config.get("vectors") is not None:
        if "base_direction" in config or "mask" in config:
            raise ConfigError("make_xsf_spin 'vectors' cannot be combined with 'base_direction' or 'mask'.")
        vectors = np.asarray(config["vectors"], dtype=float)
        if vectors.ndim != 2 or vectors.shape != (atom_count, 3):
            raise ConfigError(
                f"Explicit spin vectors must have shape ({atom_count}, 3); got shape {vectors.shape}."
            )
        if not np.all(np.isfinite(vectors)):
            raise ConfigError("Explicit spin vectors must contain only finite values.")
        return vectors

    if "base_direction" not in config:
        raise ConfigError("make_xsf_spin must define either 'vectors' or 'base_direction'.")

    base_direction = np.asarray(config["base_direction"], dtype=float)
    if base_direction.ndim != 1 or base_direction.shape != (3,):
        raise ConfigError(
            f"base_direction must be a 1D vector of length 3; got shape {base_direction.shape}."
        )
    norm = np.linalg.norm(base_direction)
    if norm == 0.0:
        raise ConfigError("base_direction must be non-zero.")

    mask = np.asarray(config.get("mask", np.ones(atom_count)), dtype=float)
    if mask.ndim != 1:
        raise ConfigError(f"Spin mask must be a 1D array of length {atom_count}.")
    if mask.shape != (atom_count,):
        raise ConfigError(f"Spin mask length {mask.shape[0]} does not match atom_count={atom_count}.")

    direction = base_direction / norm
    return mask[:, None] * direction
