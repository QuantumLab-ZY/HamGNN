from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np
import torch
from pymatgen.core.periodic_table import Element
from torch_geometric.data import Data

from DFT_interfaces.openmx.magnetism.config import ConfigError


AU_TO_ANG = 0.5291772083
PATTERN_TOTAL_ENERGY = re.compile(
    r'(?:total\s+energy|Utot)\s*[=:]\s*(\-?\d+\.?\d*(?:[Ee][+\-]?\d+)?)',
    re.IGNORECASE,
)
PATTERN_MD = re.compile(r"MD= 1  SCF=(\W*)(\d+)")
PATTERN_LATT = re.compile(
    r"<Atoms.UnitVectors.+?\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+Atoms.UnitVectors>",
    re.DOTALL,
)
PATTERN_UNIT_DECLARATION = r"^\s*{key}\s+(\S+)(?:\s+#.*)?$"
PATTERN_COOR = re.compile(
    r"\s+\d+\s+(\w+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+(\-?\d+\.?\d+)\s+\-?\d+\.?\d+\s+\-?\d+\.?\d+"
)


def basis_mask(nao_max, source_basis, target_basis):
    nao_max = int(nao_max)
    src = np.asarray(list(source_basis), dtype=int)
    tar = np.asarray(list(target_basis), dtype=int)

    if np.any(src < 0) or np.any(src >= nao_max) or np.any(tar < 0) or np.any(tar >= nao_max):
        raise ConfigError(f"Basis index outside nao_max={nao_max} in source_basis={list(src)} target_basis={list(tar)}.")

    mask = np.zeros((nao_max, nao_max), dtype=bool)
    mask[src[:, None], tar[None, :]] = True
    return mask.reshape(-1)


def parse_scf_iterations(std_text):
    matches = PATTERN_MD.findall(std_text.strip())
    if not matches:
        raise ConfigError("Could not parse SCF iteration count from OpenMX stdout text. Expected lines matching 'MD= 1  SCF='.")
    return int(matches[-1][-1])


def parse_energy(std_text):
    text = std_text.strip()
    total_energy_matches = PATTERN_TOTAL_ENERGY.findall(text)
    if total_energy_matches:
        return float(total_energy_matches[-1])

    raise ConfigError(
        "Could not parse final total energy from OpenMX stdout text. "
        "Expected a line matching 'total energy' or 'Utot.'."
    )


def should_skip_scf(max_scf, threshold):
    return int(max_scf) > int(threshold)


def _as_integer_valued_array(values, field_name):
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(int, copy=False)
    if not np.issubdtype(arr.dtype, np.number):
        raise ConfigError(f"HS payload {field_name} must contain integer values.")
    if not np.all(np.isfinite(arr)) or not np.all(np.equal(arr, np.floor(arr))):
        raise ConfigError(f"HS payload {field_name} must contain integer values.")
    return arr.astype(int)


def validate_hs_payload(payload, mode):
    if not isinstance(payload, dict):
        raise ConfigError("HS payload must be a mapping loaded from HS.json.")

    common = {"pos", "edge_index", "inv_edge_idx", "nbr_shift", "cell_shift", "Hon", "Hoff", "Son", "Soff"}
    mode_specific = {
        "collinear": set(),
        "non_collinear": {"iHon", "iHoff", "Lon", "Loff"},
    }
    if mode not in mode_specific:
        raise ConfigError(f"Unsupported HS payload mode '{mode}'.")

    required = common | mode_specific[mode]
    missing = sorted(required.difference(payload))
    if missing:
        raise ConfigError(f"Missing HS payload keys for mode '{mode}': {', '.join(missing)}")

    edge_index = _as_integer_valued_array(payload["edge_index"], "edge_index")
    if edge_index.ndim != 2:
        raise ConfigError("HS payload edge_index must have shape [2, num_edges].")
    if edge_index.shape[0] != 2:
        raise ConfigError("HS payload edge_index must have shape [2, num_edges].")

    inv_edge_idx = _as_integer_valued_array(payload["inv_edge_idx"], "inv_edge_idx")
    num_edges = edge_index.shape[1]
    if len(inv_edge_idx) != num_edges:
        raise ConfigError("HS payload inv_edge_idx length must match edge_index column count.")

    pos = np.asarray(payload["pos"], dtype=float)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ConfigError("HS payload pos must have shape [num_atoms, 3].")

    num_atoms = pos.shape[0]
    if np.any(edge_index < 0) or np.any(edge_index >= num_atoms):
        raise ConfigError("HS payload edge_index contains atom indices outside the pos array.")
    if np.any(inv_edge_idx < 0) or np.any(inv_edge_idx >= num_edges):
        raise ConfigError("HS payload inv_edge_idx contains values outside the edge_index column range.")

    reverse_edges = edge_index[:, inv_edge_idx]
    if not np.array_equal(reverse_edges[0], edge_index[1]) or not np.array_equal(reverse_edges[1], edge_index[0]):
        raise ConfigError("HS payload inv_edge_idx must map each edge to its reverse edge in edge_index.")

    nbr_shift = np.asarray(payload["nbr_shift"], dtype=float)
    if nbr_shift.ndim != 2 or nbr_shift.shape != (num_edges, 3):
        raise ConfigError("HS payload nbr_shift must have shape [num_edges, 3].")

    cell_shift = _as_integer_valued_array(payload["cell_shift"], "cell_shift")
    if cell_shift.ndim != 2 or cell_shift.shape != (num_edges, 3):
        raise ConfigError("HS payload cell_shift must have shape [num_edges, 3].")
    if not np.array_equal(cell_shift[inv_edge_idx], -cell_shift):
        raise ConfigError("HS payload cell_shift must negate across reverse edges referenced by inv_edge_idx.")
    if not np.allclose(nbr_shift[inv_edge_idx], -nbr_shift, atol=1e-6, rtol=1e-6):
        raise ConfigError("HS payload nbr_shift must negate across reverse edges referenced by inv_edge_idx.")


def _validate_payload_positions(payload_pos, parsed_coords):
    pos = np.asarray(payload_pos, dtype=float)
    coords = np.asarray(parsed_coords, dtype=float)
    if pos.shape != coords.shape:
        raise ConfigError("HS payload pos shape does not match parsed .dat coordinates.")
    if not np.allclose(pos, coords, atol=1e-6, rtol=1e-6):
        raise ConfigError("HS payload pos does not match parsed .dat coordinates.")


def symmetrize_hon(values, nao_max, sign="+"):
    if sign not in {"+", "-"}:
        raise ConfigError("sign must be '+' or '-'.")
    arr = np.asarray(values, dtype=float)
    expected = int(nao_max) ** 2
    if arr.ndim != 2 or arr.shape[1] != expected:
        raise ConfigError(f"Hon values must have shape [n, {expected}] for nao_max={nao_max}.")
    arr = arr.reshape(-1, int(nao_max), int(nao_max))
    if sign == "+":
        arr = 0.5 * (arr + arr.transpose(0, 2, 1))
    else:
        arr = 0.5 * (arr - arr.transpose(0, 2, 1))
    return arr.reshape(-1, expected)


def symmetrize_hoff(values, inv_edge_idx, nao_max, sign="+"):
    if sign not in {"+", "-"}:
        raise ConfigError("sign must be '+' or '-'.")
    arr = np.asarray(values, dtype=float)
    inv_edge_idx = np.asarray(inv_edge_idx, dtype=int)
    expected = int(nao_max) ** 2
    if arr.ndim != 2 or arr.shape[1] != expected:
        raise ConfigError(f"Hoff values must have shape [n, {expected}] for nao_max={nao_max}.")
    if len(inv_edge_idx) != len(arr):
        raise ConfigError("inv_edge_idx length must match number of off-site blocks.")
    if np.any(inv_edge_idx < 0) or np.any(inv_edge_idx >= len(arr)):
        raise ConfigError("inv_edge_idx contains values outside the off-site block range.")
    arr = arr.reshape(-1, int(nao_max), int(nao_max))
    if sign == "+":
        arr = 0.5 * (arr + arr[inv_edge_idx].transpose(0, 2, 1))
    else:
        arr = 0.5 * (arr - arr[inv_edge_idx].transpose(0, 2, 1))
    return arr.reshape(-1, expected)


def _parse_dat_text(dat_text: str):
    text = dat_text.strip()
    species_and_coordinates = PATTERN_COOR.findall(text)
    if not species_and_coordinates:
        raise ConfigError("Could not parse Atoms.SpeciesAndCoordinates from OpenMX .dat text.")

    latt_match = PATTERN_LATT.findall(text)
    if not latt_match:
        raise ConfigError("Could not parse Atoms.UnitVectors from OpenMX .dat text.")

    cell_scale = _unit_scale_from_dat_text(text, "Atoms.UnitVectors.Unit", field_name="Atoms.UnitVectors")
    coord_scale = _unit_scale_from_dat_text(
        text,
        "Atoms.SpeciesAndCoordinates.Unit",
        field_name="Atoms.SpeciesAndCoordinates",
    )

    species = [item[0] for item in species_and_coordinates]
    try:
        z = np.asarray([Element[symbol].Z for symbol in species], dtype=int)
    except KeyError as exc:
        raise ConfigError(
            f"Invalid species token '{exc.args[0]}' in OpenMX .dat Atoms.SpeciesAndCoordinates section. "
            "Expected a valid element symbol."
        ) from exc

    cell = np.asarray([float(value) for value in latt_match[0]], dtype=float).reshape(3, 3) * cell_scale
    coords = (
        np.asarray([float(value) for item in species_and_coordinates for value in item[1:]], dtype=float).reshape(-1, 3)
        * coord_scale
    )
    return species, z, coords, cell


def _unit_scale_from_dat_text(dat_text: str, unit_key: str, field_name: str) -> float:
    match = re.search(PATTERN_UNIT_DECLARATION.format(key=re.escape(unit_key)), dat_text, re.MULTILINE)
    if not match:
        raise ConfigError(
            f"Missing {unit_key} declaration in OpenMX .dat text. "
            f"Declare units explicitly for {field_name}."
        )

    declared_unit = match.group(1).strip().lower()
    if declared_unit == "ang":
        return 1.0
    if declared_unit in {"au", "bohr"}:
        return AU_TO_ANG

    raise ConfigError(
        f"Unsupported {unit_key} value '{match.group(1)}' in OpenMX .dat text. "
        "Supported units are Ang and AU/Bohr."
    )


def _species_basis(species: Iterable[str], species_settings: dict[str, dict[str, Any]]):
    resolved = []
    for symbol in species:
        settings = species_settings.get(symbol)
        if settings is None:
            raise ConfigError(f"Missing species settings for '{symbol}'.")
        basis = settings.get("basis")
        if basis is None:
            raise ConfigError(f"Species settings for '{symbol}' must include 'basis'.")
        resolved.append(list(basis))
    return resolved


def _build_spin_from_species(species: Iterable[str], species_settings: dict[str, dict[str, Any]], spin_vectors=None):
    if spin_vectors is not None:
        spin_vec = np.asarray(spin_vectors, dtype=float)
        if spin_vec.ndim != 2 or spin_vec.shape[1] != 3:
            raise ConfigError("spin_vectors must have shape [num_atoms, 3].")
        if spin_vec.shape[0] != len(list(species)):
            raise ConfigError("spin_vectors atom count must match the number of atoms.")
        if not np.all(np.isfinite(spin_vec)):
            raise ConfigError("spin_vectors must contain only finite values.")
        spin_length = np.linalg.norm(spin_vec, axis=1)
        nonzero = spin_length > 0.0
        spin_vec = spin_vec.copy()
        spin_vec[nonzero] = spin_vec[nonzero] / spin_length[nonzero, None]
        return spin_length, spin_vec

    spin_length = []
    spin_vec = []
    for symbol in species:
        spin = species_settings[symbol].get("spin")
        if not isinstance(spin, (list, tuple)) or len(spin) != 2:
            raise ConfigError(f"Species settings for '{symbol}' must include a two-value 'spin' entry.")
        moment = float(spin[0]) - float(spin[1])
        spin_length.append(abs(moment))
        if abs(moment) < 1e-12:
            spin_vec.append([0.0, 0.0, 0.0])
        else:
            spin_vec.append([0.0, 0.0, 1.0 if moment >= 0 else -1.0])
    return np.asarray(spin_length, dtype=float), np.asarray(spin_vec, dtype=float)


def _fill_blocks(blocks, basis_pairs, nao_max):
    if len(blocks) != len(basis_pairs):
        raise ConfigError(f"HS block count mismatch: got {len(blocks)} blocks but expected {len(basis_pairs)}.")
    filled = np.zeros((len(blocks), int(nao_max) ** 2), dtype=float)
    for index, (values, (source_basis, target_basis)) in enumerate(zip(blocks, basis_pairs)):
        mask = basis_mask(nao_max, source_basis, target_basis)
        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.size != int(mask.sum()):
            raise ConfigError(
                f"HS block size mismatch at index {index}: got {arr.size} values but expected {int(mask.sum())}."
            )
        filled[index, mask] = arr
    return filled


def _collinear_basis_pairs(z, edge_index, species_basis):
    on_site = [(basis, basis) for basis in species_basis]
    off_site = [
        (species_basis[int(edge_index[0, idx])], species_basis[int(edge_index[1, idx])])
        for idx in range(edge_index.shape[1])
    ]
    return on_site, off_site


def _build_collinear_hamiltonians(payload, species_basis, edge_index, nao_max):
    spins = payload["Hon"]
    if len(spins) != 2 or len(payload["Hoff"]) != 2:
        raise ConfigError("Collinear HS payload must provide two spin channels for Hon and Hoff.")

    on_site_pairs, off_site_pairs = _collinear_basis_pairs(None, edge_index, species_basis)
    hon = []
    hoff = []
    for channel in range(2):
        hon.append(_fill_blocks(payload["Hon"][channel], on_site_pairs, nao_max))
        hoff.append(_fill_blocks(payload["Hoff"][channel], off_site_pairs, nao_max))
    son = _fill_blocks(payload["Son"], on_site_pairs, nao_max)
    soff = _fill_blocks(payload["Soff"], off_site_pairs, nao_max)
    return np.stack(hon, axis=1), np.stack(hoff, axis=1), son, soff


def _build_non_collinear_hamiltonians(payload, species_basis, edge_index, nao_max):
    if len(payload["Hon"]) != 4 or len(payload["Hoff"]) != 4:
        raise ConfigError("Non-collinear HS payload must provide four real Hamiltonian channels.")
    if len(payload["iHon"]) != 3 or len(payload["iHoff"]) != 3:
        raise ConfigError("Non-collinear HS payload must provide three imaginary Hamiltonian channels.")

    on_site_pairs, off_site_pairs = _collinear_basis_pairs(None, edge_index, species_basis)
    num_atoms = len(species_basis)
    num_edges = edge_index.shape[1]
    num_blocks = num_atoms + num_edges

    hks = np.zeros((num_blocks, 4, int(nao_max) ** 2), dtype=float)
    ihks = np.zeros((num_blocks, 3, int(nao_max) ** 2), dtype=float)
    son = _fill_blocks(payload["Son"], on_site_pairs, nao_max)
    soff = _fill_blocks(payload["Soff"], off_site_pairs, nao_max)
    overlap = np.vstack([son, soff])
    lon = _fill_vector_blocks(payload["Lon"], on_site_pairs, nao_max)
    loff = _fill_vector_blocks(payload["Loff"], off_site_pairs, nao_max)

    for channel in range(4):
        hks[:num_atoms, channel, :] = _fill_blocks(payload["Hon"][channel], on_site_pairs, nao_max)
        hks[num_atoms:, channel, :] = _fill_blocks(payload["Hoff"][channel], off_site_pairs, nao_max)
    for channel in range(3):
        ihks[:num_atoms, channel, :] = _fill_blocks(payload["iHon"][channel], on_site_pairs, nao_max)
        ihks[num_atoms:, channel, :] = _fill_blocks(payload["iHoff"][channel], off_site_pairs, nao_max)

    doubled = 2 * int(nao_max)
    h_real = np.zeros((num_blocks, doubled, doubled), dtype=float)
    h_real[:, :nao_max, :nao_max] = hks[:, 0, :].reshape(-1, nao_max, nao_max)
    h_real[:, :nao_max, nao_max:] = hks[:, 2, :].reshape(-1, nao_max, nao_max)
    h_real[:, nao_max:, :nao_max] = hks[:, 2, :].reshape(-1, nao_max, nao_max)
    h_real[:, nao_max:, nao_max:] = hks[:, 1, :].reshape(-1, nao_max, nao_max)

    h_imag = np.zeros((num_blocks, doubled, doubled), dtype=float)
    h_imag[:, :nao_max, :nao_max] = ihks[:, 0, :].reshape(-1, nao_max, nao_max)
    h_imag[:, :nao_max, nao_max:] = (hks[:, 3, :] + ihks[:, 2, :]).reshape(-1, nao_max, nao_max)
    h_imag[:, nao_max:, :nao_max] = -(hks[:, 3, :] + ihks[:, 2, :]).reshape(-1, nao_max, nao_max)
    h_imag[:, nao_max:, nao_max:] = ihks[:, 1, :].reshape(-1, nao_max, nao_max)

    return h_real.reshape(num_blocks, doubled * doubled), h_imag.reshape(num_blocks, doubled * doubled), overlap, son, soff, lon, loff


def _fill_vector_blocks(blocks, basis_pairs, nao_max):
    if len(blocks) != len(basis_pairs):
        raise ConfigError(f"Vector HS block count mismatch: got {len(blocks)} blocks but expected {len(basis_pairs)}.")
    filled = np.zeros((len(blocks), int(nao_max) ** 2, 3), dtype=float)
    for index, (values, (source_basis, target_basis)) in enumerate(zip(blocks, basis_pairs)):
        mask = basis_mask(nao_max, source_basis, target_basis)
        arr = np.asarray(values, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] != int(mask.sum()):
            raise ConfigError(
                f"Vector HS block size mismatch at index {index}: got shape {arr.shape}, expected ({int(mask.sum())}, 3)."
            )
        filled[index, mask, :] = arr
    return filled


def build_collinear_graph(dat_text, std_text, hs_payload, species_settings, nao_max):
    validate_hs_payload(hs_payload, mode="collinear")
    species, z, coords, cell = _parse_dat_text(dat_text)
    if len(hs_payload["pos"]) != len(species):
        raise ConfigError("HS payload atom count does not match parsed .dat species count.")
    _validate_payload_positions(hs_payload["pos"], coords)

    edge_index = np.asarray(hs_payload["edge_index"], dtype=int)
    inv_edge_idx = np.asarray(hs_payload["inv_edge_idx"], dtype=int)
    pos = np.asarray(hs_payload["pos"], dtype=float)
    species_basis = _species_basis(species, species_settings)
    total_energy = parse_energy(std_text)
    max_scf = parse_scf_iterations(std_text)
    hon, hoff, son, soff = _build_collinear_hamiltonians(hs_payload, species_basis, edge_index, int(nao_max))
    spin_length, spin_vec = _build_spin_from_species(species, species_settings)

    return Data(
        z=torch.LongTensor(z),
        cell=torch.tensor(cell[None, :, :], dtype=torch.float32),
        total_energy=torch.tensor([total_energy], dtype=torch.float32),
        max_scf=torch.tensor([max_scf], dtype=torch.long),
        pos=torch.tensor(pos, dtype=torch.float32),
        node_counts=torch.tensor([len(z)], dtype=torch.long),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        inv_edge_idx=torch.tensor(inv_edge_idx, dtype=torch.long),
        nbr_shift=torch.tensor(np.asarray(hs_payload["nbr_shift"], dtype=float), dtype=torch.float32),
        cell_shift=torch.tensor(np.asarray(hs_payload["cell_shift"], dtype=int), dtype=torch.long),
        Hon=torch.tensor(hon, dtype=torch.float32),
        Hoff=torch.tensor(hoff, dtype=torch.float32),
        Hon0=torch.tensor(hon.copy(), dtype=torch.float32),
        Hoff0=torch.tensor(hoff.copy(), dtype=torch.float32),
        Son=torch.tensor(son, dtype=torch.float32),
        Soff=torch.tensor(soff, dtype=torch.float32),
        spin_length=torch.tensor(spin_length, dtype=torch.float32),
        spin_vec=torch.tensor(spin_vec, dtype=torch.float32),
    )


def build_non_collinear_graph(dat_text, std_text, hs_payload, species_settings, nao_max, spin_vectors=None):
    validate_hs_payload(hs_payload, mode="non_collinear")
    species, z, coords, cell = _parse_dat_text(dat_text)
    if len(hs_payload["pos"]) != len(species):
        raise ConfigError("HS payload atom count does not match parsed .dat species count.")
    pos_bohr = np.asarray(hs_payload["pos"], dtype=float)
    pos_ang = pos_bohr * AU_TO_ANG
    _validate_payload_positions(pos_ang, coords)

    edge_index = np.asarray(hs_payload["edge_index"], dtype=int)
    inv_edge_idx = np.asarray(hs_payload["inv_edge_idx"], dtype=int)
    pos = pos_ang
    species_basis = _species_basis(species, species_settings)
    total_energy = parse_energy(std_text)
    max_scf = parse_scf_iterations(std_text)
    hon, ihon, overlap, son, soff, lon, loff = _build_non_collinear_hamiltonians(
        hs_payload, species_basis, edge_index, int(nao_max)
    )
    num_atoms = len(species)
    spin_length, spin_vec = _build_spin_from_species(species, species_settings, spin_vectors=spin_vectors)

    return Data(
        z=torch.LongTensor(z),
        cell=torch.tensor(cell[None, :, :], dtype=torch.float32),
        total_energy=torch.tensor([total_energy], dtype=torch.float32),
        max_scf=torch.tensor([max_scf], dtype=torch.long),
        pos=torch.tensor(pos, dtype=torch.float32),
        node_counts=torch.tensor([len(z)], dtype=torch.long),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        inv_edge_idx=torch.tensor(inv_edge_idx, dtype=torch.long),
        nbr_shift=torch.tensor(np.asarray(hs_payload["nbr_shift"], dtype=float), dtype=torch.float32),
        cell_shift=torch.tensor(np.asarray(hs_payload["cell_shift"], dtype=int), dtype=torch.long),
        Hon=torch.tensor(hon[:num_atoms], dtype=torch.float32),
        Hoff=torch.tensor(hon[num_atoms:], dtype=torch.float32),
        iHon=torch.tensor(ihon[:num_atoms], dtype=torch.float32),
        iHoff=torch.tensor(ihon[num_atoms:], dtype=torch.float32),
        Hon0=torch.tensor(hon[:num_atoms].copy(), dtype=torch.float32),
        Hoff0=torch.tensor(hon[num_atoms:].copy(), dtype=torch.float32),
        iHon0=torch.tensor(ihon[:num_atoms].copy(), dtype=torch.float32),
        iHoff0=torch.tensor(ihon[num_atoms:].copy(), dtype=torch.float32),
        overlap=torch.tensor(overlap, dtype=torch.float32),
        Son=torch.tensor(son, dtype=torch.float32),
        Soff=torch.tensor(soff, dtype=torch.float32),
        Lon=torch.tensor(lon, dtype=torch.float32),
        Loff=torch.tensor(loff, dtype=torch.float32),
        spin_length=torch.tensor(spin_length, dtype=torch.float32),
        spin_vec=torch.tensor(spin_vec, dtype=torch.float32),
    )


def save_graphs_npz(graphs, output_file):
    output_path = Path(output_file)
    np.savez(output_path, graph=graphs)
