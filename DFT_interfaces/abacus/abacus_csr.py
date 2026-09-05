# Copyright (c) 2021-2026 HamGNN Team
# SPDX-License-Identifier: GPL-3.0-only

"""Streaming readers for ABACUS 3.10 and 3.11 real-space CSR files."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Iterator, List, Optional, Sequence, Tuple, Union


Number = Union[float, complex]
_FLOAT = r"[+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|inf|nan)"
_COMPLEX_PAIR = re.compile(rf"\(\s*({_FLOAT})\s*,\s*({_FLOAT})\s*\)", re.IGNORECASE)
_RECORD_HEADER = re.compile(r"^\s*([+-]?\d+)\s+([+-]?\d+)\s+([+-]?\d+)\s+(\d+)\s*$")


@dataclass(frozen=True)
class CSRBlock:
    cell_shift: Tuple[int, int, int]
    values: Sequence[Number]
    columns: Sequence[int]
    row_pointers: Sequence[int]


@dataclass(frozen=True)
class CSRStructure:
    """Unit-cell record embedded in an ABACUS 3.11 HContainer CSR header."""

    lattice_name: str
    lattice_constant_angstrom: float
    lattice_vectors: Tuple[Tuple[float, float, float], ...]
    species: Tuple[str, ...]
    atom_counts: Tuple[int, ...]
    coordinate_type: str
    direct_positions: Tuple[Tuple[float, float, float], ...]


class ABACUSCSRFile:
    """Read legacy sparse CSR and ABACUS 3.11 HContainer CSR exports.

    Matrix payloads are streamed one R block at a time. Appended multi-ionic-
    step files are rejected because graph generation must select one structure
    explicitly instead of silently consuming a mixture of geometries.
    """

    def __init__(self, filename: str) -> None:
        self.path = Path(filename)
        if not self.path.is_file() or self.path.stat().st_size == 0:
            raise FileNotFoundError(f"missing or empty ABACUS CSR file: {self.path}")
        self.format_version = ""
        self.label = None
        self.nspin = None
        self.spin_index = None
        self.representation_note = None
        self.structure: Optional[CSRStructure] = None
        self.no_u = 0
        self.ncell_shift = 0
        self._data_offset = 0
        self._read_header()

    @staticmethod
    def _annotated_integer(line: str, annotation: str) -> int:
        value = line.split("#", 1)[0].strip()
        try:
            return int(value)
        except ValueError as exc:
            raise ValueError(f"invalid {annotation} line: {line.rstrip()}") from exc

    def _read_header(self) -> None:
        with self.path.open("r", encoding="utf-8", errors="strict") as handle:
            first = handle.readline()
            if not first:
                raise ValueError(f"empty ABACUS CSR file: {self.path}")
            if "Ionic Step" in first:
                self._read_hcontainer_header(handle, first)
            else:
                self._read_legacy_header(handle, first)

    def _read_legacy_header(self, handle, first: str) -> None:
        self.format_version = "legacy-3.10"
        if "STEP" in first.upper():
            dimension_line = handle.readline()
        else:
            dimension_line = first
        count_line = handle.readline()
        try:
            self.no_u = int(dimension_line.split()[-1])
            self.ncell_shift = int(count_line.split()[-1])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"invalid legacy ABACUS CSR header: {self.path}") from exc
        if self.no_u <= 0 or self.ncell_shift < 0:
            raise ValueError(f"invalid legacy ABACUS CSR dimensions: {self.path}")
        self._data_offset = handle.tell()

    def _read_hcontainer_header(self, handle, first: str) -> None:
        self.format_version = "hcontainer-3.11"
        ionic_step_headers = 1
        saw_csr_format = False
        collect_structure = False
        structure_lines = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"missing CSR Format section: {self.path}")
            stripped = line.strip()
            if "Ionic Step" in line:
                ionic_step_headers += 1
            if "# print " in line and " matrix in real space " in line:
                match = re.search(r"# print\s+(\S+)\s+matrix", line)
                self.label = match.group(1) if match else None
            elif "# number of spin directions" in line:
                self.nspin = self._annotated_integer(line, "number of spin directions")
            elif "# spin index" in line:
                self.spin_index = self._annotated_integer(line, "spin index")
            elif "# number of localized basis" in line:
                self.no_u = self._annotated_integer(line, "number of localized basis")
            elif "# number of Bravais lattice vector R" in line:
                self.ncell_shift = self._annotated_integer(line, "number of R vectors")
                collect_structure = True
            elif stripped.lower().startswith("# representation:"):
                self.representation_note = stripped.split(":", 1)[1].strip()
            elif "CSR Format" in line:
                saw_csr_format = True
            elif saw_csr_format and stripped.startswith("#---"):
                self._data_offset = handle.tell()
                break
            elif collect_structure and stripped and not stripped.startswith("#"):
                structure_lines.append(stripped)

        if ionic_step_headers != 1:
            raise ValueError(f"multiple ionic steps are not supported: {self.path}")
        if self.no_u <= 0 or self.ncell_shift < 0:
            raise ValueError(f"invalid ABACUS 3.11 CSR dimensions: {self.path}")
        if not self.label or not self.nspin or not self.spin_index:
            raise ValueError(f"incomplete ABACUS 3.11 CSR metadata: {self.path}")
        self.structure = self._parse_hcontainer_structure(structure_lines)

    def _parse_hcontainer_structure(self, lines: Sequence[str]) -> CSRStructure:
        """Parse the fixed unit-cell block written by ``UcellIO::write_ucell``."""

        if len(lines) < 8:
            raise ValueError(f"incomplete ABACUS 3.11 CSR structure header: {self.path}")

        lattice_name = lines[0]
        try:
            lattice_constant = float(lines[1])
            lattice_vectors = tuple(
                tuple(float(value) for value in line.split()) for line in lines[2:5]
            )
        except ValueError as exc:
            raise ValueError(f"invalid lattice in ABACUS 3.11 CSR header: {self.path}") from exc
        if (
            not math.isfinite(lattice_constant)
            or lattice_constant <= 0
            or any(len(vector) != 3 for vector in lattice_vectors)
            or any(not math.isfinite(value) for vector in lattice_vectors for value in vector)
        ):
            raise ValueError(f"invalid lattice in ABACUS 3.11 CSR header: {self.path}")

        species = tuple(lines[5].split())
        try:
            atom_counts = tuple(int(value) for value in lines[6].split())
        except ValueError as exc:
            raise ValueError(f"invalid atom counts in ABACUS 3.11 CSR header: {self.path}") from exc
        if not species or len(atom_counts) != len(species) or any(count <= 0 for count in atom_counts):
            raise ValueError(f"invalid species/counts in ABACUS 3.11 CSR header: {self.path}")

        coordinate_type = lines[7]
        if coordinate_type.lower() != "direct":
            raise ValueError(
                f"unsupported coordinate type {coordinate_type!r} in ABACUS 3.11 CSR header: "
                f"{self.path}"
            )
        atom_count = sum(atom_counts)
        if len(lines) != 8 + atom_count:
            raise ValueError(
                f"atom-position count mismatch in ABACUS 3.11 CSR header: expected "
                f"{atom_count}, got {max(0, len(lines) - 8)} in {self.path}"
            )
        try:
            direct_positions = tuple(
                tuple(float(value) for value in line.split()) for line in lines[8:]
            )
        except ValueError as exc:
            raise ValueError(
                f"invalid atomic position in ABACUS 3.11 CSR header: {self.path}"
            ) from exc
        if (
            any(len(position) != 3 for position in direct_positions)
            or any(not math.isfinite(value) for position in direct_positions for value in position)
        ):
            raise ValueError(f"invalid atomic position in ABACUS 3.11 CSR header: {self.path}")

        return CSRStructure(
            lattice_name=lattice_name,
            lattice_constant_angstrom=lattice_constant,
            lattice_vectors=lattice_vectors,
            species=species,
            atom_counts=atom_counts,
            coordinate_type=coordinate_type,
            direct_positions=direct_positions,
        )

    @staticmethod
    def _parse_values(line: str, is_soc: bool) -> List[Number]:
        if is_soc:
            pairs = _COMPLEX_PAIR.findall(line)
            if pairs:
                if _COMPLEX_PAIR.sub("", line).strip():
                    raise ValueError("invalid trailing content in complex CSR values")
                values = [complex(float(real), float(imag)) for real, imag in pairs]
            else:
                raw = line.split()
                if len(raw) % 2:
                    raise ValueError("complex CSR value line has an odd number of components")
                values = [complex(float(raw[i]), float(raw[i + 1])) for i in range(0, len(raw), 2)]
        else:
            if "(" in line or ")" in line:
                pairs = _COMPLEX_PAIR.findall(line)
                if not pairs:
                    raise ValueError("invalid complex CSR values")
                if _COMPLEX_PAIR.sub("", line).strip():
                    raise ValueError("invalid trailing content in complex CSR values")
                if any(float(imag) != 0.0 for _, imag in pairs):
                    raise ValueError("complex ABACUS matrix requires SOC parsing")
                values = [float(real) for real, _ in pairs]
            else:
                values = [float(value) for value in line.split()]
        if any(not math.isfinite(value.real if isinstance(value, complex) else value) for value in values):
            raise ValueError("non-finite real component in ABACUS CSR values")
        if any(isinstance(value, complex) and not math.isfinite(value.imag) for value in values):
            raise ValueError("non-finite imaginary component in ABACUS CSR values")
        return values

    @staticmethod
    def _expect_label(handle, label: str) -> None:
        while True:
            comment = handle.readline()
            if not comment:
                raise ValueError(f"unexpected EOF before {label}")
            if comment.strip() == "":
                continue
            if comment.lstrip().startswith("#"):
                if label.lower() in comment.lower():
                    return
                continue
            raise ValueError(f"missing '{label}' label before: {comment.rstrip()}")

    @staticmethod
    def _read_payload(handle, expected_count: int, parser, label: str):
        payload = []
        while len(payload) < expected_count:
            line = handle.readline()
            if not line:
                raise ValueError(f"unexpected EOF in {label}")
            if line.lstrip().startswith("#"):
                raise ValueError(f"unexpected comment before completing {label}")
            if not line.strip():
                continue
            payload.extend(parser(line))
            if len(payload) > expected_count:
                raise ValueError(
                    f"too many entries in {label}: expected {expected_count}, got {len(payload)}"
                )
        if expected_count == 0:
            position = handle.tell()
            line = handle.readline()
            if line and line.strip():
                handle.seek(position)
        return payload

    def _read_hcontainer_payload(self, handle, is_soc: bool, nnz: int):
        self._expect_label(handle, "CSR values")
        values = self._read_payload(
            handle, nnz, lambda line: self._parse_values(line, is_soc), "CSR values"
        )
        self._expect_label(handle, "CSR column indices")
        columns = self._read_payload(
            handle, nnz, lambda line: [int(value) for value in line.split()], "CSR column indices"
        )
        self._expect_label(handle, "CSR row pointers")
        row_pointers = self._read_payload(
            handle,
            self.no_u + 1,
            lambda line: [int(value) for value in line.split()],
            "CSR row pointers",
        )
        return values, columns, row_pointers

    def _validate_payload(
        self,
        cell_shift: Tuple[int, int, int],
        nnz: int,
        values: Sequence[Number],
        columns: Sequence[int],
        row_pointers: Sequence[int],
    ) -> None:
        prefix = f"{self.path} R={cell_shift}"
        if len(values) != nnz or len(columns) != nnz:
            raise ValueError(
                f"CSR nnz mismatch in {prefix}: header={nnz}, values={len(values)}, columns={len(columns)}"
            )
        if len(row_pointers) != self.no_u + 1:
            raise ValueError(
                f"CSR row-pointer length mismatch in {prefix}: expected {self.no_u + 1}, got {len(row_pointers)}"
            )
        if row_pointers[0] != 0 or row_pointers[-1] != nnz:
            raise ValueError(f"invalid CSR row-pointer endpoints in {prefix}")
        if any(right < left for left, right in zip(row_pointers, row_pointers[1:])):
            raise ValueError(f"non-monotonic CSR row pointers in {prefix}")
        if any(column < 0 or column >= self.no_u for column in columns):
            raise ValueError(f"CSR column index out of range in {prefix}")
        for start, stop in zip(row_pointers, row_pointers[1:]):
            row_columns = columns[start:stop]
            if any(right <= left for left, right in zip(row_columns, row_columns[1:])):
                raise ValueError(f"duplicate or unsorted CSR column indices in {prefix}")

    def iter_blocks(self, is_soc: bool = False) -> Iterator[CSRBlock]:
        count = 0
        with self.path.open("r", encoding="utf-8", errors="strict") as handle:
            handle.seek(self._data_offset)
            while True:
                line = handle.readline()
                if not line:
                    break
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if "Ionic Step" in line:
                    raise ValueError(f"multiple ionic steps are not supported: {self.path}")
                match = _RECORD_HEADER.match(line)
                if not match:
                    raise ValueError(f"invalid CSR R-block header in {self.path}: {stripped}")
                cx, cy, cz, nnz = (int(value) for value in match.groups())
                cell_shift = (cx, cy, cz)
                if self.format_version == "hcontainer-3.11":
                    values, columns, row_pointers = self._read_hcontainer_payload(handle, is_soc, nnz)
                elif nnz == 0:
                    values, columns = [], []
                    row_pointers = [0] * (self.no_u + 1)
                else:
                    value_line = handle.readline()
                    column_line = handle.readline()
                    row_line = handle.readline()
                    if not value_line or not column_line or not row_line:
                        raise ValueError(f"unexpected EOF in legacy CSR block: {self.path}")
                    values = self._parse_values(value_line, is_soc)
                    columns = [int(value) for value in column_line.split()]
                    row_pointers = [int(value) for value in row_line.split()]

                self._validate_payload(cell_shift, nnz, values, columns, row_pointers)
                count += 1
                yield CSRBlock(cell_shift, values, columns, row_pointers)

        if count != self.ncell_shift:
            raise ValueError(
                f"CSR R-block count mismatch in {self.path}: header={self.ncell_shift}, parsed={count}"
            )

    def close(self) -> None:
        """Compatibility no-op; each iterator owns and closes its stream."""
