# Copyright (c) 2021-2026 HamGNN Team
# SPDX-License-Identifier: GPL-3.0-only

"""Parsers for ABACUS inputs, STRU, and sparse Hamiltonian / overlap CSR exports.

Loads SCF metadata, orbital indexing, and matrix elements for graph-data generation
and downstream HamGNN targets.
"""

from copy import deepcopy
import os
import re
import numpy as np
from typing import List, Dict
from scipy.sparse import csr_matrix as csr
from pymatgen.core.periodic_table import Element
from build_graph_from_coordinates import find_inverse_edge_index
from abacus_csr import ABACUSCSRFile

au2ang = 0.5291772490000065
# Match ModuleBase::BOHR_TO_A used by ABACUS 3.11 UcellIO::write_ucell.
abacus_csr_bohr_to_angstrom = 0.5291770
ry2ha  = 13.60580 / 27.21138506
float_pattern = re.compile(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?')

def convert_to_int(value):
    """
    Convert a value or a collection of values to integers.

    Parameters:
    value: int, list, or np.ndarray
        The value to convert. Can be a single integer, a list of values, or a NumPy array.

    Returns:
    int, list, or np.ndarray
        The converted integer value(s). If the input is a list or array, it returns the converted values in the same structure.
    """
    if isinstance(value, list):
        return [convert_to_int(item) for item in value]
    elif isinstance(value, np.ndarray):
        return value.astype(int).tolist()
    elif isinstance(value, (np.int32, np.int64)):
        return int(value)
    return value

def convert_to_float(value):
    """
    Convert a value or a collection of values to floats.

    Parameters:
    value: float, list, or np.ndarray
        The value to convert. Can be a single float, a list of values, or a NumPy array.

    Returns:
    float, list, or np.ndarray
        The converted float value(s). If the input is a list or array, it returns the converted values in the same structure.
    """
    if isinstance(value, list):
        return [convert_to_float(item) for item in value]
    elif isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    elif isinstance(value, (np.float32, np.float64)):
        return float(value)
    return value

def convert_complex(value):
    """
    Convert a complex number or a collection of complex numbers to their real and imaginary parts.

    Parameters:
    value: complex, list, or np.ndarray
        The complex value to convert. Can be a single complex number, a list of complex values, or a NumPy array.

    Returns:
    tuple
        A tuple containing lists of real and imaginary parts of the complex value(s).
    """
    def extract_real(value):
        if isinstance(value, list):
            return [extract_real(item) for item in value]
        elif isinstance(value, np.ndarray):
            return value.real.tolist()
        elif isinstance(value, (np.complex64, np.complex128)):
            return float(value.real)
        return value

    def extract_imaginary(value):
        if isinstance(value, list):
            return [extract_imaginary(item) for item in value]
        elif isinstance(value, np.ndarray):
            return value.imag.tolist()
        elif isinstance(value, (np.complex64, np.complex128)):
            return float(value.imag)
        return value

    real_part = extract_real(deepcopy(value))
    imaginary_part = extract_imaginary(deepcopy(value))
    return real_part, imaginary_part

def find_matching_column_index(matrix, target_column_values):
    """
    Find the index of a column in a 2D numpy array that exactly matches a given target column of values.

    This function compares each column of the matrix with the target column values and returns the index of
    the first column that matches. If no match is found, it returns None.

    Parameters:
        matrix (np.ndarray): A 2D numpy array (N×5), where N is the number of rows and 5 is the number of columns.
        target_column_values (list or np.ndarray): A 1D array or list containing the target column values to match.

    Returns:
        int or None: The index of the matching column if found, otherwise None.
    """
    # Ensure the target column is a numpy array for consistency and correct shape
    target_column_values = np.asarray(target_column_values)

    # Validate that the target column has the same number of rows as the matrix
    if target_column_values.shape[0] != matrix.shape[0]:
        raise ValueError("The number of elements in the target column must match the number of rows in the matrix.")

    # Compare each column of the matrix with the target column using broadcasting
    column_matches = np.all(matrix == target_column_values[:, None], axis=0)

    # If a match is found, return the index of the first matching column; otherwise, return None
    return np.argmax(column_matches) if column_matches.any() else None

class STRU:
    """
    Class to read and store atomic and lattice information from a file.

    Supports two input formats:
    1) STRU file (blocks such as ATOMIC_SPECIES, LATTICE_CONSTANT, LATTICE_VECTORS, ATOMIC_POSITIONS)
    2) ABACUS running_scf.log (READING UNITCELL info from OUT.ABACUS/running_scf.log)

    A validated ABACUS 3.11 HContainer CSR header can subsequently replace
    only the geometry through :meth:`use_csr_geometry`; log-only orbital and
    pseudopotential metadata are retained.

    Attributes:
        species (List[str]): List of species (element types).
        num_orbitals (List[int]): List of number of orbitals for each species.
        num_atoms_per_species (List[int]): List of number of atoms for each species.
        cell (ndarray): Lattice vectors (3x3 matrix).
        positions (ndarray): Atomic positions (Nx3 matrix).
        atomic_numbers (ndarray): Atomic numbers for each atom (1D array).
        num_species (int): Number of unique species.
        num_atoms_unit_cell (int): Total number of atoms in the unit cell.

    Methods:
        __init__(file: str) -> None: Initializes the structure by reading data from the given file.
    """

    def __init__(self, file: str) -> None:
        """
        Initialize the structure by reading data from the specified file.
        Automatically detects STRU or running_scf.log format.

        Args:
            file (str): Path to the input file (STRU or OUT.ABACUS/running_scf.log).
        """
        with open(file, 'r', encoding='utf-8', errors='replace') as fp:
            content_preview = fp.read(4096)
            fp.seek(0)
            is_log = (
                'lattice constant (bohr)' in content_preview.lower()
                or 'READING UNITCELL INFORMATION' in content_preview
            )

        if is_log:
            self._read_from_running_scf_log(file)
        else:
            self._read_from_stru_file(file)

        self.num_species = len(self.species)
        self.num_atoms_unit_cell = sum(self.num_atoms_per_species)
        self.atomic_numbers = np.array(
            [
                Element(spec).Z
                for spec, count in zip(self.species, self.num_atoms_per_species)
                for _ in range(count)
            ],
            dtype=int,
        )
        self.structure_source = file
        self.structure_format = 'running-scf-log' if is_log else 'STRU'

    @staticmethod
    def _geometry_from_csr(csr_file: ABACUSCSRFile):
        structure = csr_file.structure
        if structure is None:
            raise ValueError(f'ABACUS CSR does not contain a structure header: {csr_file.path}')

        cell = (
            np.asarray(structure.lattice_vectors, dtype=float)
            * structure.lattice_constant_angstrom
            / abacus_csr_bohr_to_angstrom
        )
        if not np.all(np.isfinite(cell)) or abs(np.linalg.det(cell)) < 1e-12:
            raise ValueError(f'invalid cell in ABACUS CSR structure header: {csr_file.path}')
        direct_positions = np.asarray(structure.direct_positions, dtype=float)
        positions = direct_positions @ cell
        return structure, cell, direct_positions, positions

    def validate_csr_geometry(self, csr_file: ABACUSCSRFile):
        """Validate one 3.11 CSR unit cell against the current structure."""

        structure, cell, direct_positions, positions = self._geometry_from_csr(csr_file)
        if list(structure.species) != self.species:
            raise ValueError(
                f'species mismatch between structure metadata {self.species} and '
                f'CSR {list(structure.species)}: {csr_file.path}'
            )
        if list(structure.atom_counts) != self.num_atoms_per_species:
            raise ValueError(
                f'atom-count mismatch between structure metadata {self.num_atoms_per_species} '
                f'and CSR {list(structure.atom_counts)}: {csr_file.path}'
            )
        if np.shape(self.cell) != (3, 3) or np.shape(self.positions) != positions.shape:
            raise ValueError(f'invalid structure metadata dimensions for CSR: {csr_file.path}')
        # ABACUS 3.11 writes lattice scale/vectors with the C++ default of six
        # significant digits, so the product can carry about 1e-5 relative error.
        if not np.allclose(self.cell, cell, rtol=1.1e-5, atol=1e-6):
            error = float(np.max(np.abs(np.asarray(self.cell) - cell)))
            raise ValueError(
                f'cell mismatch between structure metadata and CSR '
                f'(max_abs={error:.6g} Bohr): {csr_file.path}'
            )

        metadata_direct = np.linalg.solve(
            np.asarray(self.cell).T, np.asarray(self.positions).T
        ).T
        fractional_delta = metadata_direct - direct_positions
        fractional_delta -= np.rint(fractional_delta)
        position_error = float(np.max(np.linalg.norm(fractional_delta @ self.cell, axis=1)))
        if position_error > 1e-5:
            raise ValueError(
                f'atomic-position mismatch between structure metadata and CSR '
                f'(max_periodic_distance={position_error:.6g} Bohr): {csr_file.path}'
            )
        return cell, positions

    def use_csr_geometry(self, csr_file: ABACUSCSRFile) -> None:
        """Replace only cell/positions with the validated 3.11 CSR geometry."""

        cell, positions = self.validate_csr_geometry(csr_file)
        self.cell = cell
        self.positions = positions
        self.pos_type = 'cartesian'
        self.structure_source = str(csr_file.path)
        self.structure_format = csr_file.format_version

    def _read_from_running_scf_log(self, file: str) -> None:
        """Read structure information from ABACUS running_scf.log."""
        self.species = []
        self.num_orbitals = []
        self.num_atoms_per_species = []
        self.valence_electrons_per_species = []
        self.cell = []
        self.positions = []
        self.pos_type = 'cartesian'
        latconst = 1.0

        with open(file, 'r', encoding='utf-8', errors='replace') as fp:
            lines = fp.readlines()

        i = 0
        positions_are_direct = False
        while i < len(lines):
            line = lines[i]

            valence_match = re.search(
                r'valence electrons\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))',
                line,
                re.IGNORECASE,
            )
            if valence_match:
                self.valence_electrons_per_species.append(float(valence_match.group(1)))

            if 'lattice constant (bohr)' in line.lower() and '=' in line:
                latconst = float(line.split('=')[-1].strip())

            if 'reading atom type' in line.lower() and 'atom label' not in line.lower():
                i += 1
                if i >= len(lines):
                    break

                while i < len(lines) and 'atom label' not in lines[i].lower():
                    i += 1
                if i >= len(lines):
                    break

                spec = lines[i].split('=')[-1].strip()
                self.species.append(spec)

                zeta_per_l = [0, 0, 0, 0]
                i += 1
                while i < len(lines) and re.match(r'\s*L=\d+,\s*number of zeta', lines[i]):
                    l_match = re.search(r'L=(\d+)', lines[i])
                    z_match = re.search(r'zeta\s*=\s*(\d+)', lines[i])
                    if l_match and z_match:
                        l_value = int(l_match.group(1))
                        zeta = int(z_match.group(1))
                        if l_value < 4:
                            zeta_per_l[l_value] = zeta
                    i += 1

                num_orbitals = (
                    zeta_per_l[0] * 1
                    + zeta_per_l[1] * 3
                    + zeta_per_l[2] * 5
                    + zeta_per_l[3] * 7
                )
                self.num_orbitals.append(num_orbitals)

                while i < len(lines) and not re.search(
                    r'number of atoms? for this type', lines[i], re.IGNORECASE
                ):
                    i += 1
                if i < len(lines):
                    num_atoms = int(lines[i].split('=')[-1].strip())
                    self.num_atoms_per_species.append(num_atoms)
                i += 1
                continue

            if 'CARTESIAN COORDINATES' in line and 'UNIT' in line:
                self.positions = []
                i += 1
                while i < len(lines):
                    parts = lines[i].split()
                    if len(parts) >= 4:
                        try:
                            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                            self.positions.append([x, y, z])
                        except ValueError:
                            pass
                    i += 1
                    if self.positions and len(self.positions) == sum(self.num_atoms_per_species):
                        break
                positions_are_direct = False

            elif 'DIRECT COORDINATES' in line and 'K-POINTS' not in line:
                self.positions = []
                i += 1
                while i < len(lines):
                    parts = lines[i].split()
                    if len(parts) >= 4:
                        try:
                            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                            self.positions.append([x, y, z])
                        except ValueError:
                            pass
                    i += 1
                    if self.positions and len(self.positions) == sum(self.num_atoms_per_species):
                        break
                positions_are_direct = True
                self.pos_type = 'direct'

            if 'Lattice vectors:' in line and 'Cartesian' in line:
                self.cell = []
                for j in range(1, 4):
                    if i + j < len(lines):
                        vec = list(map(float, float_pattern.findall(lines[i + j])))
                        if len(vec) >= 3:
                            self.cell.append(vec[:3])
                i += 3

            i += 1

        self.cell = np.array(self.cell) if len(self.cell) == 3 else np.zeros((3, 3))
        self.cell = self.cell * latconst

        if positions_are_direct and len(self.cell) == 3 and self.positions:
            self.positions = np.array(self.positions)
            self.positions = self.positions @ self.cell
            self.pos_type = 'cartesian'
        else:
            self.positions = np.array(self.positions) if self.positions else np.zeros((0, 3))
            self.positions = self.positions * latconst
            self.pos_type = 'cartesian'

    def _read_from_stru_file(self, file: str) -> None:
        """Read structure information from STRU file."""
        with open(file, 'r', encoding='utf-8', errors='replace') as fp:
            self.species = []
            self.num_orbitals = []
            self.num_atoms_per_species = []
            self.valence_electrons_per_species = None
            self.cell = []
            self.positions = []
            active_block = None
            latconst = 1.0

            for line in fp:
                line = line.split('//')[0].split('#')[0].strip()
                if not line:
                    continue

                if 'ATOMIC_SPECIES' in line:
                    active_block = 'ATOMIC_SPECIES'
                elif 'NUMERICAL_ORBITAL' in line:
                    active_block = 'NUMERICAL_ORBITAL'
                elif 'LATTICE_CONSTANT' in line:
                    active_block = 'LATTICE_CONSTANT'
                elif 'LATTICE_VECTORS' in line:
                    active_block = 'LATTICE_VECTORS'
                elif 'ATOMIC_POSITIONS' in line:
                    active_block = 'ATOMIC_POSITIONS'

                elif active_block == 'ATOMIC_SPECIES':
                    self.species.append(line.split()[0])

                elif active_block == 'NUMERICAL_ORBITAL':
                    orbital_data = line.split('.orb')[0].split('_')[-1]
                    num_orbitals = self.parse_orbitals(orbital_data)
                    self.num_orbitals.append(num_orbitals)

                elif active_block == 'LATTICE_CONSTANT':
                    latconst = float(line)

                elif active_block == 'LATTICE_VECTORS':
                    lattice_vector = list(map(float, line.split()))
                    self.cell.append(lattice_vector)

                elif active_block == 'ATOMIC_POSITIONS':
                    self._process_atomic_positions(fp, line)

        self.cell = np.array(self.cell) * latconst
        self.positions = np.array(self.positions)

        if self.pos_type == 'direct':
            self.positions = self.convert_to_cartesian()
            self.pos_type = 'cartesian'
        else:
            self.positions = self.positions * latconst
            self.pos_type = 'cartesian'

    def parse_orbitals(self, orbital_data: str) -> int:
        """
        Parse the orbital string and calculate the total number of orbitals for the species.

        Args:
            orbital_data (str): The string containing orbital information.

        Returns:
            int: The total number of orbitals for the species.
        """
        s, p, d, f = 0, 0, 0, 0
        if 's' in orbital_data:
            s = int(re.findall(r'\d', orbital_data)[0]) * 1
        if 'p' in orbital_data:
            p = int(re.findall(r'\d', orbital_data)[1]) * 3
        if 'd' in orbital_data:
            d = int(re.findall(r'\d', orbital_data)[2]) * 5
        if 'f' in orbital_data:
            f = int(re.findall(r'\d', orbital_data)[3]) * 7
        return s + p + d + f

    def _process_atomic_positions(self, fp, line):
        """
        Process atomic positions block and parse relevant data.

        Args:
            fp (file): File pointer to read atomic data.
            line (str): Line read from the file containing atomic position info.
            latconst (float): Lattice constant used for scaling.
        """
        self.pos_type = line.strip().lower()
        for is_ in range(len(self.num_orbitals)):
            # element
            while True:
                line = fp.readline().split('//')[0].split('#')[0]
                if line.strip() == '':
                    continue
                break
            # mag
            while True:
                line = fp.readline().split('//')[0].split('#')[0]
                if line.strip() == '':
                    continue
                break
            # num
            while True:
                line = fp.readline().split('//')[0].split('#')[0]
                if line.strip() == '':
                    continue
                na = int(line)
                self.num_atoms_per_species.append(na)
                break
            # pos
            ia = 0
            while ia < na:
                line = fp.readline().split('//')[0].split('#')[0]
                if line.strip() == '':
                    continue
                tmp = line.split()
                self.positions.append([float(tmp[0]), float(tmp[1]), float(tmp[2])])
                ia += 1

    def convert_to_cartesian(self) -> np.ndarray:
        """
        Convert atomic positions from direct to Cartesian coordinates.

        Returns:
            np.ndarray: The atomic positions in Cartesian coordinates.
        """
        cartesian_positions = np.dot(self.positions, self.cell)
        return cartesian_positions

class ABACUSHS:
    """
    A class to handle the ABACUS Hamiltonian structure and related operations.

    Attributes:
        no_u (int): Number of orbitals in the unit cell.
        ncell_shift (int): Number of cell shifts.
        max_rcut (ndarray): Maximum cutoff distance for each species.
        noff (int): Number of off-site Hamiltonian terms.
        fp (file object): File pointer for reading input data.

    Methods:
        __init__(file: str): Initialize the ABACUSHS class by reading data from the specified file.
        getGraph(stru: STRU, graph: dict, skip: bool, isH: bool, isSOC: bool, calcRcut: bool, tojson: bool):
            Constructs and returns the graph (edges, Hamiltonian matrices, etc.) from the ABACUSHS data.
        getHK(stru: STRU, k: np.ndarray, isH: bool, isSOC: bool): Returns the Hamiltonian matrix for the specified k-point.
        close(): Closes the file pointer.
    """

    def __init__(self, file: str) -> None:
        """
        Initializes the ABACUSHS object by reading the data from the provided file.

        Args:
            file (str): The file containing the ABACUSHS data.
        """
        self.csr_file = ABACUSCSRFile(file)
        if self.csr_file.representation_note:
            raise ValueError(
                f'folded ABACUS matrix representation is not a real-space graph input: '
                f'{self.csr_file.representation_note}'
            )
        self.no_u = self.csr_file.no_u
        self.ncell_shift = self.csr_file.ncell_shift

    def _calculate_atom_orbitals(self, stru, repeat):
        """
        Calculate the number of orbitals for each atom and generate the corresponding indices.

        Parameters:
        stru (object): A structure object that contains:
            - species (list): A list of species.
            - num_atoms_per_species (list): A list with the number of atoms for each species.
            - num_orbitals (list): A list with the number of orbitals for each species.
        repeat (int): A scalar to multiply the calculated orbital counts by, typically used for scaling the number of orbitals.

        Returns:
        numpy.ndarray: Array of orbital counts for each atom, scaled by `repeat`.
        numpy.ndarray: Cumulative indices for each atom, based on orbital counts.
        """
        # Initialize a list to store the number of orbitals for each atom
        orbitals_per_atom = []

        # Loop through each species to compute the total number of orbitals per atom
        for species_idx in range(len(stru.species)):
            num_atoms = stru.num_atoms_per_species[species_idx]
            num_orbitals = stru.num_orbitals[species_idx]
            orbitals_per_atom += [num_orbitals] * num_atoms  # Repeat the orbital count for each atom of this species

        # Convert to numpy array and scale by repeat factor
        orbitals_per_atom = np.array(orbitals_per_atom, dtype=int) * repeat

        # Check if the total number of orbitals matches the expected value
        if orbitals_per_atom.sum() != self.no_u:
            print("STRU parse error! Mismatch in total number of orbitals.")
            raise RuntimeError("Total number of orbitals mismatch")

        # Create an array to store cumulative orbital indices for each atom
        orbital_indices = np.zeros_like(orbitals_per_atom, dtype=int)

        # Fill in the cumulative indices (skip the first atom, hence starting from index 1)
        orbital_indices[1:] = np.cumsum(orbitals_per_atom[:-1])

        return orbitals_per_atom, orbital_indices

    def getGraph(self, stru, graph: dict = {}, skip: bool = False, isH: bool = False,
                    isSOC: bool = False, calcRcut: bool = False, tojson: bool = False) -> dict:
        """
        Constructs the graph (edges, Hamiltonian matrices, etc.) from ABACUSHS data.

        Args:
            stru (STRU): The structure object containing atomic information.
            graph (dict, optional): The graph object to update, defaults to an empty dictionary.
            skip (bool, optional): If True, skip the Hamiltonian calculations, defaults to False.
            isH (bool, optional): If True, scales the Hamiltonian by `ry2ha`, defaults to False.
            isSOC (bool, optional): If True, includes spin-orbit coupling, defaults to False.
            calcRcut (bool, optional): If True, calculates the maximum cutoff distances, defaults to False.
            tojson (bool, optional): If True, converts the graph to JSON format, defaults to False.

        Returns:
            dict: The constructed graph containing edge information and Hamiltonian matrices.
        """
        assert (not graph and not skip) or (graph and skip)

        dtype = np.float32 if not isSOC else np.complex64
        repeat = 1 if not isSOC else 2
        nspin = 1 if not isSOC else 4
        edge_idx_src, edge_idx_dst, cell_shift, nbr_shift = [], [], [], []
        Hon = [[]] if not isSOC else [[], [], [], []]  # Cannot be written as [[]]*4
        Hoff = [[]] if not isSOC else [[], [], [], []]

        if skip:
            # Load pre-existing graph data
            graph_ = deepcopy(graph)
            self.noff = len(graph_['inv_edge_idx'])
            edge_idx_src = graph_['edge_index'][0]
            edge_idx_dst = graph_['edge_index'][1]
            cell_shift = graph_['cell_shift']
            Hoff = graph_['Hoff']
            for ispin in range(nspin):
                for ioff in range(self.noff):
                    Hoff[ispin][ioff] = np.zeros_like(Hoff[ispin][ioff], dtype=dtype)

        # Initialize the atomic orbital indices
        no, indo = self._calculate_atom_orbitals(stru, repeat)

        for block in self.csr_file.iter_blocks(is_soc=isSOC):
            cx, cy, cz = block.cell_shift
            hamilt = csr(
                (block.values, block.columns, block.row_pointers),
                shape=[self.no_u, self.no_u],
                dtype=dtype,
            )

            if isH:
                hamilt *= ry2ha

            if skip:
                edge_info_array = np.concatenate([np.array(graph_['edge_index']), np.array(cell_shift).T], axis=0)
            else:
                edge_info_array = None

            # Process Hamiltonian and populate graph data
            for ia in range(stru.num_atoms_unit_cell):
                for ja in range(stru.num_atoms_unit_cell):
                    ham = hamilt[indo[ia]:indo[ia] + no[ia], indo[ja]:indo[ja] + no[ja]]
                    if ia == ja and cx == 0 and cy == 0 and cz == 0:
                        # Onsite Hamiltonian
                        if not isSOC:
                            Hon[0].append(ham.toarray().flatten())
                        else:
                            Hon[0].append(ham[0::2, 0::2].toarray().flatten())  # uu
                            Hon[1].append(ham[0::2, 1::2].toarray().flatten())  # ud
                            Hon[2].append(ham[1::2, 0::2].toarray().flatten())  # du
                            Hon[3].append(ham[1::2, 1::2].toarray().flatten())  # dd
                    elif ham.getnnz() > 0:
                        # Offsite Hamiltonian
                        if not skip:
                            if not isSOC:
                                Hoff[0].append(ham.toarray().flatten())
                            else:
                                Hoff[0].append(ham[0::2, 0::2].toarray().flatten())  # uu
                                Hoff[1].append(ham[0::2, 1::2].toarray().flatten())  # ud
                                Hoff[2].append(ham[1::2, 0::2].toarray().flatten())  # du
                                Hoff[3].append(ham[1::2, 1::2].toarray().flatten())  # dd
                            edge_idx_src.append(ia)
                            edge_idx_dst.append(ja)
                            cell_shift.append(np.array([cx, cy, cz], dtype=int))
                            nbr_shift.append(np.array([cx, cy, cz]) @ stru.cell)
                        else:
                            ierr, ioff = self._fill_offsite_hamiltonian(
                                cx, cy, cz, ia, ja, edge_info_array
                            )
                            if ierr:
                                continue
                            if not isSOC:
                                Hoff[0][ioff] = ham.toarray().flatten()
                            else:
                                Hoff[0][ioff] = ham[0::2, 0::2].toarray().flatten()  # uu
                                Hoff[1][ioff] = ham[0::2, 1::2].toarray().flatten()  # ud
                                Hoff[2][ioff] = ham[1::2, 0::2].toarray().flatten()  # du
                                Hoff[3][ioff] = ham[1::2, 1::2].toarray().flatten()  # dd

        if calcRcut:
            self._calculate_rcut(stru, edge_idx_src, edge_idx_dst, cell_shift)

        if not skip:
            # Construct the edges and graph
            edge_index = [edge_idx_src, edge_idx_dst]
            self.noff = len(edge_idx_src)

            inv_edge_idx = find_inverse_edge_index(np.array(edge_index), np.array(cell_shift))

            graph_ = {}
            graph_['edge_index'] = edge_index if tojson else np.array(edge_index)
            graph_['inv_edge_idx'] = convert_to_int(inv_edge_idx) if tojson else inv_edge_idx
            graph_['cell_shift'] = convert_to_int(cell_shift) if tojson else np.array(cell_shift)
            graph_['nbr_shift'] = convert_to_float(nbr_shift) if tojson else np.array(nbr_shift)
            graph_['pos'] = convert_to_float(stru.positions) if tojson else stru.positions

        if not tojson:
            graph_['Hon'] = Hon
            graph_['Hoff'] = Hoff
        else:
            if not isSOC:
                graph_['Hon'] = convert_to_float(Hon)
                graph_['Hoff'] = convert_to_float(Hoff)
            else:
                graph_['Hon'], graph_['iHon'] = convert_complex(Hon)
                graph_['Hoff'], graph_['iHoff'] = convert_complex(Hoff)

        return graph_

    def _fill_offsite_hamiltonian(self, cx, cy, cz, ia, ja, edge_info_array):
        """
        Checks if an offsite Hamiltonian term already exists and returns the appropriate index.
        """
        ioff = find_matching_column_index(edge_info_array, [ia, ja, cx, cy, cz])

        if ioff is not None:
            return False, ioff
        else:
            return True, ioff

    def _calculate_rcut(self, stru, edge_idx_src, edge_idx_dst, cell_shift):
        """
        Calculates the maximum cutoff distance for each species.
        """
        self.max_rcut = np.zeros((len(stru.species), len(stru.species)))
        isa = np.zeros(stru.num_atoms_unit_cell, dtype=int)
        num = 0
        for is_ in range(len(stru.species)):
            for ia in range(stru.num_atoms_per_species[is_]):
                isa[num] = is_
                num += 1

        for ia, ja, cs in zip(edge_idx_src, edge_idx_dst, cell_shift):
            # Only calculate for atoms of the same species
            if isa[ia] != isa[ja]:
                continue
            distance = np.linalg.norm(stru.positions[ja] - stru.positions[ia] + (cs @ stru.cell.T))
            self.max_rcut[isa[ia], isa[ja]] = max(distance, self.max_rcut[isa[ia], isa[ja]])
            self.max_rcut[isa[ja], isa[ia]] = max(distance, self.max_rcut[isa[ja], isa[ia]])

    def getHK(self, stru, k: np.ndarray = np.array([0, 0, 0]), isH: bool = False, isSOC: bool = False):
        """
        Returns the Hamiltonian matrix for the specified k-point.

        Args:
            stru (STRU): The structure object containing atomic information.
            k (np.ndarray, optional): The k-point for which to calculate the Hamiltonian, defaults to [0,0,0].
            isH (bool, optional): If True, scales the Hamiltonian by `ry2ha`, defaults to False.
            isSOC (bool, optional): If True, includes spin-orbit coupling, defaults to False.

        Returns:
            np.ndarray: The Hamiltonian matrix for the specified k-point.
        """
        assert np.all(k == 0)  # Only support gamma point

        dtype = np.float32 if not isSOC else np.complex64
        HK = np.zeros([self.no_u, self.no_u], dtype=dtype)

        for block in self.csr_file.iter_blocks(is_soc=isSOC):
            hamilt = csr(
                (block.values, block.columns, block.row_pointers),
                shape=[self.no_u, self.no_u],
                dtype=dtype,
            )
            if isH:
                hamilt *= ry2ha

            HK += hamilt

        return HK

    def close(self):
        """
        Closes the file pointer.
        """
        self.csr_file.close()


def select_structure_for_matrices(
    crystal: STRU,
    primary_matrix: ABACUSHS,
    *other_matrices: ABACUSHS,
) -> STRU:
    """Select geometry by matrix generation while retaining log-only metadata.

    Legacy 3.10 matrices keep the historical ``running_scf.log`` geometry.
    For a 3.11 primary matrix, the embedded HContainer CSR unit cell becomes
    authoritative. Any additional 3.11 matrices must describe the same cell.
    """

    primary_csr = primary_matrix.csr_file
    if primary_csr.format_version == 'hcontainer-3.11':
        crystal.use_csr_geometry(primary_csr)
    elif primary_csr.format_version == 'legacy-3.10':
        crystal.structure_format = 'legacy-3.10-running-log'
    else:
        raise ValueError(f'unsupported ABACUS CSR format: {primary_csr.format_version}')

    for matrix in other_matrices:
        csr_file = matrix.csr_file
        if csr_file.format_version == 'hcontainer-3.11':
            crystal.validate_csr_geometry(csr_file)
    return crystal

def read_abacus_input(input_file: str) -> dict:
    """
    Read ABACUS INPUT file and extract electron-related parameters.

    Parameters:
        input_file (str): Path to the ABACUS INPUT file.

    Returns:
        dict: Dictionary containing:
            - 'nelec': Total number of electrons (if specified)
            - 'nelec_delta': Change in number of electrons (if specified)
            - 'doping_charge': Reserved field for downstream compatibility
    """
    result = {
        'nelec': None,
        'nelec_delta': None,
        'doping_charge': None
    }

    if not os.path.exists(input_file):
        return result

    with open(input_file, 'r') as f:
        for line in f:
            line = line.split('//')[0].split('#')[0].strip()
            if not line:
                continue

            line_lower = line.lower()

            if 'nelec_delta' in line_lower:
                try:
                    result['nelec_delta'] = float(line.split()[-1])
                except:
                    pass

            elif 'nelec' in line_lower and 'nelec_delta' not in line_lower:
                try:
                    result['nelec'] = float(line.split()[-1])
                except:
                    pass

    return result


def get_neutral_electrons(stru: STRU) -> float:
    """
    Calculate the number of valence electrons in a neutral system.

    Parameters:
        stru (STRU): STRU object containing atomic information.

    Returns:
        float: Total pseudopotential valence-electron count.
    """
    valence = getattr(stru, 'valence_electrons_per_species', None)
    if valence is None or len(valence) != len(stru.species):
        raise ValueError(
            'missing pseudopotential valence-electron metadata in ABACUS running log; '
            'refusing periodic-table fallback for charge labels'
        )
    return float(sum(electrons * count for electrons, count in zip(
        valence, stru.num_atoms_per_species
    )))


def calculate_doping_charge(input_params: dict, neutral_electrons: float) -> float:
    """
    Calculate the doping charge from INPUT parameters.

    Priority:
    1. If both nelec and nelec_delta are present and nelec != 0, use
       doping_charge = neutral_electrons - (nelec + nelec_delta)
    2. If only nelec is effectively set (nelec != 0), use
       doping_charge = neutral_electrons - nelec
    3. If only nelec_delta is effectively set, use -nelec_delta
    4. Otherwise, return 0.0 (neutral system)

    Notes:
    - ABACUS may write a default `nelec_delta = 0` into `OUT.ABACUS/INPUT`
      even when the user only specified `nelec`.
    - ABACUS may also write a default `nelec = 0` into `OUT.ABACUS/INPUT`
      when the user specified only `nelec_delta`.
    - To avoid misinterpreting these defaults, `nelec == 0` is treated as
      "not explicitly set" in this function.

    Parameters:
        input_params (dict): Output from read_abacus_input().
        neutral_electrons (float): Pseudopotential valence-electron count of the neutral system.

    Returns:
        float: Doping charge (positive = hole doping, negative = electron doping).
    """
    nelec = input_params.get('nelec')
    nelec_delta = input_params.get('nelec_delta')

    has_effective_nelec = nelec is not None and not np.isclose(nelec, 0.0)
    has_nelec_delta = nelec_delta is not None

    if has_effective_nelec and has_nelec_delta:
        return float(neutral_electrons - (nelec + nelec_delta))
    elif has_effective_nelec:
        return float(neutral_electrons - nelec)
    elif has_nelec_delta:
        return float(-input_params['nelec_delta'])
    else:
        return 0.0
