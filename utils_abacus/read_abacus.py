'''
Descripttion: 
version: 
Author: ChangWei Zhang & Yang Zhong
Date: 2023-01-16 13:00:43
Last Modified by:   Yang Zhong
Last Modified time: 2025-02-6 10:34:01 
'''
import numpy as np
from copy import deepcopy
import json
import os
import re
import sys
import numpy as np
from typing import List, Dict
from scipy.sparse import csr_matrix as csr
from pymatgen.core.periodic_table import Element
from build_graph_from_coordinates import find_inverse_edge_index

au2ang = 0.5291772490000065
ry2ha  = 13.60580 / 27.21138506

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
        
        Args:
            file (str): Path to the input file containing atomic and lattice information.
        """
        # Open the file and initialize attributes
        with open(file, 'r') as fp:
            self.species = []
            self.num_orbitals = []  # Number of orbitals per species.
            self.num_atoms_per_species = []  # Number of atoms per species.
            self.cell = []  # Lattice vectors.
            self.positions = []  # Atomic positions.
            
            active_block = None
            latconst = 1.0  # Default lattice constant
            
            # Process file line by line
            for line in fp:
                line = line.split('//')[0].split('#')[0].strip()  # Remove comments and whitespace
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
                    # Parse the number of orbitals for each species
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

        # After reading all lines, finalize the data
        self.num_species = len(self.species)
        self.num_atoms_unit_cell = sum(self.num_atoms_per_species)
        self.cell = np.array(self.cell) * latconst
        self.positions = np.array(self.positions)
        
        # Convert positions to Cartesian if they are in 'direct' format
        if self.pos_type == 'direct':
            self.positions = self.convert_to_cartesian()
        else:
            self.positions = self.positions * latconst
            
        # Generate atomic numbers (Z values)
        self.atomic_numbers = np.array([Element(spec).Z for spec, na in zip(self.species, self.num_atoms_per_species) for _ in range(na)], dtype=int)

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
        self.fp = open(file)
        line = self.fp.readline()  # Read first line to determine the format
        if 'STEP' in line:
            self.no_u = int(self.fp.readline().split()[-1])
        else:
            self.no_u = int(line.split()[-1])  # Number of orbitals in the unit cell.
        self.ncell_shift = int(self.fp.readline().split()[-1])

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

        while True:
            line = self.fp.readline()
            if not line:
                break
            tmp = line.split()
            cx, cy, cz = int(tmp[0]), int(tmp[1]), int(tmp[2])
            nh = int(tmp[3])
            if nh == 0:
                continue
            val = self.fp.readline()
            col = self.fp.readline().split()
            row = self.fp.readline().split()

            # Handle Hamiltonian values
            if not isSOC:
                val = list(map(float, val.split()))
            else:
                val_raw = re.findall(r'[\-\+\d\.eE]+', val)
                val_raw = np.asarray(val_raw, dtype=np.float32)
                val = np.zeros(len(val_raw) // 2, dtype=np.complex64)
                val += val_raw[0::2] + 1j * val_raw[1::2]
            
            col = list(map(int, col))
            row = list(map(int, row))
            hamilt = csr((val, col, row), shape=[self.no_u, self.no_u], dtype=dtype)

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

        while True:
            line = self.fp.readline()
            if not line:
                break
            tmp = line.split()
            cx, cy, cz = int(tmp[0]), int(tmp[1]), int(tmp[2])
            nh = int(tmp[3])
            if nh == 0:
                continue
            val = self.fp.readline()
            col = self.fp.readline().split()
            row = self.fp.readline().split()

            # Handle Hamiltonian values
            if not isSOC:
                val = list(map(float, val.split()))
            else:
                val_raw = re.findall(r'[\-\+\d\.eE]+', val)
                val_raw = np.asarray(val_raw, dtype=np.float32)
                val = np.zeros(len(val_raw) // 2, dtype=np.complex64)
                val += val_raw[0::2] + 1j * val_raw[1::2]
            
            col = list(map(int, col))
            row = list(map(int, row))
            hamilt = csr((val, col, row), shape=[self.no_u, self.no_u], dtype=dtype)
            if isH:
                hamilt *= ry2ha

            HK += hamilt

        return HK

    def close(self):
        """
        Closes the file pointer.
        """
        self.fp.close()

def process_graph_data():
    """
    Processes the Hamiltonian and overlap data, creates a merged graph, 
    and saves the graph data to a formatted JSON file.
    """
    # Load structure and Hamiltonian
    poscar = STRU(os.path.join('/public/home/zhongyang/yzhong/CsVSb/50/Training/perturbation_cal/STRU_1', 'STRU'))
    H = ABACUSHS(os.path.join('/public/home/zhongyang/yzhong/CsVSb/50/Training/perturbation_cal/STRU_1/OUT.ABACUS', 'data-H0R-sparse_SPIN0.csr'))
    graphH = H.getGraph(stru=poscar, graph={}, isH=True, tojson=True)

    # Load overlap matrix and skip Hamiltonian generation
    S = ABACUSHS(os.path.join('/public/home/zhongyang/yzhong/CsVSb/50/Training/perturbation_cal/STRU_1/OUT.ABACUS', 'data-S0R-sparse_SPIN0.csr'))
    graphS = S.getGraph(stru=poscar, graph=graphH, skip=True, tojson=True)

    # Close files
    H.close()
    S.close()

    # Merge Hamiltonian and overlap data
    graph = deepcopy(graphH)
    graph['Hon'] = [graphH['Hon']]
    graph['Hoff'] = [graphH['Hoff']]
    graph['Son'] = graphS['Hon']
    graph['Soff'] = graphS['Hoff']

    # Output file name
    fname = 'HS.json'

    # Write the graph data to the JSON file
    with open(fname, 'w') as f:
        json.dump(graph, f, separators=[',', ':'])

    # Reopen the file for formatting (newlines after commas for readability)
    with open(fname, 'r') as f:
        file_content = f.read()
        formatted_content = re.sub(r', *"', ',\n"', file_content)

    # Write the formatted content back to the file
    with open(fname, 'w') as f:
        f.write(formatted_content)


def read_abacus_input(input_file: str) -> dict:
    """
    Read ABACUS INPUT file and extract electron-related parameters.
    
    Parameters:
        input_file (str): Path to the ABACUS INPUT file.
    
    Returns:
        dict: Dictionary containing:
            - 'nelec': Total number of electrons (if specified)
            - 'nelec_delta': Change in number of electrons (if specified)
            - 'doping_charge': Computed doping charge (nelec_delta if set, else nelec - neutral)
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


def get_valence_electrons(atomic_number: int) -> int:
    """
    Get the number of valence electrons for an element.
    
    Parameters:
        atomic_number (int): Atomic number (Z).
    
    Returns:
        int: Number of valence electrons.
    """
    if atomic_number > 118 or atomic_number < 1:
        return 0
    
    from pymatgen.core.periodic_table import Element
    try:
        element = Element.from_Z(atomic_number)
        group = element.group
        period = element.period
        
        # Period 1: H=1 (1 valence), He=2 (2 valence)
        if period == 1:
            return 1 if atomic_number == 1 else 2
        
        # Period 2: Li(3)=1, Be(4)=2, B(5)=3, C(6)=4, N(7)=5, O(8)=6, F(9)=7, Ne(10)=8
        elif period == 2:
            return max(1, atomic_number - 2)
        
        # Period >= 3
        elif period >= 3:
            if atomic_number <= 12:  # s-block: Na(11)=1, Mg(12)=2
                return atomic_number - 10
            else:  # p-block elements: Al(13)=3, Si(14)=4, P(15)=5, S(16)=6
                # For p-block, valence electrons = group number - 10
                # Al: group=13, valence=3
                # Si: group=14, valence=4
                # P: group=15, valence=5
                return int(group) - 10
    
    except:
        pass
    
    # Fallback: estimate from atomic number
    # Common valence patterns:
    if atomic_number <= 2:
        return atomic_number  # H=1, He=2
    elif atomic_number <= 10:
        return atomic_number - 2  # Li=1, Be=2, B=3, C=4, N=5, O=6, F=7, Ne=8
    elif atomic_number <= 18:
        return atomic_number - 10  # Na=1, Mg=2, Al=3, Si=4, P=5, S=6, Cl=7, Ar=8
    else:
        # For heavier elements, use group-based estimate
        try:
            element = Element.from_Z(atomic_number)
            g = int(group) if group else 14
            if atomic_number > 12:  # p-block
                return g - 10
            else:  # s-block
                return atomic_number - 10
        except:
            return 4  # Default for transition metals


def get_neutral_electrons(stru: STRU) -> int:
    """
    Calculate the number of valence electrons in a neutral system.
    
    Parameters:
        stru (STRU): STRU object containing atomic information.
    
    Returns:
        int: Total number of valence electrons in the neutral system.
    """
    total_valence = 0
    for z in stru.atomic_numbers:
        total_valence += get_valence_electrons(int(z))
    return total_valence


def calculate_doping_charge(input_params: dict, neutral_electrons: int) -> float:
    """
    Calculate the doping charge from INPUT parameters.
    
    Priority:
    1. If nelec_delta is set, use it directly
    2. If nelec is set, calculate: doping_charge = nelec - neutral_electrons
    3. Otherwise, return 0.0 (neutral system)
    
    Parameters:
        input_params (dict): Output from read_abacus_input().
        neutral_electrons (int): Number of electrons in neutral system.
    
    Returns:
        float: Doping charge (positive = hole doping, negative = electron doping).
    """
    if input_params['nelec_delta'] is not None:
        return float(input_params['nelec_delta'])
    elif input_params['nelec'] is not None:
        return float(input_params['nelec'] - neutral_electrons)
    else:
        return 0.0


if __name__ == '__main__':
    process_graph_data()