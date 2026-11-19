"""Compare graph block traversal to the original atom-pair slicing order."""
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from read_abacus import _stored_atom_blocks


class StoredAtomBlocksTest(unittest.TestCase):
    def test_storage_order_and_values(self):
        # Row zero has unsorted columns, a stored negative zero and duplicates.
        for counts in ([2, 2, 2], [1, 3, 2]):
            for dtype in (np.float32, np.complex64):
                with self.subTest(counts=counts, dtype=dtype):
                    matrix = csr_matrix((
                        np.array([-0.0, 1, -1, 0.25], dtype=dtype),
                        [5, 1, 1, 3], [0, 3, 3, 3, 4, 4, 4],
                    ), shape=(6, 6))
                    offsets = np.cumsum([0] + counts)
                    expected = []
                    for i in range(len(counts)):
                        for j in range(len(counts)):
                            block = matrix[offsets[i]:offsets[i+1], offsets[j]:offsets[j+1]]
                            if block.nnz:
                                expected.append((i, j, block.toarray()))
                    actual = list(_stored_atom_blocks(
                        matrix, np.array(counts), offsets[:-1], len(counts)))
                    self.assertEqual(len(actual), len(expected))
                    for (i, j, dense), (row, col, reference) in zip(actual, expected):
                        self.assertEqual((i, j), (row, col))
                        self.assertEqual(dense.dtype, reference.dtype)
                        self.assertEqual(dense.shape, reference.shape)
                        self.assertEqual(dense.tobytes(), reference.tobytes())
