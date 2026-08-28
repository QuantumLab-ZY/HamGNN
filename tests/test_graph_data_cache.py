import pickle

import lmdb
import numpy as np
import torch
from torch_geometric.data import Data

from hamgnn.data.graph_data import LMDBGraphDataset, NPZGraphDataset


def _graph(value):
    return Data(value=torch.tensor([value], dtype=torch.int64))


def _write_lmdb(path, values):
    env = lmdb.open(str(path), map_size=1024 * 1024)
    with env.begin(write=True) as txn:
        txn.put(b"num_graphs", str(len(values)).encode())
        for index, value in enumerate(values):
            txn.put(f"graph_{index}".encode(), pickle.dumps(_graph(value)))
    env.close()


class CountingTransform:
    def __init__(self):
        self.calls = []

    def __call__(self, data):
        self.calls.append(int(data.value.item()))
        data.value = data.value + 100
        return data


def _assert_cache_behavior(dataset):
    assert int(dataset[0].value.item()) == 107
    assert int(dataset[1].value.item()) == 103
    assert dataset[2].value.item() == 104
    assert list(dataset.runtime_cache) == [3, 4]
    assert dataset[3].value.item() == 105
    assert list(dataset.runtime_cache) == [4, 5]
    assert dataset[1].value.item() == 103
    assert list(dataset.runtime_cache) == [5, 3]


def test_lmdb_preload_and_lru_use_physical_indices(tmp_path):
    path = tmp_path / "graphs.lmdb"
    _write_lmdb(path, [0, 1, 2, 3, 4, 5, 6, 7])
    transform = CountingTransform()
    dataset = LMDBGraphDataset(
        str(path), indices=[7, 3, 4, 5], transform=transform, preload=1, cache_size=2
    )

    _assert_cache_behavior(dataset)
    assert transform.calls == [7, 3, 4, 5, 3]
    assert dataset.preloaded_data[7].value.item() == 107


def test_npz_preload_and_lru_use_physical_indices(tmp_path):
    path = tmp_path / "graphs.npz"
    np.savez(path, graph=np.array({str(i): _graph(i) for i in range(8)}, dtype=object))
    transform = CountingTransform()
    dataset = NPZGraphDataset(
        str(path), indices=[7, 3, 4, 5], transform=transform, preload=1, cache_size=2
    )

    _assert_cache_behavior(dataset)
    assert transform.calls == [7, 3, 4, 5, 3]
    assert dataset.preloaded_data[7].value.item() == 107


def test_lru_caches_decoded_graph_without_transform(tmp_path):
    path = tmp_path / "graphs.lmdb"
    _write_lmdb(path, [0, 1])
    dataset = LMDBGraphDataset(str(path), cache_size=1)

    first = dataset[0]
    second = dataset[0]

    assert first is second
    assert list(dataset.runtime_cache) == [0]
