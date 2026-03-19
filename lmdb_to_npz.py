#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to convert graph_data.lmdb format to NPZ format
Usage: python lmdb_to_npz.py --input graph_data.lmdb --output graph_data.npz
"""
import argparse
import os
import pickle
import shutil

import lmdb
import numpy as np
from tqdm import tqdm


def convert_lmdb_to_npz(lmdb_path, npz_path):
    """
    Convert LMDB dataset to legacy NPZ graph format.

    Parameters
    ----------
    lmdb_path : str
        Path to the input LMDB dataset.
    npz_path : str
        Path to the output NPZ archive.
    """
    os.makedirs(os.path.dirname(os.path.abspath(npz_path)), exist_ok=True)

    if os.path.exists(npz_path):
        backup_path = npz_path + ".backup"
        print(f"Target NPZ path already exists, creating backup: {backup_path}")
        if os.path.isdir(backup_path):
            shutil.rmtree(backup_path)
        elif os.path.exists(backup_path):
            os.remove(backup_path)
        shutil.move(npz_path, backup_path)

    try:
        env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
    except Exception as e:
        print(f"Cannot open LMDB dataset: {e}")
        return False

    graph_data = {}
    try:
        with env.begin() as txn:
            raw_count = txn.get(b"num_graphs")
            if raw_count is None:
                raise RuntimeError(f"LMDB dataset {lmdb_path} does not contain 'num_graphs'")
            num_graphs = int(raw_count.decode())
            print(f"Found {num_graphs} graph data")

            for idx in tqdm(range(num_graphs), desc="Conversion progress"):
                payload = txn.get(f"graph_{idx}".encode())
                if payload is None:
                    raise IndexError(f"Missing graph_{idx} in LMDB dataset {lmdb_path}")
                graph_data[idx] = pickle.loads(payload)
    except Exception as e:
        print(f"Cannot read LMDB dataset: {e}")
        env.close()
        return False

    env.close()
    np.savez(npz_path, graph=graph_data)
    print(f"NPZ archive has been saved to: {npz_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert LMDB graph data to NPZ format"
    )
    parser.add_argument("--input", type=str, required=True, help="Input LMDB dataset path")
    parser.add_argument("--output", type=str, required=True, help="Output NPZ file path")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input path does not exist: {args.input}")
        return 1

    success = convert_lmdb_to_npz(args.input, args.output)
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
