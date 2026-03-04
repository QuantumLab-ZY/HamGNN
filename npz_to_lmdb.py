#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to convert graph_data.npz format to LMDB format
Usage: python npz_to_lmdb.py --input graph_data.npz --output graph_data.lmdb
"""
import numpy as np
import lmdb
import pickle
import os
import argparse
import time
from tqdm import tqdm
import shutil


def convert_npz_to_lmdb(npz_path, lmdb_path, map_size=None):
    """
    Convert NPZ file to LMDB format
    
    Parameters:
    npz_path: str, path to input NPZ file
    lmdb_path: str, path to output LMDB database
    map_size: int, memory size allocated for LMDB database (in bytes), default is 1TB or automatically estimated
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(lmdb_path)), exist_ok=True)

    # Load original data
    print(f"Loading NPZ file: {npz_path}")
    start_time = time.time()

    try:
        data = np.load(npz_path, allow_pickle=True)
        graphs = data['graph'].item()  # Get dictionary object
    except Exception as e:
        print(f"Cannot load NPZ file: {e}")
        return False

    print(
        f"NPZ loading complete, time elapsed: {time.time() - start_time:.2f} seconds")
    print(f"Found {len(graphs)} graph data")

    # Estimate required map_size
    if map_size is None:
        # First sample to calculate average size
        sample_size = min(100, len(graphs))
        sample_indices = np.random.choice(
            list(graphs.keys()), sample_size, replace=False)

        total_bytes = 0
        for idx in sample_indices:
            # Serialize object and calculate size
            serialized = pickle.dumps(graphs[idx])
            total_bytes += len(serialized)

        # Estimate total size (average size * number of graphs * safety factor)
        avg_size = total_bytes / sample_size
        estimated_size = int(avg_size * len(graphs) *
                             1.5)  # Add 50% safety margin

        # Ensure at least 1GB, not exceeding 1TB
        map_size = max(1_073_741_824, min(estimated_size, 1_099_511_627_776))
        print(f"Estimated database size: {map_size / (1024**3):.2f} GB")

    # If target path already exists, create a backup
    if os.path.exists(lmdb_path):
        backup_path = lmdb_path + ".backup"
        print(
            f"Target LMDB path already exists, creating backup: {backup_path}")
        if os.path.exists(backup_path):
            shutil.rmtree(backup_path)
        shutil.move(lmdb_path, backup_path)

    # Create LMDB environment
    env = lmdb.open(lmdb_path, map_size=map_size)

    # Write data
    print("Starting data conversion to LMDB...")
    start_time = time.time()

    with env.begin(write=True) as txn:
        # Store the number of graphs
        txn.put('num_graphs'.encode(), str(len(graphs)).encode())

        # Create progress bar using tqdm
        for i, idx in enumerate(tqdm(sorted(graphs.keys()), desc="Conversion progress")):
            graph = graphs[idx]

            # Serialize graph data
            serialized_data = pickle.dumps(graph)

            # Store to LMDB
            txn.put(f'graph_{i}'.encode(), serialized_data)

    # Close environment
    env.close()

    # Output results
    print(
        f"Conversion completed! Time elapsed: {time.time() - start_time:.2f} seconds")
    print(f"LMDB database has been saved to: {lmdb_path}")

    # Validate data
    validate_lmdb(lmdb_path)

    return True


def validate_lmdb(lmdb_path):
    """Validate that the LMDB database is correctly created"""
    print("Validating LMDB database...")

    try:
        env = lmdb.open(lmdb_path, readonly=True, lock=False)
        with env.begin() as txn:
            num_graphs = int(txn.get('num_graphs'.encode()).decode())
            print(f"LMDB contains {num_graphs} graph data")

            # Try to read the first and last data points
            first_key = 'graph_0'.encode()
            if txn.get(first_key) is not None:
                print("✅ Successfully read the first data item")

            last_key = f'graph_{num_graphs-1}'.encode()
            if txn.get(last_key) is not None:
                print("✅ Successfully read the last data item")

        env.close()
        print("✅ LMDB validation successful!")
    except Exception as e:
        print(f"❌ LMDB validation failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert NPZ graph data to LMDB format')
    parser.add_argument('--input', type=str, required=True,
                        help='Input NPZ file path')
    parser.add_argument('--output', type=str, required=True,
                        help='Output LMDB database path')
    parser.add_argument('--map-size', type=int, default=None,
                        help='Memory size allocated for LMDB database (bytes)')
    args = parser.parse_args()

    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file does not exist: {args.input}")
        return 1

    # Convert data
    success = convert_npz_to_lmdb(args.input, args.output, args.map_size)

    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
