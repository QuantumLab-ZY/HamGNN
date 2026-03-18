'''
Descripttion: The script to calculat bands from the results of HamGNN
version: 2.0
Author: Yang Zhong
Date: 2022-12-20 14:08:52
LastEditors: Wenhai Lu
LastEditTime: 2026-03-18 16:29:28
'''

import os

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import argparse
import math
import multiprocessing as mp
import time
from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml
from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Structure
from pymatgen.symmetry.kpath import KPathSeek
from scipy.linalg import cholesky, eigh, solve_triangular

from utils import (
    au2ang,
    au2ev,
    basis_def_13_abacus,
    basis_def_15_abacus,
    basis_def_20_abacus,
    basis_def_27_abacus,
    kpoints_generator,
    num_val,
)


DEFAULT_ENERGY_WINDOW_EV = 1.5
DEFAULT_COMPARE_SUBDIR = 'band_compare'
DEFAULT_COMPARE_ALIGNMENT_MODE = 'self_vbm'
_WORKER_STATE = None


def resolve_path(base_dir, path_value):
    if path_value in (None, ''):
        return None
    if os.path.isabs(path_value):
        return path_value
    return os.path.abspath(os.path.join(base_dir, path_value))


def load_config(config_path):
    with open(config_path, encoding='utf-8') as stream:
        config = yaml.safe_load(stream) or {}
    config_dir = os.path.dirname(os.path.abspath(config_path))

    resolved = dict(config)
    for key in ('graph_data_path', 'hamiltonian_path', 'prediction_hamiltonian_path', 'target_hamiltonian_path', 'save_dir'):
        resolved[key] = resolve_path(config_dir, resolved.get(key))

    resolved['structure_name'] = resolved.get('structure_name') or resolved.get('strcture_name') or 'structure'
    resolved['compare_subdir'] = resolved.get('compare_subdir') or DEFAULT_COMPARE_SUBDIR
    resolved['compare_alignment_mode'] = (resolved.get('compare_alignment_mode') or DEFAULT_COMPARE_ALIGNMENT_MODE).strip().lower()
    resolved['nk'] = int(resolved.get('nk', 200))
    resolved['nao_max'] = int(resolved.get('nao_max', 20))
    resolved['soc_switch'] = bool(resolved.get('soc_switch', False))
    resolved['auto_mode'] = bool(resolved.get('auto_mode', False))
    if resolved['compare_alignment_mode'] not in {'self_vbm', 'target_vbm', 'raw'}:
        raise ValueError(
            "compare_alignment_mode must be one of: 'self_vbm', 'target_vbm', 'raw'"
        )
    if resolved.get('num_workers') in (None, ''):
        resolved['num_workers'] = None
    else:
        resolved['num_workers'] = int(resolved['num_workers'])
    if resolved.get('energy_window_ev') in (None, ''):
        resolved['energy_window_ev'] = DEFAULT_ENERGY_WINDOW_EV
    else:
        resolved['energy_window_ev'] = float(resolved['energy_window_ev'])
        if resolved['energy_window_ev'] <= 0:
            raise ValueError('energy_window_ev must be positive')

    return resolved


def as_numpy(value):
    if hasattr(value, 'detach'):
        value = value.detach()
    if hasattr(value, 'cpu'):
        value = value.cpu()
    if hasattr(value, 'numpy'):
        return value.numpy()
    return np.asarray(value)


def get_basis_def(nao_max):
    if nao_max == 13:
        return basis_def_13_abacus
    if nao_max == 15:
        return basis_def_15_abacus
    if nao_max == 20:
        return basis_def_20_abacus
    if nao_max == 27:
        return basis_def_27_abacus
    raise NotImplementedError(f'Unsupported nao_max={nao_max} for ABACUS band calculation')


def build_source_paths(config):
    prediction_path = config.get('prediction_hamiltonian_path')
    target_path = config.get('target_hamiltonian_path')
    hamiltonian_path = config.get('hamiltonian_path')

    if prediction_path and target_path:
        return OrderedDict((('prediction', prediction_path), ('target', target_path))), True

    single_path = hamiltonian_path or prediction_path or target_path
    if not single_path:
        raise ValueError('Missing Hamiltonian input: provide hamiltonian_path or both prediction_hamiltonian_path and target_hamiltonian_path')

    return OrderedDict((('band', single_path),)), False


def split_non_soc_hamiltonians(graph_dataset, hamiltonian_path, nao_max):
    matrices = np.load(hamiltonian_path, mmap_mode='r').reshape(-1, nao_max, nao_max)
    split = []
    cursor = 0
    for data in graph_dataset:
        hon_len = len(data.Hon)
        hoff_len = len(data.Hoff)
        hon = np.asarray(matrices[cursor:cursor + hon_len])
        cursor += hon_len
        hoff = np.asarray(matrices[cursor:cursor + hoff_len])
        cursor += hoff_len
        split.append({'Hon': hon, 'Hoff': hoff})

    if cursor != len(matrices):
        raise ValueError(f'Hamiltonian length mismatch for {hamiltonian_path}: consumed {cursor}, total {len(matrices)}')

    return split


def get_manual_klabels(label):
    return [entry.replace('$', '') for entry in label]


def prepare_k_path(struct, latt, nk, auto_mode, k_path, label):
    if auto_mode:
        kpath_seek = KPathSeek(structure=struct)
        klabels = []
        for labels in kpath_seek.kpath['path']:
            klabels += labels
        deduped = [klabels[0]]
        [deduped.append(item) for item in klabels[1:] if item != deduped[-1]]
        klabels = deduped
        k_path = [kpath_seek.kpath['kpoints'][key] for key in klabels]
        label = [rf'${key}$' for key in klabels]
    else:
        if not k_path or not label:
            raise ValueError('k_path and label must be provided when auto_mode is False')
        if len(k_path) != len(label):
            raise ValueError(f'k_path length {len(k_path)} does not match label length {len(label)}')
        klabels = get_manual_klabels(label)

    kpts = kpoints_generator(dim_k=3, lat=latt)
    k_vec, k_dist, k_node, lat_per_inv, node_index = kpts.k_path(k_path, nk)
    k_vec = k_vec.dot(lat_per_inv[np.newaxis, :, :]).reshape(-1, 3)
    return {
        'k_vec': k_vec,
        'k_dist': k_dist,
        'k_node': k_node,
        'node_index': node_index,
        'klabels': klabels,
        'label': label,
    }


def build_orbital_indices(species, nao_max):
    basis_definition = np.zeros((99, nao_max), dtype=np.int8)
    basis_def = get_basis_def(nao_max)
    for atomic_number, orbital_indices in basis_def.items():
        basis_definition[atomic_number][orbital_indices] = 1

    occupied = basis_definition[species].reshape(-1).astype(bool)
    orbital_indices = np.flatnonzero(occupied)
    if orbital_indices.size == 0:
        raise ValueError('No occupied orbitals were selected from the ABACUS basis definition')
    return orbital_indices, np.ix_(orbital_indices, orbital_indices)


def build_edge_groups(cell_shift, nbr_shift, edge_index, soff):
    unique_cells, inverse = np.unique(cell_shift, axis=0, return_inverse=True)
    groups = []
    phase_shifts = []

    for group_idx in range(len(unique_cells)):
        edge_ids = np.flatnonzero(inverse == group_idx)
        phase_candidates = nbr_shift[edge_ids]
        unique_phase = np.unique(phase_candidates, axis=0)
        if len(unique_phase) != 1:
            raise ValueError(f'Cell-shift group {group_idx} maps to multiple nbr_shift values; cannot build grouped phase factors safely')

        groups.append({
            'rows': edge_index[0, edge_ids],
            'cols': edge_index[1, edge_ids],
            'edge_ids': edge_ids,
            'Soff': np.asarray(soff[edge_ids], dtype=np.complex64),
        })
        phase_shifts.append(unique_phase[0])

    return groups, np.asarray(phase_shifts, dtype=np.float32)


def build_phases(phase_shifts, k_vec):
    return np.exp(2j * np.pi * np.sum(phase_shifts[np.newaxis, :, :] * k_vec[:, np.newaxis, :], axis=-1)).astype(np.complex64)


def build_onsite_template(onsite_blocks, natoms, nao_max):
    onsite_template = np.zeros((natoms, natoms, nao_max, nao_max), dtype=np.complex64)
    atom_indices = np.arange(natoms)
    onsite_template[atom_indices, atom_indices, :, :] = np.asarray(onsite_blocks, dtype=np.complex64)
    return onsite_template


def build_source_groups(structure_groups, hoff):
    source_groups = []
    for group in structure_groups:
        source_groups.append({
            'rows': group['rows'],
            'cols': group['cols'],
            'blocks': np.asarray(hoff[group['edge_ids']], dtype=np.complex64),
        })
    return source_groups


def assemble_k_matrix(template, groups, phase_row, orbital_selector):
    matrix = template.copy()
    for phase, group in zip(phase_row, groups):
        matrix[group['rows'], group['cols']] += phase * group['blocks']
    matrix = np.swapaxes(matrix, 1, 2).reshape(template.shape[0] * template.shape[2], template.shape[1] * template.shape[3])
    return matrix[orbital_selector]


def build_chunks(nk, num_workers):
    if num_workers <= 1 or nk <= 1:
        return [(0, nk)]
    chunk_size = math.ceil(nk / num_workers)
    return [(start, min(start + chunk_size, nk)) for start in range(0, nk, chunk_size)]


def _solve_k_chunk(chunk):
    start, stop = chunk
    state = _WORKER_STATE
    phases = state['phases']
    source_names = state['source_names']
    source_defs = state['source_defs']
    norbs = state['norbs']

    bands = {name: np.empty((stop - start, norbs), dtype=np.float64) for name in source_names}
    for local_index, ik in enumerate(range(start, stop)):
        sk = assemble_k_matrix(state['S_template'], state['S_groups'], phases[ik], state['orbital_selector'])
        try:
            chol = cholesky(sk, lower=True, overwrite_a=True, check_finite=False)
        except np.linalg.LinAlgError as exc:
            raise np.linalg.LinAlgError(f'Overlap matrix is not positive definite at k-index {ik}') from exc

        for name in source_names:
            hk = assemble_k_matrix(source_defs[name]['template'], source_defs[name]['groups'], phases[ik], state['orbital_selector'])
            transformed = solve_triangular(chol, hk, lower=True, trans='N', check_finite=False)
            transformed = solve_triangular(chol.conj(), transformed.T, lower=True, trans='N', check_finite=False).T
            transformed = 0.5 * (transformed + transformed.conj().T)
            bands[name][local_index] = eigh(
                a=transformed,
                eigvals_only=True,
                overwrite_a=True,
                check_finite=False,
            )
    return bands


def solve_band_energies(phases, s_template, s_groups, source_defs, orbital_selector, norbs, num_workers):
    global _WORKER_STATE

    nk = phases.shape[0]
    num_workers = max(1, min(num_workers, nk))
    chunks = build_chunks(nk, num_workers)
    _WORKER_STATE = {
        'phases': phases,
        'S_template': s_template,
        'S_groups': s_groups,
        'source_defs': source_defs,
        'source_names': tuple(source_defs.keys()),
        'orbital_selector': orbital_selector,
        'norbs': norbs,
    }

    try:
        if len(chunks) == 1:
            chunk_results = [_solve_k_chunk(chunks[0])]
        else:
            ctx = mp.get_context('fork')
            with ctx.Pool(processes=len(chunks)) as pool:
                chunk_results = pool.map(_solve_k_chunk, chunks)
    finally:
        _WORKER_STATE = None

    results = {}
    for name in source_defs:
        per_k = np.concatenate([chunk[name] for chunk in chunk_results], axis=0)
        results[name] = per_k.T * au2ev
    return results


def compute_band_metrics(eigenvalues_ev, species):
    num_electrons = int(np.sum(num_val[species]))
    valence_index = math.ceil(num_electrons / 2) - 1
    conduction_index = valence_index + 1
    vbm = float(np.max(eigenvalues_ev[valence_index]))
    cbm = float(np.min(eigenvalues_ev[conduction_index]))
    return {
        'raw': eigenvalues_ev,
        'vbm': vbm,
        'cbm': cbm,
        'gap': cbm - vbm,
        'valence_index': valence_index,
    }


def shift_bands(eigenvalues_ev, energy_shift):
    return eigenvalues_ev - energy_shift


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def build_plot_ylim(energy_window_ev):
    return (-0.5 * energy_window_ev, energy_window_ev)


def resolve_compare_shifts(metrics, compare_alignment_mode):
    if compare_alignment_mode == 'self_vbm':
        return metrics['target']['vbm'], metrics['prediction']['vbm'], 'self_vbm'
    if compare_alignment_mode == 'target_vbm':
        return metrics['target']['vbm'], metrics['target']['vbm'], 'target_vbm'
    if compare_alignment_mode == 'raw':
        return 0.0, 0.0, 'raw'
    raise ValueError(f'Unsupported compare_alignment_mode={compare_alignment_mode}')


def write_band_data(path, eigenvalues, k_dist, klabels, k_node, node_index, source_name, reference_mode, energy_shift, vbm, gap):
    node_breaks = set(node_index[1:-1])
    with open(path, 'w', encoding='utf-8') as text_file:
        text_file.write(f'# reference: {reference_mode}\n')
        text_file.write(f'# energy_shift_eV: {energy_shift:.10f}\n')
        text_file.write(f'# own_vbm_eV: {vbm:.10f}\n')
        text_file.write(f'# band_gap_eV: {gap:.10f}\n')
        text_file.write('# k_lable: ')
        for label in klabels:
            text_file.write(f'{label} ')
        text_file.write('\n')

        text_file.write('# k_node: ')
        for node in k_node:
            text_file.write(f'{node:f}  ')
        text_file.write('\n')

        nk = len(k_dist)
        for nb in range(len(eigenvalues)):
            for ik in range(nk):
                text_file.write(f'{k_dist[ik]:f}    {eigenvalues[nb, ik]:f}\n')
                if ik in node_breaks:
                    text_file.write('\n')
                    text_file.write(f'{k_dist[ik]:f}    {eigenvalues[nb, ik]:f}\n')
            text_file.write('\n')


def plot_single_band(output_path, k_dist, k_node, label, eigenvalues, title, ylim):
    fig, ax = plt.subplots()
    ax.set_xlim(k_node[0], k_node[-1])
    ax.set_xticks(k_node)
    ax.set_xticklabels(label)
    for node in k_node:
        ax.axvline(x=node, linewidth=0.5, color='k')

    for band in eigenvalues:
        if np.min(band) <= ylim[1] and np.max(band) >= ylim[0]:
            ax.plot(k_dist, band, color='tab:blue', linewidth=0.8)
    ax.plot(k_dist, np.zeros_like(k_dist), linestyle='--', color='gray', linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel('Path in k-space')
    ax.set_ylabel('Band energy (eV)')
    ax.set_ylim(*ylim)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_compare_band(output_path, k_dist, k_node, label, target_eigen, prediction_eigen, ylim):
    fig, ax = plt.subplots()
    ax.set_xlim(k_node[0], k_node[-1])
    ax.set_xticks(k_node)
    ax.set_xticklabels(label)
    for node in k_node:
        ax.axvline(x=node, linewidth=0.5, color='k')

    target_labeled = False
    prediction_labeled = False
    for band in target_eigen:
        if np.min(band) <= ylim[1] and np.max(band) >= ylim[0]:
            ax.plot(k_dist, band, color='black', linewidth=0.9, label='Target' if not target_labeled else None)
            target_labeled = True
    for band in prediction_eigen:
        if np.min(band) <= ylim[1] and np.max(band) >= ylim[0]:
            ax.plot(k_dist, band, color='red', linestyle='--', linewidth=0.9, label='Prediction' if not prediction_labeled else None)
            prediction_labeled = True

    ax.plot(k_dist, np.zeros_like(k_dist), linestyle='--', color='gray', linewidth=0.8)
    ax.set_title('Prediction vs Target Band structure')
    ax.set_xlabel('Path in k-space')
    ax.set_ylabel('Band energy (eV)')
    ax.set_ylim(*ylim)
    if target_labeled or prediction_labeled:
        ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def prepare_structure_context(data, structure_index, config):
    nao_max = config['nao_max']
    species = as_numpy(data.z).astype(int)
    son = as_numpy(data.Son).reshape(-1, nao_max, nao_max)
    soff = as_numpy(data.Soff).reshape(-1, nao_max, nao_max)
    latt = as_numpy(data.cell).reshape(3, 3)
    pos = as_numpy(data.pos) * au2ang
    nbr_shift = as_numpy(data.nbr_shift)
    edge_index = as_numpy(data.edge_index).astype(int)
    cell_shift = as_numpy(data.cell_shift)

    struct = Structure(
        lattice=latt * au2ang,
        species=[Element.from_Z(atomic_number).symbol for atomic_number in species],
        coords=pos,
        coords_are_cartesian=True,
    )
    cif_path = os.path.join(config['save_dir'], f"{config['structure_name']}_{structure_index + 1}.cif")
    struct.to(filename=cif_path)

    k_info = prepare_k_path(
        struct=struct,
        latt=latt,
        nk=config['nk'],
        auto_mode=config['auto_mode'],
        k_path=config.get('k_path'),
        label=config.get('label'),
    )
    orbital_indices, orbital_selector = build_orbital_indices(species, nao_max)
    structure_groups, phase_shifts = build_edge_groups(cell_shift, nbr_shift, edge_index, soff)
    phases = build_phases(phase_shifts, k_info['k_vec'])

    return {
        'species': species,
        'structure': struct,
        'k_dist': k_info['k_dist'],
        'k_node': k_info['k_node'],
        'node_index': k_info['node_index'],
        'klabels': k_info['klabels'],
        'label': k_info['label'],
        'orbital_indices': orbital_indices,
        'orbital_selector': orbital_selector,
        'phases': phases,
        'S_template': build_onsite_template(son, len(species), nao_max),
        'S_groups': [{'rows': group['rows'], 'cols': group['cols'], 'blocks': group['Soff']} for group in structure_groups],
        'structure_groups': structure_groups,
        'norbs': len(orbital_indices),
        'natoms': len(species),
    }


def build_structure_sources(structure_groups, source_splits, structure_index, natoms, nao_max):
    source_defs = OrderedDict()
    for source_name, split_list in source_splits.items():
        current = split_list[structure_index]
        hon = np.asarray(current['Hon']).reshape(-1, nao_max, nao_max)
        hoff = np.asarray(current['Hoff']).reshape(-1, nao_max, nao_max)
        source_defs[source_name] = {
            'template': build_onsite_template(hon, natoms, nao_max),
            'groups': build_source_groups(structure_groups, hoff),
        }
    return source_defs


def report_metrics(source_name, metrics, applied_shift):
    print(f'[{source_name}] own_vbm = {metrics["vbm"]:.6f} eV')
    print(f'[{source_name}] band gap = {metrics["gap"]:.6f} eV')
    print(f'[{source_name}] applied energy shift = {applied_shift:.6f} eV')


def choose_num_workers(config, nk):
    requested = config.get('num_workers')
    if requested is None:
        return max(1, min(4, nk, os.cpu_count() or 1))
    return max(1, min(int(requested), nk))


def main():
    parser = argparse.ArgumentParser(description='band calculation')
    parser.add_argument('--config', default='band_cal.yaml', type=str, metavar='N')
    args = parser.parse_args()

    config = load_config(args.config)
    if config['soc_switch']:
        raise NotImplementedError('This optimized ABACUS workflow currently supports only soc_switch=False')

    ensure_dir(config['save_dir'])
    graph_data = np.load(config['graph_data_path'], allow_pickle=True)['graph'].item()
    graph_dataset = list(graph_data.values())

    source_paths, compare_mode = build_source_paths(config)
    source_splits = OrderedDict(
        (source_name, split_non_soc_hamiltonians(graph_dataset, path, config['nao_max']))
        for source_name, path in source_paths.items()
    )

    print(f'Loaded {len(graph_dataset)} structure(s) from {config["graph_data_path"]}')
    print(f'Sources: {", ".join(source_paths.keys())}')

    for structure_index, data in enumerate(graph_dataset):
        print(f'Processing structure {structure_index + 1}/{len(graph_dataset)}')
        context = prepare_structure_context(data, structure_index, config)
        source_defs = build_structure_sources(
            structure_groups=context['structure_groups'],
            source_splits=source_splits,
            structure_index=structure_index,
            natoms=context['natoms'],
            nao_max=config['nao_max'],
        )

        num_workers = choose_num_workers(config, context['phases'].shape[0])
        print(f'Using {num_workers} worker(s) for {context["phases"].shape[0]} k-points')
        solve_start = time.perf_counter()
        eigenvalues = solve_band_energies(
            phases=context['phases'],
            s_template=context['S_template'],
            s_groups=context['S_groups'],
            source_defs=source_defs,
            orbital_selector=context['orbital_selector'],
            norbs=context['norbs'],
            num_workers=num_workers,
        )
        solve_elapsed = time.perf_counter() - solve_start
        print(f'Solved band energies in {solve_elapsed:.2f} s')

        metrics = OrderedDict((name, compute_band_metrics(values, context['species'])) for name, values in eigenvalues.items())
        shifted = OrderedDict()

        plot_ylim = build_plot_ylim(config['energy_window_ev'])

        if compare_mode:
            output_dir = os.path.join(config['save_dir'], config['compare_subdir'])
            ensure_dir(output_dir)

            target_shift, prediction_shift, reference_mode = resolve_compare_shifts(
                metrics,
                config['compare_alignment_mode'],
            )

            shifted['target'] = shift_bands(metrics['target']['raw'], target_shift)
            shifted['prediction'] = shift_bands(metrics['prediction']['raw'], prediction_shift)

            report_metrics('target', metrics['target'], target_shift)
            report_metrics('prediction', metrics['prediction'], prediction_shift)

            write_band_data(
                os.path.join(output_dir, f'target_band_{structure_index + 1}.dat'),
                shifted['target'],
                context['k_dist'],
                context['klabels'],
                context['k_node'],
                context['node_index'],
                'target',
                reference_mode,
                target_shift,
                metrics['target']['vbm'],
                metrics['target']['gap'],
            )
            write_band_data(
                os.path.join(output_dir, f'pred_band_{structure_index + 1}.dat'),
                shifted['prediction'],
                context['k_dist'],
                context['klabels'],
                context['k_node'],
                context['node_index'],
                'prediction',
                reference_mode,
                prediction_shift,
                metrics['prediction']['vbm'],
                metrics['prediction']['gap'],
            )
            plot_compare_band(
                os.path.join(output_dir, f'compare_band_{structure_index + 1}.png'),
                context['k_dist'],
                context['k_node'],
                context['label'],
                shifted['target'],
                shifted['prediction'],
                plot_ylim,
            )
        else:
            source_name = next(iter(metrics.keys()))
            reference_shift = metrics[source_name]['vbm']
            shifted[source_name] = shift_bands(metrics[source_name]['raw'], reference_shift)
            report_metrics(source_name, metrics[source_name], reference_shift)

            write_band_data(
                os.path.join(config['save_dir'], f'band_{structure_index + 1}.dat'),
                shifted[source_name],
                context['k_dist'],
                context['klabels'],
                context['k_node'],
                context['node_index'],
                source_name,
                'self_vbm',
                reference_shift,
                metrics[source_name]['vbm'],
                metrics[source_name]['gap'],
            )
            plot_single_band(
                os.path.join(config['save_dir'], f'band_{structure_index + 1}.png'),
                context['k_dist'],
                context['k_node'],
                context['label'],
                shifted[source_name],
                'Band structure',
                plot_ylim,
            )


if __name__ == '__main__':
    main()
