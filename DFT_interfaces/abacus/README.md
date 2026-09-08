# ABACUS interface

[中文使用说明](README_zh.md)

This interface accepts both legacy ABACUS 3.10 sparse matrices and the
HContainer CSR format introduced in ABACUS 3.11. Matrix payloads are streamed
per Bravais vector and validated for dimensions, NNZ counts, CSR row pointers,
finite values, and single-geometry ownership.

Both versions use the same graph-generator CLI. A legacy ordinary H keeps the
historical geometry source, `running_scf.log`. When the ordinary H is a 3.11
HContainer CSR, its embedded lattice, species/counts, and direct coordinates
are authoritative; the log remains necessary for energy, orbital, and
pseudopotential-valence metadata.

## Matrix files

Ordinary SCF Hamiltonians are resolved from exactly one of:

- ABACUS 3.10: `data-HR-sparse_SPIN0.csr`
- ABACUS 3.11: `hrs1_nao.csr` (or one geometry-indexed `hrs1g*_nao.csr`)

H0/S0 are resolved directly from the corresponding versioned filenames:

- ABACUS 3.10: `data-H0R-sparse_SPIN0.csr` and
  `data-S0R-sparse_SPIN0.csr`
- ABACUS 3.11: `h0rs1_nao.csr` and `s0r_nao.csr`

Ordinary H and H0/S0 must all be located in the same case `OUT.ABACUS`
directory. Separate SCF and H0 roots are not supported. If both generations of
one required matrix are present, the input is ambiguous and is rejected.

The graph reader deliberately does not require an additional provenance-marker
module, so existing 3.10 datasets keep working unchanged. It requires selected
files to exist and be non-empty, then validates CSR structure while actually
reading it. Appended multi-ionic-step CSR files and missing or ambiguous
matrices fail closed. Ordinary H/S are never substituted for H0/S0.

Generate ABACUS 3.11 H0/S0 with
[`abacus_H0_export/README.md`](abacus_H0_export/README.md).
The preferred exporter is one self-contained `abacus_h0` ELF. It needs no
Python, Conda, MPI, oneMKL installation, or full ABACUS driver on the target.

## Graph generation

Pass every case root explicitly; material-specific defaults are not embedded:

```bash
python DFT_interfaces/abacus/graph_data_gen_abacus.py \
  --data-dirs /path/cases/0001 /path/cases/0002 \
  --graph-data-folder /path/graphs \
  --output-format lmdb \
  --nao-max 13 \
  --num-processes 8 \
  --worker-threads 1
```

Each case must contain `INPUT` plus one `OUT.ABACUS` holding the completed
`running_scf.log`, the ordinary H when required, and the matching H0/S0
outputs. Only this co-located layout is accepted.

No version flag is needed: `data-HR-sparse_SPIN0.csr` selects the legacy 3.10
geometry path, while `hrs1_nao.csr` (or one unambiguous `hrs1g*_nao.csr`)
selects the 3.11 CSR geometry path. All additional 3.11 H0/S0 headers are
checked against that selected structure. The graph's
`abacus_matrix_provenance` records both `structure` and `structure_format`.

The generator supports `--nao-max 13`, `15`, `27`, or `40`, NPZ, LMDB, or
both, and an explicit `--skip-dft-hamiltonian` mode. Final outputs are staged
under temporary names and installed only when every requested case succeeds.
Existing final output is preserved unless `--overwrite` is supplied; even with
that flag it is replaced only after the new conversion completes.

Hamiltonian entries are converted from Rydberg to Hartree, overlap is
dimensionless, positions and cells remain in Bohr, and ABACUS log total energy
is stored in eV. Each graph carries the selected matrix paths and detected CSR
formats in `abacus_matrix_provenance`. Charge labels use the pseudopotential valence
counts printed by ABACUS rather than periodic-table guesses; positive values
mean holes and negative values mean added electrons.

Current H0 export and graph generation cover scalar `nspin=1`, non-SOC
targets only.
