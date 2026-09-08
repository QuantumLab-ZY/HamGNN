# ABACUS 3.11 H0Lite exporter

See [README_zh.md](README_zh.md) for the complete Chinese user and build guide.

Download `abacus-h0lite-v311_source.tar.gz` from this directory. It contains
the already-patched H0Lite sources, required ABACUS source/header dependencies,
standalone CMake build, build script, and licenses. No separate ABACUS download
or patch application is needed. The legacy
`abacus-postprocess-v353_source.tar.gz` archive is retained. The loose patch is
only for maintaining a full ABACUS developer checkout.

Build prerequisites are x86_64 Linux, GCC 9+ (10.2+ recommended), CMake 3.20+,
GNU Make, binutils, `file`, GCC static runtimes (`libstdc++`, `libgcc`,
`libgomp.a`), and oneMKL static development libraries (2023.2 recommended).
Set `MKLROOT` to that oneMKL installation. Once dependencies are installed,
the build needs no network, Python, Conda, MPI, or full ABACUS installation.
The installation must also supply `license.txt` and `third-party-programs.txt`.
The build finds them beside the actual linked MKL library, including unified
oneAPI symlinks. For a nonstandard layout, set `H0LITE_MKL_LICENSE_DIR` to the
licensing directory from that exact installation; another version's notices
are not a substitute.

The current 3.11-Simpson revision was built from a fresh extraction with
GCC 10.5.0, CMake 3.31.7, oneMKL 2024.2, and glibc 2.28. Both default H0 and
`--with-vl` executed successfully; only basic glibc libraries remain dynamically
linked. Numerical validation is described below. Earlier revisions also built
with oneMKL 2023.2.0. License discovery was checked for 2023.2/2024.2 and missing
notices, and the current executable's embedded-license output was checked.

Extract outside the repository and build on allocated compute resources:

```bash
tar -xzf abacus-h0lite-v311_source.tar.gz
cd abacus-h0lite-v311_source
export MKLROOT=/path/to/oneapi/mkl/2023.2.0
BUILD_CPUS=8 bash ./build_h0lite_single.sh
./bin/abacus_h0 --version
```

The script uses `SLURM_CPUS_PER_TASK` when set, otherwise `BUILD_CPUS` (default
1). The build directory is `build-h0lite-single/`; the runtime file is
`bin/abacus_h0`. Binary redistribution also requires matching source and
license materials; see [NOTICE.md](NOTICE.md). The original
`SOURCE_DIR BUILD_DIR OUTPUT_FILE` arguments
remain supported. Edit the bundled sources and rerun to rebuild; register new
translation units in `cmake/Sources.cmake`. See `SOURCE_INFO.md` in the archive
for source provenance. Build on a glibc baseline no newer than the target
machine; CentOS 7 compatibility requires building on glibc 2.17.

Load a GCC module as well as oneMKL: loading oneAPI alone does not select a
newer `g++`. The script checks GCC before configuring and passes its absolute
path to CMake. On a compiler change it preserves the old CMake cache and
compiler metadata in `BUILD_DIR/compiler-cache-backup.XXXXXX/` and configures
again. Rerun the same command after loading GCC 9+; manual cache deletion is
not required. CMake arguments are kept in a nonempty array so the standalone
build also works under Bash 4.2 with `set -u`. The Chinese guide includes the
current mgt module commands.

`abacus_h0` is a single-file, MPI-free x86_64 Linux executable derived from
ABACUS 3.11. It reads an ordinary LCAO SCF case but evaluates only
`H0=T+Vnl` and `S0`; `--with-vl` changes H0 to `T+Vnl+Vl`. It runs no SCF
iteration and no diagonalization.

The `h0lite-v311-simpson-buffered-20260907` revision uses ABACUS 3.11's finite-grid
Simpson integrator and orbital/projector transforms. An explicitly selected
H0 table mode restores historical distance spacing, per-pair cutoff/padding
and four-point interpolation; scalar diagonal-projector D is retained.
Imported legacy table modules are no longer included or compiled. Ordinary
3.11 tables keep their default spline path; independent direct-transform rows
use OpenMP without changing per-row arithmetic. This compatibility mode is
not equivalent to native 3.11 FFT/full-projector numerics when pairing H/S.

The marker generator is now `ABACUS-3.11-H0Lite-native-simpson-v3`.
Previous v1/FFT and v2/imported-table markers are rejected without overwriting or silently reusing
their matrices. Export into a fresh case/output directory when upgrading;
do not relabel old markers as new results. CLI arguments remain unchanged.
The optional Vl contribution remains merged into H0 and is tested separately
from the historical default-H0 comparison. The retained 3.5.3 archive is a
historical reference, not a build dependency; see SOURCE_INFO.md in the archive.

Acceptance compares 10 TiO2 and 6 Si cases with the real exporter built from
the retained 3.5.3 archive, using identical structures, pseudopotentials,
orbitals and physical inputs. Both maximum-absolute and relative-Frobenius
differences must be at most `1e-8` for H0 (Ry) and S0 (dimensionless).
This coverage does not establish arbitrary pseudopotential/orbital or
unsupported calculation-mode compatibility.

On 2026-09-06 all 16 cases passed: worst absolute differences were about
`1e-10` for both matrices, with relative differences below `2.36e-12` (H0)
and `1.30e-11` (S0). Two separate 16-case Slurm jobs on the same node,
16 CPUs and four concurrent cases each took 50 seconds from submission to
completion (previous imported tables versus this revision). This observed
no end-to-end slowdown; it does not establish a significant speedup.

The 2026-09-07 buffered revision retains the existing 3.11 CSR writer but
removes per-line flushes and read-only atom-pair copies. Numerical kernels,
formatting, sparse thresholds and precision are unchanged. Independent jobs
actually exporting all 16 cases with 16 CPUs and four concurrent cases took
47 s before optimization and 29 s in both optimized runs; the genuine 3.5.3
exporter took 49/44 s (the first run included 5 s of queueing). Including Vl
improved from 73 s to 57 s. Times are Slurm Submit-to-End, excluding separate
numerical comparisons; no old outputs were reused or single cases extrapolated.

Default H0/S0 are byte-identical before/after this optimization and still pass
all old-3.5.3 numerical checks. With Vl, S0 is byte-identical; H0 differences
are at most about 1e-11 Ry. Repeating the unmodified binary also shows last-bit
variation from dynamic parallel grid integration; two single-thread controls
are byte-identical. The Vl kernel is unchanged. Compatible v3 markers remain
reusable: this serialization-only update does not require regeneration.
Backend logs now include `h0_write_ms` and `s0_write_ms`.

```bash
cd CASE
/path/to/abacus_h0

/path/to/abacus_h0 CASE --with-vl

/path/to/abacus_h0 --case-list cases.txt --tasks 8 --cpus-per-task 4
```

For one case, the CPU count defaults to `SLURM_CPUS_PER_TASK`, or the CPUs
available to the process outside Slurm. In a Slurm multi-task or array launch,
the case list is sharded automatically before each node runs its local dynamic
case queue.

The local queue sets thread counts but does not assign disjoint CPU masks to
children. For batch execution, use `OMP_PROC_BIND=FALSE` and unset `OMP_PLACES`,
`GOMP_CPU_AFFINITY` and `KMP_AFFINITY`; retain Slurm's rank CPU binding.
Forcing `OMP_PROC_BIND=close` with `OMP_PLACES=cores` can make all children
contend for the same cores despite a larger Slurm allocation.

The runtime artifact is one `abacus_h0` file; do not commit that generated ELF
back into this source directory.

## Licensing

H0Lite is modified ABACUS, not an official upstream release. Full GPLv3 and
LGPLv3 texts are provided in `COPYING` and `COPYING.LESSER`, with attribution,
modification dates and redistribution guidance in [NOTICE.md](NOTICE.md).
`--license` also embeds the actual build's oneMKL license/third-party notices,
the GCC Runtime Library Exception, and retained third-party attribution.
One-file execution does not waive corresponding-source obligations: provide
the exact modified sources and build scripts for a distributed binary, for
example alongside it at the same download location. Sources need not be
installed on the execution node.
