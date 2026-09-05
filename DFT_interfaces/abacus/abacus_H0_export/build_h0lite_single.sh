#!/usr/bin/env bash
# Build the H0Lite-only ABACUS frontend as one x86_64 Linux executable.

set -euo pipefail

if [[ $# -gt 3 ]]; then
    echo "Usage: $0 [SOURCE_DIR [BUILD_DIR [OUTPUT_FILE]]]" >&2
    exit 2
fi

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source_dir=$(cd "${1:-${script_dir}}" && pwd)
build_dir=${2:-"${source_dir}/build-h0lite-single"}
output_file=${3:-"${source_dir}/bin/abacus_h0"}
build_cpus=${SLURM_CPUS_PER_TASK:-${BUILD_CPUS:-1}}

if [[ ! -f ${source_dir}/CMakeLists.txt ]]; then
    echo "Extract abacus-h0lite-v311_source.tar.gz and pass its source directory" >&2
    exit 2
fi
if [[ ! ${build_cpus} =~ ^[1-9][0-9]*$ ]]; then
    echo "BUILD_CPUS/SLURM_CPUS_PER_TASK must be a positive integer" >&2
    exit 2
fi

if [[ -z ${MKLROOT:-} || ! -r ${MKLROOT}/include/mkl.h ]]; then
    echo "MKLROOT must point to a readable oneMKL installation" >&2
    exit 2
fi
for build_tool in cmake g++; do
    if ! command -v "${build_tool}" >/dev/null; then
        echo "Required build tool not found in PATH: ${build_tool}" >&2
        exit 2
    fi
done
cxx_compiler=$(realpath "$(type -P g++)")
cxx_version=$("${cxx_compiler}" -dumpfullversion -dumpversion)
cxx_major=${cxx_version%%.*}
if [[ ! ${cxx_major} =~ ^[0-9]+$ ]] || (( cxx_major < 9 )); then
    echo "GCC 9+ required; selected ${cxx_compiler} (${cxx_version}). Load a newer GCC module first." >&2
    exit 2
fi
echo "Using C++ compiler: ${cxx_compiler} (${cxx_version})"

libgomp_archive=$("${cxx_compiler}" -print-file-name=libgomp.a)
if [[ ! -r ${libgomp_archive} ]]; then
    echo "static libgomp.a is required: ${libgomp_archive}" >&2
    exit 2
fi

extra_options=()
if [[ ! -f ${source_dir}/cmake/Sources.cmake ]]; then
    # Retain compatibility with an already-patched full developer checkout.
    extra_options=(-DENABLE_MPI=OFF -DENABLE_LCAO=ON -DENABLE_OPENMP=ON
        -DBUILD_TESTING=OFF -DMKL_LINK=static -DH0LITE_MKL_SEQUENTIAL=ON
        -DH0LITE_PORTABLE_STATIC=ON -DCOMMIT_INFO=OFF)
fi

if [[ -f ${build_dir}/CMakeCache.txt ]]; then
    cached_cxx=$(sed -n 's/^CMAKE_CXX_COMPILER:[^=]*=//p' "${build_dir}/CMakeCache.txt")
    if [[ -n ${cached_cxx} && ${cached_cxx} != "${cxx_compiler}" ]]; then
        cache_backup=$(mktemp -d "${build_dir}/compiler-cache-backup.XXXXXX")
        for cache_item in CMakeCache.txt CMakeFiles; do
            if [[ -e ${build_dir}/${cache_item} ]]; then
                mv "${build_dir}/${cache_item}" "${cache_backup}/"
            fi
        done
        echo "Compiler changed: ${cached_cxx} -> ${cxx_compiler}; old CMake cache saved in ${cache_backup}"
    fi
fi

cmake -S "${source_dir}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER:FILEPATH="${cxx_compiler}" \
    -DOpenMP_gomp_LIBRARY="${libgomp_archive}" \
    "${extra_options[@]}"

cmake --build "${build_dir}" --target abacus_h0lite --parallel "${build_cpus}"
dynamic_info=$(readelf -d "${build_dir}/abacus_h0")
if grep -Eq 'Shared library: \[lib(mkl|stdc\+\+|gcc_s|gomp|iomp)' <<< "${dynamic_info}"; then
    echo "Build still depends on compiler/oneMKL shared libraries; static build required" >&2
    printf '%s\n' "${dynamic_info}" >&2
    exit 1
fi
mkdir -p "$(dirname "${output_file}")"
if [[ "$(realpath -m "${build_dir}/abacus_h0")" != "$(realpath -m "${output_file}")" ]]; then
    install -m 0755 "${build_dir}/abacus_h0" "${output_file}"
fi

echo "Built ${output_file}"
file "${output_file}"
grep -E 'NEEDED|RPATH|RUNPATH' <<< "${dynamic_info}" || true
