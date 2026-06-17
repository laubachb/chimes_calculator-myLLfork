#!/bin/bash
#
# install_gpu.sh -- Build LAMMPS with ChIMES + CUDA GPU acceleration
#
# Prerequisites (load before running, or set hosttype):
#   Stampede3 H100 nodes:
#     module load intel/24.0 impi/21.11 cuda/12.4
#     export hosttype=UT-TACC
#
#   Alternatively set hosttype for any of the supported machines that
#   have a corresponding modfiles/*.mod entry.  For other machines, load
#   compilers/MPI/CUDA manually and run without hosttype.
#
# Usage:
#   ./install_gpu.sh [CUDA_ARCH]
#
#   CUDA_ARCH (optional) - SM architecture number, default 90 (H100).
#                          Use 80 for A100, 86 for A30/A40, 89 for L40S.
#
# Output:
#   exe/lmp_mpi_chimes_gpu

set -e

CUDA_ARCH="${1:-90}"

echo ""
echo "=========================================================="
echo " ChIMES + CUDA LAMMPS build"
echo " CUDA SM architecture: sm_${CUDA_ARCH}"
echo "=========================================================="
echo ""

# ------------------------------------------------------------------ #
# Locate CUDA toolkit                                                  #
# ------------------------------------------------------------------ #

if [ -z "$CUDA_PATH" ]; then
    # Try common locations if the env var is not set
    for candidate in \
        "$(dirname "$(which nvcc)" 2>/dev/null)/.." \
        /usr/local/cuda \
        /opt/cuda \
        /home1/apps/nvidia/Linux_x86_64/25.3/cuda/12.8
    do
        if [ -f "${candidate}/include/cuda_runtime.h" ]; then
            export CUDA_PATH="$(realpath "${candidate}")"
            break
        fi
    done
fi

if [ -z "$CUDA_PATH" ] || [ ! -f "${CUDA_PATH}/include/cuda_runtime.h" ]; then
    echo "ERROR: Cannot find CUDA toolkit."
    echo "Please set CUDA_PATH to the CUDA installation root (e.g. /usr/local/cuda)"
    exit 1
fi

echo "Using CUDA toolkit: ${CUDA_PATH}"
echo "nvcc version:"
nvcc --version 2>&1 | head -3

# ------------------------------------------------------------------ #
# Load host-specific modules                                           #
# ------------------------------------------------------------------ #

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "$hosttype" ]; then
    echo ""
    echo "WARNING: No hosttype specified – assuming modules are already loaded."
    echo ""
elif [[ "$hosttype" == "UT-TACC" ]]; then
    source "${SCRIPT_DIR}/modfiles/UT-TACC.mod"
elif [[ "$hosttype" == "LLNL-LC" ]]; then
    source "${SCRIPT_DIR}/modfiles/LLNL-LC.mod"
elif [[ "$hosttype" == "UM-ARC" ]]; then
    source "${SCRIPT_DIR}/modfiles/UM-ARC.mod"
elif [[ "$hosttype" == "JHU-ARCH" ]]; then
    source "${SCRIPT_DIR}/modfiles/JHU-ARCH.mod"
else
    echo "ERROR: Unknown hosttype ($hosttype)."
    echo "Valid options:"
    for m in "${SCRIPT_DIR}/modfiles/"*.mod; do echo "   ${m%.mod##*/}"; done
    exit 1
fi

# ------------------------------------------------------------------ #
# Step 1: Pre-compile chimesFF_gpu.cu → static archive                #
# ------------------------------------------------------------------ #

REPO_ROOT="$(realpath "${SCRIPT_DIR}/../..")"
GPU_BUILD_DIR="${SCRIPT_DIR}/build_gpu"
mkdir -p "${GPU_BUILD_DIR}"

echo ""
echo "[1/4] Compiling chimesFF_gpu.cu with nvcc (sm_${CUDA_ARCH})..."

# Relocatable device code object
nvcc -O3 \
     -arch=sm_${CUDA_ARCH} \
     -std=c++11 \
     -Xcompiler -fPIC \
     -DUSE_CUDA \
     -DMAX_POLY_ORDER=24 \
     -I"${REPO_ROOT}/chimesFF/src" \
     -I"${CUDA_PATH}/include" \
     -dc "${REPO_ROOT}/chimesFF/src/chimesFF_gpu.cu" \
     -o "${GPU_BUILD_DIR}/chimesFF_gpu.o"

# Device-link step (generates final CUDA runtime glue)
nvcc -arch=sm_${CUDA_ARCH} \
     -dlink \
     "${GPU_BUILD_DIR}/chimesFF_gpu.o" \
     -o "${GPU_BUILD_DIR}/chimesFF_gpu_dlink.o" \
     -lcudadevrt

# Bundle into a static library so the LAMMPS Makefile can reference it cleanly
ar rcs "${GPU_BUILD_DIR}/libchimescalc_gpu.a" \
       "${GPU_BUILD_DIR}/chimesFF_gpu.o" \
       "${GPU_BUILD_DIR}/chimesFF_gpu_dlink.o"

export CHIMES_GPU_LIB="${GPU_BUILD_DIR}/libchimescalc_gpu.a"
echo "    --> ${CHIMES_GPU_LIB}"

# ------------------------------------------------------------------ #
# Step 2: Clone LAMMPS                                                 #
# ------------------------------------------------------------------ #

lammps="stable_29Aug2024_update1"
mkdir -p "${SCRIPT_DIR}/build/${lammps}"

echo ""
echo "[2/4] Cloning LAMMPS ${lammps}..."

if [ ! -d "${SCRIPT_DIR}/build/${lammps}/.git" ]; then
    git clone --depth 1 --branch ${lammps} \
        https://github.com/lammps/lammps.git \
        "${SCRIPT_DIR}/build/${lammps}"
else
    echo "    (already cloned, skipping)"
fi

# ------------------------------------------------------------------ #
# Step 3: Copy ChIMES source files                                     #
# ------------------------------------------------------------------ #

echo ""
echo "[3/4] Copying ChIMES source files to LAMMPS MANYBODY/..."

MANYBODY="${SCRIPT_DIR}/build/${lammps}/src/MANYBODY"

cp "${REPO_ROOT}/chimesFF/src/chimesFF.h"       "${MANYBODY}/"
cp "${REPO_ROOT}/chimesFF/src/chimesFF.cpp"     "${MANYBODY}/"
cp "${REPO_ROOT}/chimesFF/src/chimesFF_gpu.cuh" "${MANYBODY}/"
cp "${SCRIPT_DIR}/src/pair_chimes.h"            "${MANYBODY}/"
cp "${SCRIPT_DIR}/src/pair_chimes.cpp"          "${MANYBODY}/"
cp "${SCRIPT_DIR}/etc/pair.h"                   "${SCRIPT_DIR}/build/${lammps}/src/"
cp "${SCRIPT_DIR}/etc/pair.cpp"                 "${SCRIPT_DIR}/build/${lammps}/src/"

# LAMMPS only symlinks .cpp/.h files from MANYBODY into src/ — copy .cuh directly
# so that chimesFF.cpp can find it with a plain #include "chimesFF_gpu.cuh"
cp "${REPO_ROOT}/chimesFF/src/chimesFF_gpu.cuh" "${SCRIPT_DIR}/build/${lammps}/src/"

# Install the GPU-aware Makefile
sed \
    -e "s|\$(CUDA_PATH)|${CUDA_PATH}|g" \
    -e "s|\$(CHIMES_GPU_LIB)|${CHIMES_GPU_LIB}|g" \
    "${SCRIPT_DIR}/etc/Makefile.mpi_chimes_gpu" \
    > "${SCRIPT_DIR}/build/${lammps}/src/MAKE/Makefile.mpi_chimes_gpu"

# ------------------------------------------------------------------ #
# Step 4: Build LAMMPS                                                 #
# ------------------------------------------------------------------ #

echo ""
echo "[4/4] Building LAMMPS (lmp_mpi_chimes_gpu)..."

cd "${SCRIPT_DIR}/build/${lammps}/src"
make yes-manybody
make yes-extra-pair
make -j 4 mpi_chimes_gpu
cd "${SCRIPT_DIR}"

mkdir -p "${SCRIPT_DIR}/exe"
mv "${SCRIPT_DIR}/build/${lammps}/src/lmp_mpi_chimes_gpu" "${SCRIPT_DIR}/exe/"

echo ""
echo "=========================================================="
echo " Build complete."
echo " GPU-enabled LAMMPS executable:"
echo "   ${SCRIPT_DIR}/exe/lmp_mpi_chimes_gpu"
echo ""
echo " Quick-test (requires a GPU node):"
echo "   cd tests/test_suite-GPU_NVE"
echo "   ./run_gpu_test.sh"
echo "=========================================================="
