#!/bin/bash
#------------------------------------------------------------
# Interactive GPU test runner for Stampede3
#
# Run from inside an idev GPU session:
#   idev -p gpu-h100 -N 1 -n 1 --gpus=1 -t 00:30:00
#   cd etc/lmp/tests/test_suite-GPU_NVE
#   ./run_gpu_test.sh
#
# Or specify a different LAMMPS binary as the first argument:
#   ./run_gpu_test.sh /path/to/lmp_mpi_chimes_gpu
#------------------------------------------------------------

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LMP_EXE="${1:-${SCRIPT_DIR}/../../exe/lmp_mpi_chimes_gpu}"

if [ ! -f "${LMP_EXE}" ]; then
    echo "ERROR: GPU LAMMPS binary not found: ${LMP_EXE}"
    echo "Build it first with:  cd etc/lmp && ./install_gpu.sh"
    exit 1
fi

echo "Using executable: ${LMP_EXE}"
echo ""

# Confirm GPU visible (non-fatal – lets job continue even without nvidia-smi)
if command -v nvidia-smi &> /dev/null; then
    echo "=== GPU info ==="
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
fi

# ------------------------------------------------------------------ #
# Run                                                                  #
# ------------------------------------------------------------------ #

rm -rf "${SCRIPT_DIR}/current_output"
mkdir -p "${SCRIPT_DIR}/current_output"
cd "${SCRIPT_DIR}/current_output"
cp "${SCRIPT_DIR}/in.lammps" .

echo "=== Running LAMMPS (GPU) ==="
date
srun -n 1 "${LMP_EXE}" -i in.lammps > out.lammps 2>&1
echo "Run finished."
date
echo ""

# ------------------------------------------------------------------ #
# Compare                                                              #
# ------------------------------------------------------------------ #

CPU_REF="${SCRIPT_DIR}/../test_suite-NVE/expected_output/log.lammps"

echo "=== Comparing GPU output vs CPU reference (tol=1e-3) ==="
python3 "${SCRIPT_DIR}/../compare_logfiles.py" "${CPU_REF}" log.lammps 1e-3 \
    > compare.log 2>&1

nfail=$(wc -l < compare.log)
if [ "${nfail}" -eq 0 ]; then
    echo "PASS -- all thermo values agree within 1e-3"
else
    echo "FAIL -- ${nfail} discrepancy(ies) found (see current_output/compare.log):"
    cat compare.log
    exit 1
fi
