#!/bin/bash
# ---------------------------------------------------------------------------
# run_benchmark.sh -- interactive CPU vs GPU ChIMES benchmark
#
# Must be run from a GPU compute node (login nodes have no GPU).
# On Stampede3:
#   idev -p rtx-small -N 1 -n 1 -t 00:30:00 -A <YOUR_ALLOCATION>
#   module purge
#   module load intel/24.0 impi/21.11 gcc/13.2.0 cuda/12.8 python/3.12.11
#   cd <repo>/etc/lmp/tests/test_suite-GPU_vs_CPU_benchmark
#   ./run_benchmark.sh
# ---------------------------------------------------------------------------

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

LMP_CPU="${REPO}/etc/lmp/exe/lmp_mpi_chimes"
LMP_GPU="${REPO}/etc/lmp/exe/lmp_mpi_chimes_gpu"
DATA_FILE="${REPO}/etc/lmp/tests/test_suite-NVE/diamond_3.21gcc_512atoms.data.in"
PARAM_FILE="${REPO}/serial_interface/tests/force_fields/published_params.Carbon-2.0.Small.2+3+4b.Tersoff.txt"

OUT_CPU="${SCRIPT_DIR}/current_output_cpu"
OUT_GPU="${SCRIPT_DIR}/current_output_gpu"

# ---- sanity checks -------------------------------------------------------

if [ ! -f "${LMP_CPU}" ]; then
    echo "ERROR: CPU binary not found: ${LMP_CPU}"
    echo "Build with:  cd ${REPO}/etc/lmp && ./install.sh"
    exit 1
fi

if [ ! -f "${LMP_GPU}" ]; then
    echo "ERROR: GPU binary not found: ${LMP_GPU}"
    echo "Build with:  cd ${REPO}/etc/lmp && ./install_gpu.sh <SM_ARCH>"
    exit 1
fi

# ---- setup output dirs ---------------------------------------------------

mkdir -p "${OUT_CPU}" "${OUT_GPU}"
for d in "${OUT_CPU}" "${OUT_GPU}"; do
    cp "${SCRIPT_DIR}/in.lammps" "${d}/"
    cp "${DATA_FILE}"  "${d}/diamond_3.21gcc_512atoms.data.in"
    cp "${PARAM_FILE}" "${d}/params.txt"
done

# ---- CPU run -------------------------------------------------------------

echo "=== Running CPU ==="
date
cd "${OUT_CPU}"
srun -n 1 "${LMP_CPU}" -i in.lammps -log log.lammps > out.lammps 2>&1
echo "CPU done."
date

# ---- GPU run -------------------------------------------------------------

echo ""
echo "=== Running GPU ==="
date
cd "${OUT_GPU}"
srun -n 1 "${LMP_GPU}" -i in.lammps -log log.lammps > out.lammps 2>&1
echo "GPU done."
date

# ---- analysis ------------------------------------------------------------

echo ""
echo "=== Benchmark analysis ==="
cd "${SCRIPT_DIR}"
python3 benchmark.py \
    --cpu-log  "${OUT_CPU}/log.lammps" \
    --gpu-log  "${OUT_GPU}/log.lammps" \
    --cpu-dump "${OUT_CPU}/forces.dump" \
    --gpu-dump "${OUT_GPU}/forces.dump" \
    --tol 1e-3 | tee benchmark_results.txt

echo ""
echo "Full results saved to: ${SCRIPT_DIR}/benchmark_results.txt"
