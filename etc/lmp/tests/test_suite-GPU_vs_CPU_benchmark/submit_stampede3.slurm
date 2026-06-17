#!/bin/bash
#------------------------------------------------------------
# Stampede3 SLURM batch script: ChIMES CPU vs GPU benchmark
#
# Submit from the test directory:
#   cd etc/lmp/tests/test_suite-GPU_vs_CPU_benchmark
#   sbatch submit_stampede3.slurm
#
# Results:
#   current_output_cpu/   -- CPU run outputs
#   current_output_gpu/   -- GPU run outputs
#   benchmark_results.txt -- force consistency + speedup summary
#------------------------------------------------------------

#SBATCH -J chimes_cpu_gpu_bench
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -t 00:30:00
#SBATCH -A TG-CHM250118
#SBATCH -D /work2/09982/blaubach/stampede3/chimes_calculator-myLLfork/etc/lmp/tests/test_suite-GPU_vs_CPU_benchmark

# ------------------------------------------------------------------ #
# Environment                                                          #
# ------------------------------------------------------------------ #

module purge
module load intel/24.0
module load impi/21.11
module load gcc/13.2.0
module load cuda/12.8
module load python/3.12.11

echo "=== GPU info ==="
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader
echo ""

# ------------------------------------------------------------------ #
# Paths                                                                #
# ------------------------------------------------------------------ #

REPO=/work2/09982/blaubach/stampede3/chimes_calculator-myLLfork
TEST_DIR=${REPO}/etc/lmp/tests/test_suite-GPU_vs_CPU_benchmark
LMP_CPU=${REPO}/etc/lmp/exe/lmp_mpi_chimes
LMP_GPU=${REPO}/etc/lmp/exe/lmp_mpi_chimes_gpu
DATA_FILE=${REPO}/etc/lmp/tests/test_suite-NVE/diamond_3.21gcc_512atoms.data.in
PARAM_FILE=${REPO}/serial_interface/tests/force_fields/published_params.Carbon-2.0.Small.2+3+4b.Tersoff.txt
OUT_CPU=${TEST_DIR}/current_output_cpu
OUT_GPU=${TEST_DIR}/current_output_gpu

# ------------------------------------------------------------------ #
# Sanity checks                                                        #
# ------------------------------------------------------------------ #

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

# ------------------------------------------------------------------ #
# Prepare output directories                                           #
# ------------------------------------------------------------------ #

mkdir -p "${OUT_CPU}" "${OUT_GPU}"
for d in "${OUT_CPU}" "${OUT_GPU}"; do
    cp "${TEST_DIR}/in.lammps" "${d}/"
    cp "${DATA_FILE}"  "${d}/diamond_3.21gcc_512atoms.data.in"
    cp "${PARAM_FILE}" "${d}/params.txt"
done

# ------------------------------------------------------------------ #
# CPU run                                                              #
# ------------------------------------------------------------------ #

echo "=== Running CPU (1 rank) ==="
date
cd "${OUT_CPU}"
srun -n 1 "${LMP_CPU}" -i in.lammps -log log.lammps > out.lammps 2>&1
echo "CPU done."
date

# ------------------------------------------------------------------ #
# GPU run                                                              #
# ------------------------------------------------------------------ #

echo ""
echo "=== Running GPU (1 rank) ==="
date
cd "${OUT_GPU}"
srun -n 1 "${LMP_GPU}" -i in.lammps -log log.lammps > out.lammps 2>&1
echo "GPU done."
date

# ------------------------------------------------------------------ #
# Analysis                                                             #
# ------------------------------------------------------------------ #

echo ""
echo "=== Benchmark analysis ==="
cd "${TEST_DIR}"
python3 benchmark.py \
    --cpu-log  "${OUT_CPU}/log.lammps" \
    --gpu-log  "${OUT_GPU}/log.lammps" \
    --cpu-dump "${OUT_CPU}/forces.dump" \
    --gpu-dump "${OUT_GPU}/forces.dump" \
    --tol 1e-3 | tee benchmark_results.txt

echo ""
echo "Results saved to: ${TEST_DIR}/benchmark_results.txt"
