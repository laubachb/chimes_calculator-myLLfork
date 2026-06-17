#!/bin/bash
#------------------------------------------------------------
# Stampede3 SLURM batch script: GPU ChIMES NVE test
#
# Submit from the test directory:
#   cd etc/lmp/tests/test_suite-GPU_NVE
#   sbatch submit_stampede3.slurm
#
# Results written to current_output/:
#   out.lammps   -- LAMMPS stdout (thermo table)
#   log.lammps   -- LAMMPS log
#   compare.log  -- empty = PASS, discrepancies = FAIL
#------------------------------------------------------------

#SBATCH -J chimes_gpu_nve
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -t 00:30:00
#SBATCH -A TG-CHM250118
#SBATCH -D /work2/09982/blaubach/stampede3/chimes_calculator-myLLfork/etc/lmp/tests/test_suite-GPU_NVE

# ------------------------------------------------------------------ #
# Environment                                                          #
# ------------------------------------------------------------------ #

module purge
module load intel/24.0
module load impi/21.11
module load gcc/13.2.0
module load cuda/12.8
module load python/3.12.11

# Confirm GPU is visible
echo "=== GPU info ==="
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader
echo ""

# ------------------------------------------------------------------ #
# Paths (absolute — SLURM runs from spool dir otherwise)              #
# ------------------------------------------------------------------ #

REPO=/work2/09982/blaubach/stampede3/chimes_calculator-myLLfork
TEST_DIR=${REPO}/etc/lmp/tests/test_suite-GPU_NVE
LMP_EXE=${REPO}/etc/lmp/exe/lmp_mpi_chimes_gpu
CPU_REF=${REPO}/etc/lmp/tests/test_suite-NVE/expected_output/log.lammps
COMPARE_PY=${REPO}/etc/lmp/tests/compare_logfiles.py
DATA_FILE=${REPO}/etc/lmp/tests/test_suite-NVE/diamond_3.21gcc_512atoms.data.in
PARAM_FILE=${REPO}/serial_interface/tests/force_fields/published_params.Carbon-2.0.Small.2+3+4b.Tersoff.txt

if [ ! -f "${LMP_EXE}" ]; then
    echo "ERROR: GPU LAMMPS binary not found: ${LMP_EXE}"
    echo "Build it with:  cd etc/lmp && ./install_gpu.sh"
    exit 1
fi

# ------------------------------------------------------------------ #
# Run                                                                  #
# ------------------------------------------------------------------ #

echo "=== Running ChIMES GPU NVE test (512 C atoms, 100 steps) ==="
date

mkdir -p "${TEST_DIR}/current_output"
cd "${TEST_DIR}/current_output"
cp "${TEST_DIR}/in.lammps" .
cp "${DATA_FILE}" diamond_3.21gcc_512atoms.data.in
cp "${PARAM_FILE}" params.txt

srun -n 1 "${LMP_EXE}" -i in.lammps > out.lammps 2>&1

echo "Run finished."
date

# ------------------------------------------------------------------ #
# Compare against CPU reference                                        #
# ------------------------------------------------------------------ #

echo ""
echo "=== Comparing GPU output vs CPU reference (tol=1e-3) ==="
python3 "${COMPARE_PY}" "${CPU_REF}" log.lammps 1e-3 > compare.log 2>&1

nfail=$(wc -l < compare.log)
if [ "${nfail}" -eq 0 ]; then
    echo "PASS -- GPU thermo values match CPU reference within 1e-3"
else
    echo "FAIL -- ${nfail} value(s) exceeded tolerance (see current_output/compare.log)"
    cat compare.log
fi
