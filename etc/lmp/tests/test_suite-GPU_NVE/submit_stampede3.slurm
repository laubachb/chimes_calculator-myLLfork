#!/bin/bash
#------------------------------------------------------------
# Stampede3 SLURM batch script: GPU ChIMES NVE test
#
# Submits to the gpu-h100 partition (4x H100 per node).
# Adjust --partition to gpu-a100 if targeting A100 nodes.
#
# Submit from the test directory:
#   cd etc/lmp/tests/test_suite-GPU_NVE
#   sbatch submit_stampede3.slurm
#
# Results:
#   out.lammps     -- LAMMPS output log (thermo data)
#   log.lammps     -- LAMMPS log file
#   compare.log    -- diff vs CPU reference (empty = PASS)
#------------------------------------------------------------

#SBATCH -J chimes_gpu_nve
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH -p rtx-small
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -t 00:30:00
#SBATCH -A TG-CHM250118

# ------------------------------------------------------------------ #
# Environment                                                          #
# ------------------------------------------------------------------ #

module purge
module load gcc/13.2.0
module load impi/21.11
module load cuda/12.8
module load python3/3.12.11

# Confirm GPU is visible
echo "=== GPU info ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo ""

# ------------------------------------------------------------------ #
# Paths                                                               #
# ------------------------------------------------------------------ #

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LMP_EXE="${SCRIPT_DIR}/../../exe/lmp_mpi_chimes_gpu"

if [ ! -f "${LMP_EXE}" ]; then
    echo "ERROR: GPU LAMMPS binary not found at: ${LMP_EXE}"
    echo "Please build it first with:  cd etc/lmp && ./install_gpu.sh"
    exit 1
fi

# ------------------------------------------------------------------ #
# Run                                                                  #
# ------------------------------------------------------------------ #

echo "=== Running ChIMES GPU NVE test (512 C atoms, 100 steps) ==="
date

mkdir -p current_output
cd current_output
cp ../in.lammps .

srun -n 1 "${LMP_EXE}" -i in.lammps > out.lammps 2>&1

echo "Run finished."
date

# ------------------------------------------------------------------ #
# Compare against CPU reference                                        #
# ------------------------------------------------------------------ #

CPU_REF="${SCRIPT_DIR}/../test_suite-NVE/expected_output/log.lammps"

echo ""
echo "=== Comparing GPU output against CPU reference ==="
python3 "${SCRIPT_DIR}/../compare_logfiles.py" "${CPU_REF}" log.lammps 1e-3 \
    > compare.log 2>&1

nfail=$(wc -l < compare.log)
if [ "${nfail}" -eq 0 ]; then
    echo "PASS -- GPU thermo values match CPU reference within 1e-3"
else
    echo "FAIL -- ${nfail} value(s) exceeded tolerance (see compare.log)"
    cat compare.log
fi
