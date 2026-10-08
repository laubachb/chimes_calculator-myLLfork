#!/bin/bash

# Element-aware fingerprint example. Usage: ./run_job.sh <nprocs>
# Generates cluster files with LAMMPS, combines them, then builds 2-, 3- and 4-body
# fingerprints at several values of alpha:
#   alpha = 1.0 : structure only
#   alpha = 0.5 : equal blend of structure and composition
#   alpha = 0.0 : composition only

NP=$1


param_file=./params.txt


srun -n $NP ../../../exe/lmp_mpi_chimes -i in.lammps > out.lammps


../../../../../chimesFF/src/FP/post_process.sh


for alpha in 1.0 0.5 0.0
do
    srun -n $NP ../../../../../chimesFF/src/FP/histogram_hybrid $param_file $alpha > histogram.alpha_${alpha}.log

    for body in 2b 3b 4b
    do
        mv 0-0.${body}_clu-s.hist 0-0.${body}_clu-s.alpha_${alpha}.hist
    done
done
