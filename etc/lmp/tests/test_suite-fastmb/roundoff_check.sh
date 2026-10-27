#!/bin/bash
# Checks that "pair_style chimesFF fastmb" reproduces the default evaluation to floating-point roundoff:
# energy, virial and every force component, for a single-element and a two-element model and for a box
# small enough to contain periodic self-images, with each fastmb option, on the MPI ranks given
# (default 4). Use more than one rank: some failure modes (e.g. inconsistent cluster ownership) only
# show up then. Requires a LAMMPS executable built with ./install.sh FASTMB (../../exe/lmp_mpi_chimes),
# or set LMP.
#
#   usage: ./roundoff_check.sh [nprocs] [mpi launcher, default "srun -n"]

NP=${1:-4}
LAUNCH=${2:-"srun -n"}
LMP=${LMP:-../../exe/lmp_mpi_chimes}
FF=../../../../serial_interface/tests/force_fields
status=0

run() { # tag style data param rep
    $LAUNCH $NP $LMP -in in.roundoff -var style "$2" -var data $3 -var param_file $4 -var rep $5 -var tag $1 \
        -log log.$1 -screen none > out.$1 2>&1
}
check() { # name data param rep
    local name=$1 data=$2 param=$3 rep=$4
    run ${name}_default "chimesFF" $data $param $rep
    for opts in "" "nosimd" "norips" "orient tag" "nok4" "nosimd norips orient tag nok4"; do
        local t=${name}_fastmb_$(echo "x $opts" | tr -d ' ')
        run $t "chimesFF fastmb $opts" $data $param $rep
        python3 roundoff_compare.py "$name fastmb $opts" log.${name}_default log.$t forces_${name}_default.dump forces_$t.dump || status=1
    done
}

check carbon2.0-small liqC_2.5gcc_6000K.data.in $FF/published_params.Carbon-2.0.Small.2+3+4b.Tersoff.txt      2
check HN3             ../example-basic_HN3/data.in $FF/published_params.HN3.2+3+4b.Tersoff.special.offsets.txt 1
# 11.7 A box with a 6 A padded 3-body cutoff: periodic self-images, so cluster lists fall back to the default loops
check carbon2.0-large liqC_2.5gcc_6000K.data.in $FF/published_params.Carbon-2.0.Large.2+3+4b.Tersoff.txt      1

[ $status -eq 0 ] && echo "fastmb roundoff check passed" || echo "fastmb roundoff check FAILED"
exit $status
