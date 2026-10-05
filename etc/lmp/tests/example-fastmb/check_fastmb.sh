#!/bin/bash
# Checks that "pair_style chimesFF fastmb" reproduces the stock evaluation to roundoff:
# energy, virial and every force component, for single- and two-element models and for a box small
# enough to contain periodic self-images, with each fastmb option, on the MPI ranks given (default 4).
# Use more than one rank: some failure modes (e.g. inconsistent cluster ownership) only show up then.
#
#   usage: ./check_fastmb.sh [nprocs] [mpi launcher, default "srun -n"]
#
# Requires a LAMMPS executable built by ../../install.sh (../../exe/lmp_mpi_chimes), or set LMP.

NP=${1:-4}
LAUNCH=${2:-"srun -n"}
LMP=${LMP:-../../exe/lmp_mpi_chimes}     # or set LMP=/path/to/lmp
FF=../../../../serial_interface/tests/force_fields
status=0

run() { # tag style data param rep
    $LAUNCH $NP $LMP -in in.check -var style "$2" -var data $3 -var param_file $4 -var rep $5 -var tag $1 \
        -log log.$1 -screen none > out.$1 2>&1
}
check() { # name data param rep options...
    local name=$1 data=$2 param=$3 rep=$4; shift 4
    run ${name}_stock "chimesFF" $data $param $rep
    for opts in "" "nosimd" "norips" "orient tag"; do
        local t=${name}_fast_$(echo "x $opts" | tr -d ' ')
        run $t "chimesFF fastmb $opts" $data $param $rep
        python3 compare_fastmb.py "$name fastmb $opts" log.${name}_stock log.$t forces_${name}_stock.dump forces_$t.dump || status=1
    done
}

check carbon2.0-small liqC_2.5gcc_6000K.data      $FF/published_params.Carbon-2.0.Small.2+3+4b.Tersoff.txt          2
check HN3             ../example-basic_HN3/data.in $FF/published_params.HN3.2+3+4b.Tersoff.special.offsets.txt     1
# 11.7 A box with 6 A padded 3-body cutoff: periodic self-images, so cluster lists use the stock loops on all ranks
check carbon2.0-large liqC_2.5gcc_6000K.data      $FF/published_params.Carbon-2.0.Large.2+3+4b.Tersoff.txt          1

[ $status -eq 0 ] && echo "fastmb check passed" || echo "fastmb check FAILED"
exit $status
