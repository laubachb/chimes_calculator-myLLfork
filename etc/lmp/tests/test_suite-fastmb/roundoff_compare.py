#!/usr/bin/env python3
"""Compare a default and a fastmb single point (LAMMPS log and force dump); exit status 1 if they differ
by more than floating-point roundoff. fastmb reorders the sums of the default evaluation, so the
differences are at the 1e-15 to 1e-13 level, as between runs of the default code on different numbers
of MPI ranks."""
import sys
import numpy as np

TOL = 1e-10   # relative

def thermo(log):
    lines = open(log).read().splitlines()
    i = [k for k, s in enumerate(lines) if s.split()[:2] == ['Step', 'Atoms']][0]
    return np.array([float(x) for x in lines[i + 1].split()])

def forces(dump):
    lines = open(dump).read().splitlines()
    i = [k for k, s in enumerate(lines) if s.startswith('ITEM: ATOMS')][0]
    return np.array([[float(x) for x in s.split()[2:5]] for s in lines[i + 1:]])

name, log0, log1, dump0, dump1 = sys.argv[1:6]
a, b = thermo(log0), thermo(log1)
F0, F1 = forces(dump0), forces(dump1)
dE = abs(a[2] - b[2]) / abs(a[2])
dS = np.max(np.abs(a[4:] - b[4:])) / np.max(np.abs(a[4:]))
dF = np.max(np.abs(F0 - F1)) / np.sqrt(np.mean(F0**2))
ok = dE < TOL and dS < TOL and dF < TOL
print(f"{'PASS' if ok else 'FAIL'}  {name:34s} energy {dE:.1e}  virial {dS:.1e}  forces (max/rms) {dF:.1e}")
sys.exit(0 if ok else 1)
