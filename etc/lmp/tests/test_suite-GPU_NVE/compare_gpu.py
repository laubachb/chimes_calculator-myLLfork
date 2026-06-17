#!/usr/bin/env python3
"""
compare_gpu.py -- compare a GPU LAMMPS log against a CPU reference log.

GPU floating-point results differ from CPU due to the non-associative
order of atomicAdd reductions on the GPU.  This script uses a default
tolerance of 1e-3 (100x looser than the CPU-to-CPU default of 1e-5) to
account for that non-determinism.

Usage:
    python3 compare_gpu.py reference.log gpu.log [tolerance]

    tolerance  optional, default 1e-3

Exit code:
    0  all values within tolerance (PASS)
    1  one or more values exceed tolerance (FAIL)
"""

import sys
import re


def extract_thermo(filename):
    """Return list of thermo rows (each row is a list of floats/strings)."""
    rows = []
    in_block = False
    found_step = False
    found_loop = False

    with open(filename) as fh:
        for line in fh:
            if not in_block and re.match(r"^\s*Step\s+", line):
                in_block = True
                found_step = True
                continue
            if in_block:
                if "Loop time" in line:
                    found_loop = True
                    break
                if line.strip():
                    rows.append(line.split())

    if not found_step:
        print(f"ERROR: 'Step' header not found in {filename}")
    if not found_loop:
        print(f"ERROR: 'Loop time' footer not found in {filename}")
    if not (found_step and found_loop):
        return None
    return rows


def compare(ref_file, gpu_file, tol=1e-3):
    ref  = extract_thermo(ref_file)
    gpu  = extract_thermo(gpu_file)

    if ref is None or gpu is None:
        print("Aborting: could not parse one or both log files.")
        return False

    if len(ref) != len(gpu):
        print(f"Row count mismatch: reference has {len(ref)}, GPU has {len(gpu)}")
        return False

    failures = []
    for step_idx, (rrow, grow) in enumerate(zip(ref, gpu)):
        if len(rrow) != len(grow):
            failures.append(
                f"Step {step_idx}: column count mismatch ({len(rrow)} vs {len(grow)})"
            )
            continue
        for col_idx, (rv, gv) in enumerate(zip(rrow, grow)):
            try:
                rf, gf = float(rv), float(gv)
                err = abs(rf - gf)
                if err > tol:
                    # Compute relative error for context
                    rel = err / max(abs(rf), 1e-30)
                    failures.append(
                        f"Step {step_idx}, Col {col_idx}: "
                        f"ref={rf:.8g}  gpu={gf:.8g}  "
                        f"abs_err={err:.3e}  rel_err={rel:.3e}"
                    )
            except ValueError:
                pass  # non-numeric field (e.g. "Step")

    if failures:
        for msg in failures:
            print(msg)
        return False

    return True


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    ref_file = sys.argv[1]
    gpu_file = sys.argv[2]
    tol      = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-3

    passed = compare(ref_file, gpu_file, tol)
    sys.exit(0 if passed else 1)
