#!/usr/bin/env python3
"""
benchmark.py  --  CPU vs GPU ChIMES LAMMPS benchmark analyser.

Checks two things:
  1. Force consistency : max per-atom, per-component absolute difference
                         between the step-0 force dumps from the two runs.
  2. Thermo consistency: all per-step thermo columns match within tolerance.

Reports:
  - Per-component force statistics (max, mean abs err; pass/fail)
  - Thermo pass/fail summary
  - Wall-time speedup (GPU loop time / CPU loop time)

Usage:
    python3 benchmark.py \\
        --cpu-log  current_output_cpu/log.lammps \\
        --gpu-log  current_output_gpu/log.lammps \\
        --cpu-dump current_output_cpu/forces.dump \\
        --gpu-dump current_output_gpu/forces.dump \\
        [--tol 1e-3]

Exit code: 0 = PASS, 1 = FAIL (forces or thermo outside tolerance)
"""

import sys
import re
import argparse
from pathlib import Path


# ---------------------------------------------------------------------------
# LAMMPS log parsing
# ---------------------------------------------------------------------------

def parse_log(path):
    """
    Return (thermo_rows, loop_times) from a LAMMPS log file.

    thermo_rows : list of {col_name: float} dicts, one per thermo output line.
    loop_times  : list of float (seconds) — one per 'run N' block.
    """
    thermo_rows = []
    loop_times  = []
    headers     = []

    with open(path) as fh:
        lines = fh.readlines()

    in_block = False
    for line in lines:
        # Detect thermo header
        m_hdr = re.match(r"^\s*(Step\s+.+)", line)
        if m_hdr:
            headers = m_hdr.group(1).split()
            in_block = True
            continue

        if in_block:
            m_loop = re.match(r"\s*Loop time of\s+([\d.eE+\-]+)", line)
            if m_loop:
                loop_times.append(float(m_loop.group(1)))
                in_block = False
                continue
            parts = line.split()
            if parts and len(parts) == len(headers):
                try:
                    row = {h: float(v) for h, v in zip(headers, parts)}
                    thermo_rows.append(row)
                except ValueError:
                    pass

    return thermo_rows, loop_times


# ---------------------------------------------------------------------------
# LAMMPS dump parsing
# ---------------------------------------------------------------------------

def parse_forces_dump(path):
    """
    Parse a LAMMPS 'dump ... custom ... id type x y z fx fy fz' file.
    Returns a dict  atom_id -> {'fx': float, 'fy': float, 'fz': float}.
    Only the first timestep block is read.
    """
    forces = {}
    col_names = []

    with open(path) as fh:
        lines = fh.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("ITEM: ATOMS"):
            col_names = line.split()[2:]  # drop "ITEM:" and "ATOMS"
            i += 1
            break
        i += 1

    if not col_names:
        raise RuntimeError(f"No 'ITEM: ATOMS' header found in {path}")

    id_idx = col_names.index("id")
    fx_idx = col_names.index("fx")
    fy_idx = col_names.index("fy")
    fz_idx = col_names.index("fz")

    while i < len(lines):
        parts = lines[i].strip().split()
        if not parts or parts[0].startswith("ITEM"):
            break
        atom_id = int(parts[id_idx])
        forces[atom_id] = {
            "fx": float(parts[fx_idx]),
            "fy": float(parts[fy_idx]),
            "fz": float(parts[fz_idx]),
        }
        i += 1

    return forces


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

def compare_forces(cpu_forces, gpu_forces, tol):
    """
    Return (passed, stats_dict).
    stats_dict has keys: max_abs_fx, max_abs_fy, max_abs_fz,
                         mean_abs_fx, mean_abs_fy, mean_abs_fz,
                         n_atoms.
    """
    common_ids = sorted(set(cpu_forces) & set(gpu_forces))
    if not common_ids:
        return False, {}

    diffs = {"fx": [], "fy": [], "fz": []}
    for aid in common_ids:
        for comp in ("fx", "fy", "fz"):
            diffs[comp].append(abs(cpu_forces[aid][comp] - gpu_forces[aid][comp]))

    stats = {
        "n_atoms":    len(common_ids),
        "max_abs_fx": max(diffs["fx"]),
        "max_abs_fy": max(diffs["fy"]),
        "max_abs_fz": max(diffs["fz"]),
        "mean_abs_fx": sum(diffs["fx"]) / len(diffs["fx"]),
        "mean_abs_fy": sum(diffs["fy"]) / len(diffs["fy"]),
        "mean_abs_fz": sum(diffs["fz"]) / len(diffs["fz"]),
    }
    passed = all(stats[k] <= tol for k in ("max_abs_fx", "max_abs_fy", "max_abs_fz"))
    return passed, stats


def compare_thermo(cpu_rows, gpu_rows, tol):
    """
    Return (passed, failures_list).
    """
    if len(cpu_rows) != len(gpu_rows):
        return False, [f"Row count mismatch: CPU={len(cpu_rows)}, GPU={len(gpu_rows)}"]

    failures = []
    for step_i, (cr, gr) in enumerate(zip(cpu_rows, gpu_rows)):
        for col in cr:
            if col not in gr:
                continue
            err = abs(cr[col] - gr[col])
            if err > tol:
                rel = err / max(abs(cr[col]), 1e-30)
                failures.append(
                    f"  Row {step_i:4d}  {col:12s}: "
                    f"cpu={cr[col]:15.6f}  gpu={gr[col]:15.6f}  "
                    f"|diff|={err:.3e}  rel={rel:.3e}"
                )
    return (len(failures) == 0), failures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cpu-log",  required=True, help="CPU LAMMPS log file")
    ap.add_argument("--gpu-log",  required=True, help="GPU LAMMPS log file")
    ap.add_argument("--cpu-dump", required=True, help="CPU forces dump file")
    ap.add_argument("--gpu-dump", required=True, help="GPU forces dump file")
    ap.add_argument("--tol",      type=float, default=1e-3,
                    help="Absolute tolerance for force and thermo comparison (default 1e-3)")
    args = ap.parse_args()

    sep  = "=" * 70
    sep2 = "-" * 70
    ok   = True

    print(sep)
    print("  ChIMES CPU vs GPU Benchmark")
    print(sep)
    print(f"  CPU log : {args.cpu_log}")
    print(f"  GPU log : {args.gpu_log}")
    print(f"  CPU dump: {args.cpu_dump}")
    print(f"  GPU dump: {args.gpu_dump}")
    print(f"  Tolerance: {args.tol:.2e}")
    print(sep)

    # ------------------------------------------------------------------ #
    # 1. Force consistency
    # ------------------------------------------------------------------ #
    print("\n[1] Per-atom force consistency at step 0")
    print(sep2)
    try:
        cpu_f = parse_forces_dump(args.cpu_dump)
        gpu_f = parse_forces_dump(args.gpu_dump)
        f_pass, fstats = compare_forces(cpu_f, gpu_f, args.tol)

        print(f"  Atoms compared : {fstats.get('n_atoms', 0)}")
        for comp in ("fx", "fy", "fz"):
            print(f"  {comp}: max|Δ| = {fstats[f'max_abs_{comp}']:.4e}  "
                  f"mean|Δ| = {fstats[f'mean_abs_{comp}']:.4e}  "
                  f"(tol={args.tol:.2e})")

        if f_pass:
            print("  Force check: PASS")
        else:
            print("  Force check: FAIL  *** one or more components exceed tolerance ***")
            ok = False

    except Exception as exc:
        print(f"  ERROR parsing force dumps: {exc}")
        f_pass = False
        ok = False

    # ------------------------------------------------------------------ #
    # 2. Thermo consistency
    # ------------------------------------------------------------------ #
    print("\n[2] Thermo consistency across all steps")
    print(sep2)
    try:
        cpu_thermo, cpu_times = parse_log(args.cpu_log)
        gpu_thermo, gpu_times = parse_log(args.gpu_log)

        t_pass, failures = compare_thermo(cpu_thermo, gpu_thermo, args.tol)

        if t_pass:
            cols = list(cpu_thermo[0].keys()) if cpu_thermo else []
            print(f"  Thermo rows : {len(cpu_thermo)}")
            print(f"  Columns checked: {', '.join(cols)}")
            print("  Thermo check: PASS")
        else:
            print(f"  Thermo check: FAIL  ({len(failures)} violation(s))")
            for msg in failures[:20]:
                print(msg)
            if len(failures) > 20:
                print(f"  ... ({len(failures) - 20} more)")
            ok = False

    except Exception as exc:
        print(f"  ERROR parsing log files: {exc}")
        t_pass = False
        ok = False
        cpu_times, gpu_times = [], []

    # ------------------------------------------------------------------ #
    # 3. Timing / speedup
    # ------------------------------------------------------------------ #
    print("\n[3] Wall-time speedup")
    print(sep2)

    # The benchmark input has two run blocks:
    #   run 0   → phase 1 (force dump, near-zero time)
    #   run N   → phase 2 (timed MD loop)
    # We want the second loop time for each.
    def timed_run(times):
        if len(times) >= 2:
            return times[1]       # second run block = the actual MD
        elif len(times) == 1:
            return times[0]
        return None

    cpu_t = timed_run(cpu_times)
    gpu_t = timed_run(gpu_times)

    if cpu_t is not None and gpu_t is not None:
        speedup = cpu_t / gpu_t
        print(f"  CPU loop time : {cpu_t:.4f} s")
        print(f"  GPU loop time : {gpu_t:.4f} s")
        print(f"  Speedup (CPU/GPU): {speedup:.2f}x")
        if speedup < 1.0:
            print("  NOTE: GPU is slower — this is expected for very small systems")
            print("        or when the GPU binary was compiled for a different arch.")
        elif speedup < 2.0:
            print("  NOTE: Modest speedup. GPU overhead (transfers, kernels) dominates")
            print("        for this system size. Larger simulations benefit more.")
        else:
            print("  NOTE: Meaningful GPU speedup achieved.")
    else:
        print("  Could not extract loop times from one or both log files.")
        print(f"    CPU loop times found: {cpu_times}")
        print(f"    GPU loop times found: {gpu_times}")

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    print()
    print(sep)
    if ok:
        print("  OVERALL: PASS -- CPU and GPU results are consistent within tolerance")
    else:
        print("  OVERALL: FAIL -- see details above")
    print(sep)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
