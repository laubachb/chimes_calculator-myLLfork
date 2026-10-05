# fastmb: exact fast many-body evaluation for ChIMES in LAMMPS

`pair_style chimesFF fastmb` evaluates the 3- and 4-body ChIMES terms with the same arithmetic as
`chimesFF::compute_3B` / `compute_4B`, reordered. Energies, forces and the virial agree with the
default evaluation to floating-point roundoff. Parameter files are unchanged.

```
neighbor        0.3 bin
neigh_modify    delay 0 every 1 check yes
pair_style      chimesFF fastmb
pair_coeff      * * published_params.Carbon-2.0.Large.2+3+4b.Tersoff.txt
```

Optional keywords (A/B testing): `nosimd`, `norips`, `orient tag`. `CHIMES_TIMING=1` prints per-phase
timings at exit.

## What it does

For a cluster with M pair distances, the ChIMES energy is a sparse tensor contraction

    E = sum_p C_p prod_m X_{p_m}(r_m),   X_k(r) = fcut(r) T_k(s(r))

and the forces need only the M derivatives dE/dr_m. Every change below reorders that computation;
none approximates it.

| Change | Code | What it removes |
|---|---|---|
| Per-pair Chebyshev cache | `chimesFF/src/chimesFF_mbfast.h` | Recomputing X and dX/dr (3 exponentials + recursion) for every cluster a pair belongs to |
| Sum factorization over a CSF tree | `chimesFF_mbfast.h` | O(M^2) products per term and the force/stress scatter inside the term loop |
| SIMD batches across clusters | `chimesFF_mbfast.h`, `etc/lmp/src/pair_chimes.cpp` | Scalar tree sweeps: 8 clusters of one type share each vector instruction |
| Vietoris–Rips cluster enumeration | `pair_chimes.cpp` | Nested neighbor loops with repeated square roots; uses adjacency bitsets |
| Flat pair registry | `pair_chimes.cpp` | `std::unordered_map` overhead when assigning pair ids |
| Hash-ordered cluster ownership | `pair_chimes.cpp` | Load imbalance from tag-ordered ownership (tags are spatially correlated) |
| `get_atom_pair_index` fix | `chimesFF/src/chimesFF.cpp` | Wrong 2-body cutoffs for multi-element models whose types are not in alphabetical order |

Cluster lists fall back to the default loops on every rank when the box is smaller than twice the
padded many-body cutoff (periodic self-images); the decision is collective so all ranks use the same
ownership rule.

Build with OpenMP SIMD and the target vector ISA for the SIMD gain
(`etc/lmp/etc/Makefile.mpi_chimes.UT-TACC` uses `-qopenmp-simd -xCORE-AVX512 -qopt-zmm-usage=high`).
The code is C++11.

## Verification

* Single points, default vs fastmb, 1 and 16 MPI ranks and across rank counts, for Carbon-2.0 Small
  (liquid, diamond), Carbon-2.0 Large, nitrogen reshock, CO 2400 K (two elements, type-specific
  cutoffs), HN3 (two elements, non-alphabetical type order), liquid CO (2+3-body) and Si PBE: energy
  within 3e-13 relative, virial within 2e-13, forces within 5e-12 kcal/mol/A (1e-13 of the RMS force).
* Each option (`nosimd`, `norips`, `orient tag`, all three), `hybrid/overlay` with `momb`, and small
  periodic boxes on 1-56 ranks.
* NVE, Carbon-2.0 Small liquid, 1,600 atoms, 10 ps: drift -0.0062 (default) vs 0.0029 / 0.0058
  (fastmb, skin 1.0 / 0.3 A) meV/atom/ps, fluctuation 0.034-0.039 meV/atom; wall time 1,518 s vs 98 s.
* `etc/lmp/tests/example-fastmb/check_fastmb.sh` repeats the single-point checks for a single-element
  model, a two-element model and a small box with every option.

## Performance

One TACC Stampede3 SPR node, 112 MPI ranks, time per MD step:

| System | default | fastmb, skin 1.0 A | fastmb, skin 0.3 A | speedup |
|---|---|---|---|---|
| Carbon-2.0 Large, liquid 6000 K, 12,800 atoms | 20,581 ms | 144 ms | 111 ms | 186x |
| Nitrogen reshock, 5000 K, 13,824 atoms | 1,884 ms | 24.4 ms | 21.6 ms | 87x |
| Carbon-2.0 Small, liquid 6000 K, 102,400 atoms | 773 ms | 60.5 ms | 55.6 ms | 14x |

## Evaluated and not included

Kept out because they are not exact: float32 SIMD lanes (about 5e-6 relative force error), r-RESPA
multiple time stepping, 2B/3B tabulation, CP/SVD decompositions of the coefficient tensor. Kept out
because they were slower than the SIMD tree sweep: 3-body contraction as BLAS-3 (dgemm), reuse of
shared triangular faces between quadruplets. Found not to apply: butterfly / interpolative
decompositions and Tucker compression (the coefficient tensors have full multilinear rank),
symmetric-polynomial (invariant-theory) evaluation (ill-conditioned at practical orders), and
addition-theorem separation of the 3-body jk factor (requires very high angular order for the
Morse-mapped basis).
