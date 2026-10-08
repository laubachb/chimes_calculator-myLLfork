# Element-aware (multielement) fingerprint example

Disordered Cu<sub>50</sub>Zr<sub>50</sub> (144 atoms, 72 Cu / 72 Zr, orthorhombic cell), one
configuration from the Cu-Zr ACE training data of Leimeroth, Rohrer & Albe
(Zenodo, doi:10.5281/zenodo.8137854).

`params.txt` is a low-order (2/2/2) placeholder model: only its cutoffs, Morse
lambdas and atom types matter for fingerprinting, so the energies and forces
LAMMPS prints are not meaningful. No fifth column is given in the `# TYPEIDX #`
block, so the atomic masses are used as the atomic descriptors.

```bash
# requires: cd etc/lmp && ./install.sh FINGERPRINT   (builds exe/lmp_mpi_chimes and chimesFF/src/FP/histogram_hybrid)
./run_job.sh <nprocs>
```

`run_job.sh` runs a single-point LAMMPS calculation with
`pair_style chimesFF fingerprint 1`, combines the per-rank cluster lists with
`post_process.sh`, and then calls `histogram_hybrid` at alpha = 1.0 (structure
only), 0.5, and 0.0 (composition only). The fingerprints are written as
`0-0.{2,3,4}b_clu-s.alpha_<alpha>.hist` (bin centre, normalized frequency;
100 bins over a dissimilarity of 0 to 1). `expected_output/` holds the
reference histograms; they are independent of the number of MPI ranks.
