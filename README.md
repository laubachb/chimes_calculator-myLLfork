<p style="text-align:center;">
    <img src="./doc/ChIMES_Github_logo-2.png" alt="" width="250"/>
</p>
<hr>

The Chebyshev Interaction Model for Efficient Simulation (ChIMES) is a machine-learned interatomic potential that can target chemical reactivity. ChIMES models are able to approach quantum-accuracy through a systematically improvable explicitly many-bodied basis comprised of linear combinations of Chebyshev polynomials. ChIMES has successfully been applied to a number of condensed phase systems, including water under ambient and extreme conditions, molten carbon, and liquid carbon monoxide under planetary interior conditions. ChIMES can also be used as a many-body repulsive energy for the density functional based tight binding (DFTB) method.

The ChIMES calculator comprises a flexible tool set for evaluating ChIMES interactions (e.g. in simulations, single point calculations, etc). Users have the option of directly embedding the ChIMES calculator within their codes (e.g. see ‘’The ChIMES Calculator,’’ in the documentation for advanced users), or evaluating interactions through the beginner-friendly serial interface, each of which have Python, C++, C, and FORTRAN API’s.


<hr>

Documentation
----------------

[**Full documentation**](https://chimes-calculator.readthedocs.io/en/latest/) is available.

<hr>

Community
------------------------

Questions, discussion, and contributions (e.g. bug fixes, documentation, and extensions) are welcome. 

Additional Resources: [ChIMES Google group](https://groups.google.com/g/chimes_software).

<hr>

Contributing
------------------------

Contributions to the ChIMES calculator should be made through a pull request, with ``develop`` as the destination branch. A test suite log file should be attached to the PR. For additional contributing guidelines, see the [documentation](https://chimes-calculator.readthedocs.io/en/latest/contributing.html).

The ChIMES calculator `develop` branch has the latest contributions. Pull requests should target `develop`, and users who want the latest package versions,
features, etc. can use `develop`.

<hr>

Releases
--------

For most users, we recommend using the ChIMES calculator [stable releases](https://github.com/rk-lindsey/chimes_calculator/releases).

ChIMES releases are indicated via semantic versioning tags of the form vA.B.C. The letters A, B, and C indicate major changes impacting the API (i.e., precluding backwards compatibility), minor backwards compatible changes, and backwards compatible bug fixes, respectively

<hr>

Authors
----------------

The ChIMES calculator was developed by Rebecca K. Lindsey, Nir Goldman, and Laurence E Fried.

Contributors can be found [here](https://github.com/rk-lindsey/chimes_calculator/graphs/contributors).

<hr>

Citing
----------------

See [the documentation](https://chimes-calculator.readthedocs.io/en/latest/citing.html) for guidance on referencing ChIMES and the ChIMES calculator in a publication.

<hr>

GPU Acceleration (CUDA)
-----------------------

This fork adds optional CUDA GPU acceleration for ChIMES force evaluation.
All changes are guarded by `#ifdef USE_CUDA`; CPU-only builds are unaffected.

### How it works

The three hot loops in `chimesFF::compute_2B/3B/4B` are replaced with a
single-pass GPU path when the LAMMPS pair style is active:

1. A single pass over the LAMMPS neighbour list builds compact batched arrays
   (distances, displacement vectors, atom-type indices).
2. One kernel launch per body order processes all clusters in parallel — one
   GPU thread per cluster. Chebyshev polynomials and cutoff functions are
   evaluated on-device; forces are accumulated with `atomicAdd`.
3. Resulting forces are copied back to host and scattered into LAMMPS `f[]`.
   Energy is added to `eng_vdwl`; the global virial is handled by the
   standard LAMMPS `virial_fdotr_compute()` from the accumulated forces.

ChIMES parameters (coefficients, cutoffs, Morse lambdas, type maps) are
uploaded to GPU constant memory **once** after `pair_coeff` and reused for
every timestep.

### Requirements

- NVIDIA GPU with Compute Capability ≥ 6.0 (Pascal or newer — see table below)
- CUDA Toolkit ≥ 11.0
- CMake ≥ 3.18

#### Supported GPU architectures

| GPU | SM arch | `CUDA_ARCH` value |
|-----|---------|-------------------|
| H100 | sm_90 | `90` (default) |
| A100 | sm_80 | `80` |
| A30 / A40 / RTX 3090 | sm_86 | `86` |
| L40S / RTX 4090 | sm_89 | `89` |
| V100 | sm_70 | `70` |
| T4 | sm_75 | `75` |

`atomicAdd` for `double` requires SM ≥ 6.0 (Pascal, 2016+); older GPUs are not supported.
A comma-separated list (e.g. `CUDA_ARCH=80,90`) produces a fat binary that runs on multiple generations.

### Building the ChIMES library with GPU support

```bash
# On Stampede3 (H100 nodes):
module load gcc/13.2.0 cuda/12.8

cmake -B build_gpu \
      -DWITH_CUDA=ON \
      -DCUDA_ARCH=90 \        # 80 for A100, 90 for H100
      -DCMAKE_BUILD_TYPE=Release \
      .
cmake --build build_gpu -j8
```

The resulting `build_gpu/libchimescalc.so` links against the CUDA runtime and
contains device code for all three body orders.

For A100 nodes (SM 80):

```bash
cmake -B build_gpu -DWITH_CUDA=ON -DCUDA_ARCH=80 .
cmake --build build_gpu -j8
```

### Using GPU acceleration with LAMMPS

Build LAMMPS against the GPU-enabled `libchimescalc.so` following the
standard instructions in `etc/lmp/`. No changes to the LAMMPS input script
are required. The GPU path activates automatically when:

- The build was compiled with `USE_CUDA`, and
- No per-atom energy/virial output is requested (i.e. `eflag_atom = vflag_atom = 0`).

If per-atom thermodynamic quantities or `TABULATION`/`FINGERPRINT` mode are
requested, the code falls back silently to the original CPU path.

### GPU vs CPU validation test

A standalone test executable validates that GPU and CPU forces agree to
within 1 × 10⁻⁹ eV/Å for random 2-body and 3-body cluster geometries:

```bash
# Build the test (requires WITH_CUDA=ON build):
cmake --build build_gpu --target chimescalc-gpu-validate -j4

# Run against the included liquid-carbon force field:
./build_gpu/chimescalc-gpu-validate \
    serial_interface/tests/force_fields/published_params.liqC.2+3b.cubic.txt
```

Expected output (timings will vary by GPU):

```
Loaded: 1 atom types, 2B order 12, 3B enabled
2B cutoff: [1.000, 3.150]
...
=== 2B Results (800 pairs, 120 atoms) ===
  CPU energy:   ...
  GPU energy:   ...
  Energy abs error:   <1e-11   rel error: <1e-11
  Force  max abs err: <1e-11   max rel:   <1e-13
  2B: PASS

=== 3B Results (400 triplets, 120 atoms) ===
  ...
  3B: PASS

=== Overall: ALL PASS ===
```

> **Note on non-determinism:** GPU floating-point reductions (`atomicAdd`)
> are non-associative, so GPU energies and forces may differ from CPU results
> by ≈ 10⁻¹² – 10⁻¹⁰ eV (machine-epsilon level).  This is the standard
> trade-off for GPU parallelism and does not affect physical observables.

### New files

| File | Purpose |
|------|---------|
| `chimesFF/src/chimesFF_gpu.cuh` | C++ interface to GPU kernels (no CUDA headers required by callers) |
| `chimesFF/src/chimesFF_gpu.cu`  | CUDA kernels: `k2B`, `k3B`, `k4B`; parameter upload/teardown |
| `chimesFF/tests/gpu_validate/test_gpu_cpu.cu` | GPU vs CPU validation test |

### Modified files

| File | Change |
|------|--------|
| `config.cmake` | `WITH_CUDA` option and `CUDA_ARCH` cache variable |
| `CMakeLists.txt` | CUDA language, sources, definitions, test target |
| `chimesFF/src/chimesFF.h` | `upload_params_to_device()` / `free_device_params()` |
| `chimesFF/src/chimesFF.cpp` | Implementation of the above |
| `etc/lmp/src/pair_chimes.h` | GPU batch-array members and helper declarations |
| `etc/lmp/src/pair_chimes.cpp` | `compute_gpu()`, buffer management, `compute()` dispatch |

<hr>

License
----------------

The ChIMES calculator is distributed under terms of [LGPL v3.0 License](https://github.com/rk-lindsey/chimes_calculator/blob/main/LICENSE).

LLNL-CODE-817533
