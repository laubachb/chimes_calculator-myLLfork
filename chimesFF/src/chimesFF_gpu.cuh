/*
    ChIMES Calculator — GPU acceleration layer
    Branch: laubachb/gpu-acceleration

    This header declares the C++ interface to chimesFF_gpu.cu.
    It is included only when USE_CUDA is defined (set by CMake WITH_CUDA=ON).
    No CUDA headers are pulled in here; all CUDA-specific code lives in
    chimesFF_gpu.cu so that ordinary .cpp files can include this safely.
*/

#pragma once
#ifdef USE_CUDA

// ---------------------------------------------------------------------------
// Parameter upload / teardown
//   Call chimesFF_gpu_upload_params_flat() once after read_parameters() and
//   after build_pair_int_trip_map() / build_pair_int_quad_map().
//   All arrays must be valid host pointers; the function copies them to the
//   GPU and is safe to call again (old device memory is freed first).
// ---------------------------------------------------------------------------
void chimesFF_gpu_upload_params_flat(
    // --- 2-body ---
    int           n_pairs,   int order_2b,
    const int   * ncoeffs_2b,   const int   * offset_2b,   int total_2b_coeffs,
    const double* params_2b,    const int   * pows_2b,
    const double* cutoff_2b,                                // [n_pairs*2]: inner,outer per pair
    const double* morse,                                    // [n_pairs]
    const int   * pair_map,     int pair_map_size,          // atom_int_pair_map [natmtyps^2]
    // --- 3-body ---
    int           n_trips,   int order_3b,
    const int   * ncoeffs_3b,   const int   * offset_3b,   int total_3b_coeffs,
    const double* params_3b,    const int   * pows_3b,      // [(total_3b)*3]
    const double* cutoff_3b,                                // [n_trips*6]: [t][in_out*3+pair]
    const int   * trip_map,     int trip_map_size,          // atom_int_trip_map [natmtyps^3]
    const int   * pit,          int pit_size,               // pair_int_trip flat [natmtyps^3*3]
    // --- 4-body ---
    int           n_quads,   int order_4b,
    const int   * ncoeffs_4b,   const int   * offset_4b,   int total_4b_coeffs,
    const double* params_4b,    const int   * pows_4b,      // [(total_4b)*6]
    const double* cutoff_4b,                                // [n_quads*12]: [q][in_out*6+pair]
    const int   * quad_map,     int quad_map_size,          // atom_int_quad_map [natmtyps^4]
    const int   * piq,          int piq_size,               // pair_int_quad flat [natmtyps^4*6]
    // --- shared ---
    int natmtyps, int fcut_type, double fcut_var,           // fcut_type: 0=CUBIC 1=TERSOFF
    const double* penalty                                   // [2]: A_pen, d_pen
);

// Free all device memory allocated by chimesFF_gpu_upload_params_flat().
void chimesFF_gpu_free_params();

// ---------------------------------------------------------------------------
// Batch force / energy evaluation kernels.
//   All array pointers passed to these functions are DEVICE pointers.
//   d_forces[natoms*3] and d_energy[1] are atomically accumulated into and
//   must be zeroed by the caller before the first kernel for each timestep.
//   These functions launch kernels and return immediately (no synchronisation).
//   Call cudaDeviceSynchronize() after all body-order kernels before copying
//   d_forces / d_energy back to the host.
// ---------------------------------------------------------------------------

void chimesFF_gpu_compute_2B(
    int npairs,
    double* d_dx,  double* d_dr,  int* d_typ,   // inputs [n], [n*3], [n*2]
    int*    d_ai,  int*    d_aj,                 // atom indices [n] each
    int natoms,
    double* d_forces,   double* d_energy);       // output [natoms*3], [1]

void chimesFF_gpu_compute_3B(
    int ntriplets,
    double* d_dx,  double* d_dr,  int* d_typ,   // [n*3], [n*9], [n*3]
    int*    d_ai,  int*    d_aj,  int* d_ak,
    int natoms,
    double* d_forces,   double* d_energy);

void chimesFF_gpu_compute_4B(
    int nquads,
    double* d_dx,  double* d_dr,  int* d_typ,   // [n*6], [n*18], [n*4]
    int*    d_ai,  int*    d_aj,  int* d_ak,  int* d_al,
    int natoms,
    double* d_forces,   double* d_energy);

#endif // USE_CUDA
