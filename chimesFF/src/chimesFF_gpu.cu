/*
    ChIMES Calculator — CUDA GPU kernels
    Branch: laubachb/gpu-acceleration

    This file implements batched GPU evaluation of 2-, 3-, and 4-body
    ChIMES interactions.  One GPU thread handles one cluster (pair /
    triplet / quadruplet).  Forces are accumulated via atomicAdd back
    into a per-atom force array; energy via a single global accumulator.

    Virial (stress) is intentionally not accumulated here: pair_chimes
    relies on LAMMPS virial_fdotr_compute(), which correctly derives the
    virial from the forces written to f[] after GPU↔CPU transfer.

    Design notes
    ─────────────
    • Parameters (coefficients, cutoffs, Morse lambdas, type maps) are
      uploaded once per parameter set into GPU device memory and stored
      as a compact struct in __constant__ memory so every thread can read
      them via the L1 cache path.
    • Chebyshev polynomials are computed thread-locally on the stack; the
      compile-time constant MAX_POLY_ORDER must be ≥ the polynomial order
      in any parameter file used with this build (default 24 covers all
      published ChIMES models).
    • The 4-body force scalar uses prefix/suffix products to avoid
      division-by-zero when any Tn[p][pw] == 0.
    • atomicAdd(double*,double) requires SM ≥ 6.0 (Pascal); A100/H100
      are SM 8.0/9.0, so this is unconditionally correct on Stampede3.
*/

#include "chimesFF_gpu.cuh"

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// ============================================================
// Compile-time constants
// ============================================================

#define MAX_POLY_ORDER 24   // Max Chebyshev order supported per body term.
                             // Increase if your parameter file uses a higher order.

#define CUDA_CHECK(call) do {                                                   \
    cudaError_t _e = (call);                                                    \
    if (_e != cudaSuccess) {                                                    \
        fprintf(stderr, "CUDA error at %s:%d — %s\n",                          \
                __FILE__, __LINE__, cudaGetErrorString(_e));                    \
        exit(EXIT_FAILURE);                                                     \
    }                                                                           \
} while (0)

// ============================================================
// Device-side parameter struct (lives in constant memory)
// ============================================================

struct GPUParams {
    int     n_pairs, n_trips, n_quads;
    int     order_2b, order_3b, order_4b;
    int     natmtyps, fcut_type;   // fcut_type: 0=CUBIC  1=TERSOFF
    double  fcut_var;

    // 2B
    int    *d_ncoeffs_2b, *d_offset_2b, *d_pows_2b;
    double *d_params_2b,  *d_cutoff_2b, *d_morse;
    int    *d_pair_map;                             // [natmtyps^2]

    // 3B
    int    *d_ncoeffs_3b, *d_offset_3b, *d_pows_3b; // pows: [(total)*3]
    double *d_params_3b,  *d_cutoff_3b;              // cutoff: [n_trips*6]
    int    *d_trip_map;                              // [natmtyps^3]
    int    *d_pit;                                   // pair_int_trip [natmtyps^3 * 3]

    // 4B
    int    *d_ncoeffs_4b, *d_offset_4b, *d_pows_4b; // pows: [(total)*6]
    double *d_params_4b,  *d_cutoff_4b;              // cutoff: [n_quads*12]
    int    *d_quad_map;                              // [natmtyps^4]
    int    *d_piq;                                   // pair_int_quad [natmtyps^4 * 6]

    // Penalty
    double *d_penalty;                               // [2]: A_pen, d_pen
};

__constant__ GPUParams c_p;  // device-side read-only copy

// Host mirror for teardown tracking
static GPUParams  g_host_params;
static void      *g_allocs[64];
static int        g_n_allocs = 0;
static bool       g_initialized = false;

// ============================================================
// Host helpers: allocate + upload
// ============================================================

static void* gpu_alloc_and_track(size_t n)
{
    void *ptr;
    CUDA_CHECK(cudaMalloc(&ptr, n));
    g_allocs[g_n_allocs++] = ptr;
    return ptr;
}

template<typename T>
static T* upload(const T *host, int count)
{
    T *dev = static_cast<T*>(gpu_alloc_and_track(sizeof(T) * count));
    CUDA_CHECK(cudaMemcpy(dev, host, sizeof(T) * count, cudaMemcpyHostToDevice));
    return dev;
}

// Upload a single scalar value
template<typename T>
static T* upload_scalar(T val)
{
    T *dev = static_cast<T*>(gpu_alloc_and_track(sizeof(T)));
    CUDA_CHECK(cudaMemcpy(dev, &val, sizeof(T), cudaMemcpyHostToDevice));
    return dev;
}

// ============================================================
// Device utility: Chebyshev polynomials (Morse transform)
// ============================================================

__device__ static void cheby(double *Tn, double *Tnd,
                              double dx, double morse,
                              double rmin, double rmax, int order)
{
    double x_min  = exp(-rmin / morse);
    double x_max  = exp(-rmax / morse);
    double x_avg  = 0.5 * (x_max + x_min);
    double x_diff = -0.5 * (x_max - x_min);  // negative for Morse style

    if (dx < rmin) dx = rmin;

    double ex    = exp(-dx / morse);
    double x     = (ex - x_avg) / x_diff;
    double dx_dr = (-ex / morse) / x_diff;

    Tn[0] = 1.0;  Tn[1] = x;
    Tnd[0] = 1.0; Tnd[1] = 2.0 * x;

    for (int i = 2; i <= order; i++) {
        Tn[i]  = 2.0 * x * Tn[i-1]  - Tn[i-2];
        Tnd[i] = 2.0 * x * Tnd[i-1] - Tnd[i-2];
    }
    for (int i = order; i >= 1; i--)
        Tnd[i] = i * dx_dr * Tnd[i-1];
    Tnd[0] = 0.0;
}

// ============================================================
// Device utility: cutoff function
// ============================================================

__device__ static void fcut_fn(double dx, double rmax,
                                int fcut_type, double fcut_var,
                                double &fc, double &fcd)
{
    if (fcut_type == 0) {   // CUBIC
        double f0 = 1.0 - dx / rmax;
        fc  = f0 * f0 * f0;
        fcd = -3.0 * f0 * f0 / rmax;
    } else {                // TERSOFF
        const double PI = 3.14159265359;
        double thr = rmax * (1.0 - fcut_var);
        if (dx < thr) {
            fc = 1.0; fcd = 0.0;
        } else if (dx >= rmax) {
            fc = 0.0; fcd = 0.0;
        } else {
            double arg = (dx - thr) / (rmax - thr) * PI + PI * 0.5;
            double fd  = PI / (rmax - thr);
            fc  = 0.5 + 0.5 * sin(arg);
            fcd = 0.5 * cos(arg) * fd;
        }
    }
}

// ============================================================
// 2-body kernel
// ============================================================

__global__ void k2B(int np,
                    const double * __restrict__ dx,
                    const double * __restrict__ dr,
                    const int    * __restrict__ typ,
                    const int    * __restrict__ ai,
                    const int    * __restrict__ aj,
                    double       * __restrict__ forces,
                    double       * __restrict__ energy)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= np) return;

    double dxp = dx[pid];
    double drx = dr[pid*3+0], dry = dr[pid*3+1], drz = dr[pid*3+2];
    int    ti  = typ[pid*2+0], tj = typ[pid*2+1];
    int    a_i = ai[pid],       a_j = aj[pid];

    int N    = c_p.natmtyps;
    int pidx = c_p.d_pair_map[ti * N + tj];
    if (pidx < 0) return;

    double rmin = c_p.d_cutoff_2b[pidx*2+0];
    double rmax = c_p.d_cutoff_2b[pidx*2+1];
    if (dxp >= rmax) return;

    double Tn [MAX_POLY_ORDER+1];
    double Tnd[MAX_POLY_ORDER+1];
    cheby(Tn, Tnd, dxp, c_p.d_morse[pidx], rmin, rmax, c_p.order_2b);

    double fc, fcd;
    fcut_fn(dxp, rmax, c_p.fcut_type, c_p.fcut_var, fc, fcd);
    double dxi = (dxp > 0.0) ? 1.0 / dxp : 1e20;

    int    base = c_p.d_offset_2b[pidx];
    int    nc   = c_p.d_ncoeffs_2b[pidx];

    double eloc = 0.0, fx = 0.0, fy = 0.0, fz = 0.0;

    for (int c = 0; c < nc; c++) {
        double cv  = c_p.d_params_2b[base + c];
        int    pw  = c_p.d_pows_2b  [base + c] + 1;
        double val = fc * Tn[pw];
        double der = fc * Tnd[pw] + fcd * Tn[pw];
        double fs  = cv * der * dxi;
        eloc += cv * val;
        fx   += fs * drx;
        fy   += fs * dry;
        fz   += fs * drz;
    }

    // Penalty function (matches CPU chimesFF::get_penalty)
    double A_pen = c_p.d_penalty[0];
    double B_pen = c_p.d_penalty[1];
    double rpen  = 0.0;
    if (dxp - A_pen < rmin)
        rpen = rmin + A_pen - dxp;
    if (rpen > 0.0) {
        eloc    += rpen * rpen * rpen * B_pen;
        double fps = -3.0 * rpen * rpen * B_pen / dxp;
        fx += fps * drx;
        fy += fps * dry;
        fz += fps * drz;
    }

    atomicAdd(&forces[a_i*3+0],  fx);
    atomicAdd(&forces[a_i*3+1],  fy);
    atomicAdd(&forces[a_i*3+2],  fz);
    atomicAdd(&forces[a_j*3+0], -fx);
    atomicAdd(&forces[a_j*3+1], -fy);
    atomicAdd(&forces[a_j*3+2], -fz);
    atomicAdd(energy, eloc);
}

// ============================================================
// 3-body kernel
// ============================================================

__global__ void k3B(int nt,
                    const double * __restrict__ dx,   // [nt*3]: ij,ik,jk
                    const double * __restrict__ dr,   // [nt*9]
                    const int    * __restrict__ typ,  // [nt*3]
                    const int    * __restrict__ ai,
                    const int    * __restrict__ aj,
                    const int    * __restrict__ ak,
                    double       * __restrict__ forces,
                    double       * __restrict__ energy)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nt) return;

    int N  = c_p.natmtyps;
    int ti = typ[tid*3+0], tj = typ[tid*3+1], tk = typ[tid*3+2];
    int a_i = ai[tid], a_j = aj[tid], a_k = ak[tid];

    int type_idx = ti*N*N + tj*N + tk;
    int tripidx  = c_p.d_trip_map[type_idx];
    if (tripidx < 0) return;

    int mpi0 = c_p.d_pit[type_idx*3+0];
    int mpi1 = c_p.d_pit[type_idx*3+1];
    int mpi2 = c_p.d_pit[type_idx*3+2];

    double dxij = dx[tid*3+0], dxik = dx[tid*3+1], dxjk = dx[tid*3+2];

    // Cutoffs indexed via mapped pair index within the triplet parameter set
    double rmin0 = c_p.d_cutoff_3b[tripidx*6 + 0*3 + mpi0];
    double rmax0 = c_p.d_cutoff_3b[tripidx*6 + 1*3 + mpi0];
    double rmin1 = c_p.d_cutoff_3b[tripidx*6 + 0*3 + mpi1];
    double rmax1 = c_p.d_cutoff_3b[tripidx*6 + 1*3 + mpi1];
    double rmin2 = c_p.d_cutoff_3b[tripidx*6 + 0*3 + mpi2];
    double rmax2 = c_p.d_cutoff_3b[tripidx*6 + 1*3 + mpi2];

    if (dxij >= rmax0 || dxik >= rmax1 || dxjk >= rmax2) return;

    // Morse via 2B pair-type map (same lambda array as 2B)
    int pt_ij = c_p.d_pair_map[ti*N + tj];
    int pt_ik = c_p.d_pair_map[ti*N + tk];
    int pt_jk = c_p.d_pair_map[tj*N + tk];

    int order = c_p.order_3b;
    double Tn_ij[MAX_POLY_ORDER+1], Tnd_ij[MAX_POLY_ORDER+1];
    double Tn_ik[MAX_POLY_ORDER+1], Tnd_ik[MAX_POLY_ORDER+1];
    double Tn_jk[MAX_POLY_ORDER+1], Tnd_jk[MAX_POLY_ORDER+1];

    cheby(Tn_ij, Tnd_ij, dxij, c_p.d_morse[pt_ij], rmin0, rmax0, order);
    cheby(Tn_ik, Tnd_ik, dxik, c_p.d_morse[pt_ik], rmin1, rmax1, order);
    cheby(Tn_jk, Tnd_jk, dxjk, c_p.d_morse[pt_jk], rmin2, rmax2, order);

    double fc0, fcd0, fc1, fcd1, fc2, fcd2;
    int    ft = c_p.fcut_type;
    double fv = c_p.fcut_var;
    fcut_fn(dxij, rmax0, ft, fv, fc0, fcd0);
    fcut_fn(dxik, rmax1, ft, fv, fc1, fcd1);
    fcut_fn(dxjk, rmax2, ft, fv, fc2, fcd2);

    double fca   = fc0 * fc1 * fc2;
    double fc2_0 = fc1 * fc2 / dxij;  // product of 2 fcuts / dx (same as CPU fcut_2[])
    double fc2_1 = fc0 * fc2 / dxik;
    double fc2_2 = fc0 * fc1 / dxjk;

    double drij0 = dr[tid*9+0], drij1 = dr[tid*9+1], drij2 = dr[tid*9+2];
    double drik0 = dr[tid*9+3], drik1 = dr[tid*9+4], drik2 = dr[tid*9+5];
    double drjk0 = dr[tid*9+6], drjk1 = dr[tid*9+7], drjk2 = dr[tid*9+8];

    int    base = c_p.d_offset_3b[tripidx];
    int    nc   = c_p.d_ncoeffs_3b[tripidx];

    double eloc = 0.0;
    double fi0=0,fi1=0,fi2=0, fj0=0,fj1=0,fj2=0, fk0=0,fk1=0,fk2=0;

    for (int c = 0; c < nc; c++) {
        double cv  = c_p.d_params_3b[base + c];
        int    pw0 = c_p.d_pows_3b[(base+c)*3 + mpi0];
        int    pw1 = c_p.d_pows_3b[(base+c)*3 + mpi1];
        int    pw2 = c_p.d_pows_3b[(base+c)*3 + mpi2];

        eloc += cv * fca * Tn_ij[pw0] * Tn_ik[pw1] * Tn_jk[pw2];

        double d0 = fc0 * Tnd_ij[pw0] + fcd0 * Tn_ij[pw0];
        double d1 = fc1 * Tnd_ik[pw1] + fcd1 * Tn_ik[pw1];
        double d2 = fc2 * Tnd_jk[pw2] + fcd2 * Tn_jk[pw2];

        double fs0 = cv * d0 * fc2_0 * Tn_ik[pw1] * Tn_jk[pw2];
        double fs1 = cv * d1 * fc2_1 * Tn_ij[pw0] * Tn_jk[pw2];
        double fs2 = cv * d2 * fc2_2 * Tn_ij[pw0] * Tn_ik[pw1];

        // ij pair: force on i(+) and j(-)
        fi0 += fs0*drij0; fi1 += fs0*drij1; fi2 += fs0*drij2;
        fj0 -= fs0*drij0; fj1 -= fs0*drij1; fj2 -= fs0*drij2;
        // ik pair: force on i(+) and k(-)
        fi0 += fs1*drik0; fi1 += fs1*drik1; fi2 += fs1*drik2;
        fk0 -= fs1*drik0; fk1 -= fs1*drik1; fk2 -= fs1*drik2;
        // jk pair: force on j(+) and k(-)
        fj0 += fs2*drjk0; fj1 += fs2*drjk1; fj2 += fs2*drjk2;
        fk0 -= fs2*drjk0; fk1 -= fs2*drjk1; fk2 -= fs2*drjk2;
    }

    atomicAdd(&forces[a_i*3+0], fi0); atomicAdd(&forces[a_i*3+1], fi1); atomicAdd(&forces[a_i*3+2], fi2);
    atomicAdd(&forces[a_j*3+0], fj0); atomicAdd(&forces[a_j*3+1], fj1); atomicAdd(&forces[a_j*3+2], fj2);
    atomicAdd(&forces[a_k*3+0], fk0); atomicAdd(&forces[a_k*3+1], fk1); atomicAdd(&forces[a_k*3+2], fk2);
    atomicAdd(energy, eloc);
}

// ============================================================
// 4-body kernel
// ============================================================

// Atom assignment per pair (ij=0,ik=1,il=2,jk=3,jl=4,kl=5)
// pair p → atoms[pair_atom_a[p]] and atoms[pair_atom_b[p]]
__device__ static const int c_pa_a[6] = {0, 0, 0, 1, 1, 2};
__device__ static const int c_pa_b[6] = {1, 2, 3, 2, 3, 3};

__global__ void k4B(int nq,
                    const double * __restrict__ dx,   // [nq*6]: ij,ik,il,jk,jl,kl
                    const double * __restrict__ dr,   // [nq*18]
                    const int    * __restrict__ typ,  // [nq*4]
                    const int    * __restrict__ ai,
                    const int    * __restrict__ aj,
                    const int    * __restrict__ ak,
                    const int    * __restrict__ al,
                    double       * __restrict__ forces,
                    double       * __restrict__ energy)
{
    int qid = blockIdx.x * blockDim.x + threadIdx.x;
    if (qid >= nq) return;

    int N  = c_p.natmtyps;
    int ti = typ[qid*4+0], tj = typ[qid*4+1];
    int tk = typ[qid*4+2], tl = typ[qid*4+3];
    int atoms[4] = { ai[qid], aj[qid], ak[qid], al[qid] };

    int idx     = ti*N*N*N + tj*N*N + tk*N + tl;
    int quadidx = c_p.d_quad_map[idx];
    if (quadidx < 0) return;

    int mpi[6];
    for (int p = 0; p < 6; p++) mpi[p] = c_p.d_piq[idx*6 + p];

    double DX[6];
    for (int p = 0; p < 6; p++) DX[p] = dx[qid*6 + p];

    // Check all six cutoffs
    double rmin[6], rmax[6];
    for (int p = 0; p < 6; p++) {
        rmin[p] = c_p.d_cutoff_4b[quadidx*12 + 0*6 + mpi[p]];
        rmax[p] = c_p.d_cutoff_4b[quadidx*12 + 1*6 + mpi[p]];
        if (DX[p] >= rmax[p]) return;
    }

    // 2B pair types for Morse lookup: ij,ik,il,jk,jl,kl
    int pt[6];
    pt[0] = c_p.d_pair_map[ti*N + tj];
    pt[1] = c_p.d_pair_map[ti*N + tk];
    pt[2] = c_p.d_pair_map[ti*N + tl];
    pt[3] = c_p.d_pair_map[tj*N + tk];
    pt[4] = c_p.d_pair_map[tj*N + tl];
    pt[5] = c_p.d_pair_map[tk*N + tl];

    int order = c_p.order_4b;
    double Tn [6][MAX_POLY_ORDER+1];
    double Tnd[6][MAX_POLY_ORDER+1];
    for (int p = 0; p < 6; p++)
        cheby(Tn[p], Tnd[p], DX[p], c_p.d_morse[pt[p]], rmin[p], rmax[p], order);

    double fc[6], fcd[6];
    int    ft = c_p.fcut_type;
    double fv = c_p.fcut_var;
    for (int p = 0; p < 6; p++)
        fcut_fn(DX[p], rmax[p], ft, fv, fc[p], fcd[p]);

    double fca = fc[0]*fc[1]*fc[2]*fc[3]*fc[4]*fc[5];

    // fcut_5[p] = product of all 5 fcuts except p, divided by DX[p]
    double fcut_5[6];
    for (int p = 0; p < 6; p++) {
        double prod = 1.0;
        for (int q = 0; q < 6; q++) if (q != p) prod *= fc[q];
        fcut_5[p] = prod / DX[p];
    }

    // Displacement vectors
    double DR[6][3];
    for (int p = 0; p < 6; p++)
        for (int d = 0; d < 3; d++)
            DR[p][d] = dr[qid*18 + p*3 + d];

    int    base = c_p.d_offset_4b[quadidx];
    int    nc   = c_p.d_ncoeffs_4b[quadidx];

    double eloc   = 0.0;
    double F[4][3] = {{0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0}};

    for (int c = 0; c < nc; c++) {
        double cv  = c_p.d_params_4b[base + c];
        int    pw[6];
        for (int p = 0; p < 6; p++)
            pw[p] = c_p.d_pows_4b[(base+c)*6 + mpi[p]];

        // Prefix/suffix products of Tn — avoids division-by-zero
        double pre[7], suf[7];
        pre[0] = 1.0;
        for (int p = 0; p < 6; p++) pre[p+1] = pre[p] * Tn[p][pw[p]];
        suf[6] = 1.0;
        for (int p = 5; p >= 0; p--) suf[p] = suf[p+1] * Tn[p][pw[p]];

        double TnProd = pre[6];           // full product
        eloc += cv * fca * TnProd;

        for (int p = 0; p < 6; p++) {
            double dp    = fc[p] * Tnd[p][pw[p]] + fcd[p] * Tn[p][pw[p]];
            double TnExP = pre[p] * suf[p+1];   // product excluding pair p
            double fsp   = cv * dp * fcut_5[p] * TnExP;

            int a0 = c_pa_a[p], a1 = c_pa_b[p];
            for (int d = 0; d < 3; d++) {
                F[a0][d] += fsp * DR[p][d];
                F[a1][d] -= fsp * DR[p][d];
            }
        }
    }

    for (int a = 0; a < 4; a++)
        for (int d = 0; d < 3; d++)
            atomicAdd(&forces[atoms[a]*3 + d], F[a][d]);
    atomicAdd(energy, eloc);
}

// ============================================================
// Public C++ interface implementations
// ============================================================

void chimesFF_gpu_upload_params_flat(
    int n_pairs, int order_2b,
    const int *ncoeffs_2b, const int *offset_2b, int total_2b,
    const double *params_2b, const int *pows_2b,
    const double *cutoff_2b, const double *morse, const int *pair_map, int pair_map_size,
    int n_trips, int order_3b,
    const int *ncoeffs_3b, const int *offset_3b, int total_3b,
    const double *params_3b, const int *pows_3b,
    const double *cutoff_3b,
    const int *trip_map, int trip_map_size,
    const int *pit, int pit_size,
    int n_quads, int order_4b,
    const int *ncoeffs_4b, const int *offset_4b, int total_4b,
    const double *params_4b, const int *pows_4b,
    const double *cutoff_4b,
    const int *quad_map, int quad_map_size,
    const int *piq, int piq_size,
    int natmtyps, int fcut_type, double fcut_var,
    const double *penalty)
{
    // Free any previously uploaded parameters
    if (g_initialized) chimesFF_gpu_free_params();

    // k3B allocates ~1.5 KB and k4B ~3 KB of thread-local (stack) storage for
    // Chebyshev arrays.  The CUDA default per-thread stack is only 1 KB, which
    // causes a silent stack overflow and an illegal-memory-access in those kernels.
    // Set 8 KB per thread so all three body-order kernels have headroom.
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 8192));

    GPUParams hp = {};  // host-side params struct to fill with device pointers
    hp.n_pairs  = n_pairs;  hp.n_trips  = n_trips;  hp.n_quads  = n_quads;
    hp.order_2b = order_2b; hp.order_3b = order_3b; hp.order_4b = order_4b;
    hp.natmtyps = natmtyps; hp.fcut_type = fcut_type; hp.fcut_var = fcut_var;

    // --- 2B ---
    hp.d_ncoeffs_2b = upload(ncoeffs_2b, n_pairs);
    hp.d_offset_2b  = upload(offset_2b,  n_pairs + 1);
    if (total_2b > 0) {
        hp.d_params_2b = upload(params_2b, total_2b);
        hp.d_pows_2b   = upload(pows_2b,   total_2b);
    }
    hp.d_cutoff_2b = upload(cutoff_2b, n_pairs * 2);
    hp.d_morse     = upload(morse,     n_pairs);
    hp.d_pair_map  = upload(pair_map,  pair_map_size);

    // --- 3B ---
    if (n_trips > 0) {
        hp.d_ncoeffs_3b = upload(ncoeffs_3b, n_trips);
        hp.d_offset_3b  = upload(offset_3b,  n_trips + 1);
        if (total_3b > 0) {
            hp.d_params_3b = upload(params_3b, total_3b);
            hp.d_pows_3b   = upload(pows_3b,   total_3b * 3);
        }
        hp.d_cutoff_3b = upload(cutoff_3b, n_trips * 6);
        hp.d_trip_map  = upload(trip_map,  trip_map_size);
        hp.d_pit       = upload(pit,       pit_size * 3);
    }

    // --- 4B ---
    if (n_quads > 0) {
        hp.d_ncoeffs_4b = upload(ncoeffs_4b, n_quads);
        hp.d_offset_4b  = upload(offset_4b,  n_quads + 1);
        if (total_4b > 0) {
            hp.d_params_4b = upload(params_4b, total_4b);
            hp.d_pows_4b   = upload(pows_4b,   total_4b * 6);
        }
        hp.d_cutoff_4b = upload(cutoff_4b,  n_quads * 12);
        if (quad_map) hp.d_quad_map = upload(quad_map, quad_map_size);
        hp.d_piq       = upload(piq, piq_size * 6);
    }

    hp.d_penalty = upload(penalty, 2);

    // Copy struct of device pointers to constant memory
    CUDA_CHECK(cudaMemcpyToSymbol(c_p, &hp, sizeof(GPUParams)));
    g_host_params = hp;
    g_initialized = true;
}

void chimesFF_gpu_free_params()
{
    if (!g_initialized) return;
    for (int i = 0; i < g_n_allocs; i++) {
        cudaFree(g_allocs[i]);
        g_allocs[i] = nullptr;
    }
    g_n_allocs    = 0;
    g_initialized = false;
    memset(&g_host_params, 0, sizeof(GPUParams));
}

// ============================================================
// Kernel launchers
// ============================================================

// Per-kernel synchronize + error check used for debugging
#define CUDA_KERNEL_CHECK(label) do {                                           \
    cudaError_t _e = cudaDeviceSynchronize();                                   \
    if (_e != cudaSuccess) {                                                    \
        fprintf(stderr, "chimesFF GPU kernel [%s] failed: %s\n",               \
                label, cudaGetErrorString(_e));                                 \
        exit(EXIT_FAILURE);                                                     \
    }                                                                           \
} while (0)

void chimesFF_gpu_compute_2B(
    int npairs,
    double *d_dx, double *d_dr, int *d_typ,
    int *d_ai, int *d_aj,
    int natoms, double *d_forces, double *d_energy)
{
    if (npairs <= 0) return;
    int tpb    = 256;
    int blocks = (npairs + tpb - 1) / tpb;
    k2B<<<blocks, tpb>>>(npairs, d_dx, d_dr, d_typ, d_ai, d_aj, d_forces, d_energy);
    CUDA_KERNEL_CHECK("k2B");
}

void chimesFF_gpu_compute_3B(
    int ntriplets,
    double *d_dx, double *d_dr, int *d_typ,
    int *d_ai, int *d_aj, int *d_ak,
    int natoms, double *d_forces, double *d_energy)
{
    if (ntriplets <= 0) return;
    int tpb    = 128;
    int blocks = (ntriplets + tpb - 1) / tpb;
    k3B<<<blocks, tpb>>>(ntriplets, d_dx, d_dr, d_typ, d_ai, d_aj, d_ak, d_forces, d_energy);
    CUDA_KERNEL_CHECK("k3B");
}

void chimesFF_gpu_compute_4B(
    int nquads,
    double *d_dx, double *d_dr, int *d_typ,
    int *d_ai, int *d_aj, int *d_ak, int *d_al,
    int natoms, double *d_forces, double *d_energy)
{
    if (nquads <= 0) return;
    int tpb    = 64;
    int blocks = (nquads + tpb - 1) / tpb;
    k4B<<<blocks, tpb>>>(nquads, d_dx, d_dr, d_typ, d_ai, d_aj, d_ak, d_al, d_forces, d_energy);
    CUDA_KERNEL_CHECK("k4B");
}

#endif // USE_CUDA
