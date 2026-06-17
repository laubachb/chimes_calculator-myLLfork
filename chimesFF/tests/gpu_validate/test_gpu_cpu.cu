/*
    ChIMES Calculator — GPU vs CPU validation test
    Branch: laubachb/gpu-acceleration

    Reads a ChIMES parameter file, generates a set of random in-range pair and
    triplet geometries, evaluates forces on the CPU (one cluster at a time via
    the standard chimesFF interface) and on the GPU (batched via the
    chimesFF_gpu interface), then verifies that forces and energies agree to
    within a tight tolerance.

    Build (from the repository root after configuring with WITH_CUDA=ON):

        cmake --build build_gpu --target chimescalc-gpu-validate -j4

    Or manually:

        nvcc -std=c++14 -O2 -arch=sm_90 \
             -I chimesFF/src \
             chimesFF/tests/gpu_validate/test_gpu_cpu.cpp \
             chimesFF/src/chimesFF.cpp \
             chimesFF/src/chimesFF_gpu.cu \
             -o test_gpu_cpu

    Run:

        ./test_gpu_cpu <chimes_parameter_file>

    Example:

        ./test_gpu_cpu serial_interface/tests/force_fields/published_params.liqC.2+3b.cubic.txt
*/

#ifdef USE_CUDA

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <random>

#include <cuda_runtime.h>

#include "chimesFF.h"
#include "chimesFF_gpu.cuh"

// ============================================================
// Helpers
// ============================================================

static void cuda_check(cudaError_t e, const char *file, int line)
{
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error %s:%d — %s\n", file, line,
                cudaGetErrorString(e));
        exit(EXIT_FAILURE);
    }
}
#define CK(e) cuda_check(e, __FILE__, __LINE__)

// Generate a random double in [lo, hi]
static double randu(std::mt19937 &rng, double lo, double hi)
{
    return lo + (hi - lo) * std::uniform_real_distribution<double>(0.0,1.0)(rng);
}

// ============================================================
// Main
// ============================================================

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <chimes_parameter_file>\n", argv[0]);
        return 1;
    }
    const char *param_file = argv[1];

    // ----------------------------------------------------------
    // 1. Load ChIMES parameters
    // ----------------------------------------------------------

    chimesFF calc;
    calc.init(0);
    calc.read_parameters(param_file);
    calc.build_pair_int_trip_map();
    calc.build_pair_int_quad_map();

    // Upload to GPU
    calc.upload_params_to_device();

    int natmtyps   = calc.natmtyps;
    int order_2b   = calc.poly_orders[0];
    int has_3b     = (calc.poly_orders.size() > 1 && calc.poly_orders[1] > 0) ? 1 : 0;

    printf("Loaded: %d atom types, 2B order %d, 3B %s\n",
           natmtyps, order_2b, has_3b ? "enabled" : "disabled");

    // Get 2B cutoffs for the first pair type (0-0)
    std::vector<std::vector<double>> cuts2b;
    calc.get_cutoff_2B(cuts2b);
    double rmin2b = cuts2b[0][0];
    double rmax2b = cuts2b[0][1];

    printf("2B cutoff: [%.3f, %.3f]\n", rmin2b, rmax2b);

    // ----------------------------------------------------------
    // 2. Generate random 2B pairs
    // ----------------------------------------------------------

    const int N_ATOMS  = 120;          // virtual atoms 0..N_ATOMS-1
    const int N_PAIRS  = 800;          // random 2B interactions
    const int N_TRIPS  = has_3b ? 400 : 0;
    const double TOL   = 1e-9;         // force comparison tolerance

    std::mt19937 rng(42);

    // Batch input arrays (host)
    std::vector<double> h_dx_2b(N_PAIRS), h_dr_2b(N_PAIRS*3);
    std::vector<int>    h_typ_2b(N_PAIRS*2), h_ai_2b(N_PAIRS), h_aj_2b(N_PAIRS);

    for (int p = 0; p < N_PAIRS; p++) {
        double dx  = randu(rng, rmin2b * 1.01, rmax2b * 0.99);
        // Unit vector in a random direction
        double ux  = randu(rng,-1,1), uy = randu(rng,-1,1), uz = randu(rng,-1,1);
        double inv = dx / sqrt(ux*ux + uy*uy + uz*uz);
        h_dx_2b[p]         = dx;
        h_dr_2b[p*3+0]     = ux * inv;
        h_dr_2b[p*3+1]     = uy * inv;
        h_dr_2b[p*3+2]     = uz * inv;
        h_typ_2b[p*2+0]    = 0;  // all carbon in liqC
        h_typ_2b[p*2+1]    = 0;
        // Choose distinct atom indices (allow multiple pairs per atom → tests atomicAdd)
        h_ai_2b[p] = (int)randu(rng, 0, N_ATOMS - 1);
        do { h_aj_2b[p] = (int)randu(rng, 0, N_ATOMS - 1); }
        while (h_aj_2b[p] == h_ai_2b[p]);
    }

    // ----------------------------------------------------------
    // 3. CPU reference: compute_2B one pair at a time
    // ----------------------------------------------------------

    std::vector<double> cpu_forces(N_ATOMS * 3, 0.0);
    double cpu_energy_2b = 0.0;

    std::vector<double> tmp_dr(3), tmp_force(6), tmp_stress(6);
    std::vector<int>    tmp_typ(2);
    chimes2BTmp tmp2b(order_2b);

    for (int p = 0; p < N_PAIRS; p++) {
        tmp_dr[0]  = h_dr_2b[p*3+0];
        tmp_dr[1]  = h_dr_2b[p*3+1];
        tmp_dr[2]  = h_dr_2b[p*3+2];
        tmp_typ[0] = h_typ_2b[p*2+0];
        tmp_typ[1] = h_typ_2b[p*2+1];
        std::fill(tmp_force.begin(), tmp_force.end(), 0.0);
        std::fill(tmp_stress.begin(), tmp_stress.end(), 0.0);
        double e = 0.0;
        calc.compute_2B(h_dx_2b[p], tmp_dr, tmp_typ, tmp_force, tmp_stress, e, tmp2b);
        cpu_energy_2b        += e;
        int ai = h_ai_2b[p], aj = h_aj_2b[p];
        cpu_forces[ai*3+0]  += tmp_force[0*3+0];
        cpu_forces[ai*3+1]  += tmp_force[0*3+1];
        cpu_forces[ai*3+2]  += tmp_force[0*3+2];
        cpu_forces[aj*3+0]  += tmp_force[1*3+0];
        cpu_forces[aj*3+1]  += tmp_force[1*3+1];
        cpu_forces[aj*3+2]  += tmp_force[1*3+2];
    }

    // ----------------------------------------------------------
    // 4. GPU batch: upload inputs, run kernel, retrieve outputs
    // ----------------------------------------------------------

    double *d_dx_2b, *d_dr_2b;
    int    *d_typ_2b, *d_ai_2b, *d_aj_2b;
    double *d_forces_out, *d_energy_out;

    CK(cudaMalloc(&d_dx_2b,  sizeof(double)*N_PAIRS));
    CK(cudaMalloc(&d_dr_2b,  sizeof(double)*N_PAIRS*3));
    CK(cudaMalloc(&d_typ_2b, sizeof(int)*N_PAIRS*2));
    CK(cudaMalloc(&d_ai_2b,  sizeof(int)*N_PAIRS));
    CK(cudaMalloc(&d_aj_2b,  sizeof(int)*N_PAIRS));
    CK(cudaMalloc(&d_forces_out, sizeof(double)*N_ATOMS*3));
    CK(cudaMalloc(&d_energy_out, sizeof(double)));

    CK(cudaMemcpy(d_dx_2b,  h_dx_2b.data(),  sizeof(double)*N_PAIRS,   cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_dr_2b,  h_dr_2b.data(),  sizeof(double)*N_PAIRS*3, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_typ_2b, h_typ_2b.data(), sizeof(int)*N_PAIRS*2,    cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_ai_2b,  h_ai_2b.data(),  sizeof(int)*N_PAIRS,      cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_aj_2b,  h_aj_2b.data(),  sizeof(int)*N_PAIRS,      cudaMemcpyHostToDevice));
    CK(cudaMemset(d_forces_out, 0, sizeof(double)*N_ATOMS*3));
    CK(cudaMemset(d_energy_out, 0, sizeof(double)));

    chimesFF_gpu_compute_2B(N_PAIRS, d_dx_2b, d_dr_2b, d_typ_2b,
                             d_ai_2b, d_aj_2b,
                             N_ATOMS, d_forces_out, d_energy_out);
    CK(cudaDeviceSynchronize());

    std::vector<double> gpu_forces(N_ATOMS*3);
    double gpu_energy_2b;
    CK(cudaMemcpy(gpu_forces.data(), d_forces_out, sizeof(double)*N_ATOMS*3, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&gpu_energy_2b,    d_energy_out, sizeof(double),           cudaMemcpyDeviceToHost));

    // ----------------------------------------------------------
    // 5. Compare 2B results
    // ----------------------------------------------------------

    double max_force_err = 0.0, max_rel_err = 0.0;
    for (int i = 0; i < N_ATOMS*3; i++) {
        double err  = std::abs(gpu_forces[i] - cpu_forces[i]);
        double ref  = std::abs(cpu_forces[i]);
        double rel  = (ref > 1e-30) ? err / ref : err;
        max_force_err = std::max(max_force_err, err);
        max_rel_err   = std::max(max_rel_err,   rel);
    }
    double energy_err = std::abs(gpu_energy_2b - cpu_energy_2b);
    double energy_rel = energy_err / (std::abs(cpu_energy_2b) + 1e-30);

    printf("\n=== 2B Results (%d pairs, %d atoms) ===\n", N_PAIRS, N_ATOMS);
    printf("  CPU energy:         %+.10e\n", cpu_energy_2b);
    printf("  GPU energy:         %+.10e\n", gpu_energy_2b);
    printf("  Energy abs error:   %.3e  rel error: %.3e\n", energy_err, energy_rel);
    printf("  Force  max abs err: %.3e  max rel:   %.3e\n", max_force_err, max_rel_err);

    bool pass_2b = (max_force_err < TOL) && (energy_err < TOL);
    printf("  2B: %s\n\n", pass_2b ? "PASS" : "FAIL");

    // ----------------------------------------------------------
    // 6. 3B test (if the force field has 3B terms)
    // ----------------------------------------------------------

    bool pass_3b = true;

    if (has_3b && N_TRIPS > 0) {
        // Use 2B inner cutoff as 3B lower bound; 3B max from chimesFF method
        double rmin3b = rmin2b;
        double rmax3b = calc.max_cutoff_3B();
        if (rmax3b <= 0.0) rmax3b = rmax2b; // fallback if method returns 0
        printf("3B cutoff range used: [%.3f, %.3f]\n", rmin3b, rmax3b);

        std::vector<double> h_dx_3b(N_TRIPS*3), h_dr_3b(N_TRIPS*9);
        std::vector<int>    h_typ_3b(N_TRIPS*3), h_ai_3b(N_TRIPS), h_aj_3b(N_TRIPS), h_ak_3b(N_TRIPS);

        for (int t = 0; t < N_TRIPS; t++) {
            for (int p = 0; p < 3; p++) {
                double dx = randu(rng, rmin3b*1.01, rmax3b*0.99);
                double ux = randu(rng,-1,1), uy = randu(rng,-1,1), uz = randu(rng,-1,1);
                double inv = dx / sqrt(ux*ux + uy*uy + uz*uz);
                h_dx_3b[t*3+p]     = dx;
                h_dr_3b[t*9+p*3+0] = ux*inv;
                h_dr_3b[t*9+p*3+1] = uy*inv;
                h_dr_3b[t*9+p*3+2] = uz*inv;
                h_typ_3b[t*3+p]    = 0;
            }
            h_ai_3b[t] = (int)randu(rng, 0, N_ATOMS-1);
            do { h_aj_3b[t] = (int)randu(rng, 0, N_ATOMS-1); } while (h_aj_3b[t]==h_ai_3b[t]);
            do { h_ak_3b[t] = (int)randu(rng, 0, N_ATOMS-1); } while (h_ak_3b[t]==h_ai_3b[t] || h_ak_3b[t]==h_aj_3b[t]);
        }

        // CPU reference
        std::vector<double> cpu_forces_3b(N_ATOMS*3, 0.0);
        double cpu_energy_3b = 0.0;
        chimes3BTmp tmp3b(calc.poly_orders[1]);
        std::vector<double> t_dx(3), t_dr(9), t_force(9), t_stress(6);
        std::vector<int>    t_typ(3);

        for (int t = 0; t < N_TRIPS; t++) {
            for (int p = 0; p < 3; p++) {
                t_dx[p]    = h_dx_3b[t*3+p];
                t_typ[p]   = h_typ_3b[t*3+p];
                for (int d = 0; d < 3; d++) t_dr[p*3+d] = h_dr_3b[t*9+p*3+d];
            }
            std::fill(t_force.begin(), t_force.end(), 0.0);
            std::fill(t_stress.begin(), t_stress.end(), 0.0);
            double e = 0.0;
            calc.compute_3B(t_dx, t_dr, t_typ, t_force, t_stress, e, tmp3b);
            cpu_energy_3b += e;
            int ai=h_ai_3b[t], aj=h_aj_3b[t], ak=h_ak_3b[t];
            for (int d = 0; d < 3; d++) {
                cpu_forces_3b[ai*3+d] += t_force[0*3+d];
                cpu_forces_3b[aj*3+d] += t_force[1*3+d];
                cpu_forces_3b[ak*3+d] += t_force[2*3+d];
            }
        }

        // GPU batch
        double *d_dx_3b, *d_dr_3b;
        int    *d_typ_3b, *d_ai_3b, *d_aj_3b, *d_ak_3b;

        CK(cudaMalloc(&d_dx_3b,  sizeof(double)*N_TRIPS*3));
        CK(cudaMalloc(&d_dr_3b,  sizeof(double)*N_TRIPS*9));
        CK(cudaMalloc(&d_typ_3b, sizeof(int)*N_TRIPS*3));
        CK(cudaMalloc(&d_ai_3b,  sizeof(int)*N_TRIPS));
        CK(cudaMalloc(&d_aj_3b,  sizeof(int)*N_TRIPS));
        CK(cudaMalloc(&d_ak_3b,  sizeof(int)*N_TRIPS));

        CK(cudaMemcpy(d_dx_3b,  h_dx_3b.data(),  sizeof(double)*N_TRIPS*3, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_dr_3b,  h_dr_3b.data(),  sizeof(double)*N_TRIPS*9, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_typ_3b, h_typ_3b.data(), sizeof(int)*N_TRIPS*3,    cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_ai_3b,  h_ai_3b.data(),  sizeof(int)*N_TRIPS,      cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_aj_3b,  h_aj_3b.data(),  sizeof(int)*N_TRIPS,      cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_ak_3b,  h_ak_3b.data(),  sizeof(int)*N_TRIPS,      cudaMemcpyHostToDevice));
        CK(cudaMemset(d_forces_out, 0, sizeof(double)*N_ATOMS*3));
        CK(cudaMemset(d_energy_out, 0, sizeof(double)));

        chimesFF_gpu_compute_3B(N_TRIPS, d_dx_3b, d_dr_3b, d_typ_3b,
                                 d_ai_3b, d_aj_3b, d_ak_3b,
                                 N_ATOMS, d_forces_out, d_energy_out);
        CK(cudaDeviceSynchronize());

        std::vector<double> gpu_forces_3b(N_ATOMS*3);
        double gpu_energy_3b;
        CK(cudaMemcpy(gpu_forces_3b.data(), d_forces_out, sizeof(double)*N_ATOMS*3, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(&gpu_energy_3b,       d_energy_out, sizeof(double),           cudaMemcpyDeviceToHost));

        double mfe3 = 0.0, mre3 = 0.0;
        for (int i = 0; i < N_ATOMS*3; i++) {
            double err = std::abs(gpu_forces_3b[i] - cpu_forces_3b[i]);
            double ref = std::abs(cpu_forces_3b[i]);
            mfe3 = std::max(mfe3, err);
            mre3 = std::max(mre3, (ref > 1e-30) ? err/ref : err);
        }
        double ee3  = std::abs(gpu_energy_3b - cpu_energy_3b);
        double ere3 = ee3 / (std::abs(cpu_energy_3b) + 1e-30);

        printf("=== 3B Results (%d triplets, %d atoms) ===\n", N_TRIPS, N_ATOMS);
        printf("  CPU energy:         %+.10e\n", cpu_energy_3b);
        printf("  GPU energy:         %+.10e\n", gpu_energy_3b);
        printf("  Energy abs error:   %.3e  rel error: %.3e\n", ee3, ere3);
        printf("  Force  max abs err: %.3e  max rel:   %.3e\n", mfe3, mre3);

        pass_3b = (mfe3 < TOL) && (ee3 < TOL);
        printf("  3B: %s\n\n", pass_3b ? "PASS" : "FAIL");

        cudaFree(d_dx_3b); cudaFree(d_dr_3b); cudaFree(d_typ_3b);
        cudaFree(d_ai_3b); cudaFree(d_aj_3b); cudaFree(d_ak_3b);
    }

    // Cleanup 2B device memory
    cudaFree(d_dx_2b); cudaFree(d_dr_2b); cudaFree(d_typ_2b);
    cudaFree(d_ai_2b); cudaFree(d_aj_2b);
    cudaFree(d_forces_out); cudaFree(d_energy_out);

    calc.free_device_params();

    // ----------------------------------------------------------
    // 7. Summary
    // ----------------------------------------------------------

    bool all_pass = pass_2b && pass_3b;
    printf("=== Overall: %s ===\n", all_pass ? "ALL PASS" : "SOME TESTS FAILED");
    return all_pass ? 0 : 1;
}

#else

#include <cstdio>
int main() {
    printf("This test requires a USE_CUDA build (cmake -DWITH_CUDA=ON).\n");
    return 1;
}

#endif // USE_CUDA
