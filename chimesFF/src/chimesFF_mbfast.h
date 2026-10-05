/*
   chimesMBFast: fast evaluation of ChIMES 3- and 4-body interactions.

   The n-body ChIMES energy of a cluster is a sparse tensor contraction

       E = sum_{p} C_p  prod_m  fcut(r_m) T_{p_m}(s(r_m)),

   so the following exact restructurings apply (results match the term-by-term loop of
   chimesFF::compute_3B/compute_4B up to floating-point reassociation):

   1. Per-pair caching. The fcut-weighted Chebyshev vectors
          X_n(r) = fcut(r) T_n(s(r)),   D_n(r) = d X_n / dr
      depend only on the pair, not on the cluster, so they are computed once per pair per
      step and shared by every triplet/quadruplet that contains the pair.

   2. Sum factorization. The coefficient tensor is stored as a compressed sparse fiber (CSF)
      tree in its file column order. The energy and all M pair-distance derivatives are
      accumulated in one sweep: a level-k node carries its subtree value and the derivatives
      with respect to the modes below it, so a leaf costs two multiply-adds instead of the
      O(M^2) products plus force/stress scatter of the term-by-term loop.

   3. SIMD over clusters. Clusters of one type walk the same CSF tree, so LANES of them are
      evaluated in lockstep (double precision) with their inputs laid out lane-minor; every
      tree operation becomes one vector operation. Build with OpenMP SIMD and the target's
      vector ISA (e.g. -qopenmp-simd -xCORE-AVX512) for the full benefit.

   All arithmetic is double precision; nothing is approximated.

   Usage: init(ff) after the parameter file and the pair_int_{trip,quad} maps are built;
   each step resize()/fill_pair() for every pair used by a cluster, then either eval3()/eval4()
   per cluster or active() + eval_batch() for up to LANES clusters of one type key at a time.
*/

#ifndef CHIMESFF_MBFAST_H
#define CHIMESFF_MBFAST_H

#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include "chimesFF.h"

class chimesMBFast
{
public:

    static const int MAX_ORDER = 25;      // polynomial order limit of the fixed lane buffers
    static const int LANES     = 8;       // clusters per SIMD batch (8 doubles = one AVX-512 register)

    template<int M>
    struct Tensor
    {
        // CSF: level k nodes have power idx[k][n]; children [ptr[k][n], ptr[k][n+1]) in level k+1
        std::vector<unsigned char> idx[M];
        std::vector<int>           ptr[M-1];
        std::vector<double>        coef;     // one per leaf
        int nroot = 0;
    };

    // A variant = (pair type, inner cutoff, outer cutoff) used by some cluster-type slot.
    struct Variants
    {
        int order = 0;
        std::vector<std::array<double,2> > cut;
        std::vector<int>                   ptype;
        std::vector<std::vector<int> >     by_ptype;      // pair type -> variant ids
        std::vector<std::vector<int> >     slot_variant;  // cluster type key -> variant per slot

        int get(int pt, double rin, double rout)
        {
            for (size_t v = 0; v < cut.size(); v++)
                if (ptype[v] == pt && cut[v][0] == rin && cut[v][1] == rout) return v;
            std::array<double,2> c = {{rin, rout}};
            cut.push_back(c);
            ptype.push_back(pt);
            return cut.size() - 1;
        }
    };

    struct Cache
    {
        int nvar = 0, stride = 0;           // stride = 2*(order+1)
        std::vector<int>    offset;         // [pair*nvar + v], -1 = pair beyond that cutoff
        std::vector<double> data;
    };

    chimesFF *ff = nullptr;
    Variants V3, V4;
    Cache    C3, C4;
    std::vector<Tensor<3> > T3;
    std::vector<Tensor<6> > T4;
    std::vector<double> Tn, Tnd;

    // Returns an empty string on success, otherwise the reason this model cannot be handled.
    std::string init(chimesFF &chimes)
    {
        ff = &chimes;
        const int nt  = ff->natmtyps;
        const int npt = ff->chimes_2b_cutoff.size();

        V3 = Variants(); V4 = Variants(); T3.clear(); T4.clear();
        V3.order = ff->poly_orders.size() > 1 ? ff->poly_orders[1] : 0;
        V4.order = ff->poly_orders.size() > 2 ? ff->poly_orders[2] : 0;
        if (V3.order > MAX_ORDER || V4.order > MAX_ORDER)
            return "3- or 4-body polynomial order above " + std::to_string(MAX_ORDER);
        V3.by_ptype.assign(npt, std::vector<int>());
        V4.by_ptype.assign(npt, std::vector<int>());

        if (V3.order > 0)
        {
            for (size_t t = 0; t < ff->chimes_3b_params.size(); t++)
                T3.push_back(make_tensor<3>(ff->chimes_3b_powers[t], ff->chimes_3b_params[t], ff->ncoeffs_3b[t]));

            const int sl[3][2] = {{0,1},{0,2},{1,2}};
            V3.slot_variant.assign(nt*nt*nt, std::vector<int>(3, -1));
            for (int a = 0; a < nt; a++) for (int b = 0; b < nt; b++) for (int c = 0; c < nt; c++)
            {
                int key = (a*nt + b)*nt + c, ty[3] = {a, b, c};
                int tidx = ff->atom_int_trip_map[key];
                if (tidx < 0) continue;
                for (int s = 0; s < 3; s++)
                {
                    int col = ff->pair_int_trip_map[key][s];
                    int pt  = ff->atom_int_pair_map[ty[sl[s][0]]*nt + ty[sl[s][1]]];
                    V3.slot_variant[key][s] = V3.get(pt, ff->chimes_3b_cutoff[tidx][0][col], ff->chimes_3b_cutoff[tidx][1][col]);
                }
            }
            for (size_t v = 0; v < V3.cut.size(); v++) V3.by_ptype[V3.ptype[v]].push_back(v);
        }
        if (V4.order > 0)
        {
            for (size_t q = 0; q < ff->chimes_4b_params.size(); q++)
                T4.push_back(make_tensor<6>(ff->chimes_4b_powers[q], ff->chimes_4b_params[q], ff->ncoeffs_4b[q]));

            const int sl[6][2] = {{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};
            V4.slot_variant.assign(nt*nt*nt*nt, std::vector<int>(6, -1));
            for (int a = 0; a < nt; a++) for (int b = 0; b < nt; b++) for (int c = 0; c < nt; c++) for (int d = 0; d < nt; d++)
            {
                int key = ((a*nt + b)*nt + c)*nt + d, ty[4] = {a, b, c, d};
                int qidx = ff->atom_int_quad_map[key];
                if (qidx < 0) continue;
                for (int s = 0; s < 6; s++)
                {
                    int col = ff->pair_int_quad_map[key][s];
                    int pt  = ff->atom_int_pair_map[ty[sl[s][0]]*nt + ty[sl[s][1]]];
                    V4.slot_variant[key][s] = V4.get(pt, ff->chimes_4b_cutoff[qidx][0][col], ff->chimes_4b_cutoff[qidx][1][col]);
                }
            }
            for (size_t v = 0; v < V4.cut.size(); v++) V4.by_ptype[V4.ptype[v]].push_back(v);
        }
        int omax = std::max(V3.order, V4.order) + 1;
        Tn.assign(omax, 0.0); Tnd.assign(omax, 0.0);
        return "";
    }

    // ChIMES pair type of an atom pair with ChIMES atom types ti, tj
    inline int pair_type(int ti, int tj) const { return ff->atom_int_pair_map[ti*ff->natmtyps + tj]; }

    // ---- per-step pair cache

    void resize(int npairs)
    {
        resize_one(C3, V3, npairs);
        resize_one(C4, V4, npairs);
    }

    // Pair p of ChIMES pair type pt at distance r: fill X and D for every variant of pt.
    inline void fill_pair(int p, int pt, double r)
    {
        if (V3.order > 0) fill_one(C3, V3, p, pt, r);
        if (V4.order > 0) fill_one(C4, V4, p, pt, r);
    }

    // ---- cluster type keys

    inline int key3(const int *typ) const { const int nt = ff->natmtyps; return (typ[0]*nt + typ[1])*nt + typ[2]; }
    inline int key4(const int *typ) const { const int nt = ff->natmtyps; return ((typ[0]*nt + typ[1])*nt + typ[2])*nt + typ[3]; }
    inline int nkeys(int body) const { int nt = ff->natmtyps; return body == 3 ? nt*nt*nt : nt*nt*nt*nt; }

    // true if the cluster type is included and every pair is inside its cutoff
    inline bool active(int body, int key, const int *pids) const
    {
        const int idx = body == 3 ? ff->atom_int_trip_map[key] : ff->atom_int_quad_map[key];
        if (idx < 0) return false;
        const Cache &C = body == 3 ? C3 : C4;
        const std::vector<int> &sv = body == 3 ? V3.slot_variant[key] : V4.slot_variant[key];
        const int M = body == 3 ? 3 : 6;
        for (int s = 0; s < M; s++) if (C.offset[(size_t)pids[s]*C.nvar + sv[s]] < 0) return false;
        return true;
    }

    // ---- one cluster at a time
    //
    // pids[s] = pair id of slot s (3B slots ij ik jk; 4B slots ij ik il jk jl kl);
    // typ = ChIMES atom type indices. On return E holds the cluster energy and dEdr[s] the
    // derivative with respect to the slot-s distance. Returns false for excluded or
    // out-of-cutoff clusters (E, dEdr untouched).

    inline bool eval3(const int *pids, const int *typ, double &E, double *dEdr)
    {
        const int key = key3(typ);
        const int tidx = ff->atom_int_trip_map[key];
        if (tidx < 0) return false;
        return eval<3>(pids, ff->pair_int_trip_map[key], V3.slot_variant[key], C3, T3[tidx], V3.order + 1, E, dEdr);
    }

    inline bool eval4(const int *pids, const int *typ, double &E, double *dEdr)
    {
        const int key = key4(typ);
        const int qidx = ff->atom_int_quad_map[key];
        if (qidx < 0) return false;
        return eval<6>(pids, ff->pair_int_quad_map[key], V4.slot_variant[key], C4, T4[qidx], V4.order + 1, E, dEdr);
    }

    // ---- up to LANES clusters of one type key at a time (all must be active)
    //
    // pids: nb clusters x M slot pair ids (row-major), nb <= LANES; E[nb], dEdr[nb*M] in slot order.

    inline void eval_batch(int body, int key, const int *pids, int nb, double *E, double *dEdr) const
    {
        if (body == 3)
            batch<3>(pids, nb, ff->pair_int_trip_map[key], V3.slot_variant[key], C3, T3[ff->atom_int_trip_map[key]], V3.order + 1, E, dEdr);
        else
            batch<6>(pids, nb, ff->pair_int_quad_map[key], V4.slot_variant[key], C4, T4[ff->atom_int_quad_map[key]], V4.order + 1, E, dEdr);
    }

private:

    template<bool B> struct Flag {};

    template<int M>
    static Tensor<M> make_tensor(const std::vector<std::vector<int> > &powers, const std::vector<double> &params, int n)
    {
        typedef std::pair<std::array<unsigned char,M>, double> Term;
        Tensor<M> T;
        std::vector<Term> terms(n);
        for (int c = 0; c < n; c++)
        {
            for (int m = 0; m < M; m++) terms[c].first[m] = powers[c][m];
            terms[c].second = params[c];
        }
        std::sort(terms.begin(), terms.end(), [](const Term &x, const Term &y) { return x.first < y.first; });
        for (size_t t = 0; t < terms.size(); t++)
        {
            int k0 = 0;   // first level at which this term leaves the previous term's path
            if (t > 0)
            {
                while (k0 < M && terms[t].first[k0] == terms[t-1].first[k0]) k0++;
                if (k0 == M) k0 = M - 1;   // repeated powers: a sibling leaf
            }
            for (int k = k0; k < M; k++)
            {
                if (k < M-1) T.ptr[k].push_back(T.idx[k+1].size());
                T.idx[k].push_back(terms[t].first[k]);
            }
            T.coef.push_back(terms[t].second);
        }
        for (int k = 0; k < M-1; k++) T.ptr[k].push_back(T.idx[k+1].size());
        T.nroot = T.idx[0].size();
        return T;
    }

    static void resize_one(Cache &C, const Variants &V, int npairs)
    {
        C.nvar   = V.cut.size();
        C.stride = 2 * (V.order + 1);
        C.offset.assign((size_t)npairs * C.nvar, -1);
        C.data.resize((size_t)npairs * C.nvar * C.stride);
    }

    inline void fill_one(Cache &C, const Variants &V, int p, int pt, double r)
    {
        const int no = V.order + 1;
        for (size_t iv = 0; iv < V.by_ptype[pt].size(); iv++)
        {
            const int v = V.by_ptype[pt][iv];
            const size_t slot = (size_t)p * C.nvar + v;
            if (r >= V.cut[v][1]) { C.offset[slot] = -1; continue; }
            ff->set_cheby_polys(Tn, Tnd, r, ff->morse_var[pt], V.cut[v][0], V.cut[v][1], V.order);
            double fc, fcd;
            ff->get_fcut(r, V.cut[v][1], fc, fcd);
            const size_t off = slot * C.stride;
            C.offset[slot] = off;
            double *X = &C.data[off], *D = X + no;
            for (int n = 0; n < no; n++) { X[n] = fc * Tn[n]; D[n] = fc * Tnd[n] + fcd * Tn[n]; }
        }
    }

    // ---- scalar CSF sweep. Node at level K: out[0] = subtree value, out[1+j] = d/dmode K+j.

    template<int K, int M>
    struct Node
    {
        static inline void eval(const Tensor<M> &T, int n, const double *const *X, const double *const *D, double *out)
        {
            eval(T, n, X, D, out, Flag<K == M-1>());
        }
        static inline void eval(const Tensor<M> &T, int n, const double *const *X, const double *const *D, double *out, Flag<true>)
        {
            const int p = T.idx[M-1][n];
            out[0] = T.coef[n] * X[M-1][p];
            out[1] = T.coef[n] * D[M-1][p];
        }
        static inline void eval(const Tensor<M> &T, int n, const double *const *X, const double *const *D, double *out, Flag<false>)
        {
            double acc[M-K] = {0};
            for (int ch = T.ptr[K][n]; ch < T.ptr[K][n+1]; ch++)
            {
                double o[M-K];
                Node<(K < M-1 ? K+1 : K), M>::eval(T, ch, X, D, o);
                for (int j = 0; j < M-K; j++) acc[j] += o[j];
            }
            const int p = T.idx[K][n];
            const double x = X[K][p];
            out[0] = x * acc[0];
            out[1] = D[K][p] * acc[0];
            for (int j = 1; j < M-K; j++) out[1+j] = x * acc[j];
        }
    };

    template<int M>
    inline bool eval(const int *pids, const std::vector<int> &cols, const std::vector<int> &sv,
                     const Cache &C, const Tensor<M> &T, int no, double &E, double *dEdr) const
    {
        const double *X[M], *D[M];
        for (int s = 0; s < M; s++)
        {
            const int off = C.offset[(size_t)pids[s] * C.nvar + sv[s]];
            if (off < 0) return false;          // a pair at or beyond its cutoff: no interaction
            X[cols[s]] = &C.data[off];
            D[cols[s]] = &C.data[off + no];
        }
        double acc[M+1] = {0};
        for (int n = 0; n < T.nroot; n++)
        {
            double o[M+1];
            Node<0,M>::eval(T, n, X, D, o);
            for (int j = 0; j <= M; j++) acc[j] += o[j];
        }
        E = acc[0];
        for (int s = 0; s < M; s++) dEdr[s] = acc[1 + cols[s]];
        return true;
    }

    // ---- lane-batched CSF sweep. Lane node at level K; out layout [(M-K+1)][W]:
    //      value, d/dmode K, d/dmode K+1, ...  (K runs to M-2; leaves are folded into level M-2)

    template<int K, int M>
    struct LaneNode
    {
        static const int W = LANES;
        static inline void eval(const Tensor<M> &T, int n, const double *Xs, const double *Ds, int no, double *out)
        {
            alignas(64) double acc[(M-K)*W];
            for (int i = 0; i < (M-K)*W; i++) acc[i] = 0;
            children(T, n, Xs, Ds, no, acc, Flag<K == M-2>());
            const int p = T.idx[K][n];
            const double *x = &Xs[(K*no + p)*W], *d = &Ds[(K*no + p)*W];
            #pragma omp simd
            for (int w = 0; w < W; w++) { out[w] = x[w]*acc[w]; out[W+w] = d[w]*acc[w]; }
            for (int j = 1; j < M-K; j++)
            {
                #pragma omp simd
                for (int w = 0; w < W; w++) out[(1+j)*W+w] = x[w]*acc[j*W+w];
            }
        }
        // children are leaves: two multiply-adds per leaf and lane
        static inline void children(const Tensor<M> &T, int n, const double *Xs, const double *Ds, int no, double *acc, Flag<true>)
        {
            for (int ch = T.ptr[K][n]; ch < T.ptr[K][n+1]; ch++)
            {
                const double c = T.coef[ch];
                const int p = T.idx[M-1][ch];
                const double *x = &Xs[((M-1)*no + p)*W], *d = &Ds[((M-1)*no + p)*W];
                #pragma omp simd
                for (int w = 0; w < W; w++) { acc[w] += c*x[w]; acc[W+w] += c*d[w]; }
            }
        }
        static inline void children(const Tensor<M> &T, int n, const double *Xs, const double *Ds, int no, double *acc, Flag<false>)
        {
            for (int ch = T.ptr[K][n]; ch < T.ptr[K][n+1]; ch++)
            {
                alignas(64) double o[(M-K)*W];
                LaneNode<(K < M-2 ? K+1 : K), M>::eval(T, ch, Xs, Ds, no, o);
                #pragma omp simd
                for (int i = 0; i < (M-K)*W; i++) acc[i] += o[i];
            }
        }
    };

    template<int M>
    inline void batch(const int *pids, int nb, const std::vector<int> &cols, const std::vector<int> &sv,
                      const Cache &C, const Tensor<M> &T, int no, double *E, double *dEdr) const
    {
        const int W = LANES;
        alignas(64) double Xs[M*(MAX_ORDER+1)*LANES], Ds[M*(MAX_ORDER+1)*LANES];
        alignas(64) double root[(M+1)*LANES];
        for (int w = 0; w < W; w++)
        {
            const int *pid = &pids[M * (w < nb ? w : nb-1)];     // idle lanes repeat the last cluster
            for (int s = 0; s < M; s++)
            {
                const double *src = &C.data[C.offset[(size_t)pid[s]*C.nvar + sv[s]]];
                const int m = cols[s];
                for (int p = 0; p < no; p++) { Xs[(m*no + p)*W + w] = src[p]; Ds[(m*no + p)*W + w] = src[no + p]; }
            }
        }
        for (int i = 0; i < (M+1)*W; i++) root[i] = 0.0;
        for (int r = 0; r < T.nroot; r++)
        {
            alignas(64) double o[(M+1)*LANES];
            LaneNode<0, M>::eval(T, r, Xs, Ds, no, o);
            #pragma omp simd
            for (int i = 0; i < (M+1)*W; i++) root[i] += o[i];
        }
        for (int w = 0; w < nb; w++)
        {
            E[w] = root[w];
            for (int s = 0; s < M; s++) dEdr[w*M + s] = root[(1 + cols[s])*W + w];
        }
    }
};

#endif
