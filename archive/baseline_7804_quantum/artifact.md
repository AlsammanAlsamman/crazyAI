# APPROACH

The naive per-gate scalar sweep touches every amplitude pair with a branchy loop and re-reads/re-writes the whole state once per gate. Two orthogonal wins are available without changing semantics:

1. **Gate fusion**: consecutive gates that target the *same* qubit commute into a single 2×2 complex matrix (matrix product, left-multiplied for each new gate). Instead of N/2 pair-updates per gate, we do N/2 pair-updates per *run* of same-qubit gates — collapsing runs of k gates into 1 sweep instead of k.
2. **SIMD + branch-free indexing**: replace the `if (i & bit) continue` scan with a block/stride loop (`block += 2*bit`, inner range `[block, block+bit)`) so there's no branching and the `i`-run and `j`-run are each contiguous, which AVX (`__m256d`, 4-wide double) vectorizes cleanly. Runs shorter than 4 (qubit 0,1) fall back to a tiny scalar loop.
3. **OpenMP** over the outer block loop, gated by a size threshold so small states don't pay thread-launch overhead.

An identity-matrix short-circuit (e.g. `X;X` or `Z;Z` fusing to the identity) skips the sweep entirely at negligible cost.

# ARTIFACT

```c
#include <math.h>
#include <immintrin.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static inline void cmul(double xr, double xi, double yr, double yi, double *or_, double *oi_) {
    *or_ = xr * yr - xi * yi;
    *oi_ = xr * yi + xi * yr;
}

static void apply_matrix(int n_qubits, long long bit,
                          double ar, double ai, double br, double bi,
                          double cr, double ci, double dr, double di,
                          double *re, double *im) {
    long long N = 1LL << n_qubits;
    long long step = bit << 1;
    long long nblocks = N / step;

    if (bit >= 4) {
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static) if(N >= (1LL<<20))
        #endif
        for (long long b = 0; b < nblocks; b++) {
            long long i = b * step;
            long long j = i + bit;
            long long k = 0;
            __m256d arv = _mm256_set1_pd(ar), aiv = _mm256_set1_pd(ai);
            __m256d brv = _mm256_set1_pd(br), biv = _mm256_set1_pd(bi);
            __m256d crv = _mm256_set1_pd(cr), civ = _mm256_set1_pd(ci);
            __m256d drv = _mm256_set1_pd(dr), div = _mm256_set1_pd(di);
            for (; k + 4 <= bit; k += 4) {
                __m256d r0 = _mm256_loadu_pd(&re[i + k]);
                __m256d x0 = _mm256_loadu_pd(&im[i + k]);
                __m256d r1 = _mm256_loadu_pd(&re[j + k]);
                __m256d x1 = _mm256_loadu_pd(&im[j + k]);

                __m256d nri = _mm256_add_pd(
                    _mm256_sub_pd(_mm256_mul_pd(arv, r0), _mm256_mul_pd(aiv, x0)),
                    _mm256_sub_pd(_mm256_mul_pd(brv, r1), _mm256_mul_pd(biv, x1)));
                __m256d nxi = _mm256_add_pd(
                    _mm256_add_pd(_mm256_mul_pd(arv, x0), _mm256_mul_pd(aiv, r0)),
                    _mm256_add_pd(_mm256_mul_pd(brv, x1), _mm256_mul_pd(biv, r1)));
                __m256d nrj = _mm256_add_pd(
                    _mm256_sub_pd(_mm256_mul_pd(crv, r0), _mm256_mul_pd(civ, x0)),
                    _mm256_sub_pd(_mm256_mul_pd(drv, r1), _mm256_mul_pd(div, x1)));
                __m256d nxj = _mm256_add_pd(
                    _mm256_add_pd(_mm256_mul_pd(crv, x0), _mm256_mul_pd(civ, r0)),
                    _mm256_add_pd(_mm256_mul_pd(drv, x1), _mm256_mul_pd(div, r1)));

                _mm256_storeu_pd(&re[i + k], nri);
                _mm256_storeu_pd(&im[i + k], nxi);
                _mm256_storeu_pd(&re[j + k], nrj);
                _mm256_storeu_pd(&im[j + k], nxj);
            }
            for (; k < bit; k++) {
                double r0 = re[i + k], x0 = im[i + k], r1 = re[j + k], x1 = im[j + k];
                re[i + k] = ar * r0 - ai * x0 + br * r1 - bi * x1;
                im[i + k] = ar * x0 + ai * r0 + br * x1 + bi * r1;
                re[j + k] = cr * r0 - ci * x0 + dr * r1 - di * x1;
                im[j + k] = cr * x0 + ci * r0 + dr * x1 + di * r1;
            }
        }
    } else {
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static) if(N >= (1LL<<20))
        #endif
        for (long long b = 0; b < nblocks; b++) {
            long long base = b * step;
            for (long long k = 0; k < bit; k++) {
                long long i = base + k, j = i + bit;
                double r0 = re[i], x0 = im[i], r1 = re[j], x1 = im[j];
                re[i] = ar * r0 - ai * x0 + br * r1 - bi * x1;
                im[i] = ar * x0 + ai * r0 + br * x1 + bi * r1;
                re[j] = cr * r0 - ci * x0 + dr * r1 - di * x1;
                im[j] = cr * x0 + ci * r0 + dr * x1 + di * r1;
            }
        }
    }
}

void kernel(int n_qubits, int n_gates, const int *gate_type, const int *gate_qubit, const double *gate_param,
            double *state_re, double *state_im) {
    int g = 0;
    while (g < n_gates) {
        int q = gate_qubit[g];
        double ar = 1, ai = 0, br = 0, bi = 0, cr = 0, ci = 0, dr = 1, di = 0; /* identity */

        while (g < n_gates && gate_qubit[g] == q) {
            double gar, gai, gbr, gbi, gcr, gci, gdr, gdi;
            int gt = gate_type[g];
            if (gt == 0) {
                double s = 0.70710678118654752440;
                gar = s; gai = 0; gbr = s; gbi = 0; gcr = s; gci = 0; gdr = -s; gdi = 0;
            } else if (gt == 1) {
                gar = 0; gai = 0; gbr = 1; gbi = 0; gcr = 1; gci = 0; gdr = 0; gdi = 0;
            } else if (gt == 2) {
                gar = 1; gai = 0; gbr = 0; gbi = 0; gcr = 0; gci = 0; gdr = -1; gdi = 0;
            } else {
                double phi0 = -gate_param[g] * 0.5, phi1 = gate_param[g] * 0.5;
                gar = cos(phi0); gai = sin(phi0); gbr = 0; gbi = 0;
                gcr = 0; gci = 0; gdr = cos(phi1); gdi = sin(phi1);
            }

            double t1r, t1i, t2r, t2i, nar, nai, nbr, nbi, ncr, nci, ndr, ndi;
            cmul(gar, gai, ar, ai, &t1r, &t1i); cmul(gbr, gbi, cr, ci, &t2r, &t2i);
            nar = t1r + t2r; nai = t1i + t2i;
            cmul(gar, gai, br, bi, &t1r, &t1i); cmul(gbr, gbi, dr, di, &t2r, &t2i);
            nbr = t1r + t2r; nbi = t1i + t2i;
            cmul(gcr, gci, ar, ai, &t1r, &t1i); cmul(gdr, gdi, cr, ci, &t2r, &t2i);
            ncr = t1r + t2r; nci = t1i + t2i;
            cmul(gcr, gci, br, bi, &t1r, &t1i); cmul(gdr, gdi, dr, di, &t2r, &t2i);
            ndr = t1r + t2r; ndi = t1i + t2i;

            ar = nar; ai = nai; br = nbr; bi = nbi; cr = ncr; ci = nci; dr = ndr; di = ndi;
            g++;
        }

        int is_identity = (fabs(ar - 1.0) < 1e-15 && fabs(ai) < 1e-15 &&
                            fabs(dr - 1.0) < 1e-15 && fabs(di) < 1e-15 &&
                            fabs(br) < 1e-15 && fabs(bi) < 1e-15 &&
                            fabs(cr) < 1e-15 && fabs(ci) < 1e-15);
        if (!is_identity) {
            apply_matrix(n_qubits, 1LL << q, ar, ai, br, bi, cr, ci, dr, di, state_re, state_im);
        }
    }
}
```

PREDICTION: 4

# MEASUREMENT

No benchmarking tools (`quantum_bench`/`quantum_contract`) are exposed in this session — per the environment's own note, no tools are available here and the pipeline performs compilation/measurement externally. I cannot run the harness myself, so the number above is a reasoned estimate, not an observed result:
- AVX (4-wide double) on the hot inner loop → roughly 3–4× over scalar for compute-bound long runs (qubit ≥ 2).
- Gate fusion collapses runs of same-qubit gates into one sweep, cutting sweep count further whenever a circuit has adjacent same-qubit gates (common with repeated Rz or H·X patterns); for circuits with little same-qubit adjacency this yields close to 1× extra (no harm, since fusion of a single gate is a no-op cost-wise).
- OpenMP only engages above `N ≥ 2^20` amplitudes, so it should help large-`n_qubits` cases without hurting small ones.

# VERDICT

Implementation is a straightforward, low-risk extension of the reference algorithm (same per-pair math, just batched/vectorized/fused), so correctness risk is low: the matrix-fusion math was hand-verified against each gate definition (H, X, Z, Rz) and the composition order (`G·M`, new gate left-multiplies accumulated matrix). Expected to be reliably faster than the scalar reference; actual speedup depends on the benchmark's `n_qubits` distribution and how many same-qubit gate runs appear, which I could not verify empirically in this tool-less session.