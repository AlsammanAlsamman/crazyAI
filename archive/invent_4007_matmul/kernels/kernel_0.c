#include <immintrin.h>
#include <omp.h>

static inline double hsum256(__m256d v) {
    __m128d lo = _mm256_castpd256_pd128(v);
    __m128d hi = _mm256_extractf128_pd(v, 1);
    lo = _mm_add_pd(lo, hi);
    __m128d hh = _mm_unpackhi_pd(lo, lo);
    __m128d s = _mm_add_sd(lo, hh);
    return _mm_cvtsd_f64(s);
}

void kernel(int n, const double *A, const double *B, double *C) {
    if (n <= 0) return;
    size_t N = (size_t)n;

    /* Seed 1: cut the second table into column-troughs, at right angles to
       the row-troughs, fed from its own millrace -- transpose B so a
       "crossing" always joins two contiguous flows. */
    double *Bt = (double*)_mm_malloc(N * N * sizeof(double), 64);
    #pragma omp parallel for schedule(static)
    for (int k = 0; k < n; k++) {
        const double *Brow = B + (size_t)k * N;
        for (int j = 0; j < n; j++)
            Bt[(size_t)j * N + k] = Brow[j];
    }

    /* A batch of row-troughs that all get flooded by the same sweep across
       every column-trough before being let go ("the whole yard floods
       together, every crossing filled in the same breath"). Batch size
       chosen so the batch of A rows stays resident while Bt is swept once. */
    int IB = (int)(262144 / (8 * N));
    if (IB < 1) IB = 1;
    if (IB > 64) IB = 64;
    int nblocks = (n + IB - 1) / IB;

    #pragma omp parallel for schedule(dynamic)
    for (int bi = 0; bi < nblocks; bi++) {
        int i0 = bi * IB;
        int i1 = i0 + IB; if (i1 > n) i1 = n;

        for (int j = 0; j < n; j++) {
            const double *Brow = Bt + (size_t)j * N;

            for (int i = i0; i < i1; i++) {
                const double *Arow = A + (size_t)i * N;

                /* Seed 2/3: a stack of nested owls -- independent lanes whose
                   throats pass only the product; their drips pool by simple
                   addition, combined only at the bottom. */
                __m256d acc0 = _mm256_setzero_pd();
                __m256d acc1 = _mm256_setzero_pd();
                __m256d acc2 = _mm256_setzero_pd();
                __m256d acc3 = _mm256_setzero_pd();

                int k = 0;
                int limit = n - (n % 16);
                for (; k < limit; k += 16) {
                    acc0 = _mm256_fmadd_pd(_mm256_loadu_pd(Arow + k),      _mm256_loadu_pd(Brow + k),      acc0);
                    acc1 = _mm256_fmadd_pd(_mm256_loadu_pd(Arow + k + 4),  _mm256_loadu_pd(Brow + k + 4),  acc1);
                    acc2 = _mm256_fmadd_pd(_mm256_loadu_pd(Arow + k + 8),  _mm256_loadu_pd(Brow + k + 8),  acc2);
                    acc3 = _mm256_fmadd_pd(_mm256_loadu_pd(Arow + k + 12), _mm256_loadu_pd(Brow + k + 12), acc3);
                }
                for (; k + 4 <= n; k += 4)
                    acc0 = _mm256_fmadd_pd(_mm256_loadu_pd(Arow + k), _mm256_loadu_pd(Brow + k), acc0);

                /* the cistern is read only once the water has gone still */
                double sum = hsum256(_mm256_add_pd(_mm256_add_pd(acc0, acc1), _mm256_add_pd(acc2, acc3)));
                for (; k < n; k++) sum += Arow[k] * Brow[k];

                C[(size_t)i * N + j] = sum;
            }
        }
    }

    _mm_free(Bt);
}
