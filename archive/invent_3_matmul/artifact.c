#include <immintrin.h>

#define MR 4
#define NR 4

/* The cairn: a 4x4 tile accumulated entirely in registers (the covered
   cart) across the full k range, and written to the shared table C
   exactly once, after every boulder meant for it has arrived. */
static inline void microkernel_4x4(int n, int i0, int j0,
                                    const double *A, const double *B,
                                    double *C)
{
    __m256d c0 = _mm256_setzero_pd();
    __m256d c1 = _mm256_setzero_pd();
    __m256d c2 = _mm256_setzero_pd();
    __m256d c3 = _mm256_setzero_pd();

    const double *a0 = A + (size_t)(i0 + 0) * n;
    const double *a1 = A + (size_t)(i0 + 1) * n;
    const double *a2 = A + (size_t)(i0 + 2) * n;
    const double *a3 = A + (size_t)(i0 + 3) * n;

    for (int k = 0; k < n; k++) {
        /* the second table's street: a row of B, read once */
        __m256d b = _mm256_loadu_pd(B + (size_t)k * n + j0);

        /* the boulder: A[i,k], fired across (broadcast) into the cart */
        __m256d va0 = _mm256_set1_pd(a0[k]);
        __m256d va1 = _mm256_set1_pd(a1[k]);
        __m256d va2 = _mm256_set1_pd(a2[k]);
        __m256d va3 = _mm256_set1_pd(a3[k]);

        /* ground on the quern, stacked onto the cairn (register) */
        c0 = _mm256_fmadd_pd(va0, b, c0);
        c1 = _mm256_fmadd_pd(va1, b, c1);
        c2 = _mm256_fmadd_pd(va2, b, c2);
        c3 = _mm256_fmadd_pd(va3, b, c3);
    }

    /* the cairn is read and written into the third table exactly once */
    _mm256_storeu_pd(C + (size_t)(i0 + 0) * n + j0, c0);
    _mm256_storeu_pd(C + (size_t)(i0 + 1) * n + j0, c1);
    _mm256_storeu_pd(C + (size_t)(i0 + 2) * n + j0, c2);
    _mm256_storeu_pd(C + (size_t)(i0 + 3) * n + j0, c3);
}

/* scalar fallback for edge tiles (n not a multiple of MR/NR) */
static inline void microkernel_scalar(int n, int i0, int i1, int j0, int j1,
                                       const double *A, const double *B,
                                       double *C)
{
    for (int i = i0; i < i1; i++) {
        const double *arow = A + (size_t)i * n;
        for (int j = j0; j < j1; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++)
                sum += arow[k] * B[(size_t)k * n + j];
            C[(size_t)i * n + j] = sum;
        }
    }
}

void kernel(int n, const double *A, const double *B, double *C)
{
    const int NB = 256; /* column-panel width: keeps a B panel resident
                            in cache and reused across a thread's row tiles */

    int mfull = (n / MR) * MR;

    #pragma omp parallel for schedule(static)
    for (int ii = 0; ii < mfull; ii += MR) {
        for (int jj = 0; jj < n; jj += NB) {
            int jend  = (jj + NB < n) ? jj + NB : n;
            int jfull = jj + ((jend - jj) / NR) * NR;
            int j;
            for (j = jj; j < jfull; j += NR)
                microkernel_4x4(n, ii, j, A, B, C);
            if (j < jend)
                microkernel_scalar(n, ii, ii + MR, j, jend, A, B, C);
        }
    }

    /* leftover rows if n is not a multiple of MR */
    if (mfull < n)
        microkernel_scalar(n, mfull, n, 0, n, A, B, C);
}
