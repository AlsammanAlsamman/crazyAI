#include <string.h>
#include <omp.h>

/* C = A*B, row-major, n x n. Literal translation of "the moon":
   one synchronized whole-matrix pass per shared index k, never
   finishing a cell early, never packing/moving the two fixed
   towels (A, B), never reusing a spent pen (per-k temporaries). */
void kernel(int n, const double *A, const double *B, double *C) {
    memset(C, 0, (size_t)n * n * sizeof(double));

    for (int k = 0; k < n; k++) {                    /* one moon-turn per shared index */
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) {
            double a = A[(size_t)i * n + k];          /* west-desk nerve for this pen */
            const double *brow = B + (size_t)k * n;   /* east-desk nerve row for this pen */
            double *crow = C + (size_t)i * n;
            #pragma omp simd
            for (int j = 0; j < n; j++) {
                crow[j] += a * brow[j];                /* pen trembles out the product, poured not counted */
            }
            /* 'a' and brow are the spent pen: discarded here, never reused for another k */
        }
        /* implicit OpenMP barrier: this moon-turn is fully poured into every
           rightful cell of the whole third towel before the next k begins */
    }
}
