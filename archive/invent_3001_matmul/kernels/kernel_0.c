#include <immintrin.h>
#include <string.h>

/* Literal mapping of the tortoise/canvas method to GEMM:
   - A, B are pinned "canvases": read straight from their given
     row-major layout, never copied/repacked into a scratch layout
     ("the canvases ... pinned flat, never re-hung mid-work" - this
     is the deliberate departure from OpenBLAS's packed panels).
   - C is the third canvas, hung between them, blanked to zero
     before any ash (partial product) settles into it.
   - The grid is shaded down to one broad tile at a time: IB rows x
     JB columns. Only that tile of C, and the matching JB-wide strip
     of B, stay "awake" (cache-resident) while it is worked on.
   - The tortoise walks the shared index k at a fixed, single-step
     rhythm (k = 0,1,2,...,n-1). At every step ALL cells of the
     current tile get their crossing burned in together - the sum
     over k is NOT finished for one cell before the next is begun;
     it finishes for the whole tile only after the full walk.
   - Once the walk finishes for a tile, its ash has fully settled
     (values are final) and the shade rolls to the next tile; a
     tile's A/B values are never revisited ("spent light... swept
     out"), and tiles never overlap, so no square gets another
     tile's ash.
*/
void kernel(int n, const double *A, const double *B, double *C) {
    memset(C, 0, (size_t)n * n * sizeof(double));

    const int IB = 8;    /* row-block: the tortoise's shell width, in rows   */
    const int JB = 256;  /* column band: the shade's one broad band         */

    #pragma omp parallel for schedule(dynamic) collapse(2)
    for (int ib = 0; ib < n; ib += IB) {
        for (int jb = 0; jb < n; jb += JB) {
            int imax = (ib + IB < n) ? ib + IB : n;
            int jmax = (jb + JB < n) ? jb + JB : n;
            int jw   = jmax - jb;
            int jw4  = jw - (jw % 4);

            /* the tortoise walks the shared index, one fixed step at a time */
            for (int k = 0; k < n; k++) {
                const double *Brow = B + (size_t)k * n + jb;
                for (int i = ib; i < imax; i++) {
                    double a = A[(size_t)i * n + k];       /* the light   */
                    double *Crow = C + (size_t)i * n + jb; /* third canvas */
                    __m256d va = _mm256_set1_pd(a);
                    int j = 0;
                    for (; j < jw4; j += 4) {
                        __m256d vb = _mm256_loadu_pd(Brow + j);
                        __m256d vc = _mm256_loadu_pd(Crow + j);
                        vc = _mm256_fmadd_pd(va, vb, vc);   /* ash burns in */
                        _mm256_storeu_pd(Crow + j, vc);
                    }
                    for (; j < jw; j++)
                        Crow[j] += a * Brow[j];
                }
            }
            /* whole tile's ash has now settled; shade rolls to next tile */
        }
    }
}
