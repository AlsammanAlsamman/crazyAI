#include <stdlib.h>
#include <immintrin.h>

#define MATCH 1
#define MISMATCH -1
#define GAP -2

int kernel(int n, const char *a, const char *b) {
    if (n == 0) return 0;

    char *rb = (char *)malloc((size_t)n);           /* reversed b: rb[k] = b[n-1-k] */
    for (int k = 0; k < n; k++) rb[k] = b[n - 1 - k];

    int *prev2 = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int *prev1 = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int *cur   = (int *)malloc((size_t)(n + 1) * sizeof(int));

    cur[0] = 0;                                       /* diagonal d = 0 */
    { int *t = prev1; prev1 = cur; cur = t; }

    const __m256i vmatch    = _mm256_set1_epi32(MATCH);
    const __m256i vmismatch = _mm256_set1_epi32(MISMATCH);
    const __m256i vgap      = _mm256_set1_epi32(GAP);

    for (int d = 1; d <= 2 * n; d++) {
        int imin = d - n; if (imin < 0) imin = 0;
        int imax = d < n ? d : n;
        int lo = imin, hi = imax;

        if (lo == 0)      { cur[0] = d * GAP; lo = 1; }      /* i=0 boundary  */
        if (hi == d)      { cur[d] = d * GAP; hi = d - 1; }  /* j=0 boundary  */

        int i = lo;
        for (; i + 7 <= hi; i += 8) {
            int abase  = i - 1;
            int rbbase = n - d + i;                          /* = n - j, contiguous */

            __m128i achar  = _mm_loadl_epi64((const __m128i *)&a[abase]);
            __m128i rbchar = _mm_loadl_epi64((const __m128i *)&rb[rbbase]);
            __m256i va  = _mm256_cvtepu8_epi32(achar);
            __m256i vrb = _mm256_cvtepu8_epi32(rbchar);

            __m256i eq = _mm256_cmpeq_epi32(va, vrb);
            __m256i sc = _mm256_blendv_epi8(vmismatch, vmatch, eq);

            __m256i vp2   = _mm256_loadu_si256((const __m256i *)&prev2[i - 1]);
            __m256i vdiag = _mm256_add_epi32(vp2, sc);
            __m256i vp1a  = _mm256_loadu_si256((const __m256i *)&prev1[i - 1]);
            __m256i vup   = _mm256_add_epi32(vp1a, vgap);
            __m256i vp1b  = _mm256_loadu_si256((const __m256i *)&prev1[i]);
            __m256i vleft = _mm256_add_epi32(vp1b, vgap);

            __m256i vmax = _mm256_max_epi32(vdiag, vup);
            vmax = _mm256_max_epi32(vmax, vleft);
            _mm256_storeu_si256((__m256i *)&cur[i], vmax);
        }
        for (; i <= hi; i++) {
            int j = d - i;
            int s = (a[i - 1] == b[j - 1]) ? MATCH : MISMATCH;
            int diag = prev2[i - 1] + s;
            int up   = prev1[i - 1] + GAP;
            int left = prev1[i] + GAP;
            int best = diag;
            if (up   > best) best = up;
            if (left > best) best = left;
            cur[i] = best;
        }

        int *t = prev2; prev2 = prev1; prev1 = cur; cur = t;
    }

    int result = prev1[n];
    free(prev2); free(prev1); free(cur); free(rb);
    return result;
}
