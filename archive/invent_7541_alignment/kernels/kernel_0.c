#include <stdlib.h>
#include <string.h>
#include <immintrin.h>

#define MATCH 1
#define MISMATCH -1
#define GAP -2

int kernel(int n, const char *a, const char *b) {
    if (n == 0) return 0;

    /* lay the strings along their edges: a stays natural order,
       b is laid down reversed so a short-diagonal walk becomes a
       straight, contiguous walk through it. */
    int *ai = (int *)malloc((size_t)n * sizeof(int));
    int *brev = (int *)malloc((size_t)n * sizeof(int));
    for (int k = 0; k < n; k++) {
        ai[k] = (unsigned char)a[k];
        brev[k] = (unsigned char)b[n - 1 - k];
    }

    /* thin double row of fire: only two diagonals behind the
       crawl (plus the one being kindled) are ever kept alight. */
    const int pad = 8;
    size_t bufn = (size_t)(n + 1 + pad);
    int *prev2 = (int *)calloc(bufn, sizeof(int));
    int *prev1 = (int *)calloc(bufn, sizeof(int));
    int *curr  = (int *)calloc(bufn, sizeof(int));

    const __m256i vmatch    = _mm256_set1_epi32(MATCH);
    const __m256i vmismatch = _mm256_set1_epi32(MISMATCH);
    const __m256i vgap      = _mm256_set1_epi32(GAP);

    for (int d = 0; d <= 2 * n; d++) {
        if (d <= n) {
            curr[0] = d * GAP;   /* i == 0, the plane still bare on top */
            curr[d] = d * GAP;   /* j == 0, the plane still bare on left */
        }

        int lo = d - n; if (lo < 1) lo = 1;
        int hi = d - 1; if (hi > n) hi = n;
        if (lo > hi) {
            int *t = prev2; prev2 = prev1; prev1 = curr; curr = t;
            continue;
        }

        int off = n - d;   /* brev index = off + i, contiguous as i rises */
        int i = lo;
        int end8 = hi - 7;
        for (; i <= end8; i += 8) {
            __m256i va = _mm256_loadu_si256((const __m256i *)&ai[i - 1]);
            __m256i vb = _mm256_loadu_si256((const __m256i *)&brev[off + i]);
            __m256i eq = _mm256_cmpeq_epi32(va, vb);
            __m256i sc = _mm256_blendv_epi8(vmismatch, vmatch, eq);

            __m256i vdiag = _mm256_loadu_si256((const __m256i *)&prev2[i - 1]);
            __m256i vup   = _mm256_loadu_si256((const __m256i *)&prev1[i - 1]);
            __m256i vleft = _mm256_loadu_si256((const __m256i *)&prev1[i]);

            __m256i diagv = _mm256_add_epi32(vdiag, sc);
            __m256i upv   = _mm256_add_epi32(vup, vgap);
            __m256i leftv = _mm256_add_epi32(vleft, vgap);

            __m256i best = _mm256_max_epi32(diagv, _mm256_max_epi32(upv, leftv));
            _mm256_storeu_si256((__m256i *)&curr[i], best);
        }
        for (; i <= hi; i++) {
            int j = d - i;
            int diagv = prev2[i - 1] + (a[i - 1] == b[j - 1] ? MATCH : MISMATCH);
            int upv = prev1[i - 1] + GAP;
            int leftv = prev1[i] + GAP;
            int best = diagv;
            if (upv > best) best = upv;
            if (leftv > best) best = leftv;
            curr[i] = best;
        }

        int *t = prev2; prev2 = prev1; prev1 = curr; curr = t;
    }

    int result = prev1[n];   /* the one ember left at the far corner */

    free(ai); free(brev);
    free(prev2); free(prev1); free(curr);
    return result;
}
