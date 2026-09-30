#include <stdlib.h>
#include <stddef.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define ALN_M   1
#define ALN_X  (-1)
#define ALN_G  (-2)
#define ALN_POISON (-(1 << 28))

/* exact two-row Needleman-Wunsch: small-n path and safety fallback */
static int aln_rows(int n, const char *restrict a, const char *restrict b)
{
    int sp[64], sc[64];
    int *prev, *cur, *heap = NULL;
    if (n + 1 <= 64) { prev = sp; cur = sc; }
    else {
        heap = (int *)malloc(2u * (size_t)(n + 1) * sizeof(int));
        if (!heap) return 0;
        prev = heap; cur = heap + (n + 1);
    }
    for (int j = 0; j <= n; j++) prev[j] = j * ALN_G;
    for (int i = 1; i <= n; i++) {
        char ai = a[i - 1];
        cur[0] = i * ALN_G;
        for (int j = 1; j <= n; j++) {
            int d = prev[j - 1] + (ai == b[j - 1] ? ALN_M : ALN_X);
            int u = prev[j] + ALN_G;
            int l = cur[j - 1] + ALN_G;
            int m = d > u ? d : u;
            cur[j] = m > l ? m : l;
        }
        { int *t = prev; prev = cur; cur = t; }
    }
    { int r = prev[n]; if (heap) free(heap); return r; }
}

int kernel(int n, const char *a, const char *b)
{
    const char *restrict A = a;
    const char *restrict B = b;

    if (n <= 0) return 0;
    if (n < 64) return aln_rows(n, A, B);   /* guard: no setup overhead on small n */

    /* ---- phase 1: the crossings.  Walk the cords, count the fires. ----
       straight crossing, and every single-slip crossing in both lay orders.
       Each finished crossing is a real alignment, so the highest pile is a
       certified lower bound on the score.                                  */
    {
        int lb, W;
        int m0 = 0;
        for (int i = 0; i < n; i++) m0 += (A[i] == B[i]);
        lb = 2 * m0 - n;                                  /* no door broken */

        {   /* shadow row = B: B crouches past the quarrel at k  */
            int s = 0, bs;
            for (int i = 0; i + 1 < n; i++) s += (A[i] == B[i + 1]);
            bs = s;
            for (int k = 1; k < n; k++) {
                s += (A[k - 1] == B[k - 1]) - (A[k - 1] == B[k]);
                if (s > bs) bs = s;
            }
            { int c = 2 * bs - n - 3; if (c > lb) lb = c; }
        }
        {   /* the cords laid in the other order */
            int s = 0, bs;
            for (int i = 0; i + 1 < n; i++) s += (A[i + 1] == B[i]);
            bs = s;
            for (int k = 1; k < n; k++) {
                s += (A[k - 1] == B[k - 1]) - (A[k] == B[k - 1]);
                if (s > bs) bs = s;
            }
            { int c = 2 * bs - n - 3; if (c > lb) lb = c; }
        }

        /* ---- phase 2: how far may the door ever travel? ----
           any alignment with g gap-pairs scores <= n - 5g and never leaves
           offset g, so an optimal path obeys |i-j| <= (n - lb)/5.          */
        W = (n - lb) / 5;
        if (W < 1) W = 1;
        if (W > n) W = n;

        {   /* ---- phase 3: walk the banded wavefront ---- */
            size_t cap = (size_t)n + 8;
            int *buf = (int *)malloc(3u * cap * sizeof(int));
            char *br = (char *)malloc((size_t)n + 24);
            int *c0, *c1, *c2, res;
            if (!buf || !br) { free(buf); free(br); return aln_rows(n, A, B); }

            for (int i = 0; i < n; i++) br[i] = B[n - 1 - i];
            for (int i = 0; i < 24; i++) br[n + i] = 0;
            for (size_t i = 0; i < 3u * cap; i++) buf[i] = ALN_POISON;

            c0 = buf + 2;                 /* wavefront k-2, index = row i   */
            c1 = buf + cap + 2;           /* wavefront k-1                  */
            c2 = buf + 2 * cap + 2;       /* wavefront k, being filled      */
            c1[0] = 0;                    /* dp[0][0]                       */

            for (int k = 1; k <= 2 * n; k++) {
                int t  = k - W; if (t < 0) t = 0;
                int lo = (t + 1) >> 1;                 /* ceil((k-W)/2)     */
                int hi = (k + W) >> 1;                 /* floor((k+W)/2)    */
                int ilo, ihi, off, i;
                const int *restrict p0;
                const int *restrict p1;
                int *restrict p2;

                if (k - n > lo) lo = k - n;
                if (hi > n) hi = n;
                if (hi > k) hi = k;

                ilo = lo > 1 ? lo : 1;
                ihi = hi < k - 1 ? hi : k - 1;
                off = n - k;
                p0 = c0; p1 = c1; p2 = c2;
                i = ilo;
#if defined(__AVX2__)
                {
                    const __m256i vg  = _mm256_set1_epi32(ALN_G);
                    const __m256i vm1 = _mm256_set1_epi32(-1);
                    for (; i + 15 <= ihi; i += 16) {
                        __m128i ca = _mm_loadu_si128((const __m128i *)(A + i - 1));
                        __m128i cb = _mm_loadu_si128((const __m128i *)(br + off + i));
                        __m128i eq = _mm_cmpeq_epi8(ca, cb);
                        __m256i e0 = _mm256_cvtepi8_epi32(eq);
                        __m256i e1 = _mm256_cvtepi8_epi32(_mm_srli_si128(eq, 8));
                        __m256i s0 = _mm256_sub_epi32(vm1, _mm256_add_epi32(e0, e0));
                        __m256i s1 = _mm256_sub_epi32(vm1, _mm256_add_epi32(e1, e1));
                        __m256i d0 = _mm256_add_epi32(_mm256_loadu_si256((const __m256i *)(p0 + i - 1)), s0);
                        __m256i d1 = _mm256_add_epi32(_mm256_loadu_si256((const __m256i *)(p0 + i + 7)), s1);
                        __m256i u0 = _mm256_add_epi32(_mm256_loadu_si256((const __m256i *)(p1 + i - 1)), vg);
                        __m256i u1 = _mm256_add_epi32(_mm256_loadu_si256((const __m256i *)(p1 + i + 7)), vg);
                        __m256i l0 = _mm256_add_epi32(_mm256_loadu_si256((const __m256i *)(p1 + i)), vg);
                        __m256i l1 = _mm256_add_epi32(_mm256_loadu_si256((const __m256i *)(p1 + i + 8)), vg);
                        _mm256_storeu_si256((__m256i *)(p2 + i),
                            _mm256_max_epi32(_mm256_max_epi32(d0, u0), l0));
                        _mm256_storeu_si256((__m256i *)(p2 + i + 8),
                            _mm256_max_epi32(_mm256_max_epi32(d1, u1), l1));
                    }
                    for (; i + 7 <= ihi; i += 8) {
                        __m128i ca = _mm_loadl_epi64((const __m128i *)(A + i - 1));
                        __m128i cb = _mm_loadl_epi64((const __m128i *)(br + off + i));
                        __m128i eq = _mm_cmpeq_epi8(ca, cb);
                        __m256i e0 = _mm256_cvtepi8_epi32(eq);
                        __m256i s0 = _mm256_sub_epi32(vm1, _mm256_add_epi32(e0, e0));
                        __m256i d0 = _mm256_add_epi32(_mm256_loadu_si256((const __m256i *)(p0 + i - 1)), s0);
                        __m256i u0 = _mm256_add_epi32(_mm256_loadu_si256((const __m256i *)(p1 + i - 1)), vg);
                        __m256i l0 = _mm256_add_epi32(_mm256_loadu_si256((const __m256i *)(p1 + i)), vg);
                        _mm256_storeu_si256((__m256i *)(p2 + i),
                            _mm256_max_epi32(_mm256_max_epi32(d0, u0), l0));
                    }
                }
#endif
                for (; i <= ihi; i++) {
                    int e = (A[i - 1] == br[off + i]);
                    int d = p0[i - 1] + (e ? ALN_M : ALN_X);
                    int u = p1[i - 1] + ALN_G;
                    int l = p1[i] + ALN_G;
                    int m = d > u ? d : u;
                    p2[i] = m > l ? m : l;
                }
                if (lo == 0) p2[0] = k * ALN_G;   /* dp[0][k] */
                if (hi == k) p2[k] = k * ALN_G;   /* dp[k][0] */
                p2[lo - 1] = ALN_POISON;          /* the tower: unread cells */
                p2[hi + 1] = ALN_POISON;

                { int *tmp = c0; c0 = c1; c1 = c2; c2 = tmp; }
            }
            res = c1[n];
            free(buf); free(br);
            return res;
        }
    }
}
