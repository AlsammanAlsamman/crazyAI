#include <stdlib.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define MATCH     1
#define MISMATCH -1
#define GAP      -2

/* Small-tray fallback: the worm is not worth waking for a tiny sand-tray.
   Plain two-row Needleman-Wunsch, stack only, no malloc, no SIMD setup. */
static int nw_small(int n, const char *a, const char *b)
{
    int prev[65], cur[65];
    for (int j = 0; j <= n; j++) prev[j] = GAP * j;
    for (int i = 1; i <= n; i++) {
        int ai = (unsigned char)a[i - 1];
        cur[0] = GAP * i;
        for (int j = 1; j <= n; j++) {
            int d = prev[j - 1] + (ai == (unsigned char)b[j - 1] ? MATCH : MISMATCH);
            int u = prev[j] + GAP;
            int l = cur[j - 1] + GAP;
            int best = d;
            if (u > best) best = u;
            if (l > best) best = l;
            cur[j] = best;
        }
        for (int j = 0; j <= n; j++) prev[j] = cur[j];
    }
    return prev[n];
}

int kernel(int n, const char *a, const char *b)
{
    if (n <= 0) return 0;
    if (n < 64) return nw_small(n, a, b);           /* guard: small-tray regime */

    /* ---- the worm's first straight crawl down the crossing-line of the two
            cords: how well do the symbols agree with no slip at all?        */
    int L0 = 0;
    for (int k = 0; k < n; k++)
        L0 += ((unsigned char)a[k] == (unsigned char)b[k]) ? MATCH : MISMATCH;

    /* provable furthest furrow any optimal path can reach (score <= n - 5d) */
    int D = (n - L0) / 5;
    if (D > n) D = n;                               /* degenerates to full tray */

    /* NEG: a mound so low no path through it can ever win.
       real scores >= -4n ; NEG-derived <= NEG + 2n = -6n-64 < -4n.          */
    const int NEG = -(8 * n + 64);

    const size_t pad = (size_t)n + 24;
    int *mem = (int *)malloc(3u * pad * sizeof(int));
    if (!mem) return nw_small(n < 64 ? n : 0, a, b); /* cannot happen for n>=64 */
    int *__restrict prev = mem;
    int *__restrict cur  = mem + pad;
    int *__restrict bi   = mem + 2u * pad;

    for (size_t t = 0; t < 2u * pad; t++) mem[t] = NEG;   /* whole tray flat-low */
    for (int k = 0; k < n; k++) bi[k] = (unsigned char)b[k];
    for (size_t k = (size_t)n; k < pad; k++) bi[k] = -1;  /* never matches */

    int hi0 = (D < n) ? D : n;
    for (int j = 0; j <= hi0; j++) prev[j] = GAP * j;     /* row 0, inside band */

    for (int i = 1; i <= n; i++) {
        int lo = i - D; if (lo < 0) lo = 0;
        int hi = i + D; if (hi > n) hi = n;

        int jstart, carry;                  /* carry lives in G-space: H + 2j */
        if (lo == 0) { cur[0] = GAP * i; jstart = 1; carry = GAP * i; }
        else         { jstart = lo;      carry = NEG; }

        int ai = (unsigned char)a[i - 1];
        int j  = jstart;

#if defined(__AVX2__)
        if (hi - jstart >= 7) {
            const __m256i ninf = _mm256_set1_epi32(-(1 << 30));
            const __m256i two  = _mm256_set1_epi32(2);
            const __m256i one  = _mm256_set1_epi32(1);
            const __m256i av   = _mm256_set1_epi32(ai);
            const __m256i s1   = _mm256_setr_epi32(0,0,1,2,3,4,5,6);
            const __m256i s2   = _mm256_setr_epi32(0,0,0,1,2,3,4,5);
            const __m256i s4   = _mm256_setr_epi32(0,0,0,0,0,1,2,3);
            const __m256i last = _mm256_set1_epi32(7);
            const __m256i step = _mm256_set1_epi32(16);
            __m256i offv = _mm256_setr_epi32(2*j,    2*j+2,  2*j+4,  2*j+6,
                                             2*j+8,  2*j+10, 2*j+12, 2*j+14);
            __m256i cbv  = _mm256_set1_epi32(carry);

            for (; j + 7 <= hi; j += 8) {
                /* eight winds in one motion: eight junctions settled at once */
                __m256i bv  = _mm256_loadu_si256((const __m256i *)(bi   + j - 1));
                __m256i pm1 = _mm256_loadu_si256((const __m256i *)(prev + j - 1));
                __m256i pj  = _mm256_loadu_si256((const __m256i *)(prev + j));
                __m256i eq  = _mm256_cmpeq_epi32(bv, av);
                __m256i sc  = _mm256_sub_epi32(_mm256_and_si256(eq, two), one);
                __m256i d   = _mm256_add_epi32(pm1, sc);      /* slant   */
                __m256i u   = _mm256_sub_epi32(pj,  two);      /* straight*/
                __m256i v   = _mm256_add_epi32(_mm256_max_epi32(d, u), offv);
                __m256i t;
                /* the worm curls across its own trail: strides 1, 2, 4.
                   Each curl re-tests every junction from a farther face;
                   after the third no junction can still be improved.       */
                t = _mm256_permutevar8x32_epi32(v, s1);
                t = _mm256_blend_epi32(t, ninf, 0x01);
                v = _mm256_max_epi32(v, t);
                t = _mm256_permutevar8x32_epi32(v, s2);
                t = _mm256_blend_epi32(t, ninf, 0x03);
                v = _mm256_max_epi32(v, t);
                t = _mm256_permutevar8x32_epi32(v, s4);
                t = _mm256_blend_epi32(t, ninf, 0x0F);
                v = _mm256_max_epi32(v, t);
                /* the one mound left standing behind the worm's tail */
                v   = _mm256_max_epi32(v, cbv);
                cbv = _mm256_permutevar8x32_epi32(v, last);
                _mm256_storeu_si256((__m256i *)(cur + j),
                                    _mm256_sub_epi32(v, offv));
                offv = _mm256_add_epi32(offv, step);
            }
            carry = _mm_cvtsi128_si32(_mm256_castsi256_si128(cbv));
        }
#endif
        /* tail of the band (and the whole row when no AVX2): the same
           free-sideways-slip coordinate, so still one max per junction. */
        for (; j <= hi; j++) {
            int d = prev[j - 1] + (ai == bi[j - 1] ? MATCH : MISMATCH);
            int u = prev[j] + GAP;
            int h = (d > u) ? d : u;
            int g = h + 2 * j;
            if (g > carry) carry = g;
            cur[j] = carry - 2 * j;
        }

        int *tmp = prev; prev = cur; cur = tmp;
    }

    int result = prev[n];
    free(mem);
    return result;
}
