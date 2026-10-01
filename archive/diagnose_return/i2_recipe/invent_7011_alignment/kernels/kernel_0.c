/* Anti-diagonal "crease" Needleman-Wunsch, min-cost gauge.
   int kernel(int n, const char *a, const char *b);  match +1, mismatch -1, gap -2.

   Gauge:  E[i][j] = (i+j) - 2*H[i][j]   =>  the recipe's MIN recurrence with
           "one disagreement" = 4, "one slip" = 5, E[0][0] = 0,
           and step 12:  H = n - E/2.                                        */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define DISAGREE 4
#define SLIP     5

/* ---------------- scalar fold (no AVX2) ---------------- */
static int fold_s(int n, const unsigned char *pa, const unsigned char *rb,
                  int32_t *c0, int32_t *c1, int32_t *c2)
{
    const int32_t INF = 1 << 24;
    int d, i;
    /* step 3: light the crease whose sum is nothing at all */
    c0[0] = 0; c0[1] = INF;
    /* step 4: a bare slip-in and a bare slip-out */
    c1[0] = SLIP; c1[1] = SLIP; c1[2] = INF;

    for (d = 2; d <= 2 * n; d++) {                 /* step 9: crease after crease */
        int lo = d - n; if (lo < 0) lo = 0;
        int hi = (d < n) ? d : n;
        for (i = lo; i <= hi; i++) {
            /* step 5: the two symbols choose the cell's colour */
            int32_t cost = (pa[i - 1] == rb[n - d + i]) ? 0 : DISAGREE;
            /* step 6: read through the fold at the three layers beneath */
            int32_t dg = c0[i - 1] + cost;          /* left hand : agree straight through */
            int32_t up = c1[i - 1] + SLIP;          /* right hand: slip the down-bough    */
            int32_t lf = c1[i]     + SLIP;          /* right hand: slip the across-bough  */
            /* step 7: take whichever of those three sums is least */
            int32_t m = dg < up ? dg : up; if (lf < m) m = lf;
            c2[i] = m;
        }
        c2[hi + 1] = INF;                           /* step 7: off the edge = unreachable */
        /* step 8: let the left hand go slack; right -> left, new -> right */
        { int32_t *t = c0; c0 = c1; c1 = c2; c2 = t; }
    }
    /* step 10/11: the crease just lit holds exactly one cell */
    return (int)c1[n];
}

#if defined(__AVX2__)
/* ---------------- 32-bit fold, 8 cells of a slant at once ---------------- */
static int fold32(int n, const unsigned char *pa, const unsigned char *rb,
                  int32_t *c0, int32_t *c1, int32_t *c2)
{
    const int32_t INF = 1 << 24;
    const __m256i vmis = _mm256_set1_epi32(DISAGREE);
    const __m256i vsl  = _mm256_set1_epi32(SLIP);
    int d, i;
    c0[0] = 0; c0[1] = INF;                         /* step 3 */
    c1[0] = SLIP; c1[1] = SLIP; c1[2] = INF;        /* step 4 */

    for (d = 2; d <= 2 * n; d++) {                  /* step 9 */
        int lo = d - n; if (lo < 0) lo = 0;
        int hi = (d < n) ? d : n;
        int off = n - d;
        for (i = lo; i <= hi; i += 8) {
            /* step 5 */
            __m128i sa = _mm_loadl_epi64((const __m128i *)(pa + i - 1));
            __m128i sb = _mm_loadl_epi64((const __m128i *)(rb + off + i));
            __m256i eq = _mm256_cvtepi8_epi32(_mm_cmpeq_epi8(sa, sb));
            __m256i cost = _mm256_andnot_si256(eq, vmis);   /* equal -> 0, differ -> 4 */
            /* step 6 */
            __m256i dg = _mm256_add_epi32(
                             _mm256_loadu_si256((const __m256i *)(c0 + i - 1)), cost);
            __m256i up = _mm256_loadu_si256((const __m256i *)(c1 + i - 1));
            __m256i lf = _mm256_loadu_si256((const __m256i *)(c1 + i));
            __m256i sl = _mm256_add_epi32(_mm256_min_epi32(up, lf), vsl);
            /* step 7 */
            _mm256_storeu_si256((__m256i *)(c2 + i), _mm256_min_epi32(dg, sl));
        }
        c2[hi + 1] = INF;
        { int32_t *t = c0; c0 = c1; c1 = c2; c2 = t; }   /* step 8 */
    }
    return (int)c1[n];                              /* step 10/11 */
}

/* ---------------- 16-bit fold, 16 cells of a slant at once ----------------
   Legal whenever 5n <= 30000: every reachable spark obeys E <= 5n.
   Saturating adds keep the off-lattice garbage lanes pinned above INF.     */
static int fold16(int n, const unsigned char *pa, const unsigned char *rb,
                  int16_t *c0, int16_t *c1, int16_t *c2)
{
    const int16_t INF = 32000;
    const __m256i vmis = _mm256_set1_epi16(DISAGREE);
    const __m256i vsl  = _mm256_set1_epi16(SLIP);
    int d, i;
    c0[0] = 0; c0[1] = INF;                         /* step 3 */
    c1[0] = SLIP; c1[1] = SLIP; c1[2] = INF;        /* step 4 */

    for (d = 2; d <= 2 * n; d++) {                  /* step 9 */
        int lo = d - n; if (lo < 0) lo = 0;
        int hi = (d < n) ? d : n;
        int off = n - d;
        for (i = lo; i <= hi; i += 16) {
            /* step 5 */
            __m128i sa = _mm_loadu_si128((const __m128i *)(pa + i - 1));
            __m128i sb = _mm_loadu_si128((const __m128i *)(rb + off + i));
            __m256i eq = _mm256_cvtepi8_epi16(_mm_cmpeq_epi8(sa, sb));
            __m256i cost = _mm256_andnot_si256(eq, vmis);
            /* step 6 */
            __m256i dg = _mm256_adds_epi16(
                             _mm256_loadu_si256((const __m256i *)(c0 + i - 1)), cost);
            __m256i up = _mm256_loadu_si256((const __m256i *)(c1 + i - 1));
            __m256i lf = _mm256_loadu_si256((const __m256i *)(c1 + i));
            __m256i sl = _mm256_adds_epi16(_mm256_min_epi16(up, lf), vsl);
            /* step 7 */
            _mm256_storeu_si256((__m256i *)(c2 + i), _mm256_min_epi16(dg, sl));
        }
        c2[hi + 1] = INF;
        { int16_t *t = c0; c0 = c1; c1 = c2; c2 = t; }   /* step 8 */
    }
    return (int)c1[n];                              /* step 10/11 */
}
#endif /* __AVX2__ */

int kernel(int n, const char *a, const char *b)
{
    if (n <= 0) return 0;

    /* ---- step 1: hang the two boughs from their rows of lamp-posts.
       The across-bough is hung in reverse, because walking a slant with the
       down-place rising makes the across-place fall; reversed once here, both
       boughs are then read forward by every crease.                        */
    const size_t cs   = (size_t)n + 64;                 /* int32 slots per crease */
    const size_t sreg = (size_t)n + 64;                 /* bytes per bough copy   */
    const size_t bytes = 3 * cs * sizeof(int32_t) + 2 * sreg;

    unsigned char stackmem[32768];
    unsigned char *mem; int heap = 0;
    if (bytes <= sizeof(stackmem)) mem = stackmem;
    else { mem = (unsigned char *)malloc(bytes); if (!mem) return 0; heap = 1; }

    int32_t *cb = (int32_t *)mem;
    unsigned char *sa = mem + 3 * cs * sizeof(int32_t);
    unsigned char *sb = sa + sreg;

    unsigned char *pa = sa + 8;                          /* pa[-1] must be readable */
    memcpy(pa, a, (size_t)n);
    memset(pa + n, 0x7f, sreg - 8 - (size_t)n);          /* filler, matches nothing */
    pa[-1] = 0x7e;
    { int k; for (k = 0; k < n; k++) sb[k] = (unsigned char)b[n - 1 - k]; }
    memset(sb + n, 0x7d, sreg - (size_t)n);
    unsigned char *rb = sb;

    /* ---- step 2: number each cell by down-place + across-place; cells of one
       sum are one crease.  Three creases exist at a time and nothing else:
       the sheet is never allocated.  Every slot starts unreachable.        */
    int score;
#if defined(__AVX2__)
    if (5 * (long)n <= 30000) {
        int16_t *h = (int16_t *)cb;
        size_t hs = 2 * cs;                              /* int16 slots per crease */
        int16_t *c0 = h + 8, *c1 = h + hs + 8, *c2 = h + 2 * hs + 8;
        size_t k; for (k = 0; k < 3 * hs; k++) h[k] = 32000;
        /* steps 3-11 */
        score = n - fold16(n, pa, rb, c0, c1, c2) / 2;   /* step 12 */
    } else {
        int32_t *c0 = cb + 8, *c1 = cb + cs + 8, *c2 = cb + 2 * cs + 8;
        size_t k; for (k = 0; k < 3 * cs; k++) cb[k] = 1 << 24;
        score = n - fold32(n, pa, rb, c0, c1, c2) / 2;   /* step 12 */
    }
#else
    {
        int32_t *c0 = cb + 8, *c1 = cb + cs + 8, *c2 = cb + 2 * cs + 8;
        size_t k; for (k = 0; k < 3 * cs; k++) cb[k] = 1 << 24;
        score = n - fold_s(n, pa, rb, c0, c1, c2) / 2;   /* step 12 */
    }
#endif

    if (heap) free(mem);
    return score;   /* step 12: the agreement told as a number */
}
