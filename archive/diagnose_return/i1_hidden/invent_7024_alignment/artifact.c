/* Needleman-Wunsch score, match +1 / mismatch -1 / gap -2, |a| = |b| = n.
 *
 * Phase A ("the sunlit walk"): the maximum over EVERY crossing whose door
 *   breaks at most once (all alignments with <=1 gap-pair) is computed exactly
 *   in O(n) by three prefix walks plus a running maximum.  No matrix, no rooms.
 * Phase B ("the doors that may break W times"): score <= n - 5g proves that the
 *   optimum uses g <= W = (n - L)/5 gap-pairs, hence stays inside |i-j| <= W.
 *   That band is swept exactly with an anti-diagonal AVX2 wavefront (16 rooms
 *   per step, zero serial dependence inside a wavefront).
 * Result is bit-exact Needleman-Wunsch for every input.
 */
#include <stdlib.h>
#include <stdint.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* ---- guarded fallback: plain exact DP (small n, huge n, no memory) -------- */
static int nw_scalar_full(int n, const char *a, const char *b)
{
    int sbuf[260];
    int *prev, *cur, *heap = 0;
    int i, j, r;

    if (n <= 0) return 0;
    if (n + 1 <= 130) { prev = sbuf; cur = sbuf + 130; }
    else {
        heap = (int *)malloc(sizeof(int) * 2u * (size_t)(n + 1));
        if (!heap) return 0;
        prev = heap; cur = heap + (n + 1);
    }
    for (j = 0; j <= n; j++) prev[j] = -2 * j;
    for (i = 1; i <= n; i++) {
        char ai = a[i - 1];
        cur[0] = -2 * i;
        for (j = 1; j <= n; j++) {
            int v = prev[j - 1] + ((ai == b[j - 1]) ? 1 : -1);
            int u = prev[j] - 2;
            int l = cur[j - 1] - 2;
            if (u > v) v = u;
            if (l > v) v = l;
            cur[j] = v;
        }
        { int *t = prev; prev = cur; cur = t; }
    }
    r = prev[n];
    if (heap) free(heap);
    return r;
}

/* ---- Phase B: exact DP restricted to |i-j| <= W, anti-diagonal wavefront -- */
static int nw_band(int n, const unsigned char *pa, const unsigned char *pbr,
                   int W, int16_t *A0, int16_t *A1, int16_t *A2, int NEG)
{
    int16_t *r2 = A0, *r1 = A1, *cur = A2;
    int d, i, lo, hi, i0, i1;

    if (W > n) W = n;
    for (i = -1; i <= n + 1; i++) { A0[i] = (int16_t)NEG; A1[i] = (int16_t)NEG; A2[i] = (int16_t)NEG; }

    for (d = 0; d <= 2 * n; d++) {
        lo = 0;
        if (d - n > lo) lo = d - n;
        { int c = (d - W + 1) >> 1; if (c > lo) lo = c; }
        hi = n; if (d < hi) hi = d;
        { int f = (d + W) >> 1; if (f < hi) hi = f; }

        cur[lo - 1] = (int16_t)NEG;      /* the tower: out-of-band is unreadable */
        cur[hi + 1] = (int16_t)NEG;

        if (lo <= hi) {
            if (lo == 0) cur[0] = (int16_t)(-2 * d);   /* H(0,d) */
            if (hi == d) cur[d] = (int16_t)(-2 * d);   /* H(d,0) */
            i0 = (lo > 1) ? lo : 1;
            i1 = (hi < d - 1) ? hi : (d - 1);
            i  = i0;
#if defined(__AVX2__)
            {
                const __m256i c2   = _mm256_set1_epi16(2);
                const __m256i c1   = _mm256_set1_epi16(1);
                const __m256i zero = _mm256_setzero_si256();
                int base = n - d;
                for (; i + 15 <= i1; i += 16) {
                    __m128i va = _mm_loadu_si128((const __m128i *)(pa + (i - 1)));
                    __m128i vb = _mm_loadu_si128((const __m128i *)(pbr + (base + i)));
                    __m128i eq = _mm_cmpeq_epi8(va, vb);              /* -1 / 0 */
                    __m256i m  = _mm256_cvtepi8_epi16(eq);
                    /* s = -(2m+1):  match(-1)->+1,  mismatch(0)->-1  */
                    __m256i s  = _mm256_sub_epi16(zero,
                                   _mm256_add_epi16(_mm256_add_epi16(m, m), c1));
                    __m256i vd = _mm256_add_epi16(
                                   _mm256_loadu_si256((const __m256i *)(r2 + i - 1)), s);
                    __m256i vu = _mm256_sub_epi16(
                                   _mm256_loadu_si256((const __m256i *)(r1 + i - 1)), c2);
                    __m256i vl = _mm256_sub_epi16(
                                   _mm256_loadu_si256((const __m256i *)(r1 + i)), c2);
                    _mm256_storeu_si256((__m256i *)(cur + i),
                                   _mm256_max_epi16(_mm256_max_epi16(vd, vu), vl));
                }
            }
#endif
            for (; i <= i1; i++) {
                int s = (pa[i - 1] == pbr[n - d + i]) ? 1 : -1;
                int v = (int)r2[i - 1] + s;
                int u = (int)r1[i - 1] - 2;
                int l = (int)r1[i] - 2;
                if (u > v) v = u;
                if (l > v) v = l;
                cur[i] = (int16_t)v;
            }
        }
        { int16_t *t = r2; r2 = r1; r1 = cur; cur = t; }
    }
    return (int)r1[n];                    /* the queen's threshold: H(n,n) */
}

/* -------------------------------------------------------------------------- */
int kernel(int n, const char *a, const char *b)
{
    int *U, *Vp, *Vm;
    int t, run, bestA, bestB, L, W, Wb, W2, res;
    int NEG, row;
    unsigned char *pa = 0, *pbr = 0;
    int16_t *mem = 0, *A0, *A1, *A2;
    int *blk;

    if (n <= 0) return 0;
    /* GUARD: tiny n - setup would dominate, run the plain DP. */
    if (n < 96) return nw_scalar_full(n, a, b);

    blk = (int *)malloc(sizeof(int) * (3u * (size_t)(n + 4)));
    if (!blk) return nw_scalar_full(n, a, b);
    U = blk; Vp = U + (n + 4); Vm = Vp + (n + 4);

    /* ---- Phase A: the three walks on the sunlit water table (O(n)) ------- */
    U[0] = 0;
    for (t = 0; t < n; t++) U[t + 1] = U[t] + ((a[t] == b[t]) ? 1 : -1);
    Vp[0] = 0;
    for (t = 0; t + 1 < n; t++) Vp[t + 1] = Vp[t] + ((a[t] == b[t + 1]) ? 1 : -1);
    Vm[1] = 0;
    for (t = 1; t < n; t++) Vm[t + 1] = Vm[t] + ((a[t] == b[t - 1]) ? 1 : -1);

    /* every crossing with one broken door, both crouch directions, O(n) */
    bestA = -1000000000; run = -1000000000;
    for (t = 0; t <= n - 1; t++) {
        int c = U[t] - Vm[t + 1];
        if (c > run) run = c;
        { int v = run + (Vm[t + 1] - U[t + 1]); if (v > bestA) bestA = v; }
    }
    bestB = -1000000000; run = -1000000000;
    for (t = 0; t <= n - 1; t++) {
        int c = U[t] - Vp[t];
        if (c > run) run = c;
        { int v = run + (Vp[t] - U[t + 1]); if (v > bestB) bestB = v; }
    }
    L = U[n];
    if (U[n] - 4 + bestA > L) L = U[n] - 4 + bestA;
    if (U[n] - 4 + bestB > L) L = U[n] - 4 + bestB;
    free(blk);

    /* ---- the door-count bound: score <= n - 5g -------------------------- */
    W = (n - L) / 5;
    if (W <= 1) return L;                 /* proven exact with no matrix at all */
    if (W > n) W = n;

    /* GUARD: int16 rooms must hold every in-band value (>= -(n+2W)). */
    if (n + 2 * W + 20 > 32767) return nw_scalar_full(n, a, b);

    row = n + 68;
    pa  = (unsigned char *)malloc((size_t)n + 64);
    pbr = (unsigned char *)malloc((size_t)n + 64);
    mem = (int16_t *)malloc(sizeof(int16_t) * (size_t)(3 * row));
    if (!pa || !pbr || !mem) {
        free(pa); free(pbr); free(mem);
        return nw_scalar_full(n, a, b);
    }
    for (t = 0; t < n; t++) pa[t]  = (unsigned char)a[t];
    for (t = 0; t < n; t++) pbr[t] = (unsigned char)b[n - 1 - t];   /* the crouch */
    for (t = n; t < n + 64; t++) { pa[t] = 0xFE; pbr[t] = 0xFD; }

    A0 = mem + 1; A1 = mem + row + 1; A2 = mem + 2 * row + 1;
    NEG = -(n + 2 * W + 16);

    /* cheap narrow sweep first: it can only raise L, hence only shrink W */
    Wb = (W < 32) ? W : 32;
    res = nw_band(n, pa, pbr, Wb, A0, A1, A2, NEG);
    if (res > L) L = res;
    W2 = (n - L) / 5;
    if (W2 > Wb) {
        if (W2 > n) W2 = n;
        res = nw_band(n, pa, pbr, W2, A0, A1, A2, NEG);
        if (res > L) L = res;
    }
    free(pa); free(pbr); free(mem);
    return L;
}
