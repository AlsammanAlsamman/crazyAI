#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define MATCHS    1
#define MISMATCHS (-1)
#define GAPS      (-2)

/* ---- short cords: plain rolling-row exact DP (no sunlit-line setup) ---- */
static int nw_rows(int n, const char *a, const char *b)
{
    int *prev = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int *cur  = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int i, j, r;
    if (!prev || !cur) { free(prev); free(cur); return 0; }
    for (j = 0; j <= n; j++) prev[j] = GAPS * j;
    for (i = 1; i <= n; i++) {
        char ai = a[i - 1];
        cur[0] = GAPS * i;
        for (j = 1; j <= n; j++) {
            int d = prev[j - 1] + (ai == b[j - 1] ? MATCHS : MISMATCHS);
            int u = prev[j] + GAPS;
            int l = cur[j - 1] + GAPS;
            int best = d > u ? d : u;
            if (l > best) best = l;
            cur[j] = best;
        }
        { int *t = prev; prev = cur; cur = t; }
    }
    r = prev[n];
    free(prev); free(cur);
    return r;
}

/* ---- SEED 2: the straight crossing + EVERY single-slip crossing, O(n),   */
/*      scored with no DP cell in existence. Highest pile is brought up.    */
static int best_pile(int n, const char *a, const char *b)
{
    int i, m0 = 0, L;
    for (i = 0; i < n; i++) m0 += (a[i] == b[i]);
    L = 2 * m0 - n;                       /* straight crossing, door intact */
    if (n >= 2) {
        int e = 0, f = 0, g = 0;          /* E[q], F[q], G[q] */
        int rmF = 0, rmG = 0;             /* max_{p<=q} (E[p]-F[p]) / (E[p]-G[p]) */
        int bA = -1 << 28, bB = -1 << 28, q;
        for (q = 0; q < n; q++) {
            int eq1 = e + (a[q] == b[q]);          /* E[q+1] */
            int vA, vB;
            if (e - f > rmF) rmF = e - f;
            if (e - g > rmG) rmG = e - g;
            vA = (f - eq1) + rmF + m0;             /* matches, shadow row slips */
            vB = (g - eq1) + rmG + m0;             /* matches, near row slips   */
            if (vA > bA) bA = vA;
            if (vB > bB) bB = vB;
            e = eq1;
            if (q < n - 1) { f += (a[q] == b[q + 1]); g += (a[q + 1] == b[q]); }
        }
        if (2 * bA - n - 3 > L) L = 2 * bA - n - 3;   /* one broken door: -4 gaps, +1 column */
        if (2 * bB - n - 3 > L) L = 2 * bB - n - 3;
    }
    return L;
}

/* ---- SEED 1 (scalar): one sunlit line at a time, door-allowance w ---- */
static int wave_scalar(int n, const char *a, const char *b, int w)
{
    const int PAD = 64;
    const int sz = n + 1 + 2 * PAD;
    const int NEG = -(1 << 28);
    int *mem = (int *)malloc((size_t)3 * sz * sizeof(int));
    char *ab = (char *)malloc((size_t)n + 2 * PAD + 64);
    char *rb = (char *)malloc((size_t)n + 2 * PAD + 64);
    int *p2, *p1, *p0;
    int d, k, result = 0;
    if (!mem || !ab || !rb) { free(mem); free(ab); free(rb); return nw_rows(n, a, b); }
    for (k = 0; k < 3 * sz; k++) mem[k] = NEG;
    memset(ab, 0, (size_t)n + 2 * PAD + 64);
    memset(rb, 1, (size_t)n + 2 * PAD + 64);
    memcpy(ab, a, (size_t)n);
    for (k = 0; k < n; k++) rb[k] = b[n - 1 - k];   /* shadow row, laid reversed */
    p2 = mem + PAD; p1 = mem + sz + PAD; p0 = mem + 2 * sz + PAD;
    for (d = 0; d <= 2 * n; d++) {
        int i_min = (d > n) ? d - n : 0;
        int lo = (i_min > 1) ? i_min : 1;
        int hi = (d <= n) ? d - 1 : n;
        int l2 = (d - w >= 0) ? (d - w + 1) / 2 : 0;
        int h2 = (d + w) / 2;
        int nd = n - d, i;
        if (lo < l2) lo = l2;
        if (hi > h2) hi = h2;
#pragma omp simd
        for (i = lo; i <= hi; i++) {
            int sc = -1 + 2 * (ab[i - 1] == rb[nd + i]);   /* fire or no fire */
            int v  = p2[i - 1] + sc;                      /* straight step    */
            int u  = p1[i - 1];
            int l  = p1[i];
            int gg = (u > l ? u : l) + GAPS;              /* one coil crouches */
            p0[i] = v > gg ? v : gg;                      /* highest pile      */
        }
        for (k = 0; k < 32; k++) { p0[hi + 1 + k] = NEG; p0[lo - 1 - k] = NEG; }
        if (d <= n && d <= w) { p0[0] = GAPS * d; p0[d] = GAPS * d; }
        if (d == 2 * n) { result = p0[n]; break; }
        { int *t = p2; p2 = p1; p1 = p0; p0 = t; }
    }
    free(mem); free(ab); free(rb);
    return result;
}

#if defined(__AVX2__)
/* ---- SEED 1 (16 coils per glance): the sunlit line, int16 fires ---- */
static int wave_avx2(int n, const char *a, const char *b, int w)
{
    const int PAD = 64;
    const int sz = n + 1 + 2 * PAD;
    const int16_t NEG = -30000;
    int16_t *mem = (int16_t *)malloc((size_t)3 * sz * sizeof(int16_t));
    char *ab = (char *)malloc((size_t)n + 2 * PAD + 64);
    char *rb = (char *)malloc((size_t)n + 2 * PAD + 64);
    int16_t *p2, *p1, *p0;
    const __m256i vgap = _mm256_set1_epi16(2);
    const __m256i vm1  = _mm256_set1_epi16(-1);
    int d, k, result = 0;
    if (!mem || !ab || !rb) { free(mem); free(ab); free(rb); return nw_rows(n, a, b); }
    for (k = 0; k < 3 * sz; k++) mem[k] = NEG;
    memset(ab, 0, (size_t)n + 2 * PAD + 64);
    memset(rb, 1, (size_t)n + 2 * PAD + 64);   /* padding can never strike a fire */
    memcpy(ab, a, (size_t)n);
    for (k = 0; k < n; k++) rb[k] = b[n - 1 - k];
    p2 = mem + PAD; p1 = mem + sz + PAD; p0 = mem + 2 * sz + PAD;
    for (d = 0; d <= 2 * n; d++) {
        int i_min = (d > n) ? d - n : 0;
        int lo = (i_min > 1) ? i_min : 1;
        int hi = (d <= n) ? d - 1 : n;
        int l2 = (d - w >= 0) ? (d - w + 1) / 2 : 0;
        int h2 = (d + w) / 2;
        int nd = n - d, i;
        if (lo < l2) lo = l2;
        if (hi > h2) hi = h2;
        for (i = lo; i <= hi; i += 16) {
            __m128i ca = _mm_loadu_si128((const __m128i *)(ab + i - 1));
            __m128i cb = _mm_loadu_si128((const __m128i *)(rb + nd + i));
            __m256i msk = _mm256_cvtepi8_epi16(_mm_cmpeq_epi8(ca, cb));
            __m256i s   = _mm256_sub_epi16(vm1, _mm256_add_epi16(msk, msk));
            __m256i vd  = _mm256_add_epi16(
                              _mm256_loadu_si256((const __m256i *)(p2 + i - 1)), s);
            __m256i vu  = _mm256_loadu_si256((const __m256i *)(p1 + i - 1));
            __m256i vl  = _mm256_loadu_si256((const __m256i *)(p1 + i));
            __m256i vg  = _mm256_sub_epi16(_mm256_max_epi16(vu, vl), vgap);
            _mm256_storeu_si256((__m256i *)(p0 + i), _mm256_max_epi16(vd, vg));
        }
        for (k = 0; k < 32; k++) { p0[hi + 1 + k] = NEG; p0[lo - 1 - k] = NEG; }
        if (d <= n && d <= w) { p0[0] = (int16_t)(GAPS * d); p0[d] = (int16_t)(GAPS * d); }
        if (d == 2 * n) { result = p0[n]; break; }
        { int16_t *t = p2; p2 = p1; p1 = p0; p0 = t; }
    }
    free(mem); free(ab); free(rb);
    return result;
}
#endif

int kernel(int n, const char *a, const char *b)
{
    int L, w;
    if (n <= 0) return 0;
    if (n < 48) return nw_rows(n, a, b);          /* guard: short cord */
    L = best_pile(n, a, b);                        /* SEED 2 + SEED 3 rooms */
    w = (n - L) / 5;                               /* door-allowance (certified) */
    if (w < 0) w = 0;
    if (w > n) w = n;                              /* guard: full crossing */
#if defined(__AVX2__)
    if (n <= 12000) return wave_avx2(n, a, b, w);  /* guard: int16 range */
#endif
    return wave_scalar(n, a, b, w);
}
