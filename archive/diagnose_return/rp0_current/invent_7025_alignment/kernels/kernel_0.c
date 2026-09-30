#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define MATCH     1
#define MISMATCH -1
#define GAP      -2
#define PAD      80

/* ---------------- guarded fallback: compact exact two-row NW ---------------- */
static int nw_tworow(int n, const char *restrict a, const char *restrict b)
{
    int *buf = (int *)malloc(2u * (size_t)(n + 1) * sizeof(int));
    if (!buf) return 0;
    int *prev = buf, *cur = buf + (n + 1);
    for (int j = 0; j <= n; j++) prev[j] = GAP * j;
    for (int i = 1; i <= n; i++) {
        char ai = a[i - 1];
        cur[0] = GAP * i;
        for (int j = 1; j <= n; j++) {
            int dg = prev[j - 1] + (ai == b[j - 1] ? MATCH : MISMATCH);
            int up = prev[j] + GAP;
            int lf = cur[j - 1] + GAP;
            int bs = dg > up ? dg : up;
            cur[j] = lf > bs ? lf : bs;
        }
        int *t = prev; prev = cur; cur = t;
    }
    int r = prev[n];
    free(buf);
    return r;
}

/* ------------- the horse's walk: weigh one trial's fist of knots ------------- */
static int ham(int m, const char *restrict x, const char *restrict y)
{
    int c = 0;
    for (int i = 0; i < m; i++) c += (x[i] != y[i]);
    return c;
}

/* --------- the furrows: banded anti-diagonal wavefront, 16 slips/beat -------- */
static int wave16(int n, int W, const unsigned char *restrict pa,
                  const unsigned char *restrict pbr, int16_t *restrict buf)
{
    const size_t sz = (size_t)n + PAD;
    memset(buf, 0x88, 3u * sz * sizeof(int16_t));   /* every furrow starts -inf */
    int16_t *p2 = buf + 8, *p1 = buf + sz + 8, *c0 = buf + 2 * sz + 8;
    const int16_t NEG = (int16_t)0x8888;
    const int tmax = 2 * n;
    int res = 0;
    for (int t = 0; t <= tmax; t++) {           /* one flute note per beat */
        int lo = t - W; lo = (lo <= 0) ? 0 : ((lo + 1) >> 1);
        if (t - n > lo) lo = t - n;
        int hi = (t + W) >> 1;
        if (hi > n) hi = n;
        if (hi > t) hi = t;
        int bnd = (t <= W) && (t <= n);
        int i0 = bnd ? 1 : lo;
        int i1 = bnd ? t - 1 : hi;
        if (i1 >= i0) {
            const unsigned char *A = pa + (i0 - 1);
            const unsigned char *B = pbr + (n - t + i0);
            const int16_t *d2 = p2 + (i0 - 1);      /* diagonal predecessor  */
            const int16_t *dl = p1 + (i0 - 1);      /* up predecessor        */
            const int16_t *dr = p1 + i0;            /* left predecessor      */
            int16_t *out = c0 + i0;
            int cnt = i1 - i0 + 1;
#if defined(__AVX2__)
            const __m256i vm1 = _mm256_set1_epi16(-1);
            const __m256i v2  = _mm256_set1_epi16(2);
            const __m256i vg  = _mm256_set1_epi16(GAP);
            for (int k = 0; k < cnt; k += 16) {
                __m256i eq = _mm256_cvtepi8_epi16(
                    _mm_cmpeq_epi8(_mm_loadu_si128((const __m128i *)(A + k)),
                                   _mm_loadu_si128((const __m128i *)(B + k))));
                __m256i sc = _mm256_add_epi16(vm1, _mm256_and_si256(eq, v2));
                __m256i dg = _mm256_add_epi16(
                    _mm256_loadu_si256((const __m256i *)(d2 + k)), sc);
                __m256i gp = _mm256_add_epi16(_mm256_max_epi16(
                    _mm256_loadu_si256((const __m256i *)(dl + k)),
                    _mm256_loadu_si256((const __m256i *)(dr + k))), vg);
                _mm256_storeu_si256((__m256i *)(out + k),
                                    _mm256_max_epi16(dg, gp));
            }
#else
            for (int k = 0; k < cnt; k++) {
                int sc = (A[k] == B[k]) ? MATCH : MISMATCH;
                int dg = d2[k] + sc;
                int l = dl[k], r = dr[k];
                int gp = (l > r ? l : r) + GAP;
                out[k] = (int16_t)(dg > gp ? dg : gp);
            }
#endif
        }
        if (bnd) { int16_t v = (int16_t)(GAP * t); c0[0] = v; c0[t] = v; }
        c0[lo - 1] = NEG;                 /* furrows flung to the fish */
        c0[hi + 1] = NEG;
        res = c0[hi];
        int16_t *sw = p2; p2 = p1; p1 = c0; c0 = sw;
    }
    return res;
}

static int wave32(int n, int W, const unsigned char *restrict pa,
                  const unsigned char *restrict pbr, int32_t *restrict buf)
{
    const size_t sz = (size_t)n + PAD;
    memset(buf, 0x88, 3u * sz * sizeof(int32_t));
    int32_t *p2 = buf + 8, *p1 = buf + sz + 8, *c0 = buf + 2 * sz + 8;
    const int32_t NEG = (int32_t)0x88888888;
    const int tmax = 2 * n;
    int res = 0;
    for (int t = 0; t <= tmax; t++) {
        int lo = t - W; lo = (lo <= 0) ? 0 : ((lo + 1) >> 1);
        if (t - n > lo) lo = t - n;
        int hi = (t + W) >> 1;
        if (hi > n) hi = n;
        if (hi > t) hi = t;
        int bnd = (t <= W) && (t <= n);
        int i0 = bnd ? 1 : lo;
        int i1 = bnd ? t - 1 : hi;
        if (i1 >= i0) {
            const unsigned char *A = pa + (i0 - 1);
            const unsigned char *B = pbr + (n - t + i0);
            const int32_t *d2 = p2 + (i0 - 1);
            const int32_t *dl = p1 + (i0 - 1);
            const int32_t *dr = p1 + i0;
            int32_t *out = c0 + i0;
            int cnt = i1 - i0 + 1;
#if defined(__AVX2__)
            const __m256i vm1 = _mm256_set1_epi32(-1);
            const __m256i v2  = _mm256_set1_epi32(2);
            const __m256i vg  = _mm256_set1_epi32(GAP);
            for (int k = 0; k < cnt; k += 8) {
                __m256i eq = _mm256_cvtepi8_epi32(
                    _mm_cmpeq_epi8(_mm_loadl_epi64((const __m128i *)(A + k)),
                                   _mm_loadl_epi64((const __m128i *)(B + k))));
                __m256i sc = _mm256_add_epi32(vm1, _mm256_and_si256(eq, v2));
                __m256i dg = _mm256_add_epi32(
                    _mm256_loadu_si256((const __m256i *)(d2 + k)), sc);
                __m256i gp = _mm256_add_epi32(_mm256_max_epi32(
                    _mm256_loadu_si256((const __m256i *)(dl + k)),
                    _mm256_loadu_si256((const __m256i *)(dr + k))), vg);
                _mm256_storeu_si256((__m256i *)(out + k),
                                    _mm256_max_epi32(dg, gp));
            }
#else
            for (int k = 0; k < cnt; k++) {
                int sc = (A[k] == B[k]) ? MATCH : MISMATCH;
                int dg = d2[k] + sc;
                int l = dl[k], r = dr[k];
                int gp = (l > r ? l : r) + GAP;
                out[k] = dg > gp ? dg : gp;
            }
#endif
        }
        if (bnd) { int32_t v = GAP * t; c0[0] = v; c0[t] = v; }
        c0[lo - 1] = NEG;
        c0[hi + 1] = NEG;
        res = c0[hi];
        int32_t *sw = p2; p2 = p1; p1 = c0; c0 = sw;
    }
    return res;
}

int kernel(int n, const char *a, const char *b)
{
    if (n <= 0) return 0;
    if (n < 96) return nw_tworow(n, a, b);      /* guard: lanes would be empty */

    /* --- weigh the unslipped fist: this alone decides the regime --- */
    int h0 = ham(n, a, b);
    if (h0 == 0) return n;
    int best = n - 2 * h0;                      /* achievable => lower bound   */
    int W0 = (2 * h0) / 5;

    /* --- walk the horse through a window of doorways, weigh each fist --- */
    if (W0 >= 4) {
        int S = W0 - 1;
        if (S > 32) S = 32;
        if (S > n / 2) S = n / 2;
        for (int s = 1; s <= S; s++) {
            int v = n - 5 * s - 2 * ham(n - s, a + s, b);
            if (v > best) best = v;
            v = n - 5 * s - 2 * ham(n - s, a, b + s);
            if (v > best) best = v;
        }
    }

    /* --- keep only the furrows the lightest fist cannot rule out --- */
    int W = (n - best) / 5;                     /* provably >= optimal |i-j|   */
    if (W <= 0) return n - 2 * h0;              /* no slip can pay its own 5   */
    if (W > n) W = n;

    const size_t sz = (size_t)n + PAD;
    unsigned char *chars = (unsigned char *)malloc(2u * sz);
    void *scores = malloc(3u * sz * sizeof(int32_t));
    if (!chars || !scores) { free(chars); free(scores); return nw_tworow(n, a, b); }
    unsigned char *pa = chars, *pbr = chars + sz;
    for (int i = 0; i < n; i++) pa[i]  = (unsigned char)a[i];
    for (int i = 0; i < n; i++) pbr[i] = (unsigned char)b[n - 1 - i]; /* reversed */
    memset(pa  + n, 0xF0, sz - (size_t)n);
    memset(pbr + n, 0x0F, sz - (size_t)n);      /* padding can never match */

    int r = (n <= 12000) ? wave16(n, W, pa, pbr, (int16_t *)scores)
                         : wave32(n, W, pa, pbr, (int32_t *)scores);
    free(chars); free(scores);
    return r;
}
