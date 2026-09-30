/* The native's crossing, taken literally:
 *   pass 1  - walk the straight row and EVERY single-slip row at once (O(n)),
 *             count the fires, keep the highest pile  -> LB
 *   size    - the highest pile sizes the door: K = (n - LB)/5  (proved exact)
 *   pass 2+ - walk the band; if the pile it brings up proves the door was too
 *             narrow, throw the row in the tower and start over from the first
 *             coil with a wider door. Terminates; K=n is the full matrix.
 */
#include <stdlib.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define AL_GAP (-2)
#define AL_NEG (-(1 << 27))          /* far below any real score (|score| <= 4n) */

/* ---- pass 1: straight crossing + every single-slip crossing, O(n), no heap ---- */
static int al_crossings(int n, const char *restrict a, const char *restrict b,
                        int *diag_out)
{
    int dtot = 0, i, p;
    for (i = 0; i < n; i++) dtot += (a[i] == b[i]) ? 1 : -1;
    *diag_out = dtot;
    if (n < 2) return dtot > -4 ? dtot : -4;   /* the only slip costs -2-2 */

    int q1tot = 0, q2tot = 0;
    for (i = 0; i + 1 < n; i++) {
        q1tot += (a[i] == b[i + 1]) ? 1 : -1;  /* shadow row crouched forward */
        q2tot += (a[i + 1] == b[i]) ? 1 : -1;  /* near row crouched forward   */
    }
    /* score(slip at p) = P0[p] + (Qtot - Q[p]) - 4 ; maximise P0[p]-Q[p] */
    int p0 = 0, q1 = 0, q2 = 0, m1 = AL_NEG, m2 = AL_NEG;
    for (p = 0; p < n; p++) {
        int v1 = p0 - q1, v2 = p0 - q2;
        if (v1 > m1) m1 = v1;
        if (v2 > m2) m2 = v2;
        p0 += (a[p] == b[p]) ? 1 : -1;
        if (p + 1 < n) {
            q1 += (a[p] == b[p + 1]) ? 1 : -1;
            q2 += (a[p + 1] == b[p]) ? 1 : -1;
        }
    }
    int best = dtot, c1 = m1 + q1tot - 4, c2 = m2 + q2tot - 4;
    if (c1 > best) best = c1;
    if (c2 > best) best = c2;
    return best;
}

/* ---- one crossing with a door of half-width K: exact banded NW, 2 rows ---- */
static int al_band(int n, const char *restrict a, const char *restrict b,
                   int K, int *restrict scratch)
{
    const int W = 2 * K + 1;         /* offsets o = j - i + K, 0..2K, W = sentinel */
    const int stride = W + 2;        /* usable indices -1 .. W                     */
    int *prev = scratch + 1;
    int *cur  = scratch + stride + 1;
    int t, i, o;

    for (t = 0; t < 2 * stride; t++) scratch[t] = AL_NEG;
    {   int hi0 = 2 * K; if (hi0 > K + n) hi0 = K + n;
        for (o = K; o <= hi0; o++) prev[o] = AL_GAP * (o - K); }   /* row 0 */

#if defined(__AVX2__)
    const __m256i IDX1 = _mm256_setr_epi32(0,0,1,2,3,4,5,6);
    const __m256i IDX2 = _mm256_setr_epi32(0,0,0,1,2,3,4,5);
    const __m256i IDX4 = _mm256_setr_epi32(0,0,0,0,0,1,2,3);
    const __m256i VNEG = _mm256_set1_epi32(AL_NEG);
    const __m256i VM1  = _mm256_set1_epi32(-1);
    const __m256i VG1  = _mm256_set1_epi32(-2);
    const __m256i VG2  = _mm256_set1_epi32(-4);
    const __m256i VG4  = _mm256_set1_epi32(-8);
    const __m256i VCAR = _mm256_setr_epi32(-2,-4,-6,-8,-10,-12,-14,-16);
#endif

    for (i = 1; i <= n; i++) {
        const int o0 = K - i;                        /* offset holding j == 0 */
        int lo = o0 + 1; if (lo < 0) lo = 0;
        int hi = n - i + K; if (hi > 2 * K) hi = 2 * K;
        if (o0 >= 0 && o0 <= 2 * K) cur[o0] = AL_GAP * i;
        for (o = hi + 1; o <= hi + 2 && o <= W; o++) cur[o] = AL_NEG;
        const char ai = a[i - 1];
        const int  jb = i - K - 1;                   /* b index = jb + o      */
        o = lo;
#if defined(__AVX2__)
        if (hi - lo + 1 >= 8) {                      /* the sun lights 8 coils */
            const __m128i VAI = _mm_set1_epi8(ai);
            int carry = cur[o - 1];
            for (; o + 7 <= hi; o += 8) {
                __m128i by = _mm_loadl_epi64((const __m128i *)(b + jb + o));
                __m256i mk = _mm256_cvtepi8_epi32(_mm_cmpeq_epi8(by, VAI));
                __m256i sb = _mm256_sub_epi32(VM1, _mm256_slli_epi32(mk, 1));
                __m256i d  = _mm256_add_epi32(
                                _mm256_loadu_si256((const __m256i *)(prev + o)), sb);
                __m256i u  = _mm256_add_epi32(
                                _mm256_loadu_si256((const __m256i *)(prev + o + 1)), VG1);
                __m256i v  = _mm256_max_epi32(d, u), s;
                /* fires spread rightward on the water, dimming by 2: max-plus scan */
                s = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(v, IDX1), VNEG, 0x01);
                v = _mm256_max_epi32(v, _mm256_add_epi32(s, VG1));
                s = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(v, IDX2), VNEG, 0x03);
                v = _mm256_max_epi32(v, _mm256_add_epi32(s, VG2));
                s = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(v, IDX4), VNEG, 0x0F);
                v = _mm256_max_epi32(v, _mm256_add_epi32(s, VG4));
                v = _mm256_max_epi32(v,
                        _mm256_add_epi32(_mm256_set1_epi32(carry), VCAR));
                _mm256_storeu_si256((__m256i *)(cur + o), v);
                carry = _mm256_extract_epi32(v, 7);
            }
        }
#endif
        for (; o <= hi; o++) {
            int x = prev[o] + ((ai == b[jb + o]) ? 1 : -1);
            int u = prev[o + 1] + AL_GAP; if (u > x) x = u;
            int l = cur[o - 1]  + AL_GAP; if (l > x) x = l;
            cur[o] = x;
        }
        { int *sw = prev; prev = cur; cur = sw; }
    }
    return prev[K];                                  /* the queen's threshold */
}

int kernel(int n, const char *a, const char *b)
{
    enum { KSTAT = 16 };
    int stat[2 * (2 * KSTAT + 3)];
    int diag, lb, K, Kc, cap, best;
    int *scratch;

    if (n <= 0) return 0;

    lb = al_crossings(n, a, b, &diag);      /* highest pile of the native's rows */
    K  = (n - lb) / 5;                      /* the pile sizes the door           */
    if (K <= 0) return diag;                /* straight crossing provably optimal */
    if (K > n) K = n;

    scratch = stat; cap = KSTAT; best = lb;
    Kc = K < KSTAT ? K : KSTAT;             /* cheap narrow door first */
    for (;;) {
        int s, kreq;
        if (Kc > cap) {                     /* wider door needs a wider table */
            int *nb = (int *)malloc((size_t)(2 * (2 * Kc + 3)) * sizeof(int));
            if (!nb) break;
            if (scratch != stat) free(scratch);
            scratch = nb; cap = Kc;
        }
        s = al_band(n, a, b, Kc, scratch);
        if (s > best) best = s;
        kreq = (n - best) / 5;
        if (kreq <= Kc || Kc >= n) break;   /* certified: no deeper room is higher */
        Kc = kreq < n ? kreq : n;           /* into the tower, start from coil one */
    }
    if (scratch != stat) free(scratch);
    return best;
}
