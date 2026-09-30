#include <stdlib.h>
#include <stdint.h>

#define MATCH     1
#define MISMATCH -1
#define GAP      -2

/* ---- simpler path: exact rolling-row Needleman-Wunsch (small n / fallback) ---- */
static int nw_small(int n, const char *a, const char *b)
{
    int *prev = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int *cur  = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int r = 0;
    if (!prev || !cur) { free(prev); free(cur); return 0; }
    for (int j = 0; j <= n; j++) prev[j] = j * GAP;
    for (int i = 1; i <= n; i++) {
        char ai = a[i - 1];
        cur[0] = i * GAP;
        for (int j = 1; j <= n; j++) {
            int best = prev[j - 1] + (ai == b[j - 1] ? MATCH : MISMATCH);
            int up = prev[j] + GAP, lf = cur[j - 1] + GAP;
            if (up > best) best = up;
            if (lf > best) best = lf;
            cur[j] = best;
        }
        { int *t = prev; prev = cur; cur = t; }
    }
    r = prev[n];
    free(prev); free(cur);
    return r;
}

/* ---- the swept corridor: anti-diagonal wavefront, band half-width w ---- */
#define DEFINE_WAVE(FN, T, NEGV)                                               \
static int FN(int n, const char *a, const char *brev, int w, void *scratch)    \
{                                                                              \
    const int stride = n + 2;                                                  \
    const T NEG = (T)(NEGV);                                                   \
    T *base_ = (T *)scratch;                                                   \
    T *p2 = base_, *p1 = base_ + stride, *cu = base_ + 2 * stride;             \
    for (int t = 0; t < 3 * stride; t++) base_[t] = NEG;                       \
    p1[0] = (T)0;                                /* cell (0,0), diagonal d=0 */\
    p1[1] = NEG;                                                               \
    for (int d = 1; d <= 2 * n; d++) {                                         \
        int lo = d - n; if (lo < 0) lo = 0;                                    \
        int tt = d - w;                                                        \
        if (tt > 0) { int lb = (tt + 1) >> 1; if (lb > lo) lo = lb; }          \
        int hi = d; if (hi > n) hi = n;                                        \
        { int hb = (d + w) >> 1; if (hb < hi) hi = hb; }                       \
        if (lo <= hi) {                                                        \
            if (lo == 0) cu[0] = (T)(-2 * d);        /* rope-edge: (0,d) */     \
            if (hi == d) cu[d] = (T)(-2 * d);        /* rope-edge: (d,0) */     \
            {                                                                  \
                const int ilo = (lo > 1) ? lo : 1;                             \
                const int ihi = (hi < d - 1) ? hi : d - 1;                     \
                const int off = n - d;                                         \
                const char *restrict ra = a;                                   \
                const char *restrict rb = brev;                                \
                T *restrict A2 = p2;                                           \
                T *restrict A1 = p1;                                           \
                T *restrict AC = cu;                                           \
                _Pragma("omp simd")                                            \
                for (int i = ilo; i <= ihi; i++) {                             \
                    T sc = (ra[i - 1] == rb[i + off]) ? (T)MATCH : (T)MISMATCH;\
                    T v = (T)(A2[i - 1] + sc);                                 \
                    T u = (T)(A1[i - 1] + (T)GAP);                             \
                    T l = (T)(A1[i] + (T)GAP);                                 \
                    if (u > v) v = u;                                          \
                    if (l > v) v = l;                                          \
                    if (v < NEG) v = NEG;                                      \
                    AC[i] = v;                                                 \
                }                                                              \
            }                                                                  \
            if (lo >= 1)        cu[lo - 1] = NEG;   /* unlit bare floor */      \
            if (hi + 1 <= n + 1) cu[hi + 1] = NEG;                             \
        }                                                                      \
        { T *tmp = p2; p2 = p1; p1 = cu; cu = tmp; }                           \
    }                                                                          \
    return (int)p1[n];                                                         \
}

DEFINE_WAVE(wave16, int16_t, -20000)
DEFINE_WAVE(wave32, int32_t, -100000000)

int kernel(int n, const char *a, const char *b)
{
    if (n <= 0) return 0;
    if (n < 64) return nw_small(n, a, b);                 /* overhead guard */

    /* SEED 1: the floor-rope laid at a slant, so one sun lights both at once */
    char *brev = (char *)malloc((size_t)n + 64);
    if (!brev) return nw_small(n, a, b);
    for (int t = 0; t < n; t++) brev[t] = b[n - 1 - t];

    size_t stride = (size_t)n + 2;
    void *sc16 = NULL, *sc32 = NULL;
    int use16 = (n <= 6000);
    if (use16) sc16 = malloc(3 * stride * sizeof(int16_t));
    else       sc32 = malloc(3 * stride * sizeof(int32_t));
    if (!sc16 && !sc32) { free(brev); return nw_small(n, a, b); }

    /* SEED 2: the no-slip tally, read straight off the two ropes */
    int L = 0;
    #pragma omp simd reduction(+:L)
    for (int i = 0; i < n; i++) L += (a[i] == b[i]) ? MATCH : MISMATCH;

    /* a few knots to either side: cheap probe that tightens the bound */
    int probe = use16 ? wave16(n, a, brev, 8, sc16)
                      : wave32(n, a, brev, 8, sc32);
    if (probe > L) L = probe;

    /* the corridor's price: 5 points of score per knot of slip  =>  band */
    long long wl = ((long long)n - (long long)L) / 5;
    int w = (wl > (long long)n) ? n : (int)wl;
    if (w < 1) w = 1;

    int res;
    if (w <= 8) {
        res = probe;                      /* ropes already in step: O(n) done */
    } else {
        res = use16 ? wave16(n, a, brev, w, sc16)
                    : wave32(n, a, brev, w, sc32);
    }
    free(sc16); free(sc32); free(brev);
    return res;
}
