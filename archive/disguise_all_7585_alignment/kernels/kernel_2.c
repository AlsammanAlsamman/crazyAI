#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

#define GAP_ (-2)

typedef struct {
    const char *af, *ar, *bf, *br;   /* padded: a, reverse(a), b, reverse(b) */
    int n, use16;
} ctx_t;

/* out[j] = NW( A[0..R), B[0..j) ) for j=0..C, where Brev[t] = B[C-1-t]. */
static void dp_row_scalar(const char *A, int R, const char *Brev, int C, int *out)
{
    int *prev = (int *)malloc((size_t)(C + 1) * sizeof(int));
    int *cur  = (int *)malloc((size_t)(C + 1) * sizeof(int));
    const char *Bp = Brev + C;                 /* B[j-1] == Bp[-j] */
    for (int j = 0; j <= C; j++) prev[j] = GAP_ * j;
    for (int i = 1; i <= R; i++) {
        char ai = A[i - 1];
        cur[0] = GAP_ * i;
        for (int j = 1; j <= C; j++) {
            int v = prev[j - 1] + ((ai == Bp[-j]) ? 1 : -1);
            int u = prev[j] + GAP_;      if (u > v) v = u;
            int l = cur[j - 1] + GAP_;   if (l > v) v = l;
            cur[j] = v;
        }
        int *t = prev; prev = cur; cur = t;
    }
    memcpy(out, prev, (size_t)(C + 1) * sizeof(int));
    free(prev); free(cur);
}

#if defined(__AVX2__)
/* Anti-diagonal sweep, 32-bit lanes.  d = i + j; arrays indexed by row i.
   up   = D[i-1][j]   -> d1[i-1]
   left = D[i][j-1]   -> d1[i]
   diag = D[i-1][j-1] -> d2[i-1]                                        */
static void dp_row_i32(const char *A, int R, const char *Brev, int C, int *out)
{
    const int PAD = 16;
    size_t sz = (size_t)R + 1 + PAD;
    int *buf = (int *)malloc(3 * sz * sizeof(int));
    if (!buf) { dp_row_scalar(A, R, Brev, C, out); return; }
    memset(buf, 0, 3 * sz * sizeof(int));
    int *d2 = buf, *d1 = buf + sz, *d0 = buf + 2 * sz;
    const __m256i two  = _mm256_set1_epi32(2);
    const __m256i mone = _mm256_set1_epi32(-1);
    const __m256i gv   = _mm256_set1_epi32(GAP_);
    int dmax = R + C;
    for (int d = 0; d <= dmax; d++) {
        if (d <= C) d0[0] = GAP_ * d;          /* i = 0 boundary */
        if (d <= R) d0[d] = GAP_ * d;          /* j = 0 boundary */
        int st = d - C; if (st < 1) st = 1;
        int en = d - 1; if (en > R) en = R;
        if (en >= st) {
            const char *Bp = Brev + (C - d);   /* Bp[i] == B[d-i-1] */
            for (int i = st; i <= en; i += 8) {
                __m128i ca = _mm_loadl_epi64((const __m128i *)(A + (i - 1)));
                __m128i cb = _mm_loadl_epi64((const __m128i *)(Bp + i));
                __m256i eq = _mm256_cvtepi8_epi32(_mm_cmpeq_epi8(ca, cb));
                __m256i s  = _mm256_add_epi32(_mm256_and_si256(eq, two), mone);
                __m256i dg = _mm256_add_epi32(
                                 _mm256_loadu_si256((const __m256i *)(d2 + (i - 1))), s);
                __m256i u  = _mm256_loadu_si256((const __m256i *)(d1 + (i - 1)));
                __m256i l  = _mm256_loadu_si256((const __m256i *)(d1 + i));
                __m256i g  = _mm256_add_epi32(_mm256_max_epi32(u, l), gv);
                _mm256_storeu_si256((__m256i *)(d0 + i), _mm256_max_epi32(dg, g));
            }
        }
        if (d >= R) out[d - R] = d0[R];
        int *t = d2; d2 = d1; d1 = d0; d0 = t;
    }
    free(buf);
}

/* same sweep, 16-bit lanes (safe while |value| <= 2n < 32767) */
static void dp_row_i16(const char *A, int R, const char *Brev, int C, int *out)
{
    const int PAD = 32;
    size_t sz = (size_t)R + 1 + PAD;
    int16_t *buf = (int16_t *)malloc(3 * sz * sizeof(int16_t));
    if (!buf) { dp_row_scalar(A, R, Brev, C, out); return; }
    memset(buf, 0, 3 * sz * sizeof(int16_t));
    int16_t *d2 = buf, *d1 = buf + sz, *d0 = buf + 2 * sz;
    const __m256i two  = _mm256_set1_epi16(2);
    const __m256i mone = _mm256_set1_epi16(-1);
    const __m256i gv   = _mm256_set1_epi16((short)GAP_);
    int dmax = R + C;
    for (int d = 0; d <= dmax; d++) {
        if (d <= C) d0[0] = (int16_t)(GAP_ * d);
        if (d <= R) d0[d] = (int16_t)(GAP_ * d);
        int st = d - C; if (st < 1) st = 1;
        int en = d - 1; if (en > R) en = R;
        if (en >= st) {
            const char *Bp = Brev + (C - d);
            for (int i = st; i <= en; i += 16) {
                __m128i ca = _mm_loadu_si128((const __m128i *)(A + (i - 1)));
                __m128i cb = _mm_loadu_si128((const __m128i *)(Bp + i));
                __m256i eq = _mm256_cvtepi8_epi16(_mm_cmpeq_epi8(ca, cb));
                __m256i s  = _mm256_add_epi16(_mm256_and_si256(eq, two), mone);
                __m256i dg = _mm256_add_epi16(
                                 _mm256_loadu_si256((const __m256i *)(d2 + (i - 1))), s);
                __m256i u  = _mm256_loadu_si256((const __m256i *)(d1 + (i - 1)));
                __m256i l  = _mm256_loadu_si256((const __m256i *)(d1 + i));
                __m256i g  = _mm256_add_epi16(_mm256_max_epi16(u, l), gv);
                _mm256_storeu_si256((__m256i *)(d0 + i), _mm256_max_epi16(dg, g));
            }
        }
        if (d >= R) out[d - R] = (int)d0[R];
        int16_t *t = d2; d2 = d1; d1 = d0; d0 = t;
    }
    free(buf);
}
#endif /* __AVX2__ */

static void run_row(const ctx_t *cx, const char *A, int R,
                    const char *Brev, int C, int *out)
{
#if defined(__AVX2__)
    if (R >= 8 && C >= 32) {
        if (cx->use16) dp_row_i16(A, R, Brev, C, out);
        else           dp_row_i32(A, R, Brev, C, out);
        return;
    }
#endif
    (void)cx;
    dp_row_scalar(A, R, Brev, C, out);
}

/* score of a[i0..i1) vs b[j0..j1) */
static int solve(const ctx_t *cx, int i0, int i1, int j0, int j1, int depth)
{
    int R = i1 - i0, C = j1 - j0;
    if (R <= 0) return GAP_ * C;
    if (C <= 0) return GAP_ * R;

    if (depth <= 0 || R < 8 || C < 32) {                 /* piece is small: fill it */
        int *out = (int *)malloc((size_t)(C + 1) * sizeof(int));
        run_row(cx, cx->af + i0, R, cx->br + (cx->n - j1), C, out);
        int s = out[C];
        free(out);
        return s;
    }

    int mid = i0 + R / 2;
    int *F = (int *)malloc((size_t)(C + 1) * sizeof(int));
    int *G = (int *)malloc((size_t)(C + 1) * sizeof(int));

    /* Mia walks forward to the middle ... */
    #pragma omp task shared(F)
    run_row(cx, cx->af + i0, mid - i0, cx->br + (cx->n - j1), C, F);
    /* ... Leo walks backward from the far end to meet her. */
    run_row(cx, cx->ar + (cx->n - i1), i1 - mid, cx->bf + j0, C, G);
    #pragma omp taskwait

    int best = F[0] + G[C], bj = 0;
    for (int j = 1; j <= C; j++) {
        int v = F[j] + G[C - j];
        if (v > best) { best = v; bj = j; }
    }
    free(F); free(G);
    if (depth == 1) return best;          /* meeting value == sum of the two halves */

    int s1 = 0, s2 = 0;
    #pragma omp task shared(s1)
    s1 = solve(cx, i0, mid, j0, j0 + bj, depth - 1);
    s2 = solve(cx, mid, i1, j0 + bj, j1, depth - 1);
    #pragma omp taskwait
    return s1 + s2;
}

int kernel(int n, const char *a, const char *b)
{
    if (n <= 0) return 0;
    size_t bs = (size_t)n + 64;
    char *buf = (char *)malloc(4 * bs);
    if (!buf) return 0;
    char *af = buf, *ar = buf + bs, *bf = buf + 2 * bs, *br = buf + 3 * bs;
    memcpy(af, a, (size_t)n);
    memcpy(bf, b, (size_t)n);
    for (int i = 0; i < n; i++) { ar[i] = a[n - 1 - i]; br[i] = b[n - 1 - i]; }
    memset(af + n, 0x7e, 64); memset(ar + n, 0x7e, 64);
    memset(bf + n, 0x7f, 64); memset(br + n, 0x7f, 64);

    ctx_t cx;
    cx.af = af; cx.ar = ar; cx.bf = bf; cx.br = br;
    cx.n = n; cx.use16 = (n <= 15000);

    int depth = 1;
    if (n < 96)     depth = 0;
    if (n >= 32768) depth = 2;            /* deeper split only once cache demands it */

    int result = 0;
#ifdef _OPENMP
    if (depth > 0 && omp_get_max_threads() > 1 && (double)n * (double)n > 250000.0) {
        #pragma omp parallel
        {
            #pragma omp single nowait
            result = solve(&cx, 0, n, 0, n, depth);
        }
    } else
#endif
    {
        result = solve(&cx, 0, n, 0, n, depth);
    }
    free(buf);
    return result;
}
