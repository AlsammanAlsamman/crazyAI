#include <stdlib.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define MATCH     1
#define MISMATCH -1
#define GAP      -2

#define WAVE_MIN   32     /* short tray: crawl junction by junction      */
#define I16_MAX_N  12000  /* deeper sand needed above this (|v| <= 2n)   */
#define WPAD       80     /* slack so an over-long body spills harmlessly */

/* ---- short tray: one crossing at a time, two rows of sand, no malloc ---- */
static int nw_rows_small(int n, const char *a, const char *b)
{
    int r0[WAVE_MIN + 1], r1[WAVE_MIN + 1];
    int *prev = r0, *cur = r1;
    int i, j;
    for (j = 0; j <= n; j++) prev[j] = GAP * j;
    for (i = 1; i <= n; i++) {
        char ai = a[i - 1];
        int *t;
        cur[0] = GAP * i;
        for (j = 1; j <= n; j++) {
            int dg = prev[j - 1] + ((ai == b[j - 1]) ? MATCH : MISMATCH);
            int up = prev[j] + GAP;
            int lf = cur[j - 1] + GAP;
            int best = dg > up ? dg : up;
            if (lf > best) best = lf;
            cur[j] = best;
        }
        t = prev; prev = cur; cur = t;
    }
    return prev[n];
}

/* ---- last-resort crawl (allocation failure) ---- */
static int nw_rows_big(int n, const char *a, const char *b)
{
    int *buf = (int *)malloc((size_t)2 * (size_t)(n + 1) * sizeof(int));
    int *prev, *cur, i, j, r;
    if (!buf) return 0;
    prev = buf; cur = buf + (n + 1);
    for (j = 0; j <= n; j++) prev[j] = GAP * j;
    for (i = 1; i <= n; i++) {
        char ai = a[i - 1];
        int *t;
        cur[0] = GAP * i;
        for (j = 1; j <= n; j++) {
            int dg = prev[j - 1] + ((ai == b[j - 1]) ? MATCH : MISMATCH);
            int up = prev[j] + GAP;
            int lf = cur[j - 1] + GAP;
            int best = dg > up ? dg : up;
            if (lf > best) best = lf;
            cur[j] = best;
        }
        t = prev; prev = cur; cur = t;
    }
    r = prev[n];
    free(buf);
    return r;
}

/* ---- vast tray / no eight-wind body: same wind, deeper sand ---- */
static int nw_wave_i32(int n, const char *a, const char *b)
{
    int len = n + 2 + WPAD;
    size_t need = 3u * (size_t)len * sizeof(int) + 2u * (size_t)(n + WPAD);
    unsigned char *mem = (unsigned char *)malloc(need);
    int *f2, *f1, *f0, *tmp;
    char *A, *Br;
    int d, k, r;
    if (!mem) return nw_rows_big(n, a, b);
    f2 = (int *)mem; f1 = f2 + len; f0 = f1 + len;
    A  = (char *)(f0 + len);
    Br = A + (n + WPAD);
    memset(f2, 0, 3u * (size_t)len * sizeof(int));
    memcpy(A, a, (size_t)n); memset(A + n, 0, WPAD);
    for (k = 0; k < n; k++) Br[k] = b[n - 1 - k];   /* the cord let to slip */
    memset(Br + n, 1, WPAD);
    f2[1] = 0;                                   /* furrow 0: (0,0)        */
    f1[1] = GAP; f1[2] = GAP;                    /* furrow 1: (0,1),(1,0)  */
    for (d = 2; d <= 2 * n; d++) {
        int lo = d - n, hi = d - 1, cnt;
        const char *__restrict pa; const char *__restrict pb;
        const int *__restrict q2; const int *__restrict q1a;
        const int *__restrict q1b; int *__restrict out;
        if (lo < 1) lo = 1;
        if (hi > n) hi = n;
        cnt = hi - lo + 1;
        pa  = A + (lo - 1);
        pb  = Br + (n - d + lo);
        q2  = f2 + lo;          /* diagonal face: dp[i-1][j-1] */
        q1a = f1 + lo;          /* up face:       dp[i-1][j]   */
        q1b = f1 + lo + 1;      /* left face:     dp[i][j-1]   */
        out = f0 + lo + 1;
#ifdef _OPENMP
#pragma omp simd
#endif
        for (k = 0; k < cnt; k++) {
            int sc = (pa[k] == pb[k]) ? MATCH : MISMATCH;
            int dg = q2[k] + sc;
            int up = q1a[k] + GAP;
            int lf = q1b[k] + GAP;
            int best = dg > up ? dg : up;
            if (lf > best) best = lf;
            out[k] = best;
        }
        if (d <= n) { f0[1] = GAP * d; f0[d + 1] = GAP * d; }
        tmp = f2; f2 = f1; f1 = f0; f0 = tmp;
    }
    r = f1[n + 1];
    free(mem);
    return r;
}

#if defined(__AVX2__)
/* ---- the worm proper: one cell wide, 16 long, walking the NE wind ---- */
static int nw_wave_i16_avx2(int n, const char *a, const char *b)
{
    int len = n + 2 + WPAD;
    size_t need = 3u * (size_t)len * sizeof(short) + 2u * (size_t)(n + WPAD);
    unsigned char *mem = (unsigned char *)malloc(need + 64);
    short *f2, *f1, *f0, *tmp;
    char *A, *Br;
    int d, k, r;
    const __m256i vgap = _mm256_set1_epi16(GAP);
    const __m256i vm1  = _mm256_set1_epi16(-1);
    if (!mem) return nw_wave_i32(n, a, b);
    f2 = (short *)mem; f1 = f2 + len; f0 = f1 + len;
    A  = (char *)(f0 + len);
    Br = A + (n + WPAD);
    memset(f2, 0, 3u * (size_t)len * sizeof(short));
    memcpy(A, a, (size_t)n); memset(A + n, 0, WPAD);
    for (k = 0; k < n; k++) Br[k] = b[n - 1 - k];
    memset(Br + n, 1, WPAD);
    f2[1] = 0;
    f1[1] = GAP; f1[2] = GAP;
    for (d = 2; d <= 2 * n; d++) {
        int lo = d - n, hi = d - 1, cnt;
        const char *pa, *pb;
        const short *q2, *q1a, *q1b;
        short *out;
        if (lo < 1) lo = 1;
        if (hi > n) hi = n;
        cnt = hi - lo + 1;
        pa  = A + (lo - 1);
        pb  = Br + (n - d + lo);
        q2  = f2 + lo;
        q1a = f1 + lo;
        q1b = f1 + lo + 1;
        out = f0 + lo + 1;
        for (k = 0; k < cnt; k += 16) {
            __m128i ca = _mm_loadu_si128((const __m128i *)(pa + k));
            __m128i cb = _mm_loadu_si128((const __m128i *)(pb + k));
            __m256i t  = _mm256_cvtepi8_epi16(_mm_cmpeq_epi8(ca, cb));
            __m256i sc = _mm256_sub_epi16(vm1, _mm256_add_epi16(t, t));
            __m256i dg = _mm256_add_epi16(
                             _mm256_loadu_si256((const __m256i *)(q2 + k)), sc);
            __m256i up = _mm256_add_epi16(
                             _mm256_loadu_si256((const __m256i *)(q1a + k)), vgap);
            __m256i lf = _mm256_add_epi16(
                             _mm256_loadu_si256((const __m256i *)(q1b + k)), vgap);
            _mm256_storeu_si256((__m256i *)(out + k),
                _mm256_max_epi16(dg, _mm256_max_epi16(up, lf)));
        }
        /* boundary mounds written after the body, so spill is overwritten */
        if (d <= n) { f0[1] = (short)(GAP * d); f0[d + 1] = (short)(GAP * d); }
        tmp = f2; f2 = f1; f1 = f0; f0 = tmp;
    }
    r = (int)f1[n + 1];
    free(mem);
    return r;
}
#endif

int kernel(int n, const char *a, const char *b)
{
    if (n <= 0) return 0;
    if (n <= WAVE_MIN) return nw_rows_small(n, a, b);   /* short tray      */
#if defined(__AVX2__)
    if (n <= I16_MAX_N) return nw_wave_i16_avx2(n, a, b); /* the worm      */
#endif
    return nw_wave_i32(n, a, b);                        /* vast tray       */
}
