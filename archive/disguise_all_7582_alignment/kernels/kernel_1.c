#include <stdlib.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define GPEN (-2)

/* ---- portable banded DP over rows (fallback: no AVX2, or huge n) ---- */
static int band_rows(int n, const char *a, const char *b, int d,
                     int *prev, int *cur)
{
    const int NEG = -(1 << 28);
    int i, j, lim;
    if (d > n) d = n;
    if (d < 1) d = 1;
    for (j = 0; j <= n + 1; j++) prev[j] = NEG;
    lim = (d < n) ? d : n;
    for (j = 0; j <= lim; j++) prev[j] = -2 * j;
    for (i = 1; i <= n; i++) {
        int lo = i - d, hi = i + d, start, left;
        const char ai = a[i - 1];
        if (lo < 0) lo = 0;
        if (hi > n) hi = n;
        if (lo > 0) cur[lo - 1] = NEG;
        if (hi < n) cur[hi + 1] = NEG;
        if (lo == 0) { cur[0] = -2 * i; start = 1; } else start = lo;
        left = cur[start - 1];
        for (j = start; j <= hi; j++) {
            int best = prev[j - 1] + ((ai == b[j - 1]) ? 1 : -1);
            int t = prev[j] + GPEN; if (t > best) best = t;
            t = left + GPEN;        if (t > best) best = t;
            cur[j] = best;
            left = best;
        }
        { int *tmp = prev; prev = cur; cur = tmp; }
    }
    return prev[n];
}

#if defined(__AVX2__)
/* ---- banded anti-diagonal wavefront, 16-bit lanes ----
   ap : padded copy of a ; br : padded reverse of b
   A,B,C : three rows, valid indices [-32, n+63]                        */
static int band_diag16(int n, const char *ap, const char *br, int d,
                       short *A, short *B, short *C)
{
    const short NEG = -32000;
    short *p2 = A, *p1 = B, *cu = C;
    int k, t;
#if defined(__AVX512BW__) && defined(__AVX512VL__)
    const __m512i vgap = _mm512_set1_epi16(GPEN);
    const __m512i vpos = _mm512_set1_epi16(1);
    const __m512i vneg = _mm512_set1_epi16(-1);
    const int STEP = 32;
#else
    const __m256i vgap = _mm256_set1_epi16(GPEN);
    const __m256i vneg = _mm256_set1_epi16(-1);
    const int STEP = 16;
#endif
    for (t = -32; t <= n + 63; t++) { A[t] = NEG; B[t] = NEG; C[t] = NEG; }
    p1[0] = 0;                              /* anti-diagonal 0: cell (0,0) */

    for (k = 1; k <= 2 * n; k++) {
        int lo, hi, i0, i1, i, sh;
        lo = (k <= d) ? 0 : ((k - d + 1) >> 1);   /* ceil((k-d)/2), k-d>0 */
        if (k - n > lo) lo = k - n;
        hi = (k + d) >> 1;
        if (hi > n) hi = n;
        if (hi > k) hi = k;
        i0 = (lo > 1) ? lo : 1;                   /* interior cells only   */
        i1 = (hi < k - 1) ? hi : k - 1;
        sh = n - k;
        for (i = i0; i <= i1; i += STEP) {
#if defined(__AVX512BW__) && defined(__AVX512VL__)
            __m256i ca = _mm256_loadu_si256((const __m256i *)(ap + i - 1));
            __m256i cb = _mm256_loadu_si256((const __m256i *)(br + sh + i));
            __mmask32 eq = _mm256_cmpeq_epi8_mask(ca, cb);
            __m512i sc = _mm512_mask_blend_epi16(eq, vneg, vpos);
            __m512i dg = _mm512_loadu_si512((const void *)(p2 + i - 1));
            __m512i up = _mm512_loadu_si512((const void *)(p1 + i - 1));
            __m512i lf = _mm512_loadu_si512((const void *)(p1 + i));
            __m512i bs = _mm512_max_epi16(_mm512_add_epi16(dg, sc),
                           _mm512_add_epi16(_mm512_max_epi16(up, lf), vgap));
            _mm512_storeu_si512((void *)(cu + i), bs);
#else
            __m128i ca = _mm_loadu_si128((const __m128i *)(ap + i - 1));
            __m128i cb = _mm_loadu_si128((const __m128i *)(br + sh + i));
            __m256i eq = _mm256_cvtepi8_epi16(_mm_cmpeq_epi8(ca, cb));
            __m256i sc = _mm256_sub_epi16(vneg, _mm256_add_epi16(eq, eq));
            __m256i dg = _mm256_loadu_si256((const __m256i *)(p2 + i - 1));
            __m256i up = _mm256_loadu_si256((const __m256i *)(p1 + i - 1));
            __m256i lf = _mm256_loadu_si256((const __m256i *)(p1 + i));
            __m256i bs = _mm256_max_epi16(_mm256_add_epi16(dg, sc),
                           _mm256_add_epi16(_mm256_max_epi16(up, lf), vgap));
            _mm256_storeu_si256((__m256i *)(cu + i), bs);
#endif
        }
        if (k <= d) {                        /* cells (0,k) and (k,0)      */
            short v = (short)(-2 * k);
            cu[0] = v;
            cu[k] = v;
        }
        cu[lo - 1] = NEG;                    /* band guards, written last  */
        cu[hi + 1] = NEG;
        { short *tmp = p2; p2 = p1; p1 = cu; cu = tmp; }
    }
    return (int)p1[n];
}
#endif

int kernel(int n, const char *a, const char *b)
{
    int i, m = 0, lb, dcap, d, S = 0, it;
    size_t cn, sstride;
    char *cbuf;
    short *sbuf = NULL;
    int *ibuf = NULL;

    if (n <= 0) return 0;

    for (i = 0; i < n; i++) m += (a[i] == b[i]);
    lb = 2 * m - n;                 /* score of the gap-free alignment      */
    dcap = (n - lb + 4) / 5;        /* no optimal path can deviate further  */
    if (dcap > n) dcap = n;
    if (dcap < 1) dcap = 1;
    d = (dcap < 24) ? dcap : 24;    /* cheap probe first                    */

    cn = (size_t)n + 128;
    sstride = (size_t)n + 96;
    cbuf = (char *)malloc(2 * cn);
    if (cbuf) {
        memcpy(cbuf, a, (size_t)n);
        memset(cbuf + n, 0, 128);
        { char *br = cbuf + cn;
          for (i = 0; i < n; i++) br[i] = b[n - 1 - i];
          memset(br + n, 1, 128); }
    }

    for (it = 0; ; it++) {
        int did_v = 0;
#if defined(__AVX2__)
        if (cbuf && (n + 2 * d) <= 31000) {
            if (!sbuf) sbuf = (short *)malloc(3 * sstride * sizeof(short));
            if (sbuf) {
                S = band_diag16(n, cbuf, cbuf + cn, d,
                                sbuf + 32, sbuf + sstride + 32,
                                sbuf + 2 * sstride + 32);
                did_v = 1;
            }
        }
#endif
        if (!did_v) {
            if (!ibuf) ibuf = (int *)malloc(2 * ((size_t)n + 8) * sizeof(int));
            if (!ibuf) { free(cbuf); free(sbuf); return lb; }
            S = band_rows(n, a, b, d, ibuf, ibuf + n + 8);
        }
        /* certificate: any path leaving the band scores <= n-5(d+1) */
        { int need = (n - S + 4) / 5;
          if (need <= d || d >= n) break;
          d = (need > n) ? n : need;
          if (it >= 4) d = n; }
    }

    free(cbuf);
    free(sbuf);
    free(ibuf);
    return S;
}
