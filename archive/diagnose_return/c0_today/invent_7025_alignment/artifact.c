#include <stdlib.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define MATCH     1
#define MISMATCH -1
#define GAP      -2

/* ---------- tiny strings: plain rolling-row DP (no wavefront overhead) ---------- */
static int nw_tiny(int n, const char *restrict a, const char *restrict b)
{
    int A[130], B[130];
    int *prev = A, *cur = B;
    for (int j = 0; j <= n; j++) prev[j] = j * GAP;
    for (int i = 1; i <= n; i++) {
        char ai = a[i - 1];
        cur[0] = i * GAP;
        for (int j = 1; j <= n; j++) {
            int bst = prev[j - 1] + ((ai == b[j - 1]) ? MATCH : MISMATCH);
            int u = prev[j] + GAP;      if (u > bst) bst = u;
            int l = cur[j - 1] + GAP;   if (l > bst) bst = l;
            cur[j] = bst;
        }
        int *t = prev; prev = cur; cur = t;
    }
    return prev[n];
}

/* ---------- narrow-band regime: scalar banded rows (exact for the proved W) ---- */
static int nw_banded_scalar(int n, const char *restrict a, const char *restrict b, int W)
{
    const int NEG = -(1 << 28);
    int blk = n + 4;
    int *base = (int *)malloc((size_t)(2 * blk) * sizeof(int));
    if (!base) return 0;
    int *prev = base + 1;          /* legal indices -1 .. n+2 */
    int *cur  = base + blk + 1;

    int jhi0 = (W < n) ? W : n;
    for (int j = 0; j <= jhi0; j++) prev[j] = j * GAP;
    prev[-1] = NEG;
    prev[jhi0 + 1] = NEG;

    for (int i = 1; i <= n; i++) {
        int jlo = i - W; if (jlo < 0) jlo = 0;
        int jhi = i + W; if (jhi > n) jhi = n;
        char ai = a[i - 1];
        int left = NEG;
        int j = jlo;
        if (j == 0) { cur[0] = i * GAP; left = cur[0]; j = 1; }
        for (; j <= jhi; j++) {
            int bst = prev[j - 1] + ((ai == b[j - 1]) ? MATCH : MISMATCH);
            int u = prev[j] + GAP;  if (u > bst) bst = u;
            int l = left + GAP;     if (l > bst) bst = l;
            cur[j] = bst;
            left = bst;
        }
        cur[jlo - 1] = NEG;
        cur[jhi + 1] = NEG;
        int *t = prev; prev = cur; cur = t;
    }
    int r = prev[n];
    free(base);
    return r;
}

#if defined(__AVX2__)
/* ---------- wide-band regime: the flute-clocked wavefront -----------------------
   One tick k = i+j = one flute note.  Within a tick every house is independent.
   The horse's single permitted one-house slip is the one-lane offset:
       slipped   -> D1[i-1], D0[i-1]
       unslipped -> D1[i]
   b is stored reversed so that both crowns stream contiguously along a tick.     */
static int nw_wave_avx2(int n, const char *restrict a, const char *restrict b, int W)
{
    const int PAD = 40;
    const short NEG = -30000;                 /* below -4n for n <= 7000 */
    int stride = n + 1 + 2 * PAD;

    short *buf = (short *)malloc((size_t)(3 * stride) * sizeof(short));
    char  *ap  = (char  *)malloc((size_t)n + 64);
    char  *bp  = (char  *)malloc((size_t)n + 64);
    if (!buf || !ap || !bp) {
        free(buf); free(ap); free(bp);
        return nw_banded_scalar(n, a, b, W);
    }
    memcpy(ap, a, (size_t)n);
    memset(ap + n, 0x7f, 64);                 /* padding crowns that never match */
    for (int t = 0; t < n; t++) bp[t] = b[n - 1 - t];
    memset(bp + n, 0x01, 64);

    for (int t = 0; t < 3 * stride; t++) buf[t] = NEG;
    short *D0 = buf + PAD;                    /* tick k-2 */
    short *D1 = buf + stride + PAD;           /* tick k-1 */
    short *D2 = buf + 2 * stride + PAD;       /* tick k   */

    const __m256i vgap = _mm256_set1_epi16(GAP);
    const __m256i vm1  = _mm256_set1_epi16(-1);

    int result = 0;
    const int K = 2 * n;
    for (int k = 0; k <= K; k++) {
        int ilo = k - n; if (ilo < 0) ilo = 0;
        int t1 = k - W;  t1 = (t1 <= 0) ? 0 : (t1 + 1) / 2;   /* ceil((k-W)/2) */
        if (t1 > ilo) ilo = t1;
        int ihi = (k < n) ? k : n;
        int t2 = (k + W) / 2;                                  /* floor((k+W)/2) */
        if (t2 < ihi) ihi = t2;

        int vlo = (ilo < 1) ? 1 : ilo;
        int vhi = (ihi < k - 1) ? ihi : k - 1;
        int base = n - k;

        for (int i = vlo; i <= vhi; i += 16) {
            __m128i ca = _mm_loadu_si128((const __m128i *)(ap + (i - 1)));
            __m128i cb = _mm_loadu_si128((const __m128i *)(bp + base + i));
            __m256i eq = _mm256_cvtepi8_epi16(_mm_cmpeq_epi8(ca, cb)); /* 0 / -1 */
            /* lotus petal (+1) where beasts agree, red knot (-1) where they differ */
            __m256i sc = _mm256_sub_epi16(vm1, _mm256_add_epi16(eq, eq));
            __m256i dg = _mm256_adds_epi16(
                             _mm256_loadu_si256((const __m256i *)(D0 + i - 1)), sc);
            __m256i up = _mm256_adds_epi16(
                             _mm256_loadu_si256((const __m256i *)(D1 + i - 1)), vgap);
            __m256i lf = _mm256_adds_epi16(
                             _mm256_loadu_si256((const __m256i *)(D1 + i)), vgap);
            _mm256_storeu_si256((__m256i *)(D2 + i),
                _mm256_max_epi16(dg, _mm256_max_epi16(up, lf)));
        }
        if (ilo == 0) D2[0] = (short)(k * GAP);   /* (0,k) */
        if (ihi == k) D2[k] = (short)(k * GAP);   /* (k,0) */

        D2[ilo - 1] = NEG; D2[ilo - 2] = NEG;     /* flung fists: out of band */
        D2[ihi + 1] = NEG; D2[ihi + 2] = NEG;

        if (k == K) { result = D2[n]; break; }
        short *tmp = D0; D0 = D1; D1 = D2; D2 = tmp;
    }
    free(buf); free(ap); free(bp);
    return result;
}
#endif

static int hamming_seg(const char *restrict x, const char *restrict y, int len)
{
    int c = 0;
    for (int t = 0; t < len; t++) c += (x[t] != y[t]);
    return c;
}

int kernel(int n, const char *a, const char *b)
{
    if (n <= 0)   return 0;
    if (n <= 128) return nw_tiny(n, a, b);

    /* ---- weigh the first fists: the horse in the first few doorways, and in none.
       Each trial is an achievable alignment, so its weight is a valid lower bound. */
    int S = 8; if (S > n - 1) S = n - 1;
    int L = -4 * n;
    for (int s = 0; s <= S; s++) {
        int len = n - s;
        int v1 = (len - 2 * hamming_seg(a, b + s, len)) - 4 * s;
        if (v1 > L) L = v1;
        if (s) {
            int v2 = (len - 2 * hamming_seg(a + s, b, len)) - 4 * s;
            if (v2 > L) L = v2;
        }
    }

    /* score = n - 5g - 2*mm  =>  g_opt <= (n - S_opt)/5 <= (n - L)/5,
       and max|i-j| <= g_opt.  Every diagonal past W is provably discardable. */
    int W = (n - L) / 5;
    if (W < 2) W = 2;
    if (W > n) W = n;

#if defined(__AVX2__)
    if (n <= 7000 && (2 * W + 1) >= 48)        /* wide regime: lanes are worth it  */
        return nw_wave_avx2(n, a, b, W);
#endif
    return nw_banded_scalar(n, a, b, W);       /* near regime / no AVX2 / large n  */
}
