#include <stdlib.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define MATCH     1
#define MISMATCH -1
#define GAP      -2

/* The single hooded memory: rolling-row DP, O(n) scratch.
   Used for trays too small to be worth waking a worm, and where no winds blow. */
static int nw_scalar(int n, const char *restrict a, const char *restrict b)
{
    int *row = (int *)malloc((size_t)(n + 1) * sizeof(int));
    if (!row) return 0;
    for (int j = 0; j <= n; j++) row[j] = j * GAP;
    for (int i = 1; i <= n; i++) {
        int diag = row[0];
        row[0] = i * GAP;
        const char ai = a[i - 1];
        for (int j = 1; j <= n; j++) {
            int up = row[j];
            int v  = diag + (ai == b[j - 1] ? MATCH : MISMATCH);
            int t  = up + GAP;        if (t > v) v = t;
            t      = row[j - 1] + GAP; if (t > v) v = t;
            diag = up;
            row[j] = v;
        }
    }
    int r = row[n];
    free(row);
    return r;
}

#if defined(__AVX2__)
/* ---- the worm, shallow cups: 16 int16 mounds settled per drop ---- */
static int nw_worm16(int n, const char *restrict a, const char *restrict b)
{
    const int PAD = 40;
    char  *ap = (char *)malloc((size_t)n + 80);
    char  *br = (char *)malloc((size_t)n + 80);
    size_t dl = (size_t)n + 1 + PAD;
    short *d2 = (short *)calloc(dl, sizeof(short));
    short *d1 = (short *)calloc(dl, sizeof(short));
    short *d0 = (short *)calloc(dl, sizeof(short));
    if (!ap || !br || !d2 || !d1 || !d0) {
        free(ap); free(br); free(d2); free(d1); free(d0);
        return nw_scalar(n, a, b);
    }
    memcpy(ap, a, (size_t)n);
    memset(ap + n, 'X', 80);                       /* north cord, grain forward  */
    for (int k = 0; k < n; k++) br[k] = b[n - 1 - k]; /* east cord, laid reversed */
    memset(br + n, 'Z', 80);

    const __m256i TWO = _mm256_set1_epi16(2);
    const __m256i ONE = _mm256_set1_epi16(1);
    const __m256i GV  = _mm256_set1_epi16(GAP);

    short *A = d2, *B = d1, *C = d0;
    A[0] = 0;                                      /* diagonal 0 */
    B[0] = (short)GAP; B[1] = (short)GAP;          /* diagonal 1 */

    for (int d = 2; d <= 2 * n; d++) {
        int ilo = d - n; if (ilo < 1) ilo = 1;
        int ihi = d - 1; if (ihi > n) ihi = n;
        int cnt = ihi - ilo + 1;
        if (cnt > 0) {
            const char  *pa  = ap + (ilo - 1);
            const char  *pb  = br + (n - d + ilo);
            const short *q2m = A  + (ilo - 1);     /* diagonal d-2, face i-1 */
            const short *q1m = B  + (ilo - 1);     /* diagonal d-1, face i-1 */
            const short *q1  = B  + ilo;           /* diagonal d-1, face i   */
            short       *op  = C  + ilo;
            for (int k = 0; k < cnt; k += 16) {
                __m128i ca = _mm_loadu_si128((const __m128i *)(pa + k));
                __m128i cb = _mm_loadu_si128((const __m128i *)(pb + k));
                __m256i m  = _mm256_cvtepi8_epi16(_mm_cmpeq_epi8(ca, cb));
                __m256i sc = _mm256_sub_epi16(_mm256_and_si256(m, TWO), ONE);
                __m256i dg = _mm256_add_epi16(
                                 _mm256_loadu_si256((const __m256i *)(q2m + k)), sc);
                __m256i up = _mm256_add_epi16(
                                 _mm256_loadu_si256((const __m256i *)(q1m + k)), GV);
                __m256i lf = _mm256_add_epi16(
                                 _mm256_loadu_si256((const __m256i *)(q1  + k)), GV);
                _mm256_storeu_si256((__m256i *)(op + k),
                    _mm256_max_epi16(dg, _mm256_max_epi16(up, lf)));
            }
        }
        if (d <= n) {                              /* the two tray edges */
            short e = (short)(-2 * d);
            C[0] = e; C[d] = e;
        }
        short *t = A; A = B; B = C; C = t;
    }
    int res = (int)B[n];
    free(ap); free(br); free(d2); free(d1); free(d0);
    return res;
}

/* ---- the worm, deep cups: 8 int32 mounds, for trays long enough to overflow ---- */
static int nw_worm32(int n, const char *restrict a, const char *restrict b)
{
    const int PAD = 24;
    char *ap = (char *)malloc((size_t)n + 80);
    char *br = (char *)malloc((size_t)n + 80);
    size_t dl = (size_t)n + 1 + PAD;
    int *d2 = (int *)calloc(dl, sizeof(int));
    int *d1 = (int *)calloc(dl, sizeof(int));
    int *d0 = (int *)calloc(dl, sizeof(int));
    if (!ap || !br || !d2 || !d1 || !d0) {
        free(ap); free(br); free(d2); free(d1); free(d0);
        return nw_scalar(n, a, b);
    }
    memcpy(ap, a, (size_t)n);
    memset(ap + n, 'X', 80);
    for (int k = 0; k < n; k++) br[k] = b[n - 1 - k];
    memset(br + n, 'Z', 80);

    const __m256i TWO = _mm256_set1_epi32(2);
    const __m256i ONE = _mm256_set1_epi32(1);
    const __m256i GV  = _mm256_set1_epi32(GAP);

    int *A = d2, *B = d1, *C = d0;
    A[0] = 0;
    B[0] = GAP; B[1] = GAP;

    for (int d = 2; d <= 2 * n; d++) {
        int ilo = d - n; if (ilo < 1) ilo = 1;
        int ihi = d - 1; if (ihi > n) ihi = n;
        int cnt = ihi - ilo + 1;
        if (cnt > 0) {
            const char *pa  = ap + (ilo - 1);
            const char *pb  = br + (n - d + ilo);
            const int  *q2m = A  + (ilo - 1);
            const int  *q1m = B  + (ilo - 1);
            const int  *q1  = B  + ilo;
            int        *op  = C  + ilo;
            for (int k = 0; k < cnt; k += 8) {
                __m128i ca = _mm_loadl_epi64((const __m128i *)(pa + k));
                __m128i cb = _mm_loadl_epi64((const __m128i *)(pb + k));
                __m256i m  = _mm256_cvtepi8_epi32(_mm_cmpeq_epi8(ca, cb));
                __m256i sc = _mm256_sub_epi32(_mm256_and_si256(m, TWO), ONE);
                __m256i dg = _mm256_add_epi32(
                                 _mm256_loadu_si256((const __m256i *)(q2m + k)), sc);
                __m256i up = _mm256_add_epi32(
                                 _mm256_loadu_si256((const __m256i *)(q1m + k)), GV);
                __m256i lf = _mm256_add_epi32(
                                 _mm256_loadu_si256((const __m256i *)(q1  + k)), GV);
                _mm256_storeu_si256((__m256i *)(op + k),
                    _mm256_max_epi32(dg, _mm256_max_epi32(up, lf)));
            }
        }
        if (d <= n) { int e = -2 * d; C[0] = e; C[d] = e; }
        int *t = A; A = B; B = C; C = t;
    }
    int res = B[n];
    free(ap); free(br); free(d2); free(d1); free(d0);
    return res;
}
#endif /* __AVX2__ */

int kernel(int n, const char *a, const char *b)
{
    if (n <= 0) return 0;
#if defined(__AVX2__)
    if (n < 64)      return nw_scalar(n, a, b);  /* tray too small to wake a worm */
    if (n <= 16000)  return nw_worm16(n, a, b);  /* shallow cups: |score| < 2^15  */
    return nw_worm32(n, a, b);                   /* deep cups                     */
#else
    return nw_scalar(n, a, b);
#endif
}
