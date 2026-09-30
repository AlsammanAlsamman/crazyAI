#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
  #include <immintrin.h>
#endif

#define MATCH     1
#define MISMATCH -1
#define GAP      -2

/* ============================================================
   REGIME 0 -- "the bough is shorter than one spark-read":
   do not fold at all, walk flat.  Plain rolling-row NW on the
   stack.  Guards the folding overhead at tiny n.            */
static int nw_flat(int n, const char *restrict a, const char *restrict b)
{
    int prev[40], cur[40];              /* n < 32  =>  n+1 <= 32 */
    for (int j = 0; j <= n; j++) prev[j] = j * GAP;
    for (int i = 1; i <= n; i++) {
        const char ai = a[i - 1];
        cur[0] = i * GAP;
        for (int j = 1; j <= n; j++) {
            int s = prev[j - 1] + ((ai == b[j - 1]) ? MATCH : MISMATCH);
            int u = prev[j]     + GAP;
            int l = cur[j - 1]  + GAP;
            int best = s;
            if (u > best) best = u;
            if (l > best) best = l;
            cur[j] = best;
        }
        memcpy(prev, cur, (size_t)(n + 1) * sizeof(int));
    }
    return prev[n];
}

/* ============================================================
   REGIME 1 -- half-width lamps.  |score| <= 2n+2 < 32768.
   The fold: three creases pressed together, bottom one let go.
   cu[i] = max( p2[i-1] + s ,  max(p1[i-1], p1[i]) - 2 )
   with  s = (a[i-1] == b[d-i-1]) ? +1 : -1
   and   b stored REVERSED so b[d-i-1] == cb[n-d+i], unit stride. */
static int nw_fold16(int n, const char *a, const char *b)
{
    const int    N    = n;
    const size_t PAD  = 64;
    const size_t bufn = (size_t)N + 2 * PAD;

    int16_t *mem  = (int16_t *)malloc(3 * bufn * sizeof(int16_t));
    char    *cmem = (char    *)malloc(2 * ((size_t)N + 2 * PAD));
    if (!mem || !cmem) { free(mem); free(cmem); return nw_flat(0, a, b); }
    memset(mem, 0, 3 * bufn * sizeof(int16_t));

    int16_t *p2 = mem, *p1 = mem + bufn, *cu = mem + 2 * bufn;

    char *ca = cmem;                        /* bough A, forward  */
    char *cb = cmem + (size_t)N + 2 * PAD;  /* bough B, reversed */
    memcpy(ca, a, (size_t)N);
    memset(ca + N, 0x00, 2 * PAD);
    for (int k = 0; k < N; k++) cb[k] = b[N - 1 - k];
    memset(cb + N, 0x7f, 2 * PAD);          /* filler never matches */

    p2[0] = 0;                              /* crease d = 0 */
    p1[0] = (int16_t)GAP;                   /* crease d = 1 */
    p1[1] = (int16_t)GAP;

    for (int d = 2; d <= 2 * N; d++) {
        int ilo = d - N; if (ilo < 1) ilo = 1;
        int ihi = d - 1; if (ihi > N) ihi = N;
        const char *pb = cb + (N - d);      /* pb[i] == b[d-i-1] */

#if defined(__AVX512BW__) && defined(__AVX512F__)
        {
            const __m512i v2 = _mm512_set1_epi16(2);
            const __m512i v1 = _mm512_set1_epi16(1);
            for (int i = ilo; i <= ihi; i += 32) {
                __m512i pd = _mm512_loadu_si512((const void *)(p2 + i - 1));
                __m512i uu = _mm512_loadu_si512((const void *)(p1 + i - 1));
                __m512i ll = _mm512_loadu_si512((const void *)(p1 + i));
                __m256i av = _mm256_loadu_si256((const __m256i *)(ca + i - 1));
                __m256i bv = _mm256_loadu_si256((const __m256i *)(pb + i));
                __m256i eq = _mm256_cmpeq_epi8(av, bv);
                __m512i ss = _mm512_sub_epi16(
                                 _mm512_and_si512(_mm512_cvtepi8_epi16(eq), v2), v1);
                __m512i dg = _mm512_add_epi16(pd, ss);
                __m512i gp = _mm512_sub_epi16(_mm512_max_epi16(uu, ll), v2);
                _mm512_storeu_si512((void *)(cu + i), _mm512_max_epi16(dg, gp));
            }
        }
#elif defined(__AVX2__)
        {
            const __m256i v2 = _mm256_set1_epi16(2);
            const __m256i v1 = _mm256_set1_epi16(1);
            for (int i = ilo; i <= ihi; i += 16) {
                __m256i pd = _mm256_loadu_si256((const __m256i *)(p2 + i - 1));
                __m256i uu = _mm256_loadu_si256((const __m256i *)(p1 + i - 1));
                __m256i ll = _mm256_loadu_si256((const __m256i *)(p1 + i));
                __m128i av = _mm_loadu_si128((const __m128i *)(ca + i - 1));
                __m128i bv = _mm_loadu_si128((const __m128i *)(pb + i));
                __m128i eq = _mm_cmpeq_epi8(av, bv);
                __m256i ss = _mm256_sub_epi16(
                                 _mm256_and_si256(_mm256_cvtepi8_epi16(eq), v2), v1);
                __m256i dg = _mm256_add_epi16(pd, ss);
                __m256i gp = _mm256_sub_epi16(_mm256_max_epi16(uu, ll), v2);
                _mm256_storeu_si256((__m256i *)(cu + i), _mm256_max_epi16(dg, gp));
            }
        }
#else
        {
            const int16_t *restrict q2 = p2;
            const int16_t *restrict q1 = p1;
            int16_t       *restrict qc = cu;
            for (int i = ilo; i <= ihi; i++) {
                int s = (int)q2[i - 1] + ((ca[i - 1] == pb[i]) ? MATCH : MISMATCH);
                int u = (int)q1[i - 1] + GAP;
                int l = (int)q1[i]     + GAP;
                int best = s;
                if (u > best) best = u;
                if (l > best) best = l;
                qc[i] = (int16_t)best;
            }
        }
#endif
        if (d <= N) {                       /* the two rim sparks of this crease */
            cu[0] = (int16_t)(GAP * d);     /* dp[0][d] */
            cu[d] = (int16_t)(GAP * d);     /* dp[d][0] */
        }
        { int16_t *t = p2; p2 = p1; p1 = cu; cu = t; }  /* bottom layer drifts off */
    }

    int res = (int)p1[N];                   /* last crease, far corner */
    free(mem); free(cmem);
    return res;
}

/* ============================================================
   REGIME 2 -- full-width lamps.  Same fold, int32 sparks,
   8 per crease-read.  Used when a spark could outshine int16. */
static int nw_fold32(int n, const char *a, const char *b)
{
    const int    N    = n;
    const size_t PAD  = 64;
    const size_t bufn = (size_t)N + 2 * PAD;

    int32_t *mem  = (int32_t *)malloc(3 * bufn * sizeof(int32_t));
    char    *cmem = (char    *)malloc(2 * ((size_t)N + 2 * PAD));
    if (!mem || !cmem) { free(mem); free(cmem); return 0; }
    memset(mem, 0, 3 * bufn * sizeof(int32_t));

    int32_t *p2 = mem, *p1 = mem + bufn, *cu = mem + 2 * bufn;

    char *ca = cmem;
    char *cb = cmem + (size_t)N + 2 * PAD;
    memcpy(ca, a, (size_t)N);
    memset(ca + N, 0x00, 2 * PAD);
    for (int k = 0; k < N; k++) cb[k] = b[N - 1 - k];
    memset(cb + N, 0x7f, 2 * PAD);

    p2[0] = 0;
    p1[0] = GAP;
    p1[1] = GAP;

    for (int d = 2; d <= 2 * N; d++) {
        int ilo = d - N; if (ilo < 1) ilo = 1;
        int ihi = d - 1; if (ihi > N) ihi = N;
        const char *pb = cb + (N - d);

#if defined(__AVX2__)
        {
            const __m256i v2 = _mm256_set1_epi32(2);
            const __m256i v1 = _mm256_set1_epi32(1);
            for (int i = ilo; i <= ihi; i += 8) {
                __m256i pd = _mm256_loadu_si256((const __m256i *)(p2 + i - 1));
                __m256i uu = _mm256_loadu_si256((const __m256i *)(p1 + i - 1));
                __m256i ll = _mm256_loadu_si256((const __m256i *)(p1 + i));
                __m128i av = _mm_loadl_epi64((const __m128i *)(ca + i - 1));
                __m128i bv = _mm_loadl_epi64((const __m128i *)(pb + i));
                __m128i eq = _mm_cmpeq_epi8(av, bv);
                __m256i ss = _mm256_sub_epi32(
                                 _mm256_and_si256(_mm256_cvtepi8_epi32(eq), v2), v1);
                __m256i dg = _mm256_add_epi32(pd, ss);
                __m256i gp = _mm256_sub_epi32(_mm256_max_epi32(uu, ll), v2);
                _mm256_storeu_si256((__m256i *)(cu + i), _mm256_max_epi32(dg, gp));
            }
        }
#else
        {
            const int32_t *restrict q2 = p2;
            const int32_t *restrict q1 = p1;
            int32_t       *restrict qc = cu;
            for (int i = ilo; i <= ihi; i++) {
                int s = q2[i - 1] + ((ca[i - 1] == pb[i]) ? MATCH : MISMATCH);
                int u = q1[i - 1] + GAP;
                int l = q1[i]     + GAP;
                int best = s;
                if (u > best) best = u;
                if (l > best) best = l;
                qc[i] = best;
            }
        }
#endif
        if (d <= N) { cu[0] = GAP * d; cu[d] = GAP * d; }
        { int32_t *t = p2; p2 = p1; p1 = cu; cu = t; }
    }

    int res = p1[N];
    free(mem); free(cmem);
    return res;
}

/* ============================================================ */
int kernel(int n, const char *a, const char *b)
{
    if (n <= 0)      return 0;
    if (n < 32)      return nw_flat(n, a, b);    /* crease too short to fold  */
    if (n <= 16000)  return nw_fold16(n, a, b);  /* half-width lamps          */
    return nw_fold32(n, a, b);                   /* full-width lamps          */
}
