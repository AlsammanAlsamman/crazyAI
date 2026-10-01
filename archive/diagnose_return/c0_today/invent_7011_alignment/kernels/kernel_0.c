#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#define NW_MATCH      1
#define NW_MISMATCH (-1)
#define NW_GAP      (-2)

/* ---------------- the crease: one vector of int16 sparks ---------------- */
#if defined(__AVX512BW__)
  #include <immintrin.h>
  #define NW_VEC 1
  #define NW_VW  32
  typedef __m512i nwv;
  #define nwv_load(p)     _mm512_loadu_si512((const void *)(p))
  #define nwv_store(p,v)  _mm512_storeu_si512((void *)(p), (v))
  #define nwv_add(x,y)    _mm512_add_epi16((x),(y))
  #define nwv_max(x,y)    _mm512_max_epi16((x),(y))
  #define nwv_set1(x)     _mm512_set1_epi16((short)(x))
  static inline nwv nwv_score(const char *pa, const char *pb) {
      __m256i eq8 = _mm256_cmpeq_epi8(_mm256_loadu_si256((const __m256i *)pa),
                                      _mm256_loadu_si256((const __m256i *)pb));
      __m512i eq  = _mm512_cvtepi8_epi16(eq8);          /* 0 or -1 per lane */
      return _mm512_sub_epi16(_mm512_set1_epi16(-1), _mm512_add_epi16(eq, eq));
  }
#elif defined(__AVX2__)
  #include <immintrin.h>
  #define NW_VEC 1
  #define NW_VW  16
  typedef __m256i nwv;
  #define nwv_load(p)     _mm256_loadu_si256((const __m256i *)(p))
  #define nwv_store(p,v)  _mm256_storeu_si256((__m256i *)(p), (v))
  #define nwv_add(x,y)    _mm256_add_epi16((x),(y))
  #define nwv_max(x,y)    _mm256_max_epi16((x),(y))
  #define nwv_set1(x)     _mm256_set1_epi16((short)(x))
  static inline nwv nwv_score(const char *pa, const char *pb) {
      __m128i eq8 = _mm_cmpeq_epi8(_mm_loadu_si128((const __m128i *)pa),
                                   _mm_loadu_si128((const __m128i *)pb));
      __m256i eq  = _mm256_cvtepi8_epi16(eq8);
      return _mm256_sub_epi16(_mm256_set1_epi16(-1), _mm256_add_epi16(eq, eq));
  }
#elif defined(__SSE2__) || defined(__x86_64__) || defined(_M_X64)
  #include <emmintrin.h>
  #define NW_VEC 1
  #define NW_VW  8
  typedef __m128i nwv;
  #define nwv_load(p)     _mm_loadu_si128((const __m128i *)(p))
  #define nwv_store(p,v)  _mm_storeu_si128((__m128i *)(p), (v))
  #define nwv_add(x,y)    _mm_add_epi16((x),(y))
  #define nwv_max(x,y)    _mm_max_epi16((x),(y))
  #define nwv_set1(x)     _mm_set1_epi16((short)(x))
  static inline nwv nwv_score(const char *pa, const char *pb) {
      __m128i eq8 = _mm_cmpeq_epi8(_mm_loadl_epi64((const __m128i *)pa),
                                   _mm_loadl_epi64((const __m128i *)pb));
      __m128i eq  = _mm_unpacklo_epi8(eq8, eq8);        /* 0x0000 / 0xFFFF */
      return _mm_sub_epi16(_mm_set1_epi16(-1), _mm_add_epi16(eq, eq));
  }
#endif

/* ------------- flat-walker fallback: two rolling rows, int32 ------------- */
static int nw_rows(int n, const char *a, const char *b)
{
    int sprev[257], scur[257];
    int *prev, *cur, *heap = NULL;
    if (n <= 256) { prev = sprev; cur = scur; }
    else {
        heap = (int *)malloc((size_t)2 * (n + 1) * sizeof(int));
        if (!heap) return 0;
        prev = heap; cur = heap + (n + 1);
    }
    for (int j = 0; j <= n; ++j) prev[j] = j * NW_GAP;
    for (int i = 1; i <= n; ++i) {
        const char ai = a[i - 1];
        cur[0] = i * NW_GAP;
        int left = cur[0];
        for (int j = 1; j <= n; ++j) {
            int dg = prev[j - 1] + (ai == b[j - 1] ? NW_MATCH : NW_MISMATCH);
            int up = prev[j] + NW_GAP;
            int lf = left + NW_GAP;
            int best = dg > up ? dg : up;
            if (lf > best) best = lf;
            cur[j] = best; left = best;
        }
        int *t = prev; prev = cur; cur = t;
    }
    int r = prev[n];
    free(heap);
    return r;
}

#ifdef NW_VEC
#define NW_STACK_N 2048
#define NW_SLACK   (2 * NW_VW + 8)
#define NW_STACK_V (3 * (NW_STACK_N + 2 + 2 * 32 + 8))
#define NW_STACK_C (2 * (NW_STACK_N + 2 * 32 + 8))

/* fold the lattice along i+j; keep only three creases; band half-width W */
static int nw_diag(int n, const char *a, const char *b, int W)
{
    const int L = n + 2 + NW_SLACK;   /* one crease buffer, indexed by i */
    const int C = n + NW_SLACK;       /* one padded bough                */
    int16_t stack_v[NW_STACK_V];
    char    stack_c[NW_STACK_C];
    int16_t *v; char *ch; void *heap = NULL;

    if (n <= NW_STACK_N) { v = stack_v; ch = stack_c; }
    else {
        heap = malloc((size_t)3 * L * sizeof(int16_t) + (size_t)2 * C + 64);
        if (!heap) return nw_rows(n, a, b);
        v  = (int16_t *)heap;
        ch = (char *)((int16_t *)heap + (size_t)3 * L);
    }

    int16_t *p2 = v, *p1 = v + L, *p0 = v + 2 * L;
    char *ca = ch, *cb = ch + C;
    const int16_t NEG = (int16_t)(-2 * n - 16);   /* unreachable: dp >= -2n */

    /* one bough hangs forward, the other hangs backward, so a crease reads
       both contiguously: b[j-1] with j = d-i  ==  cb[n-d+i]               */
    memcpy(ca, a, (size_t)n);
    for (int k = n; k < C; ++k) ca[k] = 1;
    for (int k = 0; k < n; ++k) cb[k] = b[n - 1 - k];
    for (int k = n; k < C; ++k) cb[k] = 2;

    for (int k = 0; k < 3 * L; ++k) v[k] = NEG;

    p2[0] = 0;                        /* crease d=0 : dp[0][0]             */
    p1[0] = (int16_t)NW_GAP;          /* crease d=1 : dp[0][1]             */
    p1[1] = (int16_t)NW_GAP;          /*              dp[1][0]             */

    const nwv vgap = nwv_set1(NW_GAP);

    for (int d = 2; d <= 2 * n; ++d) {
        int blo = (d - W + 1) / 2;            /* band: |2i - d| <= W       */
        int bhi = (d + W) >> 1;
        int ilo = d - n; if (ilo < 0) ilo = 0;
        int ihi = d < n ? d : n;
        if (blo < ilo) blo = ilo;
        if (bhi > ihi) bhi = ihi;

        int lo = blo < 1 ? 1 : blo;           /* interior cells only       */
        int hi = bhi < d - 1 ? bhi : d - 1;

        if (hi >= lo) {
            const char    *pa = ca + (lo - 1);
            const char    *pb = cb + (n - d + lo);
            const int16_t *q2 = p2 + lo - 1;  /* layer beneath : diagonal  */
            const int16_t *q1 = p1 + lo - 1;  /* layer beside  : up / left */
            int16_t       *q0 = p0 + lo;
            const int len = hi - lo + 1;
            for (int t = 0; t < len; t += NW_VW) {
                nwv sc = nwv_score(pa + t, pb + t);
                nwv dg = nwv_add(nwv_load(q2 + t), sc);
                nwv up = nwv_add(nwv_load(q1 + t), vgap);
                nwv lf = nwv_add(nwv_load(q1 + t + 1), vgap);
                nwv_store(q0 + t, nwv_max(nwv_max(dg, up), lf));
            }
        }
        /* unlit windows just outside the band read as minus-infinity */
        if (blo - 1 >= 0) p0[blo - 1] = NEG;
        if (blo - 2 >= 0) p0[blo - 2] = NEG;
        p0[bhi + 1] = NEG;
        p0[bhi + 2] = NEG;
        /* the two border sparks, in-band only while d <= W */
        if (d <= n && d <= W) {
            int16_t g = (int16_t)(-2 * d);
            p0[0] = g; p0[d] = g;
        }
        /* the spent bottom layer drifts off into the gradient */
        int16_t *t = p2; p2 = p1; p1 = p0; p0 = t;
    }

    int res = (int)p1[n];             /* last crease, far corner */
    free(heap);
    return res;
}
#endif /* NW_VEC */

int kernel(int n, const char *a, const char *b)
{
    if (n <= 0) return 0;
#ifdef NW_VEC
    /* fold only when a crease can fill a vector and an int16 spark holds
       the whole score range ([-2n, n]); otherwise walk flat.             */
    if (n >= 64 && n <= 16000) {
        int s = 0;
        for (int k = 0; k < n; ++k) s += (a[k] == b[k]) ? NW_MATCH : NW_MISMATCH;
        /* any path at offset t scores <= n - 5t; the main crease already
           scores s, so no optimal path leaves |i-j| <= (n-s)/5.          */
        int W = (n - s) / 5 + 2;
        if (W > n) W = n;             /* W == n  =>  exact full wavefront */
        return nw_diag(n, a, b, W);
    }
#endif
    return nw_rows(n, a, b);
}
