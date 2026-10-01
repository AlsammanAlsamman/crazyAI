#include <stdlib.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define MATCH     1
#define MISMATCH (-1)
#define GAPP     (-2)
#define NEGINF   (-(1 << 28))

/* ===================== THE SUNLIT WATER TABLE =============================
   Both coil rows laid nose to tail along the same edge.  Where a shape in the
   near row coincides with its opposite in the shadow row the sun windows
   through both and a fire drops on the water.  We count fires, nothing more.
   32 coils are judged by one instruction; no grid is built.                 */
static int fires_straight(int n, const char *a, const char *b)
{
    int i = 0, m = 0;
#if defined(__AVX2__)
    for (; i + 32 <= n; i += 32) {
        __m256i va = _mm256_loadu_si256((const __m256i *)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(b + i));
        m += __builtin_popcount(
                 (unsigned)_mm256_movemask_epi8(_mm256_cmpeq_epi8(va, vb)));
    }
#endif
    for (; i < n; i++) m += (a[i] == b[i]);
    return m;
}

/* ===================== THE DOOR THAT BREAKS ONCE ==========================
   Every crossing in which the door breaks exactly once: one coil crouches
   forward past a quarrel at some place i, the rows re-square at some later
   place j, and the crossing still ends at the queen's threshold.  Its pile is
        m(i,j) = PD[n] + A[i] + B[j],     A[i] = PD[i]-PX[i+1]
                                          B[j] = PX[j+1]-PD[j+1]
   so every slip point is tried at once in a single pass carrying max A.
   dir=+1: the shadow row crouches (pairs a[p] with b[p-1] inside the slip)
   dir=-1: the near row crouches   (pairs a[p-1] with b[p] inside the slip)
   Returns the extra fires won over the straight crossing, or NEGINF.       */
static int best_one_slip_gain(int n, const char *a, const char *b, int dir)
{
    int t = 0;                  /* PD[x] - PX[x] */
    int bestA = NEGINF, best = NEGINF;
    for (int x = 0; x < n; x++) {
        int ed = (a[x] == b[x]);
        int ex = 0;
        if (x > 0) ex = (dir > 0) ? (a[x] == b[x - 1]) : (a[x - 1] == b[x]);
        if (x >= 1) {                       /* close a door opened at some i<x */
            int c = bestA + (ex - ed - t);
            if (c > best) best = c;
        }
        int A = t - ex;                     /* open a door here */
        if (A > bestA) bestA = A;
        t += ed - ex;
    }
    return best;
}

/* ===================== OPENING THE TOWER ==================================
   The attempt started over from the first coil, with the door now allowed to
   break up to kap times.  The two rows are walked together along one advancing
   front (an anti-diagonal): on that front no coil waits on the coil to its
   left, so the whole front is judged in parallel.  Ukkonen's band, reached by
   the native's own rule rather than borrowed.                              */
static int band_crossing(int n, const char *a, const char *br,
                         int kap, int *pool, int stride)
{
    int *r0 = pool + 1;                 /* front d-2 */
    int *r1 = pool + stride + 1;        /* front d-1 */
    int *r2 = pool + 2 * stride + 1;    /* front d   */
    if (kap < 1) kap = 1;
    for (int t = -1; t <= n + 1; t++) { r0[t] = NEGINF; r1[t] = NEGINF; r2[t] = NEGINF; }
    r1[0] = 0;                          /* the first coil pair */
#if defined(__AVX2__)
    const __m256i vgap = _mm256_set1_epi32(GAPP);
    const __m256i vm1  = _mm256_set1_epi32(-1);
#endif
    for (int d = 1; d <= 2 * n; d++) {
        int lo = d - kap;
        lo = (lo <= 0) ? 0 : ((lo + 1) >> 1);
        if (lo < d - n) lo = d - n;
        int hi = (d + kap) >> 1;
        if (hi > n) hi = n;
        if (hi > d) hi = d;

        int istart = lo, iend = hi;
        if (lo == 0) { r2[0] = GAPP * d; istart = 1; }   /* near row all crouched */
        if (hi == d) { r2[d] = GAPP * d; iend = d - 1; } /* shadow row all crouched */

        int i = istart;
#if defined(__AVX2__)
        int base = n - d;
        for (; i + 7 <= iend; i += 8) {
            __m128i ca  = _mm_loadl_epi64((const __m128i *)(a + (i - 1)));
            __m128i cb  = _mm_loadl_epi64((const __m128i *)(br + (base + i)));
            __m256i m32 = _mm256_cvtepi8_epi32(_mm_cmpeq_epi8(ca, cb));
            __m256i s   = _mm256_sub_epi32(vm1, _mm256_add_epi32(m32, m32));
            __m256i vd  = _mm256_add_epi32(
                              _mm256_loadu_si256((const __m256i *)(r0 + i - 1)), s);
            __m256i vu  = _mm256_add_epi32(
                              _mm256_loadu_si256((const __m256i *)(r1 + i - 1)), vgap);
            __m256i vl  = _mm256_add_epi32(
                              _mm256_loadu_si256((const __m256i *)(r1 + i)), vgap);
            _mm256_storeu_si256((__m256i *)(r2 + i),
                                _mm256_max_epi32(vd, _mm256_max_epi32(vu, vl)));
        }
#endif
        for (; i <= iend; i++) {
            int s = (a[i - 1] == br[n - d + i]) ? MATCH : MISMATCH;
            int v = r0[i - 1] + s;
            int u = r1[i - 1] + GAPP;
            int w = r1[i] + GAPP;
            if (u > v) v = u;
            if (w > v) v = w;
            r2[i] = v;
        }
        r2[lo - 1] = NEGINF;
        r2[hi + 1] = NEGINF;

        int *tmp = r0; r0 = r1; r1 = r2; r2 = tmp;
    }
    return r1[n];                       /* the queen's threshold */
}

/* plain walk, for rows too short for any of this to pay for itself */
static int plain_dp_small(int n, const char *a, const char *b)
{
    int prev[64], cur[64];
    for (int j = 0; j <= n; j++) prev[j] = GAPP * j;
    for (int i = 1; i <= n; i++) {
        char ai = a[i - 1];
        cur[0] = GAPP * i;
        for (int j = 1; j <= n; j++) {
            int v = prev[j - 1] + (ai == b[j - 1] ? MATCH : MISMATCH);
            int u = prev[j] + GAPP;
            int w = cur[j - 1] + GAPP;
            if (u > v) v = u;
            if (w > v) v = w;
            cur[j] = v;
        }
        for (int j = 0; j <= n; j++) prev[j] = cur[j];
    }
    return prev[n];
}

static int plain_dp_heap(int n, const char *a, const char *b)
{
    int *buf = (int *)malloc((size_t)2 * (n + 1) * sizeof(int));
    if (!buf) return 0;
    int *prev = buf, *cur = buf + (n + 1);
    for (int j = 0; j <= n; j++) prev[j] = GAPP * j;
    for (int i = 1; i <= n; i++) {
        char ai = a[i - 1];
        cur[0] = GAPP * i;
        for (int j = 1; j <= n; j++) {
            int v = prev[j - 1] + (ai == b[j - 1] ? MATCH : MISMATCH);
            int u = prev[j] + GAPP;
            int w = cur[j - 1] + GAPP;
            if (u > v) v = u;
            if (w > v) v = w;
            cur[j] = v;
        }
        int *t = prev; prev = cur; cur = t;
    }
    int r = prev[n];
    free(buf);
    return r;
}

int kernel(int n, const char *a, const char *b)
{
    if (n <= 0) return 0;
    if (n < 64) return plain_dp_small(n, a, b);          /* size guard */

    /* ---- room 0: the straight crossing ---- */
    int m0 = fires_straight(n, a, b);
    int L  = 2 * m0 - n;

    /* ---- one room per place the door could break, all at once ---- */
    int gp = best_one_slip_gain(n, a, b,  1);
    int gm = best_one_slip_gain(n, a, b, -1);
    int g  = (gp > gm) ? gp : gm;
    if (g > NEGINF / 2) {
        int s1 = 2 * (m0 + g) - n - 3;                   /* one broken door: -3 */
        if (s1 > L) L = s1;
    }

    /* ---- the tower: every twice-broken crossing burns at most n-10 ----
       so if the highest pile already reaches that, the tower stays shut and
       the one-slip answer is exact.                                         */
    if (L >= n - 10) return L;

    /* ---- start the attempt over, door allowed to break kap times ---- */
    int stride = n + 3;
    char *br   = (char *)malloc((size_t)n + 8);
    int  *pool = (int  *)malloc((size_t)3 * stride * sizeof(int));
    if (!br || !pool) { free(br); free(pool); return plain_dp_heap(n, a, b); }
    for (int i = 0; i < n; i++) br[i] = b[n - 1 - i];    /* shadow row reversed:
                                                            the advancing front
                                                            reads it forwards  */
    int kneed = (n - L + 4) / 5;
    if (kneed < 1) kneed = 1;
    int res;

    if (kneed > 4) {                    /* cheap rung: four broken doors */
        res = band_crossing(n, a, br, 4, pool, stride);
        if (res > L) L = res;
        if (4 >= (n - L + 4) / 5) { free(br); free(pool); return L; }
        kneed = (n - L + 4) / 5;
        if (kneed > 32) {               /* cheap rung: thirty-two */
            res = band_crossing(n, a, br, 32, pool, stride);
            if (res > L) L = res;
            if (32 >= (n - L + 4) / 5) { free(br); free(pool); return L; }
            kneed = (n - L + 4) / 5;
        }
    }
    if (kneed > n) kneed = n;
    if (kneed < 1) kneed = 1;
    res = band_crossing(n, a, br, kneed, pool, stride);  /* provably sufficient */
    free(br); free(pool);
    return res;
}
