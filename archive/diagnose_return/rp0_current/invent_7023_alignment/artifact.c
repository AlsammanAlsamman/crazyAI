#include <stdlib.h>
#include <stdint.h>
#ifdef __AVX2__
#include <immintrin.h>
#endif

#define MT   1
#define MM (-1)
#define GP (-2)
#define NBIG (-(1 << 26))   /* "outside the corridor"; safe from int32 overflow */

/* ---------------- SEED 1: the low sun and the doubled shadow ----------------
   One instruction strikes 32 knots of each rope; equal dyes cast one shadow.
   Returns the Hamming distance, i.e. the g=0 tally, without touching the floor. */
static int ham_dist(int n, const char *a, const char *b)
{
    int i = 0, h = 0;
#ifdef __AVX2__
    for (; i + 32 <= n; i += 32) {
        __m256i va = _mm256_loadu_si256((const __m256i *)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(b + i));
        unsigned m = (unsigned)_mm256_movemask_epi8(_mm256_cmpeq_epi8(va, vb));
        h += 32 - __builtin_popcount(m);
    }
#endif
    for (; i < n; i++) h += (a[i] != b[i]);
    return h;
}

/* ---------------- SEED 3: one slip, priced off the ropes only ----------------
   Alignment shape: diagonal on [0,p), then b runs ahead by s on [p,q), then
   they fall back into step on [q+s,n).  Gaps on each side = s, so
        score = n - 5s - 2x,
        x     = (P0[p]-Ps[p]) + (Ps[q]-P0[q+s]) + H.
   The inner minimisation keeps one running tally and drops every costlier
   one the instant it arrives -- a bad thread-end let fall.  O(n) per s, and
   not one cell of the floor is marked.                                        */
static int slip_bound(int n, const char *a, const char *b,
                      const int *P0, int smax)
{
    int best = NBIG;
    int H = P0[n];
    for (int s = 1; s <= smax; s++) {
        int lim = n - s;
        if (lim < 0) break;
        int Ps = 0;                 /* prefix of (a[u] != b[u+s]) */
        int bestf = 0x3fffffff;     /* min over p <= q of P0[p]-Ps[p] */
        int bestx = 0x3fffffff;
        for (int q = 0; q <= lim; q++) {
            int f = P0[q] - Ps;
            if (f < bestf) bestf = f;
            int cand = bestf + (Ps - P0[q + s]);
            if (cand < bestx) bestx = cand;
            if (q < lim) Ps += (a[q] != b[q + s]);
        }
        int sc = n - 5 * s - 2 * (H + bestx);
        if (sc > best) best = sc;
    }
    return best;
}

/* ---------------- SEED 2: the corridor, and only the corridor ----------------
   Exact Needleman-Wunsch restricted to |i-j| <= k.  Cells outside are NBIG,
   so no path can leak out.  Two rows of memory; the bare floor beyond the
   corridor is never laid.  Rows wider than the sun's beam use the prefix-max
   scan:  t[j] = max(prev[j-1]+sub, prev[j]+GP)          (independent, vector)
          cur[j] = max_{j'<=j} ( t[j'] + GP*(j-j') )     (Hillis-Steele scan) */
static int band_dp(int n, const char *a, const char *b,
                   int k, int *buf, int stride)
{
    int *prev = buf, *cur = buf + stride;
    int hp = (k < n) ? k : n;
    for (int j = 0; j <= hp; j++) prev[j] = GP * j;
    prev[hp + 1] = NBIG;
#ifdef __AVX2__
    const __m256i idx2 = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
    const __m256i vneg = _mm256_set1_epi32(NBIG);
    const __m256i vmt  = _mm256_set1_epi32(MT);
    const __m256i vmm  = _mm256_set1_epi32(MM);
    const __m256i vgp  = _mm256_set1_epi32(GP);
#endif
    for (int i = 1; i <= n; i++) {
        int lo = i - k; if (lo < 1) lo = 1;
        int hi = i + k; if (hi > n) hi = n;
        if (lo == 1) cur[0] = (i <= k) ? GP * i : NBIG;
        else         cur[lo - 1] = NBIG;
        const char ai = a[i - 1];
        int carry = cur[lo - 1];
        int j = lo;
#ifdef __AVX2__
        if (hi - lo + 1 >= 24) {
            const __m128i vai8 = _mm_set1_epi8(ai);
            for (; j + 7 <= hi; j += 8) {
                __m256i pd = _mm256_loadu_si256((const __m256i *)(prev + j - 1));
                __m256i pu = _mm256_loadu_si256((const __m256i *)(prev + j));
                __m128i bb = _mm_loadl_epi64((const __m128i *)(b + j - 1));
                __m256i eq = _mm256_cvtepi8_epi32(_mm_cmpeq_epi8(bb, vai8));
                __m256i sb = _mm256_blendv_epi8(vmm, vmt, eq);
                __m256i t  = _mm256_max_epi32(_mm256_add_epi32(pd, sb),
                                              _mm256_add_epi32(pu, vgp));
                __m256i w  = _mm256_add_epi32(t, idx2);      /* w[p] = t[p] + 2p */
                __m256i s;
                s = _mm256_alignr_epi8(w, _mm256_permute2x128_si256(w, vneg, 0x02), 12);
                w = _mm256_max_epi32(w, s);                  /* shift 1 lane */
                s = _mm256_alignr_epi8(w, _mm256_permute2x128_si256(w, vneg, 0x02), 8);
                w = _mm256_max_epi32(w, s);                  /* shift 2 lanes */
                s = _mm256_permute2x128_si256(w, vneg, 0x02);
                w = _mm256_max_epi32(w, s);                  /* shift 4 lanes */
                w = _mm256_max_epi32(w, _mm256_set1_epi32(carry + GP));
                __m256i r = _mm256_sub_epi32(w, idx2);
                _mm256_storeu_si256((__m256i *)(cur + j), r);
                carry = _mm256_extract_epi32(r, 7);
            }
        }
#endif
        for (; j <= hi; j++) {
            int d = prev[j - 1] + ((ai == b[j - 1]) ? MT : MM);
            int u = prev[j] + GP;
            int t = d > u ? d : u;
            int l = carry + GP;
            int v = t > l ? t : l;
            cur[j] = v;
            carry  = v;
        }
        cur[hi + 1] = NBIG;
        { int *tmp = prev; prev = cur; cur = tmp; }
    }
    return prev[n];
}

int kernel(int n, const char *a, const char *b)
{
    if (n <= 0) return 0;

    /* Below the cost of the sun itself: pace the room the plain way. */
    if (n < 64) {
        int prev[64], cur[64];
        for (int j = 0; j <= n; j++) prev[j] = GP * j;
        for (int i = 1; i <= n; i++) {
            char ai = a[i - 1];
            cur[0] = GP * i;
            for (int j = 1; j <= n; j++) {
                int d = prev[j - 1] + ((ai == b[j - 1]) ? MT : MM);
                int u = prev[j] + GP;
                int t = d > u ? d : u;
                int l = cur[j - 1] + GP;
                cur[j] = t > l ? t : l;
            }
            for (int j = 0; j <= n; j++) prev[j] = cur[j];
        }
        return prev[n];
    }

    /* SEED 1: the cat's census, in one sweep of sunlight. */
    int H  = ham_dist(n, a, b);
    int L0 = n - 2 * H;                 /* a real alignment, so L0 <= S      */
    int L  = L0;
    int k  = (n - L) / 5;               /* every slip costs exactly 5        */
    if (k <= 0) return L0;              /* no room to slip: answer is exact  */
    if (k > n) k = n;

    int stride = n + 18;
    int *mem = (int *)malloc((size_t)(2 * stride + n + 2) * sizeof(int));
    if (!mem) {                         /* degrade to the textbook path      */
        int *p = (int *)malloc((size_t)(2 * (n + 2)) * sizeof(int));
        if (!p) return L0;
        int *pr = p, *cu = p + (n + 2);
        for (int j = 0; j <= n; j++) pr[j] = GP * j;
        for (int i = 1; i <= n; i++) {
            char ai = a[i - 1]; cu[0] = GP * i;
            for (int j = 1; j <= n; j++) {
                int d = pr[j - 1] + ((ai == b[j - 1]) ? MT : MM);
                int u = pr[j] + GP; int t = d > u ? d : u;
                int l = cu[j - 1] + GP; cu[j] = t > l ? t : l;
            }
            { int *tm = pr; pr = cu; cu = tm; }
        }
        int r = pr[n]; free(p); return r;
    }
    int *buf = mem;
    int *P0  = mem + 2 * stride;

    /* SEED 3: if the corridor is wide, ask whether one slip explains it.
       Gated so the probe can never cost more than the floor it saves.      */
    if (k >= 16) {
        P0[0] = 0;
        for (int t = 0; t < n; t++) P0[t + 1] = P0[t] + (a[t] != b[t]);
        int smax = (k < 24) ? k : 24;
        int s1 = slip_bound(n, a, b, P0, smax);   /* b runs ahead */
        int s2 = slip_bound(n, b, a, P0, smax);   /* a runs ahead */
        if (s1 > L) L = s1;
        if (s2 > L) L = s2;
        int nk = (n - L) / 5;
        if (nk < k) k = nk;
        if (k <= 0) { free(mem); return L0; }
    }

    /* One cheap narrow pass can certify itself and retire the wide one.
       If it cannot, its score still tightens the corridor.                 */
    if (k > 48) {
        int kp = 24;
        int sp = band_dp(n, a, b, kp, buf, stride);
        int nk = (n - sp) / 5;
        if (nk <= kp) { free(mem); return sp; }   /* certificate: exact      */
        if (nk < k) k = nk;
    }

    int res = band_dp(n, a, b, k, buf, stride);
    free(mem);
    return res;
}
