#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#if defined(__AVX512F__) || defined(__AVX2__) || defined(__AVX__)
#include <immintrin.h>
#endif

/* ---- block scans: len is always a multiple of 8 and >= 8 ---- */
#if defined(__AVX512F__)
static inline double blk_min(const double *a, int len) {
    __m512d m = _mm512_loadu_pd(a);
    for (int i = 8; i < len; i += 8) m = _mm512_min_pd(m, _mm512_loadu_pd(a + i));
    return _mm512_reduce_min_pd(m);
}
static inline int blk_argmin(const double *a, int len, double *pm) {
    double mv = blk_min(a, len); *pm = mv;
    __m512d t = _mm512_set1_pd(mv);
    for (int i = 0; i < len; i += 8) {
        __mmask8 k = _mm512_cmp_pd_mask(_mm512_loadu_pd(a + i), t, _CMP_EQ_OQ);
        if (k) return i + (int)__builtin_ctz((unsigned)k);
    }
    return 0;
}
#elif defined(__AVX__)
static inline double blk_min(const double *a, int len) {
    __m256d m0 = _mm256_loadu_pd(a), m1 = _mm256_loadu_pd(a + 4);
    for (int i = 8; i < len; i += 8) {
        m0 = _mm256_min_pd(m0, _mm256_loadu_pd(a + i));
        m1 = _mm256_min_pd(m1, _mm256_loadu_pd(a + i + 4));
    }
    m0 = _mm256_min_pd(m0, m1);
    __m128d lo = _mm256_castpd256_pd128(m0);
    __m128d hi = _mm256_extractf128_pd(m0, 1);
    lo = _mm_min_pd(lo, hi);
    lo = _mm_min_sd(lo, _mm_unpackhi_pd(lo, lo));
    return _mm_cvtsd_f64(lo);
}
static inline int blk_argmin(const double *a, int len, double *pm) {
    double mv = blk_min(a, len); *pm = mv;
    __m256d t = _mm256_set1_pd(mv);
    for (int i = 0; i < len; i += 4) {
        int msk = _mm256_movemask_pd(_mm256_cmp_pd(_mm256_loadu_pd(a + i), t, _CMP_EQ_OQ));
        if (msk) return i + (int)__builtin_ctz((unsigned)msk);
    }
    return 0;
}
#else
static inline double blk_min(const double *a, int len) {
    double m = a[0];
    for (int i = 1; i < len; i++) if (a[i] < m) m = a[i];
    return m;
}
static inline int blk_argmin(const double *a, int len, double *pm) {
    double m = a[0]; int b = 0;
    for (int i = 1; i < len; i++) if (a[i] < m) { m = a[i]; b = i; }
    *pm = m; return b;
}
#endif

#define BLK   64
#define BSH    6

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out)
{
    if (n <= 0) return;
    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    if (source < 0 || source >= n) return;
    if (m < 0) m = 0;

    /* ---------------- CSR (zero-copy when src[] is already grouped) -------- */
    int *off = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int *edst_buf = NULL; double *ew_buf = NULL;
    const int *E_dst; const double *E_w;
    {
        int *cnt = (int *)calloc((size_t)n + 1, sizeof(int));
        for (int i = 0; i < m; i++) cnt[src[i]]++;
        int sorted = 1;
        for (int i = 1; i < m; i++) if (src[i] < src[i - 1]) { sorted = 0; break; }
        off[0] = 0;
        for (int i = 0; i < n; i++) off[i + 1] = off[i] + cnt[i];
        if (sorted) { E_dst = dst; E_w = weight; }
        else {
            edst_buf = (int *)malloc((size_t)(m ? m : 1) * sizeof(int));
            ew_buf   = (double *)malloc((size_t)(m ? m : 1) * sizeof(double));
            memcpy(cnt, off, (size_t)n * sizeof(int));      /* reuse as fill cursor */
            for (int i = 0; i < m; i++) {
                int u = src[i], p = cnt[u]++;
                edst_buf[p] = dst[i]; ew_buf[p] = weight[i];
            }
            E_dst = edst_buf; E_w = ew_buf;
        }
        free(cnt);
    }

    /* ---------------- block-min pyramid over the scan array ---------------- */
    int sizes[12]; int nlev = 1;
    sizes[0] = (n + (BLK - 1)) & ~(BLK - 1);
    if (sizes[0] < BLK) sizes[0] = BLK;
    while (sizes[nlev - 1] > BLK && nlev < 12) {
        sizes[nlev] = ((sizes[nlev - 1] >> BSH) + (BLK - 1)) & ~(BLK - 1);
        nlev++;
    }
    size_t tot = 0;
    for (int k = 0; k < nlev; k++) tot += (size_t)sizes[k];
    double *raw = (double *)malloc(tot * sizeof(double) + 64);
    double *base = (double *)(((uintptr_t)raw + 63) & ~(uintptr_t)63);
    double *lev[12];
    { size_t acc = 0;
      for (int k = 0; k < nlev; k++) { lev[k] = base + acc; acc += (size_t)sizes[k]; } }
    for (size_t i = 0; i < tot; i++) base[i] = INFINITY;

    double *tent = lev[0];
    tent[source] = 0.0;
    { int idx = source;
      for (int k = 1; k < nlev; k++) { idx >>= BSH;
          if (0.0 < lev[k][idx]) lev[k][idx] = 0.0; else break; } }

    /* ---------------- one lock per round ----------------------------------- */
    int remaining = n;
    while (remaining > 0) {
        double mv;
        int i = blk_argmin(lev[nlev - 1], BLK, &mv);      /* smallest of all guesses */
        if (mv == INFINITY) break;                        /* rest unreachable */
        for (int k = nlev - 2; k >= 0; k--) {
            int b = i << BSH;
            i = b + blk_argmin(lev[k] + b, BLK, &mv);
        }
        int u = i;
        double du = mv;
        dist_out[u] = du;                                 /* locked, final */
        tent[u] = INFINITY;
        { int child = u;                                  /* repair the memoized mins */
          for (int k = 1; k < nlev; k++) {
              int par = child >> BSH;
              double nv = blk_min(lev[k - 1] + (par << BSH), BLK);
              if (lev[k][par] == nv) break;
              lev[k][par] = nv;
              child = par;
          } }
        remaining--;

        int e = off[u], ee = off[u + 1];                  /* now crawl its tunnels */
        for (; e < ee; e++) {
            int v = E_dst[e];
            double nd = du + E_w[e];
            if (nd < dist_out[v]) {
                dist_out[v] = nd;
                tent[v] = nd;
                int idx = v;
                for (int k = 1; k < nlev; k++) { idx >>= BSH;
                    if (nd < lev[k][idx]) lev[k][idx] = nd; else break; }
            }
        }
    }

    free(raw); free(off); free(edst_buf); free(ew_buf);
}
