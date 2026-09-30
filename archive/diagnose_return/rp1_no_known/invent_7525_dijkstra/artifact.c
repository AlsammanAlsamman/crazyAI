#include <stdlib.h>
#include <string.h>
#include <math.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

/* ----------------------------------------------------------------------
   place              -> node 0..n-1
   road               -> NRoad { stretch w, board it ends at v }
   note / hoarseness  -> dist[u] + w
   rigid lattice      -> key[], one fixed slot per place, in rows of B;
                         each row carries a rim-mark bmin[b] = faintest
                         note in that row  (the outer cow; the slot is the
                         smaller cow inside it)
   settling a board   -> carve dist_out[u], then kill the slot with a
                         dead-note (NaN).  nd < NaN is false, so every
                         road into a settled board is dead for good, at
                         zero cost and with no flag array.
   desert             -> INFINITY (never reached) ; also the sparse regime
   cairn of birds     -> 4-ary heap, used only when the land is mostly
                         desert and walking the lattice costs more than
                         the birds themselves do
   ---------------------------------------------------------------------- */

typedef struct { double w; int v; int pad; } NRoad;   /* 16 B */
typedef struct { double d; int u; int pad; } NBird;   /* 16 B */

/* faintest note in a stretch of lattice, and which slot holds it.
   dead slots hold NaN: every comparison against them is false, so they
   are invisible here and deaf to every later bird. */
static int lat_argmin(const double *a, int cnt, double *outmin)
{
    double best = INFINITY;
    long long bi = -1;
    int i = 0;
#if defined(__AVX2__)
    if (cnt >= 8) {
        __m256d vm0 = _mm256_set1_pd(INFINITY), vm1 = vm0;
        __m256i vi0 = _mm256_set1_epi64x(-1), vi1 = vi0;
        __m256i c0 = _mm256_setr_epi64x(0, 1, 2, 3);
        __m256i c1 = _mm256_setr_epi64x(4, 5, 6, 7);
        const __m256i st = _mm256_set1_epi64x(8);
        for (; i + 8 <= cnt; i += 8) {
            __m256d x0 = _mm256_loadu_pd(a + i);
            __m256d x1 = _mm256_loadu_pd(a + i + 4);
            __m256d k0 = _mm256_cmp_pd(x0, vm0, _CMP_LT_OQ);  /* NaN -> false */
            __m256d k1 = _mm256_cmp_pd(x1, vm1, _CMP_LT_OQ);
            vm0 = _mm256_blendv_pd(vm0, x0, k0);
            vm1 = _mm256_blendv_pd(vm1, x1, k1);
            vi0 = _mm256_castpd_si256(_mm256_blendv_pd(_mm256_castsi256_pd(vi0),
                                                       _mm256_castsi256_pd(c0), k0));
            vi1 = _mm256_castpd_si256(_mm256_blendv_pd(_mm256_castsi256_pd(vi1),
                                                       _mm256_castsi256_pd(c1), k1));
            c0 = _mm256_add_epi64(c0, st);
            c1 = _mm256_add_epi64(c1, st);
        }
        {
            double bv[8]; long long bx[8]; int k;
            _mm256_storeu_pd(bv, vm0);
            _mm256_storeu_pd(bv + 4, vm1);
            _mm256_storeu_si256((__m256i *)bx, vi0);
            _mm256_storeu_si256((__m256i *)(bx + 4), vi1);
            for (k = 0; k < 8; k++) if (bv[k] < best) { best = bv[k]; bi = bx[k]; }
        }
    }
#endif
    for (; i < cnt; i++) if (a[i] < best) { best = a[i]; bi = i; }
    *outmin = best;
    return (int)bi;
}

/* re-cut a row's rim-mark.  vminpd(load, acc) returns acc when load is
   NaN, so dead slots are skipped and acc is never poisoned. */
static double lat_min(const double *a, int cnt)
{
    double best = INFINITY;
    int i = 0;
#if defined(__AVX2__)
    if (cnt >= 16) {
        __m256d m0 = _mm256_set1_pd(INFINITY), m1 = m0, m2 = m0, m3 = m0;
        double t[4]; int k;
        for (; i + 16 <= cnt; i += 16) {
            m0 = _mm256_min_pd(_mm256_loadu_pd(a + i),      m0);
            m1 = _mm256_min_pd(_mm256_loadu_pd(a + i + 4),  m1);
            m2 = _mm256_min_pd(_mm256_loadu_pd(a + i + 8),  m2);
            m3 = _mm256_min_pd(_mm256_loadu_pd(a + i + 12), m3);
        }
        m0 = _mm256_min_pd(m0, m1);
        m2 = _mm256_min_pd(m2, m3);
        m0 = _mm256_min_pd(m0, m2);
        _mm256_storeu_pd(t, m0);
        for (k = 0; k < 4; k++) if (t[k] < best) best = t[k];
    }
#endif
    for (; i < cnt; i++) if (a[i] < best) best = a[i];
    return best;
}

/* gather the roads into one bundle per place (CSR), serially */
static void gather_ser(int n, int m, const int *src, const int *dst,
                       const double *w, int *off, NRoad *E)
{
    int i, u, s = 0;
    memset(off, 0, (size_t)(n + 1) * sizeof(int));
    for (i = 0; i < m; i++) off[src[i]]++;
    for (u = 0; u < n; u++) { int t = off[u]; off[u] = s; s += t; }
    off[n] = s;
    for (i = 0; i < m; i++) { int a = src[i]; int p = off[a]++;
                              E[p].v = dst[i]; E[p].w = w[i]; }
    for (u = n; u > 0; u--) off[u] = off[u - 1];
    off[0] = 0;
}

/* the same gathering with many hands.  Legitimate to parallelise because
   it happens before any bird flies and depends on no settling order.
   Produces a byte-identical CSR to gather_ser.  Returns 0 if declined. */
static int gather_par(int n, int m, const int *src, const int *dst,
                      const double *w, int *off, NRoad *E)
{
#ifdef _OPENMP
    int T, u;
    long long s;
    int *cnt;
    if (n <= 0 || m < (1 << 21)) return 0;                 /* size guard */
    T = omp_get_max_threads();
    if (T > 16) T = 16;
    while (T >= 2 && (long long)T * (long long)n > (long long)m) T--;
    if (T < 2) return 0;                                   /* memory guard */
    cnt = (int *)calloc((size_t)T * (size_t)n, sizeof(int));
    if (!cnt) return 0;                                    /* fallback */

    #pragma omp parallel num_threads(T)
    {
        int tt = omp_get_thread_num();
        int *c = cnt + (size_t)tt * (size_t)n;
        int lo = (int)(((long long)m * tt) / T);
        int hi = (int)(((long long)m * (tt + 1)) / T);
        int k;
        for (k = lo; k < hi; k++) c[src[k]]++;
    }
    #pragma omp parallel for num_threads(T) schedule(static)
    for (u = 0; u < n; u++) {
        int sum = 0, tt;
        for (tt = 0; tt < T; tt++) sum += cnt[(size_t)tt * (size_t)n + u];
        off[u] = sum;
    }
    s = 0;
    for (u = 0; u < n; u++) { int c0 = off[u]; off[u] = (int)s; s += c0; }
    off[n] = (int)s;
    #pragma omp parallel for num_threads(T) schedule(static)
    for (u = 0; u < n; u++) {
        int run = off[u], tt;
        for (tt = 0; tt < T; tt++) {
            size_t ix = (size_t)tt * (size_t)n + (size_t)u;
            int c0 = cnt[ix]; cnt[ix] = run; run += c0;
        }
    }
    #pragma omp parallel num_threads(T)
    {
        int tt = omp_get_thread_num();
        int *c = cnt + (size_t)tt * (size_t)n;
        int lo = (int)(((long long)m * tt) / T);
        int hi = (int)(((long long)m * (tt + 1)) / T);
        int k;
        for (k = lo; k < hi; k++) { int a = src[k]; int p = c[a]++;
                                    E[p].v = dst[k]; E[p].w = w[k]; }
    }
    free(cnt);
    return 1;
#else
    (void)n; (void)m; (void)src; (void)dst; (void)w; (void)off; (void)E;
    return 0;
#endif
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out)
{
    int *off; NRoad *E;
    int i;
    double dn, dm, lg, pushes;
    int use_lattice;

    if (n <= 0) return;
    for (i = 0; i < n; i++) dist_out[i] = INFINITY;
    if (source < 0 || source >= n) return;
    if (m < 0) m = 0;

    off = (int *)malloc((size_t)(n + 1) * sizeof(int));
    E   = (NRoad *)malloc(((size_t)m + 1) * sizeof(NRoad));
    if (!off || !E) { free(off); free(E); return; }

    if (!gather_par(n, m, src, dst, weight, off, E))
        gather_ser(n, m, src, dst, weight, off, E);

    /* --- how much desert is this land? --------------------------------
       lattice walk  ~ 0.7 * n * sqrt(n) cycles of SIMD reading
       cairn of birds~ 8 * (expected pushes) * log2(n)   cycles of pointer
       chasing.  Shared edge-sweep and gathering costs cancel.          */
    dn = (double)n; dm = (double)m;
    lg = log2(dn + 2.0);
    pushes = dn * (1.0 + log2(1.0 + dm / (dn > 0.0 ? dn : 1.0)));
    if (pushes > dm) pushes = dm;
    use_lattice = (n <= 2048) || (0.7 * dn * sqrt(dn) < 8.0 * pushes * lg);

    if (use_lattice) {
        /* ---------- the rigid lattice ---------- */
        int L = 3, B, nb, it;
        size_t padded;
        double *restrict key;
        double *restrict bmin;
        const NRoad *restrict R = E;
        const int   *restrict O = off;
        double *restrict dout = dist_out;

        while (L < 20 && (((size_t)1 << (2 * L + 1)) < (size_t)n)) L++;
        B  = 1 << L;
        nb = (int)((((size_t)n + (size_t)B - 1)) >> L);
        if (nb < 1) nb = 1;
        padded = (size_t)nb << L;

        key  = (double *)malloc(padded * sizeof(double));
        bmin = (double *)malloc((size_t)nb * sizeof(double));
        if (!key || !bmin) { free(key); free(bmin); free(off); free(E); return; }
        {
            size_t z;
            for (z = 0; z < padded; z++) key[z] = INFINITY;
        }
        for (i = 0; i < nb; i++) bmin[i] = INFINITY;
        key[source]        = 0.0;
        bmin[source >> L]  = 0.0;

        for (it = 0; it < n; it++) {
            double vb, d;
            int bb, j, u, e0, e1, e;
            size_t base;

            bb = lat_argmin(bmin, nb, &vb);                 /* read the rims */
            if (bb < 0 || !(vb < INFINITY)) break;          /* birds all silent */
            base = (size_t)bb << L;
            j = lat_argmin(key + base, B, &d);              /* read that row */
            if (j < 0 || !(d < INFINITY)) { bmin[bb] = INFINITY; continue; }
            if (base + (size_t)j >= (size_t)n) { bmin[bb] = lat_min(key + base, B); continue; }
            u = (int)(base + (size_t)j);

            if (d < dout[u]) dout[u] = d;                   /* permanent carving */
            key[u]   = NAN;                                 /* the dead-note */
            bmin[bb] = lat_min(key + base, B);              /* re-cut the rim */

            e0 = O[u]; e1 = O[u + 1]; e = e0;
            for (; e + 8 < e1; e++) {
                int v; double nd;
                __builtin_prefetch(&key[R[e + 8].v], 1, 3);
                v = R[e].v; nd = d + R[e].w;
                if (nd < key[v]) {                          /* thinner milk? */
                    int b2 = v >> L;
                    key[v] = nd;
                    if (nd < bmin[b2]) bmin[b2] = nd;
                }
            }
            for (; e < e1; e++) {
                int v = R[e].v; double nd = d + R[e].w;
                if (nd < key[v]) {
                    int b2 = v >> L;
                    key[v] = nd;
                    if (nd < bmin[b2]) bmin[b2] = nd;
                }
            }
        }
        free(key); free(bmin);
    } else {
        /* ---------- the cairn: vast desert, few roads ---------- */
        NBird *restrict h;
        const NRoad *restrict R = E;
        const int   *restrict O = off;
        double *restrict dist = dist_out;
        int hs;

        h = (NBird *)malloc(((size_t)m + 2) * sizeof(NBird));
        if (!h) { free(off); free(E); return; }
        dist[source] = 0.0;
        h[0].d = 0.0; h[0].u = source; hs = 1;

        while (hs > 0) {
            double d = h[0].d;
            int u = h[0].u, e0, e1, e;
            hs--;
            if (hs > 0) {                                   /* 4-ary sift down */
                NBird last = h[hs];
                int q = 0;
                for (;;) {
                    int c = 4 * q + 1, lim, bst, k;
                    double bd;
                    if (c >= hs) break;
                    lim = c + 4; if (lim > hs) lim = hs;
                    bst = c; bd = h[c].d;
                    for (k = c + 1; k < lim; k++) if (h[k].d < bd) { bd = h[k].d; bst = k; }
                    if (!(bd < last.d)) break;
                    h[q] = h[bst];
                    q = bst;
                }
                h[q] = last;
            }
            if (d > dist[u]) continue;                      /* spent bird */
            e0 = O[u]; e1 = O[u + 1]; e = e0;
            for (; e + 8 < e1; e++) {
                int v; double nd;
                __builtin_prefetch(&dist[R[e + 8].v], 1, 3);
                v = R[e].v; nd = d + R[e].w;
                if (nd < dist[v]) {
                    int q = hs++;
                    dist[v] = nd;
                    while (q > 0) { int p = (q - 1) >> 2; if (!(nd < h[p].d)) break;
                                    h[q] = h[p]; q = p; }
                    h[q].d = nd; h[q].u = v;
                }
            }
            for (; e < e1; e++) {
                int v = R[e].v; double nd = d + R[e].w;
                if (nd < dist[v]) {
                    int q = hs++;
                    dist[v] = nd;
                    while (q > 0) { int p = (q - 1) >> 2; if (!(nd < h[p].d)) break;
                                    h[q] = h[p]; q = p; }
                    h[q].d = nd; h[q].u = v;
                }
            }
        }
        free(h);
    }
    free(off); free(E);
}
