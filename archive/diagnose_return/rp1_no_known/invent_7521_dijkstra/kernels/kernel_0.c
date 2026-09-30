#include <stdlib.h>
#include <string.h>
#include <math.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* ==========================================================================
   THE GARDEN OF NIGHTINGALE
     stone            -> node slot            letter  -> tentative distance
     thief            -> directed edge weight thread  -> one relaxation
     nightingale circuit -> vectorised argmin over the UNLOCKED stones only
     locking          -> final distance; the stone leaves the birds' circuit
     bramble / tide   -> unreachable: the circuit finds only INFINITY -> stop
     knot of three threads -> 3-ary heap, decrease-key in place (thin garden)
   ========================================================================== */

#define GARDEN_SMALL 1024      /* a garden the birds can see whole          */
#define GARDEN_THICK 125.0     /* roads thick enough to be worth circling   */

/* ---- one circuit of the birds: index of the smallest unlocked letter ---- */
static inline int nightingale_circuit(const double *restrict key, int cnt,
                                      double *best_out)
{
#if defined(__AVX2__)
    double bv = INFINITY; int bi = -1; int i = 0;
    if (cnt >= 16) {
        __m256d b0 = _mm256_set1_pd(INFINITY), b1 = b0, b2 = b0, b3 = b0;
        __m256d n0 = _mm256_set1_pd(-1.0), n1 = n0, n2 = n0, n3 = n0;
        __m256d c0 = _mm256_set_pd(3.0, 2.0, 1.0, 0.0);
        __m256d c1 = _mm256_set_pd(7.0, 6.0, 5.0, 4.0);
        __m256d c2 = _mm256_set_pd(11.0, 10.0, 9.0, 8.0);
        __m256d c3 = _mm256_set_pd(15.0, 14.0, 13.0, 12.0);
        const __m256d st = _mm256_set1_pd(16.0);
        const int lim = cnt - 16;
        for (; i <= lim; i += 16) {
            __m256d v0 = _mm256_loadu_pd(key + i);
            __m256d v1 = _mm256_loadu_pd(key + i + 4);
            __m256d v2 = _mm256_loadu_pd(key + i + 8);
            __m256d v3 = _mm256_loadu_pd(key + i + 12);
            __m256d m0 = _mm256_cmp_pd(v0, b0, _CMP_LT_OQ);
            __m256d m1 = _mm256_cmp_pd(v1, b1, _CMP_LT_OQ);
            __m256d m2 = _mm256_cmp_pd(v2, b2, _CMP_LT_OQ);
            __m256d m3 = _mm256_cmp_pd(v3, b3, _CMP_LT_OQ);
            b0 = _mm256_min_pd(b0, v0); n0 = _mm256_blendv_pd(n0, c0, m0);
            b1 = _mm256_min_pd(b1, v1); n1 = _mm256_blendv_pd(n1, c1, m1);
            b2 = _mm256_min_pd(b2, v2); n2 = _mm256_blendv_pd(n2, c2, m2);
            b3 = _mm256_min_pd(b3, v3); n3 = _mm256_blendv_pd(n3, c3, m3);
            c0 = _mm256_add_pd(c0, st); c1 = _mm256_add_pd(c1, st);
            c2 = _mm256_add_pd(c2, st); c3 = _mm256_add_pd(c3, st);
        }
        double vb[16], ib[16];
        _mm256_storeu_pd(vb,      b0); _mm256_storeu_pd(vb + 4,  b1);
        _mm256_storeu_pd(vb + 8,  b2); _mm256_storeu_pd(vb + 12, b3);
        _mm256_storeu_pd(ib,      n0); _mm256_storeu_pd(ib + 4,  n1);
        _mm256_storeu_pd(ib + 8,  n2); _mm256_storeu_pd(ib + 12, n3);
        for (int j = 0; j < 16; ++j)
            if (vb[j] < bv) { bv = vb[j]; bi = (int)ib[j]; }
    }
    for (; i < cnt; ++i) { double v = key[i]; if (v < bv) { bv = v; bi = i; } }
    *best_out = bv;
    return bi;
#else
    double bv = INFINITY;
    for (int j = 0; j < cnt; ++j) { double v = key[j]; if (v < bv) bv = v; }
    *best_out = bv;
    if (!(bv < INFINITY)) return -1;
    for (int j = 0; j < cnt; ++j) if (key[j] == bv) return j;
    return -1;
#endif
}

/* ---------------- thick / small garden: send the nightingales ------------ */
static void garden_of_nightingale(int n,
                                  const int    *restrict off,
                                  const int    *restrict edst,
                                  const double *restrict ew,
                                  int source,
                                  double *restrict dist_out,
                                  int    *restrict cand,   /* slot -> stone  */
                                  int    *restrict cpos,   /* stone -> slot  */
                                  double *restrict ckey)   /* slot -> letter */
{
    for (int i = 0; i < n; ++i) { cand[i] = i; cpos[i] = i; ckey[i] = INFINITY; }
    ckey[source] = 0.0;

    int cnt = n;
    while (cnt > 0) {
        double best;
        int p = nightingale_circuit(ckey, cnt, &best);
        if (p < 0) break;                    /* only bramble left: drop it   */

        int u = cand[p];
        dist_out[u] = best;                  /* the letter is locked         */

        --cnt;                               /* stone leaves the circuit     */
        int moved = cand[cnt];
        cand[p] = moved; ckey[p] = ckey[cnt]; cpos[moved] = p;
        cpos[u] = -1;

        const int e1 = off[u + 1];
        for (int e = off[u]; e < e1; ++e) {  /* cast threads, price by thief */
            int v = edst[e];
            int q = cpos[v];
            if (q >= 0) {                    /* locked/nowhere -> thread dropped */
                double nd = best + ew[e];
                if (nd < ckey[q]) ckey[q] = nd;   /* rechalk; stone unmoved  */
            }
        }
    }
}

/* ------------- thin, wide garden: knots of three threads ----------------- */
static void knots_of_three(int n,
                           const int    *restrict off,
                           const int    *restrict edst,
                           const double *restrict ew,
                           int source,
                           double *restrict dist_out,
                           int    *restrict heap,
                           int    *restrict pos,
                           double *restrict key)
{
    for (int i = 0; i < n; ++i) { pos[i] = -1; key[i] = INFINITY; }
    key[source] = 0.0; heap[0] = source; pos[source] = 0;
    int hs = 1;

    while (hs > 0) {
        int u = heap[0];
        double du = key[u];
        dist_out[u] = du;
        pos[u] = -2;                          /* locked, never rebuilt again */

        --hs;
        if (hs > 0) {                         /* settle the knot again       */
            int w = heap[hs]; double kw = key[w]; int i = 0;
            for (;;) {
                int c = 3 * i + 1;
                if (c >= hs) break;
                int bidx = c; double bk = key[heap[c]];
                int cend = c + 3; if (cend > hs) cend = hs;
                for (int j = c + 1; j < cend; ++j) {
                    double t = key[heap[j]];
                    if (t < bk) { bk = t; bidx = j; }
                }
                if (bk >= kw) break;
                heap[i] = heap[bidx]; pos[heap[i]] = i; i = bidx;
            }
            heap[i] = w; pos[w] = i;
        }

        const int e1 = off[u + 1];
        for (int e = off[u]; e < e1; ++e) {
            int v = edst[e];
            double nd = du + ew[e];
            if (nd < key[v]) {
                key[v] = nd;
                int pv = pos[v];
                if (pv >= 0) {                /* detach and call back on     */
                    int i = pv;
                    while (i > 0) {
                        int par = (i - 1) / 3;
                        if (key[heap[par]] <= nd) break;
                        heap[i] = heap[par]; pos[heap[i]] = i; i = par;
                    }
                    heap[i] = v; pos[v] = i;
                } else if (pv == -1) {        /* first thread to reach it    */
                    int i = hs++;
                    while (i > 0) {
                        int par = (i - 1) / 3;
                        if (key[heap[par]] <= nd) break;
                        heap[i] = heap[par]; pos[heap[i]] = i; i = par;
                    }
                    heap[i] = v; pos[v] = i;
                }
            }
        }
    }
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out)
{
    if (n <= 0) return;
    for (int i = 0; i < n; ++i) dist_out[i] = INFINITY;
    if (source < 0 || source >= n) return;
    if (m <= 0) { dist_out[source] = 0.0; return; }

    /* ---- the roads, priced one direction at a time (CSR, single pass) ---- */
    int *off = (int *)calloc((size_t)n + 2, sizeof(int));
    int *edst = (int *)malloc((size_t)m * sizeof(int));
    double *ew = (double *)malloc((size_t)m * sizeof(double));
    if (!off || !edst || !ew) { free(off); free(edst); free(ew); return; }
    for (int i = 0; i < m; ++i) off[src[i] + 2]++;
    for (int i = 2; i <= n + 1; ++i) off[i] += off[i - 1];
    for (int i = 0; i < m; ++i) {
        int p = off[src[i] + 1]++;
        edst[p] = dst[i]; ew[p] = weight[i];
    }

    int    *ia = (int *)malloc((size_t)n * sizeof(int));
    int    *ib = (int *)malloc((size_t)n * sizeof(int));
    double *da = (double *)malloc((size_t)n * sizeof(double));
    if (!ia || !ib || !da) { free(off); free(edst); free(ew);
                             free(ia); free(ib); free(da); return; }

    /* ---- count the roads per house before climbing: which garden is this? */
    double nn = (double)n * (double)n;
    int thick_or_small = (n <= GARDEN_SMALL) || ((double)m * GARDEN_THICK >= nn);

    if (thick_or_small)
        garden_of_nightingale(n, off, edst, ew, source, dist_out, ia, ib, da);
    else
        knots_of_three(n, off, edst, ew, source, dist_out, ia, ib, da);

    free(off); free(edst); free(ew); free(ia); free(ib); free(da);
}
