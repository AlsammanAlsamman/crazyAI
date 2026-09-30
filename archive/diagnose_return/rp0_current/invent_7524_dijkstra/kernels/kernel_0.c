#include <stdlib.h>
#include <string.h>
#include <math.h>
#if defined(__AVX__)
#include <immintrin.h>
#endif

/* ------------------------------------------------------------------ *
 *  "The glint and the chalk grid."
 *
 *   jewel        -> node                chalk square -> dist_out[u]
 *   cord         -> directed edge       cord length  -> weight, a DELAY
 *   glint        -> a tentative label in flight
 *   "short cord flashes at once"  -> light edge (w <= delta)
 *   "long cord keeps it waiting"  -> heavy edge (w >  delta)
 *   "waiting on the way"          -> ring of arrival slots of width delta
 *   blank + crossed square        -> INFINITY
 *
 *  No jewel is settled before it fires: it fires the instant it first
 *  catches light, and may fire again inside the same instant if a
 *  shorter cord improves it.  That is delta-stepping.
 *
 *  Before crouching she counts cords against squares: if the net is so
 *  thick that nearly every jewel touches every other, she keeps no
 *  calendar and simply sweeps her eye across the whole grid each round.
 * ------------------------------------------------------------------ */

void kernel(int n, int m, const int *src, const int *dst,
            const double *weight, int source, double *dist_out)
{
    if (n <= 0) return;
    double * __restrict d = dist_out;
    for (int i = 0; i < n; i++) d[i] = INFINITY;         /* blank, crossed */
    if (source < 0 || source >= n) return;
    d[source] = 0.0;                                     /* the one touch  */
    if (m <= 0) return;

    /* ---- survey the cords ---------------------------------------- */
    double wmax = 0.0, wsum = 0.0;
    for (int i = 0; i < m; i++) {
        double w = weight[i];
        wsum += w;
        if (w > wmax) wmax = w;
    }

    /* ---- which regime?  count cords against squares --------------- */
    const double dn    = (double)n;
    const int    dense = (n <= 64) || (dn * dn <= 6.0 * ((double)m + dn));

    /* ---- width of one instant ------------------------------------- */
    double delta;
    if (wmax <= 0.0) {
        delta = 1.0;
    } else {
        double avgdeg = (double)m / dn;
        if (avgdeg < 1.0) avgdeg = 1.0;
        delta = 4.0 * (wsum / (double)m) / avgdeg;
        double lo = wmax / 4096.0, hi = wmax * 8.0;
        if (!(delta > 0.0) || delta < lo) delta = lo;
        if (delta > hi) delta = hi;
    }
    const double thr = dense ? INFINITY : delta;   /* light / heavy split */

    /* ---- CSR: for each jewel, its short cords first --------------- */
    int    *off      = (int*)   malloc((size_t)(n + 1) * sizeof(int));
    int    *lightend = (int*)   malloc((size_t)n * sizeof(int));
    int    *cnt      = (int*)   calloc((size_t)n, sizeof(int));
    int    *lcnt     = (int*)   calloc((size_t)n, sizeof(int));
    int    *edst     = (int*)   malloc((size_t)m * sizeof(int));
    double *ew       = (double*)malloc((size_t)m * sizeof(double));
    if (!off || !lightend || !cnt || !lcnt || !edst || !ew) {
        free(off); free(lightend); free(cnt); free(lcnt); free(edst); free(ew);
        return;
    }
    for (int i = 0; i < m; i++) {
        int u = src[i];
        cnt[u]++;
        if (weight[i] <= thr) lcnt[u]++;
    }
    off[0] = 0;
    for (int u = 0; u < n; u++) {
        off[u + 1]  = off[u] + cnt[u];
        lightend[u] = off[u] + lcnt[u];
    }
    memset(cnt,  0, (size_t)n * sizeof(int));   /* reuse: light fill */
    memset(lcnt, 0, (size_t)n * sizeof(int));   /* reuse: heavy fill */
    for (int i = 0; i < m; i++) {
        int u = src[i], pos;
        if (weight[i] <= thr) pos = off[u]      + cnt[u]++;
        else                  pos = lightend[u] + lcnt[u]++;
        edst[pos] = dst[i];
        ew[pos]   = weight[i];
    }
    free(cnt); free(lcnt);

    const int    * __restrict E  = edst;
    const double * __restrict W  = ew;
    const int    * __restrict OF = off;
    const int    * __restrict LE = lightend;

    /* ================= DENSE / TINY REGIME ========================= *
     * No calendar at all: sweep the whole chalk grid each round and    *
     * take the dimmest square that is not yet lit.  O(n^2 + m).        */
    if (dense) {
        double *key = (double*)malloc((size_t)n * sizeof(double));
        if (key) {
            for (int i = 0; i < n; i++) key[i] = INFINITY;
            key[source] = 0.0;
            for (int it = 0; it < n; it++) {
                double best = INFINITY;
                int i = 0;
#if defined(__AVX__)
                {
                    __m256d vb = _mm256_set1_pd(INFINITY);
                    for (; i + 4 <= n; i += 4)
                        vb = _mm256_min_pd(vb, _mm256_loadu_pd(key + i));
                    double t[4];
                    _mm256_storeu_pd(t, vb);
                    for (int k = 0; k < 4; k++) if (t[k] < best) best = t[k];
                }
#endif
                for (; i < n; i++) if (key[i] < best) best = key[i];
                if (!(best < INFINITY)) break;          /* rest is dark */

                int u = -1; i = 0;
#if defined(__AVX__)
                {
                    __m256d vbest = _mm256_set1_pd(best);
                    for (; i + 4 <= n; i += 4) {
                        __m256d v = _mm256_loadu_pd(key + i);
                        int msk = _mm256_movemask_pd(
                                      _mm256_cmp_pd(v, vbest, _CMP_EQ_OQ));
                        if (msk) { u = i + (int)__builtin_ctz((unsigned)msk); break; }
                    }
                }
#endif
                if (u < 0) { for (; i < n; i++) if (key[i] == best) { u = i; break; } }
                if (u < 0) break;

                key[u] = INFINITY;                      /* lit; never again */
                double du = d[u];
                for (int e = OF[u], he = OF[u + 1]; e < he; e++) {
                    int v = E[e];
                    double nd = du + W[e];
                    if (nd < d[v]) { d[v] = nd; key[v] = nd; }
                }
            }
            free(key);
        }
        free(off); free(lightend); free(edst); free(ew);
        return;
    }

    /* ================= SPARSE REGIME: the calendar ================= */
    long long maxspan = (long long)(wmax / delta) + 2;
    int nb = 8;
    while ((long long)nb < maxspan + 2) nb <<= 1;
    const int mask = nb - 1;

    int       *head = (int*)      malloc((size_t)nb * sizeof(int));
    int       *nxt  = (int*)      malloc((size_t)n  * sizeof(int));
    int       *prv  = (int*)      malloc((size_t)n  * sizeof(int));
    long long *qidx = (long long*)malloc((size_t)n  * sizeof(long long));
    int       *Rl   = (int*)      malloc((size_t)n  * sizeof(int));
    char      *inR  = (char*)     calloc((size_t)n, 1);
    if (!head || !nxt || !prv || !qidx || !Rl || !inR) {
        free(head); free(nxt); free(prv); free(qidx); free(Rl); free(inR);
        free(off); free(lightend); free(edst); free(ew);
        return;
    }
    for (int b = 0; b < nb; b++) head[b] = -1;
    for (int i = 0; i < n;  i++) { qidx[i] = -1; nxt[i] = -1; prv[i] = -1; }

    long long cur  = 0, lowb = 0;
    int       nlive = 1;
    head[0] = source; qidx[source] = 0;

    /* one live glint per jewel: O(1) move between slots, no duplicates */
#define PUSH_V(v_, nd_)                                                     \
    do {                                                                    \
        int  vv_ = (v_);                                                    \
        long long nbk = (long long)((nd_) / delta);                         \
        if (nbk < lowb) nbk = lowb;                                         \
        else if (nbk > cur + maxspan) nbk = cur + maxspan;                  \
        if (qidx[vv_] != nbk) {                                             \
            if (qidx[vv_] >= 0) {                                           \
                int ob_ = (int)(qidx[vv_] & mask);                          \
                if (prv[vv_] >= 0) nxt[prv[vv_]] = nxt[vv_];                \
                else               head[ob_]     = nxt[vv_];                \
                if (nxt[vv_] >= 0) prv[nxt[vv_]] = prv[vv_];                \
                nlive--;                                                    \
            }                                                               \
            int tb_ = (int)(nbk & mask);                                    \
            nxt[vv_] = head[tb_]; prv[vv_] = -1;                            \
            if (head[tb_] >= 0) prv[head[tb_]] = vv_;                       \
            head[tb_] = vv_; qidx[vv_] = nbk; nlive++;                      \
        }                                                                   \
    } while (0)

    while (nlive > 0) {
        while (head[(int)(cur & mask)] < 0) cur++;      /* next live instant */
        int b = (int)(cur & mask);
        int Rcnt = 0;

        /* --- short cords, to a fixpoint inside this one instant ---- *
         * A jewel fires the moment it catches light; if a shorter cord *
         * improves it before the instant closes, it fires again.       */
        lowb = cur;
        while (head[b] >= 0) {
            int u = head[b];
            head[b] = nxt[u];
            if (nxt[u] >= 0) prv[nxt[u]] = -1;
            qidx[u] = -1; nlive--;
            if (!inR[u]) { inR[u] = 1; Rl[Rcnt++] = u; }
            double du = d[u];
            for (int e = OF[u], le = LE[u]; e < le; e++) {
                int v = E[e];
                double nd = du + W[e];
                if (nd < d[v]) { d[v] = nd; PUSH_V(v, nd); }   /* first light wins */
            }
        }

        /* --- long cords: the glint was kept waiting on the way ----- */
        lowb = cur + 1;
        for (int k = 0; k < Rcnt; k++) {
            int u = Rl[k];
            inR[u] = 0;
            double du = d[u];
            for (int e = LE[u], he = OF[u + 1]; e < he; e++) {
                int v = E[e];
                double nd = du + W[e];
                if (nd < d[v]) { d[v] = nd; PUSH_V(v, nd); }
            }
        }
        cur++;
    }
#undef PUSH_V

    free(head); free(nxt); free(prv); free(qidx); free(Rl); free(inR);
    free(off);  free(lightend); free(edst); free(ew);
}
