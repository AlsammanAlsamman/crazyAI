/* Shortest paths by the native's method:
 *   embers  = dist_out[]            (monotone non-increasing tentative distances)
 *   runners = out-edge relaxations from un-garrisoned heaps
 *   the throw = one synchronous land-wide round, every heap's min taken at once
 *   the garrison = the active-frontier mask; a heap whose ember did not shorten
 *                  stops sending runners until a shorter stick breaks the seal
 * No priority queue, no global argmin, no vertex ever compared to another vertex.
 */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <omp.h>

typedef uint64_t nb_u64 __attribute__((may_alias));

#define NB_BUF 2048

/* One stick laid on a heap: it survives only if it is the shortest sliver so far.
 * All distances are non-negative, so IEEE-754 bit patterns order like the doubles
 * and the min can be done as a lock-free unsigned CAS. */
static inline int nb_relax(double *slot, double cand)
{
    nb_u64 nv;
    __builtin_memcpy(&nv, &cand, sizeof nv);
    nb_u64 *p = (nb_u64 *)slot;
    nb_u64 old = __atomic_load_n(p, __ATOMIC_RELAXED);
    while (nv < old) {
        if (__atomic_compare_exchange_n(p, &old, nv, 1,
                                        __ATOMIC_RELAXED, __ATOMIC_RELAXED))
            return 1;
    }
    return 0;
}

/* One throw, done by a single hand: used whenever the un-garrisoned set is so
 * small that fork/barrier cost would exceed the work (long-thin graphs). */
static int nb_round_serial(int nc, const int *curf, int *nxtf,
                           const int *off, const int *eto, const double *ew,
                           unsigned char *inq, double *d)
{
    int nn = 0;
    for (int i = 0; i < nc; ++i) inq[curf[i]] = 0;
    for (int i = 0; i < nc; ++i) {
        int u = curf[i];
        double du = d[u];
        int e1 = off[u + 1];
        for (int e = off[u]; e < e1; ++e) {
            int v = eto[e];
            double c = du + ew[e];
            if (c < d[v]) {
                d[v] = c;
                if (!inq[v]) { inq[v] = 1; nxtf[nn++] = v; }
            }
        }
    }
    return nn;
}

/* Only if scratch memory cannot be had: plain edge-list throws, no CSR. */
static void nb_fallback(int n, int m, const int *src, const int *dst,
                        const double *w, double *d)
{
    for (int it = 0; it < n; ++it) {
        int ch = 0;
        for (int e = 0; e < m; ++e) {
            double du = d[src[e]];
            if (du == INFINITY) continue;
            double c = du + w[e];
            if (c < d[dst[e]]) { d[dst[e]] = c; ch = 1; }
        }
        if (!ch) break;
    }
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out)
{
    if (n <= 0) return;

    /* Cold ash everywhere; one banked ember where the traveler stands. */
#pragma omp parallel for schedule(static) if (n > 100000)
    for (int i = 0; i < n; ++i) dist_out[i] = INFINITY;
    if (source < 0 || source >= n) return;
    dist_out[source] = 0.0;
    if (m <= 0) return;

    int           *off = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int           *cur = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int           *eto = (int *)malloc((size_t)m * sizeof(int));
    double        *ew  = (double *)malloc((size_t)m * sizeof(double));
    int           *fa  = (int *)malloc((size_t)n * sizeof(int));
    int           *fb  = (int *)malloc((size_t)n * sizeof(int));
    unsigned char *inq = (unsigned char *)calloc((size_t)n, 1);
    if (!off || !cur || !eto || !ew || !fa || !fb || !inq) {
        free(off); free(cur); free(eto); free(ew);
        free(fa);  free(fb);  free(inq);
        nb_fallback(n, m, src, dst, weight, dist_out);
        return;
    }

    /* The roads, filed by the place they leave from. */
    memset(off, 0, (size_t)(n + 1) * sizeof(int));
#pragma omp parallel for schedule(static) if (m > 200000)
    for (int i = 0; i < m; ++i)
        __atomic_fetch_add(&off[src[i] + 1], 1, __ATOMIC_RELAXED);
    for (int i = 0; i < n; ++i) off[i + 1] += off[i];
    memcpy(cur, off, (size_t)(n + 1) * sizeof(int));
#pragma omp parallel for schedule(static) if (m > 200000)
    for (int i = 0; i < m; ++i) {
        int u = src[i];
        int p = __atomic_fetch_add(&cur[u], 1, __ATOMIC_RELAXED);
        eto[p] = dst[i];
        ew[p]  = weight[i];
    }

    int *curf = fa, *nxtf = fb;
    curf[0] = source;
    int g_nc = 1, g_nn = 0, g_stop = 0;
    long long rounds = 0;
    const long long maxr = (long long)n + 4;   /* cannot spin forever */

    int nthr    = omp_get_max_threads();
    int use_par = (nthr > 1) && (m >= 50000);

    if (!use_par) {
        /* Small land: one hand throws, no muster, no atomics. */
        while (g_nc > 0 && rounds++ <= maxr) {
            int nn = nb_round_serial(g_nc, curf, nxtf, off, eto, ew, inq, dist_out);
            int *t = curf; curf = nxtf; nxtf = t;
            g_nc = nn;
        }
    } else {
        int g_serial = (off[source + 1] - off[source]) < 20000;

        /* One muster for the whole campaign: threads are raised once, not per throw. */
#pragma omp parallel
        {
            int buf[NB_BUF];
            for (;;) {
#pragma omp barrier
                if (g_nc == 0 || g_stop) break;   /* every heap wears its garrison */

                if (g_serial) {
#pragma omp single
                    {
                        g_nn = nb_round_serial(g_nc, curf, nxtf, off, eto, ew,
                                               inq, dist_out);
                        int *t = curf; curf = nxtf; nxtf = t;
                        g_nc = g_nn; g_nn = 0;
                        g_serial = 0;
                        if (g_nc > 0 && g_nc <= 1024) {
                            long long ws = 0;
                            for (int i = 0; i < g_nc; ++i) {
                                int u = curf[i];
                                ws += off[u + 1] - off[u];
                            }
                            if (ws < 20000) g_serial = 1;
                        }
                        if (++rounds > maxr) g_stop = 1;
                    }
                    continue;
                }

                /* Lift the garrisons of the heaps that are about to send runners,
                 * so a stick arriving this throw can re-raise them. */
#pragma omp for schedule(static)
                for (int i = 0; i < g_nc; ++i) inq[curf[i]] = 0;

                /* THE THROW: every un-garrisoned heap's runners go out at once and
                 * every heap keeps only its shortest sliver, all in one breath. */
                int cnt = 0;
#pragma omp for schedule(dynamic, 32) nowait
                for (int i = 0; i < g_nc; ++i) {
                    int u = curf[i];
                    double du = dist_out[u];
                    int e1 = off[u + 1];
                    for (int e = off[u]; e < e1; ++e) {
                        int v = eto[e];
                        double c = du + ew[e];
                        if (c < dist_out[v] && nb_relax(&dist_out[v], c)) {
                            /* its ember shortened: the garrison breaks */
                            if (__atomic_exchange_n(&inq[v], (unsigned char)1,
                                                    __ATOMIC_RELAXED) == 0) {
                                buf[cnt++] = v;
                                if (cnt == NB_BUF) {
                                    int b = __atomic_fetch_add(&g_nn, cnt,
                                                               __ATOMIC_RELAXED);
                                    memcpy(nxtf + b, buf, (size_t)cnt * sizeof(int));
                                    cnt = 0;
                                }
                            }
                        }
                    }
                }
                if (cnt) {
                    int b = __atomic_fetch_add(&g_nn, cnt, __ATOMIC_RELAXED);
                    memcpy(nxtf + b, buf, (size_t)cnt * sizeof(int));
                }

#pragma omp barrier
#pragma omp single
                {
                    int *t = curf; curf = nxtf; nxtf = t;
                    g_nc = g_nn; g_nn = 0;
                    g_serial = 0;
                    if (g_nc > 0 && g_nc <= 1024) {
                        long long ws = 0;
                        for (int i = 0; i < g_nc; ++i) {
                            int u = curf[i];
                            ws += off[u + 1] - off[u];
                        }
                        if (ws < 20000) g_serial = 1;
                    }
                    if (++rounds > maxr) g_stop = 1;
                }
            }
        }
    }

    /* Heaps whose ash never caught a spark keep INFINITY: the dead field.
     * dist_out is already the recitation; nothing to extract. */
    free(off); free(cur); free(eto); free(ew);
    free(fa);  free(fb);  free(inq);
}
