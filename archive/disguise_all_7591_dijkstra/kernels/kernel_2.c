#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define INF_BITS 0x7FF0000000000000ULL
#define LBUF 512
#define PAR_EDGES 30000
#define PAR_NODES 16

static inline double b2d(uint64_t b) { double d; memcpy(&d, &b, sizeof d); return d; }
static inline uint64_t d2b(double d) { uint64_t b; memcpy(&b, &d, sizeof b); return b; }

/* lock-free "is that better than what you've got?" on non-negative doubles,
   compared as uint64 bit patterns (order-preserving for non-negative IEEE-754). */
static inline int amin_u64(uint64_t *p, uint64_t val)
{
    uint64_t old = __atomic_load_n(p, __ATOMIC_RELAXED);
    while (val < old) {
        if (__atomic_compare_exchange_n(p, &old, val, 0,
                                        __ATOMIC_RELAXED, __ATOMIC_RELAXED))
            return 1;
    }
    return 0;
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out)
{
    if (n <= 0) return;
    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    if (source < 0 || source >= n) return;
    if (m < 0) m = 0;

    size_t ms = (size_t)(m > 0 ? m : 1);
    int      *off  = (int *)calloc((size_t)n + 1, sizeof(int));
    int      *cur  = (int *)malloc((size_t)n * sizeof(int));
    int      *edst = (int *)malloc(ms * sizeof(int));
    double   *ew   = (double *)malloc(ms * sizeof(double));
    uint64_t *D    = (uint64_t *)malloc((size_t)n * sizeof(uint64_t));
    unsigned char *inq = (unsigned char *)calloc((size_t)n, 1);
    int *fr = (int *)malloc((size_t)n * sizeof(int));
    int *nx = (int *)malloc((size_t)n * sizeof(int));

    if (!off || !cur || !edst || !ew || !D || !inq || !fr || !nx) {
        /* degenerate safety: cannot proceed; leave INF except source */
        dist_out[source] = 0.0;
        free(off); free(cur); free(edst); free(ew); free(D); free(inq);
        free(fr); free(nx);
        return;
    }

    /* ---- CSR build: two passes over the edge list, no heap anywhere ---- */
    for (int i = 0; i < m; i++) off[src[i] + 1]++;
    for (int u = 0; u < n; u++) off[u + 1] += off[u];
    memcpy(cur, off, (size_t)n * sizeof(int));
    for (int i = 0; i < m; i++) {
        int u = src[i];
        int p = cur[u]++;
        edst[p] = dst[i];
        ew[p]   = weight[i];
    }

#ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(n > 200000)
#endif
    for (int i = 0; i < n; i++) D[i] = INF_BITS;

    D[source] = 0ULL;                 /* bits of +0.0 */
    fr[0] = source;
    int fsize = 1;

    /* ---- waves: no node is ever finalized, no priority structure ---- */
    while (fsize > 0) {
        long long fe = 0;
        for (int k = 0; k < fsize; k++) {
            int u = fr[k];
            inq[u] = 0;               /* u may legitimately re-activate this wave */
            fe += (long long)(off[u + 1] - off[u]);
        }
        int nsz = 0;

#ifdef _OPENMP
        if (fe >= PAR_EDGES && fsize >= PAR_NODES) {
            #pragma omp parallel
            {
                int lbuf[LBUF];
                int lc = 0;
                #pragma omp for schedule(guided) nowait
                for (int k = 0; k < fsize; k++) {
                    int u = fr[k];
                    double du = b2d(__atomic_load_n(&D[u], __ATOMIC_RELAXED));
                    int e0 = off[u], e1 = off[u + 1];
                    for (int e = e0; e < e1; e++) {
                        int v = edst[e];
                        uint64_t nb = d2b(du + ew[e]);
                        if (nb < __atomic_load_n(&D[v], __ATOMIC_RELAXED)) {
                            if (amin_u64(&D[v], nb)) {
                                if (__atomic_exchange_n(&inq[v], (unsigned char)1,
                                                        __ATOMIC_RELAXED) == 0) {
                                    lbuf[lc++] = v;
                                    if (lc == LBUF) {
                                        int base = __atomic_fetch_add(&nsz, lc,
                                                        __ATOMIC_RELAXED);
                                        memcpy(nx + base, lbuf,
                                               (size_t)lc * sizeof(int));
                                        lc = 0;
                                    }
                                }
                            }
                        }
                    }
                }
                if (lc) {
                    int base = __atomic_fetch_add(&nsz, lc, __ATOMIC_RELAXED);
                    memcpy(nx + base, lbuf, (size_t)lc * sizeof(int));
                }
            }
        } else
#endif
        {
            for (int k = 0; k < fsize; k++) {
                int u = fr[k];
                double du = b2d(D[u]);
                int e0 = off[u], e1 = off[u + 1];
                for (int e = e0; e < e1; e++) {
                    int v = edst[e];
                    uint64_t nb = d2b(du + ew[e]);
                    if (nb < D[v]) {
                        D[v] = nb;
                        if (!inq[v]) { inq[v] = 1; nx[nsz++] = v; }
                    }
                }
            }
        }

        int *t = fr; fr = nx; nx = t;
        fsize = nsz;
    }

    memcpy(dist_out, D, (size_t)n * sizeof(double));   /* bits are valid doubles */

    free(off); free(cur); free(edst); free(ew); free(D); free(inq);
    free(fr); free(nx);
}
