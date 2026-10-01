/* The garden of nightingale: in-place chalk (O(1) decrease-key), no priority
   structure touched during relaxation, priority manufactured on demand by a
   SIMD sweep over the reached-and-unlocked stones only.
   Regime fallback: a tiered roost (binary heap) chosen up front for wide/thin
   gardens, or mid-flight when the birds' wingbeats exceed their budget. */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#if defined(__AVX__)
#include <immintrin.h>
#endif

typedef struct { double d; int u; } HeapItem;

/* ---- the tiered roost: lightest letter on top (hole-based sift, no swaps) ---- */
static void roost_push(HeapItem *restrict h, int *restrict hs, double d, int u) {
    int i = (*hs)++;
    while (i > 0) {
        int p = (i - 1) >> 1;
        if (h[p].d <= d) break;
        h[i] = h[p];
        i = p;
    }
    h[i].d = d; h[i].u = u;
}
static HeapItem roost_pop(HeapItem *restrict h, int *restrict hs) {
    HeapItem top = h[0];
    int sz = --(*hs);
    if (sz > 0) {
        HeapItem last = h[sz];
        int i = 0;
        for (;;) {
            int l = 2 * i + 1;
            if (l >= sz) break;
            int r = l + 1;
            int s = (r < sz && h[r].d < h[l].d) ? r : l;
            if (h[s].d >= last.d) break;
            h[i] = h[s];
            i = s;
        }
        h[i] = last;
    }
    return top;
}

/* ---- the nightingale sweep: drop on the smallest owed letter ---- */
static int bird_drop(const double *restrict d, int nn) {
#if defined(__AVX__)
    if (nn >= 16) {
        __m256d a = _mm256_loadu_pd(d);
        __m256d b = _mm256_loadu_pd(d + 4);
        int i = 8;
        for (; i + 8 <= nn; i += 8) {
            a = _mm256_min_pd(a, _mm256_loadu_pd(d + i));
            b = _mm256_min_pd(b, _mm256_loadu_pd(d + i + 4));
        }
        a = _mm256_min_pd(a, b);
        double t[4];
        _mm256_storeu_pd(t, a);
        double mv = t[0];
        if (t[1] < mv) mv = t[1];
        if (t[2] < mv) mv = t[2];
        if (t[3] < mv) mv = t[3];
        for (; i < nn; i++) if (d[i] < mv) mv = d[i];
        __m256d vv = _mm256_set1_pd(mv);
        int j = 0;
        for (; j + 4 <= nn; j += 4) {
            int msk = _mm256_movemask_pd(
                _mm256_cmp_pd(_mm256_loadu_pd(d + j), vv, _CMP_EQ_OQ));
            if (msk) return j + (int)__builtin_ctz((unsigned)msk);
        }
        for (; j < nn; j++) if (d[j] == mv) return j;
        return 0;
    }
#endif
    {
        int best = 0;
        double bv = d[0];
        for (int i = 1; i < nn; i++) { double x = d[i]; if (x < bv) { bv = x; best = i; } }
        return best;
    }
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out)
{
    if (n <= 0) return;
    double *restrict dd = dist_out;
    for (int i = 0; i < n; i++) dd[i] = INFINITY;
    if ((unsigned)source >= (unsigned)n) return;
    dd[source] = 0.0;
    if (m < 0) m = 0;

    const int *restrict es = src;
    const int *restrict ed = dst;
    const double *restrict ww = weight;

    /* ---- cast the roads into buckets by their starting house; let dead
            threads drop (endpoint outside the garden, or a wall-loop) ---- */
    int *restrict off = (int *)calloc((size_t)n + 1u, sizeof(int));
    if (!off) return;
    for (int i = 0; i < m; i++) {
        int u = es[i], v = ed[i];
        if ((unsigned)u < (unsigned)n && (unsigned)v < (unsigned)n && u != v) off[u + 1]++;
    }
    for (int i = 0; i < n; i++) off[i + 1] += off[i];
    int mk = off[n];
    size_t am = (size_t)(mk > 0 ? mk : 1);

    int *restrict cur = (int *)malloc((size_t)n * sizeof(int));
    int *restrict av  = (int *)malloc(am * sizeof(int));
    double *restrict aw = (double *)malloc(am * sizeof(double));
    int *restrict pos = (int *)malloc((size_t)n * sizeof(int));
    int *restrict cid = (int *)malloc((size_t)n * sizeof(int));
    double *restrict cdv = (double *)malloc((size_t)n * sizeof(double));
    if (!cur || !av || !aw || !pos || !cid || !cdv) {
        free(off); free(cur); free(av); free(aw); free(pos); free(cid); free(cdv);
        return;
    }
    memcpy(cur, off, (size_t)n * sizeof(int));
    for (int i = 0; i < m; i++) {
        int u = es[i], v = ed[i];
        if ((unsigned)u < (unsigned)n && (unsigned)v < (unsigned)n && u != v) {
            int p = cur[u]++;
            av[p] = v;
            aw[p] = ww[i];
        }
    }

    /* ---- pace the wall: which regime is this garden in? ---- */
    double lg = log2((double)n + 2.0);
    double heap_scale = (double)(n + mk) * lg;
    int use_sweep = ((double)n * (double)n <= 512.0 * heap_scale);
    double budget = 1.5 * heap_scale + 4096.0;   /* bird-stone visits allowed */
    double wing = 0.0;
    int ncand = 0;

    if (use_sweep) {
        for (int i = 0; i < n; i++) pos[i] = -1;
        cid[0] = source; cdv[0] = 0.0; pos[source] = 0; ncand = 1;
        while (ncand > 0) {
            if (wing > budget) break;            /* call the birds down */
            wing += (double)ncand;
            int bi = bird_drop(cdv, ncand);
            int u = cid[bi];
            double du = cdv[bi];
            int last = --ncand;                  /* the stone leaves the sweep: locked */
            if (bi != last) {
                cdv[bi] = cdv[last];
                int mv = cid[last];
                cid[bi] = mv;
                pos[mv] = bi;
            }
            pos[u] = -1;
            int e1 = off[u + 1];
            for (int e = off[u]; e < e1; e++) {
                int v = av[e];
                double nd = du + aw[e];
                /* no lock test: no thief charges a negative toll, so a locked
                   stone can never fail this compare (dd[v] <= du <= nd). */
                if (nd < dd[v]) {
                    dd[v] = nd;
                    int p = pos[v];
                    if (p >= 0) {
                        cdv[p] = nd;             /* detached and called back on */
                    } else {
                        p = ncand++;
                        cid[p] = v; cdv[p] = nd; pos[v] = p;
                    }
                }
            }
        }
    }

    if (!use_sweep || ncand > 0) {
        HeapItem *restrict h =
            (HeapItem *)malloc(((size_t)mk + (size_t)n + 2u) * sizeof(HeapItem));
        if (h) {
            int hs = 0;
            if (use_sweep) {
                for (int k = 0; k < ncand; k++) roost_push(h, &hs, cdv[k], cid[k]);
            } else {
                roost_push(h, &hs, 0.0, source);
            }
            while (hs > 0) {
                HeapItem t = roost_pop(h, &hs);
                int u = t.u;
                double du = dd[u];
                if (t.d > du) continue;          /* stale letter; the stone is set */
                int e1 = off[u + 1];
                for (int e = off[u]; e < e1; e++) {
                    int v = av[e];
                    double nd = du + aw[e];
                    if (nd < dd[v]) { dd[v] = nd; roost_push(h, &hs, nd, v); }
                }
            }
            free(h);
        }
    }

    free(off); free(cur); free(av); free(aw); free(pos); free(cid); free(cdv);
}
