#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct { double w; int v; int pad; } Edge;

/* Safety valve only: bounded-work fallback for adversarial graphs. */
static void dijkstra_fallback(int n, int m, const int *off, const Edge *E,
                              int source, double *dist, unsigned char *done)
{
    for (int i = 0; i < n; i++) dist[i] = INFINITY;
    dist[source] = 0.0;
    memset(done, 0, (size_t)n);
    Edge *h = (Edge *)malloc(((size_t)m + 2) * sizeof(Edge));
    if (!h) return;
    int hs = 1;
    h[0].w = 0.0; h[0].v = source;
    while (hs > 0) {
        int u = h[0].v;
        hs--;
        h[0] = h[hs];
        int i = 0;
        for (;;) {
            int l = 2 * i + 1;
            if (l >= hs) break;
            int s = l;
            if (l + 1 < hs && h[l + 1].w < h[l].w) s = l + 1;
            if (!(h[s].w < h[i].w)) break;
            Edge t = h[s]; h[s] = h[i]; h[i] = t;
            i = s;
        }
        if (done[u]) continue;
        done[u] = 1;
        double du = dist[u];
        for (int e = off[u]; e < off[u + 1]; e++) {
            int v = E[e].v;
            double nd = du + E[e].w;
            if (nd < dist[v]) {
                dist[v] = nd;
                int j = hs++;
                h[j].w = nd; h[j].v = v;
                while (j > 0) {
                    int p = (j - 1) >> 1;
                    if (h[p].w <= h[j].w) break;
                    Edge t = h[p]; h[p] = h[j]; h[j] = t;
                    j = p;
                }
            }
        }
    }
    free(h);
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out)
{
    if (n <= 0) return;
    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    if (source < 0 || source >= n) return;
    dist_out[source] = 0.0;
    if (m <= 0) return;

    const size_t nn = (size_t)n;
    const size_t A  = 64;
    size_t b_off = (((nn + 2) * sizeof(int) + A - 1) / A) * A;
    size_t b_dq  = (((nn + 2) * sizeof(int) + A - 1) / A) * A;
    size_t b_ed  = (((size_t)m * sizeof(Edge) + A - 1) / A) * A;
    size_t b_inq = (((nn + A) + A - 1) / A) * A;

    char *arena = (char *)malloc(b_off + b_dq + b_ed + b_inq + A);
    if (!arena) return;
    char *p = arena;
    size_t mis = (size_t)p & (A - 1);
    if (mis) p += A - mis;

    int  *off = (int *)p;
    int  *dq  = (int *)(p + b_off);                       /* CSR cursor, then deque */
    Edge *E   = (Edge *)(p + b_off + b_dq);
    unsigned char *inq = (unsigned char *)(p + b_off + b_dq + b_ed);

    /* ---- CSR build: count, prefix, scatter (cursor aliases the deque buffer) ---- */
    memset(off, 0, (nn + 2) * sizeof(int));
    for (int i = 0; i < m; i++) off[src[i] + 1]++;
    for (int i = 0; i < n; i++) off[i + 1] += off[i];
    memcpy(dq, off, nn * sizeof(int));
    for (int i = 0; i < m; i++) {
        int u = src[i];
        int q = dq[u]++;
        E[q].v = dst[i];
        E[q].w = weight[i];
    }
    memset(inq, 0, nn);

    /* ---- pure label-correcting gossip: no "settled" set, no priority queue ---- */
    double     * restrict D  = dist_out;
    const Edge * restrict EA = E;

    const int cap = n + 1;
    int head = 0, tail = 0;
    dq[tail++] = source;
    inq[source] = 1;

    const long long budget = 32LL * ((long long)m + (long long)n) + 4096LL;
    long long work = 0;
    int overflow = 0;

    while (head != tail) {
        int u = dq[head];
        head++; if (head == cap) head = 0;
        inq[u] = 0;

        const double du = D[u];
        int e = off[u], ee = off[u + 1];

        work += (long long)(ee - e) + 1;
        if (work > budget) { overflow = 1; break; }

        const int elim = ee - 4;
        for (; e < ee; e++) {
            if (e < elim) __builtin_prefetch(&D[EA[e + 4].v], 1, 1);
            int v = EA[e].v;
            double nd = du + EA[e].w;
            if (nd < D[v]) {                 /* the one and only gate */
                D[v] = nd;
                if (!inq[v]) {
                    inq[v] = 1;
                    /* small-label-first: better-than-front goes to the front */
                    if (head != tail && nd < D[dq[head]]) {
                        head = (head == 0) ? cap - 1 : head - 1;
                        dq[head] = v;
                    } else {
                        dq[tail] = v;
                        tail++; if (tail == cap) tail = 0;
                    }
                }
            }
        }
    }

    if (overflow) dijkstra_fallback(n, m, off, E, source, dist_out, inq);

    free(arena);
}
