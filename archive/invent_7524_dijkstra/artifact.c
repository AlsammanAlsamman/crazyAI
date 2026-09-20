#include <stdlib.h>
#include <math.h>
#include <string.h>

/* ---- Known practical win for small/dense graphs: O(n^2) array scan, no heap at all ---- */
static void dijkstra_array_scan(int n, const int *off, const int *edst, const double *ew,
                                 int source, double *dist_out) {
    char *done = calloc((size_t)n, 1);
    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    dist_out[source] = 0.0;
    for (int iter = 0; iter < n; iter++) {
        int u = -1; double best = INFINITY;
        for (int v = 0; v < n; v++) {
            if (!done[v] && dist_out[v] < best) { best = dist_out[v]; u = v; }
        }
        if (u == -1) break;
        done[u] = 1;
        double du = dist_out[u];
        for (int e = off[u]; e < off[u + 1]; e++) {
            int v = edst[e];
            double nd = du + ew[e];
            if (nd < dist_out[v]) dist_out[v] = nd;
        }
    }
    free(done);
}

/* ---- Delta-stepping: jewels=vertices, cords=edges, glint pace=edge weight.
   Chalk grid = bucket array indexed by floor(tent/delta). Light cords
   ("short, flash at once") are relaxed repeatedly inside the current bucket
   phase, before the tail vertex is finally done -- this is the piece that
   breaks "a road can only be considered once its starting place is settled".
   Heavy cords ("long, glint keeps waiting") are relaxed once, only after the
   phase's vertex set R is final, preserving correctness for non-negative
   weights. ---- */
static void dijkstra_delta_stepping(int n, int m, const int *off, const int *edst, const double *ew,
                                     int source, double *dist_out) {
    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    if (m == 0) { dist_out[source] = 0.0; return; }

    int *e2 = malloc((size_t)m * sizeof(int));
    double *w2 = malloc((size_t)m * sizeof(double));
    memcpy(e2, edst, (size_t)m * sizeof(int));
    memcpy(w2, ew, (size_t)m * sizeof(double));

    double sumw = 0.0;
    for (int i = 0; i < m; i++) sumw += w2[i];
    double delta = sumw / (double)m;
    if (!(delta > 1e-12)) delta = 1.0;

    int *lightEnd = malloc((size_t)n * sizeof(int));
    for (int u = 0; u < n; u++) {
        int lo = off[u], hi = off[u + 1];
        int i = lo, j = hi - 1;
        while (i <= j) {
            while (i <= j && w2[i] <= delta) i++;
            while (i <= j && w2[j] > delta) j--;
            if (i < j) {
                int ti = e2[i]; e2[i] = e2[j]; e2[j] = ti;
                double tw = w2[i]; w2[i] = w2[j]; w2[j] = tw;
                i++; j--;
            }
        }
        lightEnd[u] = i;
    }

    double *tent = dist_out;
    int *nextInList = malloc((size_t)n * sizeof(int));
    int *prevInList = malloc((size_t)n * sizeof(int));
    int *curBucket  = malloc((size_t)n * sizeof(int));
    for (int i = 0; i < n; i++) curBucket[i] = -1;

    int cap = 64;
    int *head = malloc((size_t)cap * sizeof(int));
    for (int i = 0; i < cap; i++) head[i] = -1;

#define ENSURE_CAP(b) do { \
        if ((b) >= cap) { \
            int newcap = cap; \
            while (newcap <= (b)) newcap *= 2; \
            head = realloc(head, (size_t)newcap * sizeof(int)); \
            for (int k = cap; k < newcap; k++) head[k] = -1; \
            cap = newcap; \
        } \
    } while (0)

#define BREMOVE(u) do { \
        int b_ = curBucket[u]; \
        if (b_ != -1) { \
            if (prevInList[u] != -1) nextInList[prevInList[u]] = nextInList[u]; \
            else head[b_] = nextInList[u]; \
            if (nextInList[u] != -1) prevInList[nextInList[u]] = prevInList[u]; \
            curBucket[u] = -1; \
        } \
    } while (0)

#define BINSERT(u, b) do { \
        ENSURE_CAP(b); \
        BREMOVE(u); \
        nextInList[u] = head[b]; \
        prevInList[u] = -1; \
        if (head[b] != -1) prevInList[head[b]] = u; \
        head[b] = u; \
        curBucket[u] = b; \
    } while (0)

    tent[source] = 0.0;
    BINSERT(source, 0);

    int *Rlist = malloc((size_t)n * sizeof(int));
    char *inR = calloc((size_t)n, 1);

    int i = 0;
    for (;;) {
        while (i < cap && head[i] == -1) i++;
        if (i >= cap) break;
        int Rcount = 0;
        while (head[i] != -1) {
            int u = head[i];
            BREMOVE(u);
            if (!inR[u]) { inR[u] = 1; Rlist[Rcount++] = u; }
            double du = tent[u];
            for (int e = off[u]; e < lightEnd[u]; e++) {
                int v = e2[e];
                double nd = du + w2[e];
                if (nd < tent[v]) {
                    tent[v] = nd;
                    int b = (int)(nd / delta);
                    BINSERT(v, b);
                }
            }
        }
        for (int r = 0; r < Rcount; r++) {
            int u = Rlist[r];
            double du = tent[u];
            for (int e = lightEnd[u]; e < off[u + 1]; e++) {
                int v = e2[e];
                double nd = du + w2[e];
                if (nd < tent[v]) {
                    tent[v] = nd;
                    int b = (int)(nd / delta);
                    BINSERT(v, b);
                }
            }
            inR[u] = 0;
        }
        i++;
    }

    free(e2); free(w2); free(lightEnd);
    free(nextInList); free(prevInList); free(curBucket); free(head);
    free(Rlist); free(inR);
#undef ENSURE_CAP
#undef BREMOVE
#undef BINSERT
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight, int source, double *dist_out) {
    if (n <= 0) return;

    int *deg = calloc((size_t)n, sizeof(int));
    for (int i = 0; i < m; i++) deg[src[i]]++;
    int *off = malloc((size_t)(n + 1) * sizeof(int));
    off[0] = 0;
    for (int i = 0; i < n; i++) off[i + 1] = off[i] + deg[i];
    int *edst = malloc((size_t)(m > 0 ? m : 1) * sizeof(int));
    double *ew = malloc((size_t)(m > 0 ? m : 1) * sizeof(double));
    int *fill = calloc((size_t)n, sizeof(int));
    for (int i = 0; i < m; i++) {
        int u = src[i]; int pos = off[u] + fill[u]++;
        edst[pos] = dst[i]; ew[pos] = weight[i];
    }

    /* Guard: sequential Delta-stepping's bucket bookkeeping only pays for
       itself on larger, sparse graphs. For small or dense graphs, fall back
       to the known, validated O(n^2) array scan instead of risking overhead. */
    long long nn = (long long)n * (long long)n;
    int small_or_dense = (n <= 2000) || ((long long)m * 8LL >= nn);

    if (small_or_dense) dijkstra_array_scan(n, off, edst, ew, source, dist_out);
    else                dijkstra_delta_stepping(n, m, off, edst, ew, source, dist_out);

    free(deg); free(off); free(edst); free(ew); free(fill);
}
