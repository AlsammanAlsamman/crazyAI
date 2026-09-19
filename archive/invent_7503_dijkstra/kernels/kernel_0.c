#include <stdlib.h>
#include <math.h>

/* Literal mapping of the pit-board / sowing story to O(n^2 + m) Dijkstra:
   - place        -> node (pit)
   - road         -> directed edge (thread), thread-length -> weight
   - pit's heap   -> dist_out[v], starts "heaped full" = INFINITY, except
                     the traveler's own empty pit -> dist_out[source] = 0
   - blue flame   -> done[] flag marking a pit settled, touched once
   - "choose whichever unsettled pit holds fewest seeds" -> a full linear
                     scan over every undone pit each round (NO priority
                     queue / heap object exists anywhere in this story)
   - sowing a palmful of seeds one at a time along each thread, replacing
     the far pit's heap only when the sown count is smaller -> the
     classic Dijkstra relaxation: nd = dist[u] + w; if (nd < dist[v]) ...
*/
void kernel(int n, int m, const int *src, const int *dst, const double *weight, int source, double *dist_out) {
    /* Build the "threads leaving each pit" structure (CSR adjacency) */
    int *deg = (int *)calloc((size_t)n, sizeof(int));
    for (int i = 0; i < m; i++) deg[src[i]]++;
    int *off = (int *)malloc((size_t)(n + 1) * sizeof(int));
    off[0] = 0;
    for (int i = 0; i < n; i++) off[i + 1] = off[i] + deg[i];
    int *edst = (int *)malloc((size_t)m * sizeof(int));
    double *ew = (double *)malloc((size_t)m * sizeof(double));
    int *fill = (int *)calloc((size_t)n, sizeof(int));
    for (int i = 0; i < m; i++) {
        int u = src[i];
        int pos = off[u] + fill[u]++;
        edst[pos] = dst[i];
        ew[pos] = weight[i];
    }

    /* Every pit starts heaped to the brim (unreached), the traveler's own
       pit starts empty. */
    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    dist_out[source] = 0.0;

    char *done = (char *)calloc((size_t)n, 1);

    for (int round = 0; round < n; round++) {
        /* "each round, I choose whichever unsettled pit holds the fewest
           seeds" -- a bare scan across every remaining pit, no heap. */
        int u = -1;
        double best = INFINITY;
        for (int v = 0; v < n; v++) {
            if (!done[v] && dist_out[v] < best) {
                best = dist_out[v];
                u = v;
            }
        }
        if (u == -1) break; /* everything left is unreachable: still full pits */

        /* touch it with the blue flame: settled, never scanned again */
        done[u] = 1;

        /* sow along every thread leaving this pit */
        double du = dist_out[u];
        for (int e = off[u]; e < off[u + 1]; e++) {
            int v = edst[e];
            if (done[v]) continue;
            double nd = du + ew[e];
            if (nd < dist_out[v]) dist_out[v] = nd;
        }
    }

    free(deg); free(off); free(edst); free(ew); free(fill); free(done);
}
