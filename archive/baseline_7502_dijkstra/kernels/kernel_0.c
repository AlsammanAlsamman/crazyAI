#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <omp.h>

static inline void atomic_min_double(double *addr, double val) {
    uint64_t new_bits;
    memcpy(&new_bits, &val, sizeof(new_bits));
    uint64_t old_bits = __atomic_load_n((uint64_t*)addr, __ATOMIC_RELAXED);
    for (;;) {
        double old_val;
        memcpy(&old_val, &old_bits, sizeof(old_val));
        if (!(val < old_val)) return;
        if (__atomic_compare_exchange_n((uint64_t*)addr, &old_bits, new_bits, 1,
                                         __ATOMIC_RELAXED, __ATOMIC_RELAXED))
            return;
    }
}

static void dense_dijkstra(int n, const int *off, const int *edst, const double *ew,
                            int source, double *dist_out) {
    double *key = (double*)malloc((size_t)n * sizeof(double));
    char *done = (char*)calloc((size_t)n, 1);
    for (int i = 0; i < n; i++) key[i] = INFINITY;
    key[source] = 0.0;

    int use_par = (n >= 4000) && (omp_get_max_threads() > 1);

    for (int iter = 0; iter < n; iter++) {
        int bestu = -1;
        double best = INFINITY;
        if (use_par) {
            #pragma omp parallel
            {
                double lbest = INFINITY;
                int lu = -1;
                #pragma omp for nowait
                for (int v = 0; v < n; v++) {
                    double kv = key[v];
                    if (kv < lbest) { lbest = kv; lu = v; }
                }
                #pragma omp critical
                {
                    if (lbest < best) { best = lbest; bestu = lu; }
                }
            }
        } else {
            for (int v = 0; v < n; v++) {
                double kv = key[v];
                if (kv < best) { best = kv; bestu = v; }
            }
        }
        if (bestu < 0 || best == INFINITY) break;
        dist_out[bestu] = best;
        done[bestu] = 1;
        key[bestu] = INFINITY;
        int rs = off[bestu], re = off[bestu + 1];
        double du = best;
        if (use_par && (re - rs) > 2048) {
            #pragma omp parallel for schedule(static)
            for (int e = rs; e < re; e++) {
                int v = edst[e];
                if (!done[v]) {
                    double nd = du + ew[e];
                    if (nd < key[v]) atomic_min_double(&key[v], nd);
                }
            }
        } else {
            for (int e = rs; e < re; e++) {
                int v = edst[e];
                if (!done[v]) {
                    double nd = du + ew[e];
                    if (nd < key[v]) key[v] = nd;
                }
            }
        }
    }
    free(key);
    free(done);
}

static void sparse_dijkstra(int n, const int *off, const int *edst, const double *ew,
                             int source, double *dist_out) {
    int *pos = (int*)malloc((size_t)n * sizeof(int));
    int *heap_node = (int*)malloc((size_t)n * sizeof(int));
    double *heap_dist = (double*)malloc((size_t)n * sizeof(double));
    for (int i = 0; i < n; i++) pos[i] = -1;
    int hs = 0;

    dist_out[source] = 0.0;
    heap_node[0] = source; heap_dist[0] = 0.0; pos[source] = 0; hs = 1;

    while (hs > 0) {
        int u = heap_node[0];
        double du = heap_dist[0];
        hs--;
        if (hs > 0) {
            heap_node[0] = heap_node[hs];
            heap_dist[0] = heap_dist[hs];
            pos[heap_node[0]] = 0;
            int i = 0;
            for (;;) {
                int c0 = 4 * i + 1;
                if (c0 >= hs) break;
                int smallest = c0;
                double smallest_d = heap_dist[c0];
                int cend = c0 + 4; if (cend > hs) cend = hs;
                for (int c = c0 + 1; c < cend; c++) {
                    if (heap_dist[c] < smallest_d) { smallest = c; smallest_d = heap_dist[c]; }
                }
                if (smallest_d >= heap_dist[i]) break;
                int tn = heap_node[i]; double td = heap_dist[i];
                heap_node[i] = heap_node[smallest]; heap_dist[i] = heap_dist[smallest];
                heap_node[smallest] = tn; heap_dist[smallest] = td;
                pos[heap_node[i]] = i; pos[heap_node[smallest]] = smallest;
                i = smallest;
            }
        }
        pos[u] = -2;

        int rs = off[u], re = off[u + 1];
        for (int e = rs; e < re; e++) {
            int v = edst[e];
            if (pos[v] == -2) continue;
            double nd = du + ew[e];
            if (nd < dist_out[v]) {
                dist_out[v] = nd;
                int i;
                if (pos[v] == -1) {
                    i = hs++;
                    heap_node[i] = v; heap_dist[i] = nd; pos[v] = i;
                } else {
                    i = pos[v];
                    heap_dist[i] = nd;
                }
                while (i > 0) {
                    int p = (i - 1) / 4;
                    if (heap_dist[p] <= heap_dist[i]) break;
                    int tn = heap_node[p]; double td = heap_dist[p];
                    heap_node[p] = heap_node[i]; heap_dist[p] = heap_dist[i];
                    heap_node[i] = tn; heap_dist[i] = td;
                    pos[heap_node[p]] = p; pos[heap_node[i]] = i;
                    i = p;
                }
            }
        }
    }
    free(pos); free(heap_node); free(heap_dist);
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out) {
    if (n <= 0) return;
    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    if (source < 0 || source >= n) return;

    int *deg = (int*)calloc((size_t)n, sizeof(int));
    for (int i = 0; i < m; i++) deg[src[i]]++;
    int *off = (int*)malloc((size_t)(n + 1) * sizeof(int));
    off[0] = 0;
    for (int i = 0; i < n; i++) off[i + 1] = off[i] + deg[i];
    size_t msz = (size_t)(m > 0 ? m : 1);
    int *edst = (int*)malloc(msz * sizeof(int));
    double *ew = (double*)malloc(msz * sizeof(double));
    int *fillpos = (int*)malloc((size_t)n * sizeof(int));
    memcpy(fillpos, off, (size_t)n * sizeof(int));
    for (int i = 0; i < m; i++) {
        int u = src[i];
        int p = fillpos[u]++;
        edst[p] = dst[i];
        ew[p] = weight[i];
    }
    free(deg);
    free(fillpos);

    double logn = log2((double)n + 2.0);
    double dense_cost = (double)n * (double)n;
    double sparse_cost = ((double)n + (double)m) * logn * 3.0;

    if (dense_cost <= sparse_cost) {
        dense_dijkstra(n, off, edst, ew, source, dist_out);
    } else {
        sparse_dijkstra(n, off, edst, ew, source, dist_out);
    }

    free(off); free(edst); free(ew);
}
