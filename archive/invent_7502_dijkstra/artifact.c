#include <stdlib.h>
#include <math.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* a "peg cluster": the set of nodes currently sitting in one time-bucket */
typedef struct { int *a; int len; int cap; } Bucket;

static void bucket_push(Bucket *b, int v) {
    if (b->len == b->cap) {
        int ncap = b->cap ? b->cap * 2 : 8;
        b->a = realloc(b->a, (size_t)ncap * sizeof(int));
        b->cap = ncap;
    }
    b->a[b->len++] = v;
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight, int source, double *dist_out) {
    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    if (n <= 0) return;
    if (source < 0 || source >= n) return;
    dist_out[source] = 0.0;
    if (m <= 0 || n == 1) return;

    /* CSR: peg -> its outgoing reeds */
    int *deg = calloc((size_t)n, sizeof(int));
    for (int i = 0; i < m; i++) deg[src[i]]++;
    int *off = malloc((size_t)(n + 1) * sizeof(int));
    off[0] = 0;
    for (int i = 0; i < n; i++) off[i + 1] = off[i] + deg[i];
    int *edst = malloc((size_t)m * sizeof(int));
    double *ew = malloc((size_t)m * sizeof(double));
    int *fill = calloc((size_t)n, sizeof(int));
    for (int i = 0; i < m; i++) {
        int u = src[i];
        int pos = off[u] + fill[u]++;
        edst[pos] = dst[i];
        ew[pos] = weight[i];
    }
    free(deg); free(fill);

    /* the widest reed's bore sets the pegboard's clock granularity */
    double maxW = 0.0;
    for (int i = 0; i < m; i++) if (weight[i] > maxW) maxW = weight[i];
    const int BUCKET_TARGET = 256;
    double delta = (maxW > 0.0) ? (maxW / (double)BUCKET_TARGET) : 1.0;
    if (!(delta > 0.0)) delta = 1.0;
    int L = BUCKET_TARGET + 4; /* circular pegboard: a bucket can never be re-touched
                                  once the clock has moved past its whole reachable window */

    Bucket *buckets = calloc((size_t)L, sizeof(Bucket));
    double *lastVal = malloc((size_t)n * sizeof(double));
    for (int i = 0; i < n; i++) lastVal[i] = -1.0;

    /* pour water into the traveler's own peg */
    bucket_push(&buckets[0], source);

    long long curBucket = 0;
    long long emptyStreak = 0;
    int *batch = NULL; int batchCap = 0;

    while (emptyStreak < L) {
        int slot = (int)(((curBucket % L) + L) % L);
        if (buckets[slot].len == 0) { emptyStreak++; curBucket++; continue; }
        emptyStreak = 0;

        for (;;) {
            int blen = buckets[slot].len;
            if (blen == 0) break;
            if (blen > batchCap) { batch = realloc(batch, (size_t)blen * sizeof(int)); batchCap = blen; }
            memcpy(batch, buckets[slot].a, (size_t)blen * sizeof(int));
            buckets[slot].len = 0; /* drained; fresh pours during this round land here anew */

            #pragma omp parallel for schedule(dynamic, 64)
            for (int bi = 0; bi < blen; bi++) {
                int u = batch[bi];
                double du; int doWork = 0;
                #pragma omp critical(pegboard)
                {
                    du = dist_out[u];
                    if (lastVal[u] != du) { lastVal[u] = du; doWork = 1; }
                }
                if (!doWork) continue;
                for (int e = off[u]; e < off[u + 1]; e++) {
                    int w = edst[e];
                    double nd = du + ew[e];
                    if (nd < dist_out[w]) {           /* dirty pre-check, safe: dist only shrinks */
                        #pragma omp critical(pegboard)
                        {
                            if (nd < dist_out[w]) {   /* first wet mark wins; scrape the rest */
                                dist_out[w] = nd;
                                long long g = (long long)floor(nd / delta);
                                if (g < curBucket) g = curBucket; /* fp-boundary safety clamp */
                                int wslot = (int)(((g % L) + L) % L);
                                bucket_push(&buckets[wslot], w);
                            }
                        }
                    }
                }
            }
        }
        curBucket++;
    }

    for (int i = 0; i < L; i++) free(buckets[i].a);
    free(buckets); free(lastVal); free(batch);
    free(off); free(edst); free(ew);
}
