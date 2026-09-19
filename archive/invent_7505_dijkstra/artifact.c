#include <stdlib.h>
#include <math.h>

typedef struct { int *arr; int cnt; int cap; } Bucket;
typedef struct { Bucket *b; int cap; } Buckets;

static void buckets_ensure(Buckets *B, int idx) {
    if (idx < B->cap) return;
    int newcap = B->cap ? B->cap : 16;
    while (idx >= newcap) newcap *= 2;
    B->b = (Bucket*)realloc(B->b, (size_t)newcap * sizeof(Bucket));
    for (int i = B->cap; i < newcap; i++) { B->b[i].arr = NULL; B->b[i].cnt = 0; B->b[i].cap = 0; }
    B->cap = newcap;
}

static void bucket_add(Buckets *B, int idx, int v) {
    buckets_ensure(B, idx);
    Bucket *bk = &B->b[idx];
    if (bk->cnt == bk->cap) {
        bk->cap = bk->cap ? bk->cap * 2 : 4;
        bk->arr = (int*)realloc(bk->arr, (size_t)bk->cap * sizeof(int));
    }
    bk->arr[bk->cnt++] = v;
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight, int source, double *dist_out) {
    int *deg = (int*)calloc((size_t)n, sizeof(int));
    for (int i = 0; i < m; i++) deg[src[i]]++;
    int *off = (int*)malloc((size_t)(n + 1) * sizeof(int));
    off[0] = 0;
    for (int i = 0; i < n; i++) off[i + 1] = off[i] + deg[i];
    int *edst = (int*)malloc((size_t)(m > 0 ? m : 1) * sizeof(int));
    double *ew = (double*)malloc((size_t)(m > 0 ? m : 1) * sizeof(double));
    int *fillp = (int*)calloc((size_t)n, sizeof(int));
    double wmax = 0.0, wsum = 0.0;
    for (int i = 0; i < m; i++) {
        int u = src[i];
        int pos = off[u] + fillp[u]++;
        edst[pos] = dst[i];
        ew[pos] = weight[i];
        if (weight[i] > wmax) wmax = weight[i];
        wsum += weight[i];
    }
    free(fillp);

    double delta = (m > 0) ? (wsum / (double)m) : 1.0;
    if (!(delta > 0.0)) delta = (wmax > 0.0) ? wmax : 1.0;

    int *lcount = (int*)calloc((size_t)n, sizeof(int));
    for (int u = 0; u < n; u++) {
        int c = 0;
        for (int e = off[u]; e < off[u + 1]; e++) if (ew[e] <= delta) c++;
        lcount[u] = c;
    }
    for (int u = 0; u < n; u++) {
        int lo = off[u], hi = off[u + 1] - 1;
        while (lo < hi) {
            while (lo < hi && ew[lo] <= delta) lo++;
            while (lo < hi && ew[hi] > delta) hi--;
            if (lo < hi) {
                int td = edst[lo]; edst[lo] = edst[hi]; edst[hi] = td;
                double tw = ew[lo]; ew[lo] = ew[hi]; ew[hi] = tw;
            }
        }
    }
    int *lend = (int*)malloc((size_t)n * sizeof(int));
    for (int u = 0; u < n; u++) lend[u] = off[u] + lcount[u];
    free(lcount);

    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    if (n > 0) dist_out[source] = 0.0;

    int *bucket_of = (int*)malloc((size_t)n * sizeof(int));
    for (int i = 0; i < n; i++) bucket_of[i] = -1;
    char *done = (char*)calloc((size_t)n, 1);
    char *inR = (char*)calloc((size_t)n, 1);
    int *Rlist = (int*)malloc((size_t)(n > 0 ? n : 1) * sizeof(int));

    Buckets B; B.b = NULL; B.cap = 0;
    int maxb = 0;
    if (n > 0) {
        bucket_add(&B, 0, source);
        bucket_of[source] = 0;
    }

    for (int i = 0; i <= maxb; i++) {
        if (i >= B.cap || B.b[i].cnt == 0) continue;
        int Rn = 0;
        int cursor = 0;
        for (;;) {
            if (cursor >= B.b[i].cnt) break;
            int v = B.b[i].arr[cursor++];
            if (done[v] || bucket_of[v] != i) continue;
            bucket_of[v] = -2;
            if (!inR[v]) { inR[v] = 1; Rlist[Rn++] = v; }
            double dv = dist_out[v];
            for (int e = off[v]; e < lend[v]; e++) {
                int x = edst[e];
                double nd = dv + ew[e];
                if (nd < dist_out[x]) {
                    dist_out[x] = nd;
                    int nb = (int)(nd / delta);
                    if (nb < 0) nb = 0;
                    if (nb > maxb) maxb = nb;
                    bucket_of[x] = nb;
                    bucket_add(&B, nb, x);
                }
            }
        }
        for (int k = 0; k < Rn; k++) {
            int v = Rlist[k];
            done[v] = 1;
            inR[v] = 0;
            double dv = dist_out[v];
            for (int e = lend[v]; e < off[v + 1]; e++) {
                int x = edst[e];
                double nd = dv + ew[e];
                if (nd < dist_out[x]) {
                    dist_out[x] = nd;
                    int nb = (int)(nd / delta);
                    if (nb < 0) nb = 0;
                    if (nb > maxb) maxb = nb;
                    bucket_of[x] = nb;
                    bucket_add(&B, nb, x);
                }
            }
        }
    }

    for (int i = 0; i < B.cap; i++) free(B.b[i].arr);
    free(B.b);
    free(deg); free(off); free(edst); free(ew); free(lend);
    free(bucket_of); free(done); free(inR); free(Rlist);
}
