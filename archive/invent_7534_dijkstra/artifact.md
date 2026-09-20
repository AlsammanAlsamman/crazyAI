STORY

I light no lamp for this; I go to the cellar stair, where the point a few centimetres wide already holds every place on earth at once, and I set the traveler's own doorstep at the center of it. Every road leaving that doorstep becomes a thread of a certain length running out into the point, and every place a thread touches becomes a knot. I keep two piles: knots already fixed — given their true distance and never touched again — and a heap of embers, one ember per knot still guessing, each ember glowing exactly as bright as its current best guess is short, so the dimmest guess always floats to the top. I pull the dimmest ember — the knot with the shortest guessed distance from the doorstep. If that knot is already fixed I let the ember die as ash and reach for the next, since a second guess for a settled place is worthless. Otherwise I declare its distance final, walk every thread leaving that knot, and for each see whether doorstep-distance-to-here plus this thread's length beats whatever the far knot already holds; if it does, I write the new guess into that knot and drop a fresh ember for it, never troubling to hunt down and snuff the old dim one, because the pile of fixed knots protects me from ever trusting a stale ember twice. I do this — pull, fix, walk, drop — until the heap of embers is empty, and any knot the fire never reached I mark with the far horizon, unreachable, and leave be. Before the first ember is ever lit, though, I walk the raw pile of roads once and sort it into shelves, one shelf per starting knot holding its own roads side by side, so the fire, once it reaches a knot, never hunts the whole tangle for its roads — only reaches its own shelf. I also widen each ember-pile's branching from two children to four, so the pile stays shallow and my hand needn't climb as far to keep it in order. The shelf-sorting is what I share among many hands when the pile of roads is truly vast, since counting and placing threads needn't happen in any set order — but the fire itself, the pulling and fixing, only one hand may ever hold, for each ember's fate depends on the one pulled just before it; setting two hands on so sequential a task, or on too small a tangle, wastes more breath in the calling than it would ever save, so I call the many hands only past a certain size, and otherwise work alone, swift and plain.

ARTIFACT
```c
#include <stdlib.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef struct { double d; int u; } HeapItem;

static void hpush(HeapItem *h, int *hs, double d, int u) {
    int i = (*hs)++; h[i].d = d; h[i].u = u;
    while (i > 0) {
        int p = (i - 1) / 4;
        if (h[p].d <= h[i].d) break;
        HeapItem t = h[p]; h[p] = h[i]; h[i] = t; i = p;
    }
}

static HeapItem hpop(HeapItem *h, int *hs) {
    HeapItem top = h[0]; (*hs)--; h[0] = h[*hs];
    int i = 0;
    while (1) {
        int c0 = 4 * i + 1, s = i;
        for (int k = 0; k < 4; k++) { int c = c0 + k; if (c < *hs && h[c].d < h[s].d) s = c; }
        if (s == i) break;
        HeapItem t = h[s]; h[s] = h[i]; h[i] = t; i = s;
    }
    return top;
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight, int source, double *dist_out) {
    const int PAR_THRESHOLD = 200000;

    int *deg = calloc((size_t)n, sizeof(int));
    int *off = malloc((size_t)(n + 1) * sizeof(int));
    int *edst = malloc((size_t)(m > 0 ? m : 1) * sizeof(int));
    double *ew = malloc((size_t)(m > 0 ? m : 1) * sizeof(double));

    if (m >= PAR_THRESHOLD) {
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < m; i++) {
            #pragma omp atomic update
            deg[src[i]]++;
        }
#else
        for (int i = 0; i < m; i++) deg[src[i]]++;
#endif
    } else {
        for (int i = 0; i < m; i++) deg[src[i]]++;
    }

    off[0] = 0;
    for (int i = 0; i < n; i++) off[i + 1] = off[i] + deg[i];

    int *cursor = malloc((size_t)n * sizeof(int));
    for (int i = 0; i < n; i++) cursor[i] = off[i];

    if (m >= PAR_THRESHOLD) {
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < m; i++) {
            int u = src[i];
            int pos;
            #pragma omp atomic capture
            { pos = cursor[u]; cursor[u]++; }
            edst[pos] = dst[i];
            ew[pos] = weight[i];
        }
#else
        for (int i = 0; i < m; i++) {
            int u = src[i]; int pos = cursor[u]++;
            edst[pos] = dst[i]; ew[pos] = weight[i];
        }
#endif
    } else {
        for (int i = 0; i < m; i++) {
            int u = src[i]; int pos = cursor[u]++;
            edst[pos] = dst[i]; ew[pos] = weight[i];
        }
    }

    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    dist_out[source] = 0.0;

    char *done = calloc((size_t)n, 1);
    HeapItem *heap = malloc((size_t)(m + 2) * sizeof(HeapItem));
    int hs = 0;
    hpush(heap, &hs, 0.0, source);

    while (hs > 0) {
        HeapItem top = hpop(heap, &hs);
        int u = top.u;
        if (done[u]) continue;
        done[u] = 1;
        double du = dist_out[u];
        for (int e = off[u]; e < off[u + 1]; e++) {
            int v = edst[e];
            double nd = du + ew[e];
            if (nd < dist_out[v]) { dist_out[v] = nd; hpush(heap, &hs, nd, v); }
        }
    }

    free(deg); free(off); free(edst); free(ew); free(cursor); free(done); free(heap);
}
```

PREDICTION: 1.15