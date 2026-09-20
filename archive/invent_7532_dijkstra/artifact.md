STORY

I go to the mantel where the map book lies open, its roads inked between places, each line carrying its own weight of travel — some short as a held breath, some long as a chant repeated past counting, and a few leading only to blank leaf, useless, going nowhere a foot could stand. I set a cold coal in the hearth-bowl for every place in the book — dark, unlit, worth nothing until proven otherwise — except the traveler's own hearth, which I light first and call zero, since a place needs no road to itself. I keep one small bank of embers apart from the rest, sorted so the faintest heat always sits where my hand falls first, and I never disturb that sorting by force — I only slide a new ember in and let it find its own level, the way ash settles. Then I wait, and take, each time, the dimmest lit ember in the bank — not the brightest, the traveler is not chasing fire, only the least cost of getting there — and once a coal is lifted and its place named certain, I bank it forever and never lift it again, no matter how many other embers still whisper its name. From that place I walk its roads outward: for each road, the coal at its far end is offered a new heat, the walker's own heat plus the road's weight, and only if that offer is smaller than what that coal already holds do I relight it brighter and drop a fresh ember bearing that number into the bank. Places the roads never reach keep their cold coal forever — I do not lie and give them a number, I leave them dark, which strangers elsewhere read as unreachable. This tending is a one-back labor: coals must be picked up and banked in strict order, one after another, or the fire lies about who was cheapest to reach — so I do not call other hands to it, no matter how many places the book holds; scattered hands would only smother each other's kindling. I throw away nothing but the used matches — the visited-mark, the emptied embers — once every coal that can be lit is lit and banked.

ARTIFACT
```c
#include <stdlib.h>
#include <math.h>

typedef struct { double d; int u; } HeapItem;
typedef struct { int v; double w; } Edge;

static void hpush(HeapItem *h, int *hs, double d, int u) {
    int i = (*hs)++; h[i].d = d; h[i].u = u;
    while (i > 0) {
        int p = (i - 1) / 2;
        if (h[p].d <= h[i].d) break;
        HeapItem t = h[p]; h[p] = h[i]; h[i] = t;
        i = p;
    }
}

static HeapItem hpop(HeapItem *h, int *hs) {
    HeapItem top = h[0];
    (*hs)--;
    h[0] = h[*hs];
    int i = 0;
    while (1) {
        int l = 2 * i + 1, r = 2 * i + 2, s = i;
        if (l < *hs && h[l].d < h[s].d) s = l;
        if (r < *hs && h[r].d < h[s].d) s = r;
        if (s == i) break;
        HeapItem t = h[s]; h[s] = h[i]; h[i] = t;
        i = s;
    }
    return top;
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight, int source, double *dist_out) {
    int *deg = calloc((size_t)n, sizeof(int));
    for (int i = 0; i < m; i++) deg[src[i]]++;

    int *off = malloc((size_t)(n + 1) * sizeof(int));
    off[0] = 0;
    for (int i = 0; i < n; i++) off[i + 1] = off[i] + deg[i];

    Edge *adj = malloc((size_t)(m > 0 ? m : 1) * sizeof(Edge));
    int *cursor = malloc((size_t)n * sizeof(int));
    for (int i = 0; i < n; i++) cursor[i] = off[i];
    for (int i = 0; i < m; i++) {
        int u = src[i];
        int pos = cursor[u]++;
        adj[pos].v = dst[i];
        adj[pos].w = weight[i];
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
        int start = off[u], end = off[u + 1];
        for (int e = start; e < end; e++) {
            int v = adj[e].v;
            double nd = du + adj[e].w;
            if (nd < dist_out[v]) {
                dist_out[v] = nd;
                hpush(heap, &hs, nd, v);
            }
        }
    }

    free(deg); free(off); free(adj); free(cursor); free(done); free(heap);
}
```

PREDICTION: 1.15