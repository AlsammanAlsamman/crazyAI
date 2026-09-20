STORY

I crouch at the edge of the chalk grid where the traveler's foot rests, and before I touch anything I do what I always do first: I sort the roads. Every road scratched between squares is a straight white line with a length only I can feel by running a thread along it, and I gather all of them by the square they leave from, tying each bundle at that square's foot so that later, when I stand there, I know at a glance which threads lead onward and how far. I keep two rows of pebbles, one dark, for squares I have already fixed the true distance of and will never touch again, and one pale, for squares I have merely guessed at — and my pale pebbles I keep heaped not anyhow but by size, smallest always rolling to my palm first, so I never waste a step chasing a long road when a short one still waits unclaimed; I do not use two pebbles side by side when I could stack four together in one small cairn, because a squat heap of four pebbles is quicker for my hand to search through than a tall thin one, especially when the grid sprawls wide. I begin at the traveler's own square with a distance of nothing, drop it into the heap, and then I only ever do one thing: I lift the smallest pale pebble, and if the square under it is already dark I throw the pebble away and reach for the next, for that measure was already beaten and is worthless now — cluttered pebbles I discard without mercy, the same way one sweeps chalk dust off a finished sum. If it is not yet dark, I blacken it, walk its bundle of roads, and for each road I add its length to the distance just fixed; where that sum beats what any pale pebble already claims for the far square, I drop a fresh pale pebble there and forget the old promise entirely. Roads that lead to squares already dark I do not even weigh — they can teach me nothing more. When the grid is a small dooryard scratched by one child's hand I do all of this alone, thread and pebbles, start to finish, for calling in more hands over a handful of squares only trips people over each other; only when the grid sprawls into a whole courtyard's worth of squares and roads do I let others help me gather and bundle the roads at the very start, before the counting itself begins, since the counting itself must still be done one square at a time, in order, by one set of hands. What is left when the pale heap empties is the truest distance to every square the traveler can ever reach, and silence, forever, for the ones the roads never touch.

ARTIFACT
```c
#include <stdlib.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define HEAP_D 4

static inline void heap_sift_up(double *hd, int *hn, int i) {
    while (i > 0) {
        int p = (i - 1) / HEAP_D;
        if (hd[p] <= hd[i]) break;
        double td = hd[p]; hd[p] = hd[i]; hd[i] = td;
        int tn = hn[p]; hn[p] = hn[i]; hn[i] = tn;
        i = p;
    }
}

static inline void heap_sift_down(double *hd, int *hn, int hs, int i) {
    for (;;) {
        int first = HEAP_D * i + 1;
        if (first >= hs) break;
        int last = first + HEAP_D;
        if (last > hs) last = hs;
        int s = i;
        double sd = hd[i];
        for (int c = first; c < last; c++) {
            if (hd[c] < sd) { sd = hd[c]; s = c; }
        }
        if (s == i) break;
        double td = hd[i]; hd[i] = hd[s]; hd[s] = td;
        int tn = hn[i]; hn[i] = hn[s]; hn[s] = tn;
        i = s;
    }
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight, int source, double *dist_out)
{
    size_t msz = (size_t)(m > 0 ? m : 1);

    int *off    = (int*)malloc((size_t)(n + 1) * sizeof(int));
    int *deg    = (int*)calloc((size_t)n, sizeof(int));
    int *edst   = (int*)malloc(msz * sizeof(int));
    double *ew  = (double*)malloc(msz * sizeof(double));
    int *cursor = (int*)malloc((size_t)(n + 1) * sizeof(int));

    int use_parallel = (m > 200000) && (n > 1000);

#ifdef _OPENMP
    if (use_parallel) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < m; i++) {
            #pragma omp atomic
            deg[src[i]]++;
        }
    } else
#endif
    {
        for (int i = 0; i < m; i++) deg[src[i]]++;
    }

    off[0] = 0;
    for (int i = 0; i < n; i++) off[i + 1] = off[i] + deg[i];
    for (int i = 0; i <= n; i++) cursor[i] = off[i];

#ifdef _OPENMP
    if (use_parallel) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < m; i++) {
            int u = src[i];
            int pos;
            #pragma omp atomic capture
            { pos = cursor[u]; cursor[u]++; }
            edst[pos] = dst[i];
            ew[pos] = weight[i];
        }
    } else
#endif
    {
        for (int i = 0; i < m; i++) {
            int u = src[i];
            int pos = cursor[u]++;
            edst[pos] = dst[i];
            ew[pos] = weight[i];
        }
    }

    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;
    if (n > 0) dist_out[source] = 0.0;

    char *done = (char*)calloc((size_t)n, 1);
    int heap_cap = m + 2;
    double *hd = (double*)malloc((size_t)heap_cap * sizeof(double));
    int *hn    = (int*)malloc((size_t)heap_cap * sizeof(int));
    int hs = 0;

    if (n > 0) {
        hd[hs] = 0.0; hn[hs] = source;
        heap_sift_up(hd, hn, hs);
        hs++;
    }

    while (hs > 0) {
        double d0 = hd[0];
        int u = hn[0];
        hs--;
        hd[0] = hd[hs]; hn[0] = hn[hs];
        heap_sift_down(hd, hn, hs, 0);

        if (done[u]) continue;
        done[u] = 1;

        int e_end = off[u + 1];
        for (int e = off[u]; e < e_end; e++) {
            int v = edst[e];
            if (done[v]) continue;
            double nd = d0 + ew[e];
            if (nd < dist_out[v]) {
                dist_out[v] = nd;
                hd[hs] = nd; hn[hs] = v;
                heap_sift_up(hd, hn, hs);
                hs++;
            }
        }
    }

    free(off); free(deg); free(edst); free(ew); free(cursor);
    free(done); free(hd); free(hn);
}
```

PREDICTION: 1.3