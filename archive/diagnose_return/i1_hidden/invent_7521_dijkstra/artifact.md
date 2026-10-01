# MAPPING

**SEED 1 — "The nightingales always land first on the stone carrying the smallest owed letter."**

| world object | problem object |
|---|---|
| garden stone | vertex `0..n-1` |
| chalked letter on a stone | tentative distance `key[v]` (stored as IEEE bits) |
| "unreadable, far number" | `INFINITY` |
| **nightingale*s*** (plural flock) | the min-selection mechanism — a *batch* of extractions per flight, not one pop |
| "drop **first** onto whichever carries the smallest owed sum" | a *predicate on a value*: every stone whose letter attains the current minimum **level** |
| circling the unlocked stones | sweep over the not-yet-settled queue |
| singing the place-name down | the node id returned to the expander |
| locked stone | settled vertex |
| "the bird changes its mind about which stone to visit next" | a queued entry's priority changes *before* it is settled |

**SEED 2 — "Threads cast from the tower price each road by what its thief is owed, one direction at a time."**

| world object | problem object |
|---|---|
| casting tower at my own house | the vertex currently being scanned (`source` at first) |
| thread flung | one relaxation attempt |
| "each house the roads actually touch" | out-neighbours, i.e. CSR row `rs[u]..rs[u+1]` |
| thief's toll | `weight[i] >= 0` |
| "directional, a road paid one way is not paid the other" | directed edge `src[i]->dst[i]`; only out-adjacency is built |
| garden-stone "nothing owed" under my feet | `key[source] = 0` |

**SEED 3 — "A locked garden letter is never rebuilt again, and threads that reach nowhere are dropped and thrown away."**

| world object | problem object |
|---|---|
| locked letter never rebuilt | settled node never relaxed again (valid because no toll is negative) |
| thread into bramble / over the tide / round the wall to nowhere new | a **stale queue entry** (its node already holds a better letter) |
| "let it drop and throw it away" | lazy deletion: `if (entry.key > key[u]) continue;` |
| stones chalked with unreadable debt forever | unreachable nodes stay `INFINITY` |

# CHOSEN SEED

**SEED 1.** Seeds 2 and 3 are *textbook* Dijkstra parts (CSR relaxation; lazy-deletion heap) and both *uphold* the assumption. Seed 1 is the only one that breaks it, and it breaks it by being read **literally** rather than loosely: the native says **nightingales**, plural, a flock going up together, and says they land on "whichever carries the smallest owed sum" — a test on a *number*, satisfiable by many stones at once — not "onto the one stone I name." Taken literally, one flight locks and expands a whole **band** of stones. The story confirms this is intended: "a knot of three threads, once cast toward a stone, can be **detached** if a cheaper letter arrives **before the nightingale locks it**, and **called back on**" — entries whose priority mutates while unsettled, i.e. stones are being cast *from* and *toward* while still tentative.

Literal translation: **Δ-stepping over a band/bucket queue** (bands = the flock's landing level), with within-band iteration to fixpoint ("detached and called back on"), lazy dropping of dead threads (SEED 3), and directed CSR casting (SEED 2).

# ASSUMPTION BROKEN

"Each place's distance must be finalized before its neighbours are explored."

Here only **band boundaries** carry finality. Every stone whose letter falls in `[k·Δ,(k+1)·Δ)` is cast from while still *tentative*; it may be re-chalked and re-cast several times before its band empties (bounded Bellman-Ford inside a band, Dijkstra across bands). Correctness survives because no toll is negative, so nothing inserted while band `k` is open can land below band `k`. What is bought: the min-extraction becomes O(1) array indexing instead of O(log n) sifting, the frontier is processed in cache-friendly batches, and batches large enough to be worth threads can be relaxed in parallel with a CAS-min on the letter.

# ARTIFACT

```c
/* Single-source shortest distances, weighted directed graph, weights >= 0.
   Literal translation of SEED 1: a flock lands on a whole BAND of smallest
   letters at once; stones are cast from while still tentative; dead threads
   are dropped; a locked letter is never rebuilt.
   Guarded fallback/bailout: lazy 4-ary heap ("one bird, one stone").
   gcc -O3 -march=native -fopenmp -lm                                       */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <limits.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* A garden-stone letter is the IEEE-754 bit pattern of its distance.  For
   non-negative doubles (and +INF) unsigned order == numeric order, so every
   "is this letter higher?" test is one integer compare and the final answer
   is one memcpy.  Keeping the working array as uint64_t also makes the
   atomic CAS-min strictly legal (no type punning of double*).            */
#define GS_INF 0x7FF0000000000000ULL

static inline double  L2D(uint64_t k){ double d; memcpy(&d,&k,sizeof d); return d; }
static inline uint64_t D2L(double d){ uint64_t k; memcpy(&k,&d,sizeof k); return k; }

static inline long long band_of(double d, double invD){
    double q = d*invD;
    if(!(q < 9.0e18)) return LLONG_MAX;      /* astronomically far / +inf */
    return (long long)q;
}

/* ---------------- safe path: lazy 4-ary heap ---------------- */
typedef struct { uint64_t k; int v; } HN;
typedef struct { HN *a; long long n, cap; } Heap;

static void hp_push(Heap *h, uint64_t k, int v){
    if(h->n == h->cap){
        long long c = h->cap ? h->cap*2 : 4096;
        HN *na = (HN*)realloc(h->a, (size_t)c*sizeof(HN));
        if(!na) return;
        h->a = na; h->cap = c;
    }
    HN *a = h->a; long long i = h->n++;
    while(i > 0){ long long p=(i-1)>>2; if(a[p].k <= k) break; a[i]=a[p]; i=p; }
    a[i].k = k; a[i].v = v;
}
static HN hp_pop(Heap *h){
    HN *a = h->a, top = a[0];
    long long nn = --h->n;
    if(nn > 0){
        HN last = a[nn]; long long i = 0;
        for(;;){
            long long c = (i<<2)+1; if(c >= nn) break;
            long long e = c+4; if(e > nn) e = nn;
            long long b = c; uint64_t bk = a[c].k;
            for(long long j=c+1;j<e;j++) if(a[j].k < bk){ bk=a[j].k; b=j; }
            if(bk >= last.k) break;
            a[i] = a[b]; i = b;
        }
        a[i] = last;
    }
    return top;
}
static void run_heap(const int *rs,const int *adj,const double *wt,
                     uint64_t *key, Heap *h){
    while(h->n > 0){
        HN t = hp_pop(h);
        int u = t.v;
        if(t.k > key[u]) continue;           /* thread into bramble: dropped */
        double du = L2D(t.k);
        int e = rs[u], ee = rs[u+1];
        for(; e < ee; e++){
            int v = adj[e];
            uint64_t nk = D2L(du + wt[e]);
            if(nk < key[v]){ key[v] = nk; hp_push(h, nk, v); }
        }
    }
}

/* ---------------- bands (bags of stones) ---------------- */
typedef struct { int *a; int n, cap; } Bag;
static inline void bag_push(Bag *b, int x){
    if(b->n == b->cap){
        int c = b->cap ? b->cap*2 : 16;
        int *na = (int*)realloc(b->a, (size_t)c*sizeof(int));
        if(!na) return;
        b->a = na; b->cap = c;
    }
    b->a[b->n++] = x;
}
typedef struct { long long ix; int v; } PEnt;
typedef struct { PEnt *a; long n, cap; } PBag;
static inline void pbag_push(PBag *b, long long ix, int v){
    if(b->n == b->cap){
        long c = b->cap ? b->cap*2 : 1024;
        PEnt *na = (PEnt*)realloc(b->a, (size_t)c*sizeof(PEnt));
        if(!na) return;
        b->a = na; b->cap = c;
    }
    b->a[b->n].ix = ix; b->a[b->n].v = v; b->n++;
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out)
{
    if(n <= 0 || !dist_out) return;
    if(m <= 0 || (unsigned)source >= (unsigned)n){
        for(int i=0;i<n;i++) dist_out[i] = INFINITY;
        if((unsigned)source < (unsigned)n) dist_out[source] = 0.0;
        return;
    }

    /* ---- SEED 2: roads, directional, priced by their thief ---- */
    int    *rs  = (int*)   calloc((size_t)n+1, sizeof(int));
    int    *adj = (int*)   malloc((size_t)m*sizeof(int));
    double *wt  = (double*)malloc((size_t)m*sizeof(double));
    int    *cur = (int*)   malloc((size_t)n*sizeof(int));
    uint64_t *key = (uint64_t*)malloc((size_t)n*sizeof(uint64_t));
    uint64_t *sk  = (uint64_t*)malloc((size_t)n*sizeof(uint64_t));
    if(!rs||!adj||!wt||!cur||!key||!sk){
        for(int i=0;i<n;i++) dist_out[i] = INFINITY;
        dist_out[source] = 0.0;
        free(rs);free(adj);free(wt);free(cur);free(key);free(sk);
        return;
    }

    double wmax = 0.0, wsum = 0.0; int bad = 0;
    for(int i=0;i<m;i++){
        int s = src[i], t = dst[i];
        if((unsigned)s >= (unsigned)n || (unsigned)t >= (unsigned)n) continue;
        double x = weight[i];
        if(!(x >= 0.0) || !(x <= 1.7976931348623157e308)) bad = 1; /* neg/NaN/inf */
        if(x > wmax) wmax = x;
        wsum += x;
        rs[s+1]++;
    }
    for(int i=0;i<n;i++) rs[i+1] += rs[i];
    int me = rs[n];
    memcpy(cur, rs, (size_t)n*sizeof(int));
    for(int i=0;i<m;i++){
        int s = src[i], t = dst[i];
        if((unsigned)s >= (unsigned)n || (unsigned)t >= (unsigned)n) continue;
        int p = cur[s]++; adj[p] = t; wt[p] = weight[i];
    }
    free(cur);

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(n > 200000)
#endif
    for(int i=0;i<n;i++) key[i] = GS_INF;      /* every stone: unreadable debt */
    memset(sk, 0xFF, (size_t)n*sizeof(uint64_t));
    key[source] = 0;                            /* "nothing owed" underfoot   */

    /* ---- band width, and the guards that decide which mechanism runs ---- */
    double delta = (me > 0) ? wsum/(double)me : 0.0;   /* mean toll */
    int use_bands = 1;
    if(bad)                                  use_bands = 0; /* weird weights */
    if(wmax <= 0.0)                          use_bands = 0; /* all tolls 0   */
    if(!(delta > 0.0) || !(delta < 1e300))   use_bands = 0;
    if(me < 2048 || n < 64)                  use_bands = 0; /* too small     */
    if(use_bands && wmax/delta > 1.0e6)      use_bands = 0; /* toll spread    */

    Heap h = {NULL,0,0};

    if(!use_bands){
        hp_push(&h, 0, source);
        run_heap(rs, adj, wt, key, &h);
    } else {
        int B = 1<<14;
        if(n < B){ B = 256; while(B < n) B <<= 1; }
        Bag *ring = (Bag*)calloc((size_t)B, sizeof(Bag));
        if(!ring){
            hp_push(&h, 0, source);
            run_heap(rs, adj, wt, key, &h);
        } else {
            Bag ovf = {NULL,0,0}, front = {NULL,0,0};
            double invD = 1.0/delta;
            long long base = 0;
            int slot = 0, bail = 0;
            long long work = 0, limit = 8LL*((long long)me + (long long)n) + 4096;
            double avgdeg = (double)me/(double)n;
            int T = 1;
            PBag *pb = NULL;
#ifdef _OPENMP
            T = omp_get_max_threads();
            if(T < 1) T = 1; if(T > 64) T = 64;
            if(T > 1){ pb = (PBag*)calloc((size_t)T, sizeof(PBag)); if(!pb) T = 1; }
#endif
            bag_push(&ring[0], source);

            for(;;){
                if(slot >= B){                       /* window exhausted: refill */
                    if(ovf.n == 0) break;
                    long long mn = LLONG_MAX;
                    for(int j=0;j<ovf.n;j++){
                        long long ix = band_of(L2D(key[ovf.a[j]]), invD);
                        if(ix < mn) mn = ix;
                    }
                    base = mn; slot = 0;
                    int keep = 0;
                    for(int j=0;j<ovf.n;j++){
                        int u = ovf.a[j];
                        long long ix = band_of(L2D(key[u]), invD) - base;
                        if(ix < 0) ix = 0;
                        if(ix < (long long)B) bag_push(&ring[(int)ix], u);
                        else ovf.a[keep++] = u;
                    }
                    ovf.n = keep;
                    continue;
                }
                if(ring[slot].n == 0){ slot++; continue; }

                long long gslot = base + slot;
                /* the flock lands on this whole band and keeps landing on it
                   until nothing in it moves: stones here are cast from while
                   still tentative, detached and called back on.            */
                while(ring[slot].n > 0){
                    { Bag tmp = ring[slot]; ring[slot] = front; front = tmp; }
                    ring[slot].n = 0;
                    int  fn = front.n;
                    int *fa = front.a;
                    int par = 0;
#ifdef _OPENMP
                    if(T > 1 && (double)fn*avgdeg >= 30000.0) par = 1;
#endif
                    if(!par){
                        for(int t=0;t<fn;t++){
                            int u = fa[t];
                            uint64_t ku = key[u];
                            if(band_of(L2D(ku), invD) != gslot) continue;  /* dropped */
                            if(sk[u] <= ku) continue;                      /* dropped */
                            sk[u] = ku;
                            double du = L2D(ku);
                            int e = rs[u], ee = rs[u+1];
                            work += ee - e;
                            for(; e < ee; e++){
                                if(e+4 < ee) __builtin_prefetch(&key[adj[e+4]], 1, 1);
                                int v = adj[e];
                                double nd = du + wt[e];
                                uint64_t nk = D2L(nd);
                                if(nk < key[v]){
                                    key[v] = nk;
                                    long long ix = band_of(nd, invD) - base;
                                    if(ix < 0) ix = 0;
                                    if(ix < (long long)B) bag_push(&ring[(int)ix], v);
                                    else bag_push(&ovf, v);
                                }
                            }
                        }
                    }
#ifdef _OPENMP
                    else {
                        long long wl = 0;
                        #pragma omp parallel num_threads(T) reduction(+:wl)
                        {
                            int tid = omp_get_thread_num();
                            PBag *b = &pb[tid]; b->n = 0;
                            #pragma omp for schedule(dynamic,64)
                            for(int t=0;t<fn;t++){
                                int u = fa[t];
                                uint64_t ku = __atomic_load_n(&key[u], __ATOMIC_RELAXED);
                                if(band_of(L2D(ku), invD) != gslot) continue;
                                uint64_t prev = __atomic_exchange_n(&sk[u], ku,
                                                                   __ATOMIC_RELAXED);
                                if(prev <= ku) continue;
                                double du = L2D(ku);
                                int e = rs[u], ee = rs[u+1];
                                wl += ee - e;
                                for(; e < ee; e++){
                                    int v = adj[e];
                                    double nd = du + wt[e];
                                    uint64_t nk = D2L(nd);
                                    uint64_t old = __atomic_load_n(&key[v],
                                                                   __ATOMIC_RELAXED);
                                    while(nk < old){
                                        if(__atomic_compare_exchange_n(&key[v], &old, nk,
                                                0, __ATOMIC_RELAXED, __ATOMIC_RELAXED)){
                                            pbag_push(b, band_of(nd, invD), v);
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                        work += wl;
                        for(int q=0;q<T;q++){
                            PBag *b = &pb[q];
                            for(long j=0;j<b->n;j++){
                                long long ix = b->a[j].ix - base;
                                int v = b->a[j].v;
                                if(ix < 0) ix = 0;
                                if(ix < (long long)B) bag_push(&ring[(int)ix], v);
                                else bag_push(&ovf, v);
                            }
                            b->n = 0;
                        }
                    }
#endif
                    if(work > limit){ bail = 1; break; }
                }
                if(bail) break;
                slot++;
            }

            if(bail){
                /* Band width was a bad bet for this graph: every stone whose
                   letter can still fall is queued, so hand the whole queue to
                   the one-bird-one-stone heap and finish there.  Exact.     */
                for(int s2=slot; s2<B; s2++){
                    for(int j=0;j<ring[s2].n;j++){
                        int u = ring[s2].a[j];
                        hp_push(&h, key[u], u);
                    }
                    ring[s2].n = 0;
                }
                for(int j=0;j<ovf.n;j++){
                    int u = ovf.a[j];
                    hp_push(&h, key[u], u);
                }
                ovf.n = 0;
                run_heap(rs, adj, wt, key, &h);
            }

            for(int s2=0;s2<B;s2++) free(ring[s2].a);
            free(ring); free(ovf.a); free(front.a);
            if(pb){ for(int q=0;q<T;q++) free(pb[q].a); free(pb); }
        }
    }

    memcpy(dist_out, key, (size_t)n*sizeof(double));   /* letters -> answers */

    free(h.a); free(rs); free(adj); free(wt); free(key); free(sk);
}
```

# PREDICTION

PREDICTION: speedup_vs_heap = 2.4

(Reasoning behind the number, stated before any measurement: O(1) band indexing replaces O(log n) sifting for roughly `n` extractions and ~`m` insertions → I expect 1.6–2.2× on a sequential comparison against a CSR binary heap, mostly from branch-free bag pushes and sequential-frontier locality; the parallel band relaxation only fires on frontiers with ≳30 000 incident edges, which I expect to happen on random/mesh-like instances and essentially never on road-like ones, contributing the remaining factor on average. Redundant re-scans inside a band cost me back an estimated 10–40 % of edge work.)

# MEASUREMENT

**Not obtained.** This session was launched with no tools attached — `dijkstra_bench`, `dijkstra_contract` and the symbolic/unconventional tools were all listed as available but none are callable here, so I ran zero measurements and performed zero measured improvement rounds. I am reporting that plainly rather than inventing numbers: **the prediction above is unvalidated, and the three design revisions below were reasoned, not measured.**

What I did instead of measuring, and what each revision cost in confidence:

1. **Band queue with `in_queue` de-duplication (rejected on paper).** A single queue entry per stone needs the entry to *move* when the letter falls into an earlier band; a stale entry parked in a later band would get scanned after intervening bands closed, which is a wrong-answer bug, not a slowdown. Replaced by duplicate entries plus the SEED-3 drop test (`band_of(key[u]) != gslot`) and the `sk[]` already-scanned-at-this-value test.
2. **Sliding ring modulo B (rejected).** Replaced with non-sliding windows plus an overflow bag re-partitioned when a window empties — two linear passes per refill, no modular arithmetic in the hot loop, and refills are rare when `B·Δ` covers the distance span.
3. **Atomic min on `double*` via pointer punning (rejected).** Replaced by keeping the letters as `uint64_t` bit patterns throughout, which makes the CAS strictly legal, turns every comparison into an integer compare, and makes the final conversion a single `memcpy`.

Things a real run must check, in priority order: (a) exactness against the reference on a graph with many equal-length paths; (b) whether `Δ = mean toll` is anywhere near optimal, or whether a small constant multiple is better; (c) whether the `work > 8(m+n)` bailout ever triggers on the bench graphs, and whether the post-bailout answer still matches; (d) whether the parallel band path ever fires at all.

# VERDICT

The literal reading of SEED 1 — a *flock*, landing on *whichever stone carries the smallest sum*, a band rather than a single stone — is Δ-stepping, and it does genuinely break the target assumption: stones inside the open band are cast from while their letters are still tentative and can be re-chalked and re-cast several times before the band closes. Only band boundaries inherit the "no thief charges a negative toll" finality argument. SEEDs 2 and 3 survive intact inside it, exactly as the native tells it: casting is per-directed-road out of a CSR row priced by its own thief, and threads that reach nowhere are dropped rather than repaired.

Named conditions where this mechanism is slow, and the guard shipped for each:

- **Band width badly matched to the graph** (heavy-tailed tolls, or paths that chain many light edges inside one band) degrades toward Bellman-Ford. → Runtime guard: edge-scan counter with an `8(m+n)` ceiling, then a **correctness-preserving bailout** that dumps the entire queue into the lazy 4-ary heap and finishes as ordinary Dijkstra. Dirty-stone-implies-queued is an invariant of the band loop, which is what makes the bailout exact rather than approximate.
- **Small graphs**, where `O(B)` band setup and per-round bookkeeping dominate. → Static guard `me < 2048 || n < 64` → heap.
- **Extreme toll spread** (`wmax/Δ > 10⁶`), which makes the window span useless and the overflow bag hot. → Static guard → heap.
- **Degenerate or hostile weights** (all-zero, negative, NaN, `+inf`). → Static guard → heap, which handles them without a special case.
- **Threads on a small frontier**, where fork/join beats the work done. → The parallel band relaxation only runs when the frontier's estimated incident edges reach 30 000 and more than one thread exists; otherwise the identical serial loop runs.

Honest weak points I am *not* claiming to have solved: `Δ = mean toll` is a heuristic with no tuning behind it; the overflow refill is `O(|overflow|)` per window and could in principle be re-scanned many times on a graph with a very long distance span; and the whole artifact is unmeasured, so the 2.4× is a prediction with no evidence under it. If the harness reports below 1.0×, the most likely culprit in order is Δ, then the overflow refill, then the prefetch in the serial inner loop.