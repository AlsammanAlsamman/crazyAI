# MAPPING

**SEED 1 — "Nightingales sent down every road sing one note per stride and die at the first board, carving that board's lattice slot with their dying note."**

| World object | Problem object |
|---|---|
| the traveler's place | `source` |
| a place / a board | node id `0..n-1` |
| a road out of a place | directed edge `src[i]->dst[i]` |
| one bird per road, loosed **all at once** | the whole out-edge block of a node, relaxed in one contiguous CSR sweep |
| strides of road beneath the bird | `weight[i]` |
| hoarseness of the note at death | `dist[u] + w` |
| the bird dies at the **first** board | each edge is traversed exactly once, from its tail, ever |
| **rigid lattice of old memory, one fixed slot per place** | a flat `double key[n]` array — **not** a priority queue |
| carving the dying note into the slot | `key[v] = nd` |
| two notes at one board → keep the fainter | `if (nd < key[v])` |
| the louder rots in the desert; dead roads marked once, never flown again | settled node ⇒ `key[v] = NaN`; `nd < NaN` is false, so every road into a settled board is dead from that single value, at zero cost |

*Silent assumption broken:* **"a priority structure must be consulted before every relaxation."** The birds never touch a queue. There is no push. Tentative distances live only in fixed lattice slots.

**SEED 2 — "At each board a cow's-head woman milks a smaller cow to re-measure the note by way of the last settled board, recutting only when that milk is thinner."**

| World object | Problem object |
|---|---|
| the cow's head opened at a board | the relaxation site for node `v` |
| the **woman inside** milking a **smaller cow** | a nested, two-level measure: outer = the row-rim summary of the lattice region, inner = the slot itself |
| "the note as it would sound by way of the last board I settled, plus the stretch between" | `d_u + w(u,v)` |
| milk thinner than what's carved | `nd < key[v]` |
| scrape the slot clean and cut hers in | `key[v] = nd` (and, one level out, `bmin[v>>L] = min(bmin[v>>L], nd)`) |

*Silent assumption broken:* **"each place's distance must be finalized before its neighbors are explored."** A board's carving is provisional and gets re-cut from outside; only the *chosen* slot is frozen.

**SEED 3 — "The quietest unsettled note in the lattice is always chosen next, until no nightingale sings and unreached places stand blank as the desert."**

| World object | Problem object |
|---|---|
| reading the lattice for the quietest note | argmin over the lattice array |
| "the lattice" (a *grid*, with rows) | `key[]` laid out in rows of `B = 2^L` slots, each row carrying a rim-mark `bmin[b]` = faintest note in that row |
| never touching a settled board twice | `key[u] = NaN` forever |
| "what's thrown away stays thrown away" | no lazy duplicates, no re-insertion, no stale pops |
| blank as the pink-brown desert with no landmark | `INFINITY` |
| "when the last nightingale falls silent" | stop as soon as the global faintest note is `INFINITY` |
| reading the whole lattice in order | `dist_out[0..n-1]` |

*Silent assumption broken:* **"the next place to finalize is found by comparing against every remaining place."** The native reads *row rims first*, then one row — `O(√n)` slots touched, not `n`.

**None of the three seeds breaks "the whole graph must be explored to know any single distance."** Stated plainly: the native explores exactly the reachable component, same as Dijkstra. No seed gives a single-pair shortcut.

# CHOSEN SEED

**Seed 1**, extended by 2 and 3 (they describe the same machine: seed 1 is the lattice, seed 2 is the nested re-cut, seed 3 is the selection rule). It is the most literal — "rigid lattice of old memory that keeps one fixed slot for every place" is a flat array, said in so many words — and the most different from the known way, because the known way's entire engine is the binary heap and this seed has no heap in it at all.

# ASSUMPTION BROKEN

**"A priority structure must be consulted before every relaxation."** In the heap version, every successful relaxation costs a `hpush` — `O(log n)` pointer-chasing writes into a structure up to `m` entries long, plus a later stale pop. In the native's world a relaxation is *one conditional store into a fixed slot*. Selection is paid for separately and cheaply, by reading the lattice.

Two secondary assumptions fall with it: *"the next place to finalize is found by comparing against every remaining place"* (the lattice has rows with rim-marks, so selection costs `O(√n)` not `O(n)`), and *"a road can only be considered once its starting place is fully settled"* — inverted: a road into an already-settled board is **dead**, and the single `NaN` in that board's slot kills every one of them with no flag array and no branch.

The one thing the native's method is genuinely worse at is a vast desert: many places, almost no roads. The native says so himself ("the roads that lead nowhere at all… the pink-brown desert"). So the kernel measures the desert at runtime and, when the land is mostly empty, piles the still-singing birds into a **cairn with the faintest on top** (a 4-ary heap) instead of walking the lattice. Both regimes named in `known_way` are implemented; the choice is made from `n` and `m` before a single bird flies.

# ARTIFACT

Design decisions, in the four rounds I would have measured:

1. **Literal lattice.** Flat `key[n]`, full linear argmin, `NaN` as the dead-note sentinel (the only IEEE value `x` for which `nd < x` is false *and* which `vminpd(load, acc)` discards). This removes the `done[]` array and the second random access per relaxation entirely.
2. **The lattice is a lattice, not a line.** Rows of `B ≈ √(n/2)` slots, each with a rim-mark. Selection drops from `n` to `n/B + 2B ≈ 2.8√n` slots, i.e. `O(n^1.5)` total instead of `O(n²)`. Rim-marks stay *exact* (refreshed on settle, lowered on relax), so settle order is bit-identical to textbook Dijkstra.
3. **SIMD reading of the lattice.** AVX2 argmin with index blending (2 independent accumulator pairs to break the dependency chain) and a 4-accumulator min-only pass for the rim refresh; scalar fallback compiled in when AVX2 is absent. Interleaved 16-byte road records so one cache line delivers four whole roads. Software prefetch of the scattered slot the bird will land in, 8 roads ahead.
4. **Regime guard + the one honest use of many hands.** Runtime desert test picks lattice vs. cairn. The *only* threaded work is gathering the roads into bundles (CSR build) — which the metaphor itself licenses, because that happens before any bird flies and depends on no settling order. It is guarded by `m ≥ 2^21` **and** `T·n ≤ m`, and falls back to the serial gather on any failure. The settling loop is strictly serial: its unit of work is one row of lattice, far too small to thread, so I did not thread it.

```c
#include <stdlib.h>
#include <string.h>
#include <math.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

/* ----------------------------------------------------------------------
   place              -> node 0..n-1
   road               -> NRoad { stretch w, board it ends at v }
   note / hoarseness  -> dist[u] + w
   rigid lattice      -> key[], one fixed slot per place, in rows of B;
                         each row carries a rim-mark bmin[b] = faintest
                         note in that row  (the outer cow; the slot is the
                         smaller cow inside it)
   settling a board   -> carve dist_out[u], then kill the slot with a
                         dead-note (NaN).  nd < NaN is false, so every
                         road into a settled board is dead for good, at
                         zero cost and with no flag array.
   desert             -> INFINITY (never reached) ; also the sparse regime
   cairn of birds     -> 4-ary heap, used only when the land is mostly
                         desert and walking the lattice costs more than
                         the birds themselves do
   ---------------------------------------------------------------------- */

typedef struct { double w; int v; int pad; } NRoad;   /* 16 B */
typedef struct { double d; int u; int pad; } NBird;   /* 16 B */

/* faintest note in a stretch of lattice, and which slot holds it.
   dead slots hold NaN: every comparison against them is false, so they
   are invisible here and deaf to every later bird. */
static int lat_argmin(const double *a, int cnt, double *outmin)
{
    double best = INFINITY;
    long long bi = -1;
    int i = 0;
#if defined(__AVX2__)
    if (cnt >= 8) {
        __m256d vm0 = _mm256_set1_pd(INFINITY), vm1 = vm0;
        __m256i vi0 = _mm256_set1_epi64x(-1), vi1 = vi0;
        __m256i c0 = _mm256_setr_epi64x(0, 1, 2, 3);
        __m256i c1 = _mm256_setr_epi64x(4, 5, 6, 7);
        const __m256i st = _mm256_set1_epi64x(8);
        for (; i + 8 <= cnt; i += 8) {
            __m256d x0 = _mm256_loadu_pd(a + i);
            __m256d x1 = _mm256_loadu_pd(a + i + 4);
            __m256d k0 = _mm256_cmp_pd(x0, vm0, _CMP_LT_OQ);  /* NaN -> false */
            __m256d k1 = _mm256_cmp_pd(x1, vm1, _CMP_LT_OQ);
            vm0 = _mm256_blendv_pd(vm0, x0, k0);
            vm1 = _mm256_blendv_pd(vm1, x1, k1);
            vi0 = _mm256_castpd_si256(_mm256_blendv_pd(_mm256_castsi256_pd(vi0),
                                                       _mm256_castsi256_pd(c0), k0));
            vi1 = _mm256_castpd_si256(_mm256_blendv_pd(_mm256_castsi256_pd(vi1),
                                                       _mm256_castsi256_pd(c1), k1));
            c0 = _mm256_add_epi64(c0, st);
            c1 = _mm256_add_epi64(c1, st);
        }
        {
            double bv[8]; long long bx[8]; int k;
            _mm256_storeu_pd(bv, vm0);
            _mm256_storeu_pd(bv + 4, vm1);
            _mm256_storeu_si256((__m256i *)bx, vi0);
            _mm256_storeu_si256((__m256i *)(bx + 4), vi1);
            for (k = 0; k < 8; k++) if (bv[k] < best) { best = bv[k]; bi = bx[k]; }
        }
    }
#endif
    for (; i < cnt; i++) if (a[i] < best) { best = a[i]; bi = i; }
    *outmin = best;
    return (int)bi;
}

/* re-cut a row's rim-mark.  vminpd(load, acc) returns acc when load is
   NaN, so dead slots are skipped and acc is never poisoned. */
static double lat_min(const double *a, int cnt)
{
    double best = INFINITY;
    int i = 0;
#if defined(__AVX2__)
    if (cnt >= 16) {
        __m256d m0 = _mm256_set1_pd(INFINITY), m1 = m0, m2 = m0, m3 = m0;
        double t[4]; int k;
        for (; i + 16 <= cnt; i += 16) {
            m0 = _mm256_min_pd(_mm256_loadu_pd(a + i),      m0);
            m1 = _mm256_min_pd(_mm256_loadu_pd(a + i + 4),  m1);
            m2 = _mm256_min_pd(_mm256_loadu_pd(a + i + 8),  m2);
            m3 = _mm256_min_pd(_mm256_loadu_pd(a + i + 12), m3);
        }
        m0 = _mm256_min_pd(m0, m1);
        m2 = _mm256_min_pd(m2, m3);
        m0 = _mm256_min_pd(m0, m2);
        _mm256_storeu_pd(t, m0);
        for (k = 0; k < 4; k++) if (t[k] < best) best = t[k];
    }
#endif
    for (; i < cnt; i++) if (a[i] < best) best = a[i];
    return best;
}

/* gather the roads into one bundle per place (CSR), serially */
static void gather_ser(int n, int m, const int *src, const int *dst,
                       const double *w, int *off, NRoad *E)
{
    int i, u, s = 0;
    memset(off, 0, (size_t)(n + 1) * sizeof(int));
    for (i = 0; i < m; i++) off[src[i]]++;
    for (u = 0; u < n; u++) { int t = off[u]; off[u] = s; s += t; }
    off[n] = s;
    for (i = 0; i < m; i++) { int a = src[i]; int p = off[a]++;
                              E[p].v = dst[i]; E[p].w = w[i]; }
    for (u = n; u > 0; u--) off[u] = off[u - 1];
    off[0] = 0;
}

/* the same gathering with many hands.  Legitimate to parallelise because
   it happens before any bird flies and depends on no settling order.
   Produces a byte-identical CSR to gather_ser.  Returns 0 if declined. */
static int gather_par(int n, int m, const int *src, const int *dst,
                      const double *w, int *off, NRoad *E)
{
#ifdef _OPENMP
    int T, u;
    long long s;
    int *cnt;
    if (n <= 0 || m < (1 << 21)) return 0;                 /* size guard */
    T = omp_get_max_threads();
    if (T > 16) T = 16;
    while (T >= 2 && (long long)T * (long long)n > (long long)m) T--;
    if (T < 2) return 0;                                   /* memory guard */
    cnt = (int *)calloc((size_t)T * (size_t)n, sizeof(int));
    if (!cnt) return 0;                                    /* fallback */

    #pragma omp parallel num_threads(T)
    {
        int tt = omp_get_thread_num();
        int *c = cnt + (size_t)tt * (size_t)n;
        int lo = (int)(((long long)m * tt) / T);
        int hi = (int)(((long long)m * (tt + 1)) / T);
        int k;
        for (k = lo; k < hi; k++) c[src[k]]++;
    }
    #pragma omp parallel for num_threads(T) schedule(static)
    for (u = 0; u < n; u++) {
        int sum = 0, tt;
        for (tt = 0; tt < T; tt++) sum += cnt[(size_t)tt * (size_t)n + u];
        off[u] = sum;
    }
    s = 0;
    for (u = 0; u < n; u++) { int c0 = off[u]; off[u] = (int)s; s += c0; }
    off[n] = (int)s;
    #pragma omp parallel for num_threads(T) schedule(static)
    for (u = 0; u < n; u++) {
        int run = off[u], tt;
        for (tt = 0; tt < T; tt++) {
            size_t ix = (size_t)tt * (size_t)n + (size_t)u;
            int c0 = cnt[ix]; cnt[ix] = run; run += c0;
        }
    }
    #pragma omp parallel num_threads(T)
    {
        int tt = omp_get_thread_num();
        int *c = cnt + (size_t)tt * (size_t)n;
        int lo = (int)(((long long)m * tt) / T);
        int hi = (int)(((long long)m * (tt + 1)) / T);
        int k;
        for (k = lo; k < hi; k++) { int a = src[k]; int p = c[a]++;
                                    E[p].v = dst[k]; E[p].w = w[k]; }
    }
    free(cnt);
    return 1;
#else
    (void)n; (void)m; (void)src; (void)dst; (void)w; (void)off; (void)E;
    return 0;
#endif
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out)
{
    int *off; NRoad *E;
    int i;
    double dn, dm, lg, pushes;
    int use_lattice;

    if (n <= 0) return;
    for (i = 0; i < n; i++) dist_out[i] = INFINITY;
    if (source < 0 || source >= n) return;
    if (m < 0) m = 0;

    off = (int *)malloc((size_t)(n + 1) * sizeof(int));
    E   = (NRoad *)malloc(((size_t)m + 1) * sizeof(NRoad));
    if (!off || !E) { free(off); free(E); return; }

    if (!gather_par(n, m, src, dst, weight, off, E))
        gather_ser(n, m, src, dst, weight, off, E);

    /* --- how much desert is this land? --------------------------------
       lattice walk  ~ 0.7 * n * sqrt(n) cycles of SIMD reading
       cairn of birds~ 8 * (expected pushes) * log2(n)   cycles of pointer
       chasing.  Shared edge-sweep and gathering costs cancel.          */
    dn = (double)n; dm = (double)m;
    lg = log2(dn + 2.0);
    pushes = dn * (1.0 + log2(1.0 + dm / (dn > 0.0 ? dn : 1.0)));
    if (pushes > dm) pushes = dm;
    use_lattice = (n <= 2048) || (0.7 * dn * sqrt(dn) < 8.0 * pushes * lg);

    if (use_lattice) {
        /* ---------- the rigid lattice ---------- */
        int L = 3, B, nb, it;
        size_t padded;
        double *restrict key;
        double *restrict bmin;
        const NRoad *restrict R = E;
        const int   *restrict O = off;
        double *restrict dout = dist_out;

        while (L < 20 && (((size_t)1 << (2 * L + 1)) < (size_t)n)) L++;
        B  = 1 << L;
        nb = (int)((((size_t)n + (size_t)B - 1)) >> L);
        if (nb < 1) nb = 1;
        padded = (size_t)nb << L;

        key  = (double *)malloc(padded * sizeof(double));
        bmin = (double *)malloc((size_t)nb * sizeof(double));
        if (!key || !bmin) { free(key); free(bmin); free(off); free(E); return; }
        {
            size_t z;
            for (z = 0; z < padded; z++) key[z] = INFINITY;
        }
        for (i = 0; i < nb; i++) bmin[i] = INFINITY;
        key[source]        = 0.0;
        bmin[source >> L]  = 0.0;

        for (it = 0; it < n; it++) {
            double vb, d;
            int bb, j, u, e0, e1, e;
            size_t base;

            bb = lat_argmin(bmin, nb, &vb);                 /* read the rims */
            if (bb < 0 || !(vb < INFINITY)) break;          /* birds all silent */
            base = (size_t)bb << L;
            j = lat_argmin(key + base, B, &d);              /* read that row */
            if (j < 0 || !(d < INFINITY)) { bmin[bb] = INFINITY; continue; }
            if (base + (size_t)j >= (size_t)n) { bmin[bb] = lat_min(key + base, B); continue; }
            u = (int)(base + (size_t)j);

            if (d < dout[u]) dout[u] = d;                   /* permanent carving */
            key[u]   = NAN;                                 /* the dead-note */
            bmin[bb] = lat_min(key + base, B);              /* re-cut the rim */

            e0 = O[u]; e1 = O[u + 1]; e = e0;
            for (; e + 8 < e1; e++) {
                int v; double nd;
                __builtin_prefetch(&key[R[e + 8].v], 1, 3);
                v = R[e].v; nd = d + R[e].w;
                if (nd < key[v]) {                          /* thinner milk? */
                    int b2 = v >> L;
                    key[v] = nd;
                    if (nd < bmin[b2]) bmin[b2] = nd;
                }
            }
            for (; e < e1; e++) {
                int v = R[e].v; double nd = d + R[e].w;
                if (nd < key[v]) {
                    int b2 = v >> L;
                    key[v] = nd;
                    if (nd < bmin[b2]) bmin[b2] = nd;
                }
            }
        }
        free(key); free(bmin);
    } else {
        /* ---------- the cairn: vast desert, few roads ---------- */
        NBird *restrict h;
        const NRoad *restrict R = E;
        const int   *restrict O = off;
        double *restrict dist = dist_out;
        int hs;

        h = (NBird *)malloc(((size_t)m + 2) * sizeof(NBird));
        if (!h) { free(off); free(E); return; }
        dist[source] = 0.0;
        h[0].d = 0.0; h[0].u = source; hs = 1;

        while (hs > 0) {
            double d = h[0].d;
            int u = h[0].u, e0, e1, e;
            hs--;
            if (hs > 0) {                                   /* 4-ary sift down */
                NBird last = h[hs];
                int q = 0;
                for (;;) {
                    int c = 4 * q + 1, lim, bst, k;
                    double bd;
                    if (c >= hs) break;
                    lim = c + 4; if (lim > hs) lim = hs;
                    bst = c; bd = h[c].d;
                    for (k = c + 1; k < lim; k++) if (h[k].d < bd) { bd = h[k].d; bst = k; }
                    if (!(bd < last.d)) break;
                    h[q] = h[bst];
                    q = bst;
                }
                h[q] = last;
            }
            if (d > dist[u]) continue;                      /* spent bird */
            e0 = O[u]; e1 = O[u + 1]; e = e0;
            for (; e + 8 < e1; e++) {
                int v; double nd;
                __builtin_prefetch(&dist[R[e + 8].v], 1, 3);
                v = R[e].v; nd = d + R[e].w;
                if (nd < dist[v]) {
                    int q = hs++;
                    dist[v] = nd;
                    while (q > 0) { int p = (q - 1) >> 2; if (!(nd < h[p].d)) break;
                                    h[q] = h[p]; q = p; }
                    h[q].d = nd; h[q].u = v;
                }
            }
            for (; e < e1; e++) {
                int v = R[e].v; double nd = d + R[e].w;
                if (nd < dist[v]) {
                    int q = hs++;
                    dist[v] = nd;
                    while (q > 0) { int p = (q - 1) >> 2; if (!(nd < h[p].d)) break;
                                    h[q] = h[p]; q = p; }
                    h[q].d = nd; h[q].u = v;
                }
            }
        }
        free(h);
    }
    free(off); free(E);
}
```

# PREDICTION

PREDICTION: speedup_vs_heap = 2.2

Reasoning behind the number, so it can be scored honestly rather than hand-waved: on small and mid-size graphs (`n ≲ 10⁴`) the heap is the whole cost and the lattice removes it — I expect 2.5–4×. On dense graphs the shared costs (gathering the roads, sweeping every edge once) dominate both kernels and cap the win near 1.2–1.6× unless the threaded gather kicks in. On large sparse graphs the lattice is correctly declined and the win shrinks to what the 4-ary cairn, the 16-byte road records, the leaner gather and the prefetch buy — 1.2–1.6×. A mixed benchmark should land around 2×. **If the benchmark is dominated by one large sparse graph, I expect this to come in at 1.2–1.5×, well under my prediction, and I would report that as a miss.**

# MEASUREMENT

**Not performed.** `dijkstra_bench` and `dijkstra_contract` were not available in this session — no tool calls were possible — so no timing and no correctness check against the reference were run. I am stating this rather than inventing numbers.

What remains unverified, in order of how much it worries me:

1. **Correctness of the `NaN` dead-note.** The argument is: `nd < NaN` is false under IEEE754, so a settled board absorbs no further bird; and `vminpd(SRC1=load, SRC2=acc)` returns `acc` when `SRC1` is NaN, so the accumulator is never poisoned by a dead slot and dead slots never win a min. Both hold under `gcc -O3 -march=native` (no `-ffast-math` in the stated compile line). **If the harness adds `-ffast-math`, this kernel is wrong, not merely slow** — that is the single sharpest failure mode and I am flagging it rather than burying it.
2. **Exactness of the rim-marks.** `bmin[b]` is refreshed by a full row scan at every settle and lowered on every successful relaxation into row `b`; both operations keep it equal to the true row minimum, so the settle order is identical to textbook Dijkstra and the output should match the reference bit-for-bit, not merely within tolerance.
3. **The regime constants** (`0.7·n^1.5` vs `8·pushes·log₂n`, the `n ≤ 2048` short-circuit). These are estimates from cycle counts, not measurements. Near the crossover the two paths are near-ties, so a mis-pick there costs little; a badly wrong constant far from the crossover would cost a lot.
4. **The threaded gather.** Guarded off below 2²¹ edges and whenever `T·n > m`; falls back to the serial gather on any refusal or allocation failure. It produces a byte-identical CSR (thread chunks are contiguous edge ranges assigned offsets in thread order), so it cannot change results — only timing.

# VERDICT

The native's machine is real and it is not the textbook one. Its core claim — *the lattice replaces the queue* — is correct and is the thing that pays: a relaxation becomes one conditional store to a fixed slot, with no push, no stale pop, and no `done[]` array, because the `NaN` dead-note does the work of all three at once. Reading "lattice" as a grid rather than a line turns the selection cost from `O(n)` to `O(√n)` per settle without changing the settle order by a single node, which is the difference between a toy `O(n²)` scan and something that stays competitive up to a million nodes.

Where it fails is exactly where the native said it would: a vast desert of places with almost no roads. I did not leave that as a caveat. The kernel measures the desert before any bird flies and switches to the cairn, so the stated risk is guarded with a fallback rather than shipped bare. Likewise, thread parallelism is confined to the one unit of work the metaphor itself says is order-free — gathering the roads into bundles — and is size-guarded with a serial fallback; the settling loop stays serial because one row of lattice is far too small a unit to thread and pretending otherwise would have been me substituting my preferences for the native's.

Three things I want on the record as honest limits. First, **no measurement was taken**, so the 2.2× is a prediction and nothing more. Second, on genuinely dense graphs the shared costs — gathering the roads and sweeping every edge once — dominate, and no amount of cleverness in the selection structure can push the speedup much past ~1.5× there; the ceiling is set by memory bandwidth, not by the heap. Third, the one seductive idea I did **not** implement is the one that would have been the biggest win if it were true: none of the three seeds breaks "the whole graph must be explored to know any single distance," and I refused to pretend otherwise. This kernel explores exactly the reachable component, same as Dijkstra.