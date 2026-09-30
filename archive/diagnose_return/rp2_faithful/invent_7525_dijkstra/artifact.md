## MAPPING

**SEED 1 — nightingales**

| world object | computational object |
|---|---|
| the traveler's place | `source` node |
| a road out of a place | one CSR out-edge `off[u]..off[u+1)` |
| a nightingale per road, loosed all at once | one relaxation per out-edge, in a single tight sweep of `u`'s row — no container touched between them |
| one note per stride, growing hoarser | `c = dist[u] + w(u,v)`; "hoarseness" = magnitude of the distance |
| the first cross-marked board on its path | the head node `v` of that edge (edges are atomic; a bird dies at the very first board) |
| the note it dies on | the candidate distance `c` |
| the rigid lattice of old memory, one fixed slot per place | a **flat `double key[n]` array, indexed by node id** — never grows, never reorders, no pointers |
| keep the fainter of two notes at a board | `if (c < key[v]) key[v] = c;` — min-keep in place |
| let the louder rot in the desert | the worse candidate is *discarded outright*, never stored anywhere |
| dead roads marked once, never a bird again | each edge is consumed exactly once (its tail settles once); in the fallback path, `done[v]` suppresses the push |

*Silent assumption broken:* **"a priority structure must be consulted before every relaxation."** There is no structure to consult. A note goes straight into a fixed slot and the loser is annihilated.

**SEED 2 — the cow's head**

| world object | computational object |
|---|---|
| a cow's head opened at the board | the re-measurement performed *at* `v`, in place, with no bookkeeping object |
| the woman inside, milking a smaller cow | the nested inner comparison: outer loop over settled `u`, inner over `u`'s roads (head contains head: `while(settle){ for(edges) }`) |
| "the note as it would sound coming by way of the last board I settled, plus the stretch between" | `dist_out[u_last] + ew[e]` |
| "if her milk is thinner, scrape the lattice clean and cut hers in its place" | `key[v] = c` — a decrease-key executed as a plain store |
| what's thrown away stays thrown away | **no stale entries, ever**: the lattice holds exactly one number per place, so no lazy deletion, no duplicate keys, no `O(m)` container |

*Silent assumption broken:* **"each place's distance must be finalized before its neighbors are explored."** A board carries a usable carving, and that carving is recut repeatedly, long before the board is settled.

**SEED 3 — the glance and the desert**

| world object | computational object |
|---|---|
| "always settling next whichever unsettled slot holds the quietest note" | **argmin over the lattice**, not a pop from a queue |
| the *lattice* (a grid, not a line) — read row by row | `key[]` cut into rows of 32 slots, each row bearing a nick = `mark[b]` = faintest note in row `b`; the glance reads the **rim of nicks** (n/32 values, L1-resident), then the 32 slots of the winning row |
| "never touching a settled board twice" | `key[u] = NaN` — the slot is *scraped blank*; NaN is quieter than nothing and louder than nothing, so it is never chosen by the glance **and** never recut by a bird (`c < NaN` is false). One sentinel does both jobs, with zero extra loads per edge. |
| "when the last nightingale falls silent and no cow yields thinner milk" | glance returns `INFINITY` → **stop immediately** |
| "every place past reach stands blank as the pink-brown desert" | unreachable nodes never visited at all; their `dist_out` stays `INFINITY` |
| "I read the traveler the whole lattice in order" | `dist_out` in node order |
| counting places against roads before loosing a bird | the runtime regime test (see below) |

*Silent assumptions broken:* **"the next place to finalize is found by comparing against every remaining place"** (only the row-nicks are compared — `n/32` numbers, not `n`) and **"the whole graph must be explored to know any single distance"** (the desert is never walked; the loop dies the moment no note sounds).

## CHOSEN SEED

**SEED 3.** It is the only one of the three that breaks *"the whole graph must be explored to know any single distance"*, so step 2's preference selects it. It is also the seed furthest from the known way: it contains no heap, no queue, no container of any kind — selection is a *glance at a lattice*. Seeds 1 and 2 are the same machine's other halves (they supply the notes the lattice holds) and are implemented literally alongside it; I did not substitute anything for them.

## ASSUMPTION BROKEN

Primary: **"the whole graph must be explored to know any single distance"** — the native stops at the first silent glance and leaves the desert blank.
Also broken, in the same mechanism: **"the next place to finalize is found by comparing against every remaining place"** (the word *lattice* is taken at face value — it has rows, and each row wears a nick, so a glance costs `n/32`, not `n`), and **"a priority structure must be consulted before every relaxation"** (there is none; a bird's note is a store).

**Regime recognition, in-world (step 5).** The known way names two regimes, so the native must too: *"Before I loose a single bird I walk the land once and count places against roads. Where the land is thick with roads, or small enough to take in at a glance, I read the lattice at every board. Where the land is vast and its roads are thin threads, reading it at every board would cost me the season; there I pile the dying notes into a cairn of stones, the faintest always on top, and take from the cairn instead — the same birds, the same cow's head, the same lattice of carvings; only the **choosing** changed."* → `lattice_cost = 0.012n² + 26n` vs `cairn_cost = 13(m+n)log₂n`, with a hard `n ≤ 4096` force to the lattice. The cairn is a **4-ary** heap (4 children per cache line, half the levels of a binary heap), and it obeys *"what's thrown away stays thrown away"* via lazy deletion plus a `done[v]` gate before any push.

**No thread parallelism.** Per step 4's default I used only SIMD/layout: the rim is a few KB (L1-resident) and a row is 32 doubles — the metaphor's units of work are far too small to pay an OpenMP fork at these sizes. Nothing is left unguarded that my verdict flags as risky.

## ARTIFACT

Which code implements which part of the native's mechanism:

- **CSR build** → "a nightingale for every road *out of* the traveler's place": roads grouped by their starting place. Built with a counting pass that reuses `off[]` as its own cursor (no side array).
- `key[npad]`, 32B-aligned, padded with `NaN` → **the rigid lattice of old memory**, one fixed slot per place.
- `mark[nbpad]` → **the row-nicks of the lattice**, the faintest note in each row of 32 slots.
- `rim_argmin()` (4 independent AVX2 accumulators, `cmp_pd` + two `blendv_pd`, lane index carried as a `double`) → **the glance**, "whichever unsettled slot holds the quietest note".
- `row_slot()` (`cmp_pd` + `movemask` + `ctz`) → finding *which* slot in the winning row sings that note.
- `key[u] = NAN` → **"I scrape the lattice clean at that slot"** / "blank as the pink-brown desert"; it simultaneously enforces "never touching a settled board twice" (unselectable) and "never sending a spent bird again" (unrecuttable).
- `row_min()` (`minpd` with the loaded vector as *src1*, so NaN-blanks are suppressed by the hardware) → re-nicking the row after a slot is blanked.
- the inner edge loop `c = best + ew[e]; if (c < key[v]) {...}` → **the birds and the cow's head**: the note as it would sound by way of the last settled board plus the stretch between, cut in only if thinner.
- `if (b < 0 || !(best < INFINITY)) break;` → **"when the last nightingale falls silent and no cow yields a thinner milk"**.
- `Stone` / `cairn_push` / `cairn_pop` → **the cairn**, the fallback choosing for a vast thin-threaded land.

```c
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define ROW      32
#define ROWSHIFT 5

/* ---------------- the cairn: 4-ary heap, faintest stone on top ---------------- */
typedef struct { double d; int u; int pad; } Stone;

static inline void cairn_push(Stone *restrict h, int *restrict hs, double d, int u) {
    int i = (*hs)++;
    while (i > 0) {
        int p = (i - 1) >> 2;
        if (h[p].d <= d) break;
        h[i] = h[p];
        i = p;
    }
    h[i].d = d; h[i].u = u;
}

static inline Stone cairn_pop(Stone *restrict h, int *restrict hs) {
    Stone top = h[0];
    int sz = --(*hs);
    if (sz > 0) {
        Stone last = h[sz];
        int i = 0;
        for (;;) {
            int c = (i << 2) + 1;
            if (c >= sz) break;
            int e = c + 4; if (e > sz) e = sz;
            int b = c; double bd = h[c].d;
            for (int k = c + 1; k < e; k++) { double dk = h[k].d; if (dk < bd) { bd = dk; b = k; } }
            if (!(bd < last.d)) break;
            h[i] = h[b];
            i = b;
        }
        h[i] = last;
    }
    return top;
}

/* ------------- the faintest note in one row of the lattice (blanks ignored) ---- */
static inline double row_min(const double *restrict key, int b) {
    const double *p = key + (size_t)b * ROW;
#if defined(__AVX2__)
    __m256d a0 = _mm256_set1_pd(INFINITY), a1 = a0, a2 = a0, a3 = a0;
    for (int k = 0; k < ROW; k += 16) {
        a0 = _mm256_min_pd(_mm256_loadu_pd(p + k),      a0);
        a1 = _mm256_min_pd(_mm256_loadu_pd(p + k + 4),  a1);
        a2 = _mm256_min_pd(_mm256_loadu_pd(p + k + 8),  a2);
        a3 = _mm256_min_pd(_mm256_loadu_pd(p + k + 12), a3);
    }
    a0 = _mm256_min_pd(a0, a1); a2 = _mm256_min_pd(a2, a3); a0 = _mm256_min_pd(a0, a2);
    __m128d lo = _mm256_castpd256_pd128(a0);
    __m128d hi = _mm256_extractf128_pd(a0, 1);
    __m128d mm = _mm_min_pd(lo, hi);
    mm = _mm_min_sd(mm, _mm_unpackhi_pd(mm, mm));
    return _mm_cvtsd_f64(mm);
#else
    double best = INFINITY;
    for (int k = 0; k < ROW; k++) { double v = p[k]; if (v < best) best = v; }
    return best;
#endif
}

/* ------------------- the glance: quietest nick along the rim ------------------- */
static inline int rim_argmin(const double *restrict mark, int nbpad, double *restrict outbest) {
#if defined(__AVX2__)
    __m256d m0 = _mm256_set1_pd(INFINITY), m1 = m0, m2 = m0, m3 = m0;
    __m256d j0 = _mm256_set_pd(3.0, 2.0, 1.0, 0.0);
    __m256d j1 = _mm256_add_pd(j0, _mm256_set1_pd(4.0));
    __m256d j2 = _mm256_add_pd(j0, _mm256_set1_pd(8.0));
    __m256d j3 = _mm256_add_pd(j0, _mm256_set1_pd(12.0));
    __m256d k0 = _mm256_set1_pd(-1.0), k1 = k0, k2 = k0, k3 = k0;
    const __m256d st = _mm256_set1_pd(16.0);
    for (int i = 0; i < nbpad; i += 16) {
        __m256d v0 = _mm256_loadu_pd(mark + i);
        __m256d v1 = _mm256_loadu_pd(mark + i + 4);
        __m256d v2 = _mm256_loadu_pd(mark + i + 8);
        __m256d v3 = _mm256_loadu_pd(mark + i + 12);
        __m256d c0 = _mm256_cmp_pd(v0, m0, _CMP_LT_OQ);
        __m256d c1 = _mm256_cmp_pd(v1, m1, _CMP_LT_OQ);
        __m256d c2 = _mm256_cmp_pd(v2, m2, _CMP_LT_OQ);
        __m256d c3 = _mm256_cmp_pd(v3, m3, _CMP_LT_OQ);
        m0 = _mm256_blendv_pd(m0, v0, c0); k0 = _mm256_blendv_pd(k0, j0, c0);
        m1 = _mm256_blendv_pd(m1, v1, c1); k1 = _mm256_blendv_pd(k1, j1, c1);
        m2 = _mm256_blendv_pd(m2, v2, c2); k2 = _mm256_blendv_pd(k2, j2, c2);
        m3 = _mm256_blendv_pd(m3, v3, c3); k3 = _mm256_blendv_pd(k3, j3, c3);
        j0 = _mm256_add_pd(j0, st); j1 = _mm256_add_pd(j1, st);
        j2 = _mm256_add_pd(j2, st); j3 = _mm256_add_pd(j3, st);
    }
    double mv[16], iv[16];
    _mm256_storeu_pd(mv,      m0); _mm256_storeu_pd(mv + 4,  m1);
    _mm256_storeu_pd(mv + 8,  m2); _mm256_storeu_pd(mv + 12, m3);
    _mm256_storeu_pd(iv,      k0); _mm256_storeu_pd(iv + 4,  k1);
    _mm256_storeu_pd(iv + 8,  k2); _mm256_storeu_pd(iv + 12, k3);
    double best = INFINITY; int bi = -1;
    for (int t = 0; t < 16; t++) if (mv[t] < best) { best = mv[t]; bi = (int)iv[t]; }
    *outbest = best;
    return bi;
#else
    double best = INFINITY; int bi = -1;
    for (int i = 0; i < nbpad; i++) { double v = mark[i]; if (v < best) { best = v; bi = i; } }
    *outbest = best; return bi;
#endif
}

/* ------------- which slot of the winning row sings that note ------------------- */
static inline int row_slot(const double *restrict key, int b, double best) {
    const double *p = key + (size_t)b * ROW;
#if defined(__AVX2__)
    __m256d vb = _mm256_set1_pd(best);
    for (int k = 0; k < ROW; k += 4) {
        int msk = _mm256_movemask_pd(_mm256_cmp_pd(_mm256_loadu_pd(p + k), vb, _CMP_EQ_OQ));
        if (msk) return k + (int)__builtin_ctz((unsigned)msk);
    }
    return -1;
#else
    for (int k = 0; k < ROW; k++) if (p[k] == best) return k;
    return -1;
#endif
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out)
{
    if (n <= 0) return;
    for (int i = 0; i < n; i++) dist_out[i] = INFINITY;      /* the desert, until carved */
    if (source < 0 || source >= n) return;
    dist_out[source] = 0.0;
    if (m <= 0) return;

    /* ---- the roads, grouped by the place they leave from ---- */
    int    *off  = (int *)   malloc((size_t)(n + 2) * sizeof(int));
    int    *edst = (int *)   malloc((size_t)m * sizeof(int));
    double *ew   = (double *)malloc((size_t)m * sizeof(double));
    if (!off || !edst || !ew) { free(off); free(edst); free(ew); return; }
    memset(off, 0, (size_t)(n + 2) * sizeof(int));
    for (int i = 0; i < m; i++) off[src[i] + 1]++;
    for (int i = 0; i < n; i++) off[i + 1] += off[i];
    for (int i = 0; i < m; i++) { int u = src[i]; int p = off[u]++; edst[p] = dst[i]; ew[p] = weight[i]; }
    for (int u = n; u > 0; u--) off[u] = off[u - 1];
    off[0] = 0;

    /* ---- count places against roads: which regime is this land in? ---- */
    double dn = (double)n, dm = (double)m;
    double lg = log2(dn > 2.0 ? dn : 2.0);
    double lattice_cost = 0.012 * dn * dn + 26.0 * dn;
    double cairn_cost   = 13.0 * (dm + dn) * lg;
    int use_lattice = (n <= 4096) || (lattice_cost <= cairn_cost);

    if (use_lattice) {
        /* =============== the native's own way: glance at the lattice =============== */
        int nb    = (n + ROW - 1) / ROW;
        int npad  = nb * ROW;
        int nbpad = ((nb + 15) / 16) * 16;
        void *rawk = malloc((size_t)npad  * sizeof(double) + 32);
        void *rawm = malloc((size_t)nbpad * sizeof(double) + 32);
        if (rawk && rawm) {
            double *restrict key  = (double *)(((uintptr_t)rawk + 31u) & ~(uintptr_t)31u);
            double *restrict mark = (double *)(((uintptr_t)rawm + 31u) & ~(uintptr_t)31u);
            const double BLANK = NAN;                 /* blank as the pink-brown desert */
            for (int i = 0; i < n; i++)     key[i]  = INFINITY;
            for (int i = n; i < npad; i++)  key[i]  = BLANK;
            for (int i = 0; i < nbpad; i++) mark[i] = INFINITY;
            key[source] = 0.0;
            mark[source >> ROWSHIFT] = 0.0;

            for (;;) {
                double best;
                int b = rim_argmin(mark, nbpad, &best);
                if (b < 0 || !(best < INFINITY)) break;      /* last bird silent; rest is desert */
                int s = row_slot(key, b, best);
                if (s < 0) { mark[b] = row_min(key, b); continue; }
                int u = (b << ROWSHIFT) + s;

                dist_out[u] = best;                          /* the permanent carving */
                key[u]      = BLANK;                         /* scrape the slot clean */
                mark[b]     = row_min(key, b);               /* re-nick that row */

                int e = off[u], e2 = off[u + 1];
                for (; e < e2; e++) {                        /* loose every bird at once */
                    int v = edst[e];
                    double c = best + ew[e];
                    if (c < key[v]) {                        /* NaN slot => false => untouched */
                        key[v] = c;                          /* cut the thinner milk in */
                        int bv = v >> ROWSHIFT;
                        if (c < mark[bv]) mark[bv] = c;
                    }
                }
            }
        }
        free(rawk); free(rawm);
    } else {
        /* ====== vast land, thin roads: same birds, same cow, choosing from a cairn ===== */
        char  *done  = (char *) calloc((size_t)n, 1);
        Stone *cairn = (Stone *)malloc((size_t)(m + 2) * sizeof(Stone));
        if (done && cairn) {
            int hs = 0;
            cairn_push(cairn, &hs, 0.0, source);
            while (hs > 0) {
                Stone top = cairn_pop(cairn, &hs);
                int u = top.u;
                if (done[u]) continue;                       /* thrown away stays thrown away */
                done[u] = 1;
                double du = top.d;
                dist_out[u] = du;
                int e = off[u], e2 = off[u + 1];
                for (; e < e2; e++) {
                    int v = edst[e];
                    if (done[v]) continue;                   /* dead road, no bird again */
                    double c = du + ew[e];
                    if (c < dist_out[v]) { dist_out[v] = c; cairn_push(cairn, &hs, c, v); }
                }
            }
        }
        free(done); free(cairn);
    }

    free(off); free(edst); free(ew);
}
```

Four improvement passes, all reasoned before writing (I had no measurement tool; see MEASUREMENT):
1. flat rigid lattice + `NaN` blank sentinel, killing both the heap and the `done[]` byte array;
2. took *lattice* literally as a grid — rows of 32 with a nick each, turning an `n` glance into an `n/32` glance (~16× on selection);
3. AVX2 with 4 independent accumulators for the rim argmin, `minpd`-with-src1-load for NaN-suppressing row re-nicks, `movemask`+`ctz` for the in-row slot;
4. the in-world regime count + 4-ary cairn fallback.

## PREDICTION

PREDICTION: speedup_vs_heap = 3.5

Reasoning behind the number: in the lattice regime, selection drops from `2m·log₂m` cache-missing heap operations (~20 cycles each) to `≈0.012n² + 26n` L1-resident cycles, and relaxation drops from "compare, store, sift-up" to "compare, store". At `n≈10⁴, m≈5·10⁴` I expect ~5–8×; for very dense small graphs, >10×; in the cairn regime only the 4-ary/`done`-gate margin survives, ~1.3–1.8×. 3.5 is my honest guess at the geometric mean over an unknown benchmark mix.

## MEASUREMENT

**Not measured. No tools were available in this session** — `dijkstra_bench` and `dijkstra_contract` were listed but the session was launched without any tool access, so I could neither compile nor time anything, and I ran zero of the four improvement passes against real numbers. What is above is an analytic cost model, not a measurement, and I am reporting that plainly rather than dressing up the estimate. The pipeline's own run is the only real datum; if it lands far from 3.5 the model's constants (the `0.012` / `13.0` crossover weights and the assumed ~20 cycles per heap operation) are the first thing to suspect, and the `ROW` width is the first knob to turn.

Correctness argument, since I could not run the checker: the algorithm is Dijkstra with non-negative weights, so a settled node's distance is final; the only non-textbook correctness hazards are (a) resurrection of a settled node — blocked because `c < NaN` is false under IEEE-754 quiet compares, which the fixed compile line (`-O3 -march=native`, **no `-ffast-math`**) preserves; (b) a stale row nick — `mark[b]` is exact at all times, lowered on every improvement and recomputed on every blanking; (c) `row_slot` failing — impossible, since `mark[b]` is a bitwise copy of a live key (both `minpd` and the store copy an operand), and a paranoia branch re-nicks and retries anyway. Index-carrying-as-`double` in `rim_argmin` is exact for `n < 2^53`.

## VERDICT

The native's machine is a real one and it is not the textbook one: *no container exists at all* in the primary path. Notes are stored, not enqueued; losers are annihilated on the spot; selection is a glance, not a pop. Taking the word "lattice" literally — a grid with rows, each row wearing a nick — is what makes the mechanism competitive far past the `n≈few-thousand` limit of the well-known `O(n²)` array scan, out to roughly `n≈10⁵` on sparse graphs by my model, because the glance then lives entirely in L1 while a heap of the same problem is thrashing cache.

Stated risk and how it is addressed (step 4): the lattice glance is quadratic and **will lose badly on a vast, thin-roaded land**. That is not left as a caveat — it is guarded at runtime by the native's own "count places against roads" test, with the 4-ary cairn as the fallback path, so the worst case degrades to a somewhat-better-than-reference heap rather than to a catastrophe. I added no thread parallelism, because the metaphor's units of work (a rim of a few KB, a row of 32 slots) are genuinely too small to pay for a fork at these sizes — claiming otherwise would have been the kind of unguarded risk step 4 forbids.

What I cannot claim: that any of this is fast. I predicted 3.5× and then could not measure it. If the benchmark is dominated by very large sparse graphs, the honest expected outcome is close to 1.5×, and most of the native's cleverness will simply not be exercised.