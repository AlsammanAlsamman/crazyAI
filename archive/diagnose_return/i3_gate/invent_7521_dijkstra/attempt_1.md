## MAPPING

| World object (native's words) | Computational object | Silent assumption it breaks |
|---|---|---|
| **SEED 1** — garden of stones, each chalked with an owed letter | `dist_out[v]`, a tentative distance **living in place** at node `v` | — |
| nightingales *circle the unlocked stones* and **always drop first onto the smallest owed sum** | repeatedly settle the exact minimum tentative distance among unsettled nodes | — |
| the birds circle **over the garden**, in bands, descending to the lowest band that still holds an unlocked stone | distances are *bucketed by value* into rings of width Δ; the next node comes from the **lowest non-empty ring**, never from a global comparison tournament | **"the next place to finalize is found by comparing against every remaining place"** — the birds never compare all stones; they descend to a band and only look inside it |
| "scratch it out, chalk the smaller sum … **though the stone stays exactly where it stood**" | decrease-key with **no reordering of the container**: the value is overwritten in place; if the new value is still in the same ring, nothing moves at all | breaks the heap's premise that a key change must re-sift the item |
| "a knot of three threads … **detached** … and **called back on** — recast toward the same stone with the lower number" | a 3-link knot per stone (`prv`, `nxt`, `ringof`): unlink from the old ring, relink into the cheaper ring, O(1) | — |
| **SEED 2** — threads flung from the tower to each house the roads actually touch; "I ask the thief who guards that road what he is owed, **directional**" | relax the CSR out-adjacency of the settled node; one weight per directed arc `(u→v, w)` | "a road can only be considered once its starting place is fully settled" — *preserved*, not broken |
| a thief **owed nothing** (toll 0) | `nd == du`: the neighbour is at the current global minimum, so it is **locked in the same breath** (an immediate-lock stack) | **"each place's distance must be finalized before its neighbors are explored"** — partially: a zero-toll neighbour is finalized *during* the parent's expansion, with no queue visit at all |
| **SEED 3** — a locked letter is never rebuilt | settled ⇒ final; no `done[]` re-check needed, because non-negative tolls make `nd < dist[v]` impossible for a locked `v` | "a priority structure must be consulted before every relaxation" — no push, no pop, no duplicate entry for a superseded value |
| threads into bramble / over the tide / around a wall to nowhere new — "let them drop and throw them away" | superseded entries are **overwritten**, never accumulated (no lazy duplicates); unreachable stones keep `INFINITY` | **"the whole graph must be explored to know any single distance"** — untouched components are never allocated a queue entry |
| a garden of *a mere handful of stones*, or roads so thick the garden is one band | Δ = ∞ ⇒ **one single ring** = SIMD linear arg-min over unlocked stones (the classic dense/small O(n²) scan), the same mechanism degenerate | regime recognition (step 5) |

## CHOSEN SEED

**SEED 1: "The nightingales always land first on the stone carrying the smallest owed letter."**

It is the only seed that describes the *ordering machine*, and its literal reading is maximally far from the known way. Two details in the body forbid a heap outright: the birds **circle over the garden in bands** (they descend to a region, they do not run a tournament over all stones), and a rebuilt letter leaves **"the stone exactly where it stood"** (a decrease-key that *moves nothing*). A binary heap violates both. Taken literally, SEED 1 is a **value-bucketed monotone queue with in-place keys and O(1) unlink/relink knots** — which is the validated Dial/bucket-queue family (step 4: the mechanism arrives at a real technique rather than inventing one), generalised to real-valued tolls by letting the birds pick the true minimum *inside* the lowest band.

## ASSUMPTION BROKEN

Primary: **"the next place to finalize is found by comparing against every remaining place."** The birds locate the next stone by descending bands (a 64-ring-at-a-time bitmap skip, amortised O(1) because the bands are swept monotonically), then look only inside one band. No node is ever compared against the global population, and no key change ever reorders anything.

Stated plainly, as instructed: **none of the three seeds breaks "each place's distance must be finalized before its neighbors are explored."** The native is explicit — *"I do not walk everywhere at once"*, *"wherever a nightingale lands, that stone's letter is set"*. The single exception the metaphor does grant is the thief owed nothing: a **zero-toll road locks its far stone in the same breath**, finalising a neighbour inside the parent's expansion. I implement exactly that and nothing more; I do not smuggle in Δ-stepping, which would contradict the native's own words.

Also broken: **"a priority structure must be consulted before every relaxation"** — a relaxation that lands in the same band touches only `dist_out[v]`.

## ARTIFACT

```c
/* The garden of nightingale: distances chalked in place on stones that never move;
   the garden divided into bands (rings) of owed sum; the birds descend to the lowest
   band still holding an unlocked stone and land on its smallest letter.
   Each stone carries a knot of three threads (prv, nxt, ring) so a cheaper letter
   detaches it and calls it back on in one motion. A thief owed nothing locks his
   far stone in the same breath. A small or road-thick garden is one single band. */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define GN_LOCKED (-2)

typedef struct { double w; int v; int pad; } GnArc;   /* 16B: one stream per stone */

static int gn_pow2(int x){ int p = 1; while (p < x && p < (1<<30)) p <<= 1; return p; }

/* ---- one single band: the birds circle the whole garden (dense / tiny garden) ---- */
static void gn_one_band(int n, const int * restrict off, const GnArc * restrict arc,
                        int source, double * restrict dist, double * restrict key)
{
    for (int i = 0; i < n; i++) { dist[i] = INFINITY; key[i] = INFINITY; }
    dist[source] = 0.0; key[source] = 0.0;
    for (int it = 0; it < n; it++) {
        double b; int i = 0;
#if defined(__AVX2__)
        {
            __m256d m0 = _mm256_set1_pd(INFINITY), m1 = m0;
            for (; i + 8 <= n; i += 8) {
                m0 = _mm256_min_pd(m0, _mm256_loadu_pd(key + i));
                m1 = _mm256_min_pd(m1, _mm256_loadu_pd(key + i + 4));
            }
            m0 = _mm256_min_pd(m0, m1);
            double t4[4]; _mm256_storeu_pd(t4, m0);
            b = t4[0];
            if (t4[1] < b) b = t4[1];
            if (t4[2] < b) b = t4[2];
            if (t4[3] < b) b = t4[3];
            for (; i < n; i++) if (key[i] < b) b = key[i];
        }
#else
        b = INFINITY;
        for (; i < n; i++) if (key[i] < b) b = key[i];
#endif
        if (!(b < INFINITY)) break;
        int u = -1;
#if defined(__AVX2__)
        {
            __m256d vb = _mm256_set1_pd(b);
            int j = 0;
            for (; j + 4 <= n; j += 4) {
                int msk = _mm256_movemask_pd(
                    _mm256_cmp_pd(_mm256_loadu_pd(key + j), vb, _CMP_EQ_OQ));
                if (msk) { u = j + (int)__builtin_ctz((unsigned)msk); break; }
            }
            if (u < 0) for (; j < n; j++) if (key[j] == b) { u = j; break; }
        }
#else
        for (int j = 0; j < n; j++) if (key[j] == b) { u = j; break; }
#endif
        if (u < 0) break;
        key[u] = INFINITY;                 /* locked: never chalked over again */
        int e1 = off[u + 1];
        for (int e = off[u]; e < e1; e++) {
            int v = arc[e].v;
            double nd = b + arc[e].w;
            if (nd < dist[v]) { dist[v] = nd; key[v] = nd; }
        }
    }
}

/* ---- the three-thread knot: detach, and call back on ---- */
static inline void gn_link(int v, int p, int * restrict head, int * restrict nxt,
                           int * restrict prv, int * restrict rin, uint64_t * restrict bits)
{
    int h = head[p];
    nxt[v] = h; prv[v] = -1;
    if (h >= 0) prv[h] = v;
    head[p] = v; rin[v] = p;
    bits[p >> 6] |= (uint64_t)1 << (p & 63);
}
static inline void gn_unlink(int v, int * restrict head, int * restrict nxt,
                             int * restrict prv, int * restrict rin, uint64_t * restrict bits)
{
    int p = rin[v], a = prv[v], b = nxt[v];
    if (a >= 0) nxt[a] = b; else head[p] = b;
    if (b >= 0) prv[b] = a;
    if (head[p] < 0) bits[p >> 6] &= ~((uint64_t)1 << (p & 63));
    rin[v] = -1;
}

void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out)
{
    if (n <= 0) return;
    if (source < 0 || source >= n) { for (int i = 0; i < n; i++) dist_out[i] = INFINITY; return; }
    if (m < 0) m = 0;

    /* --- lay out the roads leaving each house, and hear every thief's price once --- */
    int   *off = (int*)malloc((size_t)(n + 1) * sizeof(int));
    GnArc *arc = (m > 0) ? (GnArc*)malloc((size_t)m * sizeof(GnArc)) : NULL;
    if (!off || (m > 0 && !arc)) { free(off); free(arc); return; }
    memset(off, 0, (size_t)(n + 1) * sizeof(int));
    double wmax = 0.0, wsum = 0.0;
    for (int i = 0; i < m; i++) {
        off[src[i] + 1]++;
        double w = weight[i];
        wsum += w;
        if (w > wmax) wmax = w;
    }
    for (int i = 0; i < n; i++) off[i + 1] += off[i];
    {
        int *fill = (int*)malloc((size_t)n * sizeof(int));
        if (!fill) { free(off); free(arc); return; }
        memcpy(fill, off, (size_t)n * sizeof(int));
        for (int i = 0; i < m; i++) {
            int u = src[i], p = fill[u]++;
            arc[p].v = dst[i]; arc[p].w = weight[i]; arc[p].pad = 0;
        }
        free(fill);
    }

    /* --- which garden am I standing in? (regime test, in the metaphor's own terms) --- */
    double nd_ = (double)n;
    int one_band = (n <= 1024) || ((double)m >= 0.05 * nd_ * nd_);
    if (one_band) {
        double *key = (double*)malloc((size_t)n * sizeof(double));
        if (key) { gn_one_band(n, off, arc, source, dist_out, key); free(key); }
        else {     /* no scratch: fall back to bands below */ one_band = 0; }
        if (one_band) { free(off); free(arc); return; }
    }

    /* --- band width chalked from the tolls actually charged --- */
    int K;
    if (wmax <= 0.0) K = 64;
    else {
        double wmean = (m > 0 && wsum > 0.0) ? wsum / (double)m : wmax;
        double ratio = wmax / (wmean > 0.0 ? wmean : wmax);
        double want = 2048.0 * ratio;
        if (!(want >= 64.0)) want = 64.0;
        if (want > 32768.0) want = 32768.0;
        K = gn_pow2((int)want);
        int cap = gn_pow2(n); if (cap < 64) cap = 64;
        if (K > cap) K = cap;
        if (K < 64) K = 64;
    }
    int R = K * 2;                       /* window needs K+2 bands; 2K is a power of two */
    int W = R >> 6;
    double invD = (wmax > 0.0) ? ((double)K) / wmax : 1.0;

    int      *head = (int*)malloc((size_t)R * sizeof(int));
    uint64_t *bits = (uint64_t*)calloc((size_t)W, sizeof(uint64_t));
    int      *nxt  = (int*)malloc((size_t)n * sizeof(int));
    int      *prv  = (int*)malloc((size_t)n * sizeof(int));
    int      *rin  = (int*)malloc((size_t)n * sizeof(int));
    int      *stk  = (int*)malloc((size_t)n * sizeof(int));
    if (!head || !bits || !nxt || !prv || !rin || !stk) {
        free(head); free(bits); free(nxt); free(prv); free(rin); free(stk);
        double *key = (double*)malloc((size_t)n * sizeof(double));
        if (key) { gn_one_band(n, off, arc, source, dist_out, key); free(key); }
        free(off); free(arc); return;
    }
    for (int i = 0; i < R; i++) head[i] = -1;
    for (int i = 0; i < n; i++) { dist_out[i] = INFINITY; rin[i] = -1; }

    /* "nothing owed" beneath my own feet */
    dist_out[source] = 0.0;
    nxt[source] = -1; prv[source] = -1; head[0] = source; rin[source] = 0;
    bits[0] |= (uint64_t)1;
    long long cur = 0;                   /* lowest band index the birds have reached */
    int qsize = 1, sp = 0;

    while (qsize > 0 || sp > 0) {
        int u; double du;
        if (sp > 0) {                    /* stones already locked by a toll-free road */
            u = stk[--sp]; du = dist_out[u];
        } else {
            /* the birds descend to the lowest band still holding an unlocked stone */
            int p = (int)(cur & (R - 1));
            int wi = p >> 6, bi = p & 63;
            uint64_t x = bits[wi] >> bi;
            long long t = -1;
            if (x) t = cur + (long long)__builtin_ctzll(x);
            else {
                long long adv = (long long)(64 - bi);
                int w2 = wi;
                for (int k = 0; k < W; k++) {
                    w2 = (w2 + 1) & (W - 1);
                    uint64_t y = bits[w2];
                    if (y) { t = cur + adv + (long long)__builtin_ctzll(y); break; }
                    adv += 64;
                }
            }
            if (t < 0) break;
            cur = t;
            p = (int)(t & (R - 1));
            int best = head[p];
            double bd = dist_out[best];
            for (int z = nxt[best]; z >= 0; z = nxt[z]) {   /* land on the smallest letter */
                double dz = dist_out[z];
                if (dz < bd) { bd = dz; best = z; }
            }
            gn_unlink(best, head, nxt, prv, rin, bits);
            qsize--;
            u = best; du = bd;
        }
        rin[u] = GN_LOCKED;               /* set, permanently, never rebuilt again */

        int e1 = off[u + 1];
        for (int e = off[u]; e < e1; e++) {           /* cast threads, one direction only */
            int v = arc[e].v;
            double nd = du + arc[e].w;
            if (nd < dist_out[v]) {
                dist_out[v] = nd;                      /* chalked in place */
                if (nd == du) {                        /* a thief owed nothing */
                    if (rin[v] >= 0) { gn_unlink(v, head, nxt, prv, rin, bits); qsize--; }
                    rin[v] = GN_LOCKED; stk[sp++] = v; /* locked in the same breath */
                } else {
                    long long ti = (long long)(nd * invD);
                    int np = (int)(ti & (R - 1));
                    int op = rin[v];
                    if (op != np) {                    /* detach the knot, call it back on */
                        if (op >= 0) { gn_unlink(v, head, nxt, prv, rin, bits); qsize--; }
                        gn_link(v, np, head, nxt, prv, rin, bits);
                        qsize++;
                    }                                  /* else: the stone stays where it stood */
                }
            }
        }
    }

    free(head); free(bits); free(nxt); free(prv); free(rin); free(stk);
    free(off); free(arc);
}
```

## PREDICTION

**PREDICTION: speedup_vs_heap = 2.8**

(Written before any measurement. Reasoning: the reference pays ~`m` heap pushes at `log n` random-access sift steps each plus lazy-deletion pops; the band flight pays one O(1) unlink/relink per *improvement* with no duplicates at all, an amortised-O(1) band descent, and a band occupancy of ~1–4 stones so the in-band arg-min is a couple of loads. On the dense/tiny regime the single-band AVX2 flight should be 3–8×; on large sparse graphs the CSR build becomes the floor and caps the win near 2.5–3×.)

## MEASUREMENT

**Not performed in this session — no tools were available** (`dijkstra_bench` / `dijkstra_contract` were not callable here), so I am reporting this as unmeasured rather than implying a number I did not observe. The pipeline will compile and run it. What I can state without a harness:

- **Contract**: signature matches `kernel(int, int, const int*, const int*, const double*, int, double*)` exactly, argument order untouched; `dist_out[i] = INFINITY` for unreachable `i`, `0` at `source`.
- **Exactness argument** (why it must match the reference within tolerance): band index `floor(d·K/wmax)` is monotone non-decreasing in `d`, so the minimum inside the lowest non-empty band **is** the global minimum over unsettled nodes — the band structure is approximate but the *landing* is exact, so the settle order is non-decreasing in true distance, i.e. textbook Dijkstra order. The circular array is safe because all live tentative distances lie in `[d_last, d_last + wmax]`, a span of at most `K+2 ≤ R` bands. Locked nodes need no `done[]` test: `nd = du + w ≥ du ≥ dist[v]` for any already-locked `v`. The zero-toll immediate lock is sound because `du` is the current global minimum and no non-negative-weight path can ever beat it.
- **Zero-weight and all-zero-weight graphs** are handled entirely by the immediate-lock stack (the rings are never touched), so `Δ` never needs to be ≤ `w_min`.
- Four improvement rounds went into *how* the mechanism is implemented, not into replacing it: (1) bands + bitmap word-skip instead of a per-pop global flight; (2) true in-place decrease-key with the 3-link knot, killing all duplicate queue entries; (3) same-band relaxations touch nothing but the chalked letter; (4) 16-byte interleaved arcs + `restrict` so each stone's roads are one sequential stream, and a 2-pass CSR build.

## VERDICT

The core of this kernel **is** SEED 1, translated literally: distances are letters chalked in place on stones that never move, the garden is divided into bands of owed sum, and the next node is found by *descending to the lowest occupied band and landing on its smallest letter* — never by a heap sift, never by a tournament against every remaining node. That mechanism, followed honestly, lands on the **Dial/bucket-queue family** (step 4: a validated real technique, generalised to real-valued weights by making the in-band landing exact), not on something I invented.

Risks I named, and how each is closed rather than left dangling:

- *"Bands cost O(R) setup and random `head[]` traffic — a loss on a tiny or road-thick garden."* Closed by the metaphor's own regime test: `n ≤ 1024 || m ≥ 0.05·n²` takes the **single-band flight**, which is the same mechanism with one ring and is exactly the known dense/small O(n²) AVX2 scan win. Both regimes named in `known_way` therefore have a path, chosen at runtime.
- *"A pathological toll spread would blow up the band count."* Closed by `K ≤ 32768` and `K ≤ pow2(n)`; correctness never depended on `Δ ≤ w_min`, only on band monotonicity, so clamping `Δ` costs occupancy, never accuracy.
- *"A long path makes the band sweep long."* Closed by the bitmap: 64 empty bands skipped per word, and the sweep is monotone, so the whole run's descent cost is amortised, not per-pop.
- *Thread parallelism deliberately omitted.* The metaphor's own unit of work is one bird's landing, which is strictly sequential; the only parallelisable part is laying out the roads (a memory-bound histogram). Per instruction I stopped at vectorisation hints (AVX2 arg-min, `restrict`, one 16-byte arc stream) rather than adding OpenMP whose units would not pay at these sizes.

Honest caveat: if the benchmark's weights are near-constant and the graph is mid-size sparse, band occupancy collapses to 1 and I expect a clear win; if the benchmark is a single very large sparse graph, the CSR build is the floor and the win will be closer to 2× than 3×. I would rather state that now than discover it and rationalise it afterwards.