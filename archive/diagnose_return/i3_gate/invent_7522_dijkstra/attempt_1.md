## MAPPING

I take every object in the description as naming something concrete, and I refuse to let "find the leader" collapse into "call a heap."

### SEED 1 — *"A stone placed on a corner marks that place's distance as finally known and never reconsidered."*

| World object | Problem object |
|---|---|
| corner of the triangular grid | vertex `v ∈ [0,n)` |
| road out of a corner | directed edge in the CSR row `off[u]..off[u+1]` |
| white stone, "dark artery click", written first | `dist[source] = 0.0`, written before anything else |
| black stone | `state[v] = 2` — permanently settled |
| "never touched again" | a settled vertex's `dist` is never rewritten *and* it is never re-enqueued |
| bare corner, "insight age that never arrives" | `dist[v] = INFINITY` |

**Breaks:** nothing. This *is* the settled-set assumption ("each place's distance must be finalized before its neighbors are explored"). SEED 1 is the one seed that agrees with the textbook.

### SEED 2 — *"A duck carries a spooled thread whose length is the running sum of every road crossed to reach it."*

| World object | Problem object |
|---|---|
| duck | one **candidate label**, created per relaxation — a carried object, plural, not a slot in a global array |
| brilliant thread spooled to exactly that road's **centimetres** | the label's value `dist[u] + w`, measured in a **discrete unit** — centimetres, not reals |
| "I *loose* the ducks" (no comparing at loose time) | enqueue = one divide + one append. **No key comparison, no sift, no structure queried** |
| peg the duck hangs on (implied by "centimetres") | bucket index `⌊d/Δ⌋`, cyclic array of `nb` buckets |
| "roads toward houses with no far side — let hang slack, throw away unspooled, never measured again" | out-degree-0 vertices: `dist` min-updated but **never** entered into the queue (`state=1`) |
| "otherwise I throw the new thread away" | `if (nd < dist[v])` — non-improving labels are never created |
| "let the wandering ducks settle... wait for **stillness**" | re-process the leading peg-band until it is a **fixed point** |
| "it may hide among its own long cousins, so I trace every shot and shuttle back to its landing before I trust it" | lazy stale check at pop: `inb[u] != tag` → this duck is an out-of-date copy, skip |

**Breaks:** *"a priority structure must be consulted before every relaxation."* A duck is loosed without consulting anything. The structure is **written**, never **queried**, during relaxation; it is queried once per round, and then only as "which peg-band is leading."
Also breaks *"the whole graph must be explored to know any single distance"* via the slack-road rule, and *"the next place to finalize is found by comparing against every remaining place."*

### SEED 3 — *"Each round, the leader duck holding the shortest carried thread among the unstoned is found and settled before any other duck moves."*

| World object | Problem object |
|---|---|
| leader duck | the global minimum tentative label |
| "walk the grid with my hand hovering, comparing thread against thread" | linear scan over all `n` keys |
| one leader per round | one extract-min per iteration |

**Breaks:** *"the next place to finalize is found by comparing against every remaining place"* — no, it **affirms** it. SEED 3 is literally the O(n²) array-scan Dijkstra. It is the seed my previous attempt built, and the reviewer was right: it is textbook.

## CHOSEN SEED

**SEED 2 — the duck and its spooled thread.**

Not SEED 3: SEED 3's hovering hand *is* the known O(n²) scan, so building it cannot break anything. Not SEED 1: it is the textbook settled set verbatim. SEED 2 is the only seed that breaks the preferred assumption, and it is the most literal: the word **centimetres** is doing real work. A thread is not a real number, it is a *count of a discrete unit*, so a duck does not need to be compared with other ducks to be filed — it can be **hung on the peg matching its centimetre count**. That single sentence is a bucket queue, and the "stillness" sentence is what makes a *band* of pegs safe to settle as a group.

Reading the whole description back with this in hand, the native's loop is: loose ducks blind → let the leading peg-band reach stillness → stone that band → loose the long roads once. That is **Δ-stepping** (Meyer & Sanders 1998) — the validated reference SSSP of Graph500/GAP, not something I invented. Per step 4, I let the mechanism arrive there rather than making up a new structure.

## ASSUMPTION BROKEN

**"A priority structure must be consulted before every relaxation."**

In a binary heap, every improving relaxation costs `O(log n)` comparisons against *other* labels, each a dependent cache miss. In the native's world, loosing a duck costs: one division, one bounds check, one store. Nothing is compared to anything. The ordering information lives in the *address* the duck is stored at (`⌊d/Δ⌋ & mask`), not in a sequence of comparisons — because threads are measured in centimetres.

Two further assumptions fall as side effects:
- *"A road can only be considered once its starting place is fully settled"* — inside a band, light roads are flown from **unsettled** ducks, repeatedly, until stillness. Stones are placed only afterward.
- *"The whole graph must be explored"* — slack roads (sinks) are never pegged, so those corners are never selected, only written.

**Regime recognition (step 5).** The `known_way` names two regimes, so the native must recognise his own: *"when the roads from every house reach nearly every other house, there are more ducks than corners, and the pegs cost more than the walk; then I hang no ducks at all and simply walk the grid each round with my hand hovering."* That is the runtime check `n ≤ 96 || m/n² > 0.05` → the pure O(n²) hovering-hand scan (SEED 3's mechanism, kept as the *fallback*, where it genuinely wins). A second in-world guard: *"if my hand runs on over empty peg after empty peg far longer than there are corners, I stop pegging, gather every living duck into my hands, and compare by hand thereafter"* — a sweep counter that escapes to lazy heap Dijkstra, which addresses the one condition under which the peg mechanism could lose (pathological weight dynamic range making `Δ` far too fine).

## ARTIFACT

```c
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ---------------------------------------------------------------------------
   A road, laid out as one 16-byte object so a duck fetches its far corner and
   its centimetres in a single cache line.
   --------------------------------------------------------------------------- */
typedef struct { double w; int v; int pad; } Road;

typedef struct { double d; int u; } HeapItem;

static void hpush(HeapItem *h, int *hs, double d, int u) {
    int i = (*hs)++; h[i].d = d; h[i].u = u;
    while (i > 0) { int p = (i - 1) / 2; if (h[p].d <= h[i].d) break;
                    HeapItem t = h[p]; h[p] = h[i]; h[i] = t; i = p; }
}
static HeapItem hpop(HeapItem *h, int *hs) {
    HeapItem top = h[0]; (*hs)--; h[0] = h[*hs];
    int i = 0;
    for (;;) {
        int l = 2 * i + 1, r = 2 * i + 2, s = i;
        if (l < *hs && h[l].d < h[s].d) s = l;
        if (r < *hs && h[r].d < h[s].d) s = r;
        if (s == i) break;
        HeapItem t = h[s]; h[s] = h[i]; h[i] = t; i = s;
    }
    return top;
}

/* ---------------------------------------------------------------------------
   THE HOVERING HAND (dense / tiny grid regime): no pegs at all.  Walk the
   whole grid each round.  Slack roads still apply: a house with no far side
   never joins the comparison.
   --------------------------------------------------------------------------- */
static void hand_walk(int n, const int *restrict off, const Road *restrict E,
                      int source, double *restrict dist)
{
    double *restrict key = (double *)malloc((size_t)n * sizeof(double));
    unsigned char *restrict done = (unsigned char *)calloc((size_t)n, 1);
    if (!key || !done) { free(key); free(done); return; }
    for (int i = 0; i < n; i++) {
        key[i] = INFINITY;
        if (off[i + 1] == off[i]) done[i] = 1;      /* slack roads only */
    }
    if (!done[source]) key[source] = 0.0;
    for (;;) {
        double best = INFINITY;                      /* vectorizable min */
        for (int i = 0; i < n; i++) { double k = key[i]; if (k < best) best = k; }
        if (!(best < INFINITY)) break;
        int u = 0; while (key[u] != best) u++;
        key[u] = INFINITY; done[u] = 1;              /* black stone */
        double du = dist[u];
        int b = off[u + 1];
        for (int e = off[u]; e < b; e++) {
            int v = E[e].v;
            double nd = du + E[e].w;
            if (nd < dist[v]) { dist[v] = nd; if (!done[v]) key[v] = nd; }
        }
    }
    free(key); free(done);
}

/* ---------------------------------------------------------------------------
   kernel: ducks on pegs.
   --------------------------------------------------------------------------- */
void kernel(int n, int m, const int *src, const int *dst, const double *weight,
            int source, double *dist_out)
{
    if (n <= 0) return;
    double *restrict dist = dist_out;
    for (int i = 0; i < n; i++) dist[i] = INFINITY;
    if (source < 0 || source >= n) return;
    dist[source] = 0.0;                              /* white stone, written first */
    if (m <= 0) return;

    /* ---- how long is a road, in centimetres ---- */
    double wmax = 0.0;
    for (int i = 0; i < m; i++) { double w = weight[i]; if (w > wmax) wmax = w; }
    if (!(wmax >= 0.0)) wmax = 0.0;

    double avgdeg = (double)m / (double)n;
    double delta = (wmax > 0.0) ? wmax / (avgdeg + 1.0) : 1.0;
    if (!(delta > 0.0) || !isfinite(delta)) delta = (wmax > 0.0) ? wmax : 1.0;

    /* peg ring: cyclic, power of two.  Sized so no live duck can ever wrap
       onto a peg the hand has already passed:  wmax/delta <= nb - 4.       */
    int nb = 4;
    {
        double need = (wmax / delta) + 4.0;
        if (!(need >= 4.0)) need = 4.0;
        if (need > 65536.0) need = 65536.0;
        while ((double)nb < need && nb < 65536) nb <<= 1;
        if ((double)(nb - 4) < wmax / delta) delta = wmax / (double)(nb - 4);
        if (!(delta > 0.0) || !isfinite(delta)) delta = 1.0;
    }
    const int mask = nb - 1;

    /* ---- CSR, light roads first in each row ---- */
    int *restrict off = (int *)malloc(((size_t)n + 1) * sizeof(int));
    int *ldeg = (int *)calloc((size_t)n, sizeof(int));
    int *tdeg = (int *)calloc((size_t)n, sizeof(int));
    Road *restrict E = (Road *)malloc((size_t)m * sizeof(Road));
    int *restrict lsplit = (int *)malloc((size_t)n * sizeof(int));
    if (!off || !ldeg || !tdeg || !E || !lsplit) {
        free(off); free(ldeg); free(tdeg); free(E); free(lsplit); return;
    }
    for (int i = 0; i < m; i++) {
        int u = src[i];
        tdeg[u]++;
        if (weight[i] <= delta) ldeg[u]++;
    }
    off[0] = 0;
    for (int u = 0; u < n; u++) off[u + 1] = off[u] + tdeg[u];
    for (int u = 0; u < n; u++) lsplit[u] = off[u] + ldeg[u];
    { int *cl = tdeg, *ch = ldeg;                    /* reuse the count arrays */
      for (int u = 0; u < n; u++) { cl[u] = off[u]; ch[u] = lsplit[u]; }
      for (int i = 0; i < m; i++) {
          int u = src[i]; double w = weight[i];
          int p = (w <= delta) ? cl[u]++ : ch[u]++;
          E[p].v = dst[i]; E[p].w = w; E[p].pad = 0;
      } }
    free(tdeg); free(ldeg);

    /* ---- REGIME: too many ducks for pegs?  Then hang none, walk the grid. ---- */
    if (n <= 96 || (double)m > 0.05 * (double)n * (double)n) {
        hand_walk(n, off, E, source, dist);
        free(off); free(lsplit); free(E);
        return;
    }

    unsigned char *restrict state = (unsigned char *)calloc((size_t)n, 1);
    int *restrict inb = (int *)calloc((size_t)n, sizeof(int));
    int *restrict R = (int *)malloc((size_t)n * sizeof(int));
    unsigned char *restrict inR = (unsigned char *)calloc((size_t)n, 1);
    int **bv = (int **)calloc((size_t)nb, sizeof(int *));
    int *bsz = (int *)calloc((size_t)nb, sizeof(int));
    int *bcap = (int *)calloc((size_t)nb, sizeof(int));
    if (!state || !inb || !R || !inR || !bv || !bsz || !bcap) {
        free(state); free(inb); free(R); free(inR); free(bv); free(bsz); free(bcap);
        hand_walk(n, off, E, source, dist);
        free(off); free(lsplit); free(E);
        return;
    }
    /* slack roads: a house with no far side is never pegged, only written */
    for (int u = 0; u < n; u++) if (off[u + 1] == off[u]) state[u] = 1;

    int *fr = NULL; int frsz = 0, frcap = 0;
    long long entries = 0, sweep = 0;
    const long long sweepLimit = 16LL * (long long)n + 64LL * (long long)nb + 4096LL;
    int cur = 0;

/* loose a duck: one divide, one store.  Nothing is compared to anything. */
#define DUCK_PUSH(v_, d_) do {                                              \
    int _i = (int)((d_) / delta);                                           \
    if (inb[(v_)] != _i + 1) {                                              \
        int _c = _i & mask;                                                 \
        if (bsz[_c] == bcap[_c]) {                                          \
            int _nc = bcap[_c] ? bcap[_c] * 2 : 32;                         \
            int *_p = (int *)realloc(bv[_c], (size_t)_nc * sizeof(int));    \
            if (!_p) break;                                                 \
            bv[_c] = _p; bcap[_c] = _nc;                                    \
        }                                                                   \
        bv[_c][bsz[_c]++] = (v_);                                           \
        inb[(v_)] = _i + 1;                                                 \
        entries++;                                                          \
    }                                                                       \
} while (0)

#define FLY(a_, b_, du_)                                                    \
    for (int e = (a_); e < (b_); e++) {                                     \
        int v = E[e].v;                                                     \
        double nd = (du_) + E[e].w;                                         \
        if (nd < dist[v]) {                                                 \
            unsigned char s = state[v];                                     \
            if (s < 2) { dist[v] = nd; if (s == 0) DUCK_PUSH(v, nd); }      \
        }                                                                   \
    }

    if (state[source] == 0) DUCK_PUSH(source, 0.0);

    while (entries > 0) {
        int c = cur & mask;
        while (bsz[c] == 0) {                        /* the hand moves forward only */
            cur++; sweep++;
            if (sweep > sweepLimit) goto escape;     /* pegs too fine: gather by hand */
            c = cur & mask;
        }
        int rsz = 0;
        const int tag = cur + 1;

        /* ---- wait for stillness on the leading peg-band ---- */
        for (;;) {
            int *tv = bv[c]; int tsz = bsz[c]; int tcap = bcap[c];
            bv[c] = fr; bsz[c] = 0; bcap[c] = frcap;     /* O(1) snapshot swap */
            fr = tv; frsz = tsz; frcap = tcap;
            entries -= tsz;
            if (frsz == 0) break;                        /* still */
            for (int k = 0; k < frsz; k++) {
                int u = fr[k];
                if (k + 1 < frsz) {                      /* next duck's landing */
                    int u2 = fr[k + 1];
                    __builtin_prefetch(&off[u2], 0, 1);
                    __builtin_prefetch(&dist[u2], 0, 1);
                }
                if (inb[u] != tag) continue;             /* a long cousin: stale */
                inb[u] = 0;
                if (state[u] >= 2) continue;
                if (!inR[u]) { inR[u] = 1; R[rsz++] = u; }
                double du = dist[u];
                FLY(off[u], lsplit[u], du)               /* short roads only */
            }
        }

        /* ---- black stones, final, never touched again ---- */
        for (int k = 0; k < rsz; k++) { state[R[k]] = 2; inR[R[k]] = 0; }
        /* ---- long roads: measured once, from the settled corners ---- */
        for (int k = 0; k < rsz; k++) {
            int u = R[k];
            double du = dist[u];
            FLY(lsplit[u], off[u + 1], du)
        }
        cur++;
    }
    goto cleanup;

escape:
    /* Every stone is final and every out-edge of every stoned corner has
       already been flown, so the live ducks are a valid frontier: gather them
       and finish by comparing in the hands. */
    {
        HeapItem *h = (HeapItem *)malloc(((size_t)n + (size_t)m + 2) * sizeof(HeapItem));
        if (h) {
            int hs = 0;
            for (int v = 0; v < n; v++)
                if (state[v] == 0 && dist[v] < INFINITY) hpush(h, &hs, dist[v], v);
            while (hs > 0) {
                HeapItem t = hpop(h, &hs);
                int u = t.u;
                if (state[u] >= 2) continue;
                if (t.d > dist[u]) continue;
                state[u] = 2;
                double du = dist[u];
                int b = off[u + 1];
                for (int e = off[u]; e < b; e++) {
                    int v = E[e].v;
                    double nd = du + E[e].w;
                    if (nd < dist[v]) {
                        unsigned char s = state[v];
                        if (s < 2) { dist[v] = nd; if (s == 0) hpush(h, &hs, nd, v); }
                    }
                }
            }
            free(h);
        }
    }

cleanup:
    for (int i = 0; i < nb; i++) free(bv[i]);
    free(fr); free(bv); free(bsz); free(bcap);
    free(state); free(inb); free(R); free(inR);
    free(off); free(lsplit); free(E);
#undef FLY
#undef DUCK_PUSH
}
```

## PREDICTION

**PREDICTION: speedup_vs_heap = 2.0**

(Stated before any measurement. Reasoning: the heap pays `O(log n)` dependent, cache-missing comparisons per improving relaxation; the peg ring pays one divide + one append, with the peg array (`nb` is typically 16–64 entries) resident in L1. Published sequential bucket-based SSSP is normally 1.5–3× a binary heap at moderate weight ranges. The dense branch should beat the heap by more than 2× when it fires; the sink rule shaves selection work on top. I weight the sparse regime most heavily and call it 2.0.)

## MEASUREMENT

**Not performed.** `dijkstra_bench` and `dijkstra_contract` are not available in this session — no tools were exposed, so I made zero measurements and zero of the four allowed improvement rounds. I will not dress up estimates as measurements. What follows is what I *believe* and why, clearly labelled as untested:

- Correctness argument (hand-checked, not machine-checked): the peg ring is sized so `wmax/Δ ≤ nb−4`, hence every live duck sits on a peg in `[cur, cur+nb−2]` and can never wrap behind the hand. Any duck loosed from a corner in band `cur` has value `≥ cur·Δ` (induction: processed corners in band `cur` have `dist ≥ cur·Δ`), so no duck ever lands on a passed peg. A *heavy* road (`w > Δ`) from a corner in band `cur` yields `dist+w > (cur+1)Δ`, so it cannot land inside band `cur` — which is exactly why deferring it to after stillness is safe. Therefore, when the band is still, every corner removed from it has its final distance. Sinks are never selected but always min-written, so their final value is correct at termination.
- Float agreement: the settled value is `dist[u]+w` summed along the same optimal path the reference sums, so bit-identical except where near-ties pick an equally-short alternative path — inside tolerance.
- Untested risk I am aware of: `Δ = wmax/(avgdeg+1)` is a heuristic. If it is badly wrong the failure is *slow, not wrong*, and the sweep counter escapes to the heap.

The honest summary is: I have a correctness proof I trust and a speed claim I have not earned yet.

## VERDICT

The core of this kernel is **SEED 2 translated literally**: a duck is a carried label, a thread is a count of centimetres, and a centimetre count is an *address*. That one substitution — ordering by address instead of by comparison — is the whole mechanism, and it is what breaks "a priority structure must be consulted before every relaxation." Relaxation here consults nothing: it divides, masks, and stores. The structure is only ever *interrogated* once per round, and then only to ask "which peg-band is leading," which the hand answers by moving forward and never back.

The other two native mechanisms survive intact rather than being paraphrased away: stones (`state=2`, written after stillness, never revisited), and slack roads (out-degree-0 corners never pegged — they receive threads and are never once compared against anything). The "long cousins" line is the lazy stale test, not decrease-key.

Where I deliberately did **not** invent: this mechanism lands on Δ-stepping, a 1998 result and the reference SSSP of Graph500/GAP. Step 4 says a validated known technique beats a novel untested one, and I would rather ship the native's idea *recognised* than the native's idea *dressed up as mine*.

Risks named, and what I did about each — none are left as caveats:
1. *"Pegs lose when there are more ducks than corners"* (dense/tiny graphs, where `known_way` says the plain `O(n²)` scan is the real win) → runtime check `n ≤ 96 || m > 0.05n²` routes to `hand_walk`, a pure hovering-hand scan. The native recognises his own regime.
2. *"Pegs lose when the centimetre is far too fine for the grid"* (huge weight dynamic range ⇒ long runs of empty pegs) → `sweep > 16n + 64·nb + 4096` gathers every live duck into a binary heap and finishes there, provably correct because stones are final and all their roads have been flown.
3. Thread parallelism: **declined.** The metaphor's unit of work is one peg-band, which at sparse benchmark sizes is a few hundred ducks — too small to pay for atomics on `dist`. I took the vectorization/cache route instead, per step 4: `restrict` throughout, roads interleaved into 16-byte objects so far-corner and centimetres arrive in one line, an `O(1)` pointer-swap snapshot instead of copying a band, a vectorizable min in the dense path, and a prefetch of the next duck's landing.

What would falsify the claim: if measurement shows speedup < 1, the likely culprit is the `Δ` heuristic causing excessive intra-band re-flights, and the fix stays inside the mechanism — retune `Δ`, not replace the pegs with a heap.