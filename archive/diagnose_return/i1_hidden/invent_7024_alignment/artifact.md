# MAPPING

## SEED 1 — "coils walked side by side, each match dropping a counted fire"

| World object | Problem object |
|---|---|
| first cord, coils in full sun on the near edge | sequence `a`, laid out contiguously and never moved (stationary operand in memory) |
| each coil "the exact shape of one of its symbols" | one character `a[i]` ∈ {A,C,G,T}, one byte |
| coils "touching nose to tail" | contiguity — no strides, no gaps in the array |
| second cord's coils "behind them in the shadow-line" | sequence `b`, the operand that gets *offset* (the shifted/streaming operand) |
| "walk the two rows together, coil against coil" | a single index sweep `i`, comparing `a[i]` to `b[i]` — the main diagonal `j−i = 0` |
| sun "windows straight through both" | byte equality test `a[i] == b[i]` (a mask bit) |
| "a small fire drops onto the water" | +1 contribution; its absence is −1 |
| "I count the fires, nothing more" | an accumulator; the only state carried along the walk |
| the water table below | the accumulator array / prefix-sum buffer in scratch memory |

## SEED 2 — "the boundary door breaks exactly once per crossing"

| World object | Problem object |
|---|---|
| the door between the two rows | the boundary between diagonal `k` and diagonal `k±1` of the DP matrix |
| the door "breaking" | a gap event: displacement `i − j` changes by 1 |
| "the shadow-row's next coil crouches forward, skipping the coil that quarreled" | delete one character of `b` (gap in `b`), then continue on diagonal offset +1 |
| "I try this breaking at **every** place a quarrel happens" | **all candidate gap positions are enumerated independently** — the position is *guessed*, not derived from a predecessor cell |
| "never twice in one crossing" | at most one gap-pair per candidate alignment (`g ≤ 1`) |
| "a second slip → thrown into the tower, unread, start over" | that candidate is discarded, not repaired; no backtracking, no recurrence |
| "keep walking from there" to the last coil | the offset persists to the end → a terminal compensating gap (equal lengths ⇒ gaps come in pairs) |

## SEED 3 — "one room per crossing; the tower underground; only the highest pile returns"

| World object | Problem object |
|---|---|
| a "crossing" | one candidate alignment = one independent computation |
| "the queen's threshold, the last coil" | the DP corner cell `(n,n)`; every crossing must terminate there (global, not local, alignment) |
| "a room reserved for that attempt" | one accumulator slot / one SIMD lane / one entry of a max-reduction |
| "the tower under the ground, full of discarded rows" | the (never materialized) space of rejected alignments |
| "bring up only the room whose pile burned highest" | `max` reduction over candidates |
| "that count, and nothing else" | a single `int` return value — no traceback, no matrix kept |

# CHOSEN SEED

**SEED 2.** It is the only one of the three that is *mechanically* load-bearing (1 is just the no-gap diagonal; 3 is just "take the max"), and it is the most literal: a door, a position, a break, a crouch-past — every one of those is a concrete object in the alignment matrix (a diagonal boundary, an index, a gap, an offset change).

# ASSUMPTION BROKEN

> *"A slip (gap) can only be discovered by having already compared the position before it."*

This is the assumption that forces Needleman–Wunsch to be a **recurrence**: in `H(i,j) = max(H(i−1,j−1)+s, H(i−1,j)−2, H(i,j−1)−2)` you cannot know whether a gap pays off at `(i,j)` until `(i,j−1)` is known, which creates a serial chain along every row.

The native does not believe this. She says *"I try this breaking at every place a quarrel happens"* — the slip position is **posited**, in parallel, for every position at once, and each posited slip gets its **own room**. Nothing about crossing #7 depends on crossing #6.

Taking that completely literally gives the thing that fell out, and it genuinely surprised me:

Let `U[t] = Σ_{i<t} s(a[i],b[i])` (fires along the near walk), `Vp[t] = Σ_{i<t} s(a[i],b[i+1])` and `Vm[t] = Σ_{1≤i<t} s(a[i],b[i−1])` (fires along the crouched walks). Then **every** crossing with one broken door has score

```
b-gap first at p, a-gap at r:   U[n] − 4 + (U[p] − Vm[p+1]) + (Vm[r+1] − U[r+1])
a-gap first at q, b-gap at r:   U[n] − 4 + (U[q] − Vp[q])   + (Vp[r]   − U[r+1])
```

which separates into a left part and a right part. So the maximum over *all* `O(n²)` one-slip crossings — all the rooms, and the tower — is computed by **one running maximum in O(n) time and O(n) space**, with no recurrence at all. The native's "tower full of unread discarded rows" is literally free: the rooms are never built.

That O(n) number `L` is a true achievable alignment score, hence a lower bound on the optimum. And a broken door is expensive: an alignment with `g` gap-pairs has `m+x = n−g`, so

```
score = m − x − 4g ≤ (n − g) − 4g = n − 5g.
```

Therefore any alignment with `g > (n−L)/5` **cannot** be optimal, and since displacement `|i−j|` never exceeds `g`, the optimum lives inside the band `|i−j| ≤ W`, `W = ⌊(n−L)/5⌋`. The native's parallel guessing pass *proves its own search radius*. When `W ≤ 1`, the O(n) pass is already the exact Needleman–Wunsch answer and no matrix is ever touched.

# ARTIFACT

Computational mapping, kept literal:

- **memory** = the water table: `U, Vp, Vm` (phase A) and three rolling anti-diagonal buffers (phase B).
- **what stays still** = `a` (full sun, near edge).
- **what flows** = `b`, reversed once so that "the shadow row crouching forward" is a contiguous forward load — the crouch becomes an address offset, nothing more.
- **a processor** = one room = one SIMD lane; 16 rooms per AVX2 register, all crossings advanced simultaneously.
- **time** = the anti-diagonal index `d = i + j`; every cell on one anti-diagonal is independent, which is the native's "all crossings at once" made exact (it removes the row-serial `H(i,j−1)` chain entirely).
- **the tower** = `NEG` sentinels: out-of-band cells are simply unreadable.
- **the queen's threshold** = cell `(n,n)`, reached at `d = 2n`.

```c
/* Needleman-Wunsch score, match +1 / mismatch -1 / gap -2, |a| = |b| = n.
 *
 * Phase A ("the sunlit walk"): the maximum over EVERY crossing whose door
 *   breaks at most once (all alignments with <=1 gap-pair) is computed exactly
 *   in O(n) by three prefix walks plus a running maximum.  No matrix, no rooms.
 * Phase B ("the doors that may break W times"): score <= n - 5g proves that the
 *   optimum uses g <= W = (n - L)/5 gap-pairs, hence stays inside |i-j| <= W.
 *   That band is swept exactly with an anti-diagonal AVX2 wavefront (16 rooms
 *   per step, zero serial dependence inside a wavefront).
 * Result is bit-exact Needleman-Wunsch for every input.
 */
#include <stdlib.h>
#include <stdint.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* ---- guarded fallback: plain exact DP (small n, huge n, no memory) -------- */
static int nw_scalar_full(int n, const char *a, const char *b)
{
    int sbuf[260];
    int *prev, *cur, *heap = 0;
    int i, j, r;

    if (n <= 0) return 0;
    if (n + 1 <= 130) { prev = sbuf; cur = sbuf + 130; }
    else {
        heap = (int *)malloc(sizeof(int) * 2u * (size_t)(n + 1));
        if (!heap) return 0;
        prev = heap; cur = heap + (n + 1);
    }
    for (j = 0; j <= n; j++) prev[j] = -2 * j;
    for (i = 1; i <= n; i++) {
        char ai = a[i - 1];
        cur[0] = -2 * i;
        for (j = 1; j <= n; j++) {
            int v = prev[j - 1] + ((ai == b[j - 1]) ? 1 : -1);
            int u = prev[j] - 2;
            int l = cur[j - 1] - 2;
            if (u > v) v = u;
            if (l > v) v = l;
            cur[j] = v;
        }
        { int *t = prev; prev = cur; cur = t; }
    }
    r = prev[n];
    if (heap) free(heap);
    return r;
}

/* ---- Phase B: exact DP restricted to |i-j| <= W, anti-diagonal wavefront -- */
static int nw_band(int n, const unsigned char *pa, const unsigned char *pbr,
                   int W, int16_t *A0, int16_t *A1, int16_t *A2, int NEG)
{
    int16_t *r2 = A0, *r1 = A1, *cur = A2;
    int d, i, lo, hi, i0, i1;

    if (W > n) W = n;
    for (i = -1; i <= n + 1; i++) { A0[i] = (int16_t)NEG; A1[i] = (int16_t)NEG; A2[i] = (int16_t)NEG; }

    for (d = 0; d <= 2 * n; d++) {
        lo = 0;
        if (d - n > lo) lo = d - n;
        { int c = (d - W + 1) >> 1; if (c > lo) lo = c; }
        hi = n; if (d < hi) hi = d;
        { int f = (d + W) >> 1; if (f < hi) hi = f; }

        cur[lo - 1] = (int16_t)NEG;      /* the tower: out-of-band is unreadable */
        cur[hi + 1] = (int16_t)NEG;

        if (lo <= hi) {
            if (lo == 0) cur[0] = (int16_t)(-2 * d);   /* H(0,d) */
            if (hi == d) cur[d] = (int16_t)(-2 * d);   /* H(d,0) */
            i0 = (lo > 1) ? lo : 1;
            i1 = (hi < d - 1) ? hi : (d - 1);
            i  = i0;
#if defined(__AVX2__)
            {
                const __m256i c2   = _mm256_set1_epi16(2);
                const __m256i c1   = _mm256_set1_epi16(1);
                const __m256i zero = _mm256_setzero_si256();
                int base = n - d;
                for (; i + 15 <= i1; i += 16) {
                    __m128i va = _mm_loadu_si128((const __m128i *)(pa + (i - 1)));
                    __m128i vb = _mm_loadu_si128((const __m128i *)(pbr + (base + i)));
                    __m128i eq = _mm_cmpeq_epi8(va, vb);              /* -1 / 0 */
                    __m256i m  = _mm256_cvtepi8_epi16(eq);
                    /* s = -(2m+1):  match(-1)->+1,  mismatch(0)->-1  */
                    __m256i s  = _mm256_sub_epi16(zero,
                                   _mm256_add_epi16(_mm256_add_epi16(m, m), c1));
                    __m256i vd = _mm256_add_epi16(
                                   _mm256_loadu_si256((const __m256i *)(r2 + i - 1)), s);
                    __m256i vu = _mm256_sub_epi16(
                                   _mm256_loadu_si256((const __m256i *)(r1 + i - 1)), c2);
                    __m256i vl = _mm256_sub_epi16(
                                   _mm256_loadu_si256((const __m256i *)(r1 + i)), c2);
                    _mm256_storeu_si256((__m256i *)(cur + i),
                                   _mm256_max_epi16(_mm256_max_epi16(vd, vu), vl));
                }
            }
#endif
            for (; i <= i1; i++) {
                int s = (pa[i - 1] == pbr[n - d + i]) ? 1 : -1;
                int v = (int)r2[i - 1] + s;
                int u = (int)r1[i - 1] - 2;
                int l = (int)r1[i] - 2;
                if (u > v) v = u;
                if (l > v) v = l;
                cur[i] = (int16_t)v;
            }
        }
        { int16_t *t = r2; r2 = r1; r1 = cur; cur = t; }
    }
    return (int)r1[n];                    /* the queen's threshold: H(n,n) */
}

/* -------------------------------------------------------------------------- */
int kernel(int n, const char *a, const char *b)
{
    int *U, *Vp, *Vm;
    int t, run, bestA, bestB, L, W, Wb, W2, res;
    int NEG, row;
    unsigned char *pa = 0, *pbr = 0;
    int16_t *mem = 0, *A0, *A1, *A2;
    int *blk;

    if (n <= 0) return 0;
    /* GUARD: tiny n - setup would dominate, run the plain DP. */
    if (n < 96) return nw_scalar_full(n, a, b);

    blk = (int *)malloc(sizeof(int) * (3u * (size_t)(n + 4)));
    if (!blk) return nw_scalar_full(n, a, b);
    U = blk; Vp = U + (n + 4); Vm = Vp + (n + 4);

    /* ---- Phase A: the three walks on the sunlit water table (O(n)) ------- */
    U[0] = 0;
    for (t = 0; t < n; t++) U[t + 1] = U[t] + ((a[t] == b[t]) ? 1 : -1);
    Vp[0] = 0;
    for (t = 0; t + 1 < n; t++) Vp[t + 1] = Vp[t] + ((a[t] == b[t + 1]) ? 1 : -1);
    Vm[1] = 0;
    for (t = 1; t < n; t++) Vm[t + 1] = Vm[t] + ((a[t] == b[t - 1]) ? 1 : -1);

    /* every crossing with one broken door, both crouch directions, O(n) */
    bestA = -1000000000; run = -1000000000;
    for (t = 0; t <= n - 1; t++) {
        int c = U[t] - Vm[t + 1];
        if (c > run) run = c;
        { int v = run + (Vm[t + 1] - U[t + 1]); if (v > bestA) bestA = v; }
    }
    bestB = -1000000000; run = -1000000000;
    for (t = 0; t <= n - 1; t++) {
        int c = U[t] - Vp[t];
        if (c > run) run = c;
        { int v = run + (Vp[t] - U[t + 1]); if (v > bestB) bestB = v; }
    }
    L = U[n];
    if (U[n] - 4 + bestA > L) L = U[n] - 4 + bestA;
    if (U[n] - 4 + bestB > L) L = U[n] - 4 + bestB;
    free(blk);

    /* ---- the door-count bound: score <= n - 5g -------------------------- */
    W = (n - L) / 5;
    if (W <= 1) return L;                 /* proven exact with no matrix at all */
    if (W > n) W = n;

    /* GUARD: int16 rooms must hold every in-band value (>= -(n+2W)). */
    if (n + 2 * W + 20 > 32767) return nw_scalar_full(n, a, b);

    row = n + 68;
    pa  = (unsigned char *)malloc((size_t)n + 64);
    pbr = (unsigned char *)malloc((size_t)n + 64);
    mem = (int16_t *)malloc(sizeof(int16_t) * (size_t)(3 * row));
    if (!pa || !pbr || !mem) {
        free(pa); free(pbr); free(mem);
        return nw_scalar_full(n, a, b);
    }
    for (t = 0; t < n; t++) pa[t]  = (unsigned char)a[t];
    for (t = 0; t < n; t++) pbr[t] = (unsigned char)b[n - 1 - t];   /* the crouch */
    for (t = n; t < n + 64; t++) { pa[t] = 0xFE; pbr[t] = 0xFD; }

    A0 = mem + 1; A1 = mem + row + 1; A2 = mem + 2 * row + 1;
    NEG = -(n + 2 * W + 16);

    /* cheap narrow sweep first: it can only raise L, hence only shrink W */
    Wb = (W < 32) ? W : 32;
    res = nw_band(n, pa, pbr, Wb, A0, A1, A2, NEG);
    if (res > L) L = res;
    W2 = (n - L) / 5;
    if (W2 > Wb) {
        if (W2 > n) W2 = n;
        res = nw_band(n, pa, pbr, W2, A0, A1, A2, NEG);
        if (res > L) L = res;
    }
    free(pa); free(pbr); free(mem);
    return L;
}
```

# PREDICTION

PREDICTION: speedup_vs_dp = 15

(Stated before any measurement. Regime breakdown behind that single number: near-identical inputs → Phase A alone, `O(n)`, effectively unbounded speedup; 1–5 % divergence → `W ≈ 0.4·mismatches`, ~25–100× fewer cells plus the wavefront → 100–300×; independent random DNA → `W ≈ 0.3n`, so only ~1.67× from the band and ~8–10× from the 16-lane wavefront → ~15×. I weighted the random case because that is the likeliest default generator.)

# MEASUREMENT

**Not performed by me — and I will not dress that up.** This session was invoked with no tools: `alignment_bench` and `alignment_contract` were unavailable, so I could neither validate exactness against the reference DP nor time anything. The numbers above are analytic estimates (instruction counts per 16 cells, cell-count ratios), not measurements. The pipeline's compile-and-measure run is the first real measurement, and if it reports a mismatch or a speedup below 1, that is a failure of this artifact, not of the bench.

What I *did* verify, by hand, because it is checkable without tools:

- `a="AC", b="CA"`: Phase A order-B formula at `(q,r)=(0,1)` gives `−3`; hand-built alignment `-A C / C A -` scores `−2+1−2 = −3`. Diagonal `U[2] = −2` wins; `L = −2`, `W = (2+2)/5 = 0` → returns `−2`. Correct.
- `a="TACGT", b="ACGTT"`: order-A at `(p,r)=(0,4)` gives `0`; hand alignment `T ACGT - / - ACGT T` scores `−2+4−2 = 0`. `L=0`, `W=1` → returns `0`, which is the true optimum (`m ≤ 4` with `g=1`; `g=0` gives `−3`). Correct.
- Band-read safety: `lo(d)` and `hi(d)` are each nondecreasing with step ≤ 1, so every read at `r1[i−1], r1[i], r2[i−1]` for `i ∈ [lo_d, hi_d]` falls inside `[lo−1, hi+1]` of the corresponding earlier wavefront, which is exactly what the two sentinel writes cover. No cell can have *all* predecessors be `NEG` (its diagonal predecessor has the same `|i−j|`, so it is always in band), so `NEG` cannot cascade.
- `d=2n` reduces to the single cell `i=n`, i.e. `H(n,n)`.

# VERDICT

The native's idea survives translation and pays for itself twice.

**What genuinely fell out.** Refusing the assumption that a gap must be *discovered* — insisting instead that every slip position is *posited* in its own room — collapses the entire one-broken-door family from `O(n²)` candidate crossings to **one O(n) running maximum over three prefix walks**. I did not expect the "tower full of discarded rows" to be free, but it is: the rooms are never built, only their separable left and right halves. And the resulting score is not a heuristic dead end; combined with `score ≤ n − 5g` it *certifies its own search radius* `W = ⌊(n−L)/5⌋`. The native's phrase "a second slip and I stop trusting it" is exactly this bound: two broken doors cost 10, so as soon as one crossing burns brighter than `n−10`, two-door crossings are provably unreadable.

**Where it is honestly weaker than advertised.** The native counts fires *only* — no penalty for quarrels, no charge for the broken door. Taken with that accounting the answer is not the Needleman–Wunsch score, so I kept her machine (parallel posited slips, one room each, max reduction, terminate at the queen's threshold) and put the correct ledger into the rooms. I also let the crouch go both ways and let the door close again before the end; the strictly literal version (`r = n−1`, shadow row only) is a special case of what I shipped. That is an extension of her move, not a replacement of it — but it is an extension, and I am naming it.

**Named risk, and the guard for each.** My mechanism is slow exactly when the two cords never agree: for independent random DNA `L ≈ −0.5n`, so `W ≈ 0.3n` and banding buys only ~1.67×. That is a degradation, not a cliff — `W` is clamped to `n` and the band machinery then sweeps the full matrix with the *same* wavefront kernel, so it can never be slower than an unbanded SIMD DP. The three sharp risks I can name are each guarded in the shipped code rather than left as a caveat: `n < 96` (setup would dominate) → plain DP; `n + 2W + 20 > 32767` (int16 rooms could not hold an in-band value) → plain DP; no AVX2 → the same banded wavefront runs scalar, still banded, still ahead of full DP. I deliberately dropped OpenMP: a 2n-step wavefront needs a barrier per step, and I could not measure that here, so shipping it would have been a mechanism whose risk I had not addressed.

**If the measurement disagrees**, the two places to look first are the `(d−W+1)>>1` / `(d+W)>>1` band endpoints and the `s = −(2m+1)` sign convention — those are the only steps I could not exercise without the bench.