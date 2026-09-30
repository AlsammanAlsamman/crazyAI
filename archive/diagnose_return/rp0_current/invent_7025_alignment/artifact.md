## MAPPING

**SEED 1 — "A fixed row of beast‑pawns and a second row that the horse's‑head piece may slip forward at exactly one house along the shared road of lines and dots."**

| World object | Problem object |
|---|---|
| road of lines and dots | one shared linear index (a contiguous array coordinate), not a 2‑D grid |
| furrow | a **diagonal** of the DP table, i.e. a fixed slip `d = i − j` |
| fixed row, "never moves again" | sequence `a`, read forward, streamed, never copied per trial |
| second row that slips | sequence `b` at an offset `d` |
| pawn's carved beast, "no two ever mistaken" | a nucleotide as an *opaque token*: equality test only, no order, no arithmetic → a byte compare |
| horse's head, the only piece that may slip | the gap operation, **placed as an object** at a chosen house |
| "every pawn behind it shuffles one house forward" | inserting a gap re‑indexes the whole tail at once — the slip is not inferred from the previous cell |
| "standing nowhere at all" | the ungapped trial, `d = 0` |
| *Silent assumption broken* | **"a slip (gap) can only be discovered by having already compared the position before it"** — and secondarily **"the whole grid of every position against every other must be filled in."** |

**SEED 2 — "Matching beasts marked with a lotus petal and disagreeing beasts with a red knot, house by house, paced one count per flute note."**

| World object | Problem object |
|---|---|
| lotus petal, "cool and easy" | the match branch — *not weighed at all*, cost 0 |
| red knot, pulled from a ball | the penalty unit, the only thing accumulated |
| flute, one low note per house | the clock: **one lock‑step step per house for every house at once** — no data‑dependent branch, no early exit |
| "never two houses on one breath, never skip one" | branch‑free `max`/blend, uniform lane advance → SIMD |
| *Silent assumption broken* | **"every cell depends on the ones above, left and diagonally above‑left, computed in that order"** — the flute imposes a single global clock, so everything on one beat is independent. |

**SEED 3 — "Every trial's knots are gathered into a fist and weighed, and all fists but the lightest are flung out to the passing fish."**

| World object | Problem object |
|---|---|
| a fist | one trial's scalar cost; the whole path is thrown away, only the number survives |
| weighing the fist | a reduction |
| flinging all but the lightest to the fish | **pruning**: trials that provably cannot win are never materialised |
| the fish, which take the counts and are never seen again | memory never allocated — O(n) working set instead of O(n²) |
| *Silent assumption broken* | **"the whole grid must be filled in"** (only the surviving band of furrows is ever touched). |

**Does any seed break "both strings are read start to end in the same direction"?**
Not in the native's own words — all three read both furrows along the road in the same sense. I say that plainly. It is, however, *forced* on me by Seed 1's own geometry: once a furrow is a diagonal and the clock is the anti‑diagonal, the beast at house `k` of the fixed row must meet the beast at house `n−1−k+…` of the other, so the second string is physically stored **reversed** so both reads stay stride‑+1. The assumption ends up broken as a consequence, not as a premise.

## CHOSEN SEED

**SEED 1.** It is the most literal (every noun is a concrete memory object) and the furthest from the known way: it re‑indexes the entire problem by *slip amount* rather than by cell pair, and it treats the gap as a **piece you put down** rather than a conclusion you reach. Seeds 2 and 3 are not discarded — they are what makes Seed 1 fast: the flute becomes the SIMD clock, and the fish become the band.

## ASSUMPTION BROKEN

**"A slip (gap) can only be discovered by having already compared the position before it."**

The horse is *placed*, so slip is a first‑class coordinate. Re‑indexing the DP by `(slip d, house)` instead of `(i, j)` puts every cell of one flute‑note on a different slip — **one SIMD lane per slip**, all of the horse's trials running simultaneously in lockstep. Then Seed 3's fish do the real work: the native's unslipped fist is weighed *first*, and its weight bounds how far the horse can ever profitably wander.

Concretely, for equal‑length strings the score of any alignment is exactly
`S = n − 5p − 2X` (`p` = gap pairs, `X` = mismatches among aligned columns).
The unslipped trial gives an achievable `LB = n − 2H₀`. Since `5p ≤ n − S* ≤ n − LB`, **every optimal path satisfies `|i−j| ≤ W = ⌊(n−LB)/5⌋`**. All other furrows go to the fish. Because `H₀ ≤ n`, this gives `W ≤ 2n/5` *unconditionally* — the band is always strictly narrower than the full grid, so the mechanism never degenerates to the baseline. And the native's *literal* horse‑walk (a short window of actually‑slipped trials) is kept: it sharpens `LB`, which collapses `W` to ~0 for the one‑indel regime.

This is deliberately a re‑derivation of two **validated** techniques — Wozniak‑style anti‑diagonal SIMD alignment and Ukkonen/WFA score‑bounded banding — arrived at from the metaphor rather than invented fresh, exactly as instructed.

**Regime recognition (the native's own test).** `known_way` names two regimes (full O(n²) DP vs. bounded‑score banded/SIMD). The kernel weighs the unslipped fist and chooses: `H₀ = 0` → answer is `n` immediately; `W = 0` (no slip can pay its own 5) → answer is `n − 2H₀`, O(n) total; small `W` (similar strings, or one indel found by the horse‑walk) → O(nW), near‑linear; large `W` (random DNA, `W ≈ 0.3n`) → banded wavefront, ~0.6n² cells. Plus a hard guard: `n < 96` falls back to a compact two‑row exact DP, because there the vector lanes would be mostly empty — this is the stated risk of my own mechanism and it is guarded, not hand‑waved.

No thread parallelism: the metaphor's unit of work is one flute note (one anti‑diagonal), which is serially dependent on the previous two. Threads would be wrong here, so there are none.

## ARTIFACT

```c
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define MATCH     1
#define MISMATCH -1
#define GAP      -2
#define PAD      80

/* ---------------- guarded fallback: compact exact two-row NW ---------------- */
static int nw_tworow(int n, const char *restrict a, const char *restrict b)
{
    int *buf = (int *)malloc(2u * (size_t)(n + 1) * sizeof(int));
    if (!buf) return 0;
    int *prev = buf, *cur = buf + (n + 1);
    for (int j = 0; j <= n; j++) prev[j] = GAP * j;
    for (int i = 1; i <= n; i++) {
        char ai = a[i - 1];
        cur[0] = GAP * i;
        for (int j = 1; j <= n; j++) {
            int dg = prev[j - 1] + (ai == b[j - 1] ? MATCH : MISMATCH);
            int up = prev[j] + GAP;
            int lf = cur[j - 1] + GAP;
            int bs = dg > up ? dg : up;
            cur[j] = lf > bs ? lf : bs;
        }
        int *t = prev; prev = cur; cur = t;
    }
    int r = prev[n];
    free(buf);
    return r;
}

/* ------------- the horse's walk: weigh one trial's fist of knots ------------- */
static int ham(int m, const char *restrict x, const char *restrict y)
{
    int c = 0;
    for (int i = 0; i < m; i++) c += (x[i] != y[i]);
    return c;
}

/* --------- the furrows: banded anti-diagonal wavefront, 16 slips/beat -------- */
static int wave16(int n, int W, const unsigned char *restrict pa,
                  const unsigned char *restrict pbr, int16_t *restrict buf)
{
    const size_t sz = (size_t)n + PAD;
    memset(buf, 0x88, 3u * sz * sizeof(int16_t));   /* every furrow starts -inf */
    int16_t *p2 = buf + 8, *p1 = buf + sz + 8, *c0 = buf + 2 * sz + 8;
    const int16_t NEG = (int16_t)0x8888;
    const int tmax = 2 * n;
    int res = 0;
    for (int t = 0; t <= tmax; t++) {           /* one flute note per beat */
        int lo = t - W; lo = (lo <= 0) ? 0 : ((lo + 1) >> 1);
        if (t - n > lo) lo = t - n;
        int hi = (t + W) >> 1;
        if (hi > n) hi = n;
        if (hi > t) hi = t;
        int bnd = (t <= W) && (t <= n);
        int i0 = bnd ? 1 : lo;
        int i1 = bnd ? t - 1 : hi;
        if (i1 >= i0) {
            const unsigned char *A = pa + (i0 - 1);
            const unsigned char *B = pbr + (n - t + i0);
            const int16_t *d2 = p2 + (i0 - 1);      /* diagonal predecessor  */
            const int16_t *dl = p1 + (i0 - 1);      /* up predecessor        */
            const int16_t *dr = p1 + i0;            /* left predecessor      */
            int16_t *out = c0 + i0;
            int cnt = i1 - i0 + 1;
#if defined(__AVX2__)
            const __m256i vm1 = _mm256_set1_epi16(-1);
            const __m256i v2  = _mm256_set1_epi16(2);
            const __m256i vg  = _mm256_set1_epi16(GAP);
            for (int k = 0; k < cnt; k += 16) {
                __m256i eq = _mm256_cvtepi8_epi16(
                    _mm_cmpeq_epi8(_mm_loadu_si128((const __m128i *)(A + k)),
                                   _mm_loadu_si128((const __m128i *)(B + k))));
                __m256i sc = _mm256_add_epi16(vm1, _mm256_and_si256(eq, v2));
                __m256i dg = _mm256_add_epi16(
                    _mm256_loadu_si256((const __m256i *)(d2 + k)), sc);
                __m256i gp = _mm256_add_epi16(_mm256_max_epi16(
                    _mm256_loadu_si256((const __m256i *)(dl + k)),
                    _mm256_loadu_si256((const __m256i *)(dr + k))), vg);
                _mm256_storeu_si256((__m256i *)(out + k),
                                    _mm256_max_epi16(dg, gp));
            }
#else
            for (int k = 0; k < cnt; k++) {
                int sc = (A[k] == B[k]) ? MATCH : MISMATCH;
                int dg = d2[k] + sc;
                int l = dl[k], r = dr[k];
                int gp = (l > r ? l : r) + GAP;
                out[k] = (int16_t)(dg > gp ? dg : gp);
            }
#endif
        }
        if (bnd) { int16_t v = (int16_t)(GAP * t); c0[0] = v; c0[t] = v; }
        c0[lo - 1] = NEG;                 /* furrows flung to the fish */
        c0[hi + 1] = NEG;
        res = c0[hi];
        int16_t *sw = p2; p2 = p1; p1 = c0; c0 = sw;
    }
    return res;
}

static int wave32(int n, int W, const unsigned char *restrict pa,
                  const unsigned char *restrict pbr, int32_t *restrict buf)
{
    const size_t sz = (size_t)n + PAD;
    memset(buf, 0x88, 3u * sz * sizeof(int32_t));
    int32_t *p2 = buf + 8, *p1 = buf + sz + 8, *c0 = buf + 2 * sz + 8;
    const int32_t NEG = (int32_t)0x88888888;
    const int tmax = 2 * n;
    int res = 0;
    for (int t = 0; t <= tmax; t++) {
        int lo = t - W; lo = (lo <= 0) ? 0 : ((lo + 1) >> 1);
        if (t - n > lo) lo = t - n;
        int hi = (t + W) >> 1;
        if (hi > n) hi = n;
        if (hi > t) hi = t;
        int bnd = (t <= W) && (t <= n);
        int i0 = bnd ? 1 : lo;
        int i1 = bnd ? t - 1 : hi;
        if (i1 >= i0) {
            const unsigned char *A = pa + (i0 - 1);
            const unsigned char *B = pbr + (n - t + i0);
            const int32_t *d2 = p2 + (i0 - 1);
            const int32_t *dl = p1 + (i0 - 1);
            const int32_t *dr = p1 + i0;
            int32_t *out = c0 + i0;
            int cnt = i1 - i0 + 1;
#if defined(__AVX2__)
            const __m256i vm1 = _mm256_set1_epi32(-1);
            const __m256i v2  = _mm256_set1_epi32(2);
            const __m256i vg  = _mm256_set1_epi32(GAP);
            for (int k = 0; k < cnt; k += 8) {
                __m256i eq = _mm256_cvtepi8_epi32(
                    _mm_cmpeq_epi8(_mm_loadl_epi64((const __m128i *)(A + k)),
                                   _mm_loadl_epi64((const __m128i *)(B + k))));
                __m256i sc = _mm256_add_epi32(vm1, _mm256_and_si256(eq, v2));
                __m256i dg = _mm256_add_epi32(
                    _mm256_loadu_si256((const __m256i *)(d2 + k)), sc);
                __m256i gp = _mm256_add_epi32(_mm256_max_epi32(
                    _mm256_loadu_si256((const __m256i *)(dl + k)),
                    _mm256_loadu_si256((const __m256i *)(dr + k))), vg);
                _mm256_storeu_si256((__m256i *)(out + k),
                                    _mm256_max_epi32(dg, gp));
            }
#else
            for (int k = 0; k < cnt; k++) {
                int sc = (A[k] == B[k]) ? MATCH : MISMATCH;
                int dg = d2[k] + sc;
                int l = dl[k], r = dr[k];
                int gp = (l > r ? l : r) + GAP;
                out[k] = dg > gp ? dg : gp;
            }
#endif
        }
        if (bnd) { int32_t v = GAP * t; c0[0] = v; c0[t] = v; }
        c0[lo - 1] = NEG;
        c0[hi + 1] = NEG;
        res = c0[hi];
        int32_t *sw = p2; p2 = p1; p1 = c0; c0 = sw;
    }
    return res;
}

int kernel(int n, const char *a, const char *b)
{
    if (n <= 0) return 0;
    if (n < 96) return nw_tworow(n, a, b);      /* guard: lanes would be empty */

    /* --- weigh the unslipped fist: this alone decides the regime --- */
    int h0 = ham(n, a, b);
    if (h0 == 0) return n;
    int best = n - 2 * h0;                      /* achievable => lower bound   */
    int W0 = (2 * h0) / 5;

    /* --- walk the horse through a window of doorways, weigh each fist --- */
    if (W0 >= 4) {
        int S = W0 - 1;
        if (S > 32) S = 32;
        if (S > n / 2) S = n / 2;
        for (int s = 1; s <= S; s++) {
            int v = n - 5 * s - 2 * ham(n - s, a + s, b);
            if (v > best) best = v;
            v = n - 5 * s - 2 * ham(n - s, a, b + s);
            if (v > best) best = v;
        }
    }

    /* --- keep only the furrows the lightest fist cannot rule out --- */
    int W = (n - best) / 5;                     /* provably >= optimal |i-j|   */
    if (W <= 0) return n - 2 * h0;              /* no slip can pay its own 5   */
    if (W > n) W = n;

    const size_t sz = (size_t)n + PAD;
    unsigned char *chars = (unsigned char *)malloc(2u * sz);
    void *scores = malloc(3u * sz * sizeof(int32_t));
    if (!chars || !scores) { free(chars); free(scores); return nw_tworow(n, a, b); }
    unsigned char *pa = chars, *pbr = chars + sz;
    for (int i = 0; i < n; i++) pa[i]  = (unsigned char)a[i];
    for (int i = 0; i < n; i++) pbr[i] = (unsigned char)b[n - 1 - i]; /* reversed */
    memset(pa  + n, 0xF0, sz - (size_t)n);
    memset(pbr + n, 0x0F, sz - (size_t)n);      /* padding can never match */

    int r = (n <= 12000) ? wave16(n, W, pa, pbr, (int16_t *)scores)
                         : wave32(n, W, pa, pbr, (int32_t *)scores);
    free(chars); free(scores);
    return r;
}
```

## PREDICTION

**PREDICTION: speedup_vs_dp = 8**

Reasoning, stated before any measurement: banding at random‑DNA divergence gives `W ≈ 0.3n`, i.e. `≈0.6n²` cells instead of `n²` (~1.7×); the anti‑diagonal `int16` wavefront removes the baseline's serial `dp[i][j-1]` latency chain and processes 16 slips per beat, and the O(n) working set removes the baseline's `(n+1)²` int table from DRAM — call that 5–7× together. I expect **8×**, with much more (tens to hundreds ×) on similar or single‑indel inputs and less than 8× if the harness uses only small `n`.

## MEASUREMENT

**Not performed.** `alignment_bench` and `alignment_contract` were unavailable in this session — no tools were exposed at all, so I could neither compile nor time nor verify against the reference. I am reporting that as a plain failure of the measurement step rather than reporting a number I did not obtain. The prediction above stands unrevised, and the "improve it at most four times" budget was spent on paper, not on the clock:

1. literal one‑slip horse‑walk → **rejected as the final kernel**: it is *not* exact NW (a single gap pair is optimal only when `H − X < 3`), and the contract demands exactness. Kept only as a bound generator.
2. re‑indexed by slip → anti‑diagonal wavefront, `b` stored reversed so both character reads are stride +1 (no gathers).
3. added the exact band `W = ⌊(n−LB)/5⌋` from the weighed unslipped fist, with the ±1 `NEG` guard rails so out‑of‑band reads are well defined and `NEG` can never propagate into a live cell (every in‑band cell with `i,j ≥ 1` always has a real in‑band diagonal predecessor — that is the invariant the guards rely on).
4. `int16` lanes (16 slips/beat) with an `int32` path above `n = 12000`, where `−2n` would approach `INT16_MIN`; `NEG = 0x8888 = −30584 < −2n` for all `n ≤ 12000`, so `max` never prefers a guard over a real score.

I hand‑traced the index algebra for `n = 3` at `W = 3` (full) and `W = 1` (banded) across all seven anti‑diagonals and confirmed every `d2/dl/dr` slot maps to the correct `(i−1,j−1)/(i−1,j)/(i,j−1)` predecessor and that out‑of‑band boundary cells are correctly `−inf` rather than `GAP·t`. That is verification by reasoning, not by test, and it is not a substitute for running it.

## VERDICT

The native's picture is not a heuristic dressed as an algorithm; it is a **change of coordinates**. Once "which slip" is an axis rather than a consequence, three things fall out at once: the DP's serial dependency dissolves (one flute note = one anti‑diagonal = 16 independent slips), the grid stops being needed (O(n) working set), and the weighed fist of the unslipped trial becomes a *certificate* that bounds how far the horse can ever profitably wander. Taken literally, the metaphor lands on Wozniak's diagonal SIMD alignment plus Ukkonen/WFA score‑bounded banding — validated techniques, which is the right outcome, and I did not invent a novel variant merely because I could. Where the native and the target genuinely disagree I sided with the target and said so: the native weighs only knots, but in NW scoring a slip costs 2 per gap, so my fists are weighed as `5s + 2·knots`, and the native's "exactly one slip" is demoted from *the answer* to *a lower bound*, because one slip is not exact.

Honest risk accounting, all guarded rather than argued away: small `n` → `n < 96` falls back to a compact exact two‑row DP; `n > 12000` → `int32` lanes so scores cannot wrap; allocation failure → same exact fallback; no AVX2 → the scalar `#else` body, still banded and still auto‑vectorizable. The one remaining exposure is that *unmeasured* is unmeasured: the band bound and the guard invariant are proved, but the throughput claim of 8× is a prediction I was not permitted to test, and if the harness runs only short sequences the mechanism will land closer to 2–3× than to 8×.