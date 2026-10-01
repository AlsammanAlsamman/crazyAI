## MAPPING

| World object | Computational object | Which silent assumption it breaks |
|---|---|---|
| **SEED 1** — the low sun, one stroke, one doubled shadow per same-dye pair | one `cmpeq_epi8` over 16/32 contiguous bytes of the two ropes → 16 match/mismatch verdicts per instruction; no per-cell character test ever executed | **"one pair of positions is judged at a time"** |
| rope on the east wall / rope at a **slant** across the floor | `i` axis and the anti-diagonal orientation `k = i+j`; the slant is why the sun lights a whole wavefront at once | — |
| shadow must fall on *contiguous* knots of both ropes | `a` kept forward, `b` stored **reversed** (`BR[t]=b[n-1-t]`): lane `t` needs `a[i-1+t]` and `BR[n-k+i+t]`, both ascending ⇒ two plain unaligned loads, no shuffle | — |
| **SEED 2** — never lay a marker on the bare floor; its cost is read off how many knots forward each rope must go | corridor cost in closed form: from a treasure to the next, `cost(di,dj) = 3·min(di,dj) − 2(di+dj)`. Applied to the whole floor it says: an alignment with `g` gaps per side scores `≤ n − 5g`. So a path that ever slips by `d` scores `≤ n − 5d`. Read `L` off the ropes ⇒ **the floor outside `|i−j| ≤ (n−L)/5` is never marked at all** | **"the whole grid of every position against every other must be filled in"** |
| **SEED 3** — keep only the cheapest tally, drop the bad thread-end | `_mm256_max_epi16(dg, max(up,lf))` — branchless, 16 tallies dropped per instruction, no back-pointers, no traceback | **"a slip can only be discovered by having already compared the position before it"** (the max is taken on whole wavefronts, not cell-by-cell in a fixed above/left/diag order) |
| "allowing the ropes to slip past each other **once**" | the cheap probe pass: an exact DP restricted to slip ≤ 16 knots. Its score is a *feasible* alignment ⇒ a lower bound `S`, which is what the bare floor's arithmetic then converts into the real slip allowance | — |
| what is memory | 3 anti-diagonal buffers (`k`, `k−1`, `k−2`), indexed by absolute `i`; working set ≈ 6·w bytes, slides one half-step per wavefront |
| what flows | the wavefront (one shadow-line) |
| what stays still | the two ropes (`A`, `BR`) — read-only, read linearly |
| what is a processor | one AVX2 lane = one crossing |
| what is time | `k = i+j`, the hour of the sun |

## CHOSEN SEED

**SEED 2** ("the bare floor's cost is read straight off how many knots forward each rope must go, never walked or marked knot by knot") — it is the only one of the three that breaks the preferred assumption, so I take it as the core, with SEED 1 and SEED 3 as the mechanism that executes inside the region SEED 2 leaves.

Honest note on literalness: the *fully* literal SEED 2 is sparse DP over lit tiles only — chaining the treasures with the closed-form corridor cost between them. I derived that and it is exactly correct (the optimum equals the best increasing chain of matches with corridor costs between), **but the native's premise "there are few of them" is false for a 4-letter alphabet**: ¼ of all crossings are lit, ≈ n²/4 treasures, and chaining them is O(m²) — catastrophically worse than the known way. So I keep SEED 2's *arithmetic* (the corridor cost is known in closed form, so the floor need never be walked) and apply it at the coarsest scale, where it is a win instead of a loss: it bounds how far the ropes can slip, and everything outside that is never marked. That is Fickett/Ukkonen score-bounded banding — a validated real technique, which per instruction 4 is what the mechanism should land on rather than a novel untested chain.

## ASSUMPTION BROKEN

"The whole grid of every position against every other must be filled in." Cells with `|i−j| > (n − L)/5` are proven unreachable by any optimal path and are never allocated, touched, or computed. Secondarily: "one pair of positions is judged at a time" (the sun) and the fixed above/left/diag order (the slant makes all three dependencies land in already-finished wavefronts).

Two regimes, recognised at runtime *by the metaphor itself*: the sun's stroke down the main diagonal gives `L₀` in O(n); a bounded-slip probe sharpens it to `S`; `need = (n−S)/5` then says which regime we are in — similar ropes ⇒ the probe's answer is already provably optimal and we stop (O(n·w)); divergent ropes ⇒ one wide wavefront pass. Guards: no AVX2, `n<64`, `n>11000`, `n+2w>20000` (int16 headroom), or `w<32` ⇒ scalar banded fallback.

## ARTIFACT

PREDICTION: speedup_vs_dp = 14

```c
#include <stdlib.h>
#include <string.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define KM   1
#define KX  (-1)
#define KG  (-2)

/* ---- the low sun: one stroke, every same-dye pair on the line at once --- */
static int sun_count_diag(int n, const char *a, const char *b)
{
    int i = 0, c = 0;
#if defined(__AVX2__)
    for (; i + 32 <= n; i += 32) {
        __m256i va = _mm256_loadu_si256((const __m256i *)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(b + i));
        c += __builtin_popcount(
                (unsigned)_mm256_movemask_epi8(_mm256_cmpeq_epi8(va, vb)));
    }
#endif
    for (; i < n; i++) c += (a[i] == b[i]);
    return c;
}

/* --- exact DP over the floor the ropes may reach when slip <= w (scalar) - */
static int band_scalar(int n, const char *a, const char *b, int w)
{
    const int NEG = -(1 << 28);
    int *mem = (int *)malloc((size_t)2 * (n + 2) * sizeof(int));
    int *prev, *cur, i, j, jlo, jhi;
    if (!mem) return 0;
    prev = mem; cur = mem + (n + 2);
    for (j = 0; j <= n + 1; j++) { prev[j] = NEG; cur[j] = NEG; }
    jhi = (w < n) ? w : n;
    for (j = 0; j <= jhi; j++) prev[j] = -2 * j;
    prev[jhi + 1] = NEG;
    for (i = 1; i <= n; i++) {
        int left; char ai = a[i - 1];
        jlo = i - w; if (jlo < 0) jlo = 0;
        jhi = i + w; if (jhi > n) jhi = n;
        j = jlo;
        if (jlo == 0) { cur[0] = -2 * i; left = cur[0]; j = 1; }
        else          { cur[jlo - 1] = NEG; left = NEG; }
        for (; j <= jhi; j++) {
            int d = prev[j - 1] + (ai == b[j - 1] ? KM : KX);
            int u = prev[j] + KG;
            int l = left + KG;
            int best = d; if (u > best) best = u; if (l > best) best = l;
            cur[j] = best; left = best;
        }
        cur[jhi + 1] = NEG;
        { int *t = prev; prev = cur; cur = t; }
    }
    i = prev[n];
    free(mem);
    return i;
}

#if defined(__AVX2__)
/* ---- the shadow sweeping the slant: one wavefront per hour of the sun --- */
static int band_wave(int n, const char *a, const char *b, int w)
{
    const short NEG = -25000;
    const int pad = 64, dlen = n + 1 + 2 * pad;
    unsigned char *blk = (unsigned char *)malloc(
            (size_t)3 * dlen * sizeof(short) + (size_t)2 * (n + 64) + 64);
    short *base, *D0, *D1, *D2;
    char *A, *BR;
    int k, t, r;
    __m256i vP1, vM1, vG;
    if (!blk) return band_scalar(n, a, b, w);
    base = (short *)blk;
    D0 = base + pad; D1 = base + dlen + pad; D2 = base + 2 * dlen + pad;
    A  = (char *)(base + 3 * dlen);
    BR = A + (n + 32);
    memcpy(A, a, (size_t)n);  memset(A + n, 0, 32);
    for (t = 0; t < n; t++) BR[t] = b[n - 1 - t];
    memset(BR + n, 1, 32);
    for (t = 0; t < 3 * dlen; t++) base[t] = NEG;
    vP1 = _mm256_set1_epi16(1);
    vM1 = _mm256_set1_epi16(-1);
    vG  = _mm256_set1_epi16(2);

    for (k = 0; k <= 2 * n; k++) {
        int lo = k - n, hi, ilo, ihi, i, m;
        if (lo < 0) lo = 0;
        m = k - w;
        if (m > 0) { int c = (m + 1) >> 1; if (c > lo) lo = c; }
        hi = (k + w) >> 1;
        m = (k < n) ? k : n; if (hi > m) hi = m;
        ilo = (lo > 1) ? lo : 1;
        ihi = (hi < k - 1) ? hi : k - 1;
        for (i = ilo; i <= ihi; i += 16) {
            __m128i va = _mm_loadu_si128((const __m128i *)(A + (i - 1)));
            __m128i vb = _mm_loadu_si128((const __m128i *)(BR + (n - k + i)));
            __m256i msk = _mm256_cvtepi8_epi16(_mm_cmpeq_epi8(va, vb));
            __m256i sc  = _mm256_blendv_epi8(vM1, vP1, msk);
            __m256i dg  = _mm256_adds_epi16(
                    _mm256_loadu_si256((const __m256i *)(D2 + i - 1)), sc);
            __m256i up  = _mm256_subs_epi16(
                    _mm256_loadu_si256((const __m256i *)(D1 + i - 1)), vG);
            __m256i lf  = _mm256_subs_epi16(
                    _mm256_loadu_si256((const __m256i *)(D1 + i)), vG);
            __m256i bs  = _mm256_max_epi16(dg, _mm256_max_epi16(up, lf));
            _mm256_storeu_si256((__m256i *)(D0 + i), bs);
        }
        D0[lo - 1] = NEG;
        D0[hi + 1] = NEG;
        if (lo == 0) D0[0] = (short)(-2 * k);
        if (hi == k) D0[k] = (short)(-2 * k);
        { short *p = D2; D2 = D1; D1 = D0; D0 = p; }
    }
    r = (int)D1[n];
    free(blk);
    return r;
}
#endif

int kernel(int n, const char *a, const char *b)
{
    int lit, L0, W0, w, S, need;
    if (n <= 0) return 0;
    if (n < 24) return band_scalar(n, a, b, n);

    /* SEED 1: one stroke of the sun down the wall-rope and the floor-rope. */
    lit = sun_count_diag(n, a, b);
    L0  = 2 * lit - n;

    /* SEED 2: the corridor's cost was never a secret, only its length.
       any alignment with g gaps per side scores <= n - 5g, so a path that
       ever slips by d scores <= n - 5d; slip beyond (n-L)/5 cannot win. */
    W0 = (n - L0) / 5;
    if (W0 > n) W0 = n;

    /* one slip, of bounded size: exact, cheap, and a valid lower bound. */
    w = (W0 < 16) ? W0 : 16;
    S = band_scalar(n, a, b, w);
    need = (n - S) / 5;
    if (need <= w) return S;              /* provably optimal already */

    w = need; if (w > n) w = n;

#if defined(__AVX2__)
    if (n >= 64 && n <= 11000 && w >= 32 && (n + 2 * w) <= 20000)
        return band_wave(n, a, b, w);     /* SEED 1 + SEED 3, 16 lanes */
#endif
    return band_scalar(n, a, b, w);
}
```

**Why it is exact.** Let `S` be any achievable score (any banded run returns one). An alignment of two length-`n` strings has `p` aligned columns and `g = n − p` gap columns on *each* side, so its score is `≤ p − 2·(2g) = n − 5g`; and reaching deviation `d = |i−j|` requires `g ≥ d`. Hence a path touching `|i−j| ≥ w+1` scores `≤ n − 5(w+1)`, which is `< S` exactly when `floor((n−S)/5) ≤ w`. That is the return test. The second pass is unconditional-optimal because `S₂ ≥ S₁ ⇒ need₂ ≤ need₁ = w`, so at most **two** passes ever run. Out-of-band cells are poisoned to `−25000`; real values are `≥ −(n+2w) ≥ −20000` by the guard, and all arithmetic is saturating, so poison can never win a `max` and garbage in the vector spill lanes (which is fenced off by the poison cell at `hi+1` and never read by an in-band cell) cannot overflow.

I hand-traced the wavefront on `n=2, a="AC", b="CA"` through all five hours `k=0..4` and it reproduces `dp[1][1]=−1`, `dp[1][2]=dp[2][1]=−1`, `dp[2][2]=−2`, the reference answer.

## PREDICTION

PREDICTION: speedup_vs_dp = 14

Reasoning, stated before measuring: for random DNA `L₀ ≈ −0.5n`, so the ropes report a slip allowance `w ≈ 0.29n` and the marked floor is `≈ 0.58 n²` instead of `n²` (**≈1.7×** from SEED 2). The wavefront body is ~13 µops per 16 crossings ⇒ ≈0.2 cycles/cell against ≈2–3 cycles/cell for the serial `left`-dependent reference, which additionally writes a full `(n+1)²` int table (67 MB at n=4096) (**≈8×** from SEED 1 + SEED 3). 1.7 × 8 ≈ 14. On similar sequences the probe alone returns and the speedup should be 100×–1000×; on random sequences 14× is the honest number.

## MEASUREMENT

**Not performed.** `alignment_bench` and `alignment_contract` were unavailable in this session (no tools exposed), so I did not measure anything and I am not going to dress up an estimate as a measurement. What I *did* verify by hand: the corridor-cost algebra, the band-optimality test, the two-pass termination argument, the int16 headroom/poison argument, the padding bounds on every load and store, and a full five-hour trace of the wavefront on a 2×2 case. What remains unmeasured and could falsify the prediction:

1. the true `cycles/cell` of the reference on the bench machine (if the 67 MB table makes it memory-bound, my speedup goes **up**; if gcc does better than 2 cyc/cell, it goes **down**);
2. the benchmark's sequence similarity, which sets `w` and therefore swings the SEED-2 factor between 1.0× and ~n/16;
3. whether `n` exceeds 11000, in which case the guard drops to the scalar banded path and only the 1.7× banding factor survives.

## VERDICT

The core of the kernel is the native's mechanism, not the textbook one: the floor outside the slip allowance is **never filled in**, and the allowance itself is *read off the ropes* by the sun's stroke plus one bounded-slip probe — never discovered by walking the corridor. Inside the allowance, the sun lights 16 crossings of the slant per instruction and 16 tallies are dropped per instruction.

Where I broke with the native: his "few treasures in a mostly empty corridor" is simply wrong for a 4-letter alphabet (¼ of crossings are lit), so the literal sparse chain over treasures would be O(n⁴/16) and I refused to ship it. I kept his arithmetic and moved it up one scale. I also declined the second half of his picture — the slip "spent or not yet spent" bookkeeping at each treasure — because it models at most one gap block, which does not match the reference score in general; instead the bounded slip appears as the probe band, which is exact for what it computes.

Named risks and how each is addressed rather than hoped away:
- *int16 could overflow for large or highly divergent input* → guarded by `n ≤ 11000 && n + 2w ≤ 20000`, fallback to the int32 scalar banded path.
- *SIMD overhead dominates for a narrow band* → guarded by `w ≥ 32`; narrower bands go scalar, where `O(n·w)` is already tiny.
- *no AVX2 / tiny n* → guarded, scalar path, still correct and still banded.
- *banding could be worthless* → in the worst case `w = need` and the result is exactly the reference score computed over at most the full grid; the mechanism degrades to the known way, never below it.
- *thread parallelism* → deliberately omitted. Anti-diagonals are serially dependent, the metaphor's unit of work (one hour of sun) is ~w/16 vector ops, far too small at these sizes to pay for a barrier; adding OpenMP here would be overhead with no fallback story.

Unresolved, and the honest weak point: with no measurement I cannot tell you whether 14 is right or whether the reference's 67 MB table makes the real number 30. The prediction stands as written and should be judged against whatever the pipeline reports.