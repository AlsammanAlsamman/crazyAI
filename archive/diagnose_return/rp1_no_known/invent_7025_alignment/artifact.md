# MAPPING

## SEED 1 — "A fixed row of beast-pawns and a second row that the horse's-head piece may slip forward at exactly one house along the shared road"

| World object | Problem object |
|---|---|
| road of lines and dots | the index axis `0..n-1`, shared by both sequences |
| furrow 1, flush to the marks, never moves again | `a` — the reference frame, index `i` |
| furrow 2, slippable | `b` — the frame allowed a relative offset |
| a pawn | one character; its carved beast = one of `{A,C,G,T}` |
| "no two beasts mistaken even in poor lamplight" | exact byte equality, no scoring matrix, no ambiguity codes |
| the horse's-head piece, the only slip-maker | a single indel event: one gap opened in `b`, one in `a` |
| "wherever I set it down, every pawn behind shuffles one house forward" | a **global** offset of `+1` applied to a whole suffix/interval, decided in advance, not discovered cell-by-cell |
| "the horse stands in each doorway once, and once more standing nowhere at all" | enumerate all `n` slip positions plus the no-slip case: `n+1` trials |
| "how far they truly disagree once the one permitted slip is placed where it does the least harm" | `min` over the trial family = a **provable lower bound on the optimal alignment score** |

**Silent assumption broken:** *"a slip (gap) can only be discovered by having already compared the position before it."* Here the slip is placed first, globally, by fiat; the comparison then happens under it. Also breaks *"the whole grid of every position against every other must be filled in"* — this family has `O(n)` trials, not `O(n²)` cells of mutual dependency.

## SEED 2 — "lotus petal for agreement, red knot for disagreement, house by house, one count per flute note"

| World object | Problem object |
|---|---|
| a house | one alignment column `(i, i+k)` at fixed slip `k` |
| lotus petal | `+1` (match) |
| red knot | `−1` (mismatch), counted as a unit of damage |
| "cool and easy" vs "tied from the ball" | matches are free; only mismatches are *accumulated objects* |
| the dark figure's flute, one low note per house | a **synchronous clock**: every house of one tick is evaluated with no reference to any other house of that tick |
| "never two houses on one breath, never skip one" | exact lockstep, no reordering, no serial carry between houses |

**Silent assumption broken:** *"every cell depends on the ones above, to the left, and diagonally above-left, computed in that order"* and *"one pair of positions is judged at a time."* One flute note judges a whole set of mutually independent houses → this is a **SIMD wavefront**, not a row-major sweep.

## SEED 3 — "every trial's knots into a fist, weigh the fist, fling all fists but the lightest to the flying fish"

| World object | Problem object |
|---|---|
| a fist of knots | the scalar reduction of a trial; the column-by-column detail is discarded |
| "the count is the whole judgment of that trial, nothing else matters" | no traceback, no matrix retained — score only |
| the lightest fist | `argmin` mismatches ⇒ `argmax` score ⇒ the best certificate found so far |
| flying fish that eat the discarded counts | **pruning**: work provably unable to win is never materialised |
| "knot for knot, the number I carry back" | the returned integer must be exact |

**Silent assumption broken:** *"the whole grid of every position against every other must be filled in."* Only a reduced weight per trial survives; and the lightest fist's weight is what licenses throwing the rest away.

# CHOSEN SEED

**Seed 1**, with Seeds 2 and 3 as its mechanics (they are not rival ideas — they are the metaphor's clock and its reduce/prune stage for the same procedure).

**On the preferred assumption:** the task asked me to prefer a seed that breaks *"both strings are read start to end in the same direction."* **Plainly: none of the three seeds breaks it.** The native's two furrows run along *the same* road of lines and dots, "house against house," and the slip is a forward shuffle in one furrow only — both strings are read start-to-end, same direction, throughout. "Nose to nose" describes two rows facing across the road, not one row reversed. I fall back to the most literal seed, as instructed. (Reversing `b` does appear in my artifact — but as a memory-layout consequence of Seed 2's flute clock, not as anything the native said. I will not claim credit for breaking an assumption the native did not break.)

Seed 1 is also the most *different* from Needleman–Wunsch: NW discovers gaps cell-locally by comparing three predecessors; the native decides the gap globally, in advance, and then never looks at a predecessor at all.

# ASSUMPTION BROKEN

Primary: **"a slip (gap) can only be discovered by having already compared the position before it."**
Secondary (from Seeds 2 and 3): **"every cell depends on above/left/diag, computed in that order"** and **"the whole grid must be filled in."**

**Where the metaphor is incomplete — stated up front, not hidden.** One permitted slip is *not* enough to be exact. For equal-length strings under (+1/−1/−2), every alignment satisfies

$$\text{score} = n - 5d - 2k$$

where `d` is the number of gap *pairs* (deletions must equal insertions when `|a|=|b|`) and `k` is the mismatch count. The native's family covers `d = 0` and a subset of `d = 1`; the true optimum can have `d ≥ 2`. A pure translation of Seed 1 would be a fast heuristic that sometimes returns the wrong number, and the contract demands the exact number.

But the metaphor does not die there — it supplies exactly the missing piece. Seed 3's *lightest fist* is a **certificate**. If the kept fist yields achievable score `S`, then because `score ≤ n − 5d` for any alignment,

$$d^\star \le \left\lfloor \frac{n - S}{5} \right\rfloor \equiv D,$$

and since at every point of an alignment path `i − j = (\text{deletions so far}) − (\text{insertions so far})`, every cell of the optimal path obeys `|i − j| ≤ d\star ≤ D`. **The horse is provably forbidden from ever standing further than `D` doorways off the road.** Everything outside that corridor goes to the fish, exactly as the native said, and now with a proof. So: Seed 1 computes the certificate in `O(n)`, Seed 3 turns it into a corridor, Seed 2 fills the corridor in flute-note lockstep. The result is exact, and the pruning is the native's own gesture rather than a textbook band heuristic bolted on.

Two extra literalities worth noting:
- The native's "weigh the fist" for *all* `n+1` trials does **not** cost `O(n²)`. Writing `M[i] = (a_i ≠ b_i)` and `N[i] = (a_i ≠ b_{i-1})`, the knot count of the trial with the bubble on `[p,q]` is `totalM − M[p] − Σ_{p<i≤q}(M[i] − N[i])`, so minimising over *all* bubbles is a **maximum-subarray (Kadane) scan** — `O(n)` for the entire family, both slip directions. The fists are weighed while they are being tied.
- Seed 2's flute makes the *anti-diagonal* (`i+j = t`) the unit of one note. Indexing each anti-diagonal buffer by `i` alone makes all three predecessors land at offsets `i−1`, `i−1`, `i` — perfectly contiguous. Reading `b` backwards is then forced, purely so that both character streams march forward in memory. (Bonus, not a claim.)

# ARTIFACT

```c
/* The horse's walk: a slip-certificate corridor with a flute-paced wavefront.
 *
 * Phase 1 (Seed 1 + Seed 3): weigh every one-slip fist in O(n) via two Kadane
 *   scans (both slip directions) plus the unslipped trial.  Keep the lightest.
 * Phase 2 (Seed 3): the kept fist's weight bounds the number of gap-pairs the
 *   true optimum may use:  d* <= (n - S)/5 = D.  Every alignment path cell then
 *   obeys |i-j| <= D.  Everything outside that corridor goes to the fish.
 * Phase 3 (Seed 2): fill the corridor anti-diagonal by anti-diagonal - one
 *   flute note per anti-diagonal, every house of that note independent, 16
 *   houses per breath in int16 SIMD.  b is stored reversed so both beast
 *   streams march forward in memory.
 *
 * Exact Needleman-Wunsch (match +1, mismatch -1, gap -2) for |a| = |b| = n.
 * Guards: tiny n -> plain rolling DP; D == 0 -> answer is the unslipped count;
 * D clamped to n so the worst case degenerates to a full SIMD wavefront and can
 * never be slower than the reference; int32 wavefront when int16 could overflow.
 */
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64)
#include <immintrin.h>
#endif

#define KMATCH     1
#define KMISMATCH -1
#define KGAP      -2

/* ---- guard: tiny inputs use the simple path, no machinery, no malloc ---- */
#define SMALL_N 64
static int small_dp(int n, const char *a, const char *b)
{
    int prev[SMALL_N + 1], cur[SMALL_N + 1];
    int i, j;
    for (j = 0; j <= n; j++) prev[j] = KGAP * j;
    for (i = 1; i <= n; i++) {
        char ai = a[i - 1];
        cur[0] = KGAP * i;
        for (j = 1; j <= n; j++) {
            int d = prev[j - 1] + ((ai == b[j - 1]) ? KMATCH : KMISMATCH);
            int u = prev[j] + KGAP;
            int l = cur[j - 1] + KGAP;
            int m = (d > u) ? d : u;
            cur[j] = (m > l) ? m : l;
        }
        for (j = 0; j <= n; j++) prev[j] = cur[j];
    }
    return prev[n];
}

/* ---- Phase 1: the horse walks every doorway; all fists weighed in O(n) ---- */
static int slip_certificate(int n, const char *a, const char *b, int *S0out)
{
    int totalM = 0, q;
    int best, m0, runA, GA, runB, GB, G, S1, S0;
    for (q = 0; q < n; q++) totalM += (a[q] != b[q]);
    S0 = n - 2 * totalM;              /* the horse standing nowhere at all */
    *S0out = S0;
    best = S0;

    /* Every one-slip trial, both slip directions, as two max-subarray scans.
     * Bubble on [p,q]:  knots = totalM - M[p] - sum_{p<i<=q} (M[i] - N[i]).   */
    m0 = (a[0] != b[0]);
    runA = m0; GA = m0;
    runB = m0; GB = m0;
    for (q = 1; q < n; q++) {
        int Mq = (a[q] != b[q]);
        int Na = (a[q] != b[q - 1]);   /* furrow b slips forward */
        int Nb = (a[q - 1] != b[q]);   /* furrow a slips forward */
        int cA = runA + (Mq - Na);
        int cB = runB + (Mq - Nb);
        runA = (Mq > cA) ? Mq : cA; if (runA > GA) GA = runA;
        runB = (Mq > cB) ? Mq : cB; if (runB > GB) GB = runB;
    }
    G = (GA > GB) ? GA : GB;
    S1 = n - 5 - 2 * (totalM - G);     /* one gap-pair costs exactly 5 */
    if (S1 > best) best = S1;
    return best;                       /* the one kept fist */
}

/* ---- Phase 3a: flute-paced wavefront, int16 houses (16 per breath) ---- */
static int wave16(int n, const unsigned char *__restrict ap,
                  const unsigned char *__restrict brp, int D,
                  int16_t *blk)
{
    const int16_t SENT = -30000;       /* the fish: never wins a max */
    const int stride = n + 48;
    int16_t *B0 = blk + 8, *B1 = blk + stride + 8, *B2 = blk + 2 * stride + 8;
    int16_t *B[3];
    int t, k;
#ifdef __AVX2__
    const __m256i vneg1 = _mm256_set1_epi16(-1);
    const __m256i vgap  = _mm256_set1_epi16(2);
#endif
    for (k = 0; k < 3 * stride; k++) blk[k] = SENT;
    B[0] = B0; B[1] = B1; B[2] = B2;
    B[0][0] = 0;                        /* dp[0][0] */

    for (t = 1; t <= 2 * n; t++) {
        int16_t       *__restrict cur = B[t % 3];
        const int16_t *__restrict p1  = B[(t + 2) % 3];   /* diagonal t-1 */
        const int16_t *__restrict p2  = B[(t + 1) % 3];   /* diagonal t-2 */
        int ilo = (t > n) ? (t - n) : 0;
        int ihi = (t < n) ? t : n;
        int blo = (t > D) ? ((t - D + 1) >> 1) : 0;       /* |2i - t| <= D */
        int bhi = (t + D) >> 1;
        int lo  = (ilo > blo) ? ilo : blo;
        int hi  = (ihi < bhi) ? ihi : bhi;
        int st = lo, en = hi, off = n - t, i;
        int16_t bv = (int16_t)(KGAP * t);

        if (t <= n) {                                    /* in-corridor edges */
            if (lo == 0) { cur[0] = bv; st = 1; }
            if (hi == t) { cur[t] = bv; en = t - 1; }
        }
        i = st;
#ifdef __AVX2__
        for (; i + 15 <= en; i += 16) {
            __m128i ca  = _mm_loadu_si128((const __m128i *)(ap  + i - 1));
            __m128i cb  = _mm_loadu_si128((const __m128i *)(brp + i + off));
            __m256i eq  = _mm256_cvtepi8_epi16(_mm_cmpeq_epi8(ca, cb));
            __m256i sv  = _mm256_sub_epi16(vneg1, _mm256_add_epi16(eq, eq));
            __m256i dg  = _mm256_add_epi16(
                              _mm256_loadu_si256((const __m256i *)(p2 + i - 1)), sv);
            __m256i uu  = _mm256_loadu_si256((const __m256i *)(p1 + i - 1));
            __m256i ll  = _mm256_loadu_si256((const __m256i *)(p1 + i));
            __m256i mx  = _mm256_sub_epi16(_mm256_max_epi16(uu, ll), vgap);
            _mm256_storeu_si256((__m256i *)(cur + i), _mm256_max_epi16(mx, dg));
        }
#endif
        for (; i <= en; i++) {
            int s = (ap[i - 1] == brp[i + off]) ? KMATCH : KMISMATCH;
            int v = (int)p2[i - 1] + s;
            int u = (int)p1[i - 1], l = (int)p1[i];
            int w = ((u > l) ? u : l) + KGAP;
            cur[i] = (int16_t)((v > w) ? v : w);
        }
        cur[lo - 1] = SENT; cur[lo - 2] = SENT;
        cur[hi + 1] = SENT; cur[hi + 2] = SENT;
    }
    return (int)B[(2 * n) % 3][n];
}

/* ---- Phase 3b: same wavefront, int32 houses (8 per breath), large n ---- */
static int wave32(int n, const unsigned char *__restrict ap,
                  const unsigned char *__restrict brp, int D,
                  int32_t *blk)
{
    const int32_t SENT = -1000000000;
    const int stride = n + 48;
    int32_t *B0 = blk + 8, *B1 = blk + stride + 8, *B2 = blk + 2 * stride + 8;
    int32_t *B[3];
    int t, k;
#ifdef __AVX2__
    const __m256i vneg1 = _mm256_set1_epi32(-1);
    const __m256i vgap  = _mm256_set1_epi32(2);
#endif
    for (k = 0; k < 3 * stride; k++) blk[k] = SENT;
    B[0] = B0; B[1] = B1; B[2] = B2;
    B[0][0] = 0;

    for (t = 1; t <= 2 * n; t++) {
        int32_t       *__restrict cur = B[t % 3];
        const int32_t *__restrict p1  = B[(t + 2) % 3];
        const int32_t *__restrict p2  = B[(t + 1) % 3];
        int ilo = (t > n) ? (t - n) : 0;
        int ihi = (t < n) ? t : n;
        int blo = (t > D) ? ((t - D + 1) >> 1) : 0;
        int bhi = (t + D) >> 1;
        int lo  = (ilo > blo) ? ilo : blo;
        int hi  = (ihi < bhi) ? ihi : bhi;
        int st = lo, en = hi, off = n - t, i;
        int32_t bv = (int32_t)(KGAP * t);

        if (t <= n) {
            if (lo == 0) { cur[0] = bv; st = 1; }
            if (hi == t) { cur[t] = bv; en = t - 1; }
        }
        i = st;
#ifdef __AVX2__
        for (; i + 7 <= en; i += 8) {
            __m128i ca  = _mm_loadl_epi64((const __m128i *)(ap  + i - 1));
            __m128i cb  = _mm_loadl_epi64((const __m128i *)(brp + i + off));
            __m256i eq  = _mm256_cvtepi8_epi32(_mm_cmpeq_epi8(ca, cb));
            __m256i sv  = _mm256_sub_epi32(vneg1, _mm256_add_epi32(eq, eq));
            __m256i dg  = _mm256_add_epi32(
                              _mm256_loadu_si256((const __m256i *)(p2 + i - 1)), sv);
            __m256i uu  = _mm256_loadu_si256((const __m256i *)(p1 + i - 1));
            __m256i ll  = _mm256_loadu_si256((const __m256i *)(p1 + i));
            __m256i mx  = _mm256_sub_epi32(_mm256_max_epi32(uu, ll), vgap);
            _mm256_storeu_si256((__m256i *)(cur + i), _mm256_max_epi32(mx, dg));
        }
#endif
        for (; i <= en; i++) {
            int s = (ap[i - 1] == brp[i + off]) ? KMATCH : KMISMATCH;
            int v = (int)p2[i - 1] + s;
            int u = (int)p1[i - 1], l = (int)p1[i];
            int w = ((u > l) ? u : l) + KGAP;
            cur[i] = (int32_t)((v > w) ? v : w);
        }
        cur[lo - 1] = SENT; cur[lo - 2] = SENT;
        cur[hi + 1] = SENT; cur[hi + 2] = SENT;
    }
    return (int)B[(2 * n) % 3][n];
}

int kernel(int n, const char *a, const char *b)
{
    int S0, Sbest, D, W, res, i;
    unsigned char *chars, *ap, *brp;

    if (n <= 0) return 0;

    /* Phase 1 + 2: the kept fist, and the corridor it certifies. */
    Sbest = slip_certificate(n, a, b, &S0);
    D = (n - Sbest) / 5;
    if (D <= 0) return S0;            /* no slip anywhere can beat the road */
    if (D > n) D = n;                 /* guard: worst case = full wavefront */

    if (n <= SMALL_N) return small_dp(n, a, b);   /* guard: tiny inputs */

    chars = (unsigned char *)malloc((size_t)2 * (n + 64));
    if (!chars) return small_dp(n < SMALL_N ? n : SMALL_N, a, b); /* unreachable */
    ap  = chars;
    brp = chars + (n + 64);
    memcpy(ap, a, (size_t)n);
    memset(ap + n, 'X', 64);
    for (i = 0; i < n; i++) brp[i] = (unsigned char)b[n - 1 - i];  /* read back */
    memset(brp + n, 'Y', 64);

    W = D;
    if (n <= 6000) {                  /* int16 houses: |dp| <= 4n < 24000 */
        int16_t *blk = (int16_t *)malloc((size_t)3 * (n + 48) * sizeof(int16_t));
        if (!blk) { free(chars); return small_dp(SMALL_N, a, b); }
        if (n >= 256 && D > 32) {     /* one narrow trial first: tighten D */
            int s = wave16(n, ap, brp, 32, blk);
            int Dn = (n - s) / 5;
            if (Dn <= 32) { free(blk); free(chars); return s; }  /* proven */
            if (Dn < W) W = Dn;
        }
        res = wave16(n, ap, brp, W, blk);
        free(blk);
    } else {
        int32_t *blk = (int32_t *)malloc((size_t)3 * (n + 48) * sizeof(int32_t));
        if (!blk) { free(chars); return small_dp(SMALL_N, a, b); }
        if (D > 32) {
            int s = wave32(n, ap, brp, 32, blk);
            int Dn = (n - s) / 5;
            if (Dn <= 32) { free(blk); free(chars); return s; }
            if (Dn < W) W = Dn;
        }
        res = wave32(n, ap, brp, W, blk);
        free(blk);
    }
    free(chars);
    return res;
}
```

# PREDICTION

**PREDICTION: speedup_vs_dp = 20**

Written before any measurement. Reasoning, so the number is falsifiable rather than decorative:

- Reference: `(n+1)²` cells, scalar, with a loop-carried `dp[i][j-1]` chain → ~2–3 cycles/cell, plus `4(n+1)²` bytes of streamed store traffic (16.8 MB at `n=2048`). Estimate ≈ `2.5 n²` cycles.
- Artifact, random/unrelated sequences: `k₀ ≈ 0.75n` ⇒ `S ≈ −0.5n` ⇒ `D ≈ 0.3n`, corridor spans `0.6n` ⇒ `≈ 0.6 n²` cells at ~12 instructions per 16 cells ≈ `0.25` cycles/cell ⇒ `≈ 0.15 n²` cycles. Ratio ≈ **17×**. Working set is three `int16` rows (~12 KB), L1-resident, so no memory term at all.
- Artifact, substitution-diverged sequences (`k₀ = 0.05n`): `D = 0.02n`, corridor `0.04n` ⇒ ≈ **250×**.
- Artifact, sequences differing by a handful of indels: the 32-wide probe proves optimality ⇒ `O(64n)` ⇒ **1000×+**.
- Artifact, identical sequences: `D = 0`, `O(n)` early return ⇒ effectively unbounded.
- Predicted worst case (adversarial `D = n`): full anti-diagonal SIMD wavefront, no corridor benefit ⇒ still ≈ **10×** from vectorisation and cache alone; never a regression.

`20` is my point estimate for a benchmark that uses independently-generated random DNA (the pessimistic assumption), with upside if it uses related sequences.

# MEASUREMENT

**Not performed. Reported plainly as a gap, not glossed.** This session was invoked with no tool access — `alignment_bench` and `alignment_contract` were unavailable, so I could not compile, could not verify exactness against the reference DP, and could not time anything. The four permitted improvement rounds were therefore not spent; the artifact above is round zero. Its prediction stands unvalidated, and the honest status is:

| Claim | Status |
|---|---|
| exactness of the corridor bound `d* ≤ ⌊(n−S)/5⌋` | proved algebraically above; **not** empirically confirmed |
| hand-traced end-to-end correctness | done on `a="ACGT", b="CGTA"` (`n=4`, `D=1`) — all 9 diagonals matched the reference DP cell for cell, final `dp[4][4] = −1` ✓ |
| sentinel non-propagation / no int16 overflow | argued (every in-corridor cell has an in-corridor diagonal predecessor, so fish values never accumulate); **not** confirmed by execution |
| speedup = 20 | **unmeasured prediction** |

Things I would look at first with the bench, in order: (1) exactness sweep over random pairs and over pairs with 1–10 planted indels at `n ∈ {65, 100, 256, 1024, 4096}` plus the `n ≤ 64`, `D = 0`, and `n > 6000` path boundaries; (2) whether the 32-wide probe pays for itself on the bench's actual sequence generator or is 5% of pure waste; (3) whether the corridor is worth its bookkeeping at all on random input, or whether the unbanded wavefront (`D = n`) is simply faster there because the inner loops run longer and the per-diagonal overhead amortises.

# VERDICT

**What the native actually contributed, and it is not decorative.** The one-slip trial family is a heuristic, and I say so without hedging: it is not exact, and shipping it alone would return wrong numbers. But its *reduction* — Seed 3's kept fist — is a certificate, and the scoring algebra `score = n − 5d − 2k` converts that certificate into a hard cap on how far any optimal path may wander off the diagonal. The native's own gesture of flinging every heavier fist to the fish becomes a proof-carrying corridor. That is a genuinely different route to the band than the literature's: SSW/KSW2 choose a band by *fiat* and lose exactness, or by score-range assumptions; here the band width is *derived* from an `O(n)` measurement of the input, and exactness is preserved by construction. Seed 2's flute then supplies the reason the corridor can be filled at 16 cells per breath instead of one cell per three dependencies.

**Risks I named, and how each is closed rather than left dangling** (per the instruction not to ship a mechanism whose own stated risk I dodge):

- *"Only helps when the corridor is narrow."* → `D` is clamped to `n`, so the wide-corridor case degenerates exactly into a full anti-diagonal SIMD wavefront. That is still `~10×` faster than the reference, so the mechanism has no regression regime, only a diminished-benefit one.
- *"Overhead if small."* → `n ≤ 64` falls back to a plain rolling-two-row DP with no malloc and no machinery.
- *"The probe may be pure waste."* → gated on `n ≥ 256 && D > 32`, bounding its cost at `64n` (~5% of the `0.6n²` main pass at `n = 2048`) and zero in the cases where it cannot help.
- *"int16 overflow."* → explicit `n ≤ 6000` guard with a structurally identical int32 wavefront above it, because `|dp| ≤ 4n` and the fish sentinel must sit below that.
- *"No AVX2."* → both wavefronts keep a scalar tail loop that is itself dependency-free across `i` with `__restrict` on distinct buffers, so GCC can auto-vectorise it; correctness never depends on the intrinsics.

**Regime detection, expressed through the metaphor itself** (per the two-regime requirement): the weight of the kept fist *is* the regime sensor. A light fist means the strings nearly share a road — narrow corridor, near-linear work. A heavy fist means they are strangers — the corridor opens toward the full board and the kernel takes the wide path. The 32-wide probe is a third reading of the same sensor, catching the case the native's single slip cannot see: several indels, where the unslipped count lies about the true distance. All three readings happen at runtime on the actual input; nothing is compiled in.

**What I refused to do.** I did not quietly replace the native's idea with Needleman–Wunsch and then credit the metaphor. The DP recurrence does appear inside the corridor — it has to, for exactness — but it is reached *through* the native's certificate, it is restricted by it, and it is evaluated in the native's flute-paced independent-houses order rather than the textbook's row-major dependent order. I also did not use OpenMP: the metaphor's unit of simultaneous work is one flute note, i.e. one anti-diagonal of at most a few thousand cells, and `2n` thread barriers would cost more than the work they distribute at these sizes. Vectorisation first, as instructed; threads not earned.

**And the honest bottom line:** the prediction of `20×` is a calculation, not a result. Until `alignment_bench` confirms both the score-for-score exactness and the timing, this is a well-argued, hand-traced, unmeasured artifact — and it should be read that way.