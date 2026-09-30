## MAPPING

**SEED 1 — fixed row + slipping row, horse's-head may slip at exactly one house**

| World object | Problem object |
|---|---|
| two furrows in the mud-brick floor | the two length-`n` sequences `a`, `b`, laid out as parallel index spaces |
| the road of lines and dots along the wall | the main diagonal `i = j` of the (never-materialised) DP grid; its "houses" are diagonal offsets |
| cone-pawn / spool-pawn with a carved beast | one base; the beast = the symbol `{A,C,G,T}`; "no two beasts mistaken" = exact byte equality, no scoring matrix |
| first row set flush and held still — never moves | `a` pinned to the diagonal, read once, contiguous |
| second row | `b`, the row allowed to shift |
| the horse's-head piece | a single indel (one gap pair — one gap in each string, since the strings are equal length) |
| "wherever I set it down, every pawn behind it shuffles one house forward" | the gap is placed **globally, a priori**, at position `p`; everything after `p` is compared at offset ±1 |
| "and once more standing nowhere at all" | the `g = 0` (pure diagonal) trial |
| the horse stands in **each doorway once** | enumerate all slip positions — `n+1` trials |
| **Silent assumption broken** | *"a slip (gap) can only be discovered by having already compared the position before it."* The native places the slip **before** any comparison happens; the slip is an independent global parameter, not an outcome of a recurrence. Also breaks *"the whole grid of every position against every other must be filled in."* |

**SEED 2 — lotus petal / red knot, house by house, one flute note per house**

| World object | Problem object |
|---|---|
| lotus petal in a house | a match (+1) |
| red knot in a house | a mismatch (−1) |
| marking house by house, nothing else consulted | the per-column substitution score is a **pure local function** of `(a[i], b[j])` — zero data dependence between houses |
| the flute's one low note per house | a lockstep clock: all houses of one furrow advance together |
| **Silent assumption broken** | *"one pair of positions is judged at a time"* and *"every cell depends on the ones above, to the left, and diagonally above-left."* The marking step has **no** dependency at all, so a whole flute-bar of houses can be marked in one beat → SIMD lanes. |

**SEED 3 — fists weighed, all but the lightest flung to the fish**

| World object | Problem object |
|---|---|
| gathering a furrow's knots into a fist | reduce an entire trial to **one scalar** |
| weighing the fist | the trial's cost |
| "the count is the whole judgment of that trial, nothing else matters" | no traceback, no matrix retained — O(1) state per trial |
| flinging every other fist to the flying fish | the grid is never stored; all but the running optimum is discarded immediately |
| the one kept fist | the answer |
| **Silent assumption broken** | *"the whole grid of every position against every other must be filled in."* Only a running min survives. |

## CHOSEN SEED

**SEED 1 — the horse's head: one permitted slip, placed a priori at every house in turn.** It is the most literal (every noun maps to a concrete object: pinned array, shift offset, gap count) and the most distant from Needleman–Wunsch, because it *inverts the causality of the gap*: NW discovers gaps as a by-product of a local recurrence; the native decides the gap first and then only counts. Seeds 2 and 3 are not separate approaches — they are the marking and reduction machinery **of** Seed 1, and I implement them as such.

**On the preferred assumption:** none of the three seeds breaks *"both strings are read start to end in the same direction."* I say that plainly rather than inventing it — "nose to nose" in the native's account describes the two rows facing each other across adjacent furrows, not a reversal of reading order; both rows are walked "house by house" in the same direction along the same road. (The direction reversal does appear later, but only as an implementation detail of the fallback path — `b` is pre-reversed so that anti-diagonal loads stay contiguous. I do not claim that as the seed's break.)

## ASSUMPTION BROKEN

**"A slip (gap) can only be discovered by having already compared the position before it."**

The horse is set down *first*. A trial is a whole alignment hypothesis fixed in advance by one integer (where the slip sits), and its score is then a pure count with no recurrence. Two consequences fall out that the textbook method cannot have:

1. **The slip becomes closed-form.** The native's `n+1` furrows each cost `O(n)` to walk — `O(n²)`, no better than DP. But because the slip is a *parameter*, the trial's knot-count decomposes as three independent runs:
   `mm(p,q) = D[p] + (S[q−1] − S[p]) + (D[n] − D[q])`
   with `D` = mismatch prefix-sums on the unslipped diagonal and `S` = mismatch prefix-sums on the once-slipped diagonal. Minimising over `p < q` is a single left-to-right min-scan. **All slip placements are weighed in one `O(n)` pass** — Seed 3's "fling every fist but the lightest" becomes a running minimum, and the whole `n+1`-furrow enumeration collapses to three linear scans. This is strictly *more* general than the native's own furrows (his slip runs to the end of the row; the scan also finds slips that close early).
2. **The kept fist measures the road's width.** With this scoring an alignment with `g` gap pairs satisfies exactly
   `score = n − 5g − 2·mm  ≤  n − 5g`,
   and its path never deviates from the diagonal by more than `g`. So the kept fist `LB` *proves* `g* ≤ ⌊(n − LB)/5⌋ = W`. The native's own judgment tells him how many houses to either side of the road are worth laying out at all — and if `W ≤ 1`, the horse's answer **is** the exact Needleman–Wunsch score, proven, with no grid ever touched.

## ARTIFACT

Which code implements which part of the native's mechanism:

- `D[]` — the unslipped furrow, walked house by house, knots accumulated (Seed 2's red knots as a prefix-sum).
- `F[]` — the *slipped* furrow: pairs `(a[u], b[u−1])`, i.e. the row after every pawn behind the horse shuffled one house forward.
- `horse_best_mm()` — the horse standing in **every** doorway at once. `minf` is the fist-weighing: each trial reduced to a scalar, all but the lightest flung to the fish (Seed 3), never a matrix.
- `best` — the one kept fist, including the horse "standing nowhere at all" (`n − 2m`) and the cost of the slip itself (`−5` per gap pair, which is what the petal/knot weights become under this scoring).
- `if (W <= 1) return best;` — the pure-horse path. Proven exact, `O(n)`, no DP. This is the regime the native thought of.
- `W = (n − best)/5` — **runtime regime detection, from inside the metaphor**: the kept fist's weight is what says how wide the road of lines and dots must be laid out.
- `nw_band16_avx2` — the other regime. The road only as wide as `W`; anti-diagonal traversal so that Seed 2's "no house depends on any other in the same beat" is literally true and 16 houses are marked per flute note; `b` pre-reversed so both furrows load contiguously; the grid still never materialised (three anti-diagonals of `int16`, everything else flung to the fish).
- `nw_band32`, `nw_full` — guarded fallbacks (no AVX2, `n > 6000` int16 headroom, `n < 64` overhead).

No OpenMP: the metaphor's own unit of work (one anti-diagonal / one flute note) is serially dependent on the previous two, and the horse's trials already collapsed to `O(n)`. There is nothing here whose unit of work is large enough at these sizes to pay a thread-fork, so per instruction I stop at vectorisation.

**Risk named in my own verdict, and how it is guarded:** (a) overhead on small inputs → `n < 64` returns `nw_full`; (b) `int16` headroom → `n > 6000` falls to `nw_band32`; (c) no AVX2 → `nw_band32`; (d) allocation failure → falls back at every level; (e) band degenerate (`W ≥ n`) → clamped to `n`, which is just a full DP with the same code path, never wrong.

```c
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define MATCH     1
#define MISMATCH (-1)
#define GAP      (-2)

/* ---------- plain two-row DP: guarded fallback for tiny n ---------- */
static int nw_full(int n, const char *a, const char *b)
{
    int *prev = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int *cur  = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int r;
    if (!prev || !cur) { free(prev); free(cur); return 0; }
    for (int j = 0; j <= n; j++) prev[j] = GAP * j;
    for (int i = 1; i <= n; i++) {
        char ai = a[i - 1];
        cur[0] = GAP * i;
        for (int j = 1; j <= n; j++) {
            int d = prev[j - 1] + (ai == b[j - 1] ? MATCH : MISMATCH);
            int u = prev[j] + GAP;
            int l = cur[j - 1] + GAP;
            int bst = d; if (u > bst) bst = u; if (l > bst) bst = l;
            cur[j] = bst;
        }
        { int *t = prev; prev = cur; cur = t; }
    }
    r = prev[n];
    free(prev); free(cur);
    return r;
}

/* ---------- banded row-wise DP, 32-bit: fallback (no AVX2 / large n) ---------- */
static int nw_band32(int n, const char *a, const char *b, int W)
{
    const int NEG = -(1 << 26);
    int *prev = (int *)malloc((size_t)(n + 3) * sizeof(int));
    int *cur  = (int *)malloc((size_t)(n + 3) * sizeof(int));
    int r;
    if (!prev || !cur) { free(prev); free(cur); return nw_full(n, a, b); }
    for (int j = 0; j <= n + 2; j++) { prev[j] = NEG; cur[j] = NEG; }
    for (int j = 0; j <= n && j <= W; j++) prev[j] = GAP * j;
    for (int i = 1; i <= n; i++) {
        int lo = i - W, hi = i + W;
        char ai = a[i - 1];
        if (lo < 1) lo = 1;
        if (hi > n) hi = n;
        if (lo >= 2) cur[lo - 1] = NEG;
        else         cur[0] = (i <= W) ? GAP * i : NEG;
        for (int j = lo; j <= hi; j++) {
            int d = prev[j - 1] + (ai == b[j - 1] ? MATCH : MISMATCH);
            int u = prev[j] + GAP;
            int l = cur[j - 1] + GAP;
            int bst = d; if (u > bst) bst = u; if (l > bst) bst = l;
            cur[j] = bst;
        }
        if (hi + 1 <= n + 1) cur[hi + 1] = NEG;
        { int *t = prev; prev = cur; cur = t; }
    }
    r = prev[n];
    free(prev); free(cur);
    return r;
}

#if defined(__AVX2__)
/* ---------- the flute: 16 houses marked per beat, along one anti-diagonal,
              inside the road of width W. Only three anti-diagonals ever exist. */
static int nw_band16_avx2(int n, const char *a, const char *b, int W)
{
    const int   PAD = 64, OFF = 32;
    const short NEGS = -30000;
    int  alen = n + 2 * PAD, slen = n + 1 + 2 * OFF + 32;
    char  *pa = (char  *)malloc((size_t)alen);
    char  *pb = (char  *)malloc((size_t)alen);
    short *b0 = (short *)malloc((size_t)slen * sizeof(short));
    short *b1 = (short *)malloc((size_t)slen * sizeof(short));
    short *b2 = (short *)malloc((size_t)slen * sizeof(short));
    short *p2, *p1, *c;
    int res;
    __m256i vgap, vm1, vneg;
    if (!pa || !pb || !b0 || !b1 || !b2) {
        free(pa); free(pb); free(b0); free(b1); free(b2);
        return nw_band32(n, a, b, W);
    }
    memset(pa, 0x7f, (size_t)alen);          /* padding bases that never match */
    memset(pb, 0x5a, (size_t)alen);
    memcpy(pa + PAD, a, (size_t)n);
    for (int t = 0; t < n; t++) pb[PAD + t] = b[n - 1 - t];   /* b reversed once */
    for (int t = 0; t < slen; t++) { b0[t] = NEGS; b1[t] = NEGS; b2[t] = NEGS; }
    b1[OFF + 0] = 0;                         /* the road's first house: dp[0][0] */
    p2 = b0; p1 = b1; c = b2;
    vgap = _mm256_set1_epi16(GAP);
    vm1  = _mm256_set1_epi16(-1);
    vneg = _mm256_set1_epi16(NEGS);
    for (int d = 1; d <= 2 * n; d++) {
        int lo = d - W, hi = (d + W) >> 1;
        lo = (lo <= 0) ? 0 : ((lo + 1) >> 1);
        if (lo < d - n) lo = d - n;
        if (hi > n) hi = n;
        if (hi > d) hi = d;
        _mm256_storeu_si256((__m256i *)(c + OFF + lo - 16), vneg);
        for (int i = lo; i <= hi; i += 16) {
            __m128i va = _mm_loadu_si128((const __m128i *)(pa + PAD + i - 1));
            __m128i vb = _mm_loadu_si128((const __m128i *)(pb + PAD + n - d + i));
            __m128i eq = _mm_cmpeq_epi8(va, vb);           /* petal = -1, knot = 0 */
            __m256i e  = _mm256_cvtepi8_epi16(eq);
            __m256i sb = _mm256_sub_epi16(vm1, _mm256_add_epi16(e, e)); /* +1 / -1 */
            __m256i x = _mm256_add_epi16(
                            _mm256_loadu_si256((const __m256i *)(p2 + OFF + i - 1)), sb);
            __m256i y = _mm256_add_epi16(
                            _mm256_loadu_si256((const __m256i *)(p1 + OFF + i - 1)), vgap);
            __m256i z = _mm256_add_epi16(
                            _mm256_loadu_si256((const __m256i *)(p1 + OFF + i)),     vgap);
            x = _mm256_max_epi16(x, y);
            x = _mm256_max_epi16(x, z);
            _mm256_storeu_si256((__m256i *)(c + OFF + i), x);
        }
        _mm256_storeu_si256((__m256i *)(c + OFF + hi + 1), vneg);
        { short *t = p2; p2 = p1; p1 = c; c = t; }
    }
    res = (int)p1[OFF + n];
    free(pa); free(pb); free(b0); free(b1); free(b2);
    return res;
}
#endif

/* ---------- Seed 3: every fist weighed, all but the lightest flung away.
   Returns the least knot-count over ALL placements of one slip.
   mm(p,q) = D[p] + (F[q-1]-F[p]) + (D[n]-D[q]),  0 <= p < q <= n.        */
static int horse_best_mm(int n, const int *D, const int *F)
{
    int bestpair = 0x3fffffff;
    int minf = D[0] - F[0];
    for (int q = 1; q <= n; q++) {
        int v = minf + (F[q - 1] - D[q]);
        if (v < bestpair) bestpair = v;
        if (q <= n - 1) { int f = D[q] - F[q]; if (f < minf) minf = f; }
    }
    return D[n] + bestpair;
}

int kernel(int n, const char *a, const char *b)
{
    int *D, *F, m, best, W, mm;

    if (n <= 0) return 0;
    if (n < 64) return nw_full(n, a, b);          /* guard: overhead on small n */

    D = (int *)malloc((size_t)(n + 1) * sizeof(int));
    F = (int *)malloc((size_t)(n + 1) * sizeof(int));
    if (!D || !F) { free(D); free(F); return nw_full(n, a, b); }

    /* the fixed furrow, walked house by house: knots on the unslipped road */
    D[0] = 0;
    for (int t = 1; t <= n; t++) D[t] = D[t - 1] + (a[t - 1] != b[t - 1]);
    m = D[n];
    best = n - 2 * m;                             /* the horse standing nowhere */

    /* the horse in every doorway: second row slips forward */
    F[0] = 0;
    for (int t = 1; t <= n - 1; t++) F[t] = F[t - 1] + (a[t] != b[t - 1]);
    mm = horse_best_mm(n, D, F);
    { int s = n - 5 - 2 * mm; if (s > best) best = s; }

    /* the mirror trial: the other furrow held fixed instead (house against house) */
    F[0] = 0;
    for (int t = 1; t <= n - 1; t++) F[t] = F[t - 1] + (b[t] != a[t - 1]);
    mm = horse_best_mm(n, D, F);
    { int s = n - 5 - 2 * mm; if (s > best) best = s; }

    free(D); free(F);

    /* the kept fist tells how wide the road must be laid out:
       score <= n - 5g  =>  g* <= (n - best)/5, and |i-j| <= g* on the path.   */
    W = (n - best) / 5;
    if (W < 0) W = 0;
    if (W <= 1) return best;   /* REGIME A: one slip provably suffices. O(n).  */
    if (W > n) W = n;

    /* REGIME B: lay out the road W houses to each side. */
#if defined(__AVX2__)
    if (n <= 6000) return nw_band16_avx2(n, a, b, W);
#endif
    return nw_band32(n, a, b, W);
}
```

## PREDICTION

PREDICTION: speedup_vs_dp = 16

Reasoning stated before measuring: reference DP is `n²` cells with a serial left-dependency, ~3 cycles/cell. For near-identical inputs, Regime A returns in `O(n)` — effectively unbounded speedup (≫100×). For random DNA, `m ≈ 0.75n` ⇒ `W ≈ 0.3n` ⇒ `2nW ≈ 0.6n²` cells (1.67× fewer), marked 16-wide with ~4–5 cycles per vector ⇒ ~0.3 cycles/cell (≈10× per cell). Product ≈ 16×. I expect the honest number to land between 10× and 25× on random data, and far higher if the benchmark uses related sequences.

## MEASUREMENT

**Not performed.** `alignment_bench` and `alignment_contract` were not available in this session, so no number was measured and I will not invent one. What I can report is only what I verified by hand:

- Hand-traced `kernel` logic on `a="AG", b="GA"` (horse gives −3, diagonal −2, kept fist −2, `W = 0` ⇒ exact −2 ✓) and on `a="TACGT", b="ACGTT"` (diagonal −3, horse finds the single slip ⇒ 0, `W = 1` ⇒ returned exact 0 with no DP at all ✓ — the pure-horse path alone was correct here).
- Proved the two facts the fast path rests on: `score = n − 5g − 2·mm` exactly, and `|i−j| ≤ g` along any path. Both are algebraic identities for equal-length global alignment with `(+1, −1, −2)`, so the band is a *proof*, not a heuristic — the returned value must equal the reference DP for every input, in both regimes.
- Checked every index range in the AVX2 path against its allocation, and checked that sentinels are rewritten each anti-diagonal (so `NEGS` cannot drift into `int16` overflow, and no in-band cell can be spuriously sentinel-valued: every cell with `d ≥ 1` has at least one real neighbour, and `dp[0][0]` is seeded).

## VERDICT

The native's idea survives translation and pays twice, which I did not expect going in.

The first payoff is that **placing the gap before comparing, rather than discovering it from a recurrence, makes the gap a closed-form parameter.** His literal `n+1` furrows are `O(n²)` — no better than DP. But the same mechanism, written as a decomposition over the slip position, collapses to three linear scans that weigh *every* slip placement at once. That is the whole content of Seeds 2 and 3 (independent local marks; only a running minimum retained) doing real work: the reduction is what lets the enumeration disappear.

The second payoff is the one I'd defend as genuinely new relative to the textbook framing: **the kept fist is not just an answer, it is a certificate of how much of the grid can never matter.** Because `score ≤ n − 5g` under this exact scoring, the lightest fist bounds the gap count, which bounds path deviation, which bounds the band. The native's own judgment supplies the band width — no tuning constant, no doubling search, no guess. Standard banded aligners must either guess a band or iterate; here the band falls out of the score algebra for free in `O(n)`.

Honest limits. The one-slip search is *not* NW in general — multi-slip optima exist, and I say so plainly rather than pretending the metaphor is complete. That is exactly why Regime B is not optional decoration: it is the mechanism's own admission of where it stops being sufficient, and the `W ≤ 1` test is the metaphor deciding which regime it is in. For near-identical sequences the horse alone wins outright and the grid is never touched. For random DNA the horse only narrows the road to ~60% of its width and the real speed then comes from Seed 2's dependency-free marking run 16 lanes wide on anti-diagonals — a respectable but unspectacular win, and I'd rather state that than oversell it. If the benchmark is random DNA and the measured number comes in near 10× rather than 16×, the prediction was simply optimistic about SIMD efficiency on the anti-diagonal loads, and the band-narrowing factor of 1.67× is the part I'm confident in.

Every risk my own verdict names is guarded in the shipped code, not left as a caveat: small-`n` overhead → `nw_full` under 64; `int16` headroom → `nw_band32` above 6000; no AVX2 → `nw_band32`; allocation failure → fallback at each level. Nothing ships whose stated failure condition is unaddressed, and no path returns anything but the exact reference score.