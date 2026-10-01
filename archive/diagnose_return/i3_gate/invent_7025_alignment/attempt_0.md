# MAPPING

## SEED 1 — "a fixed row, a second row the horse may slip forward at exactly one house"

| World object | Computational object |
|---|---|
| the road of lines and dots | the index axis `0..n-1` shared by both strings |
| furrow 1, flush against the marks, never moves | `a`, pinned to the main diagonal `i = j` |
| furrow 2, slippable | `b`, whose offset from `a` is a free parameter |
| the horse's-head piece set down at house *p* | an indel event opened at position *p* (path leaves diagonal `k=0`) |
| "pawns behind it shuffle one house forward" | all of `b` after *p* is re-indexed by −1 → path moves to diagonal `k = ±1` |
| the last pawn falling off the eaves | the closing gap that returns the path to `k = 0` at `(n,n)` |
| "unslipped and slipped at every possible house in turn" | enumerate **all** alignments with 0 or 1 gap-pair, in one sweep |
| "the horse stands nowhere at all" | the ungapped (pure-diagonal) alignment |

**Silent assumption broken:** *"a slip (gap) can only be discovered by having already compared the position before it."* The horse is **placed first**, globally, and the comparison happens afterwards. The gap is a parameter of the trial, not an outcome of the recurrence.

## SEED 2 — "lotus petal for agreement, red knot for disagreement, one per flute note"

| World object | Computational object |
|---|---|
| carved beast on a pawn's crown | one base in `{A,C,G,T}`, compared only by equality |
| lotus petal | `+1` match |
| red knot | `−1` mismatch |
| the flute's one-note-per-house clock | a fixed-stride, branch-free lane step — no data dependence between houses |
| walking two furrows together | a single linear scan of two byte streams |

**Silent assumption broken:** *"one pair of positions is judged at a time"* and *"every cell depends on above, left and diagonal, in that order."* Inside one trial nothing depends on anything else, so 8 (or 32) houses can be judged in one instruction.

## SEED 3 — "weigh the fist; fling every heavier fist to the flying fish"

| World object | Computational object |
|---|---|
| a fist of knots | one scalar per trial — its mismatch count |
| "nothing else about that trial matters" | the whole trial collapses to **one integer**; no table retained |
| keeping only the lightest fist | `max` over candidate scores → a certified lower bound `LB` on the optimum |
| flying fish eating the discarded fists | **pruning**: whole regions of `(i,j)` space proven unreachable and never allocated, never touched |

**Silent assumption broken:** *"the whole grid of every position against every other must be filled in."*

**Does any seed break "both strings are read start to end in the same direction"?** **No — plainly, none of the three does.** The native reads both furrows left to right at the same pace; the only asymmetry is *which* furrow is allowed to slip. I therefore fall back, as instructed, to the most literal seed.

# CHOSEN SEED

**SEED 1.** It is the most literal (every object — fixed row, slipping row, horse, house — has an exact counterpart) and the most distant from Needleman–Wunsch, which never names a gap position in advance.

# ASSUMPTION BROKEN

**"A slip (gap) can only be discovered by having already compared the position before it."**

Taking the horse literally forces the arithmetic the DP never does. Let the path from `(0,0)` to `(n,n)` have `k` down-moves and `k` right-moves (they must be equal for equal-length strings). Then

* aligned columns `= n − k`, gap columns `= 2k`
* **score ≤ (n − k) − 4k = n − 5k**
* the path's maximum deviation `δ = max|i−j|` satisfies `δ ≤ k`

so **score ≤ n − 5δ**. Reading that backwards: once I hold *any* real alignment's score `LB`, the optimal path is imprisoned in the band `|i − j| ≤ ⌊(n − LB)/5⌋`. The horse's walk hands me `LB` in **O(n)** — and the flying fish eat everything outside the band.

Two further consequences fall straight out of the metaphor:

1. **The horse's whole walk is O(n), not O(n²).** Trial *(down at p, right at q)* has mismatch count `X(p,q) = d₀ + f(p) + g(q)` with `f(p)=P₀[p]−M[p]`, `g(q)=M[q]−P₀[q+1]`. A running minimum of `f` gives every trial's fist in one pass.
2. **When the horse is nearly enough, it is exactly enough.** If `W ≤ 1` then `n − LB ≤ 9`, so any alignment with `k ≥ 2` scores `≤ n − 10 < LB`. The optimum therefore uses at most one slip — and the horse enumerated *all* of those. **Return `LB`; the answer is exact in O(n).**

This is where step 4 bites: rather than invent something, the mechanism lands on **score-bounded banded global alignment with iterative band refinement** — the validated technique the brief already names ("a banded/SIMD variant exploiting a bounded score range", as in KSW2/SSW). I let it arrive there instead of shipping a novelty.

**Regime detection (step 5), encoded in the metaphor itself:** the weight of the lightest fist *is* the regime detector. Light fist → `W ≤ 1` → O(n) closed form. Medium fist → narrow band. Heavy fist (random DNA, `d₀ ≈ 0.75n`) → `W ≈ 0.3n`, band ≈ 0.6n² cells, still strictly less than the full grid, run under SIMD. A cheap `W=16` probe pass refines `LB` before committing to the wide band, and the whole thing falls back to a plain two-row DP for `n < 32` or on allocation failure — so no input makes this slower than the reference.

**What is what, literally:** memory = the three anti-diagonal furrows (only 3·(n+26) ints, never the grid); what flows = the wavefront of anti-diagonals; what stays still = `a`, pinned to the diagonal; the processor = one AVX2 lane per house (8 houses judged per flute note); time = `t = i + j`, the anti-diagonal index — along which, exactly as the native says, no house waits on its neighbour. **No OpenMP**: the native's independent units (the trials) are the O(n) horse-walk, far too small to thread; the anti-diagonal wavefront is inherently serial across `t`. Vectorization only, as instructed.

# ARTIFACT

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
#define NEG_BIG  (-1061109568)      /* 0xC0C0C0C0 : memset-able sentinel */

/* ---------- fallback: plain two-row Needleman-Wunsch ---------------------- */
static int nw_simple(int n, const char *a, const char *b)
{
    int *prev, *cur, *tmp, i, j, r;
    prev = (int *)malloc((size_t)(n + 1) * sizeof(int));
    cur  = (int *)malloc((size_t)(n + 1) * sizeof(int));
    if (!prev || !cur) { free(prev); free(cur); return 0; }
    for (j = 0; j <= n; j++) prev[j] = -2 * j;
    for (i = 1; i <= n; i++) {
        char ai = a[i - 1];
        cur[0] = -2 * i;
        for (j = 1; j <= n; j++) {
            int d = prev[j - 1] + (ai == b[j - 1] ? MATCH : MISMATCH);
            int u = prev[j] + GAP;
            int l = cur[j - 1] + GAP;
            int bst = d;
            if (u > bst) bst = u;
            if (l > bst) bst = l;
            cur[j] = bst;
        }
        tmp = prev; prev = cur; cur = tmp;
    }
    r = prev[n];
    free(prev); free(cur);
    return r;
}

/* ---------- the wavefront of furrows: banded anti-diagonal DP -------------
   Cells are indexed by (t = i+j, i).  Along a fixed t nothing depends on
   anything else -- the flute's one-note-per-house clock.  Parents:
     diagonal (i-1,j-1) -> buffer t-2 at index i-1
     up       (i-1,j)   -> buffer t-1 at index i-1
     left     (i,j-1)   -> buffer t-1 at index i
   Requires W >= 1.  Each R* buffer has `slots` ints; usable index range of
   (R+2) is [-2, slots-3].                                                   */
static int nw_band(int n, const char *pa, const char *pb, int W,
                   int *R0, int *R1, int *R2, int slots)
{
    size_t nb = (size_t)slots * sizeof(int);
    int *p2, *p1, *cu, t, res = 0;
#if defined(__AVX2__)
    const __m256i vg  = _mm256_set1_epi32(GAP);
    const __m256i vp1 = _mm256_set1_epi32(MATCH);
    const __m256i vm1 = _mm256_set1_epi32(MISMATCH);
#endif
    memset(R0, 0xC0, nb); memset(R1, 0xC0, nb); memset(R2, 0xC0, nb);
    p2 = R0 + 2; p1 = R1 + 2; cu = R2 + 2;
    p2[0] = 0;                               /* dp[0][0], anti-diagonal t=0 */

    for (t = 1; t <= 2 * n; t++) {
        int lo = (t - W + 1) >> 1;           /* ceil((t-W)/2) */
        int hi = (t + W) >> 1;               /* floor((t+W)/2) */
        int c  = t - n;
        int ilo, ihi, i, *tmp;
        if (c > lo) lo = c;
        if (lo < 0) lo = 0;
        if (hi > t) hi = t;
        if (hi > n) hi = n;
        ilo = lo; ihi = hi;
        if (lo == 0) ilo = 1;                /* (0,t) is a boundary cell   */
        if (hi == t) ihi = t - 1;            /* (t,0) is a boundary cell   */

        i = ilo;
#if defined(__AVX2__)
        for (; i <= ihi; i += 8) {
            uint64_t bv, bvr;
            __m128i va, vb, eq;
            __m256i sv, d, u, l, bst;
            /* 8 beasts of furrow 1, ascending */
            va = _mm_loadl_epi64((const __m128i *)(pa + (i - 1)));
            /* 8 beasts of furrow 2, descending (j = t-i falls as i rises)  */
            memcpy(&bv, pb + (t - i - 8), 8);
            bvr = __builtin_bswap64(bv);
            vb  = _mm_loadl_epi64((const __m128i *)&bvr);
            eq  = _mm_cmpeq_epi8(va, vb);            /* lotus vs knot      */
            sv  = _mm256_blendv_epi8(vm1, vp1, _mm256_cvtepi8_epi32(eq));
            d   = _mm256_loadu_si256((const __m256i *)(p2 + i - 1));
            u   = _mm256_loadu_si256((const __m256i *)(p1 + i - 1));
            l   = _mm256_loadu_si256((const __m256i *)(p1 + i));
            bst = _mm256_add_epi32(d, sv);
            bst = _mm256_max_epi32(bst, _mm256_add_epi32(u, vg));
            bst = _mm256_max_epi32(bst, _mm256_add_epi32(l, vg));
            _mm256_storeu_si256((__m256i *)(cu + i), bst);
        }
#else
        for (; i <= ihi; i++) {
            int j = t - i;
            int s = (pa[i - 1] == pb[j - 1]) ? MATCH : MISMATCH;
            int d = p2[i - 1] + s;
            int u = p1[i - 1] + GAP;
            int l = p1[i] + GAP;
            int bst = d;
            if (u > bst) bst = u;
            if (l > bst) bst = l;
            cu[i] = bst;
        }
#endif
        /* boundary cells written after the loop: a vector tail may have
           spilled junk over index t (harmless elsewhere, never read)      */
        if (lo == 0) cu[0] = -2 * t;
        if (hi == t) cu[t] = -2 * t;
        cu[lo - 1] = NEG_BIG;                /* the flying fish            */
        cu[hi + 1] = NEG_BIG;

        if (t == 2 * n) res = cu[n];
        tmp = p2; p2 = p1; p1 = cu; cu = tmp;
    }
    return res;
}

/* ------------------------------- kernel ---------------------------------- */
int kernel(int n, const char *a, const char *b)
{
    int P0 = 0, M1 = 0, M2 = 0;
    int minf1 = 0, minf2 = 0;
    int bestX1 = 0x3FFFFFFF, bestX2 = 0x3FFFFFFF;
    int q, d0, LB, s1, s2, W, Wtry, Wneed, S, slots;
    int *R, *R0, *R1, *R2;
    char *PA, *pa, *pb;

    if (n <= 0) return 0;
    if (n < 32) return nw_simple(n, a, b);

    /* ---- the horse's walk: every 0- and 1-slip trial, O(n) total --------
       trial (down at p, right at q>=p):  X = d0 + f(p) + g(q)
       f1(p)=P0[p]-M1[p]  g1(q)=M1[q]-P0[q+1]   (furrow 2 slips forward)
       f2/g2: the mirror trial (furrow 1 slips forward).                   */
    for (q = 0; q < n; q++) {
        int f1 = P0 - M1, f2 = P0 - M2, P0n, c1, c2;
        if (f1 < minf1) minf1 = f1;
        if (f2 < minf2) minf2 = f2;
        P0n = P0 + (a[q] != b[q]);
        c1 = minf1 + M1 - P0n; if (c1 < bestX1) bestX1 = c1;
        c2 = minf2 + M2 - P0n; if (c2 < bestX2) bestX2 = c2;
        if (q + 1 < n) { M1 += (a[q + 1] != b[q]); M2 += (a[q] != b[q + 1]); }
        P0 = P0n;
    }
    d0 = P0;

    /* ---- weigh the fists, keep the lightest ---------------------------- */
    LB = n - 2 * d0;                       /* the horse stands nowhere     */
    s1 = n - 5 - 2 * bestX1; if (s1 > LB) LB = s1;
    s2 = n - 5 - 2 * bestX2; if (s2 > LB) LB = s2;

    /* score <= n - 5*deviation  =>  optimal path lives in |i-j| <= W       */
    W = (n - LB) / 5;
    if (W <= 1) return LB;   /* regime A: <=1 slip is provably optimal, O(n) */
    if (W > n) W = n;

    /* ---- regime B: banded wavefront ------------------------------------ */
    slots = n + 26;
    R  = (int  *)malloc((size_t)slots * 3 * sizeof(int));
    PA = (char *)malloc((size_t)(n + 80) * 2);
    if (!R || !PA) { free(R); free(PA); return nw_simple(n, a, b); }
    memset(PA, 0, (size_t)(n + 80) * 2);
    pa = PA + 40;
    pb = PA + (n + 80) + 40;
    memcpy(pa, a, (size_t)n);
    memcpy(pb, b, (size_t)n);
    R0 = R; R1 = R + slots; R2 = R + 2 * slots;

    Wtry = (W < 16) ? W : 16;              /* cheap probe tightens LB      */
    S = nw_band(n, pa, pb, Wtry, R0, R1, R2, slots);
    Wneed = (n - S) / 5;
    if (Wneed > Wtry) {
        if (Wneed > n) Wneed = n;
        S = nw_band(n, pa, pb, Wneed, R0, R1, R2, slots);
    }
    free(R); free(PA);
    return S;
}
```

# PREDICTION

**PREDICTION: speedup_vs_dp = 8**

(stated before any measurement; no tools were available in this session, so this is an *a priori* estimate, not a tuned one.)

Reasoning behind the number, for random DNA (`d₀ ≈ 0.75n`): band ≈ 0.6n² cells → **1.7×**; AVX2 int32, 8 houses per note, branch-free → **≈3–4×**; O(n) working set instead of a `(n+1)²` int table (67 MB at n = 4096) → a further large but unquantified cache/bandwidth win at big n. Sub-predictions I will be judged on: for ≥90 %-identical inputs the `W ≤ 1` path returns in O(n), speedup **>100×**; for adversarial all-mismatch inputs `W = 2n/5`, band ≈ 0.8n², speedup **≈4×** — never below 1×.

# MEASUREMENT

**Not measured.** `alignment_bench` and `alignment_contract` were unavailable in this session, so I ran zero of the four permitted improvement rounds and have no timing or correctness number to report. I will not dress an estimate up as a measurement. What I can state is what I verified by hand:

* the bound `score ≤ n − 5δ` proved above (down-moves = right-moves = k, aligned columns = n − k, δ ≤ k);
* the `W ≤ 1 ⇒ return LB` shortcut proved exact (n − LB ≤ 9 forces k ≤ 1, and all k ≤ 1 alignments are enumerated);
* the two-pass refinement proved exact (`S₁ ≥ LB ⇒ Wneed ≤ W`, and `S₂ ≥ S₁ ⇒ δ* ≤ Wneed`);
* the anti-diagonal indexing hand-traced cell by cell for `n = 4, W = 2` against the reference recurrence (t = 0…4, including both boundary cells and the out-of-band `dp[0][3]`, which cannot change the answer);
* buffer-lifetime analysis: `lo(t)` and `hi(t)` are monotone with step ≤ 1, so writing sentinels at `lo−1` and `hi+1` is sufficient, and vector-tail spill above `hi+1` is never read;
* all out-of-range byte accesses absorbed by 40-byte pads on both ends of both sequences.

# VERDICT

The native's horse is not a heuristic dressed as an algorithm — it is a **certificate generator**. Taking "the one permitted slip placed where it does the least harm" completely literally yields, in O(n), a real alignment whose score is a lower bound; and the arithmetic of equal-length global alignment (`score ≤ n − 5δ`) converts that bound directly into a jail cell for the optimal path. The flying fish then eat everything outside it. That the result is *exactly* score-bounded banded alignment with iterative refinement — a technique real aligners already ship — is the right outcome, not a disappointment: per step 4 I let the mechanism arrive at the validated method rather than ship a novelty.

The genuinely surprising thing the metaphor produced is the `W ≤ 1` shortcut: for highly similar sequences the horse's walk *is* the complete answer, provably, in linear time with no DP at all. The textbook method has no such off-ramp because it insists a gap can only be discovered after comparing the position before it.

Honest statement of risk, and how each is guarded:

* **Small n** — the horse-walk plus two mallocs would dominate. Guarded: `n < 32` → plain two-row DP.
* **Wide-band worst case** (all-mismatch inputs, `W = 2n/5`) — still 0.8n² cells, strictly below the reference's n², under SIMD and with an O(n) working set. Bounded above by the known way by construction; nothing to guard.
* **Probe pass overhead** — 32n cells, `O(1/n)` of the main pass.
* **No AVX2 / allocation failure** — compile-time scalar path and runtime `nw_simple` fallback.
* **Threading** — deliberately absent. The metaphor's independent units (the trials) are the O(n) walk, too small to thread; the wavefront is serial across `t`. Adding OpenMP here would be overhead with no unit of work to justify it.

The unguarded risk that remains is the one I cannot retire from here: **this code has never been compiled or run.** The band mathematics and the index algebra are proved and hand-traced, but an untested SIMD kernel is an untested SIMD kernel, and the honest confidence interval on "matches the reference exactly" is high-but-not-one. If it fails, the first thing to check is the byte-reversal of the `b` load (`__builtin_bswap64` lane order) and the boundary-cell writes at `lo == 0` / `hi == t`; the scalar `#else` branch is the same algorithm without those two hazards and can be forced on to isolate them.