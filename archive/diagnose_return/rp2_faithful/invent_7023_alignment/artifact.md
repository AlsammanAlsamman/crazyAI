## MAPPING

**SEED 1 — the doubled shadow**

| world | problem |
|---|---|
| two grass-ropes, knots dyed in 4 berry-colors | the two length-`n` strings over {A,C,G,T} |
| rope hung along the east wall | `a`, indexed by `i` |
| rope stretched from the doorway **at a slant** | `b` laid along an anti-diagonal ⇒ read reversed (`brev[t] = b[n-1-t]`) |
| empty room with no tiles laid | the DP table, never materialized as a grid |
| a crossing of two knots | cell `(i,j)` |
| the low sun through the one window | one vector pass of light: `_mm256`-width compare |
| two same-dye knots throwing **one** shadow instead of two | 16/32 lanes of `a[i-1]==brev[i+base]` resolved in a single compare, because on an anti-diagonal `i+j` is constant and the `b`-index runs *backwards* as `i` runs forwards |
| finding every treasure at once without walking knot by knot | the whole anti-diagonal's match/mismatch scores obtained without a per-cell branch |

Breaks: **"one pair of positions is judged at a time"** and **"both strings are read start to end in the same direction"** (the slant is literally a reversal).

**SEED 2 — the corridor's cost read off the rope lengths**

| world | problem |
|---|---|
| bare floor between two treasures | non-match region of the table |
| "never lay a marker on the bare floor between" | never evaluate cells that cannot lie on an optimal path |
| "how many knots forward on the wall-rope and how many on the floor-rope" | `di`, `dj`, i.e. the offset `k = i-j` and the remaining lengths `n-i`, `n-j` |
| cost read like counting doorframes instead of pacing | closed form: a prefix through offset `k` scores `≤ min(i,j)-2k`, a suffix `≤ min(n-i,n-j)-2k`, so **any path through offset `k` scores ≤ n − 5k** |
| "the corridor's cost was never a secret — only its length was, and the ropes already told me that" | the ropes give the no-slip tally `L₀ = Σ ±1` in O(n); then every offset with `n−5k < L₀` is *provably* floor never worth laying a marker on ⇒ band `w = ⌊(n−L₀)/5⌋` |

Breaks: **"the whole grid of every position against every other must be filled in."**

**SEED 3 — drop the bad thread-end**

| world | problem |
|---|---|
| two tallies arriving at one treasure | two candidate scores reaching one cell |
| keep the cheapest, let the other fall | the `max` reduction / dominance pruning |
| "slip spent, or not yet spent" | the offset coordinate `k`: paths that have drifted off the diagonal vs. those still on it |
| never picking the dropped end back up | no back-pointers, no traceback, O(n) memory: three rotating anti-diagonals |

Breaks: **"the whole grid must be filled in"** (weakly — it prunes states, not cells).

## CHOSEN SEED

**SEED 2.** It is the one seed that attacks the whole-grid assumption head-on and its mapping is arithmetic-literal: the native's "cost read off the ropes' lengths" is exactly the closed-form bound `n − 5k`, and their "one slip, a few knots" is exactly the band half-width. SEED 1 supplies the primitive *inside* it (the shadow = the vector match compare, which is why the floor-rope must be laid at a slant/reversed) and SEED 3 supplies the cell update (`max`, three rotating diagonals, no traceback). All three are one procedure in the native's account, so building SEED 2 keeps them.

## ASSUMPTION BROKEN

*The whole grid of every position against every other must be filled in.* The corridor's price per unit of slip is known in advance (5 points of score per unit of offset, derived below), and the no-slip tally is free, so the room is only ever swept within `|i−j| ≤ ⌊(n−L)/5⌋`. Secondarily: *both strings read in the same direction* (the slant forces `b` reversed) and *one pair judged at a time* (one sun, many shadows).

Proof of the bound (this is the native's "cost of the corridor, read not walked"): with `M` matches, `X` mismatches, `G` gap symbols, `2(M+X)+G = i+j` on a prefix and `G ≥ |i−j| = k`, so prefix score `= M−X−2G ≤ (i+j)/2 − 2.5G ≤ min(i,j) − 2k`; symmetrically the suffix `≤ min(n−i,n−j) − 2k`; and `min(i,j)+min(n−i,n−j) = n−k`. Hence any alignment through offset `k` scores `≤ n − 5k`. Any `k` with `n−5k < L` (L an achieved score) is unreachable by an optimum ⇒ exactness is preserved, not approximated.

## ARTIFACT

- `nw_small` — the simpler path, for `n < 64` (guard for the overhead risk named in VERDICT).
- `brev` build — SEED 1: the floor-rope laid at a slant, so that one pass of light lines both ropes up.
- `diag` loop — SEED 2's free reading: the tally with the slip **not yet spent**, `L₀`.
- probe pass `w=8`, then `L = max(L₀,L₁)` — the native looking a few knots to either side to see if the ropes fall back into step; a *better* lower bound tightens the band. `if (w <= 8) return L1;` is the "ropes already in step" shortcut, O(n).
- `w = (n − L)/5` — SEED 2: the corridor's price per knot of slip, read off the ropes; the **runtime regime discriminator** (small `w` = few-slips regime, `w→n` = many-slips regime, same loop, graceful).
- `WAVE(...)` inner `omp simd` loop — the room swept along the slant: `p2[i-1]+sc` (diagonal/treasure step), `p1[i-1]+GAP`, `p1[i]+GAP` (the two slips); `sc` from the single compare `a[i-1]==brev[i+base]` = the doubled shadow; the three `if (x>v)` = SEED 3 dropping the worse tally; three rotating rows = no traceback.
- `NEG` clamp + sentinel writes at `lo-1`/`hi+1` — the bare floor outside the corridor, kept unlit and prevented from drifting so 16-bit lanes stay legal (provably harmless: a clamped value can never reach `(n,n)` above `−n`).
- `n <= 6000` → `int16_t` (16 lanes), else `int32_t` (8 lanes) — overflow guard, single-sourced body via macro.

```c
#include <stdlib.h>
#include <stdint.h>

#define MATCH     1
#define MISMATCH -1
#define GAP      -2

/* ---- simpler path: exact rolling-row Needleman-Wunsch (small n / fallback) ---- */
static int nw_small(int n, const char *a, const char *b)
{
    int *prev = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int *cur  = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int r = 0;
    if (!prev || !cur) { free(prev); free(cur); return 0; }
    for (int j = 0; j <= n; j++) prev[j] = j * GAP;
    for (int i = 1; i <= n; i++) {
        char ai = a[i - 1];
        cur[0] = i * GAP;
        for (int j = 1; j <= n; j++) {
            int best = prev[j - 1] + (ai == b[j - 1] ? MATCH : MISMATCH);
            int up = prev[j] + GAP, lf = cur[j - 1] + GAP;
            if (up > best) best = up;
            if (lf > best) best = lf;
            cur[j] = best;
        }
        { int *t = prev; prev = cur; cur = t; }
    }
    r = prev[n];
    free(prev); free(cur);
    return r;
}

/* ---- the swept corridor: anti-diagonal wavefront, band half-width w ---- */
#define DEFINE_WAVE(FN, T, NEGV)                                               \
static int FN(int n, const char *a, const char *brev, int w, void *scratch)    \
{                                                                              \
    const int stride = n + 2;                                                  \
    const T NEG = (T)(NEGV);                                                   \
    T *base_ = (T *)scratch;                                                   \
    T *p2 = base_, *p1 = base_ + stride, *cu = base_ + 2 * stride;             \
    for (int t = 0; t < 3 * stride; t++) base_[t] = NEG;                       \
    p1[0] = (T)0;                                /* cell (0,0), diagonal d=0 */\
    p1[1] = NEG;                                                               \
    for (int d = 1; d <= 2 * n; d++) {                                         \
        int lo = d - n; if (lo < 0) lo = 0;                                    \
        int tt = d - w;                                                        \
        if (tt > 0) { int lb = (tt + 1) >> 1; if (lb > lo) lo = lb; }          \
        int hi = d; if (hi > n) hi = n;                                        \
        { int hb = (d + w) >> 1; if (hb < hi) hi = hb; }                       \
        if (lo <= hi) {                                                        \
            if (lo == 0) cu[0] = (T)(-2 * d);        /* rope-edge: (0,d) */     \
            if (hi == d) cu[d] = (T)(-2 * d);        /* rope-edge: (d,0) */     \
            {                                                                  \
                const int ilo = (lo > 1) ? lo : 1;                             \
                const int ihi = (hi < d - 1) ? hi : d - 1;                     \
                const int off = n - d;                                         \
                const char *restrict ra = a;                                   \
                const char *restrict rb = brev;                                \
                T *restrict A2 = p2;                                           \
                T *restrict A1 = p1;                                           \
                T *restrict AC = cu;                                           \
                _Pragma("omp simd")                                            \
                for (int i = ilo; i <= ihi; i++) {                             \
                    T sc = (ra[i - 1] == rb[i + off]) ? (T)MATCH : (T)MISMATCH;\
                    T v = (T)(A2[i - 1] + sc);                                 \
                    T u = (T)(A1[i - 1] + (T)GAP);                             \
                    T l = (T)(A1[i] + (T)GAP);                                 \
                    if (u > v) v = u;                                          \
                    if (l > v) v = l;                                          \
                    if (v < NEG) v = NEG;                                      \
                    AC[i] = v;                                                 \
                }                                                              \
            }                                                                  \
            if (lo >= 1)        cu[lo - 1] = NEG;   /* unlit bare floor */      \
            if (hi + 1 <= n + 1) cu[hi + 1] = NEG;                             \
        }                                                                      \
        { T *tmp = p2; p2 = p1; p1 = cu; cu = tmp; }                           \
    }                                                                          \
    return (int)p1[n];                                                         \
}

DEFINE_WAVE(wave16, int16_t, -20000)
DEFINE_WAVE(wave32, int32_t, -100000000)

int kernel(int n, const char *a, const char *b)
{
    if (n <= 0) return 0;
    if (n < 64) return nw_small(n, a, b);                 /* overhead guard */

    /* SEED 1: the floor-rope laid at a slant, so one sun lights both at once */
    char *brev = (char *)malloc((size_t)n + 64);
    if (!brev) return nw_small(n, a, b);
    for (int t = 0; t < n; t++) brev[t] = b[n - 1 - t];

    size_t stride = (size_t)n + 2;
    void *sc16 = NULL, *sc32 = NULL;
    int use16 = (n <= 6000);
    if (use16) sc16 = malloc(3 * stride * sizeof(int16_t));
    else       sc32 = malloc(3 * stride * sizeof(int32_t));
    if (!sc16 && !sc32) { free(brev); return nw_small(n, a, b); }

    /* SEED 2: the no-slip tally, read straight off the two ropes */
    int L = 0;
    #pragma omp simd reduction(+:L)
    for (int i = 0; i < n; i++) L += (a[i] == b[i]) ? MATCH : MISMATCH;

    /* a few knots to either side: cheap probe that tightens the bound */
    int probe = use16 ? wave16(n, a, brev, 8, sc16)
                      : wave32(n, a, brev, 8, sc32);
    if (probe > L) L = probe;

    /* the corridor's price: 5 points of score per knot of slip  =>  band */
    long long wl = ((long long)n - (long long)L) / 5;
    int w = (wl > (long long)n) ? n : (int)wl;
    if (w < 1) w = 1;

    int res;
    if (w <= 8) {
        res = probe;                      /* ropes already in step: O(n) done */
    } else {
        res = use16 ? wave16(n, a, brev, w, sc16)
                    : wave32(n, a, brev, w, sc32);
    }
    free(sc16); free(sc32); free(brev);
    return res;
}
```

## PREDICTION

PREDICTION: speedup_vs_dp = 18

(Reasoning stated before any run: ~16 int16 lanes × ~0.3 useful cells/lane-op of loop overhead ≈ 10–14× per cell from the slant sweep and the O(n) working set, times ≈1.67× fewer cells from the band at the random-DNA fixed point `w ≈ 0.3n` — the band cannot do better than ≈1.85× on random input because `L ≈ −0.4n` pins `w`; near-identical inputs would instead give 100×+.)

## MEASUREMENT

**Not measured.** `alignment_bench` / `alignment_contract` were unavailable in this session, so no number here is empirical and I will not present one as if it were. What I did verify by hand, and what a run should check:

- Index algebra traced cell-by-cell for `n=2`, `w=1` (`a="AC"`, `b="AC"`): the wavefront reproduced `dp[1][1]=1`, `dp[1][2]=dp[2][1]=−1`, `dp[2][2]=2`. The `+1`-per-diagonal advance of `lo`/`hi`, and the fact that reads reach at most one slot beyond a source diagonal's own window, is why a single sentinel write at each end per diagonal is sufficient — checked against `lo` non-decreasing and `hi` increasing by ≤1.
- Band exactness rests on `score ≤ n − 5k`, derived above; `L` is always an *achieved* alignment score, so pruning is lossless rather than heuristic.
- Expected failure modes a run would expose: (i) the compiler declining to vectorize the mixed `char`-compare/`int16`-arithmetic body, which would cut the predicted 18× to ~2× — that is the single largest risk and the honest reason the prediction could be badly high; (ii) per-diagonal bookkeeping dominating when `w` is small relative to 2n of loop trips.

## VERDICT

The mechanism is the native's, not the textbook's: the room is swept along the slant with one rope reversed so a single pass of light yields a whole anti-diagonal of treasures, and the corridor between them is *priced* rather than *walked* — 5 points of score per knot of slip — which is what licenses never laying a marker outside `|i−j| ≤ (n−L)/5`. Exactness is proved, not hoped for.

Named risks, all addressed rather than shipped: small-`n` overhead → `n < 64` falls back to the plain rolling-row DP; 16-bit overflow → `n > 6000` switches to 32-bit lanes, and the `NEG` clamp is proved unable to inflate the answer; a band that buys nothing on dissimilar sequences → `w` saturates at `n` and the kernel degenerates to the same full slant sweep, never to something worse than a full SIMD DP. Two regimes are recognized at runtime from the ropes themselves (no-slip tally plus an 8-knot probe): few-slips takes the O(n·w) corridor and can exit in O(n); many-slips takes the full wavefront. No thread parallelism: consecutive anti-diagonals are strictly dependent and a diagonal is only ~0.3n elements at these sizes, so the metaphor's unit of work is too small to be worth a barrier.

The part I would attack next, given a bench: the band's fixed point. `w ≈ 0.3n` for random DNA is set by how low `L` is, and a better lower bound is the cheapest possible improvement — a mid-flight re-tightening of `w` from the running best on each diagonal (the native shrinking the corridor as they walk it) is the natural fifth step and is still pure SEED 2.