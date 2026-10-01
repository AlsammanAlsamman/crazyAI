## MAPPING

### SEED 1 — "coils shaped like symbols walked side by side on a sunlit water table, each match dropping a fire counted below"

| World object | Problem object |
|---|---|
| Cord #1's coils, near edge, in full sun | `a[0..n-1]`, the near row, held still |
| Cord #2's coils, in the shadow-line, "queued" | `b[0..n-1]`, the row that is allowed to move |
| A coil "shaped like a symbol" | one base ∈ {A,C,G,T}, one byte |
| Sun windowing straight through both | `a[i] == b[j]` — an equality test, not a table lookup |
| A fire dropping on the water | `+1`; a quarrel (no fire) is `-1` |
| "I count the fires, nothing more" | the running scalar sum; no `dp[i][j]` grid is ever materialised |
| Walking the two rows together | one linear pass, position *i* against position *i* |
| The water table under both rows | a single accumulator (or an 8-lane AVX2 accumulator) |

**Silent assumption broken:** *"the whole grid of every position against every other must be filled in."* The native never builds a grid; he builds one *line* of comparisons and one scalar.

---

### SEED 2 — "a boundary door allowed to break exactly once per crossing; one coil crouches past a quarrel out of step"

| World object | Problem object |
|---|---|
| The door between the two rows | the diagonal boundary `i − j = 0` in the alignment grid |
| The door breaking | an indel: the offset `i − j` changes |
| "the shadow-row's next coil crouches forward, skipping the coil that quarreled" | delete `b[p]` and shift: `a[i] ~ b[i+1]` for all `i ≥ p` |
| "I try this breaking at **every place** a quarrel happens" | **every slip point `p` is tried at once, in parallel, without having compared `p−1` first** |
| "never twice in one crossing" | the path's maximum diagonal offset is capped at 1 |
| Crossing still "ends at the queen's threshold" | the alignment is *global*: the shift must be paid for at the end too (2 gaps, −4) |

**Silent assumption broken:** ***"a slip (gap) can only be discovered by having already compared the position before it."*** This is the one the brief asks for. In the native's world every candidate slip point is an independent crossing evaluated on its own; the slip at `p` is priced by a **prefix sum before `p`** and a **suffix sum after `p`**, both computable without any left-to-right DP dependency. It also breaks *"every cell depends on above/left/diagonal, in that order"* and *"one pair of positions is judged at a time."*

---

### SEED 3 — "every crossing's pile in its own room; twice-broken rows sink into the tower unread; only the highest pile comes back up"

| World object | Problem object |
|---|---|
| A room per crossing | one candidate score per slip point `p` |
| The pile of fire in a room | `pre[p] + suf[p] − 4` |
| Bringing up the highest pile | `max` reduction over all rooms |
| The tower under the ground, entered "unread" | **pruning**: whole classes of alignments discarded without evaluating them |
| "twice-broken rows sink, unread" | any path with max offset `d` has score ≤ `n − 5d` and is thrown away if that is below the best pile already held |
| "how many breaks I still trust" | the band half-width `W` |

**Silent assumption broken:** *"the whole grid must be filled in"* — plus the deeper one that you must *evaluate* an alignment to reject it. The tower rejects by arithmetic.

---

## CHOSEN SEED

**SEED 2**, because it is the one that breaks *"a slip (gap) can only be discovered by having already compared the position before it."* It is also the most literal: prefix-before / suffix-after is exactly "coils already walked" / "coils still queued," and "I try this at every place a quarrel happens" is exactly a `max` over all `p` computed with **no** left-to-right recurrence.

SEEDs 1 and 3 are not discarded — they are the two halves of the same machine: SEED 1 is the O(n) sunlit pass that produces the piles, SEED 3 is the tower that decides how much of the grid never needs to exist.

---

## ASSUMPTION BROKEN

> **A slip (gap) can only be discovered by having already compared the position before it.**

In the kernel, a slip at position `p` is priced as `pre[p] + suf[p] − 4` where `pre` is a forward scan and `suf` a backward scan. Both scans are computed once, independently, in O(n); every slip point is then scored in O(1) and reduced by `max`. No cell of a DP table is consulted. The gap is discovered *from both ends at once*, not from its left neighbour.

Second assumption broken, by the tower: **the whole grid need not be filled in.** And note — the tower's arithmetic lands the mechanism exactly on a **validated, real-world technique**: banded Needleman–Wunsch with a score-derived band bound plus band doubling (the edlib / ksw2 family). That is deliberate, per step 4: the metaphor *arrives at* the known-good method rather than inventing a replacement for it.

---

## THE MECHANISM, MADE LITERAL

*What is memory:* `a` and `b` themselves, plus two rows and one scratch row — never a grid.
*What stays still:* the near row `a` (full sun).
*What flows:* the shadow row `b`, which may crouch forward.
*What is a processor:* one AVX2 lane per coil — 8 coils judged in one sun-flash.
*What is time:* the row index `i`; within a row, all 8 lanes are simultaneous.
*What is the tower:* the inequality `score ≤ n − 5k` for any alignment using `k` gap-pairs.

The proof the tower rests on (equal-length strings, so insertions = deletions = `k`):
aligned columns = `n − k`, gap cost = `−4k`, so **score ≤ (n − k) − 4k = n − 5k**, and the path's maximum diagonal offset satisfies `d_max ≤ k`. Therefore if `L` is *any* achievable score, every optimal path lies inside the band `|i − j| ≤ ⌊(n − L)/5⌋`. The sun-walk supplies `L` in O(n). The tower converts it into a band. Nothing is approximated: the result is bit-exact Needleman–Wunsch.

**Regime detection (step 5), encoded in the metaphor itself:** the height of the best pile *is* the regime sensor. Tall pile (similar cords) → `W` collapses to 0–2 → the kernel is effectively O(n). Short pile (divergent cords) → `W` grows toward `0.3n` → the kernel degrades gracefully to a banded, then a full, cache-resident two-row DP. A `n < 64` guard skips the whole apparatus for tiny inputs and runs plain DP, so the mechanism can never be slower than the simple path on small problems. No thread parallelism is used: the metaphor's unit of work is one row of a band, which at these sizes is far too small to pay for a fork/join — vectorisation only, as instructed.

---

## ARTIFACT

```c
#include <stdlib.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define K_MATCH      ( 1)
#define K_MISMATCH   (-1)
#define K_GAP        (-2)
#define K_NEG        (-(1<<27))   /* out-of-band sentinel                */
#define K_FILL       (-(1<<29))   /* lane-fill for the SIMD prefix scan  */

/* ---- plain two-row DP: exact, used for tiny n and as the wide fallback ---- */
static int kd_full(int n, const char *a, const char *b, int *P, int *C)
{
    int i, j, *prev = P, *cur = C;
    for (j = 0; j <= n; j++) prev[j] = K_GAP * j;
    for (i = 1; i <= n; i++) {
        char ai = a[i - 1];
        cur[0] = K_GAP * i;
        for (j = 1; j <= n; j++) {
            int d = prev[j - 1] + ((ai == b[j - 1]) ? K_MATCH : K_MISMATCH);
            int u = prev[j] + K_GAP;
            int l = cur[j - 1] + K_GAP;
            int v = d > u ? d : u;
            if (l > v) v = l;
            cur[j] = v;
        }
        { int *t = prev; prev = cur; cur = t; }
    }
    return prev[n];
}

/* ---- the crossing: only coils within W door-breaks of the diagonal ----
   Exact whenever W >= max diagonal offset of some optimal path.
   Phase 1 (the sun-flash) is dependency-free -> 8 coils per instruction.
   Phase 2 (left-gap run) is a max-prefix-scan with decay 2, done as
   t[j]=base[j]+2j, prefix-max, minus 2j -> also vectorised.            */
static int kd_band(int n, const char *a, const char *b, int W,
                   int *P, int *C, int *B)
{
    int i, j;
    int *prev = P, *cur = C, *base = B;
    int hi0 = (W < n) ? W : n;

    for (j = 0; j <= hi0; j++) prev[j] = K_GAP * j;
    prev[hi0 + 1] = K_NEG;

#if defined(__AVX2__)
    const __m256i v1   = _mm256_set1_epi32(1);
    const __m256i vm1  = _mm256_set1_epi32(-1);
    const __m256i vgap = _mm256_set1_epi32(K_GAP);
    const __m256i vfil = _mm256_set1_epi32(K_FILL);
    const __m256i vidx = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
#endif

    for (i = 1; i <= n; i++) {
        int lo = i - W, hi = i + W, jstart;
        char ai = a[i - 1];
        if (lo < 0) lo = 0;
        if (hi > n) hi = n;
        if (lo == 0) { cur[0] = K_GAP * i; jstart = 1; }
        else         { cur[lo - 1] = K_NEG; jstart = lo; }

        /* ---- phase 1: base[j] = max(diag, up) : no loop-carried dep ---- */
        j = jstart;
#if defined(__AVX2__)
        {
            __m256i vai = _mm256_set1_epi32((int)(unsigned char)ai);
            for (; j + 7 <= hi; j += 8) {
                __m128i cb = _mm_loadl_epi64((const __m128i *)(b + j - 1));
                __m256i vb = _mm256_cvtepu8_epi32(cb);
                __m256i eq = _mm256_cmpeq_epi32(vb, vai);
                __m256i sc = _mm256_blendv_epi8(vm1, v1, eq);
                __m256i pd = _mm256_loadu_si256((const __m256i *)(prev + j - 1));
                __m256i pu = _mm256_loadu_si256((const __m256i *)(prev + j));
                _mm256_storeu_si256((__m256i *)(base + j),
                    _mm256_max_epi32(_mm256_add_epi32(pd, sc),
                                     _mm256_add_epi32(pu, vgap)));
            }
        }
#endif
        for (; j <= hi; j++) {
            int d = prev[j - 1] + ((ai == b[j - 1]) ? K_MATCH : K_MISMATCH);
            int u = prev[j] + K_GAP;
            base[j] = d > u ? d : u;
        }

        /* ---- phase 2: cur[j] = max(base[j], cur[j-1] + GAP) ---- */
        {
            int carry = cur[jstart - 1];
            j = jstart;
#if defined(__AVX2__)
            for (; j + 7 <= hi; j += 8) {
                __m256i v = _mm256_add_epi32(
                        _mm256_loadu_si256((const __m256i *)(base + j)), vidx);
                __m256i t, r;
                t = _mm256_permute2x128_si256(v, v, 0x08);
                r = _mm256_blend_epi32(_mm256_alignr_epi8(v, t, 12), vfil, 0x01);
                v = _mm256_max_epi32(v, r);
                t = _mm256_permute2x128_si256(v, v, 0x08);
                r = _mm256_blend_epi32(_mm256_alignr_epi8(v, t, 8),  vfil, 0x03);
                v = _mm256_max_epi32(v, r);
                r = _mm256_permute2x128_si256(v, v, 0x08);
                r = _mm256_blend_epi32(r, vfil, 0x0F);
                v = _mm256_max_epi32(v, r);
                v = _mm256_max_epi32(v, _mm256_set1_epi32(carry + K_GAP));
                _mm256_storeu_si256((__m256i *)(cur + j),
                                    _mm256_sub_epi32(v, vidx));
                carry = _mm256_extract_epi32(v, 7) - 14;
            }
#endif
            for (; j <= hi; j++) {
                int v = base[j];
                int l = carry + K_GAP;
                if (l > v) v = l;
                cur[j] = v;
                carry  = v;
            }
        }

        cur[hi + 1] = K_NEG;
        { int *t = prev; prev = cur; cur = t; }
    }
    return prev[n];
}

int kernel(int n, const char *a, const char *b)
{
    int *mem, *P, *C, *B, *pre, *suf;
    int i, s, L, Wb, W, R;

    if (n <= 0) return 0;

    mem = (int *)malloc((size_t)5 * (size_t)(n + 4) * sizeof(int));
    if (!mem) return 0;
    P = mem; C = P + (n + 4); B = C + (n + 4);
    pre = B + (n + 4); suf = pre + (n + 4);

    /* size guard: below this the machinery costs more than it saves */
    if (n < 64) { R = kd_full(n, a, b, P, C); free(mem); return R; }

    /* ================= THE SUN-WALK (SEED 1 + SEED 2) =================
       Straight crossing, plus every single-door-break crossing in both
       directions, each scored as prefix-before + suffix-after - 4.
       Every slip point is judged without having compared the one before
       it.  O(n), auto-vectorised.                                      */
    s = 0; pre[0] = 0;
    for (i = 0; i < n; i++) { s += (a[i] == b[i]) ? 1 : -1; pre[i + 1] = s; }
    L = s;                                   /* the straight crossing    */

    {   /* shadow row crouches forward: a[i] ~ b[i+1] for i >= p        */
        int t = 0;
        suf[n - 1] = 0;
        for (i = n - 2; i >= 0; i--) { t += (a[i] == b[i + 1]) ? 1 : -1; suf[i] = t; }
        for (i = 0; i < n; i++) { int v = pre[i] + suf[i] - 4; if (v > L) L = v; }
        /* mirror: near row crouches forward: a[i+1] ~ b[i] for i >= p  */
        t = 0; suf[n - 1] = 0;
        for (i = n - 2; i >= 0; i--) { t += (a[i + 1] == b[i]) ? 1 : -1; suf[i] = t; }
        for (i = 0; i < n; i++) { int v = pre[i] + suf[i] - 4; if (v > L) L = v; }
    }

    /* ================= THE TOWER (SEED 3) =================
       Any alignment using k gap-pairs scores at most n - 5k, and its
       maximum diagonal offset is at most k.  So every crossing that
       breaks the door more than Wb times sinks unread.                 */
    Wb = (n - L) / 5;
    if (Wb < 1) Wb = 1;
    if (Wb > n) Wb = n;

    /* regime switch: tall pile -> narrow band; short pile -> widen,
       doubling, until the tower itself certifies optimality.           */
    W = (Wb < 32) ? Wb : 32;
    for (;;) {
        R = kd_band(n, a, b, W, P, C, B);
        if (W >= Wb || W >= n)        break;   /* bound already proved  */
        if (n - 5 * (W + 1) < R)      break;   /* tighter proof from R  */
        W <<= 1;
        if (W > Wb) W = Wb;
        if (W > n)  W = n;
    }

    free(mem);
    return R;
}
```

---

## PREDICTION

**PREDICTION: speedup_vs_dp = 12**

(Stated before any measurement; no benchmark tool was reachable in this session, so this number is unvalidated.)

Reasoning behind it: the reference `malloc`s an `(n+1)²` table (16 MB at n = 2000) and streams it through memory at roughly 4 cycles/cell. My kernel touches two rows that live in L1, restricted to a band of half-width `W`. For divergent (near-random) sequences the sun-walk yields `L ≈ −0.5n`, so `W ≈ 0.3n` and the band is ~0.6 n² cells at roughly 0.5 cycles/cell with the AVX2 pair of phases — about 15×, minus a factor for the doubling loop's wasted narrow passes, giving ~8×. For similar sequences (any mutation-based generator) `W` collapses to 0–3 and the kernel is O(n) against O(n²) — hundreds of ×. Twelve is my honest midpoint across those two regimes.

---

## MEASUREMENT

**Not performed.** `alignment_bench` and `alignment_contract` were not available in this session, so I have no measured speedup and no independently verified exactness check. I am reporting that plainly rather than substituting an estimate for a measurement. What I *can* offer instead:

- **Hand-traced correctness** on `n = 1` (`A`/`C` → −1), `n = 2` (`AC`/`CA` → −2, band path traced cell-by-cell), and the one-indel case (`L = n−5`, `W = 1`), all matching the reference by hand.
- **A proof, not a heuristic**, that the band never excludes an optimal path: for equal-length strings, insertions = deletions = `k`, aligned columns = `n − k`, so `score ≤ n − 5k` and `d_max ≤ k`; with `L` achievable, any path with `d_max ≥ ⌊(n−L)/5⌋ + 1` scores strictly below `L`. Exactness therefore does not depend on the benchmark's sequence distribution.
- **A verified SIMD prefix-scan**: the decay-2 max-scan was checked by hand on `base = [5,0,0,0,0,0,0,0]` → `[5,3,1,0,0,0,0,0]`, and each `permute2x128`/`alignr` lane pattern was worked out explicitly.

The correct next step is to run `alignment_contract` first (exactness, including adversarial random pairs where the band is widest) and only then `alignment_bench`.

---

## VERDICT

The native's picture is not a loose analogy here — it is a working algorithm that a real aligner would recognise. The three seeds decompose cleanly into (1) a dependency-free O(n) sunlit pass, (2) a gap discovered from both ends simultaneously rather than from its left neighbour, and (3) a pruning rule that rejects whole families of alignments by arithmetic rather than by evaluating them. Together they are, exactly, **banded Needleman–Wunsch with a score-derived band bound and band doubling** — which is the right outcome: a validated technique reached through the metaphor beats a novel one invented to be different.

Where the native is genuinely wrong, and I will not paper over it: **one door-break is not enough.** For divergent sequences the true optimum uses many gap-pairs, and a strict one-slip kernel would return a wrong answer, not a slow one. I kept the one-slip crossing for what it is actually good at — producing a strong lower bound `L` in O(n) — and let the tower's inequality decide how many breaks are worth trusting. That is a generalisation of the native's rule, and I am flagging it as a departure rather than pretending the metaphor already contained it.

Risks I named and then addressed, as required:
- *"Overhead on small inputs"* → guarded by `n < 64` falling back to plain two-row DP.
- *"Only helps if the band is narrow"* → the doubling loop caps at `Wb` and at `n`, so the worst case is a full two-row DP, which is still strictly faster than the reference's `n²`-table version. There is no input on which this kernel is asymptotically worse.
- *"Thread parallelism might not pay"* → dropped entirely; a band row is too small a work unit at these sizes.

The honest residual risk is not correctness of the idea but correctness of the hand-written AVX2 — untested here. If `alignment_contract` fails, the first thing to do is compile with `-UAVX2`-equivalent (delete the `#if defined(__AVX2__)` blocks) and re-check: the scalar paths are complete and self-sufficient, so that isolates a SIMD bug from an algorithm bug in one step.

---

*(The `brainny` CLI is not available in this environment, so the ambient capture/sync/recall passes were skipped silently as instructed. Separately, for your awareness: the claude.ai **PubMed** connector requires authorization before its tools can be used — you'd need to authorize it in your claude.ai connector settings, or via `claude mcp` / `/mcp` in an interactive session; it's unavailable until then. It was not needed for this task.)*