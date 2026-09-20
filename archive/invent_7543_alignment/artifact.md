# MAPPING

| World object | Problem object | Silent assumption it touches |
|---|---|---|
| **Seed 1** | | |
| two chessboards, one per file of marks | the two DP-adjacent index axes for `a` and `b` | — |
| pawn on each square | one dp-relevant character position | "one pair of positions judged at a time" |
| walking both boards "rank against rank", holding up one hand from each board at the same rank | comparing `a[i]` to `b[i]` at the *same* index, not through a 2-D table | "one pair of positions judged at a time" (breaks it: all ranks are independent, judgeable together) |
| hands "closing true" / crowned queen | match → +1 | scoring rule |
| orphaned hand / ash | mismatch → −1 (ash, discarded) | scoring rule |
| **Seed 2** | | |
| "the second board's boy climbs once into the oak, one rank, and never comes down" | a single, permanent index offset between the `a`-walk and the `b`-walk from some rank onward | **"a slip (gap) can only be discovered by having already compared the position before it"** — this is the one that literally claims you can pick the slip *without* having walked the whole prefix first |
| "every mark on board two answers to the mark one square along" | `b[j]` is now compared against `a[j+1]` (or vice versa) after the climb | same |
| "a boy only climbs once … never a second one" | exactly one gap-open event, no closing gap | same, extended: a *single* irreversible reindexing |
| **Seed 3** | | |
| two hearts beating in one chest, run side by side | two independent candidate scoring strategies computed in parallel | "the whole grid… must be filled" (challenges whether you need the *full* table, or just enough to compare two hypotheses) |
| chest / oxygen taken in | running score accumulator fed by matches | — |
| port / ash carried off | discarded penalty, not retained in the kept total | — |
| keep the heart with more oxygen, less ash | `max()` over the two strategies' final tallies | — |

# CHOSEN SEED

**Seed 2** is the most literal and the only one that *claims* to break the target assumption ("gap discovered without having compared the position before it") — it says the climb-rank can be chosen once, up front, rather than discovered by sequential comparison.

**But it fails a hard check.** Both input strings have length `n`. Any valid global-alignment path runs from `(0,0)` to `(n,n)`. If `U` = number of "up" (gap-in-b) moves and `L` = number of "left" (gap-in-a) moves, then row-progress `U+D=n` and column-progress `L+D=n` force **`U=L`** always. A "climb once and never come back down" is a lone `U` (or `L`) move with no matching partner later — it ends at offset `≠0`, i.e. it literally cannot land back on `(n,n)`. So the pure, literal Seed-2 heart is not a valid representation of *any* global alignment on equal-length strings in general, and cannot be used as-is without risking wrong scores on inputs whose optimal alignment needs more than one paired detour (e.g. sequences with a real indel plus noise). Since the contract requires **exact** agreement with the reference DP, I cannot ship an approximation that only handles ≤1 gap-pair.

Per instruction 4, I let the mechanism fall through to the well-known, validated technique that satisfies the *spirit* of Seed 2 (judge many marks "at once," not rank-by-rank in row order) while staying exact: **anti-diagonal (wavefront) computation of the full DP table**, vectorized. This is the same family named in the prompt as "known" (SSW/KSW2-style SIMD), and I arrive at it honestly rather than inventing something untested.

# ASSUMPTION BROKEN

Not assumption 4 (I could not preserve correctness while breaking it — stated plainly, as instructed). Instead, the artifact breaks:
- "every cell depends on the ones above/left/diagonal, **computed in that order**" — cells are computed by anti-diagonal, not row-major order.
- "one pair of positions is judged at a time" — 8 cells of a diagonal are judged simultaneously via AVX2.

One genuine surviving piece of Seed 2's imagery: **reversing board two** (`rb[k]=b[n-1-k]`) is exactly "every mark on board two answers to the mark one square along" applied once, permanently, at setup — it turns board two's naturally-reversed access pattern (within a diagonal, `j` decreases as `i` increases) into a forward, contiguous one, which is precisely what makes the SIMD loads possible.

# ARTIFACT

Memory: three rolling `O(n)` arrays (`prev2`,`prev1`,`cur`) instead of the reference's `O(n²)` table — "chest" (kept, small, living data) vs. "ash" (the old diagonal, overwritten/discarded each step, never retained). Flow: diagonal index `d` is time; the processor lane is 8 `int32` cells judged at once; `a` and the reversed `b` are read forward and contiguously.

```c
#include <stdlib.h>
#include <immintrin.h>

#define MATCH 1
#define MISMATCH -1
#define GAP -2

int kernel(int n, const char *a, const char *b) {
    if (n == 0) return 0;

    char *rb = (char *)malloc((size_t)n);           /* reversed b: rb[k] = b[n-1-k] */
    for (int k = 0; k < n; k++) rb[k] = b[n - 1 - k];

    int *prev2 = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int *prev1 = (int *)malloc((size_t)(n + 1) * sizeof(int));
    int *cur   = (int *)malloc((size_t)(n + 1) * sizeof(int));

    cur[0] = 0;                                       /* diagonal d = 0 */
    { int *t = prev1; prev1 = cur; cur = t; }

    const __m256i vmatch    = _mm256_set1_epi32(MATCH);
    const __m256i vmismatch = _mm256_set1_epi32(MISMATCH);
    const __m256i vgap      = _mm256_set1_epi32(GAP);

    for (int d = 1; d <= 2 * n; d++) {
        int imin = d - n; if (imin < 0) imin = 0;
        int imax = d < n ? d : n;
        int lo = imin, hi = imax;

        if (lo == 0)      { cur[0] = d * GAP; lo = 1; }      /* i=0 boundary  */
        if (hi == d)      { cur[d] = d * GAP; hi = d - 1; }  /* j=0 boundary  */

        int i = lo;
        for (; i + 7 <= hi; i += 8) {
            int abase  = i - 1;
            int rbbase = n - d + i;                          /* = n - j, contiguous */

            __m128i achar  = _mm_loadl_epi64((const __m128i *)&a[abase]);
            __m128i rbchar = _mm_loadl_epi64((const __m128i *)&rb[rbbase]);
            __m256i va  = _mm256_cvtepu8_epi32(achar);
            __m256i vrb = _mm256_cvtepu8_epi32(rbchar);

            __m256i eq = _mm256_cmpeq_epi32(va, vrb);
            __m256i sc = _mm256_blendv_epi8(vmismatch, vmatch, eq);

            __m256i vp2   = _mm256_loadu_si256((const __m256i *)&prev2[i - 1]);
            __m256i vdiag = _mm256_add_epi32(vp2, sc);
            __m256i vp1a  = _mm256_loadu_si256((const __m256i *)&prev1[i - 1]);
            __m256i vup   = _mm256_add_epi32(vp1a, vgap);
            __m256i vp1b  = _mm256_loadu_si256((const __m256i *)&prev1[i]);
            __m256i vleft = _mm256_add_epi32(vp1b, vgap);

            __m256i vmax = _mm256_max_epi32(vdiag, vup);
            vmax = _mm256_max_epi32(vmax, vleft);
            _mm256_storeu_si256((__m256i *)&cur[i], vmax);
        }
        for (; i <= hi; i++) {
            int j = d - i;
            int s = (a[i - 1] == b[j - 1]) ? MATCH : MISMATCH;
            int diag = prev2[i - 1] + s;
            int up   = prev1[i - 1] + GAP;
            int left = prev1[i] + GAP;
            int best = diag;
            if (up   > best) best = up;
            if (left > best) best = left;
            cur[i] = best;
        }

        int *t = prev2; prev2 = prev1; prev1 = cur; cur = t;
    }

    int result = prev1[n];
    free(prev2); free(prev1); free(cur); free(rb);
    return result;
}
```

No thread-level parallelism added: diagonals are only `O(n)` wide and there are `O(n)` of them, so a thread barrier per diagonal would dominate at the sizes this contract implies (DNA alignment benches typically run n in the hundreds–low thousands). SIMD is the guarded, size-appropriate win; threading is left out rather than shipped as a risk I don't address.

PREDICTION: speedup_vs_dp = 6.0

# MEASUREMENT

No `alignment_bench`/`alignment_contract` tool was available in this session to execute directly (per the session note, the pipeline compiles and measures the artifact after this response). I am reporting the prediction above honestly as unmeasured-by-me; the actual number should come from the pipeline's run against the reference kernel, and I have not substituted a guessed "measured" figure for a real one.

# VERDICT

Correctness: the anti-diagonal recurrence is algebraically identical to the reference (same three predecessors, same boundary values `i*GAP`/`j*GAP`), just reordered and batched — it should match the reference exactly, unlike a literal Seed-2 single-shift heuristic, which I showed cannot even reach `(n,n)` for equal-length strings in general and was therefore discarded rather than shipped.

Expected win: `O(n)` memory instead of `O(n²)` (avoids a large malloc and cache-thrashing over the big table) plus 8-wide AVX2 throughput on the recurrence's compare/add/max — both should help monotonically with `n`; there's no condition under which this artifact should be *worse* than the naive reference, since it does strictly less memory traffic and the SIMD path is only entered for full 8-lane chunks (the scalar tail path is identical in cost profile to the reference for the leftover cells), so no separate fallback guard was needed beyond the tail loop already in the code.