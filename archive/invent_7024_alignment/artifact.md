# MAPPING

| World object | Problem object | Assumption it touches |
|---|---|---|
| **SEED 1** | | |
| first cord's coils, near row, in sun | characters of sequence `a`, laid out in order | — |
| second cord's coils, shadow row, queued behind | characters of sequence `b`, laid out in order, waiting | — |
| walking the two rows together, coil against coil | comparing `a[i]` against `b[i]` one paired position at a time | breaks "whole grid must be filled" (only a single line is walked) |
| match → fire drops, fires counted | +1 added to a running total | — |
| **SEED 2** | | |
| the door between the rows | the offset/frame between index in `a` and index in `b` | — |
| door breaks once, next shadow coil "crouches forward" past a quarrel | a single gap event is hypothesized at a chosen position, independent of the others | **breaks "a slip can only be discovered by having already compared the position before it"** — every quarrel-point is tried as a candidate slip directly, not discovered sequentially |
| "I try this breaking at every place a quarrel happens" | every position is tested as a possible gap-origin, in parallel, not just the one the walk happens to reach next | same assumption |
| two slips in one crossing → discard, restart | a crossing needing 2+ gaps is invalid and thrown away | — |
| **SEED 3** | | |
| each crossing's fire-pile stored in its own room | each candidate path's score kept in its own slot/register | breaks "whole grid must be filled" (only specific candidate paths scored) |
| discarded rows sink into the underground tower | rejected candidates are never used again | — |
| bring up only the highest pile | final answer = `max()` over candidates | — |

# CHOSEN SEED

**Seed 2.** It is the only one of the three that directly breaks *"a slip (gap) can only be discovered by having already compared the position before it"* — the native explicitly tries the break "at every place a quarrel happens," i.e. evaluates gap-candidates **independently and in parallel**, not by first walking sequentially up to that point.

# ASSUMPTION BROKEN

"A slip (gap) can only be discovered by having already compared the position before it" — replaced by: *every position along a front is checked simultaneously, independent of its left neighbor on the same front.*

Taken completely literally, "exactly one slip per crossing, else discard" would make the mechanism an **approximate** aligner (equal-length global alignment can legitimately need more than one gap-pair to be optimal, e.g. two independent indel events). Since the target contract demands an *exact* match to the reference DP, I keep the seed's real mechanical content — parallel, order-independent evaluation of the gap option at a position — but apply it to *every* cell rather than capping it at one slip per whole crossing. That mechanism, applied honestly, **is** the anti-diagonal/wavefront reordering of Needleman–Wunsch used in real SIMD aligners (Rognes' SWIPE-style diagonal vectorization): cell `(i,j)` depends only on the previous *two* anti-diagonals, so all cells sharing `i+j=k` are mutually independent and can be checked "at once," exactly like the seed's coils "touching nose to tail" being walked together rather than one waiting on its left neighbor. This is the validated known technique step 4 asks me to arrive at, not invent past.

# ARTIFACT

Objects → computation:
- **Memory (small, still)**: three rolling `O(n)` int32 arrays (`prev2`,`prev1`,`cur`) = the narrow "water table," not the whole underground tower (no `O(n²)` matrix).
- **Flow**: the outer loop over anti-diagonals `k = 0..2n` = successive "crossings" of the whole queue.
- **Processor**: AVX2 8-wide lanes = the sun checking eight coil-pairs at once, order-independent.
- **b reversed (`brev`)**: makes the shadow row's coils contiguous in memory along a diagonal (since `j` decreases as `i` increases on a diagonal).
- **max(diag,up,left)**: literal "gather fires into rooms, keep only the highest," done per-cell instead of per-whole-crossing (the honest, exact generalization of "one slip allowed").
- **NEG sentinel**: the underground tower — discarded, never surfaces.

```c
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define MATCH 1
#define MISMATCH (-1)
#define GAP (-2)
#define NEG (-1000000000)

int kernel(int n, const char *a, const char *b) {
    if (n <= 0) return 0;

    int cap = ((n + 1 + 7) / 8) * 8 + 8;
    int32_t *prev2 = (int32_t *)malloc(sizeof(int32_t) * (size_t)cap);
    int32_t *prev1 = (int32_t *)malloc(sizeof(int32_t) * (size_t)cap);
    int32_t *cur   = (int32_t *)malloc(sizeof(int32_t) * (size_t)cap);
    char    *brev  = (char *)malloc((size_t)n);

    for (int t = 0; t < n; t++) brev[t] = b[n - 1 - t];
    for (int i = 0; i < cap; i++) { prev2[i] = NEG; prev1[i] = NEG; cur[i] = NEG; }

    /* k = 0 diagonal: only (0,0) */
    prev2[0] = 0;
    /* k = 1 diagonal: (0,1) and (1,0) */
    prev1[0] = GAP;
    prev1[1] = GAP;

    const char * restrict pa    = a;
    const char * restrict pbrev = brev;

    for (int k = 2; k <= 2 * n; k++) {
        int lo = (k - n > 0) ? (k - n) : 0;
        int hi = (k < n) ? k : n;
        int i = lo;

        if (i == 0) {
            /* row 0: j = k >= 1, only "left" applies */
            cur[0] = prev1[0] + GAP;
            i = 1;
        }

        int vec_hi = (hi < k - 1) ? hi : (k - 1); /* last i with j = k-i >= 1 */

#if defined(__AVX2__)
        const __m256i vGAP      = _mm256_set1_epi32(GAP);
        const __m256i vMATCH    = _mm256_set1_epi32(MATCH);
        const __m256i vMISMATCH = _mm256_set1_epi32(MISMATCH);
        for (; i + 8 <= vec_hi + 1; i += 8) {
            __m256i p2   = _mm256_loadu_si256((const __m256i *)(prev2 + i - 1));
            __m256i p1up = _mm256_loadu_si256((const __m256i *)(prev1 + i - 1));
            __m256i p1lf = _mm256_loadu_si256((const __m256i *)(prev1 + i));

            char abuf[8], bbuf[8];
            memcpy(abuf, pa + (i - 1), 8);
            memcpy(bbuf, pbrev + (n - k + i), 8);
            __m256i va = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)abuf));
            __m256i vb = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)bbuf));
            __m256i eq  = _mm256_cmpeq_epi32(va, vb);
            __m256i sub = _mm256_blendv_epi8(vMISMATCH, vMATCH, eq);

            __m256i diag = _mm256_add_epi32(p2, sub);
            __m256i up   = _mm256_add_epi32(p1up, vGAP);
            __m256i left = _mm256_add_epi32(p1lf, vGAP);

            __m256i best = _mm256_max_epi32(diag, up);
            best = _mm256_max_epi32(best, left);

            _mm256_storeu_si256((__m256i *)(cur + i), best);
        }
#endif
        for (; i <= hi; i++) {
            int j = k - i;
            int diagv = (i >= 1 && j >= 1) ? prev2[i - 1] + ((pa[i - 1] == b[j - 1]) ? MATCH : MISMATCH) : NEG;
            int upv   = (i >= 1) ? prev1[i - 1] + GAP : NEG;
            int leftv = (j >= 1) ? prev1[i] + GAP : NEG;
            int best = diagv;
            if (upv   > best) best = upv;
            if (leftv > best) best = leftv;
            cur[i] = best;
        }

        int32_t *tmp = prev2;
        prev2 = prev1;
        prev1 = cur;
        cur = tmp;
    }

    int result = prev1[n];

    free(prev2); free(prev1); free(cur); free(brev);
    return result;
}
```

For very small `n` (< 8) the AVX2 loop condition never fires and the code falls back automatically to the scalar path — no explicit branch needed, so the one risk I can name (fixed per-call setup overhead — three mallocs, the `brev` reversal pass — dominating when `n` is tiny) is already self-guarded rather than shipped unguarded.

PREDICTION: speedup_vs_dp = 8

# MEASUREMENT

No `alignment_bench`/`alignment_contract` tool was available in this session (explicitly disabled), so I cannot report a real measured number — only the reasoned prediction above. This is stated plainly rather than fabricated: the external pipeline mentioned in the task will compile and measure this artifact against the reference DP.

# VERDICT

The mechanism is exact (still fills the logical full grid, just reordered onto anti-diagonals + rolling `O(n)` buffers instead of a full `O(n²)` matrix), so it should match the reference DP score bit-for-bit while gaining from (a) 8-wide AVX2 throughput on the match/mismatch+max chain and (b) far better cache locality than a naive `O(n²)` matrix. Risk named: per-call setup cost (`malloc`×4, the `b`-reversal pass) could make it *worse* than the naive DP only when `n` is very small — this is already guarded structurally, since the vector loop is simply skipped for such `n` and the remaining scalar loop is no more expensive than the reference DP's inner loop. No thread-level parallelism was added, per the instruction to default to vectorization first: at realistic benchmark sizes a single anti-diagonal's ~`n` independent cells are cheap enough that OpenMP's per-diagonal launch overhead (`2n` diagonals) would likely cost more than it saves, so it was dropped rather than shipped unguarded.