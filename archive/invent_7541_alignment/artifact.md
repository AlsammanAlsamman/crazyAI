## MAPPING

**SEED 1** — "The body threads the plane along its short diagonals, curling back on itself at each end so a single unbroken crawl visits every cell once."

| World object | Problem object |
|---|---|
| bare plane | (n+1)×(n+1) DP grid |
| two edges, notched strings | rows = string `a`, columns = string `b` |
| body, one cell wide | the wavefront / frontier of cells being evaluated |
| short diagonal, top-edge to left-edge | anti-diagonal `d = i+j = const` |
| "no cell needs its neighbor on the same diagonal" | cells on one anti-diagonal are mutually independent — only diagonals `d-1`,`d-2` are needed |
| curling back, "never two passes" | single sweep `d = 0..2n`, in place |

Breaks: **"one pair of positions is judged at a time"** — an entire diagonal of (i,j) pairs can be judged simultaneously, since none of them depends on another in the same diagonal.

**SEED 2** — "kindles an ember, brightest of the three behind it plus its own kindling/coal/toll."

| World object | Problem object |
|---|---|
| ember | dp[i][j] score |
| three embers behind | diag(i-1,j-1), up(i-1,j), left(i,j-1) |
| kindling / coal / toll | match(+1) / mismatch(-1) / gap(-2) |

This is just the Needleman-Wunsch recurrence restated in metaphor. It breaks none of the listed assumptions — it's the per-cell rule itself, one cell at a time.

**SEED 3** — "Two diagonals back, embers are no longer needed... thin double row of fire."

| World object | Problem object |
|---|---|
| thin double row of fire | two live diagonal buffers, length O(n) each |
| snuffing/sweeping | freeing/reusing the diagonal from `d-2` |

Breaks the *space* complexity of the standard method (O(n) instead of O(n²) memory), but not "one pair at a time" — it's an implementation detail of storage, orthogonal to which/how-many pairs get judged per step.

## CHOSEN SEED
**SEED 1**, combined with SEED 3's rolling-buffer detail to make it viable in O(n) memory. SEED 1 is the only one of the three that breaks "one pair of positions is judged at a time" — the others don't touch that assumption at all.

## ASSUMPTION BROKEN
"one pair of positions is judged at a time" — replaced by "one whole anti-diagonal of pairs is judged at once," since no cell on a diagonal depends on any other cell on that same diagonal.

## ARTIFACT
Mapping to hardware: memory = three O(n) int arrays (`prev2`,`prev1`,`curr`) instead of the O(n²) table; what flows = the diagonal index `d`, moving forward in time; what stays still = the two strings, pre-laid along contiguous axes (`b` reversed so its diagonal-walk becomes a straight contiguous walk — this is the actual trick that falls out of taking "short diagonal" literally); processor = 8 SIMD lanes (AVX2 int32) computing 8 independent cells of one diagonal per instruction; time = the outer loop over `2n+1` diagonals.

This lands exactly on the classical, validated anti-diagonal SIMD technique for Needleman-Wunsch/Smith-Waterman (Wozniak 1997 and successors), rather than inventing something new — as required by the instructions.

```c
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>

#define MATCH 1
#define MISMATCH -1
#define GAP -2

int kernel(int n, const char *a, const char *b) {
    if (n == 0) return 0;

    /* lay the strings along their edges: a stays natural order,
       b is laid down reversed so a short-diagonal walk becomes a
       straight, contiguous walk through it. */
    int *ai = (int *)malloc((size_t)n * sizeof(int));
    int *brev = (int *)malloc((size_t)n * sizeof(int));
    for (int k = 0; k < n; k++) {
        ai[k] = (unsigned char)a[k];
        brev[k] = (unsigned char)b[n - 1 - k];
    }

    /* thin double row of fire: only two diagonals behind the
       crawl (plus the one being kindled) are ever kept alight. */
    const int pad = 8;
    size_t bufn = (size_t)(n + 1 + pad);
    int *prev2 = (int *)calloc(bufn, sizeof(int));
    int *prev1 = (int *)calloc(bufn, sizeof(int));
    int *curr  = (int *)calloc(bufn, sizeof(int));

    const __m256i vmatch    = _mm256_set1_epi32(MATCH);
    const __m256i vmismatch = _mm256_set1_epi32(MISMATCH);
    const __m256i vgap      = _mm256_set1_epi32(GAP);

    for (int d = 0; d <= 2 * n; d++) {
        if (d <= n) {
            curr[0] = d * GAP;   /* i == 0, the plane still bare on top */
            curr[d] = d * GAP;   /* j == 0, the plane still bare on left */
        }

        int lo = d - n; if (lo < 1) lo = 1;
        int hi = d - 1; if (hi > n) hi = n;
        if (lo > hi) {
            int *t = prev2; prev2 = prev1; prev1 = curr; curr = t;
            continue;
        }

        int off = n - d;   /* brev index = off + i, contiguous as i rises */
        int i = lo;
        int end8 = hi - 7;
        for (; i <= end8; i += 8) {
            __m256i va = _mm256_loadu_si256((const __m256i *)&ai[i - 1]);
            __m256i vb = _mm256_loadu_si256((const __m256i *)&brev[off + i]);
            __m256i eq = _mm256_cmpeq_epi32(va, vb);
            __m256i sc = _mm256_blendv_epi8(vmismatch, vmatch, eq);

            __m256i vdiag = _mm256_loadu_si256((const __m256i *)&prev2[i - 1]);
            __m256i vup   = _mm256_loadu_si256((const __m256i *)&prev1[i - 1]);
            __m256i vleft = _mm256_loadu_si256((const __m256i *)&prev1[i]);

            __m256i diagv = _mm256_add_epi32(vdiag, sc);
            __m256i upv   = _mm256_add_epi32(vup, vgap);
            __m256i leftv = _mm256_add_epi32(vleft, vgap);

            __m256i best = _mm256_max_epi32(diagv, _mm256_max_epi32(upv, leftv));
            _mm256_storeu_si256((__m256i *)&curr[i], best);
        }
        for (; i <= hi; i++) {
            int j = d - i;
            int diagv = prev2[i - 1] + (a[i - 1] == b[j - 1] ? MATCH : MISMATCH);
            int upv = prev1[i - 1] + GAP;
            int leftv = prev1[i] + GAP;
            int best = diagv;
            if (upv > best) best = upv;
            if (leftv > best) best = leftv;
            curr[i] = best;
        }

        int *t = prev2; prev2 = prev1; prev1 = curr; curr = t;
    }

    int result = prev1[n];   /* the one ember left at the far corner */

    free(ai); free(brev);
    free(prev2); free(prev1); free(curr);
    return result;
}
```

Notes on the guard required by step 4: no thread-level parallelism was added. Each diagonal has a hard synchronization point (rotation) before the next can start, and there are `2n+1` such points; for the diagonal lengths this benchmark's `n` will actually exercise, OpenMP-per-diagonal would pay barrier overhead far more often than it recoups in vector work, so it is deliberately left out — SIMD only, per the instructions' stated preference order. Small-`n` fallback is implicit and automatic: the vector loop body only executes when a diagonal's interior length ≥ 8; below that, only the scalar remainder loop runs, which is independently correct, so there is no separate code path to maintain and no risk of the vector path misbehaving on tiny inputs.

I traced this by hand for n=1 (match and mismatch cases) against the reference recurrence and it reproduces the reference scores exactly.

PREDICTION: speedup_vs_dp = 4.0

## MEASUREMENT
No `alignment_bench` tool call was available in this session (the environment note states tools are disabled here and the external pipeline will compile/measure the artifact). I am not fabricating a number — the prediction above is what I commit to before any measurement exists; the actual speedup should be filled in by that external run.

## VERDICT
The mechanism breaks "one pair of positions is judged at a time" by exploiting the genuine independence of cells on a Needleman-Wunsch anti-diagonal — the same fact the classical anti-diagonal SIMD technique (Wozniak-style) is built on, so this lands on a validated known method rather than a novel one, as required. The one condition under which this could lose to the plain DP is small `n`, where SIMD/setup overhead (allocation, precomputing `ai`/`brev`, per-diagonal bookkeeping) isn't amortized — that risk is addressed structurally, not just noted: the vector path only fires when a diagonal's interior is ≥ 8 wide, so tiny `n` silently runs the same scalar recurrence as the baseline with only O(n) extra bookkeeping, not a separate risky mechanism. No thread parallelism was added, consistent with the instruction to prefer vectorization and only add threads if the metaphor's own units of work are large enough — here the true unit of work (one diagonal) is serialized against its neighbors by construction, so threading would fight the mechanism rather than help it at realistic benchmark sizes.