# MAPPING

**SEED 1** — "A maker pricks coal-prints only where two shell-strings name the same old-speech word, leaving all other wall-stone bare and uncounted."

| World object | Problem object |
|---|---|
| two strings of counted shells | sequences `a`, `b` (length n) |
| a shell | a base (A/C/G/T) at a position |
| "old-speech name … the empty name whoever speaks commands" | the character identity at a position |
| brick wall | the (i,j) comparison space |
| coal print | a marked, scored cell — specifically where `a[i]==b[i]`, compared **shell for shell** (position-for-position, not all-pairs) |
| bare stone, never chalked/walked/counted | cells never computed, never allocated |

Breaks: **"the whole grid of every position against every other must be filled in"** — explicitly and literally (most of the wall is never chalked, walked, or counted).

**SEED 2** — "A maker threads those true prints in one forward-climbing line, never stepping back along either string…"

| World object | Problem object |
|---|---|
| true prints | positions where `a[i]==b[i]` |
| thread, forward-climbing, never back | a single monotonic left-to-right sweep, i and j advancing together |
| "longest steady agreement" | the running direct-diagonal match tally |

Breaks the same grid-filling assumption from a different angle: it replaces a nested nested double loop with **one** linear pass, and it fixes the processing order (used below to derive the band bound in one shot rather than rediscovering it cell-by-cell).

**SEED 3** — "…lets the thread go slack once and rejoin further down the wall … a fixed length of thread, no more."

| World object | Problem object |
|---|---|
| thread going slack once, rejoining past unpricked stone | a single permitted gap event |
| fixed thread-length cost | a fixed penalty for that one slip |

Breaks: **"a slip (gap) can only be discovered by having already compared the position before it"** (and implicitly, "gaps can occur anywhere, any number of times"). Taken *literally* — exactly one gap, fixed cost, discovered once — this is a **heuristic**: real optimal NW alignments can need many gaps in adversarial cases, so a literal single-slip chain would not reproduce the exact reference score. Per instruction 4, I do not ship this literal version.

# CHOSEN SEED

**SEED 1**, folding in SEED 2's forward-only single sweep. It is the most literal seed and the only one that squarely breaks "the whole grid must be filled in."

# ASSUMPTION BROKEN

"The whole grid of every position against every other must be filled in." Instead: run the shell-for-shell (diagonal) comparison once to see how much true agreement already exists without any slip, use that to derive a **provable radius** beyond which no optimal path can ever go, and only chalk (allocate + compute) that band of the wall — with **O(n·W) time and O(W) memory** instead of O(n²) and O(n²).

SEED 3's literal "one slip, fixed cost" is not exact for arbitrary DNA (an adversarial input can need many gaps). Per instruction 4, I let it point at the real, validated technique named in the prompt itself — banded/Ukkonen-style DP (the family KSW2/SSW belong to) — and derive the band width from data instead of hard-coding "exactly one slip."

**Derivation of the band radius.** For equal-length n, any alignment path uses `g` up-moves and `g` left-moves (g gap-pairs) and `n-g` diagonal moves. Score ≤ (n-g)·1 − 4g = n − 5g (best case all diagonal moves are matches, 2g gap columns at GAP=−2 each). The pure diagonal path (no gaps) is always a legal candidate, so OPT ≥ S0. Hence any path that could be optimal satisfies n−5g ≥ S0 ⇒ **g ≤ (n−S0)/5**, and the path's drift from the diagonal never exceeds g. So W = ⌈(n−S0)/5⌉+1 (margin) is a provably sufficient band half-width — clamped to n, which degenerates gracefully (same asymptotic cost as the reference, never worse) for maximally dissimilar sequences, while still keeping memory at O(n) instead of O(n²) even in that worst case.

# ARTIFACT

```c
#include <stdlib.h>
#include <limits.h>

#define MATCH 1
#define MISMATCH -1
#define GAP -2
#define NEG_INF (INT_MIN / 2)

static inline int max3(int x, int y, int z) {
    int m = x;
    if (y > m) m = y;
    if (z > m) m = z;
    return m;
}

int kernel(int n, const char *a, const char *b) {
    if (n == 0) return 0;

    /* "shell for shell": hold both strings side by side at matching
       counted positions and press a coal print (score +1) only where
       the old-speech names truly agree; leave mismatched stone bare
       (-1). This is the single forward pass of SEED 1 + SEED 2. */
    long S0 = 0;
    for (int i = 0; i < n; i++)
        S0 += (a[i] == b[i]) ? MATCH : MISMATCH;

    /* Any path that could tie-or-beat the direct thread spends g
       gap-pairs with g <= (n - S0) / 5 (score <= n - 5g, and OPT >= S0
       because the direct path is always legal). Drift from the
       diagonal never exceeds g. W is that radius, with a +1 safety
       margin -- the one place we "pay a fixed length" up front,
       computed once, not rediscovered cell by cell. Clamped to n so
       the worst case degrades to a full pass, never worse. */
    long Wl = (n - S0 + 4) / 5 + 1;
    if (Wl < 0) Wl = 0;
    if (Wl > n) Wl = n;
    int W = (int)Wl;

    int width = 2 * W + 1;
    int *prev = (int *)malloc((size_t)width * sizeof(int));
    int *cur  = (int *)malloc((size_t)width * sizeof(int));
    if (!prev || !cur) { free(prev); free(cur); return 0; }

    for (int k = 0; k < width; k++) {
        int j = (0 - W) + k;
        prev[k] = (j < 0 || j > n) ? NEG_INF : j * GAP;
    }

    for (int i = 1; i <= n; i++) {
        int lo = (i - W < 0) ? 0 : i - W;
        int hi = (i + W > n) ? n : i + W;
        for (int k = 0; k < width; k++) {
            int j = (i - W) + k;
            if (j < lo || j > hi) { cur[k] = NEG_INF; continue; }
            if (j == 0) { cur[k] = i * GAP; continue; }
            int diagv = prev[k] + (a[i - 1] == b[j - 1] ? MATCH : MISMATCH);
            int upv   = (k + 1 < width) ? prev[k + 1] + GAP : NEG_INF;
            int leftv = (k - 1 >= 0)    ? cur[k - 1] + GAP  : NEG_INF;
            cur[k] = max3(diagv, upv, leftv);
        }
        int *tmp = prev; prev = cur; cur = tmp;
    }

    int result = prev[W];
    free(prev);
    free(cur);
    return result;
}
```

I hand-verified the index arithmetic on n=1 (match and mismatch cases), which reproduces the exact reference scores.

PREDICTION: speedup_vs_dp = 4.0

# MEASUREMENT

No tool access was available in this session (explicitly noted for this turn), so `alignment_bench`/`alignment_contract` could not be executed here. I cannot self-verify the prediction; it must be measured by the downstream pipeline that compiles and runs the artifact. I am reporting this plainly rather than presenting an unmeasured guess as a result.

# VERDICT

Two honest regimes exist, and I did not paper over the bad one:

- **Similar/related sequences** (S0 close to n): W shrinks toward O(1)–O(√n)-ish, giving near-linear time and a large speedup (potentially >>4×).
- **Maximally dissimilar/adversarial diagonal** (S0 near −n): W approaches ~0.4n, so cell count only drops to ~0.8n² — the named risk condition where the mechanism is little better than the known DP. I addressed this directly rather than leaving it unguarded: (a) W is clamped to n, so time never exceeds the reference's asymptotic cost; (b) memory is **always** O(n) via rolling rows, never the reference's O(n²) malloc, which is an unconditional win (better cache behavior, no huge allocation) even in the worst case. So the risky regime degrades to "roughly as fast, but lighter on memory," never to "slower."

I did not force SIMD: within a row, `cur[k]` depends on `cur[k-1]` (the "left" move), a genuine loop-carried dependency inherent to Needleman–Wunsch row order, so auto-vectorization of the recurrence itself is not honestly available without restructuring to anti-diagonal (wavefront) order — a change I considered (SEED 2's "always climbing" maps cleanly onto increasing i+j) but declined to ship without the ability to compile-test its trickier index bookkeeping, per the instruction to favor a validated, provably-correct path over an untested clever one. I also skipped OpenMP: the natural unit of parallel work here (one row's band, width 2W+1) is a poor fit for thread-level parallelism at the sizes this bound produces, so per the size-guard instruction I left it out rather than add overhead-only threading.