# MAPPING

| World object | Problem object | Silent assumption it breaks |
|---|---|---|
| **Seed 1 — sliding comb, trailing bead dropped** | | |
| the strand | `seq[0..n-1]` | — |
| a bead | one base (`A/C/G/T`) | — |
| the comb, fixed at k teeth | a k-wide code register, 2k bits | — |
| bead caught under a tooth | one 2-bit field currently held in the running code | — |
| bead falling off the trailing tooth, dropped in the basket, "owed nothing more" | the oldest base's 2-bit field, discarded by `& mask` when the code is shifted | breaks **(1)** "code must be built fully from scratch" — the code is *carried forward*, not rebuilt; breaks **(4)** "every base contributes independently to the window it starts" — each base is folded in once and rides along in k−1 more windows via the carried code, it is never re-read |
| "I never look at it twice" | single forward index `i`, no backtracking | reinforces, does not break, assumption (5) |
| **Seed 2 — market stall, flower-of-geometry unique to a face** | | |
| the face (k beads under the comb) | the packed 2k-bit code | — |
| the stall shaped so only that face fits | `counts[code]`, a bijective direct address | this is already exactly what the naive kernel does (`counts[code]++`); it doesn't break any of the five listed assumptions, it just re-describes direct addressing — least novel seed |
| **Seed 3 — one pebble per visit, one closing walk reads the tally** | | |
| pebble on the scale | `counts[code]++` | — |
| closing walk read at the very end | a final reduction pass, implying tallying can be *deferred/local* (e.g. per-thread private tables) until one merge at the close | breaks **(3)** "count table can only be updated one position's result at a time" — visits could accumulate into private scales, merged once; hints at breaking (5) by allowing chunked/parallel visitation, read out only at the close |

# CHOSEN SEED

**Seed 1** — the sliding comb that drops only the trailing bead. It is the most literal translation (register holds exactly k live bases, one shift-and-mask per step) and it is maximally different from the known way, because the naive kernel's entire cost (`O(n·k)`) comes precisely from rebuilding the k-length code from scratch at every position — which is the one thing this seed forbids ("I never look at it twice").

# ASSUMPTION BROKEN

(1) "each k-mer's code must be built fully from scratch, one base at a time, before it can be counted" and (4) "every base contributes independently to the window it starts." The comb instead carries one running code; each incoming bead is read exactly once, contributes to k consecutive windows, and the outgoing bead is simply masked off — no re-derivation of the other k−1 bases.

# ARTIFACT

```c
#include <stdint.h>

/* one fixed lookup: translate a raw bead (ASCII base) straight to its
   2-bit code, so reading a bead under the comb is O(1) with no branch,
   matching the naive kernel's A=0,C=1,G=2,default=3 convention */
static const unsigned char CODE[256] = {
    [0 ... 255] = 3,
    ['A'] = 0, ['C'] = 1, ['G'] = 2, ['T'] = 3
};

void kernel(int n, const char *seq, int k, uint64_t *counts) {
    if (k <= 0 || n < k) return;

    uint64_t mask = (k >= 32) ? ~(uint64_t)0 : (((uint64_t)1 << (2 * k)) - 1);

    uint64_t code = 0;
    int i = 0;

    /* prime the comb: load the first k-1 beads under the teeth without
       counting yet -- the window isn't full, no face to carry to market */
    for (; i < k - 1; i++) {
        code = ((code << 2) | (uint64_t)CODE[(unsigned char)seq[i]]) & mask;
    }

    /* slide the comb one bead at a time down the strand, never looking
       back: fold in the new bead, the trailing bead is masked off and
       owed nothing more, read the fresh face, drop one pebble at its
       stall */
    for (; i < n; i++) {
        code = ((code << 2) | (uint64_t)CODE[(unsigned char)seq[i]]) & mask;
        counts[code]++;
    }
}
```

PREDICTION: speedup_vs_naive = 6.0

Reasoning: naive does `O(k)` work (a full re-encode) per window; the comb does `O(1)` amortized work (one shift/mask/table-lookup/increment) per window. Asymptotically the win should scale roughly with k, but fixed per-position overhead (the increment itself, memory traffic into the `counts` array, loop control) eats into that ideal ratio, and for small k the naive inner loop is cheap/well-predicted to begin with. For the k range typical of such a benchmark (roughly 8–16), I expect the incremental version to land around 6x faster than naive rather than the full k-fold ideal.

# MEASUREMENT

Not run in this session — no `kmer_bench`/`kmer_contract` tool access was available here (per the task note, "No tools are available in this session"). The prediction above was fixed before any measurement, as required; actual timing is to be produced by the external pipeline that compiles and benchmarks this artifact.

# VERDICT

Provisional, pending the pipeline's run: the artifact is a literal, mechanical realization of Seed 1 — a fixed-width comb (2k-bit register) that ingests one bead per step, drops exactly the trailing bead via masking, and reads the surviving k-bead face as a single index into a bijective stall table (`counts`), with one pebble (`++`) set per visit and no re-derivation of any bead already under the teeth. This directly removes the naive kernel's `O(k)` per-position rebuild, so a real speedup over naive is expected; the magnitude should be checked against the PREDICTION of 6.0x once `kmer_bench` is run, and if it falls short, the likely next lever (not yet applied, to stay within the "improve at most four times" budget and because it changes the contract's memory footprint) would be per-thread private `counts` tables over OpenMP-split chunks, merged in one closing pass — which is exactly what Seed 3 describes and was deliberately left out of this first artifact pending a real measurement.