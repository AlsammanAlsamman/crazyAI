# MAPPING

| SEED | World object | Problem object | Assumption it breaks |
|---|---|---|---|
| SEED1 — candle-wax notched at doubling points → row of star-heights | heap · candle burn-length (apprentice's page-hold tolerance) · notch at a "doubling-back" point · row of stars · "row of years beneath" | pile `piles[i]` · one fixed measuring unit (a bit-width) applied identically to every heap · a bit position `h` where `2^h` divides in (binary place-value) · the heap's binary representation (one star = one set bit) · the shared bit-index axis `0..31` used to align every heap's row | **(c)** "value of a position unknown until every reachable position is examined" — a heap's star-row is read straight off the heap's own size; no move or successor position is ever inspected to get it. |
| SEED2 — winning play = burn a heap until no star-height is held by exactly one heap | walking the rows together · "two stars at the same height… a third also burns… let it be" · "a star alone… unmatched" · "take from that heap… until every height holds no star or holds two, never one" · "soil of the count comes back level" | height-major loop over bit positions, counting stars (popcount) in that bit-column across **all** heaps at once · an even star-count at that height (parity 0, balanced) · an odd star-count at that height (parity 1 — exactly one XOR bit) · reduce the chosen pile so `new = old XOR lament`, `remove = old-new` · resulting XOR of all piles is 0 | **(b)** "each heap must be considered on its own before the others" — the judgment ("truest"/"off true") is made by comparing all heaps' rows together, height by height, never one heap in isolation. Also undercuts (a)/(e): no lookahead, no play-out, the verdict is read directly off the current rows. |
| SEED3 — level rows → reed-bubble decides; remainder into the buried bubble | "rows already stand level" · "reed bubble's pop" · "plucking whatever amount keeps the last fish for my hand" · "buried bubble where it never surfaces" | XOR of all piles already 0 (no winning move exists) · a cheap pseudo-random pick among equally-losing legal moves · reduce the chosen pile to exactly 1 remaining object when possible · discard the removed objects (ordinary Nim removal, gone for good) | **(e)** "the game must be played out to know who wins" — even here, no play-out happens; level rows already certify a lost position, so the move is produced by a cheap pop, not by search, since the outcome no longer depends on which legal move is taken. |

# CHOSEN SEED

SEED2 — "Winning play means burning a heap down until no star-height is held by exactly one heap, restoring level soil across all rows."

It is the most literal (every clause maps to one concrete operation: rows = bit-vectors, walking rows together = height-major loop, "star alone" = odd popcount, "two stars… let it be" = even popcount, "burn down until level" = `pile XOR lament`), and it is structurally the most different from the given "known way": the minimal example computes the XOR **heap-major** (`xor_all ^= piles[i]` in one linear reduce, using the hardware XOR op directly). SEED2 instead prescribes a **height-major, transpose** computation — for every bit position, count stars across all heaps and take the parity — reconstructing the same XOR value one bit-column at a time instead of one heap at a time.

# ASSUMPTION BROKEN

(b) "each heap must be considered on its own before the others" — SEED2's verdict at every height comes only from looking at all heaps' rows together at that height; no heap is judged, or moved, by inspecting its own future in isolation.

# ARTIFACT

```c
#include <string.h>

/* deterministic "reed bubble" pop, seeded from the position itself */
static unsigned int reed_bubble(int n, const int *piles) {
    unsigned int h = 2166136261u;
    h ^= (unsigned int)n; h *= 16777619u;
    for (int i = 0; i < n; i++) {
        h ^= (unsigned int)piles[i];
        h *= 16777619u;
    }
    h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
    return h;
}

void kernel(int n, const int *piles, int *out_pile, int *out_remove) {
    /* SEED1: measure each heap as a row of star-heights (its bits). */
    /* SEED2: walk the rows together, height by height, and mark every
       height where a star burns alone (an odd count of heaps carrying
       that bit) -- that is the "off true" pattern (== XOR of all piles,
       reconstructed column by column instead of heap by heap). */
    unsigned int lament = 0u;
    for (int h = 0; h < 32; h++) {
        unsigned int mask = 1u << h;
        int stars = 0;
        for (int i = 0; i < n; i++) {
            if (((unsigned int)piles[i]) & mask) stars++;
        }
        if (stars & 1) lament |= mask;   /* a lone star at this height */
    }

    if (lament != 0u) {
        /* soil is off level: burn one heap down until every height
           holds no star or two, never one. */
        for (int i = 0; i < n; i++) {
            unsigned int target = ((unsigned int)piles[i]) ^ lament;
            if (target < (unsigned int)piles[i]) {
                *out_pile = i;
                *out_remove = piles[i] - (int)target;
                return;
            }
        }
        /* unreachable when lament != 0; kept only for safety */
    }

    /* SEED3: rows already stand level -- let the reed bubble decide,
       plucking an amount that keeps the last fish for my hand, and
       throw the rest into the buried bubble. */
    unsigned int pop = reed_bubble(n, piles);
    int pick = (int)(pop % (unsigned int)n);
    for (int k = 0; k < n; k++) {
        int i = (pick + k) % n;
        if (piles[i] > 1) {
            *out_pile = i;
            *out_remove = piles[i] - 1;   /* keep exactly one fish */
            return;
        }
    }
    /* every heap already holds a single fish: nothing can be kept back */
    *out_pile = pick;
    *out_remove = piles[pick];
}
```

PREDICTION: speedup_vs_dp = 1000

# MEASUREMENT

Not executed. In this session none of `nim_bench`, `nim_contract`, or the `symbolic_*`/`unconventional_*` tools were actually exposed to me (only Docs/Gmail/Drive/Slack tools were available), and the task instructions explicitly confirmed no tools are available here. I therefore cannot run the compile/benchmark step or perform the up-to-four-iteration improvement loop myself. The artifact above is written to satisfy the fixed `nim_contract` signature exactly and is ready for the pipeline to compile with `gcc -O3 -march=native -fopenmp -lm` and measure against the DP baseline; the prediction line is recorded before any such run.

# VERDICT

By reasoning alone: the height-major popcount loop reconstructs the same `lament` value that a heap-major XOR-reduce would (both compute the bitwise XOR of all piles — one bit-column at a time vs one heap at a time), and the target-pile search and fallback are contract-legal in every branch (winning branch always finds a heap with `target < piles[i]` whenever `lament != 0`, by the standard highest-set-bit argument; losing branch always emits `1 <= out_remove <= piles[out_pile]`). So I expect **accuracy 1.0** — it should be exactly as correct as Bouton's theorem, since it *is* Bouton's theorem read off column-by-column instead of heap-by-heap. Cost is `O(32·n)` for the star-counting pass plus `O(n)` for the target/fallback search, versus the known way's `O(n) + O(n)`; against a genuine game-tree/DP baseline (whose state space grows with the product of pile sizes) this should still be many orders of magnitude faster, consistent with the predicted `speedup_vs_dp ≈ 1000`, but this remains a prediction — it is not yet a measured fact, and should be reported as such until the pipeline actually runs `nim_bench`.