# MAPPING

**SEED 1 — "Graduated doubling limestone weights decompose each heap into ranked tallies across the senet board's columns."**

| World object | Problem object |
|---|---|
| heap on the bench | one pile, `piles[i]` |
| case of weights 1,2,4,8,16,... (each doubled) | the powers of two `weight[r] = 2^r` |
| "lay down the largest weight it will bear, then the next against what remains" | greedy binary decomposition of `piles[i]` |
| clay tally pressed into a square | a bit of `piles[i]`'s binary representation, `tally[i][r]∈{0,1}` |
| senet board, rows/columns | rows = heaps, columns = weight-rank (bit position) |

Assumption broken: **"a winning move can only be found by looking ahead through the game's possible futures."** The heap is weighed once, directly, against a fixed physical reference (the weight case) — no future position is ever consulted.

**SEED 2 — "Columns holding an odd count of tallies are marked with red ochre... it burns."**

| World object | Problem object |
|---|---|
| reading straight down a column | summing `tally[i][r]` over all heaps `i`, for fixed rank `r` |
| odd count → ochre burns | that bit of the XOR of all piles is 1 |
| even count → still stone | that bit of the XOR is 0 |
| "if nothing burns" | `xor_all == 0`, a losing position |

Assumption broken: **"each heap must be considered on its own before the others."** The column reading deliberately aggregates *all* heaps at once, rank by rank — heaps are never judged in isolation; the whole board is read together.

**SEED 3 — "Sweep that heap's excess counters... into the discard jar... rebuild the heap smaller until every burning column comes up even."**

| World object | Problem object |
|---|---|
| heap whose tally lies in the highest burning column | pile `i` with the highest set bit of `xor_all` also set |
| sweeping counters into the discard jar | `*out_remove` objects taken from that pile |
| rebuilding until every burning column is even | new pile value = old value with every burning-column bit flipped |
| "the last object is always mine" | the resulting position has `xor_all == 0` for the opponent |

Assumption broken: **"the game must be played out to know who wins."** The mover never plays the game forward — the win is certified purely by the parity pattern on the board *right now*, before any further moves are made.

# CHOSEN SEED

**SEED 2** (the column-burning seed). It is the most literal (a column-count parity check, nothing more) and the most mechanically different from "the known way": Bouton's theorem is normally computed as a chain of `^` folds across the whole integers; here the identical quantity is produced by an entirely different concrete process — laying tallies into a rank/heap grid and counting *per column*, never folding two numbers together at all.

# ASSUMPTION BROKEN

"Each heap must be considered on its own before the others" — the column read explicitly considers all heaps simultaneously, rank by rank, which is exactly how the burning pattern (≡ XOR) is obtained without ever reducing two heaps into one running total.

# ARTIFACT

Literal mapping used in the code: heap = pile; weight case = powers of two; senet-board square = one bit of one pile, stored in a `tally[i][r]` grid; "reading down a column" = summing a column of the grid; "burning" = odd column sum; "highest burning column" = highest set bit of the XOR, found without ever computing the XOR as a folded integer; "sweeping/rebuilding a heap" = re-summing that heap's row after flipping exactly the burning-column squares.

```c
#include <stdlib.h>

void kernel(int n, const int *piles, int *out_pile, int *out_remove) {
    enum { NBITS = 31 };                         /* the scribe's case of doubled weights: 1,2,4,...,2^30 */
    int weight[NBITS];
    for (int r = 0; r < NBITS; r++) weight[r] = 1 << r;

    int *tally = (int *)malloc((size_t)n * NBITS * sizeof(int)); /* the senet board: row=heap, col=rank */

    /* weigh each heap, largest weight first, press a tally into each square it bears */
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        int remaining = piles[i];
        int *row = tally + (size_t)i * NBITS;
        for (int r = NBITS - 1; r >= 0; r--) {
            if (weight[r] <= remaining) { row[r] = 1; remaining -= weight[r]; }
            else row[r] = 0;
        }
    }

    /* read straight down each column and count its tallies */
    int column_count[NBITS];
    for (int r = 0; r < NBITS; r++) {
        int c = 0;
        for (int i = 0; i < n; i++) c += tally[(size_t)i * NBITS + r];
        column_count[r] = c;
    }

    /* an odd column burns with red ochre; an even column is left as still stone */
    int burning[NBITS];
    int any_burn = 0;
    for (int r = 0; r < NBITS; r++) { burning[r] = column_count[r] & 1; any_burn |= burning[r]; }

    if (!any_burn) {
        /* nothing burns: position already lost; take a single token from the heap that loses least */
        int best = 0;
        for (int i = 1; i < n; i++) if (piles[i] > piles[best]) best = i;
        *out_pile = best;
        *out_remove = 1;
        free(tally);
        return;
    }

    /* the highest burning column */
    int hcol = NBITS - 1;
    while (hcol >= 0 && !burning[hcol]) hcol--;

    /* the heap whose tally lies in that column */
    for (int i = 0; i < n; i++) {
        if (tally[(size_t)i * NBITS + hcol]) {
            /* sweep excess counters off; rebuild rank by rank until every burning column is even */
            int new_value = 0;
            int *row = tally + (size_t)i * NBITS;
            for (int r = 0; r < NBITS; r++) {
                int keep = burning[r] ? (row[r] ^ 1) : row[r];
                new_value += keep * weight[r];
            }
            *out_pile = i;
            *out_remove = piles[i] - new_value;
            free(tally);
            return;
        }
    }

    *out_pile = 0;
    *out_remove = 1;
    free(tally);
}
```

# PREDICTION

Accuracy is predicted to be 100% — the column-parity/rebuild procedure is algebraically identical to `piles[i] ^ xor_all` (flipping exactly the bits where the XOR is 1 toggles each burning column's parity from odd to even, and the highest-bit dominance guarantees the move is a legal reduction), so this should match Bouton's theorem move-for-move.

PREDICTION: speedup_vs_dp = 2000

(Reasoning for the number: this kernel is O(n·31) plus one malloc, versus a DP/game-tree baseline whose state space is exponential in pile sizes or heap count — even a modest benchmark instance makes that gap enormous; I discount from "millions" toward a few thousand because the per-call malloc and the 31-wide inner loops add real constant-factor overhead compared to the one-line `^`-fold minimal solution, which itself would already beat DP by many orders of magnitude.)

# MEASUREMENT

Not performed in this session — no tools (including `nim_bench`/`nim_contract`) were available to me here; the instructions for this turn state the pipeline will compile and measure the artifact externally. I am not fabricating a result. Based on static analysis only:
- Correctness: expected 100% (shown algebraically above — burning-column rebuild ≡ `piles[i]^xor_all`, no-burn case falls back to the same "take 1 from the largest pile" fallback as the minimal example).
- Performance risk: the `malloc`/`free` per call and the O(n·31) grid (vs. O(n) for the one-line XOR fold) are the most likely sources of a smaller-than-predicted speedup versus even the minimal correct solution, though both are still trivial next to an exponential DP baseline.

# VERDICT

Unverified pending actual `nim_bench` execution. The construction is a faithful, literal translation of SEED 2 (and the other two seeds it depends on) — it never folds pile values with `^`, only counts tallies per rank-column and rebuilds one heap's row — and it should be exactly as correct as Bouton's theorem while being asymptotically no different from it (O(n) heaps × O(log max_pile) bits). If measurement later shows the malloc/parallel-loop overhead dominates for small `n`, the first improvement (not yet applied, since no measurement exists to justify it) would be to replace the heap-allocated `tally` grid with the direct bit test `(piles[i]>>r)&1`, keeping the same column-parity logic but dropping the O(n·31) storage — that remains for a future iteration once real numbers are in hand, per the "measure before changing" discipline.