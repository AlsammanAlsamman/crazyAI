## MAPPING

| World-object | Problem-object | Assumption it breaks |
|---|---|---|
| **notched ferry-rod, cut for crossings not distance** | An array of *bit positions* (doubling marks), not a heap's length — a structure that records parity across heaps, not the size of any one heap | "a move only affects the one heap it touches" (the rod is a *shared* structure, not per-heap) |
| **walking each heap once per doubling mark (1s, 2s, 4s…), laying a pebble face-up/down** | For each bit `b`, loop over *all* piles and tally how many have bit `b` set (face-up = yes); this is a bit-major/heap-minor loop, the transpose of `xor_all ^= piles[i]` | "each heap must be considered on its own before the others" — here no heap is fully processed alone; every heap contributes one vote per mark, mark by mark |
| **a mark with an odd count of yeses = unbalanced, a debt to the water** | `xor_all`'s bit `b` = 1 (parity of set bits at position `b`) | "the value of a position is unknown until every reachable position is examined" — the imbalance is read directly, no lookahead |
| **go to a heap with a yes at the highest unbalanced mark, throw into the dark carry-basket until all marks flip face-down** | pick `i` where bit `highest(xor_all)` of `piles[i]` is 1; remove `piles[i] - (piles[i]^xor_all)` objects | "a winning move can only be found by looking ahead through the game's possible futures" — the exact removal count is read off the current rod state alone |
| **sit still — stillness is a deliberate second move, wait for the opponent to unbalance the rod again** | after emitting the move, return; no simulation of opponent replies is needed to know the outcome | "the game must be played out to know who wins" |

## CHOSEN SEED

**SEED 1**: *"Each heap I walk twice: once counting by the ones, once by the twos, once by fours, laying a pebble face-up wherever a heap answers yes… face-down wherever no."*

This is the most literal *and* the most structurally different from the known way. The minimal/known solution computes `xor_all` heap-major (`for i: xor_all ^= piles[i]`), letting hardware XOR handle all bit positions at once, invisibly. SEED 1 insists on the opposite loop nest — bit-major, heap-minor: walk the *marks* (bit positions) one at a time, and for each mark walk *all* heaps to tally an odd/even vote. That's a genuine transpose of the computation, not just a rephrasing of it.

## ASSUMPTION BROKEN

"**Each heap must be considered on its own before the others**." In the literal translation, no heap is ever examined in isolation to completion — every heap only ever contributes a single yes/no vote to whichever doubling-mark is currently being read, and the imbalance is a property of the *rod* (all heaps combined, column-wise), never of one heap read start-to-finish first.

## ARTIFACT

```c
void kernel(int n, const int *piles, int *out_pile, int *out_remove) {
    /* The ferry-rod: one doubling-mark per bit. Piles are >=1, fit in
       a signed 32-bit int, so 31 marks (bit 31 is the sign bit) suffice. */
    const int NBITS = 31;
    int rod[31]; /* rod[b] = 1 (face-up / unbalanced) iff an ODD number of
                    heaps answer "yes" at doubling-mark 2^b */

    for (int b = 0; b < NBITS; b++) {
        int yes_count = 0;
        for (int i = 0; i < n; i++) {
            if ((piles[i] >> b) & 1) yes_count++;   /* walk every heap for this mark */
        }
        rod[b] = yes_count & 1;                      /* odd yeses -> face-up */
    }

    /* find the highest unbalanced mark */
    int highest = -1;
    for (int b = NBITS - 1; b >= 0; b--) {
        if (rod[b]) { highest = b; break; }
    }

    if (highest == -1) {
        /* whole rod already all face-down: no winning move exists,
           take anything legal (mirrors the minimal contract's fallback) */
        *out_pile = 0;
        *out_remove = 1;
        return;
    }

    /* reconstruct the full imbalance pattern from the rod */
    int xor_all = 0;
    for (int b = 0; b < NBITS; b++) {
        if (rod[b]) xor_all |= (1 << b);
    }

    /* go to any heap carrying a "yes" at the highest unbalanced mark,
       and throw enough into the dark carry-basket to flip every
       unbalanced mark for that heap -> rod goes all face-down */
    for (int i = 0; i < n; i++) {
        if ((piles[i] >> highest) & 1) {
            int target = piles[i] ^ xor_all;   /* flips exactly the unbalanced marks */
            if (target < piles[i]) {
                *out_pile = i;
                *out_remove = piles[i] - target;
                return;
            }
        }
    }

    /* unreachable if Bouton's theorem holds, kept as a safety net */
    *out_pile = 0;
    *out_remove = 1;
}
```

Correctness note: this is mathematically identical to `xor_all = XOR(piles)` — the rod's bit `b` is exactly the parity of bit `b` across all piles, which *is* the definition of XOR at that bit. So it must produce the same move as the minimal example, just computed column-by-column (31 passes over `n` piles) instead of via the hardware `^` operator (one pass). It is still `O(n)` (constant factor 31), never inspects a reachable position, and stops the instant the rod reads all-face-down — matching every piece of SEED 1–3.

PREDICTION: speedup_vs_dp = 500

(Reasoning for the number: against a DP/minimax baseline that must enumerate reachable positions — combinatorially large even for modest pile counts/sizes — an O(31·n) tally is still asymptotically trivial. I discount it well below the "no search, O(n)" ideal of the minimal example because of the ~31× constant-factor overhead from doing 31 explicit passes instead of one hardware XOR instruction, so I expect a large but not astronomical measured speedup — hence 500 rather than 10⁴–10⁶.)

## MEASUREMENT

Not performed: no tools (`nim_contract`, `nim_bench`) were available in this session, per the environment note. The kernel above is offered as the artifact for the pipeline to compile and benchmark; I have not silently substituted a measured number for the prediction above.

## VERDICT

Pending actual `nim_bench` results. Expected outcome: **accuracy = 100%** (the rod-tally computation is provably equal to Bouton's XOR, bit for bit, so every emitted move is a legal winning move whenever one exists, and the same safe fallback otherwise). Expected speed: large speedup over DP, somewhat under the minimal example's likely speedup due to the deliberate 31-pass, bit-major structure the seed demands — that gap is the honest cost of taking SEED 1 literally rather than quietly collapsing it back into a single `^=` loop.