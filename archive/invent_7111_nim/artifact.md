# MAPPING

**Seed 1** — "Heaps are decomposed into doubling shells stitched as shared rows on one embroidery cloth by heap-assigned girls in the tower."

| World object | Problem object |
|---|---|
| heap of sticks | `piles[i]` |
| girl (heap-assigned) | index `i` into the pile array |
| tower | the whole kernel's working context (all piles at once) |
| cloth | conceptual n×B bit-matrix of all piles |
| doubling shells / rows (1,2,4,8,…) | binary place values / bit positions `b` |
| "girl stitches into a row only if her heap, broken into shells, has that shell lit" | `(piles[i] >> b) & 1` |
| "threads cross in the same row without tangling — shared row, belongs to no single heap" | a bit-column `b` is a global object spanning all heaps simultaneously |

Breaks: **"each heap must be considered on its own before the others"** — the row is explicitly shared/global; it is a structure indexed by bit, not by heap.

**Seed 2** — "A row where crossing threads fail to pair off evenly marks an unbalanced binary place across all heaps at once."

| World object | Problem object |
|---|---|
| row | bit position `b` |
| crossing threads in that row | the set of heaps with bit `b` set |
| "fail to pair off evenly" | odd parity of that bit-count across heaps |
| "unbalanced binary place, across all heaps at once" | XOR-of-all-piles has a 1 at bit `b` |
| walking cloth top to bottom, hunting highest unbalanced row | scanning bits high→low for the first odd parity |

Breaks: same assumption — determining "unbalanced" is a simultaneous, cross-heap judgment made row-by-row, not heap-by-heap.

**Seed 3** — "The winning move is unpicking one heap's threads down to re-pair every unbalanced row, discarding sticks into the hotel room that empties by noon."

| World object | Problem object |
|---|---|
| unpicking one heap's threads down to a new count | reducing `piles[i]` to `piles[i]^xor_all` |
| "re-pair every unbalanced row from there down" | that target value clears the top bit and rebalances all lower bits |
| hotel room that empties by noon | the removed sticks (`out_remove`), discarded/irrelevant afterward |
| no such girl / idle handful, wait for the next uneven row | fallback when `xor_all==0` (losing position): take 1 from pile 0 |

This seed maps almost verbatim onto the **known** algorithm's final search loop (`target = piles[i]^xor_all; if target<piles[i] ...`) — least different from the textbook method.

# CHOSEN SEED

Seed 2 (with Seed 1's shell/row substrate needed to realize it). Both 1 and 2 break the targeted assumption; Seed 3 does not (it's the standard move-selection step almost unchanged). Seed 2 is chosen over Seed 1 because it names the actual *decision procedure* ("hunting the highest row where threads fail to pair off"), not just the data layout — it is the more literal, load-bearing translation, and it forces a genuinely different loop nest than the textbook fold.

# ASSUMPTION BROKEN

"Each heap must be considered on its own before the others." The standard kernel is heap-major: loop over heaps, XOR-accumulate (`for i: xor_all ^= piles[i]`) — one heap fully processed, folded into a scalar, then the next. The seed's cloth is row-major: for each shared row (bit position), look across *all* girls' heaps at once and count parity; only after finishing that row does the hunt move to the next row. This is a literal transpose of the loop nest (bit-outer, heap-inner) rather than the standard (heap-outer, bit-folded-by-hardware).

# ARTIFACT

```c
void kernel(int n, const int *piles, int *out_pile, int *out_remove) {
    /* Tower cloth: rows are doubling shells (bit positions), shared across
       all heap-girls. A row is "unbalanced" if an odd number of girls'
       threads cross it (odd parity of that bit across all heaps). We hunt
       the cloth top (highest shell) to bottom for the first unbalanced
       row, building the XOR value one shared row at a time instead of
       folding heap-by-heap. */

    int maxpile = 0;
    for (int i = 0; i < n; i++) if (piles[i] > maxpile) maxpile = piles[i];

    int top_bit = -1;
    while ((1 << (top_bit + 1)) <= maxpile) top_bit++;
    if (top_bit < 0) top_bit = 0;

    int xor_all = 0;
    int highest_unbalanced_row = -1;

    /* walk the cloth top to bottom: row by row (shell by shell), count how
       many girls' threads cross this row */
    for (int b = top_bit; b >= 0; b--) {
        int crossing = 0;
        for (int i = 0; i < n; i++) {
            if ((piles[i] >> b) & 1) crossing ^= 1; /* even count = pairs off evenly */
        }
        if (crossing) {
            xor_all |= (1 << b);
            if (highest_unbalanced_row < 0) highest_unbalanced_row = b;
        }
    }

    if (highest_unbalanced_row < 0) {
        /* every row already pairs off evenly: no such girl, no such thread;
           take an idle handful and wait for the next uneven row to open */
        *out_pile = 0;
        *out_remove = 1;
        return;
    }

    /* find the one girl whose heap, trimmed at the highest unbalanced row,
       re-pairs every row from there down; her removed sticks go to the
       hotel room that empties by noon (out_remove), gone, not tracked */
    for (int i = 0; i < n; i++) {
        if ((piles[i] >> highest_unbalanced_row) & 1) {
            int target = piles[i] ^ xor_all;
            if (target < piles[i]) {
                *out_pile = i;
                *out_remove = piles[i] - target;
                return;
            }
        }
    }

    *out_pile = 0;
    *out_remove = 1;
}
```

Object-level computational mapping: a heap = `piles[i]` (int); a row/shell = one bit position `b`, a global object spanned by all heaps, not owned by any one; "knowing" whether a row is unbalanced = an explicit parity count taken across the whole heap array for that single bit, computed before any heap's full value is ever folded into a scalar; a move = picking the heap that owns the top unbalanced row's bit and trimming it to `piles[i]^xor_all`. Correctness is identical to Bouton's theorem (same `xor_all`, same target), only the *order* of computing it is transposed from heap-major to row(bit)-major, per the chosen seed.

PREDICTION: speedup_vs_dp = 4000

Reasoning for the number: this kernel is still `O(n·log(maxpile))` — a constant-factor (~30×, one pass per bit) more work than the textbook `O(n)` hardware-XOR fold, but still linear in `n` and utterly unlike an exponential game-tree/DP baseline over pile states. Against a `dp`/minimax baseline (exponential in total stones or heap sizes), the asymptotic gap dominates the constant-factor overhead from the bit-plane restructuring, so I expect a very large but not maximal speedup — somewhat below what the pure-XOR minimal kernel would show, hence 4000 rather than, say, 50000.

# MEASUREMENT

Not available in this session — no `nim_bench`/`nim_contract` execution tool is present among the tools I was given here (only Docs/Gmail/Drive/Slack). Per the task's own instruction ("No tools are available in this session... the pipeline will compile and measure it"), I am not fabricating a result. The kernel above is submitted for the pipeline to compile with `gcc -O3 -march=native -fopenmp -lm` and measure against `nim_contract`/`nim_bench`; I have not iterated beyond one version since there is no measured feedback to iterate against in this session.

# VERDICT

Correctness argument (not a measurement): the bit-plane parity computed for each `b` equals bit `b` of XOR-of-all-piles, so `xor_all` reconstructed row-by-row is exactly Bouton's XOR; `highest_unbalanced_row` is exactly the top set bit of `xor_all`; the search loop and target formula are the standard, provably-correct Bouton move. So the algorithm should be 100% accurate against the theorem regardless of the loop-order transposition — the transposition changes *how* the same fact is discovered (row-shared, all-heaps-at-once) but not *what* is discovered. Speed prediction (4000×) is untested in this session and stands only as a prediction pending the pipeline's actual `nim_bench` run.